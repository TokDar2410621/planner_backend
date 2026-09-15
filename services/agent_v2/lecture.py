"""
LIRE en mode ombre: une lecture typee du message tape, qui ne decide RIEN.

Pourquoi. AGIR et les gardes du code lisent le francais brut par des regex, et
une faute (« aujourdui », « demin », « mercedi ») saute une garantie sans
bruit. Le fondateur interdit de faire grossir ces regex. LIRE est un appel a
part, sans outil ni raisonnement, qui ne voit que le message TAPE et un
contexte construit par le code, et rend une LectureTour validee
(lecture_schema.py). Avant qu'une regle ne s'en serve, elle tourne en
production pour mesurer, sur le vrai trafic, combien de fois elle contredit
les lecteurs regex geles.

Ce que ce module garantit (core/test_agent_v2_lire_ombre.py le verifie):
1. Rien ne decide sur la lecture. Le tour lit seulement les metadonnees
   rendues par finir(), qui ne vont qu'au message persiste.
2. Aucun ORM dans le pool: preparer() lit la base dans le thread de la
   requete, lire() n'appelle que le fournisseur.
3. Un pool DEDIE et reutilise. Jamais _POOL_AGIR (un fournisseur lent
   affamerait AGIR), jamais un thread par tour (gel du deuxieme tour le
   2026-08-28, voir agent.py).
4. Le tour attend au plus LIRE_ATTENTE_S, a sa fin, hors de tout outil et de
   tout verrou. Au-dela: hors_budget; une lecture encore en file est annulee,
   une lecture en cours abandonne son repli et n'est que journalisee a son
   arrivee. La file est bornee (FILE_MAX): au-dela, erreur file_pleine, sans
   meme lire la base.
5. Rien de l'utilisateur dans les journaux: statuts, noms de fournisseur,
   classes d'erreur et accords categoriels. Jamais str(e) ni exc_info: une
   erreur de validation pydantic recopie les valeurs lues, donc ses mots.

Statuts: ok, partielle (l'ancrage a retire au moins un champ), absente
(sortie invalide, enveloppe comprise, ou lecture vide sur un message non
vide), hors_budget, desactivee (LIRE_OMBRE), erreur (aucun fournisseur n'a
repondu), sautee (chemin rapide du code: LIRE n'est pas soumise).
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from concurrent.futures import TimeoutError as _DelaiDepasse
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Optional

from django.conf import settings
from django.utils import timezone

from services.agent_v2 import demandes as dem
from services.agent_v2 import lecture_schema as schema

logger = logging.getLogger(__name__)

OK = "ok"
PARTIELLE = "partielle"
ABSENTE = "absente"
HORS_BUDGET = "hors_budget"
DESACTIVEE = "desactivee"
ERREUR = "erreur"
SAUTEE = "sautee"

# Delai client par appel, sans relance: la sonde 2 mesure un p95 de 1,70 s et
# aucun appel au-dela de 5 s.
DELAI_LIRE = 6.0
# Gemini refuse toute echeance sous 10 s: HTTP 400 « Minimum allowed deadline
# is 10s » a chaque appel (smoke test du 2026-09-15). L'attente du tour reste
# bornee par LIRE_ATTENTE_S; seul le travailleur du pool attend plus longtemps.
DELAI_GEMINI = 10.0
BASE_DEEPSEEK = "https://api.deepseek.com"
BASE_GEMINI = None  # None: l'adresse par defaut de google-genai
ATTENTE_DEFAUT = 1.5
ATTENTE_MAX = 10.0
MODELES_DEFAUT = ("deepseek-flash", "gemini-2.5-flash")
PREFIXE_FORMULAIRE = "Voici mes réponses"

CLES_METADONNEES = ("lecture", "lecture_statut", "lecture_fournisseur", "lecture_ms")

# Quatre travailleurs: un appel dure 1 a 2 s et le trafic reste modeste. Sous
# une rafale, les lectures attendent en file et celles d'un tour deja conclu
# sont annulees avant de partir.
_TRAVAILLEURS = getattr(settings, "AGENT_V2_THREADS_LIRE", 4)
_POOL_LIRE = ThreadPoolExecutor(max_workers=_TRAVAILLEURS, thread_name_prefix="lire")

# La file du ThreadPoolExecutor n'a pas de borne. Pendant une panne DeepSeek,
# chaque lecture tient un travailleur jusqu'a son delai, et les tours suivants
# s'empilaient sans limite avec leur contexte. _en_file compte les lectures
# soumises et pas encore terminees, en cours ou en attente.
FILE_MAX = 2 * _TRAVAILLEURS
_FILE_VERROU = threading.Lock()
_en_file = 0


def _reserver() -> bool:
    global _en_file
    with _FILE_VERROU:
        if _en_file >= FILE_MAX:
            return False
        _en_file += 1
        return True


def _liberer(_futur=None) -> None:
    global _en_file
    with _FILE_VERROU:
        _en_file = max(0, _en_file - 1)


def en_file() -> int:
    """Lectures soumises et pas encore terminees."""
    with _FILE_VERROU:
        return _en_file

LIGNE = ("agent_v2 lire statut=%s fournisseur=%s ms=%d accords=%s"
         " attente=%d rejets=%d erreur=%s")


# ------------------------------------------------------------------ reglages

_VALEURS_COUPEES = {"0", "false", "off", "non", "no"}


def ombre_active() -> bool:
    """Actif si LIRE_OMBRE est absent: aucune variable a poser en production."""
    valeur = getattr(settings, "LIRE_OMBRE", None)
    if valeur is None:
        return True
    if isinstance(valeur, bool):
        return valeur
    return str(valeur).strip().lower() not in _VALEURS_COUPEES


def attente_s() -> float:
    try:
        secondes = float(getattr(settings, "LIRE_ATTENTE_S", ATTENTE_DEFAUT))
    except (TypeError, ValueError):
        return ATTENTE_DEFAUT
    if secondes != secondes:  # NaN
        return ATTENTE_DEFAUT
    return min(max(secondes, 0.0), ATTENTE_MAX)


def modeles_lire() -> tuple[str, ...]:
    valeur = getattr(settings, "LIRE_MODELES", None)
    if valeur is None:
        return MODELES_DEFAUT
    noms = valeur.split(",") if isinstance(valeur, str) else list(valeur or [])
    return tuple(n.strip() for n in noms if isinstance(n, str) and n.strip())


# ------------------------------------------------------------------- donnees

@dataclass(frozen=True)
class Preparation:
    """Tout ce que lire() consomme, deja lu en base."""
    message: str
    aujourdhui: date
    contexte: str
    refs: dict
    modeles: tuple


@dataclass(frozen=True)
class Resultat:
    statut: str
    lecture: object = None
    fournisseur: str = ""
    ms: int = 0
    erreur: str = ""
    rejets: int = 0
    attente_ms: int = 0

    def metadonnees(self) -> dict:
        return {
            "lecture": self.lecture.model_dump(mode="json") if self.lecture is not None else None,
            "lecture_statut": self.statut,
            "lecture_fournisseur": self.fournisseur or None,
            "lecture_ms": int(self.ms),
        }


@dataclass
class Suivi:
    preparation: Optional[Preparation] = None
    futur: object = None
    depart: float = 0.0
    fixe: Optional[Resultat] = None
    abandon: Optional[threading.Event] = None


def _ms(depart: float) -> int:
    return int((time.perf_counter() - depart) * 1000) if depart else 0


# ------------------------------------------------- thread de la requete (ORM)

def preparer(user, message_brut: str) -> Preparation:
    """Le contexte de LIRE, lu en base dans le thread de la requete."""
    from core.models import ConversationMessage, RecurringBlock, Task

    message = message_brut if isinstance(message_brut, str) else ""
    maintenant = timezone.localtime()
    aujourdhui = maintenant.date()

    groupes: dict = {}
    blocs = (RecurringBlock.objects
             .filter(user=user, active=True)
             .exclude(end_date__lt=aujourdhui)
             .order_by("day_of_week", "start_time", "pk"))
    for bloc in blocs:
        if not isinstance(bloc.day_of_week, int) or not 0 <= bloc.day_of_week <= 6:
            continue
        cle = (bloc.title, bloc.start_time, bloc.end_time, bloc.is_flexible, bloc.block_type)
        groupe = groupes.setdefault(cle, {
            "titre": bloc.title, "jours": [], "ids": [],
            "debut": bloc.start_time.strftime("%H:%M"),
            "fin": bloc.end_time.strftime("%H:%M"),
            "type": bloc.block_type,
        })
        if bloc.day_of_week not in groupe["jours"]:
            groupe["jours"].append(bloc.day_of_week)
        groupe["ids"].append(bloc.pk)
    semaine = sorted(groupes.values(), key=lambda g: (min(g["jours"]), g["debut"], g["titre"]))

    taches = [t.title for t in Task.objects.filter(user=user, completed=False)[:10]]

    # Le message precedent du tour: [0] est le message courant, deja sauve.
    question, formulaire = "", []
    deux = list(ConversationMessage.objects.filter(user=user).order_by("-pk")[:2])
    if len(deux) == 2 and deux[1].role == "assistant" and isinstance(deux[1].metadata, dict):
        meta = deux[1].metadata
        question = str(meta.get("question") or "")
        formulaire = [
            {"id": str(c.get("id")), "type": str(c.get("type") or ""),
             "label": str(c.get("label") or "")}
            for c in meta.get("interactive_inputs") or []
            if isinstance(c, dict) and c.get("id")
        ]

    attente = []
    for demande in dem.demandes_en_attente(user)[:4]:
        libelles: dict = {}
        for puce in demande.get("chips") or []:
            if isinstance(puce, dict) and puce.get("option") and puce.get("label"):
                libelles.setdefault(puce["option"], str(puce["label"]))
        attente.append({
            "motif": str(demande.get("motif") or ""),
            "cible_titre": str((demande.get("cible") or {}).get("titre") or ""),
            "question": question,
            "options": [{"id": o["id"], "libelle": libelles[o["id"]]}
                        for o in demande.get("options") or []
                        if isinstance(o, dict) and o.get("id") in libelles],
            "cle": demande.get("cle"),
        })

    origine = "formulaire" if message.strip().startswith(PREFIXE_FORMULAIRE) else "tape"
    texte, refs = schema.contexte_lire(aujourdhui, maintenant.strftime("%H:%M"), origine,
                                       semaine, taches, attente, formulaire, question)
    return Preparation(message=message, aujourdhui=aujourdhui, contexte=texte,
                       refs=refs, modeles=modeles_lire())


def sautee() -> Suivi:
    return Suivi(fixe=Resultat(SAUTEE))


def demarrer(user, message_brut: str) -> Suivi:
    """Prepare et soumet la lecture. Ne leve jamais, n'attend jamais."""
    if not ombre_active():
        return Suivi(fixe=Resultat(DESACTIVEE))
    # Avant preparer(): une file pleine ne coute aucune requete en base.
    if not _reserver():
        return Suivi(fixe=Resultat(ERREUR, erreur="file_pleine"))
    depart = time.perf_counter()
    try:
        prep = preparer(user, message_brut)
    except Exception as e:  # noqa: BLE001 - une mesure ne casse pas un tour
        _liberer()
        logger.warning("agent_v2 lire contexte illisible erreur=%s", type(e).__name__)
        return Suivi(fixe=Resultat(ERREUR, ms=_ms(depart), erreur=type(e).__name__))
    abandon = threading.Event()
    try:
        futur = _POOL_LIRE.submit(lire, prep, depart, abandon)
    except Exception as e:  # noqa: BLE001 - pool ferme a l'arret du processus
        _liberer()
        return Suivi(preparation=prep, fixe=Resultat(ERREUR, ms=_ms(depart), erreur=type(e).__name__))
    # Rappel appele une seule fois, a la fin comme a l'annulation (Future.cancel
    # invoque les rappels): la place se libere dans tous les cas.
    futur.add_done_callback(_liberer)
    return Suivi(preparation=prep, futur=futur, depart=depart, abandon=abandon)


# ------------------------------------------------------ pool LIRE (sans ORM)

@asynccontextmanager
async def _modele(nom: str):
    """Le modele d'un identifiant de LIRE_MODELES, ou None (cle absente, inconnu).

    Clients NEUFS, ouverts et fermes dans la boucle de CETTE lecture. Smoke test
    du 2026-09-15: GoogleProvider(api_key=...) seul reprend le client httpx mis
    en cache pour tout le processus, et un appel depuis la boucle d'un autre
    thread bloquait 30 s; un AsyncOpenAI jamais ferme levait « Event loop is
    closed » a la sortie du processus. Aucun client ne survit a sa lecture."""
    if nom.startswith("deepseek"):
        cle = getattr(settings, "DEEPSEEK_API_KEY", "")
        if not cle:
            yield None
            return
        from openai import AsyncOpenAI
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.deepseek import DeepSeekProvider

        async with AsyncOpenAI(api_key=cle, base_url=BASE_DEEPSEEK,
                               timeout=DELAI_LIRE, max_retries=0) as client:
            yield OpenAIChatModel(nom, provider=DeepSeekProvider(openai_client=client))
        return
    if nom.startswith("gemini"):
        cle = getattr(settings, "GEMINI_API_KEY", "")
        if not cle:
            yield None
            return
        import httpx
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        async with httpx.AsyncClient(timeout=DELAI_GEMINI) as client:
            yield GoogleModel(nom, provider=GoogleProvider(api_key=cle, http_client=client,
                                                           base_url=BASE_GEMINI))
        return
    yield None


def delai(nom: str) -> float:
    return DELAI_GEMINI if nom.startswith("gemini") else DELAI_LIRE


async def _appeler(nom: str, modele, prep: Preparation):
    from pydantic_ai import Agent, ToolOutput

    # output_retries=0: une sortie invalide rend absente, sans second appel paye.
    agent = Agent(modele, output_type=ToolOutput(schema.LectureTour, name=schema.NOM_OUTIL),
                  output_retries=0)
    # Jamais d'historique: le message tape seul, le contexte en instructions
    # apres le prompt fixe (prefixe mis en cache chez le fournisseur).
    resultat = await agent.run(prep.message,
                               instructions=f"{schema.PROMPT_LIRE}\n\n{prep.contexte}",
                               model_settings=schema.reglages_lire(nom, delai(nom)))
    return resultat.output


def _classe(e: BaseException) -> str:
    code = getattr(e, "status_code", None)
    return f"{type(e).__name__}:{code}" if isinstance(code, int) else type(e).__name__


def _sortie_invalide(e: BaseException) -> bool:
    from pydantic import ValidationError
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    return isinstance(e, (UnexpectedModelBehavior, ValidationError))


def _evaluer(prep: Preparation, nom: str, sortie, depart: float) -> Resultat:
    try:
        lecture = schema.LectureTour.model_validate_json(sortie.model_dump_json())
    except Exception as e:  # noqa: BLE001
        return Resultat(ABSENTE, fournisseur=nom, ms=_ms(depart), erreur=type(e).__name__)
    lecture, rejets = schema.ancrer(lecture, prep.message, prep.aujourdhui, prep.refs)
    if prep.message.strip() and not lecture.elements and not lecture.reponses:
        return Resultat(ABSENTE, fournisseur=nom, ms=_ms(depart), erreur="lecture_vide",
                        rejets=len(rejets))
    return Resultat(PARTIELLE if rejets else OK, lecture=lecture, fournisseur=nom,
                    ms=_ms(depart), rejets=len(rejets))


def lire(prep: Preparation, depart: float = 0.0,
         abandon: Optional[threading.Event] = None) -> Resultat:
    """Dans le pool LIRE. AUCUN ORM: tout ce qui vient de la base est dans prep.

    Repli sur panne du fournisseur seulement (HTTP, delai, reseau). Une sortie
    invalide rend absente sans repli: la mesure porte sur le lecteur, et un
    second appel doublerait cout et latence.

    Une boucle asyncio PAR lecture (asyncio.run), jamais celle du thread: les
    clients des fournisseurs y naissent et y meurent (voir _modele)."""
    try:
        return asyncio.run(_lire(prep, depart, abandon))
    except Exception as e:  # noqa: BLE001 - le pool ne remonte jamais d'exception
        return Resultat(ERREUR, ms=_ms(depart), erreur=type(e).__name__)


async def _lire(prep: Preparation, depart: float,
                abandon: Optional[threading.Event] = None) -> Resultat:
    erreurs: list[str] = []
    for nom in prep.modeles:
        # Tour deja conclu en hors_budget: plus personne n'attend ni cette
        # lecture ni son repli. Sous une panne DeepSeek, le repli Gemini tenait
        # le travailleur environ 16 s pour un resultat seulement journalise.
        if abandon is not None and abandon.is_set():
            return Resultat(ERREUR, ms=_ms(depart), erreur="|".join(erreurs + ["abandon"]))
        try:
            async with _modele(nom) as modele:
                if modele is None:
                    continue
                sortie = await _appeler(nom, modele, prep)
        except Exception as e:  # noqa: BLE001
            if _sortie_invalide(e):
                return Resultat(ABSENTE, fournisseur=nom, ms=_ms(depart), erreur=_classe(e))
            code = getattr(e, "status_code", None)
            if nom.startswith("deepseek") and code in (401, 402):
                # Un credit mort ne se voit nulle part ailleurs: le repli
                # repond et le tour continue sans bruit.
                logger.warning("agent_v2 lire fournisseur=%s http=%d cle refusee ou credit "
                               "epuise, repli sur le suivant", nom, code)
            # Chaque fournisseur tente laisse sa classe: la panne du repli ne se
            # cache pas derriere celle du premier.
            erreurs.append(f"{nom.split('-', 1)[0]}:{_classe(e)}")
            continue
        return _evaluer(prep, nom, sortie, depart)
    return Resultat(ERREUR, ms=_ms(depart), erreur="|".join(erreurs) or "AucunFournisseur")


# --------------------------------------------------- fin du tour (requete)

def conclure(suivi: Suivi) -> Resultat:
    """Attend au plus LIRE_ATTENTE_S. Ne leve jamais."""
    if suivi.fixe is not None:
        return suivi.fixe
    try:
        return suivi.futur.result(timeout=attente_s())
    except _DelaiDepasse:
        pass
    except Exception as e:  # noqa: BLE001 - lire() ne leve pas; filet
        return Resultat(ERREUR, ms=_ms(suivi.depart), erreur=type(e).__name__)
    if suivi.abandon is not None:
        suivi.abandon.set()
    if not suivi.futur.cancel():
        prep = suivi.preparation
        suivi.futur.add_done_callback(lambda futur: _tardive(futur, prep))
    return Resultat(HORS_BUDGET, ms=_ms(suivi.depart))


def _tardive(futur, prep: Preparation) -> None:
    """Rappel du pool: journalise, n'ecrit rien en base."""
    try:
        if futur.cancelled():
            return
        r = futur.result()
        logger.info("agent_v2 lire tardive statut=%s fournisseur=%s ms=%d accords=%s",
                    r.statut, r.fournisseur or "-", r.ms,
                    formater_accords(accords(r.lecture, prep.message, prep.aujourdhui, prep.refs)))
    except Exception as e:  # noqa: BLE001
        logger.warning("agent_v2 lire tardive illisible erreur=%s", type(e).__name__)


def journaliser(suivi: Suivi, resultat: Resultat) -> None:
    """LA ligne du tour. Valeurs categorielles seulement."""
    prep = suivi.preparation
    if prep is None or resultat.lecture is None:
        acc = dict.fromkeys(LECTEURS, NA)
    else:
        acc = accords(resultat.lecture, prep.message, prep.aujourdhui, prep.refs)
    logger.info(LIGNE, resultat.statut, resultat.fournisseur or "-", resultat.ms,
                formater_accords(acc), resultat.attente_ms, resultat.rejets,
                resultat.erreur or "-")


def finir(suivi: Suivi) -> dict:
    """Attend (borne), journalise, rend les metadonnees du message. Ne leve jamais."""
    try:
        debut = time.perf_counter()
        resultat = conclure(suivi)
        resultat = replace(resultat, attente_ms=int((time.perf_counter() - debut) * 1000))
        journaliser(suivi, resultat)
        return resultat.metadonnees()
    except Exception as e:  # noqa: BLE001 - une mesure ne casse pas un tour
        logger.warning("agent_v2 lire conclusion illisible erreur=%s", type(e).__name__)
        return Resultat(ERREUR, erreur=type(e).__name__).metadonnees()


# ------------------------------------------ accords avec les lecteurs geles
#
# Fonctions pures. Pour chaque lecteur regex gele, le meme message lu des deux
# cotes: ok (meme valeur), diff (valeurs differentes, ou un seul cote lit
# quelque chose), na (aucun des deux ne lit rien, ou le lecteur ne s'applique
# pas a ce tour). Les dates typees passent par schema.jours_resolus: une
# reference que le code ne peut pas trancher vaut QUESTION, et un lecteur regex
# qui tranche la ou le code demanderait est en desaccord.

NA, ACCORD, DESACCORD = "na", "ok", "diff"
LECTEURS = ("heures", "dates", "jour_vise", "date_visee", "suppression",
            "evenement_unique", "cette_semaine", "puces_date")
_RETRAITS = frozenset({"supprimer", "sauter_une_fois"})
_PLANIFIER = frozenset({"ajouter", "deplacer"})


def _booleens(regex: bool, typee: bool) -> str:
    if not regex and not typee:
        return NA
    return ACCORD if regex == typee else DESACCORD


def _retraits(lecture) -> list:
    return [e for e in lecture.elements if e.polarite == "demande" and e.operation in _RETRAITS]


def _dates_typees(elements, aujourdhui: date) -> tuple[set, bool]:
    """Les jours nommes d'une lecture: dates resolues hors fenetres, et les
    jours d'une habitude a leur prochaine occurrence, comme _dates_nommees."""
    dates, question = set(), False
    for e in elements:
        for genre, valeur in schema.jours_resolus(e, aujourdhui):
            if genre in schema.FENETRES:
                continue
            if valeur == schema.QUESTION:
                question = True
            else:
                dates.add(valeur)
        for jour in schema.jours_de_semaine(e):
            dates.add(dem.prochaine_occurrence(jour, aujourdhui))
    return dates, question


def accord_heures(lecture, message: str) -> str:
    """demandes.heures_dites. Une heure typee a deux lectures (H, H+12) est
    d'accord avec une regex qui en lit une."""
    regex = set(dem.heures_dites(message))
    typees = [set(h.lectures) for e in lecture.elements for h in e.heures if h.lectures]
    if not regex and not typees:
        return NA
    couvertes = set().union(*typees) if typees else set()
    d_accord = regex <= couvertes and all(t & regex for t in typees)
    return ACCORD if d_accord else DESACCORD


def accord_dates(lecture, message: str, aujourdhui: date) -> str:
    """demandes._dates_nommees."""
    regex = set(dem._dates_nommees(message, aujourdhui))
    typees, question = _dates_typees(lecture.elements, aujourdhui)
    if not regex and not typees and not question:
        return NA
    if question:
        return DESACCORD
    return ACCORD if regex == typees else DESACCORD


def accord_jour_vise(lecture, message: str, aujourdhui: date) -> str:
    """demandes.jour_vise, qui choisit la forme de la question de suppression:
    ne s'applique qu'a un retrait, lu d'un cote ou de l'autre."""
    retraits = _retraits(lecture)
    if not retraits and not dem.suppression_demandee(message):
        return NA
    dates, question = _dates_typees(retraits or lecture.elements, aujourdhui)
    typee = bool(dates) or question
    return ACCORD if typee == dem.jour_vise(message) else DESACCORD


def accord_date_visee(lecture, message: str, aujourdhui: date, refs: dict) -> str:
    """demandes.date_visee: un seul retrait type, lie a un seul groupe de la
    semaine type qui tombe un seul jour."""
    retraits = _retraits(lecture)
    if len(retraits) != 1:
        return NA
    element = retraits[0]
    groupes = [(refs.get("s") or {}).get(c) for c in element.candidats
               if isinstance(c, str) and c.startswith("s")]
    if len(groupes) != 1 or not groupes[0] or len(groupes[0].get("jours") or []) != 1:
        return NA
    dow = groupes[0]["jours"][0]
    typee = None
    for genre, valeur in schema.jours_resolus(element, aujourdhui):
        if genre in schema.FENETRES:
            continue
        if valeur == schema.QUESTION:
            return DESACCORD
        if valeur.weekday() == dow:
            typee = valeur
            break
    if typee is None:
        typee = dem.prochaine_occurrence(dow, aujourdhui)
    return ACCORD if typee == dem.date_visee(message, dow, aujourdhui) else DESACCORD


def accord_suppression(lecture, message: str) -> str:
    """demandes.suppression_demandee."""
    return _booleens(dem.suppression_demandee(message), bool(_retraits(lecture)))


def accord_evenement_unique(lecture, message: str) -> str:
    """outils.jour_sans_recurrence, le lecteur de _evenement_unique. Une lecture
    qui ne dit rien de la recurrence (non_dit partout) ne tranche pas: na, et
    non diff contre une regex qui lit un jour sans mot de recurrence."""
    from services.agent_v2.outils import jour_sans_recurrence

    if all(e.recurrence == "non_dit" for e in lecture.elements):
        return NA
    return _booleens(bool(jour_sans_recurrence(message)),
                     any(e.recurrence == "unique" for e in lecture.elements))


def semaine_bornee_regex(message: str) -> bool:
    """La condition de texte de outils._borner_semaine, sur ses regex gelees."""
    from services.agent_v2 import outils

    plat = dem.sans_accents(message)
    return bool(outils._SEMAINE.search(plat)) and not outils._SEMAINE_SANS_FIN.search(plat)


def accord_cette_semaine(lecture, message: str) -> str:
    return _booleens(semaine_bornee_regex(message),
                     any(e.recurrence == "cette_semaine_seulement" for e in lecture.elements))


_ISO_V1 = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def date_puces_v1(message: str, aujourdhui: date) -> Optional[date]:
    """Miroir de la jambe date de services/agent/agent.py _chips_from_message
    (aujourd'hui, demain, ISO), sans verbe, heures ni base. Rien ne decide
    dessus; un test epingle la parite avec l'original."""
    if not message:
        return None
    norm = unicodedata.normalize("NFKD", message).encode("ascii", "ignore").decode("ascii").lower()
    if "aujourd'hui" in norm or "aujourdhui" in norm:
        return aujourdhui
    if "demain" in norm:
        return aujourdhui + timedelta(days=1)
    m = _ISO_V1.search(norm)
    if not m:
        return None
    try:
        return date.fromisoformat(m.group(1))
    except ValueError:
        return None


def accord_puces_date(lecture, message: str, aujourdhui: date) -> str:
    """Jambe date des creneaux forces: la date de placement unique d'une
    demande d'ajout ou de deplacement."""
    regex = date_puces_v1(message, aujourdhui)
    places, question = set(), False
    for e in lecture.elements:
        if e.polarite != "demande" or e.operation not in _PLANIFIER:
            continue
        for genre, valeur in schema.jours_resolus(e, aujourdhui):
            if genre != "placement":
                continue
            if valeur == schema.QUESTION:
                question = True
            elif valeur >= aujourdhui:
                places.add(valeur)
    if question:
        typee = schema.QUESTION
    elif len(places) == 1:
        typee = next(iter(places))
    else:
        typee = "plusieurs" if places else None
    if regex is None and typee is None:
        return NA
    return ACCORD if typee == regex else DESACCORD


def accords(lecture, message: str, aujourdhui: date, refs: dict) -> dict:
    """{lecteur: ok|diff|na} pour LECTEURS. Ne leve jamais."""
    if lecture is None or not isinstance(message, str):
        return dict.fromkeys(LECTEURS, NA)
    calculs = {
        "heures": lambda: accord_heures(lecture, message),
        "dates": lambda: accord_dates(lecture, message, aujourdhui),
        "jour_vise": lambda: accord_jour_vise(lecture, message, aujourdhui),
        "date_visee": lambda: accord_date_visee(lecture, message, aujourdhui, refs or {}),
        "suppression": lambda: accord_suppression(lecture, message),
        "evenement_unique": lambda: accord_evenement_unique(lecture, message),
        "cette_semaine": lambda: accord_cette_semaine(lecture, message),
        "puces_date": lambda: accord_puces_date(lecture, message, aujourdhui),
    }
    sortie = {}
    for nom in LECTEURS:
        try:
            sortie[nom] = calculs[nom]()
        except Exception as e:  # noqa: BLE001 - une mesure ne casse pas un tour
            logger.warning("agent_v2 lire accord=%s illisible erreur=%s", nom, type(e).__name__)
            sortie[nom] = NA
    return sortie


def formater_accords(acc: dict) -> str:
    return ",".join(f"{nom}:{acc.get(nom, NA)}" for nom in LECTEURS)
