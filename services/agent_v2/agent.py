"""
La boucle unique (2026-09-29): COMPRENDRE+OUTILLER, RECONCILIER, RENDRE.

La boucle outille et alimente un registre ecrit par le runtime. Apres une
lecture ou un echec d'outil, le modele repond a la personne. Apres une mutation
reussie, le code rend son recu verifie. Les affirmations d'effet non confirmees
sont ecartees avant l'assemblage.

Depuis le 2026-09-14 (lots 1 a 3), un tour s'affiche en trois sections
streamees dans leur ordre final: FAITS (code), PROSE (boucle verifiee),
QUESTION (une seule, choisie par PRIORITE). done.response est exactement la
concatenation des deltas, et le message persiste porte ses boutons et ses
demandes en metadonnees pour que le tour suivant sache a quoi l'utilisateur
repond.

La surface publique porte les QUATRE points d'entree que core/views.py et le
banc exigent. views.py:861 lit result['response'] par indexation DIRECTE: une
cle manquante rend un 500 a l'utilisateur.
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import re
import time
from copy import copy
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextvars import copy_context
from typing import Optional
from types import SimpleNamespace

from django.conf import settings
from django.contrib.auth.models import User
from django.db import close_old_connections
from django.db.models import F
from django.utils import timezone
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (ModelRequest, ModelResponse, PartDeltaEvent,
                                  PartStartEvent, TextPart,
                                  ThinkingPart, ThinkingPartDelta, UserPromptPart)
from pydantic_ai.usage import RunUsage, UsageLimits

from core.models import BudgetJetonsJournalier, ConversationMessage, UploadedDocument
from services.agent_v2 import lecture, regles
from services.agent_v2 import memoire as memoire_prefs
from services.agent_v2 import jugement as _jugement
from services.agent_v2.mesure import verifier_prose
from services.agent_v2.modeles import modele_agir
from services.agent_v2.outils import outils_pour, outils_pour_le_modele
from services.agent_v2.prompts import prompt_agir, prompt_planification
from services.agent_v2.reconciliation import detecter_ecarts, reconcilier
from services.agent_v2.importation import inscrire_import
from services.agent_v2.rendu import plage
from services.agent_v2.redaction import (LECTURES_RENDUES, ReponseDire,
                                         bloc_factuel, bloc_reste, composer,
                                         contient_question, marqueurs_bruts,
                                         question_code, sans_tiret_long)
from services.agent_v2.registre import Registre
from services.agent_v2.validation_sortie import (jugement_exige_outil,
                                                 question_sans_outil, retour_validation)

logger = logging.getLogger(__name__)

BUDGET_ETAPES = 10
HISTORIQUE_MAX = 20


# ── Garde-fous de disponibilite et de cout ────────────────────────────────
#
# Deux risques documentes mais non bornes: un tour sans deadline murale
# (397 s mesures en prod le 2026-08-29) et une facture LLM sans plafond.
# La philosophie reste la meme que budget_epuise: le tour est tronque,
# jamais rate. Le registre survit, les faits deja vrais sont rendus.


def _delai_tour() -> float:
    """Duree murale maximale d'un tour (AGIR + DIRE), en secondes."""
    try:
        return max(1.0, float(getattr(settings, "AGENT_V2_DELAI_TOUR", 180.0)))
    except (TypeError, ValueError):
        return 180.0


def _budget_jetons_agir() -> int:
    try:
        return max(0, int(getattr(settings, "AGENT_V2_BUDGET_JETONS_AGIR", 300000)))
    except (TypeError, ValueError):
        return 300000


def _budget_jetons_dire() -> int:
    try:
        return max(0, int(getattr(settings, "AGENT_V2_BUDGET_JETONS_DIRE", 100000)))
    except (TypeError, ValueError):
        return 100000


def _budget_jetons_jour(user=None) -> int:
    """Plafond journalier de jetons pour l'utilisateur.

    Les comptes anonymes (mode sans inscription) ont un plafond resserre
    (anti-abus) : quand il est epuise, le tour suggere la creation d'un
    compte pour continuer. Un profil manquant vaut compte normal (direction
    prudente : on ne resserre jamais par defaut).
    """
    from core.anonyme import est_anonyme
    if user is not None and est_anonyme(user):
        try:
            return max(0, int(getattr(settings, "AGENT_V2_BUDGET_JETONS_JOUR_ANON", 400000)))
        except (TypeError, ValueError):
            return 400000
    try:
        return max(0, int(getattr(settings, "AGENT_V2_BUDGET_JETONS_JOUR", 2000000)))
    except (TypeError, ValueError):
        return 2000000


def _limites_jetons(budget: int) -> dict:
    """Parametres UsageLimits pour un budget jetons (total = entree + sortie).
    Un budget nul ou negatif desactive la garde plutot qu'imposer un plafond
    de zero qui tronquerait chaque tour."""
    if budget > 0:
        return {"total_tokens_limit": budget}
    return {}


def _jetons_phase(cout: dict) -> int:
    """Jetons consommes par une phase, depuis son dict _cout."""
    if not isinstance(cout, dict):
        return 0
    # Les jetons de raisonnement sont deja inclus dans les jetons de sortie.
    return sum(int(cout.get(k, 0) or 0) for k in ("entree", "sortie"))


def _budget_jour_epuise(user) -> bool:
    """Sans lecture du compteur, aucun nouvel appel payant n'est admis."""
    from services.agent_v2.trace_tour import trace_courante
    reservation = getattr(trace_courante.get(), 'reservation', None)
    if reservation is not None and reservation.user_id == user.pk:
        return reservation.fermee
    if _budget_jetons_jour(user) <= 0:
        return True
    try:
        ligne = BudgetJetonsJournalier.objects.filter(
            user=user, jour=timezone.localdate()).first()
    except Exception:  # noqa: BLE001
        return True
    if ligne is None:
        return False
    from services.agent_v2.budget_tour import plafond_garde_cout
    plafond = _budget_jetons_jour(user)
    return (ligne.jetons_mesures >= plafond or
            ligne.jetons + ligne.jetons_reserves >= plafond_garde_cout(user, plafond))


def _enregistrer_jetons(user, total: int) -> None:
    """Ajoute les jetons du tour au compteur journalier (increment atomique)."""
    if total <= 0:
        return
    ligne, _ = BudgetJetonsJournalier.objects.get_or_create(
        user=user, jour=timezone.localdate(), defaults={"jetons": 0})
    BudgetJetonsJournalier.objects.filter(pk=ligne.pk).update(
        jetons=F("jetons") + int(total))

# Une phrase TERMINEE dans un flux de texte: ponctuation finale, guillemets
# fermants eventuels, puis une espace. Tant que l'espace n'est pas arrivee,
# la phrase peut encore s'allonger (« 18 h. » contre « 18 h.30 »).
_FIN_DE_PHRASE_STREAMEE = re.compile(r"[.!?…][\"'»)\]]*\s")

# Une question par tour, la premiere presente dans cet ordre. Les gardes du
# code passent avant tout: une suppression retenue sans sa question serait un
# refus muet. Le formulaire passe avant le choix du modele, et DIRE ferme la
# marche.
PRIORITE = [
    "portee_jour", "portee_changement", "destructif", "heure_refusee", "creation_en_masse",
    "optimisation", "formulaire", "choix_modele", "question_libre", "chevauchement",
    "fin_recurrence", "creneaux", "dire",
]
CHIPS_MAX = 4

# Audit 2026-10-13 (#6): des actions ont ete tentees mais aucun rendu ne les
# a couvertes. Ce n'est pas de l'incomprehension, c'est un trou de rendu:
# on le dit honnetement au lieu d'accuser la demande.
REPLI_RESULTAT = ("J'ai traité ta demande mais je n'ai pas pu t'en montrer "
                  "le détail pour l'instant. Redis-le-moi si ce n'est pas "
                  "ce que tu attendais.")


def _parametres_journal(parametres: dict | None) -> dict:
    """Parametres d'outil expurges pour le journal d'execution.

    On garde les cles et la forme des valeurs (type, apercu court des
    chaines), jamais le texte integral ni les secrets. Suffit pour
    diagnostiquer « quel appel, avec quels criteres » sans rejouer les
    donnees personnelles dans les metadonnees.
    """
    journal = {}
    for cle, valeur in (parametres or {}).items():
        if isinstance(valeur, str):
            journal[cle] = {"type": "str",
                            "apercu": valeur[:40]}
        elif isinstance(valeur, bool):
            journal[cle] = {"type": "bool", "valeur": valeur}
        elif isinstance(valeur, (int, float)):
            journal[cle] = {"type": type(valeur).__name__, "valeur": valeur}
        elif isinstance(valeur, (list, tuple)):
            journal[cle] = {"type": "list", "longueur": len(valeur)}
        elif isinstance(valeur, dict):
            journal[cle] = {"type": "dict", "cles": sorted(valeur.keys())[:10]}
        elif valeur is None:
            journal[cle] = {"type": "null"}
        else:
            journal[cle] = {"type": type(valeur).__name__}
    return journal
# Quand la boucle a ANNONCE une liste sans la montrer et que le juge n'a pas
# tranche ce qu'il fallait lire: on demande quoi voir, on ne dit pas qu'on n'a
# pas compris (defaut A, prod du 2026-09-29).
REPLI_LISTE = "Tu veux voir quoi : ta journée, ta semaine ou tes tâches ?"
CHIPS_LISTE = [
    {"label": "Ma journée", "value": "Montre-moi ma journée"},
    {"label": "Ma semaine", "value": "Montre-moi ma semaine"},
    {"label": "Mes tâches", "value": "Montre-moi mes tâches"},
]
PROSE_FORMULAIRE = "Il me manque quelques précisions."
PROSE_REPRISE = "Il me faut juste cette précision avant de toucher à ton horaire."
PROSE_ANNULEE = "D'accord, je ne change rien."
PROSE_DELAI = ("J'ai mis trop de temps à te répondre, je m'arrête ici. "
               "Ce qui est déjà fait est affiché plus haut.")
# Style OpenCode : quand la boucle tombe en panne (tous les fournisseurs ont
# echoue), on le dit honnetement au lieu de pretendre ne pas avoir compris.
PROSE_PANNE = ("J'ai eu un problème technique en traitant ta demande. "
               "Rien n'a été modifié. Réessaie dans un moment.")
def prose_budget_jour(user, motif=None):
    from core.anonyme import est_anonyme
    if motif is None:
        from services.agent_v2.budget_tour import statut_usage
        try:
            motif = statut_usage(user)['bloque_par']
        except Exception:
            motif = 'indisponible'
    if motif == 'indisponible':
        return "Le contrôle du budget est indisponible. Réessaie plus tard."
    cause = ("ton quota de jetons mesurés" if motif == 'quota_mesure'
             else "le garde de coût des appels non mesurés")
    texte = (f"J'ai atteint ma limite d'IA pour aujourd'hui : {cause}. "
             "Réessaie demain à 0 h, heure de Toronto.")
    if est_anonyme(user):
        texte += " Crée un compte gratuit pour accéder à un quota plus élevé."
    return texte

# POOL DE THREADS REUTILISES, et le mot « reutilises » porte tout le poids.
#
# AGIR doit tourner a cote du generateur pour qu'on puisse emettre pendant
# qu'il travaille. La premiere version creait un thread NEUF par tour: mesure
# du 2026-08-28, le DEUXIEME tour d'un meme processus se bloque
# indefiniment, alors que trois tours dans le thread principal passent sans
# rien. Quatre tours sur un thread reutilise passent aussi, et douze tours
# concurrents sur un pool de quatre egalement.
#
# Consequence pratique: ne JAMAIS remplacer ce pool par un threading.Thread
# cree a la volee. Le symptome est un blocage silencieux au second message,
# donc invisible sur un essai unique.
_POOL_AGIR = ThreadPoolExecutor(
    max_workers=getattr(settings, "AGENT_V2_THREADS", 8),
    thread_name_prefix="agir",
)

# Le drainage attend par tranches plutot qu'indefiniment: si le thread meurt
# sans poser sa sentinelle, un get() sans delai gelerait la connexion SSE
# jusqu'au timeout du serveur.
ATTENTE_PENSEE = 0.5


# ── Chargeurs de boutons.py et outils.py (simulables par les tests) ──────


def _charger_question_forcee():
    """boutons.question_forcee, charge par une fonction pour que les tests la simulent."""
    from services.agent_v2.boutons import question_forcee
    return question_forcee


def _charger_creneaux_types():
    """boutons.creneaux_types (regle creneaux), charge par une fonction pour les tests."""
    from services.agent_v2.boutons import creneaux_types
    return creneaux_types


def _charger_appliquer_choix():
    """outils.appliquer_choix_en_attente, charge par une fonction pour les tests."""
    from services.agent_v2.outils import appliquer_choix_en_attente
    return appliquer_choix_en_attente


def _charger_tour_decide():
    """outils.tour_entierement_decide_par_le_code, ou None s'il n'existe pas.

    Contrat du round 6 avec les gardes: vrai quand le message ne fait que
    repondre (ou echouer a repondre) a une demande en attente, sans nouvelle
    requete. Absente, la fonction ne decide rien et AGIR tourne comme avant.
    """
    from services.agent_v2 import outils
    return getattr(outils, "tour_entierement_decide_par_le_code", None)


def _abandonnee(action) -> bool:
    return bool((action.donnees or {}).get("abandonnee_par_le_code"))


def _garde_a_retenu(registre: Registre) -> bool:
    """Une garde du code a-t-elle retenu une action ce tour ?"""
    for a in registre.actions:
        if a.succes or _abandonnee(a):
            continue
        d = a.donnees or {}
        if (isinstance(d.get("demande"), dict) or d.get("needs_confirmation")
                or d.get("requires_confirmation") or d.get("heure_dite")):
            return True
    return False


def _brouillon_interdit(registre: Registre) -> bool:
    """Le brouillon d'AGIR est-il ecarte du brief ce tour ?

    Revue de verite du round 4: _garde_a_retenu ignorait une date passee, une
    erreur d'outil et les autres echecs, et le brouillon racontait l'action
    ratee (« Je l'ai mis jeudi a 9 h, veux-tu... ? »). Toute action en echec
    ce tour, retenue ou non, ecarte le brouillon entier.
    """
    # Une demande laissee tombee par le code (D2) n'est pas un echec du tour:
    # son brouillon peut encore porter la question d'une nouvelle requete.
    return _garde_a_retenu(registre) or any(
        not a.succes and a.outil != "import_recent" and not _abandonnee(a)
        for a in registre.actions)


def _cout(resultat, duree: float) -> dict:
    """Ce qu'une phase a reellement coute: allers-retours, jetons, secondes.

    `usage()` peut manquer selon le fournisseur : les compteurs restent alors
    inconnus (None), avec usage_connu=False dans les metadonnees.
    """
    vide = {"etapes": 0, "entree": None, "sortie": None, "raisonnement": None,
            "cache": None, "duree": duree, "usage_connu": False}
    try:
        u = resultat.usage()
    except Exception:  # noqa: BLE001
        return vide
    # `details` porte ce que le fournisseur ajoute. Sur DeepSeek: les jetons de
    # RAISONNEMENT, et le partage entre succes et echecs de cache de prefixe.
    #
    # Un succes de cache ne prouve ni gratuite ni faible latence. Le compteur
    # standard du SDK prime ; les details fournisseur restent un secours.
    details = getattr(u, "details", None) or {}
    return {
        "etapes": getattr(u, "requests", 0) or 0,
        "entree": getattr(u, "input_tokens", 0) or 0,
        "sortie": getattr(u, "output_tokens", 0) or 0,
        "raisonnement": details.get("reasoning_tokens", 0) or 0,
        "cache": getattr(u, 'cache_read_tokens', 0) or details.get("prompt_cache_hit_tokens", 0) or 0,
        "duree": duree,
        "usage_connu": True,
    }
# Une lecture de semaine chargee depasse largement ce volume; on tronque plutot
# que de laisser un seul outil manger le contexte de la redaction.
EXTRAIT_MAX = 4000
BROUILLON_MAX = 1500
ECHANGE_MAX = 400


def _extrait(donnees: dict) -> str:
    brut = json.dumps(donnees, ensure_ascii=False, default=str)
    if len(brut) <= EXTRAIT_MAX:
        return brut
    return f"{brut[:EXTRAIT_MAX]}... (tronque)"


def _json_sur(valeur):
    """Les metadonnees passent en JSONField: tout ce qui n'est pas natif devient texte."""
    return json.loads(json.dumps(valeur, ensure_ascii=False, default=str))


def _chips_propres(chips, garder_option: bool) -> list[dict]:
    propres: list[dict] = []
    for chip in chips or []:
        if not isinstance(chip, dict) or not chip.get("label") or not chip.get("value"):
            continue
        propre = {"label": sans_tiret_long(str(chip["label"])),
                  "value": sans_tiret_long(str(chip["value"]))}
        if garder_option and chip.get("option") is not None:
            propre["option"] = chip["option"]
        propres.append(propre)
    return propres[:CHIPS_MAX]


def _chips_reponse(chips, demandes) -> list[dict]:
    """Les quick replies envoyees au front, avec le postback des demandes.

    Une puce qui appartient a une demande RENDUE porte `demande` (la cle) et
    `option` (l'id): au tap, le front les renvoie tels quels et l'egalite
    d'identifiants remplace la comparaison de texte. Champs additifs: v1,
    le MCP et un front ancien les ignorent sans rien casser.
    """
    propres = _chips_propres(chips, garder_option=False)
    par_texte: dict[tuple, tuple] = {}
    for d in demandes or []:
        cle = (d or {}).get("cle") if isinstance(d, dict) else None
        if not cle:
            continue
        for chip in d.get("chips") or []:
            if not isinstance(chip, dict) or chip.get("option") is None:
                continue
            ref = (sans_tiret_long(str(chip.get("label") or "")),
                   sans_tiret_long(str(chip.get("value") or "")))
            par_texte.setdefault(ref, (cle, chip["option"]))
    for propre in propres:
        info = par_texte.get((propre["label"], propre["value"]))
        if info:
            propre["demande"], propre["option"] = info
    return propres


def _rang(motif) -> int:
    # Un motif inconnu est traite comme informatif, au rang du chevauchement.
    return PRIORITE.index(motif) if motif in PRIORITE else PRIORITE.index("chevauchement")


class PlannerAgentV2:
    """Le nom est fixe par benchmarks/harness.py, qui l'importe tel quel."""

    # La vue chat ne passe `tap` (postback structure d'une puce) qu'aux
    # agents qui l'annoncent: v1 ne le connait pas et ne doit pas le recevoir.
    accepte_tap = True

    def __init__(self, user: Optional[User] = None):
        self.user = user
        # Poses ici et pas seulement dans le flux: _boucle et _historique sont
        # appelables directement (banc, tests), et une instance a demi
        # initialisee leverait un AttributeError loin de sa cause.
        self._tache: str = ""
        self._exclu: Optional[int] = None
        self._file_pensees: Optional[queue.Queue] = None
        self._message_brut: Optional[str] = None
        self._tap: Optional[dict] = None
        self._registre_courant: Optional[Registre] = None

    def pousser_pensee(self, texte: str) -> None:
        """Emet un fragment de raisonnement vers le flux, s'il y a un flux.

        Sans file (appel direct depuis le banc ou un test), l'appel est sans
        effet: la primitive ne doit jamais imposer au reste du code de savoir
        s'il tourne dans un contexte streame.
        """
        if self._file_pensees is not None and texte:
            self._file_pensees.put(("thinking", texte))

    def signaler_outil(self, action) -> None:
        """Diffuse un appel d'outil vers le flux, s'il y a un flux.

        On envoie de quoi AFFICHER (nom, succes, message rendu par l'outil),
        jamais le dictionnaire d'arguments: il peut porter du contenu
        utilisateur, et le flux part vers le client.
        """
        if self._file_pensees is None or action is None:
            return
        self._file_pensees.put(("tool", {
            "id": action.id,
            "name": action.outil,
            "ok": bool(action.succes),
            "message": action.message or "",
        }))

    # ------------------------------------------------------------------ AGIR

    async def _sur_evenements(self, _contexte, evenements) -> None:
        """Capte le RAISONNEMENT au fil de sa production, et le TEXTE final.

        run_sync execute le graphe en entier, donc tous les outils tournent;
        run_stream_sync s'arreterait a la premiere sortie « finale » et
        sauterait les appels suivants. Le handler est le seul moyen d'avoir
        les deux.

        Deux defauts corriges le 2026-09-14: la version precedente lisait
        `content_delta` sur tout evenement, donc les brouillons en francais
        de la boucle (TextPartDelta) se melaient au raisonnement anglais du
        volet; et elle ignorait PartStartEvent, qui porte le PREMIER fragment
        de chaque partie (d'ou « user wants » sans « The »).

        Boucle unique (2026-09-29): seul le raisonnement est streame en
        direct. Le texte final est une sortie structuree (JSON): il n'est
        jamais pousse tel quel, la reponse verifiee part en deltas ordonnes
        apres composition.
        """
        async for evenement in evenements:
            if isinstance(evenement, PartStartEvent):
                if isinstance(evenement.part, ThinkingPart):
                    self.pousser_pensee(evenement.part.content)
            elif isinstance(evenement, PartDeltaEvent):
                if isinstance(evenement.delta, ThinkingPartDelta):
                    self.pousser_pensee(evenement.delta.content_delta)

    def _boucle(self, user: User, message: str, registre: Registre) -> ReponseDire | None:
        """Une boucle outillee : plusieurs requetes modele peuvent se succeder.

        Le modele comprend, outille et rend une reponse STRUCTUREE
        (ReponseDire): prose + refs vers le registre + lecture typee pour les
        regles du code. Plus d'etapes LIRE ni DIRE separees.

        Le registre est alimente par l'adaptateur d'outils a chaque execution:
        cette methode ne l'ecrit pas elle-meme, et c'est voulu. Une action ne
        peut entrer dans le registre qu'en ayant reellement ete executee.
        """
        # Les regles de la garde (confirmation, heure dite, portee d'un jour)
        # lisent le message TAPE, jamais sa version enrichie du document.
        brut = self._message_brut if self._message_brut is not None else message
        outils = outils_pour_le_modele(
            user, registre, message_du_tour=message, tache=self._tache,
            signaler=self.signaler_outil, message_brut=brut, tap=self._tap)
        # instructions= et non system_prompt=: pydantic-ai n'ajoute les system
        # prompts QUE si l'historique est vide. Sur tout tour de suivi, la
        # boucle tournait sans date, sans table de decision ni semaine type,
        # et a place une revision en 2025 (banc du 2026-09-14). Les
        # instructions partent a chaque requete.
        # Facon OpenCode : pas de output_type (la sortie structuree forcee via
        # tool_choice=required est incompatible avec le mode thinking sur tous
        # les providers). La reflexion reste ACTIVE. Le modele agit librement
        # avec ses outils ; sa reponse aux lectures et aux echecs passe par
        # verifier_prose. Le code rend les recus des mutations confirmees.
        from services.agent_v2.budget_tour import modele_borne, BudgetEpuise
        agent = Agent(
            modele_borne(modele_agir(), 'planning'),
            instructions=prompt_agir(user),
            tools=outils,
            output_retries=2,
            # Pas de output_type, pas de model_settings : thinking actif.
        )

        reprises_sortie = 0

        async def valider_sortie(_contexte, sortie: str) -> str:
            nonlocal reprises_sortie
            if registre.mutations():
                # La relecture est faite AVANT la décision finale. Elle est
                # répétée après la boucle pour le rendu et les écarts.
                from asgiref.sync import sync_to_async
                try:
                    await sync_to_async(reconcilier, thread_sensitive=True)(user, registre)
                except Exception as exc:  # noqa: BLE001 - le reçu reste non confirmé
                    logger.warning("agent_v2 relecture avant sortie impossible type=%s",
                                   type(exc).__name__)
            faits_controle = (bloc_factuel(registre)
                             if self._lecture_sans_demande(registre) else "")
            question_exige_outil = False
            if question_sans_outil(sortie, registre):
                extrait_message = (message if len(message) <= 1200 else
                                   f"{message[:600]}\n[...]\n{message[-600:]}")
                extrait_sortie = (sortie if len(sortie) <= 1200 else
                                  f"{sortie[:600]}\n[...]\n{sortie[-600:]}")
                verdicts = await asyncio.to_thread(
                    _jugement.juger,
                    {"message": extrait_message,
                     "reponse_assistant": extrait_sortie},
                    {"question_requiert_outil": _jugement.q_question_requiert_outil()},
                )
                verdict = (verdicts or {}).get("question_requiert_outil") or {}
                question_exige_outil = jugement_exige_outil(verdict)
            retour = retour_validation(sortie, registre,
                                       faits_controle=faits_controle,
                                       question_exige_outil=question_exige_outil)
            if retour and (question_exige_outil
                           or reprises_sortie < 2):
                reprises_sortie += 1
                logger.info("agent_v2 sortie rejetee reprise=%d actions=%d",
                            reprises_sortie, len(registre.actions))
                raise ModelRetry(retour)
            # Le garde final reste actif si la limite de reprises est atteinte.
            return sortie

        # Les doubles de test anciens n'implémentent pas le décorateur.
        enregistrer = getattr(agent, "output_validator", None)
        if callable(enregistrer):
            enregistrer(valider_sortie)

        depart = time.perf_counter()
        usage = RunUsage()
        resultat = None
        try:
            resultat = self._executer_boucle(
                agent, message,
                usage=usage,
                delai=self._delai_restant(),
                message_history=self._historique(user),
                usage_limits=UsageLimits(request_limit=BUDGET_ETAPES,
                                         **_limites_jetons(_budget_jetons_agir())),
                event_stream_handler=self._sur_evenements,
            )
        except BudgetEpuise as exc:
            registre.budget_epuise = True
            registre.motif_budget = exc.motif
            return None
        except UsageLimitExceeded:
            # Le tour est tronque, pas rate: les outils deja executes ont
            # ecrit. Le bloc factuel le dira, c'est tout l'interet du registre.
            # Declenche par le budget d'etapes OU le budget jetons.
            registre.budget_epuise = True
            registre.motif_budget = "etapes" if usage.requests >= BUDGET_ETAPES else "jetons"
            return None
        except asyncio.TimeoutError:
            registre.delai_depasse = True
            return None
        finally:
            self._cout_boucle = _cout(resultat or SimpleNamespace(usage=lambda: usage),
                                     time.perf_counter() - depart)
            if resultat is None and not usage.total_tokens:
                self._cout_boucle.update(usage_connu=False, entree=None, sortie=None,
                                        raisonnement=None, cache=None)
        # Facon OpenCode : le modele rend du texte libre (pas de output_type).
        # On l'emballe dans un ReponseDire minimal : la prose passe par
        # verifier_prose, les questions par poser_question/present_choices,
        # les recus des mutations par le registre.
        # Le champ lecture reste vide : les regles typees ne s'activent pas,
        # le systeme fonctionne sans elles.
        from services.agent_v2.redaction import ReponseDire
        texte = getattr(resultat, "output", None)
        if isinstance(texte, str) and texte.strip():
            return ReponseDire(ouverture=texte.strip())
        if isinstance(texte, ReponseDire):
            return texte
        return None

    @staticmethod
    def _executer_boucle(agent, message, **kwargs):
        """La boucle, avec retry facon OpenCode.

        - Backoff exponentiel 2s, 4s, 8s puis 10s ; max 10 tentatives.
        - `retry-after` honore, plafonne a 15 min.
        - Timeouts : max 3 essais.
        - Seules les erreurs reessayables (transport, rate limit, 5xx) sont
          retentees ; les 4xx et erreurs de validation remontent aussitot.
        - Le stream vide (« Streamed response ended without content or tool
          calls », defaut prod du 2026-09-29) est reessayable.
        """
        from services.agent_v2 import retry as _retry
        delai = kwargs.pop('delai', None)

        async def _exec_async():
            from services.agent_v2.modeles import fermer_modele
            timeouts = 0
            derniere = None
            for tentative in range(1, _retry.TENTATIVES_MAX + 1):
                try:
                    return await agent.run(message, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    derniere = exc
                    vide = 'without content or tool calls' in str(exc)
                    if not vide and not _retry.est_reessayable(exc):
                        raise
                    if _retry.est_timeout(exc):
                        timeouts += 1
                        if timeouts >= _retry.TENTATIVES_MAX_TIMEOUT:
                            raise
                    if tentative >= _retry.TENTATIVES_MAX:
                        raise
                    attente = _retry.delai_avant_reessai(tentative, exc)
                    logger.warning("Boucle: echec tentative %d/%d type=%s, reessai dans %.1fs",
                                   tentative, _retry.TENTATIVES_MAX,
                                   type(exc).__name__, attente)
                    await asyncio.sleep(attente)
                finally:
                    await fermer_modele(getattr(agent, 'model', None))
            raise derniere

        def _exec_sync():
            from services.agent_v2.modeles import fermer_modele
            timeouts = 0
            derniere = None
            for tentative in range(1, _retry.TENTATIVES_MAX + 1):
                try:
                    return agent.run_sync(message, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    derniere = exc
                    vide = 'without content or tool calls' in str(exc)
                    if not vide and not _retry.est_reessayable(exc):
                        raise
                    if _retry.est_timeout(exc):
                        timeouts += 1
                        if timeouts >= _retry.TENTATIVES_MAX_TIMEOUT:
                            raise
                    if tentative >= _retry.TENTATIVES_MAX:
                        raise
                    attente = _retry.delai_avant_reessai(tentative, exc)
                    logger.warning("Boucle: echec tentative %d/%d type=%s, reessai dans %.1fs",
                                   tentative, _retry.TENTATIVES_MAX,
                                   type(exc).__name__, attente)
                    import time as _time
                    _time.sleep(attente)
                finally:
                    try:
                        import asyncio as _aio
                        _aio.run(fermer_modele(getattr(agent, 'model', None)))
                    except Exception:  # noqa: BLE001
                        pass
            raise derniere

        if delai is not None:
            return asyncio.run(asyncio.wait_for(_exec_async(), timeout=max(0.0, delai)))
        return _exec_sync()

    def process_message_stream(
        self, user, message, attachment=None, *, use_streaming=True,
        generate_quick_replies=False, tap=None, conversation_id=None, question_answer=None,
    ):
        from services.agent_v2.trace_tour import trace_courante
        from services.agent_v2.budget_tour import ouvrir_trace, fermer_trace, BudgetIndisponible
        trace = trace_courante.get()
        proprietaire = trace is None
        if proprietaire:
            try:
                trace = ouvrir_trace(user)
            except BudgetIndisponible as exc:
                from services.agent_v2.budget_tour import BudgetEpuise
                prose = prose_budget_jour(user, exc.motif if isinstance(exc, BudgetEpuise)
                                          else 'indisponible')
                yield {'type': 'delta', 'text': prose}
                yield {'type': 'done', 'response': prose, 'quick_replies': [],
                       'blocks_created': [], 'tasks_created': [], 'raisonnement': ''}
                return
        if trace.reservation is None or trace.reservation.user_id != user.pk:
            raise BudgetIndisponible('réservation étrangère au tour')
        self._trace = trace
        jeton = trace_courante.set(trace)
        try:
            yield from self._process_message_stream(user, message, attachment,
                use_streaming=use_streaming, generate_quick_replies=generate_quick_replies, tap=tap,
                conversation_id=conversation_id, question_answer=question_answer)
        finally:
            try:
                if proprietaire:
                    fermer_trace(trace)
            finally:
                trace_courante.reset(jeton)

    def _resoudre_conversation(self, user, conversation_id=None):
        """La conversation du tour : celle demandee (si elle appartient a
        l'utilisateur), sinon la plus recente non archivee, sinon une neuve."""
        from core.models import Conversation
        if conversation_id:
            try:
                conv = Conversation.objects.get(pk=conversation_id, user=user)
                conv.save(update_fields=['updated_at'])
                return conv
            except (Conversation.DoesNotExist, ValueError):
                pass
        conv = (Conversation.objects
                .filter(user=user, archivee=False)
                .order_by('-updated_at').first())
        if conv is None:
            conv = Conversation.objects.create(user=user)
        else:
            conv.save(update_fields=['updated_at'])
        return conv

    def _process_message_stream(
        self,
        user: User,
        message: str,
        attachment: Optional[UploadedDocument] = None,
        *,
        use_streaming: bool = True,
        generate_quick_replies: bool = False,
        tap: Optional[dict] = None,
        conversation_id=None,
        question_answer=None,
    ):
        """Contrat SSE additif: status, thinking, tool, delta, done. Les deltas
        arrivent dans l'ordre final (faits, prose, question) et done.response
        en est exactement la concatenation.

        `tap` est le postback structure d'une puce touchee:
        {"demande": cle, "option": id}. Il ne vaut que contre la demande en
        attente qui porte cette cle (memes gardes, meme fenetre); tout le
        reste du tour lit le message comme avant."""
        self.user = user
        depart_tour = getattr(self, '_depart_admission', None) or time.perf_counter()
        self._depart_tour = depart_tour  # lu par _delai_restant (deadline du tour)
        self._cout_boucle = {}
        # Le message TAPE, avant tout enrichissement: c'est lui que lisent la
        # garde et les boutons forces.
        self._message_brut = message
        self._tap = tap if isinstance(tap, dict) and tap.get("demande") and tap.get("option") else None
        # La conversation du tour (le sous-classe l'a deja resolue ; ici on
        # s'assure qu'elle existe pour les appels directs).
        if getattr(self, '_conversation', None) is None:
            self._conversation = self._resoudre_conversation(user, conversation_id)
        # Verdict du juge plombe en pre-boucle (voie rapide sociale): la question
        # consultation, consultee a un autre moment du tour, decide si une
        # lecture reussie s'affiche. Reinitialise a chaque tour: sans verdict
        # frais, jamais de masquage.
        # Style OpenCode : plus de verdict pre-boucle. Sans classification
        # prealable, une lecture reussie s'affiche toujours (les mutations ne
        # sont jamais masquees de toute facon).
        self._verdict_consultation = {}
        self._journaliser_reponse_formulaire(user, message)

        # Persiste d'abord, puis exclut CETTE ligne de l'historique par son id.
        # v1 devait s'en remettre a un filet (B9: message sauve, relu, puis
        # rajoute, donc duplique a chaque requete); ici la duplication est
        # structurellement impossible.
        courant = ConversationMessage.objects.create(
            user=user, role="user", content=message,
            conversation=self._conversation,
            metadata={"tap": self._tap} if self._tap else {})
        documents = getattr(self, '_chat_attachments', None) or ([attachment] if attachment else [])
        courant.attachments.set(documents)
        self._exclu = courant.pk
        # Titre auto : le premier message donne son nom à la conversation
        # (style OpenCode : génération de titre, ici déterministe).
        if (self._conversation and self._conversation.titre in ("Nouvelle conversation", "Conversation")
                and not ConversationMessage.objects.filter(
                    conversation=self._conversation, role="user").exclude(pk=courant.pk).exists()):
            self._conversation.titre = message.strip()[:60] or "Nouvelle conversation"
            self._conversation.save(update_fields=['titre', 'updated_at'])
        # Identifie CE tour pour les cles d'idempotence: deux tours
        # distincts peuvent legitimement refaire la meme action, un meme
        # tour rejoue ne doit l'executer qu'une fois.
        self._tache = f"{user.pk}:{courant.pk}"

        registre = Registre()
        registre.lecture_seule = False
        registre.echeance = time.monotonic() + self._delai_restant()
        self._registre_courant = registre

        # Le document et l'import recent DOIVENT entrer dans le message vu par
        # AGIR. Sans cela, un horaire envoye est perdu en silence et l'agent
        # decrit le planning deja en base en laissant croire qu'il a lu le
        # document (defaut observe le 2026-08-26 sur un tour reel). L'envoi
        # d'horaire est le premier chemin d'entree du produit.
        message_enrichi = message
        if question_answer is not None:
            from services.agent_v2.reponse_question import (consommer_reponse,
                                                            ReponseQuestionRefusee)
            try:
                precise = consommer_reponse(user, self._conversation, courant, question_answer)
            except ReponseQuestionRefusee as exc:
                texte_refus = str(exc)
                question_reposee = getattr(exc, "question", None)
                assistant = ConversationMessage.objects.create(
                    user=user, role="assistant", content=texte_refus,
                    conversation=self._conversation,
                    metadata={"agent": "v2", "en_reponse_a": courant.pk,
                              "question_posee": bool(question_reposee), "quick_replies": [],
                              "question_custom": question_reposee,
                              "interactive_inputs": question_reposee.get("fields", []) if question_reposee else []})
                yield {"type": "delta", "text": texte_refus}
                resultat_refus = {"type": "done", "assistant_message_id": assistant.pk,
                                  "conversation_id": self._conversation.pk if self._conversation else None,
                                  "response": texte_refus, "quick_replies": [],
                                  "question_posee": bool(question_reposee),
                                  "blocks_created": [], "tasks_created": [], "raisonnement": ""}
                if question_reposee:
                    resultat_refus["question_custom"] = question_reposee
                    resultat_refus["interactive_inputs"] = question_reposee.get("fields", [])
                yield resultat_refus
                return
            message_enrichi += "\n\nRÉPONSE VALIDÉE AU FORMULAIRE : " + json.dumps(
                precise, ensure_ascii=False)
        # Lu AVANT l'attente: seul un document qui bascule pendant ce tour
        # compte comme « traite ce tour » (equivalence avec v1 detaillee dans
        # la docstring de boutons_forces).
        deja_traite = attachment is not None and attachment.processed
        attente_fin = time.monotonic() + getattr(settings, "ATTACHMENT_WAIT_SECONDS", 45)
        for index, document in enumerate(documents):
            attente_restante = max(0, attente_fin - time.monotonic())
            for evenement, complement in self._contexte_document(
                user, document, wait_seconds=attente_restante,
                include_recent=index == len(documents) - 1,
            ):
                if evenement:
                    yield evenement
                if complement:
                    message_enrichi = f"{message_enrichi}\n\n{complement}"
        if not documents:
            for evenement, complement in self._contexte_document(user, None):
                if evenement:
                    yield evenement
                if complement:
                    message_enrichi = f"{message_enrichi}\n\n{complement}"
        attachment_traite_ce_tour = (
            attachment is not None and not deja_traite and attachment.processed)

        # L'import fait par le systeme entre au registre AVANT AGIR, comme
        # une action deja accomplie. Sans cela DIRE lit « registre vide » et
        # propose de renvoyer l'horaire qui vient d'etre importe (deux tours
        # reels le 2026-09-01). Detail dans importation.py.
        try:
            if documents:
                for document in documents:
                    inscrire_import(registre, user, document)
            else:
                inscrire_import(registre, user, None)
        except Exception:  # noqa: BLE001 - un recap absent vaut mieux qu'un tour tombe
            logger.error("Import du document non inscrit au registre", exc_info=True)

        # Commande memoire deterministe (memoire.py): « souviens-toi que... »,
        # « oublie... », « que sais-tu de moi ? ». Executee par le code AVANT
        # AGIR, sans passer par le modele. Le resultat entre au registre pour
        # que DIRE l'annonce; AGIR ne recoit que le reste eventuel du message.
        # Un tap structure (chip d'une question en attente) n'est jamais une
        # commande memoire.
        try:
            commande_memoire = (memoire_prefs.interpreter(message)
                                if self._tap is None else None)
        except Exception:  # noqa: BLE001 - une commande illisible ne casse pas le tour
            commande_memoire = None
            logger.error("Interpretation memoire impossible", exc_info=True)
        if commande_memoire is not None:
            try:
                from services.agent.tools.base import ToolResult
                if commande_memoire.genre == "memoriser":
                    phrase = memoire_prefs.memoriser(user, commande_memoire.enonce)
                elif commande_memoire.genre == "oublier":
                    phrase = memoire_prefs.oublier(user, commande_memoire.enonce)
                else:
                    phrase = memoire_prefs.lister(user)
                registre.ajouter("memoire", {"commande": commande_memoire.genre},
                                 ToolResult(success=True, message=phrase))
                if commande_memoire.pure and message_enrichi == message:
                    # Rien d'autre a faire ce tour: AGIR ne doit pas relire
                    # une commande memoire comme une demande de planification.
                    message_enrichi = ("(Commande memoire deja executee par le "
                                       "code: aucune action de planification.)")
                elif commande_memoire.reste:
                    message_enrichi = commande_memoire.reste
            except Exception:  # noqa: BLE001 - le tour continue sans la memoire
                logger.error("Commande memoire non executee", exc_info=True)

        # La reponse a une question du tour precedent s'execute par le CODE,
        # avant AGIR : un tap sur « Tous les jeudis » supprime la serie sans
        # qu'un modele ait a le refaire (ni a pouvoir le rater). AGIR recoit
        # le bilan pour ne pas le rejouer; le message brut reste intact.
        # Sans decideur pre-boucle, on tente toujours d'appliquer : sans
        # demande en attente ou sans correspondance, c'est un no-op.
        self._file_pensees = queue.Queue()
        choix: list = []
        try:
            # `tap` seulement s'il existe: les simulations de tests et tout
            # remplacant sans ce parametre restent valides.
            extra_tap = {"tap": self._tap} if self._tap else {}
            choix = list(_charger_appliquer_choix()(
                user, registre, message, tache=self._tache,
                signaler=self.signaler_outil, **extra_tap) or [])
        except Exception:  # noqa: BLE001 - un choix non applique se redemande
            logger.error("Choix en attente non appliques", exc_info=True)
            choix = []
        yield from self._vider_file()
        choix = [c for c in choix if isinstance(c, dict)]
        choix_code = sum(1 for c in choix if c.get("action_id"))
        # Une reponse floue a une question gardee ne perd pas la demande: les
        # gardes la reposent UNE fois et l'inscrivent au registre avec
        # reposee_par_le_code (contrat du round 6). La voix ne relit plus
        # l'attente elle-meme: une seule source decide de reposer.
        reemises = [a.donnees["demande"] for a in registre.actions
                    if (a.donnees or {}).get("reposee_par_le_code")
                    and isinstance(a.donnees.get("demande"), dict)]
        resumes = [str(c["resume"]) for c in choix if c.get("resume")]
        if resumes:
            message_enrichi = (f"{message_enrichi}\n\nSUITE AU CHOIX DE L'UTILISATEUR:\n- "
                               + "\n- ".join(resumes))

        # CHEMIN RAPIDE (D6). Banc du round 5, s05-2: le code avait repose la
        # demande a 0,06 s, puis AGIR a raisonne 105,6 s sans outil et son
        # brouillon a ete jete. Quand le message ne fait que repondre (ou ne
        # pas repondre) a une demande en attente, le tour se construit depuis
        # la decision du code: ni AGIR, ni DIRE.
        # Round 8: une piece jointe est toujours une demande pour AGIR, meme
        # quand le texte tape ne fait que repondre.
        par_le_code = attachment is None and self._tour_decide(registre, message)
        # Compteur journalier epuise: on saute les phases LLM (LIRE, AGIR,
        # DIRE). Le tour continue avec le registre tel quel: les choix du
        # code et les faits restent rendus, sans couter un jeton de plus.
        budget_jour_epuise = not par_le_code and _budget_jour_epuise(user)
        if budget_jour_epuise:
            logger.warning("agent_v2 budget jetons journalier epuise user=%s", user.pk)
        # Voie rapide sociale: une salutation ou un remerciement ne merite
        # pas la boucle lourde (15k tokens de contexte). Le juge semantique
        # tranche, jamais une liste de mots. La reponse breve entre dans un
        # ReponseDire pour suivre le chemin de composition normal.
        voie_rapide = (
            not par_le_code and not budget_jour_epuise
            and self._voie_rapide_sociale(user, message, attachment)
        )
        if voie_rapide:
            texte_rapide = self._reponse_rapide(user, message)
            if texte_rapide:
                reponse_boucle = ReponseDire(ouverture=texte_rapide)
                logger.info("agent_v2 voie rapide sociale user=%s", user.pk)
            else:
                voie_rapide = False
        # Boucle unique: un seul appel modele par tour (plus de LIRE ni de
        # DIRE separes). Le chemin rapide et le budget epuise sautent la
        # boucle; le registre (choix du code, faits) est rendu tel quel.
        reponse_boucle: ReponseDire | None = reponse_boucle if voie_rapide else None
        panne = None
        raisonnement = ""
        if par_le_code or budget_jour_epuise or voie_rapide:
            self._file_pensees = None
        else:
            yield {"type": "status", "text": "Réflexion..."}
            reponse_boucle, panne, raisonnement = yield from self._boucle_en_fond(
                user, message_enrichi, registre)
        if panne is not None:
            # Une panne de la boucle ne doit pas effacer ce que les outils
            # ont deja ecrit: le registre survit et le tour continue.
            logger.error("Boucle a echoue type=%s", type(panne).__name__)
        if not isinstance(reponse_boucle, ReponseDire):
            reponse_boucle = None

        etat: dict = {}
        if registre.mutations():
            yield {"type": "status", "text": "Je relis ton planning..."}
            etat = reconcilier(user, registre)
            detecter_ecarts(registre)

        # La lecture typee vient de la boucle elle-meme (champ `lecture` de
        # sa reponse structuree): plus d'appel LIRE separe. Elle alimente les
        # deux regles qui en ont besoin (formulaire_cours, creneaux).
        configurees = frozenset() if par_le_code else lecture.regles_actives()
        lecture_typee = (self._lecture_de_boucle(user, message, reponse_boucle)
                         if configurees else None)
        actives = configurees if lecture_typee is not None else frozenset()
        # La regle pose son formulaire dans le registre et produit la phrase
        # factuelle qui l'accompagne. Cette phrase prime sur une prose du
        # modele qui peut contredire les candidats verifies.
        regle, prose_regle = "-", ""
        if regles.FORMULAIRE_COURS in actives:
            try:
                prose_regle = regles.appliquer_formulaire_cours(
                    lecture_typee, registre, attachment=attachment, reemises=reemises)
            except Exception as e:  # noqa: BLE001 - une regle ne casse pas un tour
                logger.warning("agent_v2 regle=%s illisible erreur=%s",
                               regles.FORMULAIRE_COURS, type(e).__name__)
                prose_regle = ""
            if prose_regle:
                regle = regles.FORMULAIRE_COURS

        # La question du tour est choisie AVANT les faits: les actions
        # retenues que cette question couvre n'ont pas a etre redites, les
        # autres recoivent leur ligne « pas encore ».
        gagnant = self._choisir_question(
            user, message, attachment, registre, attachment_traite_ce_tour,
            reemises=reemises, sans_forcee=par_le_code,
            lecture_typee=lecture_typee if regles.CRENEAUX in actives else None)
        if gagnant and gagnant.get("regle"):
            regle = gagnant["regle"]
        par_demande = bool(gagnant) and gagnant.get("source") == "demande"
        cles_posees = set(gagnant.get("cles_posees") or []) if par_demande else set()

        # Une lecture qui n'a servi qu'a preparer un formulaire ou un choix ne
        # se deverse pas au-dessus de la question (banc du round 4, s02-1).
        sans_lecture = bool(gagnant) and (
            gagnant.get("source") == "formulaire" or gagnant.get("motif") in ("choix_modele", "question_libre")
        ) and not any(a.succes and a.est_mutation for a in registre.actions)
        titres_vises = [((d or {}).get("cible") or {}).get("titre")
                        for d in (gagnant.get("demandes") or [])] if par_demande else []
        affichage = self._assurer_consultation(user, message, registre)
        if affichage is not registre and any(a.succes and a.outil in LECTURES_RENDUES
                                             for a in affichage.actions):
            sans_lecture = False
        faits = bloc_factuel(affichage, cles_posees=cles_posees, sans_lecture=sans_lecture,
                             titres_vises=[t for t in titres_vises if t])
        # La section RESTE: demande contre place, une soustraction rendue par
        # du code. Elle rejoint les faits AVANT la redaction et le flux: le
        # manque se nomme au meme instant que le succes qu'il tempere.
        reste = bloc_reste(message, registre)
        if reste:
            faits = f"{faits}\n{reste}" if faits else reste
        faits = sans_tiret_long(faits or "")
        faits_pour_controle = faits

        # LIRE n'est pas MONTRER. Une lecture reussie pour raisonner (le
        # modele a lu sans que personne ne demande a voir) ne s'affiche
        # pas; seule une lecture qui repond a une demande est rendue. Le
        # registre garde les faits: seul l'affichage est coupe.
        # Audit 2026-10-13 (#4): l'ancien `faits = ""` effacait tout le bloc
        # factuel, y compris les explications d'echec deja composees. On ne
        # masque que les listes lues pour raisonner; le bilan d'execution
        # (succes comme echecs) reste visible.
        lecture_masquee = self._lecture_sans_demande(registre)
        if lecture_masquee:
            logger.info("agent_v2 lecture masquee: listes cachees, bilan conserve")
            faits = sans_tiret_long(bloc_factuel(
                affichage, cles_posees=cles_posees, sans_lecture=True,
                titres_vises=[t for t in titres_vises if t]) or "")
            # La section RESTE (calculee plus haut) n'est pas une liste lue
            # pour raisonner: elle survit au masquage comme le bilan.
            if reste:
                faits = f"{faits}\n{reste}" if faits else reste
                faits = sans_tiret_long(faits or "")

        supprimees = 0
        fuites: list[str] = []
        # Chemin rapide et budget journalier epuise: la boucle n'a pas tourne,
        # il n'y a aucune prose de modele a soumettre.
        t0_verif = time.perf_counter()
        if par_le_code or budget_jour_epuise:
            compo = composer(None, registre, faits, gagnant)
        elif reponse_boucle is not None:
            # Verifier-puis-rendre: la prose de la boucle ne survit que contre
            # des refs verifiees (ids d'actions reelles du registre). Sur un
            # tour de reprise, le code parle seul (PROSE_REPRISE plus bas).
            brut = (reponse_boucle.model_copy(update={"ouverture": "", "suite": ""})
                    if reemises else reponse_boucle)
            brut, supprimees, fuites = verifier_prose(
                brut, registre, autoriser_conversation=voie_rapide,
                sans_faits=not faits)
            # Une lecture masquée ne doit pas réapparaître dans la prose du
            # modèle. On la fournit au dédoublonneur comme preuve de contrôle,
            # tout en n'émettant que les faits autorisés plus bas.
            compo = composer(brut, registre,
                             faits_pour_controle if lecture_masquee else faits,
                             gagnant)
        else:
            # La boucle n'a rien rendu (panne, budget d'etapes): les faits
            # parlent, et le repli aussi (plus bas).
            compo = composer(None, registre, faits, gagnant)

        # Une lecture ou un échec d'outil a déjà été renvoyé au modèle. Sa
        # réponse finale prend la place du résumé `public` si elle survit aux
        # gardes. Le rendu déterministe reste le repli lorsque le modèle n'a
        # rien produit d'utilisable. Les mutations réussies gardent leurs
        # reçus relus et les questions du code gardent leur priorité.
        mutation_reussie = any(a.succes and a.est_mutation for a in registre.actions)
        modele_prioritaire = bool(
            reponse_boucle is not None and not par_le_code and not budget_jour_epuise
            and not registre.budget_epuise and not lecture_masquee and not reemises
            and not mutation_reussie and not gagnant and not faits
            and not prose_regle)
        lecture_repondue_par_modele = False
        if modele_prioritaire:
            # Composer directement sans résumé évite qu'un dédoublonneur
            # retire la liste demandée puis laisse seulement un commentaire.
            candidate = composer(brut, registre, "", None)
            if candidate.prose or candidate.question:
                faits = ""
                compo = candidate
                lecture_repondue_par_modele = True
        duree_verif = time.perf_counter() - t0_verif

        emis: list[str] = []
        if faits:
            emis.append(faits)
            yield {"type": "delta", "text": faits}

        # Zero tiret long dans ce que lit l'utilisateur, quelle que soit la
        # source (banc du round 3, s06-1).
        prose, question = sans_tiret_long(compo.prose), sans_tiret_long(compo.question)
        motif, chips = compo.motif, compo.chips
        if prose_regle:
            prose = sans_tiret_long(prose_regle)
        if reemises and par_demande and cles_posees & {d.get("cle") for d in reemises}:
            # Revue de lisibilite du round 4: DIRE lisait une reponse de garde
            # et ecrivait « D'accord, je garde ta chimie. » juste avant la
            # question de suppression reposee. Sur ce tour, le code parle seul.
            # Round 6 (D5): « avant de toucher a ton horaire » serait faux
            # apres une mutation reussie; les faits parlent, puis la question.
            prose = "" if mutation_reussie else PROSE_REPRISE
        if not question and not gagnant and mutation_reussie:
            # Defaut B (prod du 2026-09-29): l'occurrence sautee, rien a la
            # place, et la reponse souhaitait bonne chance. Le code demande.
            manquant = self._remplacant_manquant(
                user, self._message_brut or message, registre)
            if manquant:
                question, motif, chips = manquant, "remplacant_manquant", []

        annulee = (any(c.get("decision_code") == "annulee" for c in choix)
                   or any((a.donnees or {}).get("decision_code") == "annulee"
                          for a in registre.actions))
        if par_le_code and annulee and not faits and not prose and not question:
            # Une garde fermee par « laisse faire »: une ligne, pas un silence.
            prose = PROSE_ANNULEE
        formulaire = gagnant.get("interactive_inputs") if gagnant and \
            gagnant.get("source") == "formulaire" else None
        if panne is not None and reponse_boucle is None and faits and not prose:
            # La boucle est tombee. Le delai depasse se dit, parce que les
            # faits ne disent pas pourquoi le tour s'arrete la. Une panne sans
            # delai n'a rien a ajouter: les faits rendus juste au-dessus disent
            # ce qui a change, et « Voici ce qui a change » les redisait sans
            # rien apprendre a personne.
            if registre.delai_depasse:
                prose = PROSE_DELAI
        if formulaire and not faits and not prose:
            prose = PROSE_FORMULAIRE
        if budget_jour_epuise and not faits and not prose and not question and not formulaire:
            # Le compteur journalier est epuise et le tour n'a rien produit:
            # on le dit plutot que de laisser un silence. Un compte anonyme
            # au plafond resserre se voit proposer la sortie (creer un
            # compte), pas l'attente.
            prose, motif = prose_budget_jour(user), "budget_jour"
        elif not faits and not prose and not question and not formulaire:
            # La boucle a repondu mais n'a ni lu ni mute, et sa prose est
            # tombee: avant de dire qu'on n'a pas compris, le code lit ce que
            # la personne voulait voir (defaut A, prod du 2026-09-29).
            rien_lu = not any(a.succes and (a.est_mutation
                                            or a.outil in LECTURES_RENDUES)
                              for a in registre.actions)
            outil_secours = ""
            if reponse_boucle is not None and rien_lu:
                outil_secours = self._lecture_de_secours(
                    user, self._message_brut or message, registre)
            if outil_secours:
                faits = sans_tiret_long(bloc_factuel(registre) or "")
                motif = "lecture_secours"
            if faits:
                emis.append(faits)
                yield {"type": "delta", "text": faits}
            elif compo.lecture_sans_liste:
                question, motif = REPLI_LISTE, "dire"
                chips = [dict(c) for c in CHIPS_LISTE]
            elif not registre.actions:
                # Style OpenCode : aucun repli « je n'ai pas compris ». Si la
                # boucle est tombee en panne, on le dit honnetement. Si le
                # modele n'a simplement rien produit d'exploitable, le tour
                # s'arrete sans mentir : un tour vide vaut mieux qu'un faux
                # aveu d'incomprehension.
                if panne is not None:
                    prose = PROSE_PANNE
                    motif = "panne"
            else:
                question, motif, chips = REPLI_RESULTAT, "dire", []

        if prose:
            morceau = ("\n\n" if emis else "") + prose
            emis.append(morceau)
            yield {"type": "delta", "text": morceau}
        if question:
            morceau = ("\n\n" if emis else "") + question
            emis.append(morceau)
            yield {"type": "delta", "text": morceau}
        response = "".join(emis)

        quick_replies = _chips_reponse(chips, compo.demandes if par_demande else [])
        # Une question restee dans la prose compte aussi: sinon des puces
        # differees viennent contredire la question (banc du round 3, s05-2).
        question_posee = bool(gagnant) or bool(question) or contient_question(prose)
        lecture_reussie = any(a.succes and a.outil in LECTURES_RENDUES
                              for a in registre.actions)
        # Un masquage voulu n'est pas le defaut « lu sans rien montrer »:
        # il a son compteur propre, sinon la metrique deviendrait aveugle.
        lecture_sans_liste = (compo.lecture_sans_liste
                              or (lecture_reussie and not faits
                                  and not lecture_repondue_par_modele)) and not lecture_masquee
        try:
            marqueurs = list(marqueurs_bruts(response))
        except Exception:  # noqa: BLE001 - une mesure ne casse pas un tour
            marqueurs = []
        rejetees = compo.rejetees

        # Recueillie avant la question quand une regle est configuree; sinon
        # ici seulement, une fois la reponse figee. La ligne du tour nomme la
        # regle qui a decide.
        metadonnees_lire = {"lecture": "boucle"}

        # Une seule ligne par tour, mais pas toujours au meme niveau: une
        # reference rejetee est un mensonge que la garantie structurelle vient
        # d'attraper, et une fuite est une affirmation d'action sans recu.
        # Ce sont LES deux signaux du projet; en INFO ils se noieraient dans
        # le bruit et personne ne les verrait passer.
        cout = getattr(self, "_cout_boucle", None) or {}
        anormal = bool(rejetees or fuites)
        logger.log(
            logging.WARNING if anormal else logging.INFO,
            "agent_v2 tour actions=%d rejetees=%d fuites=%d supprimees=%d ecarts=%d%s"
            " boucle=%.1fs/%dep/%d->%dj/r%d/c%d verif=%.2fs"
            " asked=%d form=%d choices=%d read_without_list=%d masques=%d redites=%d"
            " raw_marker_count=%d"
            " motif=%s choix_code=%d chemin=%s tour=%.2fs",
            len(registre.actions),
            rejetees,
            len(fuites),
            supprimees,
            len(registre.ecarts),
            f" types={','.join(fuites)}" if fuites else "",
            # CHRONO. Il entre dans LA ligne du tour plutot que d'en ouvrir une
            # seconde: le contrat est une ligne agregee par tour, et c'est
            # aussi ce qui permet de correler duree et verite sans jointure.
            # En production le 2026-08-29, la mediane etait de 57 s et un tour
            # a atteint 397 s sans qu'aucun log ne dise ou passait le temps: on
            # ne pouvait qu'inferer des ecarts entre lignes httpx.
            cout.get("duree", 0.0), cout.get("etapes", 0),
            cout.get("entree") or 0, cout.get("sortie") or 0,
            cout.get("raisonnement") or 0, cout.get("cache") or 0,
            duree_verif,
            # MESURE DES QUESTIONS (lot 3g): a-t-on demande, par quel canal,
            # combien de boutons, une lecture sans liste, du texte machine.
            1 if question_posee else 0,
            1 if formulaire else 0,
            len(quick_replies),
            1 if lecture_sans_liste else 0,
            1 if lecture_masquee else 0,
            compo.redites,
            len(marqueurs),
            motif or "-",
            choix_code,
            # Latence du chemin rapide (D6) contre la boucle complete.
            "code" if par_le_code else "boucle",
            time.perf_counter() - depart_tour,
        )

        question_affichee = "" if motif == "formulaire" else question
        question_custom = None
        if gagnant:
            for action in reversed(registre.actions):
                candidate = (action.donnees or {}).get("question_custom")
                if action.outil != "ask_question_custom" or not action.succes or not candidate:
                    continue
                if candidate.get("kind") in ("form", "text") and formulaire:
                    question_custom = candidate
                    break
                if candidate.get("kind") == "choice" and question_affichee == candidate.get("question"):
                    question_custom = candidate
                    break

        # Les appels réconcilient leur propre provision. Aucun second débit par phase.

        metadonnees = {
            "agent": "v2",
            "contrat_tour": 'boucle',
            "usage": list(getattr(getattr(self, '_trace', None), 'usages', []))
                     or [{**cout, 'phase': 'planning'}],
            "en_reponse_a": self._exclu,
            "quick_replies": quick_replies,
            "interactive_inputs": formulaire or [],
            "question_posee": question_posee,
            "question": question_affichee,
            "question_motif": motif or "",
            "question_custom": question_custom,
            "activation_notifications": any(
                a.outil == 'send_notification'
                or (a.succes and a.outil in ('manage_reminder', 'manage_recurrence')
                    and (a.outil == 'manage_reminder'
                         and a.parametres.get('action') in ('create', 'update')
                         or a.outil == 'manage_recurrence'
                         and a.parametres.get('action') == 'create'
                         and (a.parametres.get('kind') == 'reminder'
                              or bool(a.parametres.get('due_time')))))
                or (a.succes and a.outil in ('create_task', 'update_task')
                    and bool((a.donnees.get('task') or {}).get('alert_at')))
                for a in registre.actions
            ),
            # SEULEMENT les demandes rendues dans la question gagnante: une
            # demande que l'utilisateur n'a pas vue ne doit jamais pouvoir
            # etre autorisee par sa reponse.
            "demandes": [
                {**d, "chips": _chips_propres(d.get("chips"), garder_option=True)}
                for d in compo.demandes
            ] if par_demande else [],
            "faits_rendus": faits,
            "raw_markers": marqueurs,
            "lecture_sans_liste": lecture_sans_liste,
            "lecture_masquee": lecture_masquee,
            "redites": compo.redites,
            # Le nom que l'utilisateur a donne au cours du formulaire du code:
            # au tour de la reponse, create_block le garde (outils.py).
            "formulaire_nom": next((str((a.donnees or {}).get("nom") or "")
                                    for a in registre.actions
                                    if a.outil == "present_form" and a.succes
                                    and (a.donnees or {}).get("par_le_code")), ""),
            "actions": [
                {"id": a.id, "outil": a.outil, "succes": bool(a.succes),
                 "par_le_code": bool((a.donnees or {}).get("par_le_code")),
                 # Journal d'execution (audit 2026-10-13, ref OpenCode:
                 # la session persiste chaque appel complet). Parametres
                 # expurges: cles + type, aprecu court pour les chaines,
                 # jamais de secret ni de texte integral.
                 "parametres": _parametres_journal(a.parametres),
                 "issue": a.issue or "",
                 "cause": a.cause or "",
                 "duree_ms": round(a.duree_ms or 0.0, 1)}
                for a in registre.actions
            ],
        }
        metadonnees.update(metadonnees_lire)
        # Le serveur connait l'ensemble des appareils du compte, pas celui
        # qui affiche ce chat. Le frontend verifie CE terminal avant de montrer
        # le bouton. Un navigateur abonne ne doit pas masquer le bouton iPhone.
        assistant = ConversationMessage.objects.create(
            user=user, role="assistant", content=response,
            conversation=self._conversation,
            metadata=_json_sur(metadonnees))

        evenement = {
            "type": "done",
            "assistant_message_id": assistant.pk,
            "conversation_id": self._conversation.pk if self._conversation else None,
            "response": response,
            "quick_replies": quick_replies,
            "activation_notifications": metadonnees['activation_notifications'],
            "blocks_created": self._crees(registre, "create_block", "created"),
            "tasks_created": self._crees(registre, "create_task", "task"),
            "raisonnement": raisonnement,
            "question_posee": question_posee,
            "question": question_affichee,
            "question_motif": motif or "",
            "question_custom": question_custom,
        }
        if formulaire:
            evenement["interactive_inputs"] = formulaire
            # Mesure: le denominateur du taux de remplissage. Une ligne par
            # formulaire AFFICHE, donc par tour, meme si le modele en a
            # presente plusieurs (seul le dernier atteint l'utilisateur).
            logger.info(
                "formulaire presente user=%s champs=%d types=%s",
                user.id, len(formulaire),
                ",".join(str(champ.get("type", "")) for champ in formulaire),
            )
        yield evenement

    @staticmethod
    def _evenement_de_file(element) -> dict:
        # La file transporte deux formes: un fragment de raisonnement (texte
        # nu) et un appel d'outil (dictionnaire deja pret).
        genre, charge = element
        return ({"type": genre, "text": charge} if isinstance(charge, str)
                else {"type": genre, **charge})

    def _vider_file(self):
        """Emet ce que la file contient deja, sans attendre."""
        if self._file_pensees is None:
            return
        while True:
            try:
                element = self._file_pensees.get_nowait()
            except queue.Empty:
                return
            if element is not None:
                yield self._evenement_de_file(element)

    def _delai_restant(self) -> float:
        """Secondes restantes avant la deadline du tour.

        Apres l'echeance, aucun temps supplementaire n'est accorde. Hors tour,
        le delai complet est disponible pour les appels directs.
        """
        depart = getattr(self, "_depart_tour", None)
        from services.agent_v2.trace_tour import trace_courante
        trace = trace_courante.get()
        if trace is not None:
            return trace.restant()
        if depart is None:
            return _delai_tour()
        return max(0.0, _delai_tour() - (time.perf_counter() - depart))

    def _boucle_en_fond(self, user: User, message_enrichi: str, registre: Registre):
        """La boucle dans le pool, ses pensees streamees.

        Rend (ReponseDire|None, panne, raisonnement): le raisonnement est le
        texte pense collecte pendant le drainage, pour l'evenement done.

        La boucle tourne dans un THREAD pour qu'on puisse emettre pendant
        qu'elle travaille. Mesure du 2026-08-28: sur une demande multi-etapes
        la phase LLM occupe 15 s des 25 s du tour, et l'utilisateur n'avait
        rien a lire pendant ce temps. Le raisonnement etait bien capte, mais
        emis apres coup: il decrivait une reflexion deja terminee.
        """
        reponse, panne = None, None
        pensees: list[str] = []
        file_agir = self._file_pensees
        # Deadline murale du tour: le drainage s'interrompt quand elle est
        # depassee au lieu de tenir la connexion SSE ouverte indefiniment
        # L'appel asynchrone recoit la meme echeance. Les outils synchrones
        # verifient aussi la deadline ; une mutation DB tardive est annulee
        # avant commit. Une requete externe deja envoyee n'est pas retractable.
        echeance = time.monotonic() + self._delai_restant()

        def travailler():
            nonlocal reponse, panne
            # Ce thread vit hors du cycle de requete Django, qui ferme les
            # connexions: on s'en charge des deux cotes.
            close_old_connections()
            try:
                reponse = self._boucle(user, message_enrichi, registre)
            except Exception as e:  # noqa: BLE001
                panne = e
            finally:
                close_old_connections()
                file_agir.put(None)  # sentinelle de fin

        contexte = copy_context()
        futur = _POOL_AGIR.submit(contexte.run, travailler)
        fragments = 0
        while True:
            restant = echeance - time.monotonic()
            if restant <= 0:
                registre.delai_depasse = True
                futur.cancel()
                panne = panne or TimeoutError("delai du tour depasse pendant la boucle")
                logger.warning("agent_v2 delai depasse pendant la boucle user=%s", user.pk)
                break
            try:
                element = file_agir.get(timeout=min(ATTENTE_PENSEE, restant))
            except queue.Empty:
                # Filet: si le thread s'est termine sans poser sa sentinelle
                # (arret brutal du worker), on sort au lieu d'attendre pour
                # toujours et de geler la connexion SSE.
                if futur.done():
                    break
                continue
            if element is None:
                break
            fragments += 1
            evt = self._evenement_de_file(element)
            if evt["type"] == "thinking":
                pensees.append(evt["text"])
            yield evt
        if not registre.delai_depasse:
            try:
                futur.result(timeout=max(0.1, echeance - time.monotonic()))
            except FuturesTimeoutError:
                # Paranoia: le drainage a vu la sentinelle mais le thread ne
                # rend pas la main. Meme traitement que ci-dessus.
                registre.delai_depasse = True
                panne = panne or TimeoutError(
                    "delai du tour depasse: la boucle ne rend pas la main")
                logger.warning("agent_v2 la boucle ne rend pas la main user=%s", user.pk)
        self._file_pensees = None

        if registre.delai_depasse and not futur.done():
            try:
                futur.result(timeout=0.25)
            except FuturesTimeoutError:
                pass
        return reponse, panne, "".join(pensees)

    @staticmethod
    def _lecture_de_boucle(user: User, message: str,
                           reponse_boucle: ReponseDire | None):
        """La lecture typee pour les regles, produite par la boucle unique.

        Remplace l'appel LIRE separe: le champ `lecture` de la reponse
        structuree traverse les memes regles (lecture_du_tour), sur une
        preparation faite par le code avec le message TAPE.
        """
        lecture_boucle = getattr(reponse_boucle, "lecture", None)
        if lecture_boucle is None:
            return None
        try:
            preparation = lecture.preparer(user, message)
        except Exception:  # noqa: BLE001
            return None
        suivi = SimpleNamespace(preparation=preparation)
        resultat = SimpleNamespace(lecture=lecture_boucle, statut=lecture.OK)
        try:
            return regles.lecture_du_tour(suivi, resultat)
        except Exception:  # noqa: BLE001 - une lecture ne casse pas un tour
            logger.error("Lecture de boucle illisible", exc_info=True)
            return None

    @staticmethod
    def _tour_decide(registre: Registre, message: str) -> bool:
        """Le code a-t-il tout decide ce tour (D6) ? Faux au moindre doute."""
        try:
            decide = _charger_tour_decide()
            return bool(decide(registre, message)) if callable(decide) else False
        except Exception:  # noqa: BLE001 - dans le doute, AGIR tourne
            logger.error("Decision du code illisible", exc_info=True)
            return False

    @staticmethod
    def _dernier_mot_de_l_agent(user: User) -> str:
        """Ce que l'assistant vient de dire, pour juger ce qui lui repond.

        Un message n'a pas le meme sens seul ou apres une question. Borne a
        400 caracteres: le juge a besoin du sens, pas de la liste affichee.
        """
        from core.models import ConversationMessage
        dernier = (ConversationMessage.objects
                   .filter(user=user, role="assistant")
                   .order_by("-pk")
                   .values_list("content", flat=True)
                   .first())
        return (dernier or "")[:400]

    def _voie_rapide_sociale(self, user: User, message: str, attachment) -> bool:
        """Le message merite-t-il la voie rapide (simple interaction sociale) ?

        La decision est SEMANTIQUE (juge Jev, question typee), jamais une
        liste de mots ni une regex: la doctrine l'interdit. Les garde-fous
        sont STRUCTURELS (pas d'intention): pas de piece jointe, pas de tap,
        pas de demande en attente, message court. Seuil de confiance eleve
        (0.9): rater une optimisation vaut mieux que rater une vraie demande.
        """
        try:
            # Garde-fous structurels: jamais de voie rapide si le tour a
            # autre chose a traiter qu'un simple message texte court.
            if attachment is not None:
                return False
            if self._tap is not None:
                return False
            if len(message.split()) > 8:
                return False
            from services.agent_v2 import demandes as _demandes
            if _demandes.demandes_en_attente(user):
                return False
            # Le JUGE tranche, pas une liste de mots.
            # Le message ET ce qui le precede: une reponse a une question
            # n'est pas une simple politesse, meme avec les memes mots.
            # Deux questions, UN appel: le juge traite le dict en une fois. La
            # seconde (consultation) ne sert pas la voie rapide; elle est
            # plombee pour le rendu, ou elle decide si une lecture reussie
            # s'affiche. Le cout marginal est nul: l'appel etait deja paye.
            resultats = _jugement.juger(
                {"message": message,
                 "agent_a_dit": self._dernier_mot_de_l_agent(user)},
                {"sociale": _jugement.q_interaction_sociale(),
                 "consultation": _jugement.q_consultation()})
            self._verdict_consultation = (resultats or {}).get("consultation") or {}
            rep = (resultats or {}).get("sociale") or {}
            return (
                rep.get("statut") == _jugement.STATUT_DECISION
                # Une reponse noul est un BOOLEEN (jugement.py l.507), pas la
                # chaine « oui »: la comparer a « oui » rendait cette voie
                # morte, et chaque salutation payait un tour complet.
                and rep.get("valeur") is True
                and float(rep.get("confiance") or 0) >= 0.9
            )
        except Exception:  # noqa: BLE001 - dans le doute, la boucle tourne
            logger.warning("Voie rapide sociale illisible", exc_info=True)
            return False

    def _lecture_sans_demande(self, registre: Registre) -> bool:
        """Une lecture reussie qu'aucune demande ne justifie: a masquer.

        Garde de rendu, separee de la voie rapide: la question consultation
        du juge (plombee en pre-boucle) dit si la personne demandait a VOIR
        quelque chose. « aucune » a >= 0,9 masque; l'incertitude ne masque
        jamais: une demande sans reponse est pire qu'un affichage non
        demande. Les mutations ne sont jamais masquees: leur recu est du.
        Sans verdict pre-boucle (gardes, par_le_code), pas de masquage.
        """
        if not any(a.succes and a.outil in LECTURES_RENDUES
                   for a in registre.actions):
            return False
        if any(a.succes and a.est_mutation for a in registre.actions):
            return False
        verdict = getattr(self, "_verdict_consultation", None) or {}
        return (
            verdict.get("statut") == _jugement.STATUT_DECISION
            and verdict.get("valeur") == "aucune"
            and float(verdict.get("confiance") or 0) >= 0.9
        )

    def _assurer_consultation(self, user: User, message: str, registre: Registre) -> Registre:
        """Sans decideur pre-boucle, rien a assurer : le modele lit ce qu'il
        veut via ses outils, et le registre reflete ce qui a ete execute."""
        return registre

    def _lecture_de_secours(self, user: User, message: str,
                            registre: Registre) -> str:
        """Le tour n'a rien lu ni mute: le code lit ce que la personne veut voir.

        La decision est SEMANTIQUE (juge Jev, question typee), jamais une liste
        de mots. Seuil eleve (0.9): afficher une liste que personne n'a
        demandee est pire que poser la question. Rend le nom de l'outil
        execute, ou une chaine vide.
        """
        from services.agent_v2 import outils as _outils
        outil_de = {
            "journee": ("get_today_schedule", {}),
            "demain": ("get_today_schedule",
                       {"date": (timezone.localdate()
                                 + timedelta(days=1)).isoformat()}),
            "semaine": ("get_week_schedule", {}),
            "taches": ("list_tasks", {}),
        }
        try:
            resultats = _jugement.juger(
                message, {"consultation": _jugement.q_consultation()})
            rep = (resultats or {}).get("consultation") or {}
            if rep.get("statut") != _jugement.STATUT_DECISION:
                return ""
            if float(rep.get("confiance") or 0) < 0.9:
                return ""
            choix = outil_de.get(str(rep.get("valeur") or ""))
            if choix is None:
                return ""
            outil, parametres = choix
            action = _outils.executer_lecture_par_le_code(
                user, registre, outil, tache=self._tache, message=message,
                brut=message, **parametres)
            return outil if action is not None and action.succes else ""
        except Exception:  # noqa: BLE001 - un filet ne casse pas un tour
            logger.warning("Lecture de secours illisible", exc_info=True)
            return ""

    def _remplacant_manquant(self, user: User, message: str,
                             registre: Registre) -> str:
        """Une occurrence sautee, rien a la place: quelle question poser ?

        Rend la question, ou une chaine vide. La decision est SEMANTIQUE (juge
        Jev): « annule mon cours de mardi » est un retrait legitime, et rien
        ne doit etre demande. Seuil eleve, prudent dans le doute.
        """
        sauts = [a for a in registre.actions
                 if a.succes and a.outil == "skip_block_occurrence"]
        if not sauts:
            return ""
        if any(a.succes and a.outil in ("schedule_task_at",
                                        "replace_block_occurrence",
                                        "create_block")
               for a in registre.actions):
            return ""
        try:
            resultats = _jugement.juger(
                message, {"remplacant": _jugement.q_remplacant_annonce()})
            rep = (resultats or {}).get("remplacant") or {}
            if (rep.get("statut") != _jugement.STATUT_DECISION
                    or rep.get("valeur") is not True
                    or float(rep.get("confiance") or 0) < 0.9):
                return ""
            donnees = sauts[-1].donnees or {}
            debut, fin = donnees.get("start_time"), donnees.get("end_time")
            if debut and fin:
                return f"Tu mets quoi à la place, de {plage(debut, fin)} ?"
            return "Tu mets quoi à la place ?"
        except Exception:  # noqa: BLE001 - un filet ne casse pas un tour
            logger.warning("Filet du remplacant illisible", exc_info=True)
            return ""

    def _reponse_rapide(self, user: User, message: str) -> str:
        """Repond a une interaction sociale sans la boucle lourde.

        Prompt minimal, aucun outil, aucun contexte planning: le message
        n'attend qu'une reponse sociale breve. En cas d'echec, chaine vide
        (l'appelant bascule sur la boucle normale).
        """
        try:
            from services.agent_v2.modeles import modele_agir
            agent = Agent(
                modele_agir(),
                instructions=(
                    "Tu es l'assistant Planner, chaleureux et direct, "
                    "tutoiement, francais quebecois. Reponds en UNE phrase "
                    "breve a ce simple message social. Ne parle jamais de "
                    "planning, d'horaire ou de taches: l'utilisateur n'a "
                    "rien demande."
                ),
            )
            resultat = self._executer_boucle(agent, message, delai=self._delai_restant())
            texte = (getattr(resultat, "output", "") or "").strip()
            return texte if isinstance(texte, str) else ""
        except Exception:  # noqa: BLE001 - la boucle normale prend le relais
            logger.warning("Reponse rapide impossible", exc_info=True)
            return ""

    def _choisir_question(self, user: User, message: str, attachment,
                          registre: Registre, attachment_traite_ce_tour: bool,
                          reemises=(), sans_forcee: bool = False, lecture_typee=None):
        """La question UNIQUE du tour, selon PRIORITE, ou None.

        Rend {"source", "motif", "question", "chips", "demandes",
        "cles_posees", "interactive_inputs"}. Les demandes viennent du
        registre (gardes du code, present_choices), dans l'ordre des actions,
        puis des demandes reemises apres une reponse floue (une cle deja au
        registre n'est pas doublee). Une demande abandonnee par le code (D2)
        n'est jamais reposee: elle a sa ligne dans les faits. `sans_forcee`:
        sur le chemin rapide, aucune question forcee ne s'ajoute a la
        decision du code.
        """
        demandes = [a.donnees["demande"] for a in registre.actions
                    if isinstance((a.donnees or {}).get("demande"), dict)
                    and not _abandonnee(a)]
        deja = {d.get("cle") for d in demandes}
        demandes += [d for d in reemises or [] if d.get("cle") not in deja]
        formulaire = self._dernier_formulaire(registre)
        rang_formulaire = PRIORITE.index("formulaire")

        if demandes:
            haut = min(_rang(d.get("motif")) for d in demandes)
            if not formulaire or haut < rang_formulaire:
                code = self._question_des_demandes(demandes)
                if code:
                    return code

        if formulaire:
            return {"source": "formulaire", "motif": "formulaire", "question": "",
                    "chips": [], "demandes": [], "cles_posees": [],
                    "interactive_inputs": formulaire}

        if sans_forcee:
            return None
        if lecture_typee is not None:
            # Regle creneaux: la jambe typee passe d'abord; si elle ne rend
            # rien, question_forcee tourne exactement comme sur main.
            try:
                typee = _charger_creneaux_types()(user, message, attachment, registre, lecture_typee)
            except Exception as e:  # noqa: BLE001 - un bouton ne fait jamais tomber un tour
                logger.warning("agent_v2 regle=creneaux illisible erreur=%s", type(e).__name__)
                typee = None
            if isinstance(typee, dict):
                chips = _chips_propres(typee.get("chips"), garder_option=False)
                texte = (typee.get("question") or "").strip()
                if texte or chips:
                    return {"source": "forcee", "motif": typee.get("motif") or "creneaux",
                            "question": texte, "chips": chips, "demandes": [],
                            "cles_posees": [], "interactive_inputs": None,
                            "regle": "creneaux"}
        try:
            forcee = _charger_question_forcee()(
                user, message, attachment, registre, attachment_traite_ce_tour)
        except Exception:  # noqa: BLE001 - un bouton ne fait jamais tomber un tour
            logger.error("Question forcee indisponible", exc_info=True)
            forcee = None
        if isinstance(forcee, dict):
            chips = _chips_propres(forcee.get("chips"), garder_option=False)
            texte = (forcee.get("question") or "").strip()
            if texte or chips:
                return {"source": "forcee", "motif": forcee.get("motif") or "creneaux",
                        "question": texte, "chips": chips, "demandes": [],
                        "cles_posees": [], "interactive_inputs": None}
        return None

    @staticmethod
    def _question_des_demandes(demandes: list[dict]):
        texte, chips, cles = question_code(demandes)
        texte = (texte or "").strip()
        chips = _chips_propres(chips, garder_option=True)
        if not texte and not chips:
            return None
        cles = [c for c in (cles or []) if c]
        rendues = set(cles)
        posees: list[dict] = []
        vues: set = set()
        for d in demandes:
            cle = d.get("cle")
            if cle in rendues and cle not in vues:
                vues.add(cle)
                posees.append({**d, "chips": chips})
        if posees:
            motif = posees[0].get("motif") or ""
        else:
            motif = min(demandes, key=lambda d: _rang(d.get("motif"))).get("motif") or ""
        return {"source": "demande", "motif": motif, "question": texte,
                "chips": chips, "demandes": posees, "cles_posees": cles,
                "interactive_inputs": None}

    def _deux_derniers_echanges(self, user: User) -> str:
        """Les quatre derniers messages avant celui-ci, courts, pour DIRE."""
        try:
            lignes = list(ConversationMessage.objects
                          .filter(user=user).exclude(pk=self._exclu)
                          .order_by("-created_at", "-pk")[:4])
        except Exception:  # noqa: BLE001 - un contexte absent ne casse pas un tour
            logger.debug("Historique court illisible", exc_info=True)
            return ""
        sortie = []
        for ligne in reversed(lignes):
            qui = "Utilisateur" if ligne.role == "user" else "Assistant"
            contenu = " ".join((ligne.content or "").split())
            if len(contenu) > ECHANGE_MAX:
                contenu = f"{contenu[:ECHANGE_MAX]}..."
            sortie.append(f"{qui}: {contenu}")
        return "\n".join(sortie)

    @staticmethod
    def _journaliser_reponse_formulaire(user: User, message: str) -> None:
        """Mesure: numerateur (repondu) et abandon explicite (passe) du
        formulaire. Les deux phrases sont celles que le frontend envoie
        (InteractiveInputs.tsx, ChatContainer.tsx)."""
        propre = (message or "").strip()
        if propre.startswith("Voici mes réponses"):
            logger.info("formulaire repondu user=%s", user.id)
        elif propre == "On verra ça plus tard, continuons sans formulaire.":
            logger.info("formulaire passe user=%s", user.id)

    @staticmethod
    def _dernier_formulaire(registre: Registre):
        """Le dernier formulaire presente avec succes, releve du registre.

        Le prompt d'AGIR recommande present_form pour demander plusieurs
        infos d'un coup, et la vue ne relaie que la cle interactive_inputs
        du resultat. Sans ce releve, v2 presenterait des formulaires que
        l'utilisateur ne verrait jamais."""
        for a in reversed(registre.actions):
            if a.outil in ("present_form", "ask_question_custom") and a.succes:
                return (a.donnees or {}).get("interactive_inputs")
        return None

    def process_message(
        self,
        user: User,
        message: str,
        attachment: Optional[UploadedDocument] = None,
        generate_quick_replies: bool = True,
        tap: Optional[dict] = None,
        conversation_id=None,
        question_answer=None,
    ) -> dict:
        """Enveloppe non streamee: draine le flux, seule source de verite."""
        done: dict = {}
        question_extra = {"question_answer": question_answer} if question_answer is not None else {}
        for event in self.process_message_stream(
            user, message, attachment,
            use_streaming=False,
            generate_quick_replies=generate_quick_replies,
            tap=tap,
            conversation_id=conversation_id,
            **question_extra,
        ):
            if event.get("type") == "done":
                done = {k: v for k, v in event.items() if k != "type"}
        return done

    def quick_replies_for(
        self, user: User, user_message: str, assistant_response: str,
    ) -> list[dict]:
        """Une vue l'appelle et avale les exceptions: sans cette methode, les
        chips disparaitraient en silence pour tout compte bascule.

        Les suggestions n'ont rien a voir avec la verite d'action, et v1 les
        rend bien: on delegue plutot que de dupliquer.
        La chip « Mémoriser ? » (memoire inferee) passe devant: un tap
        rejoue la voie explicite, confirmation incluse.
        """
        chips: list[dict] = []
        try:
            chip = memoire_prefs.chip_inference(user, user_message or "")
            if chip:
                chips.append(chip)
        except Exception:  # noqa: BLE001 - une suggestion ne remonte jamais d'erreur
            logger.debug("Chip memoire indisponible", exc_info=True)
        try:
            from services.agent.agent import PlannerAgent
            chips.extend(PlannerAgent().quick_replies_for(
                user, user_message, assistant_response) or [])
        except Exception:  # noqa: BLE001 - une suggestion ne remonte jamais d'erreur
            logger.debug("Suggestions indisponibles", exc_info=True)
        return chips

    # ------------------------------------------------------------------ util

    @staticmethod
    def _contexte_document(user: User, attachment, *, wait_seconds=None, include_recent=True):
        """Rend des couples (evenement a emettre, complement de message).

        Les deux formateurs viennent de v1 et sont repris tels quels: ils
        portent des consignes produit affinees par l'audit (ne jamais nier un
        import, ne jamais promettre un resume qu'on ne livrera pas, presenter
        les blocs comme le RESULTAT de l'import). Les reecrire les ferait
        deriver.
        """
        from services.agent.agent import PlannerAgent
        v1 = PlannerAgent()

        if attachment is not None:
            if not attachment.processed:
                # Meme attente cooperative que v1: gunicorn tourne en workers
                # gevent avec monkey.patch_all(), ce sommeil ne suspend que
                # cette conversation. Une reponse vraie en un tour vaut mieux
                # qu'une promesse cassee en trois secondes.
                attente = (getattr(settings, "ATTACHMENT_WAIT_SECONDS", 45)
                           if wait_seconds is None else wait_seconds)
                if attente:
                    yield {"type": "status", "text": "J'analyse ton document..."}, None
                    for tic in range(int(attente * 2)):
                        time.sleep(0.5)
                        attachment.refresh_from_db()
                        if attachment.processed:
                            break
                        if attachment.processing_error:
                            # L'analyse a ECHOUE: inutile d'attendre la borne
                            # de 45 s pour annoncer un "retard" qui n'en est
                            # pas un. Le contexte ci-dessous le dit franchement.
                            break
                        if tic and tic % 16 == 0:
                            yield ({"type": "status",
                                    "text": "J'analyse ton document... (presque fini)"}, None)
            yield None, v1._build_attachment_context(attachment)

        # Vaut AVEC ou SANS piece jointe: un « c'est bon ? » au tour suivant
        # doit voir l'import, sinon l'agent invente une limitation et dit a
        # l'utilisateur l'inverse de la verite.
        if include_recent:
            recent = v1._recent_import_context(user)
            if recent:
                yield None, recent

    def _historique(self, user: User) -> list:
        conversation = getattr(self, '_conversation', None)
        filtre = {'user': user}
        if conversation is not None:
            filtre['conversation'] = conversation
        lignes = (ConversationMessage.objects
                  .filter(**filtre).exclude(pk=getattr(self, "_exclu", None))
                  .order_by("-created_at")[:HISTORIQUE_MAX])
        messages = []
        restant = 16000
        for ligne in lignes:
            if restant <= 0:
                break
            texte = ligne.content
            # Les boutons ne sont plus colles au texte: sans cette ligne, le
            # modele ne saurait pas que « 15 h 50 à 17 h 20 » repond a un choix
            # qu'il a lui-meme propose au tour precedent.
            proposes = (ligne.metadata or {}).get("quick_replies") or []
            libelles = [str(c.get("label")) for c in proposes
                        if isinstance(c, dict) and c.get("label")]
            if libelles and ligne.role != 'user':
                texte = f"{texte}\n[Choix proposés : {' | '.join(libelles)}]"
            limite = min(4000, restant)
            if len(texte) > limite:
                marque = '\n[message tronque]'
                texte = texte[:max(0, limite-len(marque))] + marque[:limite]
            restant -= len(texte)
            if ligne.role == 'user':
                messages.append(ModelRequest(parts=[UserPromptPart(content=texte)]))
            else:
                messages.append(ModelResponse(parts=[TextPart(content=texte)]))
        return list(reversed(messages))

    @staticmethod
    def _raisonnement(resultat) -> str:
        """Le raisonnement est ephemere: affiche, jamais persiste."""
        morceaux = []
        try:
            for message in resultat.all_messages():
                for part in getattr(message, "parts", []):
                    if type(part).__name__ == "ThinkingPart":
                        morceaux.append(getattr(part, "content", "") or "")
        except Exception:  # noqa: BLE001 - un volet d'affichage ne casse pas un tour
            logger.debug("Raisonnement illisible", exc_info=True)
        return "\n".join(m for m in morceaux if m)

    @staticmethod
    def _crees(registre: Registre, outil: str, cle: str) -> list:
        sortie: list = []
        for action in registre.actions:
            if action.outil != outil or not action.succes or (action.est_mutation and not action.etat_verifie):
                continue
            valeur = action.donnees.get(cle)
            if isinstance(valeur, list):
                sortie.extend(valeur)
            elif valeur:
                sortie.append(valeur)
        return sortie
