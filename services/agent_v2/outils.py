"""
Adaptateur: les outils de v1 exposes a PydanticAI, INCHANGES.

Leur description et leur schema portent des regles produit; on les transmet
tels quels, jamais reecrits. Chaque execution alimente le registre, seule
source de verite du tour.

Trois pieges, tous verifies plutot que supposes:

1. Tool(fonction, name=..., description=...) derive le schema des annotations
   Python. Sur une fonction **kwargs, cela produit {"properties": {}}, donc des
   outils sans le moindre parametre et un agent incapable de rien creer. Seul
   Tool.from_schema transmet le vrai schema (sonde du 2026-08-24).

2. requires_confirmation est applique par agent.py (lignes 921 a 944) et PAS
   par execute_tool. Un adaptateur qui appelle les outils en direct contourne
   la garde, et clear_all_blocks efface un planning sans confirmation.

3. L'ORM doit passer par sync_to_async, mais avec thread_sensitive=FALSE.
   Sous ASGI, Django execute la vue synchrone dans un thread de son pool, et
   on y demarre une boucle asyncio pour PydanticAI. Avec thread_sensitive=True
   asgiref veut rejouer l'ORM dans ce meme thread, deja bloque a attendre la
   boucle: interblocage, observe en production le 2026-08-27. Les connexions
   sont fermees des deux cotes, ce thread vivant hors du cycle de requete.

LES GARDES DU CODE (lot 1d et 2d, 2026-09-14). L'ancienne garde cherchait
« supprime|oui|ok » n'importe ou dans le message: la demande de suppression
etait donc sa propre confirmation. Desormais une action retenue pose une
DEMANDE (demandes.py), et seule la reponse du tour SUIVANT, lue par demande et
liee a l'identite de la cible, peut l'autoriser. Quand l'utilisateur touche une
puce, c'est le code qui execute l'effet choisi (appliquer_choix_en_attente),
avant meme que le modele ne tourne.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
import unicodedata
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from asgiref.sync import sync_to_async
from django.contrib.auth.models import User
from django.db import close_old_connections
from django.utils import timezone
from pydantic_ai.tools import Tool

from services.agent.tools import ALL_TOOLS, TOOL_MAP
from services.agent.tools.base import ToolResult
from services.agent_v2 import demandes as dem
from services.agent_v2 import chargeur
from services.agent_v2.registre import (OUTILS_DE_MUTATION, Registre,
                                        _empreinte, boucle_detectee)

logger = logging.getLogger(__name__)

# Repris a l'identique de la garde v1 (agent.py:160-172) et garde pour la
# parite des tests v1. La garde v2 NE L'UTILISE PLUS: le message qui demande
# une suppression ne peut pas etre sa propre confirmation.
_CONFIRME = re.compile(
    r"\b(efface|supprime|vide|enleve|retire|reset|recommence|reinitialise|"
    r"oui|confirme|d'accord|ok|vas-y|fais-le|je confirme|c'est bon)\b",
    re.IGNORECASE,
)


def autorise_destructif(message) -> bool:
    """L'utilisateur a-t-il exprime une intention destructrice explicite ?"""
    if not message or not isinstance(message, str):
        return False
    plat = unicodedata.normalize("NFKD", message).encode("ascii", "ignore").decode("ascii")
    return bool(_CONFIRME.search(plat))


# ------------------------------------------------------------------ constantes

MESSAGE_RETENUE = (
    "Action retenue par le code: une question est posee a l'utilisateur. "
    "N'agis pas sur ce point et ne repose pas la question."
)
# L'utilisateur a deja choisi une AUTRE option pour cette cible (annuler,
# seulement l'occurrence...). On ne consigne rien: une demande neuve ferait
# reposer une question a laquelle il vient de repondre.
MESSAGE_DEJA_TRANCHE = (
    "L'utilisateur a deja repondu autrement a cette question ce tour: "
    "n'agis pas sur ce point et ne repose pas la question."
)
DEJA_FAIT = "Deja fait par le code ce tour."

DESTRUCTIFS = {"delete_block", "clear_all_blocks", "delete_task", "cancel_scheduled_block"}
CREATEURS = {"create_block", "schedule_task_at", "create_task",
             "replace_block_occurrence"}
# Au-dela de cinq creations dans un tour, on demande avant de continuer.
SEUIL_CREATIONS = 5

AUTORISANTES = {
    "destructif": {"confirmer"},
    "optimisation": {"confirmer"},
    "creation_en_masse": {"confirmer"},
    "portee_jour": {"delete_block": {"serie"},
                    "skip_block_occurrence": {"occurrence"},
                    "replace_block_occurrence": {"occurrence"}},
    "portee_changement": {"update_block": {"serie"},
                          "replace_block_occurrence": {"occurrence"}},
}
MOTIFS_GARDES = {"portee_jour", "portee_changement", "destructif",
                 "creation_en_masse", "optimisation"}

_SEMAINE = re.compile(r"\bcette semaine\b|\bpour la semaine\b")
_SEMAINE_SANS_FIN = re.compile(
    r"(a partir|des|depuis) (de )?cette semaine|et les suivantes|et apres|"
    r"toutes les semaines|chaque semaine"
)
_NOMS_JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


def _autorise(motif, outil: str, option) -> bool:
    regle = AUTORISANTES.get(motif)
    if isinstance(regle, dict):
        regle = regle.get(outil, set())
    return option is not None and option in (regle or set())


# ------------------------------------------------------------- etat d'un tour

@dataclass
class _EtatTour:
    """Ce que partagent les appels d'un meme tour (un registre = un tour).

    Le verrou ne serialise plus QUE LES MUTATIONS: la garde de creations en
    masse compte puis ecrit, et deux mutations paralleles verraient sinon le
    meme compte; le cache d'idempotence est aussi un tester-puis-poser. Les
    LECTURES s'executent en parallele: leurs gardes ne font que lire l'etat
    du tour et la base, et le registre a son propre verrou pour l'ecriture.
    Voir _fabriquer: pydantic-ai dispatche deja les appels batchés en
    parallele, sauf si l'un d'eux est marque sequential (les mutations).
    """
    verrou: threading.RLock = field(default_factory=threading.RLock)
    verrou_init: threading.Lock = field(default_factory=threading.Lock)
    cache: dict = field(default_factory=dict)
    attente: dict = field(default_factory=dict)
    armes: list = field(default_factory=list)


_ETATS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_ETATS_VERROU = threading.Lock()


def _etat_du_tour(registre: Registre) -> _EtatTour:
    with _ETATS_VERROU:
        etat = _ETATS.get(registre)
        if etat is None:
            etat = _EtatTour()
            _ETATS[registre] = etat
        return etat


@dataclass
class _Contexte:
    user: User
    registre: Registre
    tache: str
    texte: str
    signaler: object = None
    tap: dict | None = None

    @property
    def etat(self) -> _EtatTour:
        return _etat_du_tour(self.registre)


def _attente(ctx: _Contexte) -> list[dict]:
    """Les demandes en attente, lues UNE fois par tour.

    L'initialisation paresseuse est protegee: les lectures s'executent
    desormais en parallele et deux threads pouvaient la declencher.
    """
    etat = ctx.etat
    if "liste" not in etat.attente:
        with etat.verrou_init:
            if "liste" not in etat.attente:
                try:
                    etat.attente["liste"] = dem.demandes_en_attente(ctx.user)
                except Exception:  # noqa: BLE001 - sans attente lisible, rien n'est autorise
                    logger.error("Demandes en attente illisibles", exc_info=True)
                    etat.attente["liste"] = []
    return etat.attente["liste"]


# -------------------------------------------------------------------- petits

def _entier(valeur):
    if isinstance(valeur, bool):
        return None
    try:
        return int(valeur)
    except (TypeError, ValueError):
        return None


def _date_iso(valeur):
    if isinstance(valeur, date):
        return valeur
    try:
        return datetime.strptime(str(valeur).strip()[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _heure_normale(valeur):
    from services.scheduling.overlap import parse_time

    if valeur in (None, ""):
        return None
    try:
        return parse_time(valeur).strftime("%H:%M")
    except (TypeError, ValueError):
        return None


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _fmt(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _ascii(texte) -> str:
    if not isinstance(texte, str):
        return ""
    return unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode("ascii")


def _cible_bloc(block) -> dict:
    return {
        "titre": block.title,
        "jour": block.day_of_week,
        "debut": block.start_time.strftime("%H:%M"),
        "fin": block.end_time.strftime("%H:%M"),
        "block_type": block.block_type,
        "debut_serie": block.start_date.isoformat() if block.start_date else None,
        "fin_serie": block.end_date.isoformat() if block.end_date else None,
    }


# ---------------------------------------------------------------------- cles

def _cle_portee(block_id, jour_iso: str) -> str:
    return dem.cle_demande("portee_jour", {"block_id": int(block_id), "date": jour_iso})


# Tout ce qu'un update_block peut changer sur une serie, hors bornes de dates.
# La cle et les deux options les portent toutes: sans cela le code annonçait
# FAIT alors qu'une partie du geste etait tombee.
CHAMPS_CHANGEABLES = ("title", "start_time", "end_time", "day_of_week",
                      "block_type", "location", "flexibility")
# Celles qui changent l'IDENTITE de la serie: elles valent pour toutes les
# semaines, donc leur portee se demande.
CHAMPS_IDENTITE = ("title", "day_of_week")


def _cle_changement(block_id, params: dict) -> str:
    """Cle d'une portee de CHANGEMENT. Les valeurs visees en font partie: une
    puce qui accepte « midi » n'autorise pas ensuite « 14 h »."""
    return dem.cle_demande("portee_changement", {
        "block_id": int(block_id),
        **{c: str(params.get(c) or "") for c in CHAMPS_CHANGEABLES}})


def _cle_bornes(block_id, params: dict) -> str:
    """Cle d'une fin ou d'un depart de serie: les dates en font partie, pour
    qu'une puce confirmant le 15 octobre n'autorise pas le 1er decembre."""
    def _iso(cle):
        jour = _date_iso(params.get(cle)) if str(params.get(cle) or "").strip() else None
        return jour.isoformat() if jour else ""
    return dem.cle_demande("update_block:fin", {"block_id": int(block_id),
                                                "end_date": _iso("end_date"),
                                                "start_date": _iso("start_date")})


def _cle_destructive(nom: str, params: dict):
    """Cle d'identite d'une action retenue, a partir de ses seuls arguments."""
    params = params or {}
    if nom == "delete_block":
        bid = _entier(params.get("block_id"))
        return None if bid is None else dem.cle_demande("delete_block", {"block_id": bid})
    if nom == "update_block":
        bid = _entier(params.get("block_id"))
        return None if bid is None else _cle_bornes(bid, params)
    if nom == "delete_task":
        tid = _entier(params.get("task_id"))
        return None if tid is None else dem.cle_demande("delete_task", {"task_id": tid})
    if nom == "cancel_scheduled_block":
        jour = _date_iso(params.get("date"))
        if jour is None:
            return None
        titre = str(params.get("title") or "").casefold().strip()
        return dem.cle_demande("cancel_scheduled_block", {"date": jour.isoformat(), "title": titre})
    if nom == "clear_all_blocks":
        return "clear_all_blocks"
    if nom == "optimize_week":
        return "optimize_week:apply"
    return None


def _bloc_de_demande(demande: dict):
    bid = _entier((demande.get("parametres") or {}).get("block_id"))
    if bid is not None:
        return bid
    for option in demande.get("options") or []:
        effet = option.get("effet") if isinstance(option, dict) else None
        if isinstance(effet, dict) and effet.get("outil") == "delete_block":
            bid = _entier((effet.get("parametres") or {}).get("block_id"))
            if bid is not None:
                return bid
    return None


# ------------------------------------------------ cible relue avant l'effet
#
# Round 9 (K1, trouve par Codex): l'effet stocke ne verifiait que sa cle et
# ses parametres. Un bloc modifie, desactive ou remplace entre la question et
# la puce (autre appareil, MCP, application web) etait supprime dans son
# NOUVEL etat. Avant tout effet destructif, la cible est relue et comparee a
# demande.cible; a la moindre difference, rien ne s'execute.

# Les champs d'etat compares. « date » n'en est pas: elle vient du message.
_CHAMPS_CIBLE = ("titre", "jour", "debut", "fin", "block_type", "complete", "echeance",
                 "ids", "nombre", "evenements", "debut_serie", "fin_serie")

MESSAGE_CIBLE_CHANGEE = (
    "La cible a change depuis la question: le code n'a rien execute et laisse la "
    "demande de cote. N'agis pas sur ce point sans nouvelle demande explicite.")


def _cible_tache(task) -> dict:
    return {
        "titre": task.title,
        "complete": bool(task.completed),
        "echeance": task.deadline.isoformat() if task.deadline else None,
    }


def _evenements(user, jour, titre: str) -> list:
    from core.models import ScheduledBlock

    qs = ScheduledBlock.objects.filter(user=user, date=jour).select_related("task")
    if titre:
        qs = qs.filter(task__title__icontains=titre)
    return list(qs.order_by("start_time", "id"))


def _cible_evenements(blocs: list, jour: date) -> dict:
    premier = blocs[0]
    return {
        "titre": premier.task.title if premier.task_id else "",
        "date": jour.isoformat(),
        "debut": premier.start_time.strftime("%H:%M"),
        "fin": premier.end_time.strftime("%H:%M"),
        "ids": sorted(b.id for b in blocs),
        # Round 9 (revue Codex): une photo par evenement. Une deuxieme rangee
        # deplacee le meme jour garde les memes ids et la premiere rangee.
        "evenements": [
            {"id": b.id, "titre": b.task.title if b.task_id else "",
             "debut": b.start_time.strftime("%H:%M"), "fin": b.end_time.strftime("%H:%M")}
            for b in sorted(blocs, key=lambda x: x.id)],
    }


def _cible_planning(user) -> dict:
    from core.models import RecurringBlock

    ids = sorted(RecurringBlock.objects.filter(user=user, active=True).values_list("id", flat=True))
    return {"nombre": len(ids), "ids": ids}


def _cible_actuelle(user, demande: dict):
    """L'etat COURANT de la cible d'une demande destructive, dans la forme de
    demande.cible, ou None quand elle n'existe plus (ou plus active)."""
    from core.models import RecurringBlock, Task

    outil = demande.get("outil")
    parametres = demande.get("parametres") or {}
    if demande.get("motif") == "portee_jour" or outil in ("delete_block", "update_block",
                                                          "skip_block_occurrence"):
        bid = _bloc_de_demande(demande)
        block = (RecurringBlock.objects.filter(id=bid, user=user, active=True).first()
                 if bid is not None else None)
        return _cible_bloc(block) if block is not None else None
    if outil == "delete_task":
        tid = _entier(parametres.get("task_id"))
        task = Task.objects.filter(id=tid, user=user).first() if tid is not None else None
        return _cible_tache(task) if task is not None else None
    if outil == "cancel_scheduled_block":
        jour = _date_iso(parametres.get("date"))
        if jour is None:
            return None
        titre = str(parametres.get("title") or "").strip()
        blocs = _evenements(user, jour, titre)
        if not blocs or (not titre and len({b.task.title if b.task_id else "" for b in blocs}) > 1):
            return None
        return _cible_evenements(blocs, jour)
    if outil == "clear_all_blocks":
        return _cible_planning(user)
    return None


def _cible_changee(user, demande: dict) -> bool:
    """Vrai quand la cible d'une demande destructive n'est plus celle de la
    question. Seuls les champs presents dans demande.cible sont compares: une
    demande d'avant le round 9 (sans « ids ») garde les autres."""
    if demande.get("motif") not in ("portee_jour", "portee_changement", "destructif"):
        return False
    stockee = demande.get("cible") or {}
    actuelle = _cible_actuelle(user, demande)
    if actuelle is None:
        return True
    return any(actuelle.get(k) != stockee[k] for k in _CHAMPS_CIBLE if k in stockee)


def _ligne_cible_changee(demande: dict) -> str:
    """La ligne du code, en francais du Quebec, pour une cible changee."""
    outil = demande.get("outil")
    titre = str((demande.get("cible") or {}).get("titre") or "").strip()
    if titre:
        titre = titre[0].upper() + titre[1:]
    if outil == "clear_all_blocks":
        sujet, verbe = "Ton planning", "supprimé"
    elif outil == "delete_task":
        sujet, verbe = (f"La tâche {titre}" if titre else "Cette tâche"), "supprimé"
    elif outil == "cancel_scheduled_block":
        sujet, verbe = (titre or "Cet événement"), "annulé"
    elif outil == "update_block":
        sujet, verbe = (titre or "Ce créneau"), "changé"
    else:
        sujet, verbe = (titre or "Ce créneau"), "supprimé"
    return (f"{sujet} a changé depuis ma question, je n'ai rien {verbe}. "
            "Redis-le si tu veux toujours.")


def _abandon_cible_changee(ctx, demande: dict):
    """Consigne l'abandon d'une demande dont la cible a change, et la retire
    du tour: ni la puce ni le modele ne peuvent plus l'executer."""
    cle = demande.get("cle")
    ctx.etat.attente.setdefault("abandonnees", set()).add(cle)
    ctx.etat.attente.setdefault("cibles_changees", set()).add(cle)
    logger.warning("Cible changee depuis la question, rien d'execute cle=%s", cle)
    resultat = ToolResult(
        success=False,
        data={"demande": {k: v for k, v in demande.items() if k != "chips"},
              "abandonnee_par_le_code": True, "cible_changee": True,
              "decision_code": DECISION_ABANDONNEE,
              "ligne_cible_changee": _ligne_cible_changee(demande)},
        message=MESSAGE_CIBLE_CHANGEE)
    return _consigner_decision(ctx, demande, resultat)


# -------------------------------------------------------------------- gardes

@dataclass
class _Garde:
    motif: str
    cle: str
    cles: set
    outil: str
    parametres: dict = field(default_factory=dict)
    cible: dict = field(default_factory=dict)
    options: list = field(default_factory=list)
    actif: bool = True
    type: str = "confirmation"

    def demande(self) -> dict:
        return dem.construire_demande(self.type, self.motif, self.outil, self.parametres,
                                      self.cible, self.options, self.cle)


@dataclass
class _Preparation:
    borne_auto: dict | None = None
    bloc_update: dict | None = None


def _options_confirmer(outil: str, parametres: dict, cible: dict) -> list[dict]:
    return [
        {"id": "confirmer", "effet": {"outil": outil, "parametres": dict(parametres)},
         "cible": dict(cible)},
        {"id": "annuler", "effet": None, "cible": dict(cible)},
    ]


def _options_portee(block, jour: date, cible: dict) -> list[dict]:
    return [
        {"id": "occurrence",
         "effet": {"outil": "skip_block_occurrence",
                   "parametres": {"date": jour.isoformat(), "title": block.title,
                                  "block_type": block.block_type}},
         "cible": dict(cible)},
        {"id": "serie",
         "effet": {"outil": "delete_block", "parametres": {"block_id": block.id}},
         "cible": dict(cible)},
        {"id": "annuler", "effet": None, "cible": dict(cible)},
    ]


def _options_changement(block, jour: date, cible: dict, params: dict) -> list[dict]:
    """Une occurrence: le remplacant prend le creneau. La serie: update_block."""
    titre = str(params.get("title") or "").strip() or block.title
    remplacement = {"date": jour.isoformat(), "title": block.title,
                    "block_type": block.block_type, "replacement_title": titre}
    serie = {"block_id": block.id}
    # Tout ce que l'appel portait suit l'option « serie »: un geste partiel
    # annonce comme fait serait un mensonge. L'option « occurrence » ne prend
    # que ce qui a un sens pour un evenement date.
    for cle in CHAMPS_CHANGEABLES:
        valeur = str(params.get(cle) or "").strip()
        if valeur:
            serie[cle] = valeur
            if cle in ("start_time", "end_time"):
                remplacement[cle] = valeur
    return [
        {"id": "occurrence",
         "effet": {"outil": "replace_block_occurrence", "parametres": remplacement},
         "cible": dict(cible)},
        {"id": "serie",
         "effet": {"outil": "update_block", "parametres": serie},
         "cible": dict(cible)},
        {"id": "annuler", "effet": None, "cible": dict(cible)},
    ]


def _saut_suspect(texte: str) -> bool:
    """Un saut d'occurrence qui ressemble a une suppression large (« efface
    tout jeudi »). Le saut unique explicite (« saute mon gym demain ») passe.

    Le jugement remplace les anciennes regex _TOUT/_UNE_SEULE_FOIS: en cas de
    doute, le saut est suspect et le code pose la question de portee.
    """
    return dem.saut_suspect(texte)


def _appel_composite(nom: str, kwargs: dict):
    """Un update_block qui change l'IDENTITE de la serie ET ses bornes.

    Une seule question ne peut pas trancher les deux gestes: la question des
    bornes ne parle que de la date, et celle de la portee ne parle que du
    changement. Mesure du 2026-09-30: un end_date non destructif desarmait
    toute garde, et une fin de serie confirmee renommait la serie au passage.
    """
    if nom != "update_block":
        return None
    identite = any(str(kwargs.get(c) or "").strip() for c in CHAMPS_IDENTITE)
    bornes = any(str(kwargs.get(c) or "").strip() for c in ("end_date", "start_date"))
    if not (identite and bornes):
        return None
    return ToolResult(
        success=False,
        data={"appel_composite": True},
        message=("Refuse par le code: un meme appel change le nom ou le jour de "
                 "la serie ET ses dates de debut ou de fin. Chacun a sa propre "
                 "question a l'utilisateur. Fais deux appels separes."))


def _garde_changement(ctx: _Contexte, kwargs: dict):
    """La garde de portee d'un changement de titre ou d'heures, ou None."""
    from core.models import RecurringBlock

    # L'IDENTITE de la serie: son nom et son jour. Les HEURES ont deja leur
    # garde (« heure dite », round r9): la doubler d'une question de portee
    # masquerait un refus plus precis, et un tour ne pose qu'une question.
    titre = str(kwargs.get("title") or "").strip()
    jour_vise = str(kwargs.get("day_of_week") or "").strip()
    if not titre and not jour_vise:
        return None
    bid = _entier(kwargs.get("block_id"))
    if bid is None:
        return None
    block = RecurringBlock.objects.filter(id=bid, user=ctx.user, active=True).first()
    if block is None:
        return None
    # Comparaison normalisee: une apostrophe courbe redressee par le modele
    # n'est pas un changement de nom.
    change_nom = bool(titre) and dem.normaliser(titre) != dem.normaliser(block.title)
    change_jour = bool(jour_vise) and _entier(jour_vise) != block.day_of_week
    if not (change_nom or change_jour):
        return None
    jour = dem.date_visee(ctx.texte, block.day_of_week, timezone.localdate())
    cle = _cle_changement(bid, kwargs)
    cible = _cible_bloc(block)  # noqa: E501
    cible["date"] = jour.isoformat()
    parametres = dict(kwargs)
    parametres["block_id"] = bid
    # Le juge ne tranche pas a la place de la personne: il EVITE la question
    # quand la serie entiere est clairement visee. Muet ou incertain, on
    # demande, et la garde reste active. Une question DEJA posee sur cette
    # cible ne se tranche que par sa puce (round 6, D1): le juge ne la
    # court-circuite pas au tour suivant.
    en_attente = any(d.get("cle") == cle for d in _attente(ctx))
    actif = en_attente or not dem.serie_entiere_visee(ctx.texte)
    return _Garde("portee_changement", cle, {cle}, "update_block", parametres, cible,
                  _options_changement(block, jour, cible, kwargs),
                  actif=actif, type="choix")


def _analyser(ctx: _Contexte, nom: str, kwargs: dict):
    """La garde destructive d'un appel, ou None. Une garde inactive sert
    seulement a reconnaitre ce que le code a deja fait ce tour."""
    from core.models import RecurringBlock, Task

    user, texte = ctx.user, ctx.texte
    if nom == "delete_block":
        bid = _entier(kwargs.get("block_id"))
        if bid is None:
            return None
        cle = dem.cle_demande("delete_block", {"block_id": bid})
        cles = {cle}
        for demande in _attente(ctx):
            if demande.get("motif") == "portee_jour" and _bloc_de_demande(demande) == bid:
                cles.add(demande["cle"])
        block = RecurringBlock.objects.filter(id=bid, user=user, active=True).first()
        if block is None:
            return _Garde("destructif", cle, cles, nom, actif=False)
        cible = _cible_bloc(block)
        if dem.jour_vise(texte):
            jour = dem.date_visee(texte, block.day_of_week, timezone.localdate())
            cle_portee = _cle_portee(block.id, jour.isoformat())
            cles.add(cle_portee)
            cible["date"] = jour.isoformat()
            return _Garde("portee_jour", cle_portee, cles, nom, {"block_id": block.id},
                          cible, _options_portee(block, jour, cible), type="choix")
        return _Garde("destructif", cle, cles, nom, {"block_id": block.id}, cible,
                      _options_confirmer(nom, {"block_id": block.id}, cible))

    if nom == "skip_block_occurrence":
        from services.agent.tools.blocks import VALID_BLOCK_TYPES, _resolve_day_blocks

        jour = _date_iso(kwargs.get("date"))
        block_type = kwargs.get("block_type") or None
        if jour is None or (block_type and block_type not in VALID_BLOCK_TYPES):
            return None
        trouves = _resolve_day_blocks(user, jour, block_type, kwargs.get("title") or None)
        if len(trouves) != 1:
            return None
        block = trouves[0]
        cle = _cle_portee(block.id, jour.isoformat())
        cible = _cible_bloc(block)
        cible["date"] = jour.isoformat()
        # Round 6 (D1): une question de portee en attente sur CE bloc et CE
        # jour ne se tranche que par sa puce. « juste celui-la » ne laisse pas
        # le modele sauter l'occurrence a la place de la puce.
        en_attente = any(d.get("cle") == cle for d in _attente(ctx))
        return _Garde("portee_jour", cle, {cle}, nom, dict(kwargs), cible,
                      _options_portee(block, jour, cible),
                      actif=_saut_suspect(texte) or en_attente, type="choix")

    if nom == "clear_all_blocks":
        # K1: l'ensemble exact des blocs, relu avant d'executer la puce.
        cible = _cible_planning(user)
        return _Garde("destructif", "clear_all_blocks", {"clear_all_blocks"}, nom,
                      {"confirm": True}, cible,
                      _options_confirmer(nom, {"confirm": True}, cible))

    if nom == "delete_task":
        tid = _entier(kwargs.get("task_id"))
        if tid is None:
            return None
        cle = dem.cle_demande("delete_task", {"task_id": tid})
        task = Task.objects.filter(id=tid, user=user).first()
        if task is None:
            return _Garde("destructif", cle, {cle}, nom, actif=False)
        cible = _cible_tache(task)
        return _Garde("destructif", cle, {cle}, nom, {"task_id": tid}, cible,
                      _options_confirmer(nom, {"task_id": tid, "confirm": True}, cible))

    if nom == "cancel_scheduled_block":
        jour = _date_iso(kwargs.get("date"))
        if jour is None:
            return None
        titre = str(kwargs.get("title") or "").strip()
        cle = _cle_destructive(nom, kwargs)
        blocs = _evenements(user, jour, titre)
        distincts = {b.task.title if b.task_id else "" for b in blocs}
        if not blocs or (len(distincts) > 1 and not titre):
            # Rien a supprimer, ou l'outil va demander lequel: rien a retenir.
            return _Garde("destructif", cle, {cle}, nom, actif=False)
        cible = _cible_evenements(blocs, jour)
        parametres = {"date": jour.isoformat(), "title": titre} if titre else {"date": jour.isoformat()}
        return _Garde("destructif", cle, {cle}, nom, parametres, cible,
                      _options_confirmer(nom, parametres, cible))

    if nom == "update_block":
        # Une fin avancee ou un depart repousse enleve des seances. La garde se
        # decide sur les dates et l'etat du bloc, jamais sur les mots du
        # message: « arrete mon quart a partir de decembre » passait sans
        # question faute d'un verbe de suppression reconnu (enquete du
        # 2026-09-14). Prolonger une serie ou avancer son depart n'enleve rien.
        fin = _date_iso(kwargs.get("end_date")) if str(kwargs.get("end_date") or "").strip() else None
        depart = _date_iso(kwargs.get("start_date")) if str(kwargs.get("start_date") or "").strip() else None
        if fin is None and depart is None:
            # PORTEE D'UN CHANGEMENT: le titre ou les heures d'une serie
            # changent pour TOUTES les semaines. « J'ai examen a la place du
            # cours » a renomme la serie en production le 2026-09-29. Le code
            # demande, sauf quand le juge voit clairement la serie visee.
            return _garde_changement(ctx, kwargs)
        bid = _entier(kwargs.get("block_id"))
        if bid is None:
            return None
        cle = _cle_bornes(bid, kwargs)
        block = RecurringBlock.objects.filter(id=bid, user=user, active=True).first()
        if block is None:
            return _Garde("destructif", cle, {cle}, nom, actif=False)
        aujourdhui = timezone.localdate()
        fin_destructive = fin is not None and (block.end_date is None or fin < block.end_date)
        depart_destructif = (depart is not None and depart > aujourdhui
                             and (block.start_date is None or depart > block.start_date))
        if fin_destructive and not depart_destructif \
                and _fin_de_recurrence_repondue(user, block, fin, aujourdhui, ctx.tache):
            fin_destructive = False
        if not (fin_destructive or depart_destructif):
            return None
        cible = _cible_bloc(block)
        cible["date"] = (fin if fin_destructive else depart).isoformat()
        parametres = dict(kwargs)
        parametres["block_id"] = bid
        return _Garde("destructif", cle, {cle}, nom, parametres, cible,
                      _options_confirmer(nom, parametres, cible))

    if nom == "optimize_week" and kwargs.get("apply"):
        cle = "optimize_week:apply"
        return _Garde("optimisation", cle, {cle}, nom, dict(kwargs))
    return None


def _fin_de_recurrence_repondue(user, block, fin, aujourdhui, tache: str = "") -> bool:
    """La reponse IMMEDIATE a la question de fin de recurrence posee apres le
    dernier import: le message juste avant le message courant est cette
    question, le bloc vient du dernier document envoye et n'a pas encore de
    fin, et la fin tombe au-dela des 7 prochains jours. Le message de
    l'utilisateur ne porte pas sa piece jointe en v2: le document se lit par
    l'ordre des envois."""
    from core.models import ConversationMessage, UploadedDocument
    from services.agent_v2.boutons import MOTIF_FIN_RECURRENCE

    if block.source_document_id is None or block.end_date is not None:
        return False
    if fin <= aujourdhui + timedelta(days=7):
        return False
    dernier_doc = UploadedDocument.objects.filter(user=user).order_by("-pk").values_list(
        "pk", flat=True).first()
    if dernier_doc != block.source_document_id:
        return False
    # La tache vaut « user:message courant » (agent.py). Sans ce message, un
    # tour plus ancien du meme utilisateur profiterait de la reponse d'un autre.
    _, _, courant = str(tache or "").rpartition(":")
    if not courant.isdigit():
        return False
    deux = list(ConversationMessage.objects.filter(user=user).order_by("-pk")[:2])
    if (len(deux) < 2 or deux[0].pk != int(courant) or deux[0].role != "user"
            or deux[1].role != "assistant"):
        return False
    meta = deux[1].metadata if isinstance(deux[1].metadata, dict) else {}
    return meta.get("question_motif") == MOTIF_FIN_RECURRENCE


def _garde_critique(nom: str, kwargs: dict) -> bool:
    """Si la garde elle-meme tombe en panne, ces appels sont refuses."""
    return (
        nom in DESTRUCTIFS
        or nom == "skip_block_occurrence"
        or (nom == "update_block" and bool(str(kwargs.get("end_date") or "").strip()))
        or (nom == "update_block" and bool(str(kwargs.get("start_date") or "").strip()))
        or (nom == "update_block" and bool(str(kwargs.get("title") or "").strip()))
        or (nom == "optimize_week" and bool(kwargs.get("apply")))
    )


def _deja_fait(ctx: _Contexte, nom: str, garde) -> bool:
    if garde is None or not garde.cles:
        return False
    return any(
        a.succes and a.outil == nom and a.donnees.get("par_le_code")
        and a.donnees.get("cle_demande") in garde.cles
        for a in ctx.registre.actions
    )


def _reponse(ctx: _Contexte, cles: set, nom: str):
    """(autorise, demande, en_suspens) pour les demandes en attente sur la
    MEME cible, evaluees demande par demande.

    autorise: la demande dont l'option autorise cet outil.
    demande (non autorise): une demande a laquelle l'utilisateur a repondu
    AUTREMENT (annuler, l'autre portee).
    en_suspens: une demande sans reponse claire, a reposer telle quelle.
    """
    repondue = en_suspens = None
    abandonnees = ctx.etat.attente.get("abandonnees") or set()
    for demande in _attente(ctx):
        if demande.get("cle") not in cles:
            continue
        if demande.get("cle") in abandonnees:
            # Abandonnee par le code ce tour (D2): une nouvelle suppression de
            # la meme cible pose une question NEUVE, jamais la perimee.
            continue
        option = dem.option_choisie(ctx.texte, demande, tap=ctx.tap)
        if _autorise(demande.get("motif"), nom, option):
            return True, demande, None
        if option is not None:
            repondue = repondue or demande
        elif en_suspens is None:
            en_suspens = demande
    return False, repondue, en_suspens


# Une demande n'est reposee par le code qu'une fois: au-dela, elle collait a
# la conversation et une suppression suivante y repondait (revue du round 4).
REEMISSIONS_MAX = 1

# Les decisions que appliquer_choix_en_attente consigne (ToolResult.data
# « decision_code »), lues par la voix et par tour_entierement_decide_par_le_code.
DECISION_EXECUTE = "execute"
DECISION_REPOSEE = "reposee"
DECISION_ABANDONNEE = "abandonnee"
DECISION_ANNULEE = "annulee"
DECISIONS_DU_CODE = {DECISION_EXECUTE, DECISION_REPOSEE, DECISION_ABANDONNEE, DECISION_ANNULEE}
# Nom de registre des decisions sans outil execute. Pas une mutation.
OUTIL_DECISION = "decision_du_code"
MESSAGE_ABANDON = ("Question laissee de cote par le code: rien n'a change. "
                   "N'agis pas sur ce point sans nouvelle demande explicite.")
MESSAGE_ANNULEE = "Refuse par l'utilisateur: rien n'a change, n'y touche pas."


def _reposer(demande: dict) -> dict:
    """La meme question, sans les puces du tour passe. emise_le reste celle
    d'ORIGINE: la fenetre d'attente doit pouvoir expirer."""
    copie = {k: v for k, v in demande.items() if k != "chips"}
    copie["reemissions"] = int(demande.get("reemissions") or 0) + 1
    if not copie.get("emise_le"):
        copie["emise_le"] = timezone.now().isoformat()
    return copie


def _refus(demande: dict, destructif: bool) -> ToolResult:
    donnees = {"demande": demande}
    if destructif:
        donnees["needs_confirmation"] = True
    return ToolResult(success=False, data=donnees, message=MESSAGE_RETENUE)


# ------------------------------------------------------------- optimisation

def _empreinte_plan(donnees: dict) -> str:
    """Empreinte d'un plan propose, stable pour un meme etat. Les listes sont
    triees: un ordre instable ferait reposer la question sans fin."""
    base = donnees.get("moves") or donnees.get("placed")
    if base is None:
        def _tri(elements):
            return sorted(
                (e for e in elements or [] if isinstance(e, dict)),
                key=lambda e: (str(e.get("start_time") or ""), str(e.get("title") or "")),
            )
        base = {
            "start_date": donnees.get("start_date"),
            "days": [
                {"date": j.get("date"), "placed": _tri(j.get("placed")),
                 "overnight_kept": _tri(j.get("overnight_kept")),
                 "skipped": _tri(j.get("skipped"))}
                for j in donnees.get("days") or [] if isinstance(j, dict)
            ],
        }
    brut = json.dumps(base, sort_keys=True, default=str)
    return hashlib.sha1(brut.encode("utf-8")).hexdigest()[:12]


_PLAN_SEMAINE_TTL = 1800  # 30 min: une confirmation arrive vite; au-dela on re-resout.


def _cle_plan_semaine(user_id: int, start_iso: str) -> str:
    return f"agentv2:plan_semaine:v1:{user_id}:{start_iso}"


def _plan_propose(outil, user, kwargs: dict):
    """(proposition, empreinte, arrangements), avec cache inter-tours.

    Le solveur est deterministe (random_seed = 0): si ses entrees n'ont pas
    change depuis la proposition, on ressert le plan en cache au lieu de
    re-resoudre 7 jours (~5 s mesures, plafond 14 s). Sinon on re-resout
    (comportement historique): la fraicheur reste garantie par l'empreinte
    des ENTREES (empreinte_entrees_semaine, colocalisee avec le solveur),
    pas par un re-calcul systematique.
    """
    from django.core.cache import cache
    from services.agent.tools.schedule import _resoudre_semaine, _resultat_semaine
    from services.scheduling.solve_day import empreinte_entrees_semaine

    parametres = {k: v for k, v in kwargs.items() if k not in ("plan_hash", "_arrangements")}
    raw = parametres.get("start_date")
    try:
        start = datetime.strptime(raw, "%Y-%m-%d").date() if raw else timezone.localdate()
    except (ValueError, TypeError):
        # Date invalide: l'outil echouera de lui-meme; pas de cache.
        proposition = outil.execute(user, **{**parametres, "apply": False})
        return proposition, None, None
    start_iso = start.isoformat()

    cle = _cle_plan_semaine(user.pk, start_iso)
    entrees = empreinte_entrees_semaine(user, start)
    cached = cache.get(cle)
    if cached and cached.get("entrees") == entrees:
        logger.info("Plan semaine %s: servi du cache (entrees inchangees)", start_iso)
        proposition = ToolResult(success=True, data=cached["data"], message=cached["message"])
        return proposition, cached["empreinte"], cached["arrangements"]

    days_data, lines, skipped_total, arrangements = _resoudre_semaine(user, start)
    proposition = _resultat_semaine(apply=False, start=start, days_data=days_data,
                                    lines=lines, moved_total=0, skipped_total=skipped_total)
    empreinte = _empreinte_plan(proposition.data or {})
    cache.set(cle, {"entrees": entrees, "empreinte": empreinte,
                    "data": proposition.data, "message": proposition.message,
                    "arrangements": arrangements},
              _PLAN_SEMAINE_TTL)
    return proposition, empreinte, arrangements


_PLAN_JOUR_TTL = 1800  # 30 min: meme logique que le plan semaine.


def _cle_plan_jour(user_id: int, date_iso: str) -> str:
    return f"agentv2:plan_jour:v1:{user_id}:{date_iso}"


def _jour_organize(raw):
    """Parse la date d'organize_day; None si absente ou invalide."""
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date() if raw else None
    except (ValueError, TypeError):
        return None


def _arrangement_jour_cache(user, jour):
    """(arrangement, entrees): l'arrangement en cache si ses entrees sont
    inchangees, sinon (None, entrees_fraiches).

    Le solveur est deterministe: si ses entrees n'ont pas change depuis la
    proposition, on reutilise l'arrangement en cache au lieu de re-resoudre
    (~1-2 s mesures, plafond 2 s par jour).
    """
    from django.core.cache import cache
    from services.scheduling.solve_day import empreinte_entrees_jour

    cle = _cle_plan_jour(user.pk, jour.isoformat())
    entrees = empreinte_entrees_jour(user, jour)
    cached = cache.get(cle)
    if cached and cached.get("entrees") == entrees:
        logger.info("Plan jour %s: servi du cache (entrees inchangees)", jour.isoformat())
        return cached["arrangement"], entrees
    return None, entrees


def _plan_propose_jour(outil, user, kwargs: dict):
    """(proposition, entrees, arrangement) pour organize_day, avec cache.

    Meme contrat que _plan_propose mais pour un seul jour: la proposition
    (apply=False) alimente le cache; l'apply le lit via
    _arrangement_jour_cache.
    """
    from django.core.cache import cache
    from services.agent.tools import schedule as _sched

    jour = _jour_organize(kwargs.get("date"))
    if jour is None:
        # Date invalide: l'outil echouera de lui-meme; pas de cache.
        proposition = outil.execute(user, **kwargs)
        return proposition, None, None

    arrangement, entrees = _arrangement_jour_cache(user, jour)
    if arrangement is None:
        # Via le module (pas d'import direct): les tests peuvent espionner
        # solve_placement, et le comportement reste identique.
        arrangement = _sched.solve_placement(user, jour)
        proposition = _sched._executer_organize_day(user, jour, apply=False, arrangement=arrangement)
        cache.set(_cle_plan_jour(user.pk, jour.isoformat()),
                  {"entrees": entrees, "data": proposition.data,
                   "message": proposition.message, "arrangement": arrangement},
                  _PLAN_JOUR_TTL)
    else:
        proposition = _sched._executer_organize_day(user, jour, apply=False, arrangement=arrangement)
    return proposition, entrees, arrangement


def _demande_optimisation(kwargs: dict, empreinte: str, proposition: ToolResult) -> dict:
    # Les kwargs prives (prefixe _) ne font jamais partie d'une demande
    # persistee: ce sont des hints d'execution, pas l'intention metier.
    parametres = {k: v for k, v in kwargs.items() if k != "plan_hash" and not k.startswith("_")}
    effet = dict(parametres)
    effet["apply"] = True
    cible = {"date": (proposition.data or {}).get("start_date")}
    options = _options_confirmer("optimize_week", effet, cible)
    options[0]["effet"]["parametres"] = effet
    return dem.construire_demande(
        "confirmation", "optimisation", "optimize_week",
        {**parametres, "apply": True, "plan_hash": empreinte}, cible, options,
        "optimize_week:apply")


def _garde_optimisation(ctx: _Contexte, outil, kwargs: dict, garde: _Garde):
    autorise, demande, _en_suspens = _reponse(ctx, garde.cles, outil.name)
    proposition, empreinte, arrangements = _plan_propose(outil, ctx.user, kwargs)
    if empreinte is None:
        return None  # l'outil echouera de lui-meme, rien a proteger
    if autorise and (demande.get("parametres") or {}).get("plan_hash") == empreinte:
        # Plan confirme a l'identique: l'execution reutilise l'arrangement
        # valide au lieu de re-resoudre (kwarg prive, retire apres l'appel).
        kwargs["_arrangements"] = arrangements
        return None
    if demande is not None and not autorise:
        return MESSAGE_DEJA_TRANCHE
    return _refus(_demande_optimisation(kwargs, empreinte, proposition), destructif=False)


# ----------------------------------------------------------- heure refusee

def _creneaux(user, jour: date, duree: int, debut_min: int) -> list[tuple[int, int]]:
    """Deux ou trois vrais creneaux libres de la meme duree, les plus proches
    de l'heure demandee, dans la fenetre eveillee de find_free_slots."""
    from services.agent.tools.schedule import (DAY_END_MIN, DAY_START_MIN,
                                               _free_slots_from_intervals)
    from services.scheduling.placement import open_intervals

    if duree <= 0:
        return []
    maintenant = timezone.localtime()
    if jour < maintenant.date():
        return []
    intervalles = open_intervals(user, jour, DAY_START_MIN, DAY_END_MIN)
    if jour == maintenant.date():
        now_min = ((maintenant.hour * 60 + maintenant.minute + 4) // 5) * 5
        intervalles = [(max(s, now_min), e) for s, e in intervalles if e > now_min]
    departs: list[int] = []
    for libre in _free_slots_from_intervals(intervalles, duree):
        debut, fin = _minutes(libre["start_time"]), _minutes(libre["end_time"])
        for depart in (debut, fin - duree):
            if depart not in departs:
                departs.append(depart)
    departs.sort(key=lambda d: (abs(d - debut_min), d))
    return [(d, d + duree) for d in sorted(departs[:3])]


def _demande_heure_refusee(ctx: _Contexte, nom: str, kwargs: dict, titre: str,
                           jour: date, debut: str, fin: str, avec, recurrent: bool) -> dict:
    duree = (_minutes(fin) - _minutes(debut)) % (24 * 60)
    identite = ({"jour": jour.weekday(), "debut": debut} if recurrent
                else {"date": jour.isoformat(), "debut": debut})
    cle = dem.cle_demande("heure_refusee", identite)
    cible = {"titre": titre, "date": jour.isoformat(), "jour": jour.weekday(),
             "debut": debut, "fin": fin}
    if recurrent:
        # Un bloc hebdomadaire se dit « le mercredi », jamais « mer. 16 sept. »:
        # la date n'est que la prochaine occurrence qui a servi au calcul.
        cible["recurrent"] = True
    if isinstance(avec, dict) and avec:
        cible["avec"] = dict(avec)
    options = []
    for i, (s, e) in enumerate(_creneaux(ctx.user, jour, duree, _minutes(debut)), start=1):
        cible_option = {"titre": titre, "date": jour.isoformat(),
                        "jour": jour.weekday(), "debut": _fmt(s), "fin": _fmt(e)}
        if recurrent:
            cible_option["recurrent"] = True
        options.append({"id": f"creneau_{i}", "effet": None, "cible": cible_option})
    autre = {"titre": titre, "jour": jour.weekday(), "date": jour.isoformat()}
    if recurrent:
        autre["recurrent"] = True
    options.append({"id": "autre_jour", "effet": None, "cible": autre})
    return dem.construire_demande("choix", "heure_refusee", nom, dict(kwargs), cible, options, cle)


def _demande_chevauchement(nom: str, kwargs: dict, titre: str, dow: int, debut, fin, avec) -> dict:
    cle = dem.cle_demande("chevauchement", {"titre": (titre or "").casefold(), "jour": dow})
    cible = {"titre": titre, "jour": dow, "date": dem.prochaine_occurrence(dow).isoformat()}
    if debut:
        cible["debut"] = debut
    if fin:
        cible["fin"] = fin
    if isinstance(avec, dict) and avec:
        cible["avec"] = dict(avec)
    options = [{"id": "autre_heure", "effet": None, "cible": dict(cible)},
               {"id": "annuler", "effet": None, "cible": dict(cible)}]
    return dem.construire_demande("choix", "chevauchement", nom, dict(kwargs), cible, options, cle)


def _cibles_creation(ctx: _Contexte, nom: str, kwargs: dict, prep: _Preparation):
    """(dates, jours, recurrent, debut) d'un appel createur, ou None."""
    from services.agent.tools.blocks import normaliser_jours

    if nom == "schedule_task_at":
        jour = _date_iso(kwargs.get("date"))
        if jour is None:
            return None
        return {jour}, {jour.weekday()}, False, _heure_normale(kwargs.get("start_time"))
    if nom == "create_block":
        return set(), set(normaliser_jours(kwargs.get("days"))), True, _heure_normale(kwargs.get("start_time"))
    if nom == "update_block" and prep.bloc_update is not None:
        if all(kwargs.get(k) in (None, "") for k in ("start_time", "end_time", "day_of_week")):
            return None
        bloc = prep.bloc_update
        dow = bloc["jour"]
        if kwargs.get("day_of_week") not in (None, ""):
            compris = normaliser_jours(kwargs.get("day_of_week"))
            if compris:
                dow = compris[0]
        debut = _heure_normale(kwargs.get("start_time")) or bloc["debut"]
        return set(), {dow}, True, debut
    return None


def _arme_touche(arme: dict, dates: set, jours: set, recurrent: bool) -> bool:
    if arme.get("date") is not None:
        cible = arme["date"]
        return cible.weekday() in jours if recurrent else cible in dates
    return arme.get("jour") in jours


def _refus_heure_armee(ctx: _Contexte, nom: str, kwargs: dict, prep: _Preparation):
    if not ctx.etat.armes or nom not in ("schedule_task_at", "create_block", "update_block"):
        return None
    cibles = _cibles_creation(ctx, nom, kwargs, prep)
    if cibles is None:
        return None
    dates, jours, recurrent, debut = cibles
    if debut is None:
        return None
    # Revue du 2026-09-14 (round 3): la garde armee retenait TOUT ajout du
    # jour, meme un autre element, sous la cle de la premiere question. Rendu
    # masquait cette retenue et la seconde demande disparaissait sans un mot.
    # Elle ne vise plus que l'element dont l'heure a ete refusee; un element
    # renomme reste couvert par la garde d'heure dite, qui suit.
    titre = str(kwargs.get("title") or (prep.bloc_update or {}).get("titre") or "")
    for arme in ctx.etat.armes:
        if not _arme_touche(arme, dates, jours, recurrent) or debut in arme["heures"]:
            continue
        titre_arme = str(((arme.get("demande") or {}).get("cible") or {}).get("titre") or "")
        if not _meme_element(titre, titre_arme):
            continue
        return ToolResult(success=False, data={"demande": arme["demande"]},
                          message=MESSAGE_RETENUE)
    return None


# ------------------------------------------------ heure dite, avant l'essai
#
# Revue du 2026-09-14: la garde d'heure ne s'armait qu'APRES un refus de
# l'heure dite. Un modele qui lisait find_free_slots, voyait 10 h 30 pris et
# reservait 13 h directement changeait l'heure en silence. Desormais, avant
# tout appel createur qui vise le jour de la proposition ou l'utilisateur a
# donne une heure ferme pour CE titre, une autre heure est retenue.

_SEPARATEURS = re.compile(r"[,;.!?\n]|\bet\b|\bpuis\b|\bensuite\b|\bmais\b")
# Une heure precedee de ces mots est une borne, pas l'heure du rendez-vous.
_HEURE_SOUPLE_AVANT = re.compile(
    r"\b(?:avant|apres|a partir d[e']|des|jusqu'?a|jusque|entre|au plus tard|passe|pas)\s*$")
_HEURE_APPROX_AVANT = re.compile(r"\b(?:vers|environ|autour de|genre)\s*$")
TOLERANCE_APPROX = 30
_MOTS_VIDES_TITRE = {"avec", "pour", "dans", "chez", "cours", "bloc", "rendez", "vous",
                     "tache", "evenement", "seance", "le", "la", "les", "de", "du", "des",
                     "un", "une", "au", "aux", "en", "et", "ma", "mon", "mes", "ton", "ta",
                     "tes", "sa", "son", "ses", "sur", "par"}


def _mots_du_titre(titre: str) -> set:
    """Mots porteurs d'un titre. Deux lettres suffisent: « Gym » ou « Bac »
    donnaient un ensemble vide et la garde d'heure sautait (revue r3)."""
    mots = set()
    for mot in re.findall(r"[a-z0-9]+", dem.sans_accents(titre or "")):
        if len(mot) < 2 or mot in _MOTS_VIDES_TITRE:
            continue
        mots.add(mot[:-1] if mot.endswith("s") and len(mot) > 4 else mot)
    return mots


def _meme_element(titre_a: str, titre_b: str) -> bool:
    """Deux titres designent-ils le meme element ? « Stats » et
    « Statistiques », « RDV dentiste » et « Dentiste » oui; « Lecture » et
    « Dentiste » non. Sans mot porteur, seule l'egalite compte."""
    mots_a, mots_b = _mots_du_titre(titre_a), _mots_du_titre(titre_b)
    if not mots_a or not mots_b:
        return dem.normaliser(titre_a) == dem.normaliser(titre_b)
    for a in mots_a:
        for b in mots_b:
            if a == b:
                return True
            commun = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                commun += 1
            if commun >= 4:
                return True
    return False


def _propositions(plat: str) -> list[tuple[int, int]]:
    bornes, debut = [], 0
    for m in _SEPARATEURS.finditer(plat):
        bornes.append((debut, m.start()))
        debut = m.end()
    bornes.append((debut, len(plat)))
    return [(s, e) for s, e in bornes if plat[s:e].strip()]


def _heure_dite_ignoree(ctx: _Contexte, nom: str, kwargs: dict, prep: _Preparation):
    """ToolResult de refus quand l'appel pose une AUTRE heure que celle que
    l'utilisateur a donnee pour ce titre et ce jour, sinon None."""
    if nom not in ("schedule_task_at", "create_block", "update_block"):
        return None
    if nom == "update_block" and kwargs.get("start_time") in (None, ""):
        return None
    cibles = _cibles_creation(ctx, nom, kwargs, prep)
    if cibles is None:
        return None
    dates, jours, recurrent, debut = cibles
    if debut is None or not jours:
        return None
    titre = str(kwargs.get("title") or (prep.bloc_update or {}).get("titre") or "").strip()
    # « Cours » ou « Rendez-vous » n'ont que des mots vides: leurs mots bruts
    # servent alors a retrouver la proposition.
    mots = _mots_du_titre(titre) or {
        m[:-1] if m.endswith("s") and len(m) > 4 else m
        for m in re.findall(r"[a-z0-9]+", dem.sans_accents(titre)) if len(m) >= 2}
    plat = dem.sans_accents(ctx.texte)
    positions = dem.heures_dites_positions(ctx.texte)
    if not positions:
        return None
    aujourdhui = timezone.localdate()
    dates_message = dem._dates_nommees(ctx.texte, aujourdhui)

    def _jour_cible(dates_visees):
        if dates_visees:
            if recurrent:
                communs = [d for d in dates_visees if d.weekday() in jours]
            else:
                communs = [d for d in dates_visees if d in dates]
            if not communs:
                return None
            return communs[0] if not recurrent else dem.prochaine_occurrence(communs[0].weekday())
        return next(iter(dates)) if dates else dem.prochaine_occurrence(min(jours))

    # Round 6 (D3): le repli « une seule heure ferme dans le message » est
    # retire. Il imposait l'heure donnee a autre chose (« mon cours finit a
    # 15 h, place ma lecture ») et chaque revue trouvait une tournure neuve.
    # La garde ne vaut plus que si le titre de l'appel est dans une
    # proposition qui porte sa propre heure ferme.
    #
    # ECART ACCEPTE: un titre renomme par le modele (« Entrainement » pour
    # « gym ») ou une heure donnee par pronom dans une autre proposition
    # (« ajoute gym jeudi et mets-le a 15 h ») n'est pas garde par le code.
    # La regle du prompt d'AGIR le couvre. Production main n'a aucune garde
    # d'heure dite: c'est strictement mieux. La garde armee (R1), elle, tient
    # toujours le meme element apres un refus.
    #
    # Revue du round 6: une heure DITE n'importe ou dans le message n'est
    # jamais refusee. « Gym jeudi a 15 h, en fait non, a 17 h » posait 15 h;
    # « Mon gym de 15 h, deplace-le a 17 h » ne se deplacait plus. La garde ne
    # retient que l'heure que l'utilisateur n'a dite nulle part.
    toutes = _heures_fermes(plat, positions, 0, len(plat))
    if any(abs(_minutes(debut) - _minutes(v)) <= tol for v, _ou, tol in toutes):
        return None
    # L'heure actuelle d'un bloc deplace le NOMME, elle ne dit pas ou il va.
    actuel = _heure_normale((prep.bloc_update or {}).get("debut")) if nom == "update_block" else None
    # Round 9 (K3): une heure nue dont UNE lecture est l'heure actuelle ne
    # nomme le bloc que si le message donne une autre heure (« mon gym de 7 h,
    # deplace-le a 9 h »). Seule, elle dit ou il va (« deplace mon gym a 7 h »
    # sur un gym de 19 h): son autre lecture reste ferme. Retirer toute la
    # position laissait passer n'importe quelle heure.
    #
    # Revue du round 9: l'autre heure doit viser le MEME bloc. Une heure d'un
    # autre element (« et mon yoga a 9 h », « j'ai un rendez-vous a 10 h »)
    # ne compte pas; seule compte une heure liee au titre dans sa proposition,
    # ou celle d'une proposition sans element propre (« deplace-le a 9 h »).
    heures_sans_element = {
        ou for s2, e2 in _propositions_rattachees(plat) if _proposition_sans_element(plat, s2, e2)
        for _v, ou, _tol in _heures_fermes(plat, positions, s2, e2)}
    for s, e in _propositions_rattachees(plat):
        morceau = plat[s:e]
        spans = [(s + m.start(), s + m.end()) for m in re.finditer(r"[a-z0-9]+", morceau)
                 if (m.group()[:-1] if m.group().endswith("s") and len(m.group()) > 4
                     else m.group()) in mots]
        if not spans:
            continue
        dites = [(v, ou, tol) for v, ou, tol in _heures_fermes(plat, positions, s, e)
                 if _heure_liee_au_titre(plat, ou, spans)]
        nommees = {ou for v, ou, _tol in dites if v == actuel}
        autre_heure = bool(({ou for _v, ou, _tol in dites} | heures_sans_element) - nommees)
        fermes = [(v, tol, ou) for v, ou, tol in dites
                  if v != actuel and not (ou in nommees and autre_heure)]
        if not fermes:
            continue
        jour_cible = _jour_cible(dem._dates_nommees(morceau, aujourdhui) or dates_message)
        if jour_cible is None:
            # Le titre porte sa propre heure, pour un autre jour: rien a opposer.
            return None
        return _heure_dite_contredite(ctx, nom, kwargs, prep, titre, recurrent, debut,
                                      jour_cible, fermes)
    return None


_VERBE_DE_PLACEMENT = re.compile(
    r"(?:deplac|boug|met|mis|plac|decal|avanc|recul|pass|pouss|chang|remet|fix|cal)\w*")
_PRONOMS_DE_RAPPEL = {"le", "la", "les", "l", "y", "lui", "ca", "moi", "plutot", "donc"}


def _proposition_sans_element(plat: str, s: int, e: int) -> bool:
    """La proposition ne nomme aucun element: seulement une heure, des liants,
    un verbe de placement et un pronom (« deplace-le a 9 h »)."""
    if not dem._RE_HEURE.search(plat[s:e]):
        return False
    mots = re.findall(r"[a-z]+", dem._RE_HEURE.sub(" ", plat[s:e]))
    return all(w in _LIANTS_TITRE_HEURE or w in _PRONOMS_DE_RAPPEL
               or _VERBE_DE_PLACEMENT.fullmatch(w) for w in mots)


# Une proposition faite seulement d'une heure (« Gym jeudi. A 15 h. ») se
# rattache a la precedente: c'est la meme phrase coupee par la ponctuation.
_MOTS_D_HEURE_SEULE = {"a", "au", "vers", "environ", "autour", "de", "genre", "pile"}


def _propositions_rattachees(plat: str) -> list[tuple[int, int]]:
    sortie: list[tuple[int, int]] = []
    for s, e in _propositions(plat):
        morceau = plat[s:e]
        reste = re.findall(r"[a-z]+", dem._RE_HEURE.sub(" ", morceau))
        if (sortie and dem._RE_HEURE.search(morceau)
                and all(m in _MOTS_D_HEURE_SEULE for m in reste)):
            sortie[-1] = (sortie[-1][0], e)
            continue
        sortie.append((s, e))
    return sortie


# Ce qui peut separer un titre de SON heure: jour, date, preposition. Tout
# autre mot (« apres mon cours a 15 h », « avant le souper a 18 h ») rattache
# l'heure a un autre element: la garde se tait, comme sur main. Une liste
# incomplete ne coute qu'une garde absente, jamais une heure imposee.
_LIANTS_TITRE_HEURE = {
    "a", "au", "aux", "vers", "environ", "autour", "de", "du", "des", "d", "genre", "pile",
    "le", "la", "les", "l", "ce", "cette", "prochain", "prochaine", "pour", "et", "chaque",
    "tous", "toutes", "semaine", "demain", "aujourd", "hui", "soir", "matin", "midi", "er",
    "soirs", "matins", "soiree", "matinee", "aprem", "cet",
    "janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre",
    "sept", "octobre", "oct", "novembre", "nov", "decembre", "dec",
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
    "lundis", "mardis", "mercredis", "jeudis", "vendredis", "samedis", "dimanches",
}


def _heure_liee_au_titre(plat: str, ou: int, spans: list[tuple[int, int]]) -> bool:
    """L'heure a la position `ou` suit (ou precede) directement un mot du
    titre, separee seulement par des liants."""
    m = dem._RE_HEURE.match(plat, ou)
    fin_heure = m.end() if m else ou
    avant = [sp for sp in spans if sp[1] <= ou]
    if avant:
        segment = plat[avant[-1][1]:ou]
    else:
        apres = [sp for sp in spans if sp[0] >= fin_heure]
        if not apres:
            return False
        segment = plat[fin_heure:apres[0][0]]
    segment = re.sub(r"(?:apres|avant)[- ](?:demain|midi)", " demain ", segment)
    mots = re.findall(r"[a-z]+", dem._RE_HEURE.sub(" ", segment))
    return all(w in _LIANTS_TITRE_HEURE for w in mots)


_MARQUE_MATIN = re.compile(r"[ \t]*(?:du matin\b|am\b|a\.m\.)")
_MARQUE_APRES_MIDI = re.compile(
    r"[ \t]*(?:du soir\b|pm\b|p\.m\.|de l'?[ \t]*apres[- ]midi\b|de l'?[ \t]*aprem\b)")
# Round 9 (K4): un moment de la journee dit dans la meme proposition, avant
# ou apres l'heure (« jeudi soir a 6 h », « a 6 h le soir », « tous les
# matins a 7 h »), choisit la lecture stricte. « souper » est un soir.
_MOMENT_MATIN = re.compile(r"\b(?:matin(?:s|ee|ees)?|avant[- ]midi|am|a\.m\.)(?!\w)")
_MOMENT_SOIR = re.compile(
    r"\b(?:soir(?:s|ee|ees)?|soupers?|apres[- ]midi|aprem|pm|p\.m\.)(?!\w)")


# Revue du round 9: le mot doit QUALIFIER l'heure. Complement d'un autre nom
# (« avant le souper », « mon quart du soir », « ma soiree »), il ne choisit
# rien et les deux lectures restent valides. Dans le doute, rien n'est choisi.
_AVANT_COMPLEMENT = {"du", "de", "d", "au", "aux", "pour", "avant", "apres", "pendant",
                     "durant", "depuis", "jusqu", "sans", "par", "sur", "dans", "pas",
                     "sauf", "ni", "que", "quart", "shift", "cours"}
_DETERMINANTS = {"le", "la", "les", "l", "un", "une", "des", "mon", "ma", "mes", "ton",
                 "ta", "tes", "son", "sa", "ses", "notre", "votre", "nos", "vos", "leur",
                 "leurs"}
_POSSESSIFS = {"mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses", "notre", "votre",
               "nos", "vos", "leur", "leurs"}


def _moment_qualifie_l_heure(plat: str, s: int, mot, heure: tuple[int, int]) -> bool:
    # 1. Seuls des liants entre le mot et l'heure.
    if mot.end() <= heure[0]:
        entre = plat[mot.end():heure[0]]
    else:
        entre = plat[heure[1]:mot.start()]
    if not all(w in _LIANTS_TITRE_HEURE for w in re.findall(r"[a-z]+", entre)):
        return False
    # 2. Pas complement d'un autre nom. « apres-midi »/« avant-midi » portent
    # leur propre preposition: on regarde avant le mot entier.
    avant = re.findall(r"[a-z]+", plat[s:mot.start()])
    if not avant:
        return True
    precedent = avant[-1]
    if precedent in _AVANT_COMPLEMENT:
        return False
    if precedent in _DETERMINANTS:
        if precedent in _POSSESSIFS and not mot.group().startswith("souper"):
            return False  # « ma soiree », « mon matin »: un nom
        if len(avant) >= 2 and avant[-2] in _AVANT_COMPLEMENT:
            return False  # « avant le souper », « pas le soir »
    return True


def _moment_de_la_journee(plat: str, ou: int) -> str | None:
    """« matin » ou « soir » quand un mot de moment de la journee de la
    proposition de l'heure a la position `ou` s'y rattache: parmi les heures
    de la proposition, c'est elle la plus proche du mot. Le mot le plus
    proche l'emporte."""
    borne = next(((s, e) for s, e in _propositions(plat) if s <= ou < e), None)
    if borne is None:
        return None
    s, e = borne
    heures = [(s + h.start(), s + h.end()) for h in dem._RE_HEURE.finditer(plat[s:e])]
    if not heures:
        return None

    def _distance(a, b):
        return max(0, b[0] - a[1], a[0] - b[1])

    meilleur = None
    for genre, motif in (("matin", _MOMENT_MATIN), ("soir", _MOMENT_SOIR)):
        for mot in motif.finditer(plat, s, e):
            span = (mot.start(), mot.end())
            proche = min(heures, key=lambda h: _distance(h, span))
            if proche[0] != ou or not _moment_qualifie_l_heure(plat, s, mot, proche):
                continue
            d = _distance(proche, span)
            if meilleur is None or d < meilleur[0]:
                meilleur = (d, genre)
    return meilleur[1] if meilleur else None


def _lectures_d_heure(plat: str, ou: int, valeur: str) -> list[str]:
    """Les lectures d'une heure dite. Round 8: au Quebec, une heure nue de 1 a
    11 (« souper jeudi a 6 h ») vaut aussi H+12. « du matin », « am » gardent
    H; « du soir », « pm », « de l'apres-midi » donnent H+12. Midi (12 h) et une
    heure ecrite avec un zero (« 07:00 ») restent telles quelles. Round 9: un
    moment de la journee dit dans la proposition choisit aussi la lecture."""
    m = dem._RE_HEURE.match(plat, ou)
    if m is None or m.group(5):
        return [valeur]
    ecrite = m.group(1) if m.group(1) is not None else m.group(3)
    h = int(valeur[:2])
    if not 1 <= h <= 12 or ecrite.startswith("0"):
        return [valeur]
    suite = plat[m.end():]
    apres_midi = f"{h + 12:02d}{valeur[2:]}" if h != 12 else valeur
    if _MARQUE_APRES_MIDI.match(suite):
        return [apres_midi]
    if h == 12 or _MARQUE_MATIN.match(suite):
        return [valeur]
    moment = _moment_de_la_journee(plat, ou)
    if moment == "soir":
        return [apres_midi]
    if moment == "matin":
        return [valeur]
    return [valeur, apres_midi]


def _heures_fermes(plat: str, positions, s: int, e: int) -> list[tuple[str, int, int]]:
    """(valeur, position, tolerance) des heures fermes entre s et e: une
    borne (« avant 10 h ») n'en est pas une, une approximation a sa marge.
    Une heure nue donne une entree par lecture, a la meme position."""
    fermes = []
    for valeur, ou in positions:
        if not (s <= ou < e):
            continue
        avant = plat[:ou]
        if _HEURE_SOUPLE_AVANT.search(avant):
            continue
        tolerance = TOLERANCE_APPROX if _HEURE_APPROX_AVANT.search(avant) else 0
        for lecture in _lectures_d_heure(plat, ou, valeur):
            fermes.append((lecture, ou, tolerance))
    return fermes


def _heure_dite_contredite(ctx: _Contexte, nom: str, kwargs: dict, prep: _Preparation,
                           titre: str, recurrent: bool, debut: str, jour_cible: date, fermes):
    """Refus quand l'appel ne pose aucune des heures fermes dites, sinon None."""
    from services.scheduling.placement import open_intervals

    if any(abs(_minutes(debut) - _minutes(v)) <= tol for v, tol, _ou in fermes):
        return None
    # Les lectures de la premiere heure dite: une heure nue en a deux, et le
    # code n'en impose jamais une seule (round 8).
    lectures = [v for v, _tol, ou in fermes if ou == fermes[0][2]]
    dite = lectures[0]
    fin_appel = _heure_normale(kwargs.get("end_time")) or (prep.bloc_update or {}).get("fin")
    duree = ((_minutes(fin_appel) - _minutes(debut)) % (24 * 60)) if fin_appel else 60
    duree = duree or 60
    fin_dite = _fmt((_minutes(dite) + duree) % (24 * 60))
    ouverts = open_intervals(ctx.user, jour_cible, 0, 24 * 60)

    def _libre(heure):
        debut_min = _minutes(heure)
        fin_min = debut_min + duree
        return fin_min <= 24 * 60 and any(a <= debut_min and fin_min <= b for a, b in ouverts)

    if any(_libre(v) for v in lectures):
        if len(lectures) > 1:
            consigne = (f"l'utilisateur a dit {' ou '.join(lectures)} pour {_ascii(titre)} "
                        "(heure sans matin ni soir, les deux lectures valent). Refais l'appel avec "
                        + " ou ".join(f"start_time={v}" for v in lectures)
                        + " selon le contexte. Si aucune ne convient")
        else:
            consigne = (f"l'utilisateur a dit {dite} pour {_ascii(titre)}. "
                        f"Refais l'appel avec start_time={dite}. Si cette heure ne convient pas")
        return ToolResult(
            success=False, data={"heure_dite": dite},
            message=f"Retenu par le code: {consigne}, pose la question au lieu de changer l'heure.")
    demande = _demande_heure_refusee(ctx, nom, kwargs, titre, jour_cible, dite, fin_dite,
                                     None, recurrent=recurrent)
    heures = [v for v, _tol, _ou in fermes]
    if recurrent:
        _armer(ctx, demande, heures, jour=jour_cible.weekday())
    else:
        _armer(ctx, demande, heures, date_armee=jour_cible)
    return ToolResult(success=False, data={"demande": demande}, message=MESSAGE_RETENUE)


def _date_passee(nom: str, kwargs: dict):
    """schedule_task_at ne place jamais un evenement a une date deja passee:
    sans date du jour, AGIR a deja reserve en 2025 (banc du 2026-09-14)."""
    if nom != "schedule_task_at":
        return None
    jour = _date_iso(kwargs.get("date"))
    aujourdhui = timezone.localdate()
    if jour is None or jour >= aujourdhui:
        return None
    return ToolResult(
        success=False, data={"date_passee": jour.isoformat()},
        message=(f"Refuse par le code: {jour.isoformat()} est deja passe. Aujourd'hui, "
                 f"c'est le {aujourdhui.isoformat()}. Verifie le jour et l'annee, puis refais l'appel."))


# --------------------------------------------------- echeance sans jour choisi

_JOURS_RE = "lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche"
_ECHEANCE = re.compile(
    r"\b(avant|d'ici(?: a)?|au plus tard)\s+(?:(?:ce|le)\s+)?"
    r"(" + _JOURS_RE + r"|demain|la fin de (?:la )?semaine|la fin de semaine)\b"
    r"|\bdans la semaine\b")
_MOIS_LONGS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
               "septembre", "octobre", "novembre", "décembre"]
MAX_JOURS_PROPOSES = 4


def _jours_avant_echeance(m, aujourdhui: date) -> list[date]:
    """Les jours candidats, aujourd'hui compris, jusqu'a l'echeance nommee.
    « avant X » exclut X; « d'ici X » et « au plus tard X » l'incluent."""
    if m.group(1) is None:  # « dans la semaine »
        fin = aujourdhui + timedelta(days=6 - aujourdhui.weekday())
        return [aujourdhui + timedelta(days=i) for i in range((fin - aujourdhui).days + 1)]
    borne = m.group(2)
    if borne == "demain":
        echeance = aujourdhui + timedelta(days=1)
    elif borne.startswith("la fin de"):
        echeance = aujourdhui + timedelta(days=6 - aujourdhui.weekday())
    else:
        dow = _NOMS_JOURS.index(borne)
        echeance = aujourdhui + timedelta(days=(dow - aujourdhui.weekday()) % 7 or 7)
    inclus = m.group(1) != "avant"
    derniere = echeance if inclus else echeance - timedelta(days=1)
    return [aujourdhui + timedelta(days=i) for i in range((derniere - aujourdhui).days + 1)]


def _jour_a_choisir(ctx: _Contexte, nom: str, kwargs: dict):
    """schedule_task_at lance sur une echeance (« avant vendredi ») sans jour
    choisi: le code retient l'appel et propose les jours libres.

    Banc du 2026-09-14 (s09-1): AGIR a place la revision aujourd'hui en
    trouvant la question « borderline ». La reponse depend du tirage du
    modele; la garde la rend stable. Un message qui nomme un jour hors de
    l'echeance (« mercredi avant vendredi »), ou une puce de choix, passe.
    """
    from services.scheduling.placement import open_intervals

    if nom != "schedule_task_at":
        return None
    plat = dem.sans_accents(ctx.texte)
    m = _ECHEANCE.search(plat)
    if m is None:
        return None
    reste = plat[:m.start()] + " " + plat[m.end():]
    # Un jour nomme hors de l'echeance (« mercredi avant vendredi ») passe:
    # le jour est choisi. En cas de doute du juge, la garde reste active et
    # propose les jours libres plutot que de laisser AGIR placer au hasard.
    vise, tranche = dem.jour_vise_tranche(reste)
    if vise and tranche:
        return None
    debut, fin = _heure_normale(kwargs.get("start_time")), _heure_normale(kwargs.get("end_time"))
    duree = ((_minutes(fin) - _minutes(debut)) % (24 * 60)) if debut and fin else 60
    duree = duree or 60
    maintenant = timezone.localtime()
    aujourdhui = timezone.localdate()
    libres: list[date] = []
    for jour in _jours_avant_echeance(m, aujourdhui):
        plancher = maintenant.hour * 60 + maintenant.minute if jour == aujourdhui else 0
        if any(min(b, 24 * 60) - max(a, plancher) >= duree
               for a, b in open_intervals(ctx.user, jour, 0, 24 * 60)):
            libres.append(jour)
        if len(libres) == MAX_JOURS_PROPOSES:
            break
    if len(libres) < 2:
        return None

    titre = str(kwargs.get("title") or "").strip()
    options = []
    for jour in libres:
        if jour == aujourdhui:
            libelle, nomme = "Aujourd'hui", "aujourd'hui"
        elif jour == aujourdhui + timedelta(days=1):
            libelle, nomme = "Demain", "demain"
        else:
            nomme = f"{_NOMS_JOURS[jour.weekday()]} {jour.day} {_MOIS_LONGS[jour.month - 1]}"
            libelle = nomme[0].upper() + nomme[1:]
        valeur = f"Place {titre} {nomme}." if titre else f"Place-le {nomme}."
        options.append({"label": libelle, "value": valeur, "cible": {"date": jour.isoformat()}})
    question = (f"Quel jour veux-tu placer {titre} ?" if titre else "Quel jour te convient ?")
    cle = "choix:" + hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
    demande = dem.construire_demande(
        "choix", "choix_modele", nom, dict(kwargs), {"titre": titre},
        [{"id": f"o{rang}", "effet": None, "cible": o["cible"],
          "libelle": o["label"], "valeur": o["value"]}
         for rang, o in enumerate(options, start=1)],
        cle)
    demande["question"] = question
    demande["source"] = "jours"
    return ToolResult(success=False, data={"demande": demande}, message=MESSAGE_RETENUE)


# ------------------------------------------- jour nomme sans mot de recurrence

# Round 10 (P2, banc r9 k4-1): « souper jeudi soir a 6 h » est devenu une
# serie hebdomadaire. Le raisonnement citait la regle « horaires habituels
# AVEC jours et heures -> create_block » et l'absence de « ce » devant jeudi.
# Un jour nomme sans l'un de ces mots designe UN jour: evenement date.
_RECURRENCE = re.compile(
    r"\b(?:chaque|tous\s+les|toutes\s+les|par\s+semaine|fois\s+par|hebdo\w*|habitud\w*"
    r"|toujours|normalement|regulier\w*|a\s+partir\s+d\w*|jusqu'?\s*(?:a|au)\b|horaire\w*"
    r"|semaine\s+type|session"
    r"|(?:les|le)\s+(?:" + _JOURS_RE + r")s?\b(?!\s*\d)"
    r"|(?:" + _JOURS_RE + r")s)\b")
_JOUR_UNIQUE = re.compile(r"\b(?:" + _JOURS_RE + r"|demain|apres-demain|aujourd'?hui)\b")
_REPONSE_FORMULAIRE = re.compile(r"^\s*voici mes r[ée]ponses", re.IGNORECASE)
# Un cours, un quart ou le sommeil reviennent chaque semaine par defaut (K6).
_TYPES_QUI_REVIENNENT = frozenset({"course", "work", "sleep"})


def jour_sans_recurrence(texte) -> bool:
    """Le message nomme-t-il un jour sans dire que ca revient ?"""
    if not isinstance(texte, str) or not texte.strip() or _REPONSE_FORMULAIRE.match(texte):
        return False
    plat = dem.sans_accents(texte).lower()
    return bool(_JOUR_UNIQUE.search(plat)) and not _RECURRENCE.search(plat)


def _reponse_a_une_habitude(ctx: _Contexte) -> bool:
    """Le message repond-il a une question posee sur une demande qui disait
    deja la recurrence (« ajoute du yoga chaque semaine », puis « samedi a
    10 h ») ?"""
    from core.models import ConversationMessage

    derniers = ConversationMessage.objects.filter(user=ctx.user).order_by("-pk")[:6]
    for message in derniers:
        if message.role != "assistant":
            continue
        meta = message.metadata or {}
        if not meta.get("question_posee") and not meta.get("demandes") \
                and not meta.get("interactive_inputs"):
            return False
        origine = ConversationMessage.objects.filter(
            user=ctx.user, pk=meta.get("en_reponse_a"), role="user").first()
        return bool(origine and _RECURRENCE.search(dem.sans_accents(origine.content or "").lower()))
    return False


def _evenement_unique(ctx: _Contexte, nom: str, kwargs: dict):
    """create_block d'une activite lance sur un jour nomme sans recurrence:
    le code le retient et dit d'appeler schedule_task_at a la bonne date."""
    from services.agent.tools.blocks import normaliser_jours

    if nom != "create_block" or str(kwargs.get("block_type") or "") in _TYPES_QUI_REVIENNENT:
        return None
    if not jour_sans_recurrence(ctx.texte) or _reponse_a_une_habitude(ctx):
        return None
    aujourdhui = timezone.localdate()
    nommes = {d.weekday() for d in dem._dates_nommees(ctx.texte, aujourdhui)}
    dows = {dow for dow in (_entier(j) for j in normaliser_jours(kwargs.get("days")))
            if dow is not None and 0 <= dow <= 6}
    # Le jour nomme doit etre celui de CET appel: « efface le quart de jeudi,
    # ajoute mes etudes » ne dit rien des jours des etudes.
    if not dows or not dows <= nommes:
        return None
    dates = [dem.prochaine_occurrence(dow, aujourdhui) for dow in dows]
    isos = sorted({d.isoformat() for d in dates})
    return ToolResult(
        success=False, data={"evenement_unique": isos},
        message=("Retenu par le code: l'utilisateur a nomme un jour sans mot de recurrence "
                 "(chaque, tous les, les lundis). C'est un seul jour, pas une habitude: "
                 "n'appelle pas create_block. Appelle schedule_task_at avec le meme titre et "
                 "les memes heures, date=" + " puis date=".join(isos)
                 + ". S'il voulait chaque semaine, il le dira."))


# ------------------------------------ formulaire: l'heure avec les jours (P4)

# Round 10 (P4, banc r9 s06-1): le formulaire demandait les jours et la duree
# de la gym, pas l'heure; au tour suivant AGIR a choisi 16 h seul. Un
# formulaire qui demande les jours d'une activite demande aussi sa plage
# horaire, sans defaut invente. Un volume total (« 4 h d'etude en tout ») se
# repartit et n'a pas d'heure; le sommeil garde son defaut produit.
_NOMS_JOURS_FORM = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_TOTAL_FORM = re.compile(
    r"\b(?:total\w*|en tout|heures d|heures de|temps d|temps de|nombre d'?heures|volume)\b")
_PAR_SEANCE_FORM = re.compile(r"\b(?:seances?|par fois|chaque fois|une fois)\b")
_SOMMEIL_FORM = re.compile(r"\b(?:sommeil|dormir|dors|coucher|couches|reveil\w*|nuit|dodo)\b")


def _plat_form(champ: dict) -> str:
    return dem.sans_accents(" ".join(str(champ.get(k) or "") for k in ("id", "label", "question"))).lower()


def _champ_de_jours(champ: dict) -> bool:
    if champ.get("type") != "checkbox" or not isinstance(champ.get("options"), list):
        return False
    vus = set()
    for option in champ["options"]:
        if not isinstance(option, dict):
            continue
        libelle = dem.sans_accents(str(option.get("label") or "")).lower().strip().rstrip(".")
        for rang, nom in enumerate(_NOMS_JOURS_FORM):
            if libelle == nom or libelle == nom[:3]:
                vus.add(rang)
    return len(vus) >= 5


def formulaire_avec_heure(texte: str, kwargs: dict) -> dict:
    """Les arguments de present_form, completes d'une plage horaire sans
    defaut quand le formulaire demande des jours sans demander l'heure."""
    inputs = kwargs.get("inputs")
    if not isinstance(inputs, list):
        return kwargs
    champs = [c for c in inputs if isinstance(c, dict)]
    if not any(_champ_de_jours(c) for c in champs):
        return kwargs
    if any(c.get("type") in ("duration", "number") and _TOTAL_FORM.search(_plat_form(c))
           and not _PAR_SEANCE_FORM.search(_plat_form(c)) for c in champs):
        return kwargs
    dites = dem.heures_dites(texte or "")
    heure_demandee = False
    nouveaux = []
    for champ in inputs:
        if isinstance(champ, dict) and champ.get("type") in ("time_range", "time"):
            heure_demandee = True
            defaut = champ.get("default")
            debut = defaut.get("start") if isinstance(defaut, dict) else defaut
            if "default" in champ and not _SOMMEIL_FORM.search(_plat_form(champ)) \
                    and _heure_normale(debut) not in dites:
                champ = {k: v for k, v in champ.items() if k != "default"}
        nouveaux.append(champ)
    if not heure_demandee and not dites:
        ids = {c.get("id") for c in champs}
        ident = "plage_horaire"
        while ident in ids:
            ident += "_2"
        nouveaux.append({"id": ident, "type": "time_range", "label": "Plage horaire",
                         "question": "De quelle heure à quelle heure ?"})
    return {**kwargs, "inputs": nouveaux}


# ------------------------------------------------------- creations en masse

def _crees_ce_tour(registre: Registre) -> tuple[int, list[str]]:
    """Elements crees ce tour, dedupliques par identifiant: une action
    rejouee par l'idempotence est consignee deux fois sans rien creer."""
    vus: set = set()
    titres: list[str] = []

    def _noter(ident, titre):
        if ident in vus:
            return
        vus.add(ident)
        if titre and titre not in titres:
            titres.append(titre)

    for a in registre.actions:
        if not a.succes:
            continue
        donnees = a.donnees or {}
        if a.outil == "create_block":
            for cree in donnees.get("created") or []:
                if isinstance(cree, dict):
                    _noter(("bloc", cree.get("id"), a.id if cree.get("id") is None else None),
                           cree.get("title"))
        elif a.outil in ("schedule_task_at", "replace_block_occurrence"):
            sb = donnees.get("scheduled_block") or {}
            _noter(("evenement", sb.get("id") or a.id), sb.get("title"))
        elif a.outil == "create_task" and not donnees.get("deja_presente"):
            tache = donnees.get("task") or {}
            _noter(("tache", tache.get("id") or a.id), tache.get("title"))
    return len(vus), titres


def _titre_du_formulaire(ctx: _Contexte, nom: str, kwargs: dict):
    """Au tour de la reponse au formulaire du code (regle formulaire_cours),
    un create_block sous le titre d'un cours DEJA a l'horaire est renvoye UNE
    fois au modele avec le nom que l'utilisateur a donne.

    Banc reel du 2026-09-15: « mon labo de chimie » est devenu une seconde
    serie « Chimie generale (labo) ». La decision se lit sur des donnees
    structurees (le nom garde dans les metadonnees du formulaire, les titres en
    base), jamais sur les mots du message. Un second appel au meme titre passe:
    l'utilisateur voulait peut-etre vraiment une seance de plus de ce cours."""
    from core.models import ConversationMessage, RecurringBlock

    if nom != "create_block" or ctx.etat.attente.get("titre_formulaire_renvoye"):
        return None
    titre = str(kwargs.get("title") or "").strip()
    _, _, courant = str(ctx.tache or "").rpartition(":")
    if not titre or not courant.isdigit():
        return None
    deux = list(ConversationMessage.objects.filter(user=ctx.user).order_by("-pk")[:2])
    if len(deux) < 2 or deux[0].pk != int(courant) or deux[1].role != "assistant":
        return None
    meta = deux[1].metadata if isinstance(deux[1].metadata, dict) else {}
    nom_donne = str(meta.get("formulaire_nom") or "").strip()
    plat, plat_donne = dem.normaliser(titre), dem.normaliser(nom_donne)
    # Le nom de l'utilisateur porte deja ce titre (« mon cours de chimie
    # generale » contre « Chimie generale »): c'est bien ce cours-la qu'il
    # nomme, une seance de plus ne se discute pas. En MOTS ENTIERS et a partir
    # de trois lettres: « Art » se cachait dans « mon cours de cartographie »
    # et laissait passer un titre sans rapport (relecture Codex).
    porte_le_titre = len(plat) >= 3 and f" {plat} " in f" {plat_donne} "
    if not nom_donne or plat == plat_donne or porte_le_titre:
        return None
    existants = RecurringBlock.objects.filter(user=ctx.user, active=True).values_list("title", flat=True)
    if dem.normaliser(titre) not in {dem.normaliser(t) for t in existants}:
        return None
    ctx.etat.attente["titre_formulaire_renvoye"] = True
    return (f"L'utilisateur vient de donner les jours et les heures de « {nom_donne} », un cours "
            f"qu'il AJOUTE. « {titre} » est le titre d'un cours deja a son horaire: cree ce "
            f"nouveau cours sous le nom de l'utilisateur, sans le determinant (mon, ma, mes), "
            f"par exemple « {nom_donne} » mis en forme de titre. Seulement s'il voulait vraiment "
            f"une seance de plus de « {titre} », rappelle create_block avec ce titre.")


MOTIFS_QUESTION_MODELE = ("choix_modele", "question_libre")


def _question_deja_posee(ctx: _Contexte, nom: str, kwargs: dict):
    """Une seule question du modele par tour. Si le registre porte deja une
    demande de motif choix_modele ou question_libre emise ce tour, le second
    appel est refuse avec une consigne, pas execute: deux questions ne se
    rendent pas, et la seconde ecraserait la premiere au tap."""
    if nom not in ("poser_question", "present_choices"):
        return None
    for a in ctx.registre.actions:
        demande = (a.donnees or {}).get("demande")
        if (isinstance(demande, dict)
                and demande.get("motif") in MOTIFS_QUESTION_MODELE):
            return ToolResult(
                success=False,
                message=("Question non posee : tu as deja pose une question ce tour. "
                         "Attends la reponse de l'utilisateur."))
    return None


def _garde_creations(ctx: _Contexte, nom: str, kwargs: dict):
    from services.agent.tools.blocks import normaliser_jours

    if nom not in CREATEURS:
        return None
    deja, titres = _crees_ce_tour(ctx.registre)
    prospectif = len(normaliser_jours(kwargs.get("days"))) if nom == "create_block" else 1
    if deja + prospectif <= SEUIL_CREATIONS:
        return None
    for demande in _attente(ctx):
        if demande.get("motif") != "creation_en_masse":
            continue
        option = dem.option_choisie(ctx.texte, demande, tap=ctx.tap)
        if option == "confirmer":
            return None
        if option is not None:
            return MESSAGE_DEJA_TRANCHE
    titre = str(kwargs.get("title") or "").strip()
    # « titres » ne nomme que ce qui est RETENU: la question « Je continue
    # avec X ? » ne doit jamais lister ce qui vient d'etre ajoute (revue du
    # 2026-09-14). Les titres deja crees vivent a part, pour le compte.
    cible = {"nombre": deja + prospectif, "deja": deja, "titre": titre,
             "titres": [titre] if titre else [], "crees": list(titres)}
    options = [{"id": "confirmer", "effet": None, "cible": dict(cible)},
               {"id": "annuler", "effet": None, "cible": dict(cible)}]
    demande = dem.construire_demande("confirmation", "creation_en_masse", nom, dict(kwargs),
                                     cible, options, "creation_en_masse")
    return _refus(demande, destructif=False)


# --------------------------------------------------------- semaine bornee

def _borner_semaine(ctx: _Contexte, nom: str, kwargs: dict):
    """« cette semaine » ne cree pas d'habitude sans fin: la fin tombe le
    dimanche de la semaine courante. Pas de question, la borne est rendue."""
    if nom != "create_block" or str(kwargs.get("end_date") or "").strip():
        return None
    plat = dem.sans_accents(ctx.texte)
    if not _SEMAINE.search(plat) or _SEMAINE_SANS_FIN.search(plat):
        return None
    aujourdhui = timezone.localdate()
    fin = aujourdhui + timedelta(days=6 - aujourdhui.weekday())
    debut_brut = str(kwargs.get("start_date") or "").strip()
    if debut_brut:
        debut = _date_iso(debut_brut)
        if debut is not None and debut > fin:
            return None
    else:
        kwargs["start_date"] = aujourdhui.isoformat()
    kwargs["end_date"] = fin.isoformat()
    return {"end_date": fin.isoformat()}


# -------------------------------------------------------- apres execution

def _armer(ctx: _Contexte, demande: dict, heures: list, date_armee=None, jour=None) -> None:
    ctx.etat.armes.append({"date": date_armee, "jour": jour, "heures": list(heures),
                           "demande": demande})


def _apres(ctx: _Contexte, nom: str, kwargs: dict, resultat: ToolResult,
           prep: _Preparation) -> ToolResult:
    donnees = dict(resultat.data or {})
    change = False
    if prep.borne_auto:
        donnees["borne_auto"] = prep.borne_auto
        change = True
    heures = dem.heures_dites(ctx.texte)
    demande = None

    if nom == "schedule_task_at" and not resultat.success and isinstance(donnees.get("conflict"), dict):
        debut = _heure_normale(kwargs.get("start_time"))
        fin = _heure_normale(kwargs.get("end_time"))
        jour = _date_iso(kwargs.get("date"))
        if heures and debut in heures and fin and jour is not None:
            conflit = donnees["conflict"]
            avec = {"titre": conflit.get("titre"), "debut": conflit.get("start_time"),
                    "fin": conflit.get("end_time")}
            demande = _demande_heure_refusee(ctx, nom, kwargs, str(kwargs.get("title") or "").strip(),
                                             jour, debut, fin, avec, recurrent=False)
            _armer(ctx, demande, heures, date_armee=jour)

    elif nom == "create_block":
        chevauches = [s for s in donnees.get("skipped") or []
                      if isinstance(s, dict) and s.get("motif") == "chevauchement"]
        if chevauches:
            touches = [s for s in chevauches if heures and s.get("debut") in heures]
            titre = str(kwargs.get("title") or "").strip()
            if touches:
                premier = touches[0]
                dow = int(premier.get("day"))
                demande = _demande_heure_refusee(
                    ctx, nom, kwargs, titre, dem.prochaine_occurrence(dow),
                    premier["debut"], premier.get("fin") or premier["debut"],
                    premier.get("avec"), recurrent=True)
                for s in touches:
                    _armer(ctx, demande, heures, jour=int(s.get("day")))
            elif not donnees.get("created"):
                premier = chevauches[0]
                demande = _demande_chevauchement(nom, kwargs, titre, int(premier.get("day")),
                                                 premier.get("debut"), premier.get("fin"),
                                                 premier.get("avec"))

    elif nom == "update_block" and not resultat.success and isinstance(donnees.get("conflit"), dict):
        conflit = donnees["conflit"]
        bloc = prep.bloc_update or {}
        dow = conflit.get("jour", bloc.get("jour"))
        debut = _heure_normale(kwargs.get("start_time")) or bloc.get("debut")
        fin = _heure_normale(kwargs.get("end_time")) or bloc.get("fin")
        titre = str(kwargs.get("title") or bloc.get("titre") or "").strip()
        avec = {"titre": conflit.get("titre"), "debut": conflit.get("debut"), "fin": conflit.get("fin")}
        if dow is not None and heures and debut in heures and fin:
            demande = _demande_heure_refusee(ctx, nom, kwargs, titre, dem.prochaine_occurrence(int(dow)),
                                             debut, fin, avec, recurrent=True)
            _armer(ctx, demande, heures, jour=int(dow))
        elif dow is not None:
            demande = _demande_chevauchement(nom, kwargs, titre, int(dow), debut, fin, avec)

    if demande is not None:
        donnees["demande"] = demande
        change = True
    if not change:
        return resultat
    return ToolResult(success=resultat.success, data=donnees, message=resultat.message)


# -------------------------------------------------------------------- noyau

def _consigner(ctx: _Contexte, nom: str, kwargs: dict, resultat: ToolResult,
               cle_operation: str = ""):
    action = ctx.registre.ajouter(nom, kwargs, resultat, cle_operation=cle_operation)
    if ctx.signaler:
        ctx.signaler(action)
    return action


def _pour_empreinte(nom: str, kwargs: dict) -> dict:
    """Arguments de la cle d'idempotence. `confirm` est force comme le fera
    la garde, pour qu'un rejeu retrouve l'appel autorise. Les kwargs prives
    (prefixe _) sont exclus: ce sont des hints d'execution, pas l'intention
    metier (ex: _arrangements du plan OR-Tools valide)."""
    if nom in ("delete_task", "clear_all_blocks"):
        return {**kwargs, "confirm": True}
    return {k: v for k, v in (kwargs or {}).items() if not k.startswith("_")}


# Les ecritures dont une reemission aveugle creerait un doublon visible.
# Pour les autres (update, delete, ...), la reconciliation est une relecture
# d'etat que le modele fait lui-meme avec ses outils de lecture.
_CREATIONS_RECONCILIABLES = {"create_block", "create_task", "schedule_task_at"}


def _transitoire(erreur: BaseException) -> bool:
    """Une exception qui laisse le resultat d'une ecriture AMBIGU.

    Timeout, connexion perdue, 429, base momentanement indisponible: dans
    tous ces cas l'ecriture a pu etre validee cote base avant que l'erreur
    ne remonte. Le resultat n'est ni un succes ni un echec, c'est une
    tentative en attente de confirmation.
    """
    from django.db import OperationalError
    nom = type(erreur).__name__
    if isinstance(erreur, (TimeoutError, OperationalError, ConnectionError)):
        return True
    return nom in {"TimeoutError", "ConnectTimeout", "ReadTimeout",
                   "TooManyRequests", "RateLimitError", "ServiceUnavailable",
                   "OperationalError", "InterfaceError", "InternalError"}


def _doublon_recent(user, nom: str, kwargs: dict):
    """L'objet que la tentative ambigue aurait cree, s'il existe en base.

    Fenetre: cree depuis le debut du tour (bornée a 10 minutes, le tour ne
    dure jamais plus longtemps). Retourne l'objet trouve ou None. Lecture
    seule: aucun effet de bord.
    """
    from django.utils import timezone

    from core.models import RecurringBlock, ScheduledBlock, Task
    debut = timezone.now() - timezone.timedelta(minutes=10)
    titre = str(kwargs.get("title") or "").strip()
    if not titre:
        return None
    try:
        if nom == "create_block":
            jours = kwargs.get("days") or kwargs.get("day_of_week")
            if isinstance(jours, int):
                jours = [jours]
            filtre = RecurringBlock.objects.filter(
                user=user, active=True, title=titre, created_at__gte=debut)
            if jours:
                filtre = filtre.filter(day_of_week__in=list(jours))
            debut_h = str(kwargs.get("start_time") or "")
            if debut_h:
                filtre = filtre.filter(start_time=debut_h)
            return filtre.order_by("-created_at").first()
        if nom == "create_task":
            return Task.objects.filter(
                user=user, title=titre, created_at__gte=debut).order_by("-created_at").first()
        if nom == "schedule_task_at":
            # Filtre precis (tache, date, heures) : sans lui, la fenetre de
            # 10 minutes confondrait l'operation ambigue avec une creation
            # anterieure legitime du meme tour.
            filtre = ScheduledBlock.objects.filter(
                user=user, created_at__gte=debut)
            if titre:
                filtre = filtre.filter(task__title=titre)
            jour = kwargs.get("date")
            if jour:
                filtre = filtre.filter(date=jour)
            debut_h = str(kwargs.get("start_time") or "")
            if debut_h:
                filtre = filtre.filter(start_time=debut_h)
            fin_h = str(kwargs.get("end_time") or "")
            if fin_h:
                filtre = filtre.filter(end_time=fin_h)
            return filtre.order_by("-created_at").first()
    except Exception:  # noqa: BLE001 - la reconciliation ne casse jamais un tour
        logger.error("Reconciliation: recherche de doublon en panne", exc_info=True)
    return None


def _reconcilier(ctx: _Contexte, nom: str, kwargs: dict, cle: str) -> str | None:
    """Avant de reemettre une mutation dont une tentative est ambigue.

    Rend la chaine a renvoyer au modele si la reconciliation tranche, None
    pour laisser l'execution suivre son cours. Deux issues tranchent:
    - l'objet existe en base -> recu reconcilie rejoue comme succes;
    - l'objet n'existe pas et l'outil est une creation reconciliable ->
      None (la reemission est sure : rien n'a ete ecrit).
    Pour les autres outils, on laisse le modele verifier lui-meme: le
    message de pending_confirmation le lui a dit.
    """
    attente = ctx.registre.en_attente(cle)
    if attente is None or nom not in _CREATIONS_RECONCILIABLES:
        return None
    doublon = _doublon_recent(ctx.user, nom, kwargs)
    if doublon is None:
        logger.info("Reconciliation %s: rien en base, reemission autorisee", cle)
        return None
    resultat = ToolResult(
        success=True,
        message=f"{attente.message} (confirme par reconciliation)",
        data={**(attente.donnees or {}), "pending_confirmation": False,
              "reconciliation": True,
              "block_id": getattr(doublon, "pk", None)},
    )
    # La cle d'idempotence retient le recu reconcilie: une troisieme
    # tentative rejouera celui-ci, jamais une nouvelle ecriture.
    ctx.etat.cache[cle] = resultat
    _consigner(ctx, nom, kwargs, resultat, cle_operation=cle)
    logger.info("Reconciliation %s: ecriture retrouvee en base, recu rejoue", cle)
    return resultat.to_string()


def _executer_appel(ctx: _Contexte, outil, kwargs: dict, choix: dict | None = None) -> str:
    """Le chemin UNIQUE de tout appel d'outil du tour, qu'il vienne du modele
    ou d'un choix de l'utilisateur execute par le code (choix non nul).
    Synchrone: il tourne dans un thread d'executeur, verrou du tour tenu
    pour les MUTATIONS seulement (voir _fabriquer)."""
    nom = outil.name
    kwargs = dict(kwargs or {})
    registre = ctx.registre
    etat = ctx.etat
    prep = _Preparation()
    garde = None

    if choix is None:
        try:
            garde = _analyser(ctx, nom, kwargs)
            if _deja_fait(ctx, nom, garde):
                logger.info("Idempotence: %s deja fait par le code ce tour", nom)
                return DEJA_FAIT
            prep.borne_auto = _borner_semaine(ctx, nom, kwargs)
            if nom == "update_block":
                from core.models import RecurringBlock

                bid = _entier(kwargs.get("block_id"))
                block = (RecurringBlock.objects.filter(id=bid, user=ctx.user, active=True).first()
                         if bid is not None else None)
                if block is not None:
                    prep.bloc_update = _cible_bloc(block)
        except Exception:  # noqa: BLE001
            logger.error("Garde du code en panne sur %s", nom, exc_info=True)
            if _garde_critique(nom, kwargs):
                refus = ToolResult(success=False, data={"needs_confirmation": True},
                                   message=MESSAGE_RETENUE)
                _consigner(ctx, nom, kwargs, refus)
                return refus.to_string()
            garde = None
    else:
        existante = next(
            (a for a in registre.actions
             if a.succes and a.outil == nom and a.donnees.get("cle_demande") == choix.get("cle")),
            None)
        if existante is not None:
            return DEJA_FAIT

    # IDEMPOTENCE, sur les ECRITURES seulement. Observe en production le
    # 2026-08-27: un tour bloque a fait reessayer le client trois fois, et
    # create_block est parti trois fois avec les memes arguments.
    #
    # La cle vient de l'identite METIER de l'action (tache, outil,
    # arguments normalises) et JAMAIS d'un identifiant tire a chaque
    # tentative. Un choix execute par le code a sa propre cle, liee a la
    # demande. Les LECTURES en sont exclues: elles doivent refleter l'etat
    # courant, sinon l'agent devient aveugle a ses propres ecritures.
    cle = None
    if nom in OUTILS_DE_MUTATION:
        if choix is not None:
            cle = f"{ctx.tache}:choix:{choix.get('cle')}"
        else:
            cle = f"{ctx.tache}:{_empreinte(nom, _pour_empreinte(nom, kwargs))}"
        if cle in etat.cache:
            deja = etat.cache[cle]
            logger.info(f"Executing tool: {nom}({kwargs})")
            logger.info("Idempotence: %s deja execute ce tour, resultat rejoue", nom)
            _consigner(ctx, nom, kwargs, deja, cle_operation=cle)
            return deja.to_string()
        # RECONCILIATION (boucle unique, 2026-09-29). Une tentative precedente
        # de la meme operation est restee ambigue (timeout, exception
        # transitoire): on ne reemet JAMAIS a l'aveugle. On verifie d'abord
        # en base si l'ecriture a bien eu lieu; si oui, le recu reconcilie
        # est rejoue comme un succes, sans reexecuter.
        if choix is None:
            reconcilie = _reconcilier(ctx, nom, kwargs, cle)
            if reconcilie is not None:
                return reconcilie

    if choix is None:
        try:
            issue = _appel_composite(nom, kwargs)
            if issue is None and garde is not None and garde.actif:
                if garde.motif == "optimisation":
                    issue = _garde_optimisation(ctx, outil, kwargs, garde)
                elif garde.cles & (etat.attente.get("cibles_changees") or set()):
                    # K1: la puce de ce tour visait une cible qui a change.
                    # Ni execution ni nouvelle question: la ligne du code le dit.
                    issue = MESSAGE_CIBLE_CHANGEE
                else:
                    autorise, repondue, en_suspens = _reponse(ctx, garde.cles, nom)
                    if autorise and _cible_changee(ctx.user, repondue):
                        _abandon_cible_changee(ctx, repondue)
                        issue = MESSAGE_CIBLE_CHANGEE
                    elif autorise:
                        if nom in ("delete_task", "clear_all_blocks"):
                            kwargs["confirm"] = True
                    elif repondue is not None:
                        issue = MESSAGE_DEJA_TRANCHE
                    elif en_suspens is not None and garde.motif != "portee_jour":
                        # « oui » a une question de portee: c'est la portee
                        # qu'on repose, pas une confirmation generique.
                        issue = _refus(_reposer(en_suspens), destructif=True)
                    else:
                        issue = _refus(garde.demande(), destructif=True)
            if issue is None:
                issue = _date_passee(nom, kwargs)
            if issue is None:
                issue = _refus_heure_armee(ctx, nom, kwargs, prep)
            if issue is None:
                issue = _heure_dite_ignoree(ctx, nom, kwargs, prep)
            if issue is None:
                issue = _jour_a_choisir(ctx, nom, kwargs)
            if issue is None:
                issue = _evenement_unique(ctx, nom, kwargs)
            if issue is None:
                issue = _garde_creations(ctx, nom, kwargs)
            if issue is None:
                issue = _titre_du_formulaire(ctx, nom, kwargs)
            if issue is None:
                issue = _question_deja_posee(ctx, nom, kwargs)
            if nom == "present_form":
                kwargs = formulaire_avec_heure(ctx.texte, kwargs)
        except Exception:  # noqa: BLE001
            logger.error("Garde du code en panne sur %s", nom, exc_info=True)
            issue = (ToolResult(success=False, data={"needs_confirmation": True},
                                message=MESSAGE_RETENUE)
                     if _garde_critique(nom, kwargs) else None)
        if isinstance(issue, str):
            return issue
        if issue is not None:
            _consigner(ctx, nom, kwargs, issue)
            return issue.to_string()
    elif nom == "optimize_week" and kwargs.get("apply"):
        # Le plan confirme doit etre CELUI qu'on applique: s'il a change
        # depuis la question, on redemande au lieu d'appliquer autre chose.
        # _plan_propose sert le plan en cache quand les entrees sont
        # inchangees (0 s de solveur), re-resout sinon.
        try:
            proposition, empreinte, arrangements = _plan_propose(outil, ctx.user, kwargs)
        except Exception as e:  # noqa: BLE001
            logger.error("Proposition de plan en panne: %s", e, exc_info=True)
            proposition, empreinte, arrangements = (
                ToolResult(success=False, data={}, message=f"Erreur de l'outil: {e}"), None, None)
        if empreinte is None or empreinte != (choix.get("parametres") or {}).get("plan_hash"):
            donnees = {"cle_demande": choix.get("cle"), "par_le_code": True,
                       "decision_code": DECISION_EXECUTE}
            if empreinte is not None:
                donnees["demande"] = _demande_optimisation(kwargs, empreinte, proposition)
                message = MESSAGE_RETENUE
            else:
                message = proposition.message or MESSAGE_RETENUE
            refus = ToolResult(success=False, data=donnees, message=message)
            _consigner(ctx, nom, kwargs, refus)
            return refus.to_string()
        # Niveau 3: l'apply reutilise l'arrangement valide au lieu de
        # re-resoudre (kwarg prive, retire apres l'execution pour ne pas
        # polluer le registre).
        kwargs["_arrangements"] = arrangements

    if nom == "organize_day" and kwargs.get("apply"):
        # Niveau 3 (jour): l'apply reutilise l'arrangement valide quand les
        # entrees sont inchangees, au lieu de re-resoudre (kwarg prive,
        # retire apres l'execution pour ne pas polluer le registre).
        # Pas de garde dediee pour organize_day: le chemin est direct.
        try:
            jour = _jour_organize(kwargs.get("date"))
            if jour is not None:
                arrangement, _ = _arrangement_jour_cache(ctx.user, jour)
                if arrangement is not None:
                    kwargs["_arrangement"] = arrangement
        except Exception:  # noqa: BLE001
            logger.error("Cache plan jour en panne", exc_info=True)

    # Meme marqueur que v1: le banc capte les appels d'outils par le
    # logger parent « services », et cette ligne est ce qu'il cherche.
    # Les kwargs prives (ex: _arrangements) sont exclus de l'affichage:
    # le banc ne lit que le nom, et un dump multi-Ko polluerait les logs.
    kwargs_publics = {k: v for k, v in kwargs.items() if not k.startswith("_")}
    logger.info(f"Executing tool: {nom}({kwargs_publics})")
    try:
        if nom == "optimize_week" and not kwargs.get("apply"):
            # La proposition passe par _plan_propose pour ALIMENTER le cache
            # inter-tours: a la confirmation, le plan est resservi sans
            # re-resoudre si les entrees sont inchangees. Resultat identique
            # a l'execution directe (memes helpers de mise en forme).
            resultat, _, _ = _plan_propose(outil, ctx.user, kwargs)
        elif nom == "organize_day" and not kwargs.get("apply"):
            # La proposition passe par _plan_propose_jour pour ALIMENTER le
            # cache inter-tours: a l'apply, l'arrangement est reutilise sans
            # re-resoudre si les entrees sont inchangees. Resultat identique
            # a l'execution directe (memes helpers de mise en forme).
            resultat, _, _ = _plan_propose_jour(outil, ctx.user, kwargs)
        else:
            resultat = outil.execute(ctx.user, **kwargs)
    except Exception as e:  # noqa: BLE001
        # v1 degrade une exception d'outil en ToolResult d'echec. Sans
        # cela, l'exception avorterait le run ET le registre ne garderait
        # aucune trace de la mutation tentee.
        logger.error("Tool %s a leve: %s", nom, e, exc_info=True)
        if cle is not None and _transitoire(e):
            # AMBIGU, pas echoue: l'ecriture a peut-etre eu lieu (timeout
            # apres commit, 429, connexion perdue). Ni succes ni echec:
            # pending_confirmation. Le modele doit VERIFIER par une lecture
            # avant toute reemission, et le code reconciliera par requete
            # (voir _reconcilier). On ne met PAS en cache: un echec mis en
            # cache empecherait toute reprise, et un succes serait un
            # mensonge.
            resultat = ToolResult(
                success=False,
                data={"pending_confirmation": True, "cle_operation": cle,
                      "erreur": type(e).__name__},
                message=(f"{nom}: resultat incertain ({type(e).__name__}), "
                         "action en attente de confirmation. Verifie l'etat "
                         "reel avec un outil de lecture avant de reessayer; "
                         "ne reemets pas a l'aveugle."),
            )
        else:
            resultat = ToolResult(success=False, data={}, message=f"Erreur de l'outil: {e}")
    # Kwargs prives du niveau 3: ne doivent pas polluer le registre ni le rendu.
    kwargs.pop("_arrangements", None)
    kwargs.pop("_arrangement", None)

    if choix is not None:
        resultat = ToolResult(
            success=resultat.success,
            data={**(resultat.data or {}), "cle_demande": choix.get("cle"), "par_le_code": True,
                  "decision_code": DECISION_EXECUTE},
            message=resultat.message,
        )
    else:
        try:
            resultat = _apres(ctx, nom, kwargs, resultat, prep)
        except Exception:  # noqa: BLE001 - une question en moins, jamais un tour tombe
            logger.error("Post-traitement de %s en panne", nom, exc_info=True)

    # On ne met en cache que les SUCCES: un echec peut etre transitoire
    # (429, timeout), et rejouer un echec empecherait toute reprise.
    # Un pending_confirmation n'est ni l'un ni l'autre: il reste
    # reconcilable par _reconcilier a la prochaine tentative.
    if cle is not None and resultat.success:
        etat.cache[cle] = resultat
    # Diffuse au fil de l'execution: c'est ce qui meuble l'attente cote
    # interface.
    _consigner(ctx, nom, kwargs, resultat, cle_operation=cle or "")

    # Garde de terminaison, ici parce que c'est le seul point par lequel
    # TOUS les appels passent. On rend au modele une consigne explicite
    # plutot que de lever: il peut encore rediger une reponse utile.
    if boucle_detectee(registre):
        registre.boucle_interrompue = True
        logger.warning(
            "Boucle detectee: %s rejoue a l'identique, tour interrompu", nom)
        return (
            "ARRET: tu viens de rejouer trois fois la meme action avec les "
            "memes arguments sans progresser. N'appelle plus d'outil. "
            "Reponds a l'utilisateur avec ce que tu as deja."
        )
    return resultat.to_string()


def _fabriquer(outil, user: User, registre: Registre, message_du_tour: str,
               tache: str, cache: dict | None = None, signaler=None,
               message_brut: str | None = None, tap: dict | None = None):
    """Rend la coroutine que PydanticAI appellera avec les arguments du modele.

    `cache` n'est plus lu: l'idempotence vit dans l'etat du tour, partage
    avec les choix executes par le code. Les regles de message lisent le
    message BRUT quand il est fourni, jamais le message enrichi du document.

    Parallelisme: pydantic-ai execute en parallele les appels d'outils batchés
    dans une meme etape, sauf si l'un est marque sequential. Les MUTATIONS
    gardent le verrou du tour sur toute leur execution (garde tester-puis-
    poser + cache d'idempotence + ecriture); les LECTURES s'executent sans
    lui, le registre etant desormais thread-safe pour l'ecriture.
    """
    ctx = _Contexte(user=user, registre=registre, tache=tache,
                    texte=message_brut if message_brut is not None else (message_du_tour or ""),
                    signaler=signaler, tap=tap)
    est_mutation = outil.name in OUTILS_DE_MUTATION

    def _appel_ferme(kwargs):
        """L'ORM tourne dans un thread du pool d'asgiref, hors du cycle de
        requete qui ferme les connexions. On les ferme donc nous-memes des
        deux cotes, comme le fait database_sync_to_async de channels."""
        close_old_connections()
        try:
            if est_mutation:
                with ctx.etat.verrou:
                    return _executer_appel(ctx, outil, kwargs)
            return _executer_appel(ctx, outil, kwargs)
        finally:
            close_old_connections()

    # thread_sensitive=FALSE, et c'est mesure, pas theorique. Sous ASGI, Django
    # execute la vue synchrone dans un thread de son pool; on y demarre une
    # boucle asyncio pour PydanticAI. Avec thread_sensitive=True, asgiref veut
    # rejouer l'ORM dans CE thread, lequel est bloque a attendre la boucle:
    # interblocage, observe en production le 2026-08-27.
    executer_sync = sync_to_async(_appel_ferme, thread_sensitive=False)

    async def executer(**kwargs) -> str:
        return await executer_sync(kwargs)

    executer.__name__ = outil.name
    return executer


# poser_question et present_choices forcent eux aussi le batch en sequentiel
# (pydantic-ai): deux questions du modele dans le meme batch s'executeraient
# en parallele et passeraient toutes les deux la garde « une seule question
# par tour » avant que la premiere soit consignee. En sequentiel, la seconde
# voit la demande de la premiere dans le registre et est refusee avec une
# consigne. Ce ne sont pas des mutations (pas de verrou du tour, pas de
# traitement « mutation » dans le rendu): seulement l'ordre d'execution.
# Les outils toujours exposes au modele. Tires de la mesure du 2026-10-01 sur
# 600 tours reels: ceux qui servent, plus les trois outils de question (le code
# relaie leurs puces) et le remplacement d'occurrence, trop recent pour
# apparaitre dans un echantillon retrospectif. Les autres passent derriere
# chercher_outils, qui rend leur schema a la demande.
OUTILS_EXPOSES = frozenset({
    # lectures du quotidien
    "get_today_schedule", "get_week_schedule", "find_free_slots",
    "list_tasks", "list_blocks",
    # ecritures du quotidien
    "create_block", "update_block", "schedule_task_at",
    "skip_block_occurrence", "replace_block_occurrence",
    "cancel_scheduled_block", "create_task", "complete_task",
    # questions posees par le modele, relayees par le code
    "present_form", "present_choices", "poser_question",
})
NOM_CHERCHEUR = "chercher_outils"
NOM_APPEL = "appeler_outil"

SCHEMA_CHERCHEUR = {
    "type": "object",
    "properties": {
        "besoin": {
            "type": "string",
            "description": ("Ce que tu cherches a faire, en quelques mots "
                            "(ex: « supprimer une tache », « envoyer une "
                            "notification », « reorganiser la journee »)."),
        },
    },
    "required": ["besoin"],
}
DESCRIPTION_CHERCHEUR = (
    "Cherche un outil que tu n'as pas sous la main. Tes outils courants "
    "(lire le planning, creer, deplacer, liberer, planifier, cocher, poser "
    "une question) sont deja la: ne passe PAS par ici pour eux. Pour tout le "
    "reste (supprimer, restaurer, vider, reorganiser, optimiser, objectifs, "
    "preferences, notification, faisabilite, statistiques), appelle cet outil "
    "avec ton besoin: il rend les outils qui correspondent et leurs "
    "parametres, que tu lances ensuite avec appeler_outil."
)

SCHEMA_APPEL = {
    "type": "object",
    "properties": {
        "nom": {
            "type": "string",
            "description": "Le nom exact rendu par chercher_outils.",
        },
        "parametres": {
            "type": "object",
            "description": ("Les parametres de cet outil, en objet "
                            "(ex: {\"task_id\": 12})."),
        },
    },
    "required": ["nom"],
}
DESCRIPTION_APPEL = (
    "Lance un outil trouve par chercher_outils, avec ses parametres. "
    "Reserve a ces outils-la: tes outils courants s'appellent directement, "
    "par leur nom. Le code applique les memes gardes que pour un appel "
    "direct: une suppression demande toujours sa confirmation."
)

OUTILS_QUESTION = frozenset({"poser_question", "present_choices"})


# Descriptions V2 (doctrine 2026-09-29): le modele choisit l'outil par sa
# description, jamais par des declencheurs ecrits dans le prompt. Ces
# surcharges ne s'appliquent qu'a la boucle V2: v1 garde les descriptions
# d'origine pour les utilisateurs non migres. Les regles d'usage qui vivaient
# dans REGLES_AGIR demenagent ici, sans vocabulaire declencheur.
DESCRIPTIONS_V2 = {
    # Remplacement complet, par USAGE et non par date (defaut mesure en prod
    # le 2026-09-29): interdire cet outil pour aujourd'hui, alors que la prose
    # n'a pas le droit de recopier le contexte, privait « montre-moi ma
    # journee » de tout chemin et servait un repli « je n'ai pas compris ».
    "get_today_schedule": (
        "Lit le planning EFFECTIF d'UN jour (blocs recurrents aux heures "
        "PLACEES, occurrences annulees exclues, taches planifiees, creneaux "
        "libres; parametre date AAAA-MM-JJ, defaut = aujourd'hui). "
        "APPELLE-LE des que la personne DEMANDE a voir une journee, "
        "aujourd'hui comprise: le systeme affiche la liste au-dessus de ta "
        "reponse et tu ne la recris jamais. N'appelle cette lecture QUE si la "
        "personne DEMANDE a voir. Un message qui n'exprime aucune demande n'en "
        "est pas une, quels que soient ses mots: aucune lecture, une phrase et "
        "rien d'autre. "
        "Ce qu'un tour precedent a lu ne se relit pas parce qu'il l'a lu. "
        "Pour seulement RAISONNER sur aujourd'hui "
        "(placer quelque chose, verifier une heure), le PLANNING AUJOURD'HUI "
        "de ton prompt suffit: inutile de le relire. Pour un AUTRE jour, "
        "consulte-le avant d'affirmer ou se trouve une activite ou si elle a "
        "bouge, et parle des heures effectives, jamais de memoire ni d'apres "
        "l'historique (un bloc souple peut etre place a une autre heure que "
        "son heure habituelle)."
    ),
}

COMPLEMENTS_V2 = {
    "skip_block_occurrence": (
        "Si quelque chose PREND LA PLACE de l'occurrence ce jour-la (examen a la "
        "place du cours, reunion a la place du quart), n'utilise pas cet outil: "
        "replace_block_occurrence libere le creneau ET place le remplacant en un "
        "seul geste."
    ),
    "schedule_task_at": (
        "Si l'evenement PREND LA PLACE de l'occurrence d'un bloc recurrent ce "
        "jour-la, n'utilise pas cet outil seul: replace_block_occurrence libere le "
        "creneau ET place l'evenement en un seul geste, dans le bon ordre."
    ),
    "optimize_week": (
        "apply=true seulement apres confirmation explicite de l'utilisateur: "
        "propose toujours d'abord avec apply=false."
    ),
    "create_block": (
        "Avant de creer, verifie la SEMAINE TYPE deja dans ton prompt: si le "
        "bloc y figure deja, modifie-le (update_block) au lieu de le recreer. "
        "Un ajout avec jours et heures, y compris en reponse a ta propre "
        "question: si ces jours et heures ne sont pas deja ceux d'un cours de "
        "la SEMAINE TYPE, c'est un nouveau cours: create_block dans ce tour, "
        "avec le nom que l'utilisateur a dit. Ne demande ni lequel, ni de "
        "confirmer la recurrence: un cours revient chaque semaine par defaut."
    ),
    "update_block": (
        "Pour borner une serie dans le temps (debut/fin de session), passe "
        "start_date/end_date, jamais en supprimant et recreant le bloc."
    ),
}


def description_v2(outil) -> str:
    """La description effective d'un outil dans la boucle V2."""
    if outil.name in DESCRIPTIONS_V2:
        return DESCRIPTIONS_V2[outil.name]
    complement = COMPLEMENTS_V2.get(outil.name)
    if complement:
        return f"{outil.description} {complement}"
    return outil.description


def outils_pour(user: User, registre: Registre, message_du_tour: str = "",
                tache: str = "", signaler=None, message_brut: str | None = None,
                tap: dict | None = None) -> list[Tool]:
    """Les outils de v1, prets pour PydanticAI, branches sur ce registre.

    `tache` identifie le tour: il entre dans la cle d'idempotence pour que
    deux tours distincts puissent legitimement refaire la meme action, alors
    qu'un meme tour rejoue ne l'execute qu'une fois.

    `message_brut` est ce que l'utilisateur a TAPE. Toute regle qui lit le
    message (portee d'une suppression, confirmation, heures dites, « cette
    semaine ») le lit, jamais le message enrichi du document ou de l'import.
    """
    return [_outil_pydantic(outil, user, registre, message_du_tour, tache,
                            signaler, message_brut, tap)
            for outil in ALL_TOOLS]


def _outil_pydantic(outil, user, registre, message_du_tour, tache,
                    signaler, message_brut, tap) -> Tool:
    return Tool.from_schema(
        _fabriquer(outil, user, registre, message_du_tour, tache, None,
                   signaler, message_brut, tap),
        outil.name,
        description_v2(outil),
        outil.parameters,
        # Une mutation dans le batch force TOUT le batch en sequentiel
        # (pydantic-ai): les lectures pures, elles, partent en parallele.
        # Les outils de question aussi (voir OUTILS_QUESTION): deux
        # questions du modele ne doivent jamais s'executer en parallele.
        sequential=(outil.name in OUTILS_DE_MUTATION
                    or outil.name in OUTILS_QUESTION),
    )


def outils_pour_le_modele(user: User, registre: Registre, message_du_tour: str = "",
                          tache: str = "", signaler=None,
                          message_brut: str | None = None,
                          tap: dict | None = None) -> list[Tool]:
    """Ce que la BOUCLE voit: les outils exposes, plus le chargeur.

    Mesure du 2026-10-01 sur 600 tours: 75 % des tours n'appellent aucun outil
    et les 33 outils pesaient 10 400 jetons a chaque tour. Les 17 rares passent
    derriere `chercher_outils` et `appeler_outil`, qui rendent leur schema a la
    demande.

    `outils_pour` garde son contrat (TOUS les outils) pour les tests et pour
    les executions par le code.
    """
    caches = [o for o in ALL_TOOLS if o.name not in OUTILS_EXPOSES]

    async def chercher_outils(besoin: str = "") -> str:
        return chargeur.chercher(besoin, caches, description_v2)

    async def appeler_outil(nom: str = "", parametres=None) -> str:
        """Lance un outil cache par le MEME chemin que les appels directs."""
        vise = str(nom or "").strip()
        outil = TOOL_MAP.get(vise)
        if outil is None:
            connus = ", ".join(sorted(o.name for o in caches))
            return (f"Outil inconnu: {vise!r}. Cherche-le d'abord avec "
                    f"{NOM_CHERCHEUR}. Disponibles ici: {connus}.")
        if vise in OUTILS_EXPOSES:
            return (f"{vise} est deja dans tes outils: appelle-le directement, "
                    f"pas par {NOM_APPEL}.")
        lus, erreur = chargeur.lire_parametres(parametres)
        if erreur:
            return f"Refuse par le code: {erreur}"
        manque = chargeur.requis_manquants(outil.parameters, lus)
        if manque:
            return (f"Refuse par le code: {vise} exige "
                    f"{', '.join(manque)}. Relance avec ces parametres.")
        # Le MEME executeur que pour un appel direct: verrou du tour, gardes,
        # registre, idempotence. Aucun chemin parallele.
        executer = _fabriquer(outil, user, registre, message_du_tour, tache,
                              None, signaler, message_brut, tap)
        return await executer(**lus)

    exposes = [_outil_pydantic(o, user, registre, message_du_tour, tache,
                               signaler, message_brut, tap)
               for o in ALL_TOOLS if o.name in OUTILS_EXPOSES]
    return exposes + [
        Tool.from_schema(chercher_outils, NOM_CHERCHEUR, DESCRIPTION_CHERCHEUR,
                         SCHEMA_CHERCHEUR),
        # Sequentiel: il peut muter, et deux mutations en parallele
        # contourneraient le verrou du tour.
        Tool.from_schema(appeler_outil, NOM_APPEL, DESCRIPTION_APPEL,
                         SCHEMA_APPEL, sequential=True),
    ]


# ------------------------------------------------- choix executes par le code

_POOL_CHOIX = ThreadPoolExecutor(max_workers=2, thread_name_prefix="choix")


def _sans_boucle(fabrique):
    """Execute la coroutine hors de toute boucle deja en cours dans ce thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(fabrique())
    return _POOL_CHOIX.submit(lambda: asyncio.run(fabrique())).result()


# Les seules lectures que le code s'autorise a executer lui-meme (filet de
# consultation, agent.py). Le code ne MUTE jamais a la place du modele.
LECTURES_DU_CODE = frozenset({"get_today_schedule", "get_week_schedule",
                              "list_tasks"})


def executer_lecture_par_le_code(user: User, registre: Registre, outil: str,
                                 tache: str = "", message: str = "",
                                 brut: str = "", **kwargs):
    """Execute une LECTURE au nom du code et l'inscrit au registre.

    Toute ecriture faite hors de la boucle d'outils doit entrer au registre,
    sinon la phase de rendu la nie; une lecture suit la meme regle, c'est elle
    qui fera la liste affichee.
    """
    if outil not in LECTURES_DU_CODE:
        raise ValueError(f"lecture refusee au code: {outil}")
    par_nom = {t.name: t for t in outils_pour(
        user, registre, message_du_tour=message, tache=tache, message_brut=brut)}
    tool = par_nom.get(outil)
    if tool is None:
        return None
    _sans_boucle(lambda: tool.function_schema.function(**kwargs))
    return registre.actions[-1] if registre.actions else None


def _effet_valide(demande: dict, option: str, effet: dict) -> bool:
    """L'effet stocke doit viser EXACTEMENT la cible de la cle. Une demande
    alteree ne peut pas faire executer autre chose que ce qu'elle nomme."""
    outil = effet.get("outil")
    parametres = effet.get("parametres")
    if outil not in TOOL_MAP or not isinstance(parametres, dict):
        return False
    motif, cle = demande.get("motif"), demande.get("cle")
    if motif == "portee_jour":
        bid = _bloc_de_demande(demande)
        jour = (demande.get("cible") or {}).get("date")
        if bid is None or not jour or _cle_portee(bid, jour) != cle:
            return False
        if option == "occurrence":
            return outil == "skip_block_occurrence" and parametres.get("date") == jour
        if option == "serie":
            return outil == "delete_block" and _entier(parametres.get("block_id")) == bid
        return False
    if motif == "portee_changement":
        bid = _bloc_de_demande(demande)
        jour = (demande.get("cible") or {}).get("date")
        if bid is None or not jour:
            return False
        if _cle_changement(bid, demande.get("parametres") or {}) != cle:
            return False
        if option == "occurrence":
            return (outil == "replace_block_occurrence"
                    and parametres.get("date") == jour)
        if option == "serie":
            if outil != "update_block" or _entier(parametres.get("block_id")) != bid:
                return False
            # Aucun parametre clandestin: l'effet ne porte que ce que la
            # demande stockait, aux memes valeurs.
            stockes = demande.get("parametres") or {}
            for cle_param, valeur in parametres.items():
                if cle_param == "block_id":
                    continue
                if cle_param not in CHAMPS_CHANGEABLES:
                    return False
                if str(stockes.get(cle_param) or "") != str(valeur or ""):
                    return False
            return True
        return False
    if motif == "destructif":
        return (option == "confirmer" and outil in DESTRUCTIFS | {"update_block"}
                and _cle_destructive(outil, parametres) == cle)
    if motif == "optimisation":
        return option == "confirmer" and outil == "optimize_week" and cle == "optimize_week:apply"
    return False


def _sujet(demande: dict) -> str:
    cible = demande.get("cible") or {}
    titre = _ascii(cible.get("titre") or "")
    motif = demande.get("motif")
    if motif == "portee_jour":
        jour = cible.get("jour")
        nom_jour = _NOMS_JOURS[jour] if isinstance(jour, int) and 0 <= jour <= 6 else ""
        return f"portee de la suppression de {titre} ({nom_jour} {cible.get('date') or ''})".replace("  ", " ")
    if motif == "portee_changement":
        jour = cible.get("jour")
        nom_jour = _NOMS_JOURS[jour] if isinstance(jour, int) and 0 <= jour <= 6 else ""
        return (f"portee du changement de {titre} ({nom_jour} "
                f"{cible.get('date') or ''})").replace("  ", " ")
    if motif == "creation_en_masse":
        return "ajouts en serie"
    if motif == "optimisation":
        return "application du plan de la semaine"
    return f"{demande.get('outil') or ''} {titre}".strip()


def _resume_sans_effet(demande: dict, option: str) -> str:
    motif = demande.get("motif")
    cible = demande.get("cible") or {}
    titre = _ascii(cible.get("titre") or "")
    choisie = next((o for o in demande.get("options") or []
                    if isinstance(o, dict) and o.get("id") == option), {}) or {}
    cible_option = choisie.get("cible") or {}
    if motif == "creation_en_masse":
        if option == "confirmer":
            return "CONFIRME: les ajouts en serie peuvent continuer ce tour"
        return "REFUSE PAR L'UTILISATEUR: ajouts en serie, n'ajoute rien de plus de ce lot"
    if motif == "optimisation" and option == "annuler":
        return ("REFUSE PAR L'UTILISATEUR: optimize_week apply, montre la proposition "
                "sans l'appliquer (apply=false)")
    if motif == "heure_refusee":
        if option.startswith("creneau_"):
            jour = cible_option.get("jour", cible.get("jour"))
            if (cible_option.get("recurrent") or cible.get("recurrent")) \
                    and isinstance(jour, int) and 0 <= jour <= 6:
                return (f"CHOISI PAR L'UTILISATEUR: {titre} les {_NOMS_JOURS[jour]}s "
                        f"(bloc recurrent) de {cible_option.get('debut')} a {cible_option.get('fin')}")
            return (f"CHOISI PAR L'UTILISATEUR: {titre} le {cible_option.get('date')} "
                    f"de {cible_option.get('debut')} a {cible_option.get('fin')}")
        return f"CHOISI PAR L'UTILISATEUR: un autre jour pour {titre}"
    if motif == "chevauchement":
        if option == "autre_heure":
            return f"CHOISI PAR L'UTILISATEUR: une autre heure pour {titre}"
        return f"REFUSE PAR L'UTILISATEUR: {titre}, laisse faire"
    if motif in ("choix_modele", "question_libre"):
        return f"CHOISI PAR L'UTILISATEUR: {_ascii(choisie.get('valeur') or choisie.get('libelle') or '')}"
    if option == "annuler":
        return f"REFUSE PAR L'UTILISATEUR: {_sujet(demande)}, n'y touche pas"
    return f"CHOISI PAR L'UTILISATEUR: {_sujet(demande)} ({option})"


def _consigner_decision(ctx: _Contexte, demande: dict, resultat: ToolResult):
    """Une decision du code sans outil execute (annulee, abandonnee). Consignee
    sous un nom qui n'est pas une mutation: rendu.py ne la raconte jamais comme
    un fait, et la voix la rend depuis decision_code."""
    return _consigner(ctx, OUTIL_DECISION, {"cle": demande.get("cle")}, resultat)


def _appliquer(ctx: _Contexte) -> list[dict]:
    sorties: list[dict] = []
    attente = _attente(ctx)
    etat = ctx.etat
    abandonnees = etat.attente.setdefault("abandonnees", set())
    options = {id(d): dem.option_choisie(ctx.texte, d, tap=ctx.tap) for d in attente}
    # D2: un message qui ne repond a aucune demande porte une nouvelle
    # requete. Le juge tranche par demande: « nouvelle_requete » partout (et
    # aucune option choisie) abandonne; un doute (« incertain ») repose la
    # question au lieu d'abandonner. Cette lecture choisit seulement entre
    # reposer et abandonner; elle n'autorise jamais rien.
    classes = {id(d): dem.classification_reponse(ctx.texte, d) for d in attente}
    nouvelle = bool(attente) and not any(options.values()) and all(
        c == "nouvelle_requete" for c in classes.values())
    decisions: dict = {}
    codes: dict = {}
    for demande in attente:
        cle, motif = demande.get("cle"), demande.get("motif")
        option = options[id(demande)]
        if option is None:
            if motif not in MOTIFS_GARDES:
                decisions[cle] = None
                continue
            if not nouvelle and int(demande.get("reemissions") or 0) < REEMISSIONS_MAX:
                # Banc du 2026-09-14 (s05-3): apres un oui vague, la demande
                # se perdait. Le CODE repose la meme demande, UNE fois, avec
                # sa date d'origine: la fenetre de 30 minutes doit expirer.
                reposee = ToolResult(
                    success=False,
                    data={"demande": _reposer(demande), "reposee_par_le_code": True,
                          "decision_code": DECISION_REPOSEE,
                          **({"needs_confirmation": True} if motif != "creation_en_masse" else {})},
                    message=MESSAGE_RETENUE)
                action = _consigner(ctx, str(demande.get("outil") or ""),
                                    dict(demande.get("parametres") or {}), reposee)
                decisions[cle] = codes[cle] = DECISION_REPOSEE
                sorties.append({"cle": cle, "motif": motif, "option": None, "action_id": None,
                                "resume": (f"SANS REPONSE CLAIRE: {_sujet(demande)}, n'agis pas "
                                           f"({action.id}: le code repose la question)")})
                continue
            # D2: deja reposee une fois, ou nouvelle requete. La demande est
            # abandonnee et consignee pour que la voix le dise en une ligne;
            # elle ne revient jamais sur un tour sans rapport.
            abandon = ToolResult(
                success=False,
                data={"demande": {k: v for k, v in demande.items() if k != "chips"},
                      "abandonnee_par_le_code": True, "decision_code": DECISION_ABANDONNEE},
                message=MESSAGE_ABANDON)
            _consigner_decision(ctx, demande, abandon)
            abandonnees.add(cle)
            decisions[cle] = codes[cle] = DECISION_ABANDONNEE
            sorties.append({"cle": cle, "motif": motif, "option": None, "action_id": None,
                            "resume": (f"QUESTION LAISSEE DE COTE: {_sujet(demande)}, "
                                       "l'utilisateur est passe a autre chose; n'agis pas "
                                       "sur ce point sans nouvelle demande explicite")})
            continue
        choisie = next((o for o in demande.get("options") or []
                        if isinstance(o, dict) and o.get("id") == option), None) or {}
        effet = choisie.get("effet")
        if not effet:
            if option == "annuler":
                _consigner_decision(ctx, demande, ToolResult(
                    success=True,
                    data={"decision_code": DECISION_ANNULEE, "cle_demande": cle, "motif": motif,
                          "cible": dict(demande.get("cible") or {}), "par_le_code": True},
                    message=MESSAGE_ANNULEE))
                codes[cle] = DECISION_ANNULEE
                # « Montre d'abord » (optimisation) demande encore a AGIR de
                # montrer la proposition: le tour n'est pas decide par le code.
                decisions[cle] = DECISION_ANNULEE if motif != "optimisation" else None
            else:
                # Un creneau, un jour, la suite des ajouts: AGIR doit agir.
                decisions[cle] = None
            sorties.append({"cle": cle, "motif": motif, "option": option, "action_id": None,
                            "resume": _resume_sans_effet(demande, option)})
            continue
        if not isinstance(effet, dict) or not _effet_valide(demande, option, effet):
            logger.warning("Effet de demande rejete cle=%s option=%s", cle, option)
            decisions[cle] = None
            sorties.append({"cle": cle, "motif": motif, "option": option, "action_id": None,
                            "resume": f"SANS REPONSE CLAIRE: {_sujet(demande)}, n'agis pas"})
            continue

        if _cible_changee(ctx.user, demande):
            # K1: la cible a change depuis la question. Rien ne s'execute,
            # la demande tombe et la voix le dit en une ligne.
            abandon = _abandon_cible_changee(ctx, demande)
            decisions[cle] = codes[cle] = DECISION_ABANDONNEE
            sorties.append({"cle": cle, "motif": motif, "option": option, "action_id": None,
                            "resume": (f"CIBLE CHANGEE ({abandon.id}): {_sujet(demande)} a change "
                                       "depuis la question, le code n'a rien execute; n'agis pas "
                                       "sur ce point sans nouvelle demande explicite")})
            continue

        outil = TOOL_MAP[effet["outil"]]
        decisions[cle] = codes[cle] = DECISION_EXECUTE
        retour = _executer_appel(ctx, outil, dict(effet["parametres"]), choix=demande)
        action = next((a for a in reversed(ctx.registre.actions)
                       if a.outil == outil.name and a.donnees.get("cle_demande") == cle), None)
        action_id = action.id if action is not None else None
        titre = _ascii((demande.get("cible") or {}).get("titre") or "")
        sujet = f"{outil.name} {titre}".strip()
        if retour == DEJA_FAIT:
            resume = f"DEJA FAIT PAR LE CODE ({action_id}): {sujet}, ne le refais pas"
        elif action is not None and action.succes:
            resume = f"FAIT PAR LE CODE ({action_id}): {sujet}, ne le refais pas"
        elif action is not None and action.donnees.get("demande"):
            resume = ("PLAN CHANGE: la proposition de la semaine a change depuis la question, "
                      "une nouvelle confirmation est demandee, n'applique rien")
        else:
            resume = f"ECHEC DU CODE ({action_id}): {sujet} n'a pas abouti, n'insiste pas"
        sorties.append({"cle": cle, "motif": motif, "option": option,
                        "action_id": action_id, "resume": resume})
    etat.attente["decide"] = {"texte": ctx.texte, "decisions": decisions, "nouvelle": nouvelle}
    for sortie in sorties:
        code = codes.get(sortie["cle"])
        if code:
            sortie["decision_code"] = code
    return sorties


def tour_entierement_decide_par_le_code(registre: Registre, message) -> bool:
    """D6: le message ne fait que repondre (ou ne pas repondre) aux demandes
    en attente, et le code a tout tranche. AGIR peut alors etre saute.

    Vrai seulement si appliquer_choix_en_attente a tourne sur CE registre et
    CE message, que chaque demande en attente a recu une decision du code
    (execute, reposee, abandonnee, annulee) et que le message ne porte aucune
    nouvelle requete. Une puce qui demande une suite au modele (creneau,
    jour, ajouts confirmes, « Montre d'abord ») rend Faux.
    """
    with _ETATS_VERROU:
        try:
            etat = _ETATS.get(registre)
        except TypeError:
            return False
    info = etat.attente.get("decide") if etat is not None else None
    if not isinstance(info, dict) or info.get("texte") != (message or ""):
        return False
    decisions = info.get("decisions") or {}
    if not decisions or info.get("nouvelle"):
        return False
    return all(v in DECISIONS_DU_CODE for v in decisions.values())


def appliquer_choix_en_attente(user, registre: Registre, message_brut: str, tache: str,
                               signaler=None, tap: dict | None = None) -> list[dict]:
    """Execute par le code l'option que l'utilisateur vient de choisir.

    Appele avant AGIR: une puce touchee (« Tous les jeudis ») supprime la
    serie sans attendre que le modele le refasse, par le MEME chemin que ses
    appels (registre, signal, idempotence). Le modele ne peut pas s'attribuer
    ces actions: ce sont des actions du registre, rendues par le code comme
    toute mutation. Rend un resume par demande en attente, destine au modele.
    """
    ctx = _Contexte(user=user, registre=registre, tache=tache,
                    texte=message_brut or "", signaler=signaler, tap=tap)

    def _ferme():
        close_old_connections()
        try:
            with ctx.etat.verrou:
                return _appliquer(ctx)
        finally:
            close_old_connections()

    try:
        return _sans_boucle(lambda: sync_to_async(_ferme, thread_sensitive=False)())
    except Exception:  # noqa: BLE001 - un choix non applique se redemande
        logger.error("Choix en attente non appliques", exc_info=True)
        return []


def schema_expose(tool: Tool) -> dict:
    """Le schema JSON tel que le modele le verra, pour le test de parite."""
    return tool.function_schema.json_schema
