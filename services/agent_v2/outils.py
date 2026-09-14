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
CREATEURS = {"create_block", "schedule_task_at", "create_task"}
# Au-dela de cinq creations dans un tour, on demande avant de continuer.
SEUIL_CREATIONS = 5

AUTORISANTES = {
    "destructif": {"confirmer"},
    "optimisation": {"confirmer"},
    "creation_en_masse": {"confirmer"},
    "portee_jour": {"delete_block": {"serie"}, "skip_block_occurrence": {"occurrence"}},
}
MOTIFS_GARDES = {"portee_jour", "destructif", "creation_en_masse", "optimisation"}

_TOUT = re.compile(r"\btou(t|s|te|tes)\b")
_UNE_SEULE_FOIS = re.compile(r"\b(seulement|juste|cette fois|demain|aujourd'?hui)\b")
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

    Le verrou serialise les appels: la garde de creations en masse compte
    puis ecrit, et deux appels paralleles du modele verraient sinon le meme
    compte. Les outils sont des ecritures ORM de quelques millisecondes.
    """
    verrou: threading.RLock = field(default_factory=threading.RLock)
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

    @property
    def etat(self) -> _EtatTour:
        return _etat_du_tour(self.registre)


def _attente(ctx: _Contexte) -> list[dict]:
    """Les demandes en attente, lues UNE fois par tour."""
    etat = ctx.etat
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
    }


# ---------------------------------------------------------------------- cles

def _cle_portee(block_id, jour_iso: str) -> str:
    return dem.cle_demande("portee_jour", {"block_id": int(block_id), "date": jour_iso})


def _cle_destructive(nom: str, params: dict):
    """Cle d'identite d'une action retenue, a partir de ses seuls arguments."""
    params = params or {}
    if nom == "delete_block":
        bid = _entier(params.get("block_id"))
        return None if bid is None else dem.cle_demande("delete_block", {"block_id": bid})
    if nom == "update_block":
        bid = _entier(params.get("block_id"))
        return None if bid is None else dem.cle_demande("update_block:fin", {"block_id": bid})
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


def _saut_suspect(texte: str) -> bool:
    """Un saut d'occurrence qui ressemble a une suppression large (« efface
    tout jeudi »). Le saut unique explicite (« saute mon gym demain ») passe."""
    plat = dem.sans_accents(texte)
    if not dem.suppression_demandee(texte) or not dem.jour_vise(texte):
        return False
    return bool(_TOUT.search(plat)) or not _UNE_SEULE_FOIS.search(plat)


def _analyser(ctx: _Contexte, nom: str, kwargs: dict):
    """La garde destructive d'un appel, ou None. Une garde inactive sert
    seulement a reconnaitre ce que le code a deja fait ce tour."""
    from core.models import RecurringBlock, ScheduledBlock, Task

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
        return _Garde("portee_jour", cle, {cle}, nom, dict(kwargs), cible,
                      _options_portee(block, jour, cible), actif=_saut_suspect(texte),
                      type="choix")

    if nom == "clear_all_blocks":
        nombre = RecurringBlock.objects.filter(user=user, active=True).count()
        cible = {"nombre": nombre}
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
        cible = {"titre": task.title}
        return _Garde("destructif", cle, {cle}, nom, {"task_id": tid}, cible,
                      _options_confirmer(nom, {"task_id": tid, "confirm": True}, cible))

    if nom == "cancel_scheduled_block":
        jour = _date_iso(kwargs.get("date"))
        if jour is None:
            return None
        titre = str(kwargs.get("title") or "").strip()
        cle = _cle_destructive(nom, kwargs)
        qs = ScheduledBlock.objects.filter(user=user, date=jour).select_related("task")
        if titre:
            qs = qs.filter(task__title__icontains=titre)
        blocs = list(qs.order_by("start_time"))
        distincts = {b.task.title if b.task_id else "" for b in blocs}
        if not blocs or (len(distincts) > 1 and not titre):
            # Rien a supprimer, ou l'outil va demander lequel: rien a retenir.
            return _Garde("destructif", cle, {cle}, nom, actif=False)
        premier = blocs[0]
        cible = {
            "titre": premier.task.title if premier.task_id else "",
            "date": jour.isoformat(),
            "debut": premier.start_time.strftime("%H:%M"),
            "fin": premier.end_time.strftime("%H:%M"),
        }
        parametres = {"date": jour.isoformat(), "title": titre} if titre else {"date": jour.isoformat()}
        return _Garde("destructif", cle, {cle}, nom, parametres, cible,
                      _options_confirmer(nom, parametres, cible))

    if nom == "update_block":
        # Revue du 2026-09-14: « supprime mon cours de chimie » + end_date=demain
        # terminait la serie sans question. Une fin (ou un depart repousse)
        # est destructive quand le message parle de supprimer, ou quand la
        # fin tombe dans les 7 prochains jours. « mon quart finit le 15
        # octobre » (fin lointaine, aucun verbe de suppression) passe.
        aujourdhui = timezone.localdate()
        suppression = dem.suppression_demandee(texte)
        fin = _date_iso(kwargs.get("end_date")) if str(kwargs.get("end_date") or "").strip() else None
        depart = _date_iso(kwargs.get("start_date")) if str(kwargs.get("start_date") or "").strip() else None
        fin_destructive = fin is not None and (suppression or fin <= aujourdhui + timedelta(days=7))
        depart_destructif = depart is not None and suppression and depart > aujourdhui
        if not (fin_destructive or depart_destructif):
            return None
        if fin is None:
            fin = depart
        bid = _entier(kwargs.get("block_id"))
        if bid is None:
            return None
        cle = dem.cle_demande("update_block:fin", {"block_id": bid})
        block = RecurringBlock.objects.filter(id=bid, user=user, active=True).first()
        if block is None:
            return _Garde("destructif", cle, {cle}, nom, actif=False)
        cible = _cible_bloc(block)
        cible["date"] = fin.isoformat()
        parametres = dict(kwargs)
        parametres["block_id"] = bid
        return _Garde("destructif", cle, {cle}, nom, parametres, cible,
                      _options_confirmer(nom, parametres, cible))

    if nom == "optimize_week" and kwargs.get("apply"):
        cle = "optimize_week:apply"
        return _Garde("optimisation", cle, {cle}, nom, dict(kwargs))
    return None


def _garde_critique(nom: str, kwargs: dict) -> bool:
    """Si la garde elle-meme tombe en panne, ces appels sont refuses."""
    return (
        nom in DESTRUCTIFS
        or nom == "skip_block_occurrence"
        or (nom == "update_block" and bool(str(kwargs.get("end_date") or "").strip()))
        or (nom == "update_block" and bool(str(kwargs.get("start_date") or "").strip()))
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
    for demande in _attente(ctx):
        if demande.get("cle") not in cles:
            continue
        option = dem.option_choisie(ctx.texte, demande)
        if _autorise(demande.get("motif"), nom, option):
            return True, demande, None
        if option is not None:
            repondue = repondue or demande
        elif en_suspens is None:
            en_suspens = demande
    return False, repondue, en_suspens


def _reposer(demande: dict) -> dict:
    """La meme question, remise a l'heure, sans les puces du tour passe."""
    copie = {k: v for k, v in demande.items() if k != "chips"}
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


def _plan_propose(outil, user, kwargs: dict):
    parametres = {k: v for k, v in kwargs.items() if k != "plan_hash"}
    parametres["apply"] = False
    proposition = outil.execute(user, **parametres)
    if not proposition.success:
        return proposition, None
    return proposition, _empreinte_plan(proposition.data or {})


def _demande_optimisation(kwargs: dict, empreinte: str, proposition: ToolResult) -> dict:
    parametres = {k: v for k, v in kwargs.items() if k != "plan_hash"}
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
    proposition, empreinte = _plan_propose(outil, ctx.user, kwargs)
    if empreinte is None:
        return None  # l'outil echouera de lui-meme, rien a proteger
    if autorise and (demande.get("parametres") or {}).get("plan_hash") == empreinte:
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
    mots = _mots_du_titre(titre)
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

    titre_situe = False
    for s, e in _propositions(plat):
        morceau = plat[s:e]
        mots_morceau = {m[:-1] if m.endswith("s") and len(m) > 4 else m
                        for m in re.findall(r"[a-z0-9]+", morceau)}
        if not mots & mots_morceau:
            continue
        titre_situe = True
        fermes = [(v, tol) for v, _ou, tol in _heures_fermes(plat, positions, s, e)]
        if not fermes:
            continue
        jour_cible = _jour_cible(dem._dates_nommees(morceau, aujourdhui) or dates_message)
        if jour_cible is None:
            continue
        return _heure_dite_contredite(ctx, nom, kwargs, prep, titre, recurrent, debut,
                                      jour_cible, fermes)
    if titre_situe:
        return None

    # Revue du 2026-09-14 (round 3): un titre court ou vide de sens (« Gym »,
    # « cours ») ou renomme par le modele (« Entraînement » pour « gym ») ne se
    # retrouve dans aucune proposition. Si le message ne donne qu'UNE heure
    # ferme, elle vaut pour tout ajout du jour qu'il nomme (ou sans jour nomme).
    fermes = _sans_fins_de_plage(plat, _heures_fermes(plat, positions, 0, len(plat)))
    if len({v for v, _ou, _tol in fermes}) != 1:
        return None
    jour_cible = _jour_cible(dates_message)
    if jour_cible is None:
        return None
    return _heure_dite_contredite(ctx, nom, kwargs, prep, titre, recurrent, debut, jour_cible,
                                  [(v, tol) for v, _ou, tol in fermes])


def _heures_fermes(plat: str, positions, s: int, e: int) -> list[tuple[str, int, int]]:
    """(valeur, position, tolerance) des heures fermes entre s et e: une
    borne (« avant 10 h ») n'en est pas une, une approximation a sa marge."""
    fermes = []
    for valeur, ou in positions:
        if not (s <= ou < e):
            continue
        avant = plat[:ou]
        if _HEURE_SOUPLE_AVANT.search(avant):
            continue
        tolerance = TOLERANCE_APPROX if _HEURE_APPROX_AVANT.search(avant) else 0
        fermes.append((valeur, ou, tolerance))
    return fermes


_ENTRE_PLAGE = re.compile(r"\s*(?:a|au|-)\s*$")


def _sans_fins_de_plage(plat: str, fermes):
    """« de 14 h a 15 h » ou « 10:30 - 11:30 » donnent UNE heure de debut."""
    fins = {m.start(): m.end() for m in dem._RE_HEURE.finditer(plat)}
    gardees, fin_precedente = [], None
    for valeur, ou, tolerance in sorted(fermes, key=lambda f: f[1]):
        if fin_precedente is not None and _ENTRE_PLAGE.match(plat[fin_precedente:ou]):
            fin_precedente = None
            continue
        gardees.append((valeur, ou, tolerance))
        fin_precedente = fins.get(ou)
    return gardees


def _heure_dite_contredite(ctx: _Contexte, nom: str, kwargs: dict, prep: _Preparation,
                           titre: str, recurrent: bool, debut: str, jour_cible: date, fermes):
    """Refus quand l'appel ne pose aucune des heures fermes dites, sinon None."""
    from services.scheduling.placement import open_intervals

    if any(abs(_minutes(debut) - _minutes(v)) <= tol for v, tol in fermes):
        return None
    dite = fermes[0][0]
    fin_appel = _heure_normale(kwargs.get("end_time")) or (prep.bloc_update or {}).get("fin")
    duree = ((_minutes(fin_appel) - _minutes(debut)) % (24 * 60)) if fin_appel else 60
    duree = duree or 60
    fin_dite = _fmt((_minutes(dite) + duree) % (24 * 60))
    debut_min = _minutes(dite)
    fin_min = debut_min + duree
    libre = fin_min <= 24 * 60 and any(
        a <= debut_min and fin_min <= b for a, b in open_intervals(ctx.user, jour_cible, 0, 24 * 60))
    if libre:
        return ToolResult(
            success=False, data={"heure_dite": dite},
            message=(f"Retenu par le code: l'utilisateur a dit {dite} pour {_ascii(titre)}. "
                     f"Refais l'appel avec start_time={dite}. Si cette heure ne convient pas, "
                     "pose la question au lieu de changer l'heure."))
    demande = _demande_heure_refusee(ctx, nom, kwargs, titre, jour_cible, dite, fin_dite,
                                     None, recurrent=recurrent)
    heures = [v for v, _tol in fermes]
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
    if dem.jour_vise(reste):
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
        elif a.outil == "schedule_task_at":
            sb = donnees.get("scheduled_block") or {}
            _noter(("evenement", sb.get("id") or a.id), sb.get("title"))
        elif a.outil == "create_task" and not donnees.get("deja_presente"):
            tache = donnees.get("task") or {}
            _noter(("tache", tache.get("id") or a.id), tache.get("title"))
    return len(vus), titres


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
        option = dem.option_choisie(ctx.texte, demande)
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

def _consigner(ctx: _Contexte, nom: str, kwargs: dict, resultat: ToolResult):
    action = ctx.registre.ajouter(nom, kwargs, resultat)
    if ctx.signaler:
        ctx.signaler(action)
    return action


def _pour_empreinte(nom: str, kwargs: dict) -> dict:
    """Arguments de la cle d'idempotence. `confirm` est force comme le fera
    la garde, pour qu'un rejeu retrouve l'appel autorise."""
    if nom in ("delete_task", "clear_all_blocks"):
        return {**kwargs, "confirm": True}
    return kwargs


def _executer_appel(ctx: _Contexte, outil, kwargs: dict, choix: dict | None = None) -> str:
    """Le chemin UNIQUE de tout appel d'outil du tour, qu'il vienne du modele
    ou d'un choix de l'utilisateur execute par le code (choix non nul).
    Synchrone: il tourne dans un thread d'executeur, verrou du tour tenu."""
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
            _consigner(ctx, nom, kwargs, deja)
            return deja.to_string()

    if choix is None:
        try:
            issue = None
            if garde is not None and garde.actif:
                if garde.motif == "optimisation":
                    issue = _garde_optimisation(ctx, outil, kwargs, garde)
                else:
                    autorise, repondue, en_suspens = _reponse(ctx, garde.cles, nom)
                    if autorise:
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
                issue = _garde_creations(ctx, nom, kwargs)
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
        try:
            proposition, empreinte = _plan_propose(outil, ctx.user, kwargs)
        except Exception as e:  # noqa: BLE001
            logger.error("Proposition de plan en panne: %s", e, exc_info=True)
            proposition, empreinte = ToolResult(success=False, data={}, message=f"Erreur de l'outil: {e}"), None
        if empreinte is None or empreinte != (choix.get("parametres") or {}).get("plan_hash"):
            donnees = {"cle_demande": choix.get("cle"), "par_le_code": True}
            if empreinte is not None:
                donnees["demande"] = _demande_optimisation(kwargs, empreinte, proposition)
                message = MESSAGE_RETENUE
            else:
                message = proposition.message or MESSAGE_RETENUE
            refus = ToolResult(success=False, data=donnees, message=message)
            _consigner(ctx, nom, kwargs, refus)
            return refus.to_string()

    # Meme marqueur que v1: le banc capte les appels d'outils par le
    # logger parent « services », et cette ligne est ce qu'il cherche.
    logger.info(f"Executing tool: {nom}({kwargs})")
    try:
        resultat = outil.execute(ctx.user, **kwargs)
    except Exception as e:  # noqa: BLE001
        # v1 degrade une exception d'outil en ToolResult d'echec. Sans
        # cela, l'exception avorterait le run ET le registre ne garderait
        # aucune trace de la mutation tentee.
        logger.error("Tool %s a leve: %s", nom, e, exc_info=True)
        resultat = ToolResult(success=False, data={}, message=f"Erreur de l'outil: {e}")

    if choix is not None:
        resultat = ToolResult(
            success=resultat.success,
            data={**(resultat.data or {}), "cle_demande": choix.get("cle"), "par_le_code": True},
            message=resultat.message,
        )
    else:
        try:
            resultat = _apres(ctx, nom, kwargs, resultat, prep)
        except Exception:  # noqa: BLE001 - une question en moins, jamais un tour tombe
            logger.error("Post-traitement de %s en panne", nom, exc_info=True)

    # On ne met en cache que les SUCCES: un echec peut etre transitoire
    # (429, timeout), et rejouer un echec empecherait toute reprise.
    if cle is not None and resultat.success:
        etat.cache[cle] = resultat
    # Diffuse au fil de l'execution: c'est ce qui meuble l'attente cote
    # interface.
    _consigner(ctx, nom, kwargs, resultat)

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
               message_brut: str | None = None):
    """Rend la coroutine que PydanticAI appellera avec les arguments du modele.

    `cache` n'est plus lu: l'idempotence vit dans l'etat du tour, partage
    avec les choix executes par le code. Les regles de message lisent le
    message BRUT quand il est fourni, jamais le message enrichi du document.
    """
    ctx = _Contexte(user=user, registre=registre, tache=tache,
                    texte=message_brut if message_brut is not None else (message_du_tour or ""),
                    signaler=signaler)

    def _appel_ferme(kwargs):
        """L'ORM tourne dans un thread du pool d'asgiref, hors du cycle de
        requete qui ferme les connexions. On les ferme donc nous-memes des
        deux cotes, comme le fait database_sync_to_async de channels."""
        close_old_connections()
        try:
            with ctx.etat.verrou:
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


def outils_pour(user: User, registre: Registre, message_du_tour: str = "",
                tache: str = "", signaler=None, message_brut: str | None = None) -> list[Tool]:
    """Les outils de v1, prets pour PydanticAI, branches sur ce registre.

    `tache` identifie le tour: il entre dans la cle d'idempotence pour que
    deux tours distincts puissent legitimement refaire la meme action, alors
    qu'un meme tour rejoue ne l'execute qu'une fois.

    `message_brut` est ce que l'utilisateur a TAPE. Toute regle qui lit le
    message (portee d'une suppression, confirmation, heures dites, « cette
    semaine ») le lit, jamais le message enrichi du document ou de l'import.
    """
    return [
        Tool.from_schema(
            _fabriquer(outil, user, registre, message_du_tour, tache, None,
                       signaler, message_brut),
            outil.name,
            outil.description,
            outil.parameters,
        )
        for outil in ALL_TOOLS
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
    if motif == "choix_modele":
        return f"CHOISI PAR L'UTILISATEUR: {_ascii(choisie.get('valeur') or choisie.get('libelle') or '')}"
    if option == "annuler":
        return f"REFUSE PAR L'UTILISATEUR: {_sujet(demande)}, n'y touche pas"
    return f"CHOISI PAR L'UTILISATEUR: {_sujet(demande)} ({option})"


def _appliquer(ctx: _Contexte) -> list[dict]:
    sorties: list[dict] = []
    for demande in _attente(ctx):
        cle, motif = demande.get("cle"), demande.get("motif")
        option = dem.option_choisie(ctx.texte, demande)
        if option is None:
            if motif in MOTIFS_GARDES:
                # Banc du 2026-09-14 (s05-3): apres un oui vague, la demande
                # se perdait et « Tous les jeudis » au tour suivant ne trouvait
                # plus rien a trancher. Le CODE repose la meme demande: elle
                # entre au registre, la question du tour la rend avec ses
                # puces et la persiste, et une reponse claire la tranchera.
                reposee = ToolResult(
                    success=False,
                    data={"demande": _reposer(demande), "reposee_par_le_code": True,
                          **({"needs_confirmation": True} if motif != "creation_en_masse" else {})},
                    message=MESSAGE_RETENUE)
                action = _consigner(ctx, str(demande.get("outil") or ""),
                                    dict(demande.get("parametres") or {}), reposee)
                sorties.append({"cle": cle, "motif": motif, "option": None, "action_id": None,
                                "resume": (f"SANS REPONSE CLAIRE: {_sujet(demande)}, n'agis pas "
                                           f"({action.id}: le code repose la question)")})
            continue
        choisie = next((o for o in demande.get("options") or []
                        if isinstance(o, dict) and o.get("id") == option), None) or {}
        effet = choisie.get("effet")
        if not effet:
            sorties.append({"cle": cle, "motif": motif, "option": option, "action_id": None,
                            "resume": _resume_sans_effet(demande, option)})
            continue
        if not isinstance(effet, dict) or not _effet_valide(demande, option, effet):
            logger.warning("Effet de demande rejete cle=%s option=%s", cle, option)
            sorties.append({"cle": cle, "motif": motif, "option": option, "action_id": None,
                            "resume": f"SANS REPONSE CLAIRE: {_sujet(demande)}, n'agis pas"})
            continue

        outil = TOOL_MAP[effet["outil"]]
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
    return sorties


def appliquer_choix_en_attente(user, registre: Registre, message_brut: str, tache: str,
                               signaler=None) -> list[dict]:
    """Execute par le code l'option que l'utilisateur vient de choisir.

    Appele avant AGIR: une puce touchee (« Tous les jeudis ») supprime la
    serie sans attendre que le modele le refasse, par le MEME chemin que ses
    appels (registre, signal, idempotence). Le modele ne peut pas s'attribuer
    ces actions: ce sont des actions du registre, rendues par le code comme
    toute mutation. Rend un resume par demande en attente, destine au modele.
    """
    ctx = _Contexte(user=user, registre=registre, tache=tache,
                    texte=message_brut or "", signaler=signaler)

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
