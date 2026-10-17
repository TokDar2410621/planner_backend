"""Outil present_choices."""
import re
from datetime import date
from typing import Literal

from django.contrib.auth.models import User
from django.utils import timezone
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import ToolResult
from . import commun
from .commun import _fold, _iso_date

SOURCES_CHOIX = ("creneaux", "blocs", "taches", "jours")
WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
MAX_QUESTION_CHOIX = 140
MAX_LIBELLE_CHOIX = 40
MAX_VALEUR_CHOIX = 200
MAX_OPTIONS_CHOIX = 4
MIN_OPTIONS_CHOIX = 2
DUREE_CRENEAU_PAR_DEFAUT = 30
MOIS_COURTS_PLATS = ("janv", "fevr", "mars", "avr", "mai", "juin",
                     "juil", "aout", "sept", "oct", "nov", "dec")
_APOSTROPHES_CHOIX = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
_HEURE_CHOIX = re.compile(r"(?<!\d)(\d{1,2})\s*(?:h|:)\s*(\d{2})?(?!\d)|\b(midi|minuit)\b")
_JOUR_CHOIX = re.compile(
    r"\b(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|demain|apres-demain"
    r"|aujourd'hui|aujourdhui)\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}\s+(?:%s)" % "|".join(MOIS_COURTS_PLATS)
    + r"|\b\d{1,2}/\d{1,2}\b")

# Source de verite : SOURCES_CHOIX. Le Literal ci-dessous est le miroir type
# du schema ; l'assertion garde les deux synchronises.
SourceChoix = Literal["creneaux", "blocs", "taches", "jours"]
assert set(SourceChoix.__args__) == set(SOURCES_CHOIX), \
    "derive entre le schema type et SOURCES_CHOIX"


def _maintenant():
    """L'instant mural, isole pour que les tests le figent sans toucher au
    module timezone de Django."""
    return timezone.localtime()


def _plat(texte) -> str:
    """Minuscules, accents retires, apostrophes droites, espaces reduits."""
    return " ".join(_fold(str(texte).translate(_APOSTROPHES_CHOIX)).split())


def _texte_court(valeur, maximum: int):
    if not isinstance(valeur, str):
        return None
    propre = " ".join(valeur.split())
    if not propre or len(propre) > maximum:
        return None
    return propre


def _titre_contenu(libelle_plat: str, titres: dict):
    """Le titre reel (le plus long) que le libelle contient en mots entiers."""
    trouve = None
    for plat, original in titres.items():
        if not plat:
            continue
        if re.search(r"(?<!\w)" + re.escape(plat) + r"(?!\w)", libelle_plat):
            if trouve is None or len(plat) > len(_plat(trouve)):
                trouve = original
    return trouve


def _minutes_du_libelle(libelle: str):
    """(debut, fin) en minutes lus dans un libelle de creneau, fin par defaut
    debut + 30 min. None si le libelle ne porte pas d'heure valide."""
    heures = []
    for trouve in _HEURE_CHOIX.finditer(_plat(libelle)):
        if trouve.group(3):
            heures.append(720 if trouve.group(3) == "midi" else 0)
            continue
        h, m = int(trouve.group(1)), int(trouve.group(2) or 0)
        if h > 23 or m > 59:
            return None
        heures.append(h * 60 + m)
        if len(heures) == 2:
            break
    if not heures:
        return None
    debut = heures[0]
    if len(heures) == 1:
        return debut, debut + DUREE_CRENEAU_PAR_DEFAUT
    fin = heures[1]
    if fin == 0:
        fin = 24 * 60  # « 23 h a minuit »
    if fin <= debut:
        return None
    return debut, fin


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _demande_de_choix(question: str, source: str, options: list, parametres: dict,
                      cible: dict) -> dict:
    """La DEMANDE de motif choix_modele, meme forme que les demandes des gardes."""
    import hashlib

    return {
        "type": "choix",
        "motif": "choix_modele",
        "cle": "choix:" + hashlib.sha1(question.encode("utf-8")).hexdigest()[:12],
        "outil": "present_choices",
        "parametres": parametres,
        "cible": cible,
        "options": [
            {"id": f"o{rang}", "effet": None, "cible": option["cible"],
             "libelle": option["label"], "valeur": option["value"]}
            for rang, option in enumerate(options, start=1)
        ],
        "question": question,
        "source": source,
        "emise_le": timezone.now().isoformat(),
    }


def _refus(message: str, **data) -> ToolResult:
    return ToolResult(success=False, data=data, message=message)


def _ancrer(user, source: str, jour, candidates: list, rejetees: list) -> list:
    """Garde les options qui designent une vraie entite, avec leur cible."""
    if source in ("blocs", "taches"):
        titres = _titres(user, source)
        gardees = []
        for option in candidates:
            titre = _titre_contenu(_plat(option["label"]), titres)
            if titre is None:
                rejetees.append(option["label"])
                continue
            gardees.append({**option, "cible": {"titre": titre}})
        return gardees

    if source == "jours":
        gardees = []
        for option in candidates:
            trouve = _JOUR_CHOIX.search(_plat(option["label"]))
            if trouve is None:
                rejetees.append(option["label"])
                continue
            cible = {}
            mot = trouve.group(1)
            if mot in WEEKDAYS:
                cible["jour"] = WEEKDAYS.index(mot)
            elif _iso_date(trouve.group(0)):
                cible["date"] = trouve.group(0)
            gardees.append({**option, "cible": cible})
        return gardees

    # creneaux: chaque plage doit tenir ENTIERE dans un trou libre du jour,
    # et ne pas etre deja passee.
    from services.scheduling.placement import open_intervals

    maintenant = _maintenant()
    aujourdhui = maintenant.date()
    if jour < aujourdhui:
        rejetees.extend(option["label"] for option in candidates)
        return []
    plancher = maintenant.hour * 60 + maintenant.minute if jour == aujourdhui else 0
    libres = open_intervals(user, jour, 0, 24 * 60)
    gardees = []
    for option in candidates:
        plage = _minutes_du_libelle(option["label"])
        if plage is None:
            rejetees.append(option["label"])
            continue
        debut, fin = plage
        if debut < plancher or not any(s <= debut and fin <= e for s, e in libres):
            rejetees.append(option["label"])
            continue
        gardees.append({**option, "cible": {
            "date": jour.isoformat(), "debut": _hhmm(debut),
            "fin": _hhmm(fin % (24 * 60))}})
    return gardees


def _seul_element_reel(user, source: str, brutes) -> str | None:
    """Le titre du SEUL bloc ou tache reel que nomment les options, sinon None.

    Lit les libelles bruts: une option reelle ecartee parce que sa valeur
    raconte une action compte encore comme element reel.
    """
    if source not in ("blocs", "taches") or not isinstance(brutes, list):
        return None
    titres = _titres(user, source)
    reels = set()
    for option in brutes:
        if not isinstance(option, dict):
            continue
        label = _texte_court(option.get("label"), MAX_LIBELLE_CHOIX)
        if label is None:
            continue
        titre = _titre_contenu(_plat(label), titres)
        if titre is not None:
            reels.add(titre)
    return reels.pop() if len(reels) == 1 else None


def _titres(user, source: str) -> dict:
    if source == "taches":
        from core.models import Task
        noms = Task.objects.filter(user=user).values_list("title", flat=True)
    else:
        from core.models import RecurringBlock, ScheduledBlock
        noms = list(RecurringBlock.objects.filter(user=user, active=True)
                    .values_list("title", flat=True))
        noms += list(ScheduledBlock.objects.filter(user=user)
                     .values_list("task__title", flat=True))
    return {_plat(nom): nom for nom in noms if nom and _plat(nom)}


NAME = "present_choices"
DESCRIPTION = (
    "Pose UNE question courte avec 2 à 4 réponses en un tap, tirées du planning réel. "
    "Utilise-le quand la réponse est bornée: lequel de plusieurs blocs ou tâches existants "
    "(source blocs ou taches), quel créneau libre (source creneaux, avec date; lis d'abord "
    "find_free_slots), quel jour (source jours). Le code rejette toute option qui n'existe "
    "pas et tout ce qui affirme une action. N'affirme rien: la question se pose avant d'agir."
)


class OptionChoix(BaseModel):
    """Option brute. Le schema reste permissif : execute() ecarte les
    options mal formees (pas un objet, label/value manquants), les
    doublons et celles qui affirment une action."""
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _non_dict_vaut_vide(cls, v):
        return v if isinstance(v, dict) else {}

    label: str | None = Field(
        default=None,
        description="Texte du bouton, ex: '13 h à 14 h' ou 'Poterie débutant'",
    )
    value: str | None = Field(
        default=None,
        description="Message envoyé au tap, ex: 'Va pour 13 h à 14 h jeudi.'",
    )


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str = Field(
        description="La question, courte (140 caractères max), qui finit par « ? ». Ex: 'Laquelle de tes séances de poterie ?'",
    )
    options: list[OptionChoix] = Field(
        description="2 à 4 réponses. Chaque option a un label court (40 caractères max) et une value: la phrase complète envoyée au tap (200 caractères max).",
    )
    source: SourceChoix = Field(
        description="D'où viennent les options: creneaux (créneaux libres d'une date), blocs, taches, jours.",
    )
    date: str | None = Field(
        default=None,
        description="Date des créneaux au format YYYY-MM-DD. Obligatoire quand source vaut creneaux.",
    )


PARAMETERS = commun.inline_refs(Parametres.model_json_schema())


def execute(user: User, **kwargs) -> ToolResult:
    from services.agent_v2.mesure import fuite_question

    question = _texte_court(kwargs.get("question"), MAX_QUESTION_CHOIX)
    if question is None or not question.endswith("?"):
        return _refus(
            "Choix non présenté : la question doit être courte (140 caractères max) "
            "et finir par « ? ».")
    source = kwargs.get("source")
    if source not in SOURCES_CHOIX:
        return _refus(
            f"Choix non présenté : source inconnue '{source}'. "
            f"Sources valides : {', '.join(SOURCES_CHOIX)}.")
    if fuite_question(question):
        return _refus(
            "Choix non présenté : la question affirme une action. "
            "Pose-la sans rien raconter de ce qui a été fait.")

    jour = None
    brute = kwargs.get("date")
    if brute not in (None, ""):
        iso = _iso_date(brute)
        jour = date.fromisoformat(iso) if iso else None
    if source == "creneaux" and jour is None:
        return _refus(
            "Choix non présenté : une date YYYY-MM-DD est obligatoire pour des créneaux.")

    candidates, vus, rejetees = [], set(), []
    brutes = kwargs.get("options")
    for option in brutes if isinstance(brutes, list) else []:
        if not isinstance(option, dict):
            continue
        label = _texte_court(option.get("label"), MAX_LIBELLE_CHOIX)
        value = _texte_court(option.get("value"), MAX_VALEUR_CHOIX)
        if label is None or value is None:
            continue
        if _plat(label) in vus:
            continue
        if fuite_question(label) or fuite_question(value):
            rejetees.append(label)
            continue
        vus.add(_plat(label))
        candidates.append({"label": label, "value": value})

    ancrees = _ancrer(user, source, jour, candidates, rejetees)
    ancrees = ancrees[:MAX_OPTIONS_CHOIX]
    if len(ancrees) < MIN_OPTIONS_CHOIX:
        detail = f" Options écartées : {', '.join(rejetees)}." if rejetees else ""
        seul = _seul_element_reel(user, source, brutes)
        if seul is not None:
            # Banc r8, s03-2: un seul cours proche plus « Un autre cours »
            # inventé. « Pose plutôt une question courte » faisait
            # redemander au lieu de créer un cours dont l'utilisateur
            # venait de donner les jours et les heures.
            return _refus(
                f"Choix non présenté : un seul élément réel correspond (« {seul} »), "
                "il n'y a donc rien à choisir. Si l'utilisateur ajoute un cours ou une "
                "activité avec ses jours et ses heures, crée-le (create_block) sans "
                "demander lequel. Sinon, vise cet élément." + detail,
                options_ecartees=rejetees)
        return _refus(
            "Choix non présenté : il faut au moins 2 options réelles (créneaux libres, "
            "blocs, tâches ou jours existants). Pose plutôt une question courte." + detail,
            options_ecartees=rejetees)

    parametres = {
        "question": question,
        "options": [{"label": o["label"], "value": o["value"]} for o in ancrees],
        "source": source,
    }
    cible = {}
    if jour is not None:
        parametres["date"] = jour.isoformat()
        cible["date"] = jour.isoformat()
    demande = _demande_de_choix(question, source, ancrees, parametres, cible)
    return ToolResult(
        success=True,
        data={"demande": demande},
        message=(
            f"Choix présenté à l'utilisateur ({len(ancrees)} options). "
            "Attends sa réponse, ne pose pas d'autre question."
        ),
    )
