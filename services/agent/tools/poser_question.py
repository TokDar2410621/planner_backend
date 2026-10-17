"""Outil poser_question."""
import re

from django.contrib.auth.models import User
from django.utils import timezone
from pydantic import BaseModel, ConfigDict, Field

from . import commun
from .base import ToolResult
from .commun import _fold

MAX_QUESTION_LIBRE = 140
MAX_LIBELLE_LIBRE = 40
MAX_VALEUR_LIBRE = 200
MAX_OPTIONS_LIBRE = 4
MIN_OPTIONS_LIBRE = 2

# Lexique local a l'agent v1: il detecte qu'une question du modele ressemble
# a une confirmation de suppression, pour que ce soit le code qui la pose.
# L'agent v2 n'utilise plus aucun lexique ecrit a la main pour lire
# l'intention (2026-09-29): il est parti dans la couche de jugement.
LEXIQUE_SUPPRESSION = (
    r"\b(supprim\w*|effac\w*|enlev\w*|retir\w*|annul\w*|vide[rz]?|debarrasse\w*)\b"
)

_APOSTROPHES_CHOIX = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})


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


def _refus(message: str, **data) -> ToolResult:
    return ToolResult(success=False, data=data, message=message)


def _confirmation_destructive(question: str, options: list) -> bool:
    """La question ressemble-t-elle a une confirmation de suppression ou
    d'annulation ? Une telle confirmation ne se pose jamais par le modele,
    c'est le code qui la pose quand le modele appelle l'outil vise."""
    destructif = re.compile(LEXIQUE_SUPPRESSION)
    textes = [question or ""]
    for option in options or []:
        if isinstance(option, dict):
            textes.append(option.get("label") or "")
            textes.append(option.get("value") or "")
    return any(destructif.search(t) for t in textes)


def _demande_question_libre(question: str, options: list) -> dict:
    """La DEMANDE de motif question_libre, meme forme que choix_modele.

    Les options ne portent AUCUN effet: un tap ne peut jamais executer un
    outil. La valeur revient comme message utilisateur et AGIR l'interprete
    au tour suivant. La cle derive de la question: la reposer a l'identique
    ne duplique pas la demande au rendu.
    """
    import hashlib

    return {
        "type": "choix",
        "motif": "question_libre",
        "cle": "question:" + hashlib.sha1(question.encode("utf-8")).hexdigest()[:12],
        "outil": "poser_question",
        "parametres": {
            "question": question,
            "options": [{"label": o["label"], "value": o["value"]} for o in options],
        },
        "cible": {},
        "options": [
            {"id": f"o{rang}", "effet": None,
             "libelle": option["label"], "valeur": option["value"]}
            for rang, option in enumerate(options, start=1)
        ],
        "question": question,
        "emise_le": timezone.now().isoformat(),
    }


NAME = "poser_question"
DESCRIPTION = (
    "Pose UNE question courte avec 2 a 4 reponses en un tap, a choix libre. "
    "Utilise-le quand la reponse est bornee mais ne porte sur aucun element "
    "du planning (present_choices est reserve aux creneaux libres, blocs, "
    "taches et jours existants). Exemples: une preference oui/non, un format "
    "de reponse, une clarification binaire. N'affirme jamais d'action dans "
    "la question ni dans les options: la question se pose avant d'agir. "
    "Une seule question par tour: attends la reponse avant d'en poser "
    "une autre. Ne t'en sers jamais pour une confirmation destructive: "
    "appelle l'outil vise, le code pose la question lui-meme."
)


class OptionLibre(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str
    value: str


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str = Field(
        description="La question, courte (140 caracteres max), qui finit par « ? ».",
    )
    options: list[OptionLibre] = Field(
        description=(
            "2 a 4 reponses. Chaque option a un label court "
            "(40 caracteres max, texte du bouton) et une value "
            "(200 caracteres max, phrase complete envoyee au tap)."
        ),
    )


PARAMETERS = commun.inline_refs(Parametres.model_json_schema())


def execute(user: User, **kwargs) -> ToolResult:
    from services.agent_v2.mesure import fuite_question

    question = _texte_court(kwargs.get("question"), MAX_QUESTION_LIBRE)
    if question is None or not question.endswith("?"):
        return _refus(
            "Question non posee : elle doit etre courte (140 caracteres max) "
            "et finir par « ? ».")
    if fuite_question(question):
        return _refus(
            "Question non posee : elle affirme une action. "
            "Pose-la sans rien raconter de ce qui a ete fait.")

    brutes = kwargs.get("options")
    candidates, vus, rejetees = [], set(), []
    for option in brutes if isinstance(brutes, list) else []:
        if not isinstance(option, dict):
            continue
        label = _texte_court(option.get("label"), MAX_LIBELLE_LIBRE)
        value = _texte_court(option.get("value"), MAX_VALEUR_LIBRE)
        if label is None or value is None:
            continue
        if _plat(label) in vus:
            continue
        if fuite_question(label) or fuite_question(value):
            rejetees.append(label)
            continue
        vus.add(_plat(label))
        candidates.append({"label": label, "value": value})
    candidates = candidates[:MAX_OPTIONS_LIBRE]
    if len(candidates) < MIN_OPTIONS_LIBRE:
        detail = f" Options ecartees : {', '.join(rejetees)}." if rejetees else ""
        return _refus(
            "Question non posee : il faut 2 a 4 options valides." + detail)
    if _confirmation_destructive(question, candidates):
        return _refus(
            "Question non posee : elle ressemble a une confirmation de "
            "suppression ou d'annulation. Si une action destructive est en "
            "jeu, appelle directement l'outil vise et laisse le code poser "
            "la question ; sinon reformule sans verbe de suppression.")

    demande = _demande_question_libre(question, candidates)
    return ToolResult(
        success=True,
        data={"demande": demande},
        message=(
            f"Question posee a l'utilisateur ({len(candidates)} options). "
            "Attends sa reponse, ne pose pas d'autre question."
        ),
    )
