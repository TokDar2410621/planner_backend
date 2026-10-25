"""Question unique du modele. L'outil ne peut executer aucune action."""
from uuid import uuid4
from typing import Literal

from django.contrib.auth.models import User
from pydantic import BaseModel, ConfigDict, Field

from . import commun, poser_question, present_choices, present_form
from .base import ToolResult

NAME = "ask_question_custom"
DESCRIPTION = (
    "Pose une seule question avant d'agir. choice: 2 a 4 choix exhaustifs; "
    "source blocs/taches/jours/creneaux pour des options du planning verifiees. "
    "Si les choix ne sont pas exhaustifs, utilise form avec un champ radio "
    "allow_other=true. text: reponse libre. form: plusieurs champs manquants. "
    "Ne tronque jamais des possibilites: pose une question de tri si plus "
    "de quatre choix restent. confirmation: appelle l'outil d'action vise; "
    "le backend lie alors la confirmation a sa cible et la valide."
)


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["choice", "text", "form", "confirmation"]
    question: str
    options: list[poser_question.OptionLibre] = Field(default_factory=list)
    source: Literal["creneaux", "blocs", "taches", "jours"] | None = None
    date: str | None = None
    fields: list[present_form.ChampFormulaire] = Field(default_factory=list)


PARAMETERS = commun.inline_refs(Parametres.model_json_schema())


def execute(user: User, **kwargs) -> ToolResult:
    from services.agent_v2.mesure import fuite_question

    kind = kwargs.get("kind")
    question = poser_question._texte_court(kwargs.get("question"), 140)
    if not question or not question.endswith("?"):
        return ToolResult(False, message="Question non posee: 140 caracteres maximum, terminee par ?")
    if fuite_question(question):
        return ToolResult(False, message="Question non posee: elle affirme un effet non confirme.")
    if kind == "confirmation":
        return ToolResult(False, message=(
            "Confirmation non posee: appelle l'outil d'action vise. Son garde "
            "construit une demande liee a l'action et a la cible verifiee."))
    if kind == "choice":
        if kwargs.get("fields"):
            return ToolResult(False, message="Choice ne prend aucun champ de formulaire.")
        options = kwargs.get("options")
        if not isinstance(options, list) or not 2 <= len(options) <= 4:
            return ToolResult(False, message=(
                "Il faut 2 a 4 choix exhaustifs. S'il y en a davantage, "
                "pose une question de tri; si la liste peut etre incomplete, "
                "utilise form avec radio et Autre."))
        source = kwargs.get("source")
        if source:
            result = present_choices.execute(user, question=question, options=options,
                                             source=source, date=kwargs.get("date"))
        else:
            result = poser_question.execute(user, question=question, options=options)
        if result.success:
            result.data["question_custom"] = {
                "id": str(uuid4()), "kind": "choice", "question": question,
            }
        return result
    if kind in ("text", "form"):
        if kwargs.get("options") or kwargs.get("source"):
            return ToolResult(False, message="Utilise des champs pour text ou form, pas des options globales.")
        if kind == "text":
            if kwargs.get("fields"):
                return ToolResult(False, message="Text ne prend pas de champs supplementaires.")
            fields = [{"id": "reponse", "type": "text", "label": "Réponse",
                       "question": question}]
        else:
            fields = kwargs.get("fields")
            if not isinstance(fields, list) or not 1 <= len(fields) <= 6:
                return ToolResult(False, message="Le formulaire demande de 1 a 6 champs.")
            ids = [field.get("id") for field in fields if isinstance(field, dict)]
            if (len(ids) != len(fields) or any(not isinstance(i, str) or not i for i in ids)
                    or len(set(ids)) != len(fields)):
                return ToolResult(False, message="Chaque champ doit avoir un id distinct.")
            if any(field.get("type") in ("radio", "select")
                   and len(field.get("options") or []) > 4 for field in fields):
                return ToolResult(False, message="Un choix unique accepte au plus quatre options. Pose une question de tri.")
        result = present_form.execute(user, inputs=fields)
        if result.success:
            if any(fuite_question(field.get("question", "")) for field in result.data["interactive_inputs"]):
                return ToolResult(False, message="Question non posee: un champ affirme un effet non confirme.")
            result.data["question_custom"] = {
                "id": str(uuid4()), "kind": kind, "question": question,
                "fields": result.data["interactive_inputs"],
            }
        return result
    return ToolResult(False, message="Type de question inconnu.")
