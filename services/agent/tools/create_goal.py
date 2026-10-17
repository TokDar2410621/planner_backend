"""Outil create_goal."""
from typing import Literal

from django.contrib.auth.models import User
from pydantic import BaseModel, ConfigDict, Field

from core.models import Goal
from .base import ToolResult
from .commun import (
    GOAL_TITLE_MAX_LENGTH,
    VALID_GOAL_TYPES,
    _goal_to_dict,
    _parse_goal_deadline,
)

NAME = "create_goal"
DESCRIPTION = "Crée un nouvel objectif pour l'utilisateur."

# Source de verite : les choix du modele. Le Literal ci-dessous est le
# miroir type du schema ; l'assertion garde les deux synchronises.
TypeObjectif = Literal["short_term", "long_term"]
assert set(TypeObjectif.__args__) == VALID_GOAL_TYPES, (
    "derive entre le schema type et Goal.GOAL_TYPE_CHOICES"
)


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(
        description="Titre de l'objectif",
        max_length=GOAL_TITLE_MAX_LENGTH,
    )
    goal_type: TypeObjectif = Field(
        description="Type d'objectif",
    )
    description: str | None = Field(
        default=None,
        description="Description détaillée (optionnel)",
    )
    deadline: str | None = Field(
        default=None,
        description="Date limite au format YYYY-MM-DD (optionnel)",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    # NB : le type et la longueur du titre sont valides par le harnais
    # contre le schema (Literal, max_length) avant execute ; plus de
    # validate_choice/validate_max_length ici.
    deadline, derr = _parse_goal_deadline(kwargs.get("deadline"))
    if derr:
        return ToolResult(success=False, data={}, message=derr,
                          public="Je n'ai pas créé l'objectif : la date limite est invalide. Utilise le format AAAA-MM-JJ.")

    goal = Goal.objects.create(
        user=user,
        title=kwargs["title"],
        goal_type=kwargs["goal_type"],
        description=kwargs.get("description", ""),
        deadline=deadline,
    )
    type_label = "court terme" if goal.goal_type == "short_term" else "long terme"
    return ToolResult(
        success=True,
        data={"goal": _goal_to_dict(goal)},
        message=f"Objectif '{goal.title}' créé ({type_label}).",
    )
