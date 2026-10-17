"""Outil update_goal."""
from typing import Literal

from django.contrib.auth.models import User
from pydantic import BaseModel, ConfigDict, Field

from core.models import Goal
from .base import ToolResult
from .commun import (
    GOAL_TITLE_MAX_LENGTH,
    VALID_GOAL_STATUSES,
    _goal_to_dict,
    _parse_goal_deadline,
)

NAME = "update_goal"
DESCRIPTION = "Met à jour un objectif existant (progression, statut, titre, etc.)."

# Source de verite : les choix du modele. Le Literal ci-dessous est le
# miroir type du schema ; l'assertion garde les deux synchronises.
StatutObjectif = Literal["active", "completed", "paused", "cancelled"]
assert set(StatutObjectif.__args__) == VALID_GOAL_STATUSES, (
    "derive entre le schema type et Goal.STATUS_CHOICES"
)


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    goal_id: int = Field(
        description="ID de l'objectif à modifier",
    )
    title: str | None = Field(
        default=None,
        description="Nouveau titre",
        max_length=GOAL_TITLE_MAX_LENGTH,
    )
    description: str | None = Field(
        default=None,
    )
    progress: int | None = Field(
        default=None,
        description="Progression (0-100)",
        ge=0,
        le=100,
    )
    status: StatutObjectif | None = Field(
        default=None,
    )
    deadline: str | None = Field(
        default=None,
        description="Nouvelle deadline (YYYY-MM-DD)",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    goal_id = kwargs["goal_id"]
    try:
        goal = Goal.objects.get(id=goal_id, user=user)
    except Goal.DoesNotExist:
        return ToolResult(success=False, data={}, message=f"Objectif #{goal_id} introuvable.",
                          public="Je n'ai pas modifié l'objectif : je ne l'ai pas trouvé. Dis-moi lequel.")

    # NB : le statut et la longueur du titre sont valides par le harnais
    # contre le schema (Literal, max_length) avant execute ; plus de
    # validate_choice/validate_max_length ici.
    if kwargs.get("deadline") is not None:
        deadline, derr = _parse_goal_deadline(kwargs["deadline"])
        if derr:
            return ToolResult(success=False, data={}, message=derr,
                              public="Je n'ai pas modifié l'objectif : la date limite est invalide. Utilise le format AAAA-MM-JJ.")
        goal.deadline = deadline

    for field in ["title", "description", "progress", "status"]:
        if field in kwargs and kwargs[field] is not None:
            setattr(goal, field, kwargs[field])

    # Auto-complete if progress reaches 100
    if goal.progress >= 100 and goal.status == "active":
        goal.status = "completed"

    goal.save()
    return ToolResult(
        success=True,
        data={"goal": _goal_to_dict(goal)},
        message=f"Objectif '{goal.title}' mis à jour (progression: {goal.progres_effectif()}%).",
    )
