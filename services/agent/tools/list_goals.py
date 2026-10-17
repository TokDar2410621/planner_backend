"""Outil list_goals."""
from typing import Literal

from django.contrib.auth.models import User
from pydantic import BaseModel, ConfigDict, Field

from core.models import Goal
from .base import ISSUE_TERMINEE, ToolResult
from .commun import VALID_GOAL_STATUSES, _goal_to_dict

NAME = "list_goals"
DESCRIPTION = "Liste les objectifs de l'utilisateur. Peut filtrer par statut (active, completed, paused)."

# Source de verite : les choix du modele. Le Literal ci-dessous est le
# miroir type du schema ; l'assertion garde les deux synchronises.
StatutObjectif = Literal["active", "completed", "paused", "cancelled"]
assert set(StatutObjectif.__args__) == VALID_GOAL_STATUSES, (
    "derive entre le schema type et Goal.STATUS_CHOICES"
)


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: StatutObjectif | None = Field(
        default=None,
        description="Filtrer par statut",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    goals = Goal.objects.filter(user=user)

    status = kwargs.get("status")
    if status:
        goals = goals.filter(status=status)

    goal_list = [_goal_to_dict(g) for g in goals]

    # Contrat d'issue (audit 2026-10-13): texte affichable ecrit par
    # l'outil, avec les titres reels (donnees deja lues, rien d'invente).
    if goal_list:
        lignes = "\n".join(f"- {g.get('title', '').strip()}"
                           for g in goal_list if g.get("title"))
        public = f"Tes objectifs :\n{lignes}" if lignes else \
            f"{len(goal_list)} objectif(s)."
    else:
        public = ("Tu n'as pas encore d'objectif"
                  + (" avec ce statut." if status else "."))
    return ToolResult(
        success=True,
        data={"goals": goal_list, "count": len(goal_list)},
        message=f"{len(goal_list)} objectif(s) trouvé(s).",
        issue=ISSUE_TERMINEE, cause="", public=public,
    )
