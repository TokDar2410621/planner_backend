"""Outil update_preferences."""
from typing import Literal

from django.contrib.auth.models import User
from pydantic import BaseModel, ConfigDict, Field

from core.models import UserProfile
from .base import ToolResult

NAME = "update_preferences"
DESCRIPTION = "Met à jour les préférences de planification de l'utilisateur."

# Source de verite : les choix du modele. Le Literal ci-dessous est le
# miroir type du schema ; l'assertion garde les deux synchronises.
VALID_PRODUCTIVITY_TIMES = {c[0] for c in UserProfile.PRODUCTIVITY_CHOICES}
PicProductivite = Literal["morning", "afternoon", "evening"]
assert set(PicProductivite.__args__) == VALID_PRODUCTIVITY_TIMES, (
    "derive entre le schema type et UserProfile.PRODUCTIVITY_CHOICES"
)


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    min_sleep_hours: int | None = Field(
        default=None, ge=5, le=12,
        description="Heures de sommeil minimum (5-12)",
    )
    peak_productivity_time: PicProductivite | None = Field(
        default=None,
        description="Moment de pic de productivité",
    )
    transport_time_minutes: int | None = Field(
        default=None, ge=0, le=180,
        description="Temps de transport en minutes",
    )
    max_deep_work_hours_per_day: int | None = Field(
        default=None, ge=1, le=12,
        description="Heures max de travail concentré par jour",
    )
    onboarding_completed: bool | None = Field(
        default=None,
        description="Marquer l'onboarding comme terminé",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    profile = user.profile
    updated = []

    # NB : le choix du pic est valide par le harnais contre le schema
    # (Literal) avant execute ; plus de validate_choice ici.
    for field in ["min_sleep_hours", "peak_productivity_time", "transport_time_minutes",
                   "max_deep_work_hours_per_day", "onboarding_completed"]:
        if field in kwargs and kwargs[field] is not None:
            setattr(profile, field, kwargs[field])
            updated.append(field)

    if updated:
        profile.save()

    return ToolResult(
        success=True,
        data={"updated_fields": updated},
        message=f"Préférences mises à jour: {', '.join(updated)}" if updated else "Aucune modification.",
    )
