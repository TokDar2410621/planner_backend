"""Outil get_preferences."""
from django.contrib.auth.models import User
from pydantic import BaseModel, ConfigDict

from core.models import UserProfile
from .base import ISSUE_TERMINEE, ToolResult

NAME = "get_preferences"
DESCRIPTION = (
    "Récupère toutes les préférences de planification de l'utilisateur "
    "(heures de sommeil, productivité, transport, etc.)."
)


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    profile = user.profile

    prefs = {
        "min_sleep_hours": profile.min_sleep_hours,
        "peak_productivity_time": profile.peak_productivity_time,
        "transport_time_minutes": profile.transport_time_minutes,
        "max_deep_work_hours_per_day": profile.max_deep_work_hours_per_day,
        "onboarding_completed": profile.onboarding_completed,
    }

    # Contrat d'issue (audit 2026-10-13): l'outil ecrit son texte
    # affichable; le rendu le relaie au lieu de se taire.
    pic = {"morning": "le matin", "afternoon": "l'après-midi",
           "evening": "le soir"}.get(profile.peak_productivity_time,
                                     profile.peak_productivity_time)
    return ToolResult(
        success=True,
        data=prefs,
        message="Préférences récupérées.",
        issue=ISSUE_TERMINEE, cause="",
        public=(f"Tes préférences : sommeil min. {profile.min_sleep_hours} h, "
                f"pic de productivité {pic}, "
                f"transport {profile.transport_time_minutes} min, "
                f"travail concentré max "
                f"{profile.max_deep_work_hours_per_day} h par jour."),
    )
