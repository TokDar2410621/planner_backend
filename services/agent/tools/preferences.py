"""
User preferences tools for the Planner AI agent.
"""
from django.contrib.auth.models import User

from core.models import UserProfile
from .base import BaseTool, ToolResult, validate_choice, ISSUE_TERMINEE

# Enforced at the tool layer (not just in the JSON schema) before save().
VALID_PRODUCTIVITY_TIMES = {c[0] for c in UserProfile.PRODUCTIVITY_CHOICES}


class GetPreferencesTool(BaseTool):
    name = "get_preferences"
    description = "Récupère toutes les préférences de planification de l'utilisateur (heures de sommeil, productivité, transport, etc.)."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
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


class UpdatePreferencesTool(BaseTool):
    name = "update_preferences"
    description = "Met à jour les préférences de planification de l'utilisateur."
    parameters = {
        "type": "object",
        "properties": {
            "min_sleep_hours": {
                "type": "integer",
                "description": "Heures de sommeil minimum (5-12)",
                "minimum": 5,
                "maximum": 12,
            },
            "peak_productivity_time": {
                "type": "string",
                "description": "Moment de pic de productivité",
                "enum": ["morning", "afternoon", "evening"],
            },
            "transport_time_minutes": {
                "type": "integer",
                "description": "Temps de transport en minutes",
                "minimum": 0,
                "maximum": 180,
            },
            "max_deep_work_hours_per_day": {
                "type": "integer",
                "description": "Heures max de travail concentré par jour",
                "minimum": 1,
                "maximum": 12,
            },
            "onboarding_completed": {
                "type": "boolean",
                "description": "Marquer l'onboarding comme terminé",
            },
        },
        "required": [],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        profile = user.profile
        updated = []

        # Validation choices AVANT save() (le schéma JSON n'est qu'indicatif).
        err = validate_choice(
            kwargs.get("peak_productivity_time"),
            VALID_PRODUCTIVITY_TIMES,
            "peak_productivity_time",
        )
        if err:
            return ToolResult(success=False, data={}, message=err)

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
