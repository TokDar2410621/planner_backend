"""Outil create_task."""
from typing import Literal

from django.contrib.auth.models import User
from pydantic import Field

from core.models import Task
from . import commun
from .base import ToolResult

NAME = "create_task"
DESCRIPTION = (
    "Crée une nouvelle tâche pour l'utilisateur. Si la tâche a lieu quelque "
    "part (rendez-vous, réunion, cours ponctuel), renseigne place_name (et "
    "place_address si connue) pour activer les rappels de départ."
)

# Source de verite : les choix du modele (via commun). Le Literal
# ci-dessous est le miroir type du schema ; l'assertion garde les
# deux synchronises (miroir du pattern PicProductivite de
# update_preferences.py).
TypeTache = Literal["deep_work", "shallow", "errand"]
assert set(TypeTache.__args__) == commun.VALID_TASK_TYPES, (
    "derive entre le schema type et Task.TASK_TYPE_CHOICES"
)


class Parametres(commun.ChampsLieu):
    title: str = Field(
        max_length=commun.TASK_TITLE_MAX_LENGTH,
        description="Titre de la tâche",
    )
    task_type: TypeTache | None = Field(
        default=None,
        description="Type de tâche (deep_work=travail concentré, shallow=tâche légère, errand=course/admin)",
    )
    priority: int | None = Field(
        default=None, ge=1, le=10,
        description="Priorité de 1 (basse) à 10 (haute). Défaut: 5",
    )
    description: str | None = Field(
        default=None,
        description="Description détaillée (optionnel)",
    )
    deadline: str | None = Field(
        default=None,
        description="Échéance YYYY-MM-DD ou YYYY-MM-DDTHH:MM (heure précise si dite)",
    )
    alert_at: str | None = Field(
        default=None, description="Alerte YYYY-MM-DDTHH:MM si elle doit précéder l'échéance",
    )
    estimated_duration_minutes: int | None = Field(
        default=None, ge=5,
        description="Durée estimée en minutes (optionnel)",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    task_type = kwargs.get("task_type", "shallow")
    title = kwargs["title"]

    # NB : le type et la longueur du titre sont valides par le harnais
    # contre le schema (Literal / max_length) avant execute ; plus de
    # validate_choice/validate_max_length ici.

    # Idempotency: guard against the model re-creating a task it already
    # mentioned earlier in the conversation. If an ACTIVE (non-completed)
    # task with the same title already exists, return it instead of
    # duplicating.
    place = commun._resolve_place(
        user,
        place_name=kwargs.get("place_name"),
        place_address=kwargs.get("place_address"),
        travel_minutes=kwargs.get("travel_minutes"),
    )

    deadline = commun._parse_deadline(kwargs.get('deadline'))
    if kwargs.get('deadline') and deadline is None:
        return ToolResult(False, message='Échéance invalide.', public='Précise le jour et l’heure de la tâche.')
    alert_at = commun._parse_deadline(kwargs.get('alert_at'))
    if kwargs.get('alert_at') and alert_at is None:
        return ToolResult(False, message='Alerte invalide.', public='Précise l’heure de l’alerte.')
    # Un jour seul n'autorise pas une alerte. Une heure explicite, oui.
    has_hour = bool(kwargs.get('deadline') and ('T' in kwargs['deadline'] or ' ' in kwargs['deadline']))
    if alert_at is None and has_hour:
        alert_at = deadline
    if alert_at and (not deadline or alert_at > deadline):
        return ToolResult(False, message='Alerte après échéance.',
                          public="L’alerte doit arriver au plus tard à l’heure de la tâche.")
    existing_qs = Task.objects.filter(user=user, completed=False, title__iexact=title.strip())
    if deadline is not None:
        existing_qs = existing_qs.filter(deadline=deadline)
    else:
        existing_qs = existing_qs.filter(deadline__isnull=True)
    existing = existing_qs.first()
    if existing is not None:
        # N'attache le lieu à la tâche existante que s'il n'en a pas encore un.
        if place is not None and existing.place_id is None:
            existing.place = place
            existing.save(update_fields=["place"])
        return ToolResult(
            success=True,
            data={"task": commun._task_to_dict(existing), "deja_presente": True},
            message=f"Tâche '{existing.title}' déjà présente (non dupliquée).",
        )

    task = Task.objects.create(
        user=user,
        title=kwargs["title"],
        task_type=kwargs.get("task_type", "shallow"),
        priority=kwargs.get("priority", 5),
        description=kwargs.get("description", ""),
        deadline=deadline,
        alert_at=alert_at,
        estimated_duration_minutes=kwargs.get("estimated_duration_minutes"),
        place=place,
    )
    loc = f" à {place.name}" if place else ""
    return ToolResult(
        success=True,
        data={"task": commun._task_to_dict(task), "deja_presente": False},
        message=f"Tâche '{task.title}'{loc} créée (priorité {task.priority}).",
    )
