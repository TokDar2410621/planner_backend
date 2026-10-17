"""Outil update_task."""
from typing import Literal

from django.contrib.auth.models import User
from pydantic import Field

from core.models import Task
from . import commun
from .base import ToolResult

NAME = "update_task"
DESCRIPTION = "Modifie une tâche existante. Résous la tâche par son nom via list_tasks; ne demande jamais un identifiant à l'utilisateur."

# Source de verite : les choix du modele (via commun). Le Literal
# ci-dessous est le miroir type du schema ; l'assertion garde les
# deux synchronises (miroir du pattern PicProductivite de
# update_preferences.py).
TypeTache = Literal["deep_work", "shallow", "errand"]
assert set(TypeTache.__args__) == commun.VALID_TASK_TYPES, (
    "derive entre le schema type et Task.TASK_TYPE_CHOICES"
)


class Parametres(commun.ChampsLieu):
    task_id: int = Field(
        description="ID de la tâche à modifier",
    )
    title: str | None = Field(
        default=None, max_length=commun.TASK_TITLE_MAX_LENGTH,
        description="Nouveau titre",
    )
    priority: int | None = Field(
        default=None, ge=1, le=10,
    )
    deadline: str | None = Field(
        default=None,
        description="Nouvelle deadline (YYYY-MM-DD)",
    )
    description: str | None = Field(
        default=None,
    )
    task_type: TypeTache | None = Field(
        default=None,
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    task_id = kwargs["task_id"]
    try:
        task = Task.objects.get(id=task_id, user=user)
    except Task.DoesNotExist:
        return ToolResult(success=False, data={}, message=f"Tâche #{task_id} introuvable.",
                          public="Je n'ai pas modifié la tâche : je ne l'ai pas trouvée. Dis-moi laquelle.")

    # NB : le type et la longueur du titre sont valides par le harnais
    # contre le schema (Literal / max_length) avant execute ; plus de
    # validate_choice/validate_max_length ici.

    for field in ["title", "priority", "description", "task_type"]:
        if field in kwargs and kwargs[field] is not None:
            setattr(task, field, kwargs[field])
    if kwargs.get("deadline") is not None:
        task.deadline = commun._parse_deadline(kwargs["deadline"])

    if kwargs.get("place_name") or kwargs.get("place_address"):
        place = commun._resolve_place(
            user,
            place_name=kwargs.get("place_name"),
            place_address=kwargs.get("place_address"),
            travel_minutes=kwargs.get("travel_minutes"),
        )
        if place is not None:
            task.place = place

    task.save()
    return ToolResult(
        success=True,
        data={"task": commun._task_to_dict(task)},
        message=f"Tâche '{task.title}' mise à jour.",
    )
