"""Outil complete_task."""
from django.contrib.auth.models import User
from django.utils import timezone
from pydantic import BaseModel, ConfigDict, Field

from core.models import Task
from . import commun
from .base import ToolResult

NAME = "complete_task"
DESCRIPTION = "Marque une tâche comme terminée. Résous la tâche par son nom via list_tasks; ne demande jamais un identifiant à l'utilisateur."


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_id: int = Field(
        description="ID de la tâche à compléter",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    task_id = kwargs["task_id"]
    try:
        task = Task.objects.get(id=task_id, user=user)
    except Task.DoesNotExist:
        return ToolResult(success=False, data={}, message=f"Tâche #{task_id} introuvable.",
                          public="Je n'ai pas marqué la tâche comme terminée : je ne l'ai pas trouvée. Dis-moi laquelle.")

    if hasattr(task, "mark_completed"):
        task.mark_completed()
    else:
        task.completed = True
        task.completed_at = timezone.now()
        task.save()
    return ToolResult(
        success=True,
        data={"task": commun._task_to_dict(task)},
        message=f"Tâche '{task.title}' marquée comme terminée.",
    )
