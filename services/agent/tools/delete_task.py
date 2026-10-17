"""Outil delete_task."""
from django.contrib.auth.models import User
from pydantic import BaseModel, ConfigDict, Field

from core.models import Task
from .base import ToolResult

NAME = "delete_task"
DESCRIPTION = (
    "Supprime définitivement une tâche (hard delete, cascade sur les blocs "
    "planifiés associés). Opération destructive et IRRÉVERSIBLE: elle DOIT "
    "être confirmée par l'utilisateur hors-bande (dialogue de confirmation "
    "côté frontend), jamais sur la seule initiative du modèle. Passe "
    "confirm=true uniquement après une confirmation explicite de "
    "l'utilisateur. Pour marquer une tâche faite, préfère complete_task."
)

# Flag expose a la couche orchestration/frontend : cet appel doit etre
# verrouille par une confirmation humaine explicite hors-bande, pas par
# un booleen fourni par le LLM (ex requires_confirmation de la classe).
REQUIRES_CONFIRMATION = True


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_id: int = Field(
        description="ID de la tâche à supprimer",
    )
    confirm: bool = Field(
        description=(
            "Doit être true. Ne mets true que si l'utilisateur a "
            "explicitement confirmé vouloir supprimer cette tâche."
        ),
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    task_id = kwargs["task_id"]

    # Le modèle Task n'a pas de champ de soft-delete: la suppression est
    # irréversible. On exige donc une confirmation explicite avant de
    # détruire quoi que ce soit (S10).
    if not kwargs.get("confirm"):
        return ToolResult(
            success=False,
            data={"requires_confirmation": True},
            message=(
                "Suppression annulée: confirm doit être true et la "
                "suppression doit être confirmée explicitement par "
                "l'utilisateur (action irréversible)."
            ),
        )

    try:
        task = Task.objects.get(id=task_id, user=user)
    except Task.DoesNotExist:
        return ToolResult(success=False, data={}, message=f"Tâche #{task_id} introuvable.",
                          public="Je n'ai pas supprimé la tâche : je ne l'ai pas trouvée. Dis-moi laquelle.")

    title = task.title
    task.delete()
    return ToolResult(
        success=True,
        data={"deleted_id": task_id, "title": title},
        message=f"Tâche '{title}' supprimée.",
    )
