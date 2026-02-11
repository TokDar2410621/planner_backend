"""
Task tools for the Planner AI agent.
"""
from django.contrib.auth.models import User
from django.utils import timezone

from core.models import Task
from .base import BaseTool, ToolResult


def _task_to_dict(task: Task) -> dict:
    return {
        "id": task.id,
        "title": task.title,
        "description": task.description or None,
        "task_type": task.task_type,
        "priority": task.priority,
        "deadline": task.deadline.isoformat() if task.deadline else None,
        "estimated_duration_minutes": task.estimated_duration_minutes,
        "completed": task.completed,
    }


class ListTasksTool(BaseTool):
    name = "list_tasks"
    description = "Liste les tâches de l'utilisateur. Peut filtrer par statut (complétée ou non), type, ou limite."
    parameters = {
        "type": "object",
        "properties": {
            "completed": {
                "type": "boolean",
                "description": "Filtrer par statut (true=terminées, false=en cours)",
            },
            "task_type": {
                "type": "string",
                "description": "Filtrer par type",
                "enum": ["deep_work", "shallow", "errand"],
            },
            "limit": {
                "type": "integer",
                "description": "Nombre max de tâches à retourner (défaut: 20)",
                "minimum": 1,
                "maximum": 50,
            },
        },
        "required": [],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        tasks = Task.objects.filter(user=user)

        completed = kwargs.get("completed")
        if completed is not None:
            tasks = tasks.filter(completed=completed)

        task_type = kwargs.get("task_type")
        if task_type:
            tasks = tasks.filter(task_type=task_type)

        limit = kwargs.get("limit", 20)
        tasks = tasks[:limit]
        task_list = [_task_to_dict(t) for t in tasks]

        return ToolResult(
            success=True,
            data={"tasks": task_list, "count": len(task_list)},
            message=f"{len(task_list)} tâche(s) trouvée(s).",
        )


class CreateTaskTool(BaseTool):
    name = "create_task"
    description = "Crée une nouvelle tâche pour l'utilisateur."
    parameters = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Titre de la tâche",
            },
            "task_type": {
                "type": "string",
                "description": "Type de tâche (deep_work=travail concentré, shallow=tâche légère, errand=course/admin)",
                "enum": ["deep_work", "shallow", "errand"],
            },
            "priority": {
                "type": "integer",
                "description": "Priorité de 1 (basse) à 10 (haute). Défaut: 5",
                "minimum": 1,
                "maximum": 10,
            },
            "description": {
                "type": "string",
                "description": "Description détaillée (optionnel)",
            },
            "deadline": {
                "type": "string",
                "description": "Date limite au format YYYY-MM-DD (optionnel)",
            },
            "estimated_duration_minutes": {
                "type": "integer",
                "description": "Durée estimée en minutes (optionnel)",
                "minimum": 5,
            },
        },
        "required": ["title"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        task = Task.objects.create(
            user=user,
            title=kwargs["title"],
            task_type=kwargs.get("task_type", "shallow"),
            priority=kwargs.get("priority", 5),
            description=kwargs.get("description", ""),
            deadline=kwargs.get("deadline"),
            estimated_duration_minutes=kwargs.get("estimated_duration_minutes"),
        )
        return ToolResult(
            success=True,
            data={"task": _task_to_dict(task)},
            message=f"Tâche '{task.title}' créée (priorité {task.priority}).",
        )


class UpdateTaskTool(BaseTool):
    name = "update_task"
    description = "Modifie une tâche existante."
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {"type": "integer", "description": "ID de la tâche à modifier"},
            "title": {"type": "string", "description": "Nouveau titre"},
            "priority": {"type": "integer", "minimum": 1, "maximum": 10},
            "deadline": {"type": "string", "description": "Nouvelle deadline (YYYY-MM-DD)"},
            "description": {"type": "string"},
            "task_type": {"type": "string", "enum": ["deep_work", "shallow", "errand"]},
        },
        "required": ["task_id"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        task_id = kwargs["task_id"]
        try:
            task = Task.objects.get(id=task_id, user=user)
        except Task.DoesNotExist:
            return ToolResult(success=False, data={}, message=f"Tâche #{task_id} introuvable.")

        for field in ["title", "priority", "deadline", "description", "task_type"]:
            if field in kwargs and kwargs[field] is not None:
                setattr(task, field, kwargs[field])

        task.save()
        return ToolResult(
            success=True,
            data={"task": _task_to_dict(task)},
            message=f"Tâche '{task.title}' mise à jour.",
        )


class DeleteTaskTool(BaseTool):
    name = "delete_task"
    description = "Supprime une tâche."
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {"type": "integer", "description": "ID de la tâche à supprimer"},
        },
        "required": ["task_id"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        task_id = kwargs["task_id"]
        try:
            task = Task.objects.get(id=task_id, user=user)
        except Task.DoesNotExist:
            return ToolResult(success=False, data={}, message=f"Tâche #{task_id} introuvable.")

        title = task.title
        task.delete()
        return ToolResult(
            success=True,
            data={"deleted_id": task_id},
            message=f"Tâche '{title}' supprimée.",
        )


class CompleteTaskTool(BaseTool):
    name = "complete_task"
    description = "Marque une tâche comme terminée."
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {"type": "integer", "description": "ID de la tâche à compléter"},
        },
        "required": ["task_id"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        task_id = kwargs["task_id"]
        try:
            task = Task.objects.get(id=task_id, user=user)
        except Task.DoesNotExist:
            return ToolResult(success=False, data={}, message=f"Tâche #{task_id} introuvable.")

        task.completed = True
        task.completed_at = timezone.now()
        task.save()
        return ToolResult(
            success=True,
            data={"task": _task_to_dict(task)},
            message=f"Tâche '{task.title}' marquée comme terminée.",
        )
