"""
Goal tools for the Planner AI agent.
"""
from django.contrib.auth.models import User

from core.models import Goal
from .base import BaseTool, ToolResult


def _goal_to_dict(goal: Goal) -> dict:
    return {
        "id": goal.id,
        "title": goal.title,
        "description": goal.description or None,
        "goal_type": goal.goal_type,
        "deadline": goal.deadline.isoformat() if goal.deadline else None,
        "progress": goal.progress,
        "status": goal.status,
        "created_at": goal.created_at.isoformat(),
    }


class ListGoalsTool(BaseTool):
    name = "list_goals"
    description = "Liste les objectifs de l'utilisateur. Peut filtrer par statut (active, completed, paused)."
    parameters = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "Filtrer par statut",
                "enum": ["active", "completed", "paused", "cancelled"],
            },
        },
        "required": [],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        goals = Goal.objects.filter(user=user)

        status = kwargs.get("status")
        if status:
            goals = goals.filter(status=status)

        goal_list = [_goal_to_dict(g) for g in goals]

        return ToolResult(
            success=True,
            data={"goals": goal_list, "count": len(goal_list)},
            message=f"{len(goal_list)} objectif(s) trouvé(s).",
        )


class CreateGoalTool(BaseTool):
    name = "create_goal"
    description = "Crée un nouvel objectif pour l'utilisateur."
    parameters = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Titre de l'objectif",
            },
            "goal_type": {
                "type": "string",
                "description": "Type d'objectif",
                "enum": ["short_term", "long_term"],
            },
            "description": {
                "type": "string",
                "description": "Description détaillée (optionnel)",
            },
            "deadline": {
                "type": "string",
                "description": "Date limite au format YYYY-MM-DD (optionnel)",
            },
        },
        "required": ["title", "goal_type"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        goal = Goal.objects.create(
            user=user,
            title=kwargs["title"],
            goal_type=kwargs["goal_type"],
            description=kwargs.get("description", ""),
            deadline=kwargs.get("deadline"),
        )
        type_label = "court terme" if goal.goal_type == "short_term" else "long terme"
        return ToolResult(
            success=True,
            data={"goal": _goal_to_dict(goal)},
            message=f"Objectif '{goal.title}' créé ({type_label}).",
        )


class UpdateGoalTool(BaseTool):
    name = "update_goal"
    description = "Met à jour un objectif existant (progression, statut, titre, etc.)."
    parameters = {
        "type": "object",
        "properties": {
            "goal_id": {
                "type": "integer",
                "description": "ID de l'objectif à modifier",
            },
            "title": {"type": "string", "description": "Nouveau titre"},
            "description": {"type": "string"},
            "progress": {
                "type": "integer",
                "description": "Progression (0-100)",
                "minimum": 0,
                "maximum": 100,
            },
            "status": {
                "type": "string",
                "enum": ["active", "completed", "paused", "cancelled"],
            },
            "deadline": {"type": "string", "description": "Nouvelle deadline (YYYY-MM-DD)"},
        },
        "required": ["goal_id"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        goal_id = kwargs["goal_id"]
        try:
            goal = Goal.objects.get(id=goal_id, user=user)
        except Goal.DoesNotExist:
            return ToolResult(success=False, data={}, message=f"Objectif #{goal_id} introuvable.")

        for field in ["title", "description", "progress", "status", "deadline"]:
            if field in kwargs and kwargs[field] is not None:
                setattr(goal, field, kwargs[field])

        # Auto-complete if progress reaches 100
        if goal.progress >= 100 and goal.status == "active":
            goal.status = "completed"

        goal.save()
        return ToolResult(
            success=True,
            data={"goal": _goal_to_dict(goal)},
            message=f"Objectif '{goal.title}' mis à jour (progression: {goal.progress}%).",
        )
