"""
Analytics/productivity tools for the Planner AI agent.
"""
from datetime import timedelta

from django.contrib.auth.models import User
from django.db.models import Count, Q
from django.utils import timezone

from core.models import Task, RecurringBlockCompletion, RecurringBlock
from .base import BaseTool, ToolResult


class GetProductivityStatsTool(BaseTool):
    name = "get_productivity_stats"
    description = "Récupère les statistiques de productivité : taux de complétion des tâches, blocs complétés, streak de jours consécutifs."
    parameters = {
        "type": "object",
        "properties": {
            "period_days": {
                "type": "integer",
                "description": "Période en jours pour les stats (défaut: 7)",
                "minimum": 1,
                "maximum": 90,
            },
        },
        "required": [],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        period = kwargs.get("period_days", 7)
        now = timezone.now()
        start_date = (now - timedelta(days=period)).date()

        # Task completion stats
        total_tasks = Task.objects.filter(user=user, created_at__date__gte=start_date).count()
        completed_tasks = Task.objects.filter(
            user=user, completed=True, completed_at__date__gte=start_date
        ).count()
        completion_rate = round(completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

        # Recurring block completion stats
        total_blocks = RecurringBlock.objects.filter(user=user, active=True).count()
        completions = RecurringBlockCompletion.objects.filter(
            user=user, date__gte=start_date
        ).count()

        # Streak calculation (consecutive days with at least one completion)
        streak = 0
        check_date = timezone.localdate()
        while True:
            day_completions = RecurringBlockCompletion.objects.filter(
                user=user, date=check_date
            ).count()
            task_completions = Task.objects.filter(
                user=user, completed=True, completed_at__date=check_date
            ).count()
            if day_completions + task_completions > 0:
                streak += 1
                check_date -= timedelta(days=1)
            else:
                break

        # Pending tasks
        pending_tasks = Task.objects.filter(user=user, completed=False).count()
        overdue_tasks = Task.objects.filter(
            user=user, completed=False, deadline__lt=now
        ).count()

        return ToolResult(
            success=True,
            data={
                "period_days": period,
                "tasks_created": total_tasks,
                "tasks_completed": completed_tasks,
                "task_completion_rate": completion_rate,
                "recurring_blocks_total": total_blocks,
                "recurring_completions": completions,
                "streak_days": streak,
                "pending_tasks": pending_tasks,
                "overdue_tasks": overdue_tasks,
            },
            message=f"Stats ({period}j): {completion_rate}% tâches complétées, streak de {streak} jour(s), {overdue_tasks} tâche(s) en retard.",
        )
