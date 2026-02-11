"""
Planning optimization tools for the Planner AI agent.
"""
from datetime import timedelta, time

from django.contrib.auth.models import User
from django.utils import timezone

from core.models import RecurringBlock, Task
from .base import BaseTool, ToolResult

DAY_NAMES = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']


def _time_to_minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def _minutes_to_str(minutes: int) -> str:
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


class SuggestOptimizationTool(BaseTool):
    name = "suggest_schedule_optimization"
    description = "Analyse le planning de l'utilisateur et identifie des points d'amélioration : créneaux vides, manque de pauses, déséquilibres, etc."
    parameters = {
        "type": "object",
        "properties": {
            "focus": {
                "type": "string",
                "description": "Domaine d'optimisation",
                "enum": ["balance", "productivity", "rest", "general"],
            },
        },
        "required": [],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        focus = kwargs.get("focus", "general")
        profile = user.profile

        blocks = RecurringBlock.objects.filter(user=user, active=True)
        pending_tasks = Task.objects.filter(user=user, completed=False)

        suggestions = []
        analysis = {
            "total_blocks": blocks.count(),
            "pending_tasks": pending_tasks.count(),
            "days_with_blocks": set(),
            "block_types": {},
            "daily_hours": {},
        }

        # Analyze blocks per day and type
        for b in blocks:
            analysis["days_with_blocks"].add(b.day_of_week)
            analysis["block_types"][b.block_type] = analysis["block_types"].get(b.block_type, 0) + 1

            start = _time_to_minutes(b.start_time)
            end = _time_to_minutes(b.end_time)
            duration = (end - start) if end > start else (24 * 60 - start + end)
            day_name = DAY_NAMES[b.day_of_week]
            analysis["daily_hours"][day_name] = analysis["daily_hours"].get(day_name, 0) + duration / 60

        analysis["days_with_blocks"] = list(analysis["days_with_blocks"])

        # Check for missing sleep blocks
        if "sleep" not in analysis["block_types"]:
            suggestions.append({
                "type": "missing_sleep",
                "priority": "high",
                "message": "Aucun bloc de sommeil n'est configuré. Ajoute tes heures de sommeil pour un planning plus réaliste.",
            })

        # Check for missing meal blocks
        if "meal" not in analysis["block_types"]:
            suggestions.append({
                "type": "missing_meals",
                "priority": "medium",
                "message": "Aucun bloc de repas n'est configuré. Ajouter des pauses déjeuner/dîner aide à structurer la journée.",
            })

        # Check for days without blocks
        empty_days = [DAY_NAMES[d] for d in range(7) if d not in analysis["days_with_blocks"]]
        if empty_days and len(empty_days) < 7:
            suggestions.append({
                "type": "empty_days",
                "priority": "low",
                "message": f"Pas de blocs le {', '.join(empty_days)}. Tu veux ajouter des activités pour ces jours ?",
            })

        # Check for overloaded days (>10h)
        overloaded = [d for d, h in analysis["daily_hours"].items() if h > 10]
        if overloaded:
            suggestions.append({
                "type": "overloaded_days",
                "priority": "high",
                "message": f"Journée(s) chargée(s) (>10h): {', '.join(overloaded)}. Pense à ajouter des pauses.",
            })

        # Check for overdue tasks
        overdue = pending_tasks.filter(deadline__lt=timezone.now()).count()
        if overdue > 0:
            suggestions.append({
                "type": "overdue_tasks",
                "priority": "high",
                "message": f"{overdue} tâche(s) en retard. Il faudrait les replanifier ou les compléter.",
            })

        # Check for unscheduled high-priority tasks
        high_priority = pending_tasks.filter(priority__gte=7).count()
        if high_priority > 0:
            suggestions.append({
                "type": "high_priority_tasks",
                "priority": "medium",
                "message": f"{high_priority} tâche(s) haute priorité en attente. Trouve-leur un créneau !",
            })

        return ToolResult(
            success=True,
            data={
                "analysis": {
                    "total_blocks": analysis["total_blocks"],
                    "pending_tasks": analysis["pending_tasks"],
                    "block_types": analysis["block_types"],
                    "daily_hours": analysis["daily_hours"],
                },
                "suggestions": suggestions,
            },
            message=f"{len(suggestions)} suggestion(s) d'optimisation trouvée(s).",
        )


class DetectConflictsTool(BaseTool):
    name = "detect_conflicts"
    description = "Détecte les chevauchements et conflits dans le planning de l'utilisateur."
    parameters = {
        "type": "object",
        "properties": {
            "day_of_week": {
                "type": "integer",
                "description": "Vérifier un jour spécifique (0=Lundi, 6=Dimanche). Si absent, vérifie tous les jours.",
                "minimum": 0,
                "maximum": 6,
            },
        },
        "required": [],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        day = kwargs.get("day_of_week")
        blocks = RecurringBlock.objects.filter(user=user, active=True)

        if day is not None:
            days_to_check = [day]
        else:
            days_to_check = range(7)

        conflicts = []

        for d in days_to_check:
            day_blocks = list(blocks.filter(day_of_week=d).order_by("start_time"))

            for i in range(len(day_blocks)):
                for j in range(i + 1, len(day_blocks)):
                    a = day_blocks[i]
                    b = day_blocks[j]

                    a_start = _time_to_minutes(a.start_time)
                    a_end = _time_to_minutes(a.end_time)
                    b_start = _time_to_minutes(b.start_time)
                    b_end = _time_to_minutes(b.end_time)

                    # Handle non-night-shift blocks
                    if not a.is_night_shift and not b.is_night_shift:
                        if a_start < b_end and b_start < a_end:
                            overlap_start = max(a_start, b_start)
                            overlap_end = min(a_end, b_end)
                            conflicts.append({
                                "day": d,
                                "day_name": DAY_NAMES[d],
                                "block_a": {"id": a.id, "title": a.title, "time": f"{a.start_time.strftime('%H:%M')}-{a.end_time.strftime('%H:%M')}"},
                                "block_b": {"id": b.id, "title": b.title, "time": f"{b.start_time.strftime('%H:%M')}-{b.end_time.strftime('%H:%M')}"},
                                "overlap_minutes": overlap_end - overlap_start,
                            })

        return ToolResult(
            success=True,
            data={"conflicts": conflicts, "count": len(conflicts)},
            message=f"{len(conflicts)} conflit(s) détecté(s)." if conflicts else "Aucun conflit détecté.",
        )
