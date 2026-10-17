"""Outil suggest_schedule_optimization."""
from typing import Literal

from django.contrib.auth.models import User
from django.utils import timezone
from pydantic import BaseModel, ConfigDict, Field

from core.models import RecurringBlock, Task
from .base import ISSUE_TERMINEE, ToolResult
from .commun import DAY_NAMES, _time_to_minutes

NAME = "suggest_schedule_optimization"
DESCRIPTION = "Analyse le planning de l'utilisateur et identifie des points d'amélioration : créneaux vides, manque de pauses, déséquilibres, etc."

FocusOptimisation = Literal["balance", "productivity", "rest", "general"]


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    focus: FocusOptimisation | None = Field(
        default=None,
        description="Domaine d'optimisation",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
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

    # Contrat d'issue (audit 2026-10-13): les suggestions portent deja
    # des messages affichables, on les relaie tels quels.
    if suggestions:
        lignes = "\n".join(f"- {s.get('message', '').strip()}"
                           for s in suggestions if s.get("message"))
        public = f"Quelques pistes pour ton horaire :\n{lignes}"
    else:
        public = "Ton horaire est bien équilibré, rien à signaler."
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
        issue=ISSUE_TERMINEE, cause="", public=public,
    )
