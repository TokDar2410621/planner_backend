"""Outil get_week_schedule."""
from datetime import datetime, timedelta

from django.contrib.auth.models import User
from django.utils import timezone
from pydantic import BaseModel, ConfigDict, Field

from services.scheduling.day_view import effective_day_entries
from .base import ToolResult
from .commun import DAY_NAMES

NAME = "get_week_schedule"
DESCRIPTION = (
    "Récupère un résumé du planning de la semaine : nombre de blocs par jour, "
    "heures occupées, et jours les plus chargés."
)


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start_date: str | None = Field(
        default=None,
        description="Date de début de la semaine (YYYY-MM-DD, défaut: lundi de la semaine en cours)",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    start_str = kwargs.get("start_date")
    if start_str:
        try:
            start = datetime.strptime(start_str, "%Y-%m-%d").date()
        except ValueError:
            return ToolResult(success=False, data={}, message=(
                f"Format de date invalide : {start_str!r}. Utilise YYYY-MM-DD."),
                public="Je n'ai pas pu lire ta semaine : la date est invalide. Utilise le format AAAA-MM-JJ.")
    else:
        today = timezone.localdate()
        start = today - timedelta(days=today.weekday())  # Monday

    days_summary = []
    total_hours = 0

    for i in range(7):
        day = start + timedelta(days=i)
        day_of_week = day.weekday()

        detail = effective_day_entries(user, day)

        day_minutes = 0
        block_list = []
        for b in detail:
            start_h, start_m = map(int, b["start_time"].split(":"))
            end_h, end_m = map(int, b["end_time"].split(":"))
            start_min = start_h * 60 + start_m
            end_min = end_h * 60 + end_m
            duration = end_min - start_min if end_min > start_min else (24 * 60 - start_min + end_min)
            day_minutes += duration
            block_list.append(f"{b['title']} ({b['start_time']}-{b['end_time']})")

        total_hours += day_minutes / 60

        days_summary.append({
            "date": day.isoformat(),
            "day_name": DAY_NAMES[day_of_week],
            "block_count": len(detail),
            "occupied_hours": round(day_minutes / 60, 1),
            "blocks": block_list,
            "detail": detail,
        })

    return ToolResult(
        success=True,
        data={
            "week_start": start.isoformat(),
            "days": days_summary,
            "total_hours": round(total_hours, 1),
        },
        message=f"Semaine du {start.isoformat()}: {round(total_hours, 1)}h planifiées sur 7 jours.",
    )
