"""Outil get_today_schedule."""
from datetime import datetime

from django.contrib.auth.models import User
from django.utils import timezone
from pydantic import BaseModel, ConfigDict, Field

from services.scheduling.day_view import effective_day_entries
from services.scheduling.placement import open_intervals
from .base import ISSUE_ECHEC, ToolResult
from .commun import DAY_NAMES, DAY_START_MIN, DAY_END_MIN, _free_slots_from_intervals

NAME = "get_today_schedule"
DESCRIPTION = (
    "Récupère le planning EFFECTIF d'un jour (blocs récurrents aux heures PLACÉES, "
    "tâches planifiées, créneaux libres). Consulte-le avant d'affirmer où se trouve "
    "une activité ou si elle a bougé, et parle des heures effectives, jamais de mémoire "
    "ni d'après l'historique (un bloc souple peut être placé à une autre heure que son "
    "heure habituelle)."
)


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: str | None = Field(
        default=None,
        description="Date au format YYYY-MM-DD (défaut: aujourd'hui)",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    target_date_str = kwargs.get("date")
    if target_date_str:
        try:
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        except ValueError:
            return ToolResult(success=False, data={}, message="Format de date invalide. Utilise YYYY-MM-DD.",
                          issue=ISSUE_ECHEC, cause="date_illisible",
                          public="Je n'ai pas pu vérifier ce créneau : la date était illisible.")
    else:
        target_date = timezone.localdate()

    # Day of week (0=Monday)
    day_of_week = target_date.weekday()
    day_name = DAY_NAMES[day_of_week]

    blocks = effective_day_entries(user, target_date)

    # Créneaux libres 7h-23h via la logique unique overnight-aware (compte
    # correctement un quart de nuit du jour ET le débordement de la veille,
    # + les blocs déjà planifiés). Corrige le faux "libre 7h-23h".
    intervals = open_intervals(user, target_date, DAY_START_MIN, DAY_END_MIN)
    # AUJOURD'HUI, le passé n'est pas libre, comme dans find_free_slots.
    # Vécu (banc du 2026-09-15, 16:31): la ligne « Libre : 15 h 50 à 18 h »
    # de cette lecture est devenue un bouton « 15 h 50 à 18 h ».
    now_local = timezone.localtime()
    if target_date == now_local.date():
        now_min = ((now_local.hour * 60 + now_local.minute + 4) // 5) * 5
        intervals = [(max(s, now_min), e) for s, e in intervals if e > now_min]
    free_slots = _free_slots_from_intervals(intervals, 30)

    return ToolResult(
        success=True,
        data={
            "date": target_date.isoformat(),
            "day_name": day_name,
            "blocks": blocks,
            "free_slots": free_slots,
            "total_blocks": len(blocks),
            "total_free_minutes": sum(s["duration_minutes"] for s in free_slots),
        },
        message=f"Planning du {day_name} {target_date.isoformat()}: {len(blocks)} bloc(s), {len(free_slots)} créneau(x) libre(s).",
    )
