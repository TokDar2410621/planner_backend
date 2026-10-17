"""Outil find_free_slots."""
from datetime import datetime

from django.contrib.auth.models import User
from django.utils import timezone
from pydantic import BaseModel, ConfigDict, Field

from services.scheduling.placement import open_intervals
from .base import ToolResult
from .commun import DAY_NAMES, DAY_START_MIN, DAY_END_MIN, _free_slots_from_intervals

NAME = "find_free_slots"
DESCRIPTION = (
    "Trouve les créneaux libres d'un jour (overnight-aware: un quart de nuit occupe "
    "la soirée, le travail de la veille occupe le matin), avec une durée minimum "
    "optionnelle. Appelle-le AVANT de déclarer un jour 'libre' ou de choisir un "
    "créneau toi-même. Activité souple sans heure (lecture, étude, sport): choisis "
    "un créneau libre. Pour un rendez-vous, un cours, un quart, une réunion ou une "
    "leçon fixé par quelqu'un d'autre sans heure de début ou de fin, ou sans date: "
    "ne choisis pas à sa place, demande-lui (present_form ou present_choices)."
)


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: str = Field(
        description="Date au format YYYY-MM-DD",
    )
    min_duration_minutes: int | None = Field(
        default=None, ge=15,
        description="Durée minimum du créneau en minutes (défaut: 30)",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    try:
        target_date = datetime.strptime(kwargs["date"], "%Y-%m-%d").date()
    except ValueError:
        return ToolResult(success=False, data={}, message=(
            f"Format de date invalide : {kwargs.get('date')!r}. Utilise YYYY-MM-DD."),
            public="Je n'ai pas pu vérifier les créneaux libres : la date est invalide. Utilise le format AAAA-MM-JJ.")

    day_of_week = target_date.weekday()
    min_duration = kwargs.get("min_duration_minutes", 30)

    # Créneaux libres 7h-23h via la logique unique overnight-aware: un quart
    # de nuit (ex: 19h-07h) occupe bien la soirée, le débordement de la veille
    # occupe le matin, et les blocs déjà planifiés comptent aussi.
    intervals = open_intervals(user, target_date, DAY_START_MIN, DAY_END_MIN)

    # AUJOURD'HUI, le passé n'est pas libre. Vécu (2026-08-13, 19:48):
    # l'agent proposait « 07:00 à 08:00 » pour une tâche « dans 6
    # minutes ». On rogne à l'heure murale courante, arrondie aux 5 min
    # supérieures; le filtre de durée minimum s'applique après rognage.
    now_local = timezone.localtime()
    if target_date == now_local.date():
        now_min = ((now_local.hour * 60 + now_local.minute + 4) // 5) * 5
        intervals = [(max(s, now_min), e) for s, e in intervals if e > now_min]

    free = _free_slots_from_intervals(intervals, min_duration)

    return ToolResult(
        success=True,
        data={
            "date": target_date.isoformat(),
            "day_name": DAY_NAMES[day_of_week],
            "free_slots": free,
            "total_free_minutes": sum(s["duration_minutes"] for s in free),
        },
        message=f"{len(free)} créneau(x) libre(s) le {DAY_NAMES[day_of_week]} {target_date.isoformat()} (min {min_duration}min).",
    )
