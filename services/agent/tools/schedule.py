"""
Schedule query tools for the Planner AI agent.
"""
from datetime import date, datetime, time, timedelta

from django.contrib.auth.models import User
from django.utils import timezone

from core.models import RecurringBlock, ScheduledBlock, Task
from services.scheduling.exceptions import skipped_block_ids
from services.scheduling.overlap import parse_time
from services.scheduling.placement import (
    fixed_busy_intervals,
    occupied_intervals,
    open_intervals,
    place_day,
)
from .base import BaseTool, ToolResult, validate_choice

DAY_NAMES = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
VALID_TASK_TYPES = {c[0] for c in Task.TASK_TYPE_CHOICES}

# Fenêtre "éveillée" par défaut pour les créneaux libres proposés (7h-23h).
DAY_START_MIN = 7 * 60
DAY_END_MIN = 23 * 60


def _time_to_minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def _minutes_to_str(minutes: int) -> str:
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def _free_slots_from_intervals(intervals, min_duration: int) -> list:
    """(start,end)-minutes -> dicts de créneaux libres, filtrés par durée min."""
    out = []
    for s, e in intervals:
        duration = e - s
        if duration >= min_duration:
            out.append({
                "start_time": _minutes_to_str(s),
                "end_time": _minutes_to_str(e),
                "duration_minutes": duration,
            })
    return out


class GetTodayScheduleTool(BaseTool):
    name = "get_today_schedule"
    description = "Récupère le planning complet d'aujourd'hui : blocs récurrents, tâches planifiées, et créneaux libres."
    parameters = {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "Date au format YYYY-MM-DD (défaut: aujourd'hui)",
            },
        },
        "required": [],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        target_date_str = kwargs.get("date")
        if target_date_str:
            try:
                target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
            except ValueError:
                return ToolResult(success=False, data={}, message="Format de date invalide. Utilise YYYY-MM-DD.")
        else:
            target_date = timezone.localdate()

        # Day of week (0=Monday)
        day_of_week = target_date.weekday()
        day_name = DAY_NAMES[day_of_week]

        # Recurring blocks for this day (moins les occurrences ignorées ce jour-là)
        recurring = RecurringBlock.objects.filter(
            user=user, day_of_week=day_of_week, active=True
        ).exclude(
            id__in=skipped_block_ids(user, target_date)
        ).order_by("start_time")

        placements = {
            p["block_id"]: p
            for p in place_day(user, target_date)
        }
        blocks = []
        for b in recurring:
            start_time = b.start_time.strftime("%H:%M")
            end_time = b.end_time.strftime("%H:%M")
            if b.is_flexible:
                placement = placements.get(b.id)
                if placement is None or placement["skipped"]:
                    continue
                start_time = placement["start_time"]
                end_time = placement["end_time"]

            blocks.append({
                "id": b.id,
                "title": b.title,
                "type": "recurring",
                "block_type": b.block_type,
                "start_time": start_time,
                "end_time": end_time,
                "location": b.location or None,
            })

        # Scheduled tasks for this day
        scheduled = ScheduledBlock.objects.filter(
            user=user, date=target_date
        ).select_related("task").order_by("start_time")

        for sb in scheduled:
            blocks.append({
                "id": sb.id,
                "title": sb.task.title,
                "type": "scheduled_task",
                "task_type": sb.task.task_type,
                "start_time": sb.start_time.strftime("%H:%M"),
                "end_time": sb.end_time.strftime("%H:%M"),
                "completed": sb.actually_completed,
            })

        # Sort all blocks by start time
        blocks.sort(key=lambda x: x["start_time"])

        # Créneaux libres 7h-23h via la logique unique overnight-aware (compte
        # correctement un quart de nuit du jour ET le débordement de la veille,
        # + les blocs déjà planifiés). Corrige le faux "libre 7h-23h".
        free_slots = _free_slots_from_intervals(
            open_intervals(user, target_date, DAY_START_MIN, DAY_END_MIN), 30
        )

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


class GetWeekScheduleTool(BaseTool):
    name = "get_week_schedule"
    description = "Récupère un résumé du planning de la semaine : nombre de blocs par jour, heures occupées, et jours les plus chargés."
    parameters = {
        "type": "object",
        "properties": {
            "start_date": {
                "type": "string",
                "description": "Date de début de la semaine (YYYY-MM-DD, défaut: lundi de la semaine en cours)",
            },
        },
        "required": [],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        start_str = kwargs.get("start_date")
        if start_str:
            try:
                start = datetime.strptime(start_str, "%Y-%m-%d").date()
            except ValueError:
                return ToolResult(success=False, data={}, message="Format de date invalide.")
        else:
            today = timezone.localdate()
            start = today - timedelta(days=today.weekday())  # Monday

        days_summary = []
        total_hours = 0

        for i in range(7):
            day = start + timedelta(days=i)
            day_of_week = day.weekday()

            blocks = RecurringBlock.objects.filter(
                user=user, day_of_week=day_of_week, active=True
            ).exclude(id__in=skipped_block_ids(user, day))

            day_minutes = 0
            block_list = []
            for b in blocks:
                start_min = _time_to_minutes(b.start_time)
                end_min = _time_to_minutes(b.end_time)
                duration = end_min - start_min if end_min > start_min else (24 * 60 - start_min + end_min)
                day_minutes += duration
                block_list.append(f"{b.title} ({b.start_time.strftime('%H:%M')}-{b.end_time.strftime('%H:%M')})")

            total_hours += day_minutes / 60

            days_summary.append({
                "date": day.isoformat(),
                "day_name": DAY_NAMES[day_of_week],
                "block_count": blocks.count(),
                "occupied_hours": round(day_minutes / 60, 1),
                "blocks": block_list,
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


class FindFreeSlotsTool(BaseTool):
    name = "find_free_slots"
    description = "Trouve les créneaux libres sur un jour donné, avec une durée minimum optionnelle."
    parameters = {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "Date au format YYYY-MM-DD",
            },
            "min_duration_minutes": {
                "type": "integer",
                "description": "Durée minimum du créneau en minutes (défaut: 30)",
                "minimum": 15,
            },
        },
        "required": ["date"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        try:
            target_date = datetime.strptime(kwargs["date"], "%Y-%m-%d").date()
        except ValueError:
            return ToolResult(success=False, data={}, message="Format de date invalide.")

        day_of_week = target_date.weekday()
        min_duration = kwargs.get("min_duration_minutes", 30)

        # Créneaux libres 7h-23h via la logique unique overnight-aware: un quart
        # de nuit (ex: 19h-07h) occupe bien la soirée, le débordement de la veille
        # occupe le matin, et les blocs déjà planifiés comptent aussi.
        free = _free_slots_from_intervals(
            open_intervals(user, target_date, DAY_START_MIN, DAY_END_MIN), min_duration
        )

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


class ScheduleTaskAtTool(BaseTool):
    name = "schedule_task_at"
    description = (
        "Planifie un événement PONCTUEL daté à une heure précise (ex: 'lecture ce "
        "samedi 9h-11h', 'rdv mardi 14h-15h'). C'est l'outil pour un événement UNIQUE "
        "sur une date donnée: il crée directement le créneau (verrouillé, la "
        "replanification ne le bouge pas). N'utilise PAS create_block (qui crée une "
        "habitude répétée CHAQUE semaine) pour un événement ponctuel. Si l'utilisateur "
        "ne donne pas d'heure, choisis toi-même un créneau libre (find_free_slots) au "
        "lieu de lui demander."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Titre de l'événement (ex: 'Lecture')."},
            "date": {"type": "string", "description": "Date de l'événement (YYYY-MM-DD). Déduis-la de la DATE du jour."},
            "start_time": {"type": "string", "description": "Heure de début (HH:MM)."},
            "end_time": {"type": "string", "description": "Heure de fin (HH:MM), après le début."},
            "task_type": {
                "type": "string",
                "enum": ["deep_work", "shallow", "errand"],
                "description": "Type (optionnel, défaut shallow).",
            },
            "priority": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Priorité 1-10 (optionnel)."},
            "description": {"type": "string", "description": "Détails (optionnel)."},
        },
        "required": ["title", "date", "start_time", "end_time"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        title = (kwargs.get("title") or "").strip()
        if not title:
            return ToolResult(success=False, data={}, message="Titre requis.")

        try:
            target_date = datetime.strptime(kwargs["date"], "%Y-%m-%d").date()
        except (ValueError, KeyError, TypeError):
            return ToolResult(success=False, data={}, message="Format de date invalide. Utilise YYYY-MM-DD.")

        try:
            start_t = parse_time(kwargs["start_time"])
            end_t = parse_time(kwargs["end_time"])
        except (ValueError, KeyError):
            return ToolResult(success=False, data={}, message="Heure invalide (attendu HH:MM).")

        s = _time_to_minutes(start_t)
        e = _time_to_minutes(end_t)
        if e <= s:
            return ToolResult(
                success=False,
                data={},
                message="L'heure de fin doit être après l'heure de début (un événement ponctuel ne passe pas minuit ici).",
            )

        err = validate_choice(kwargs.get("task_type"), VALID_TASK_TYPES, "task_type")
        if err:
            return ToolResult(success=False, data={}, message=err)

        # Chevauchement avec les murs RÉELS du jour (fixes + blocs datés déjà
        # planifiés), fenêtre pleine 0-24h. Les récurrents souples ne sont pas
        # des murs: l'événement ponctuel verrouillé les fera se replacer.
        for bs, be in fixed_busy_intervals(user, target_date):
            if s < be and bs < e:
                return ToolResult(
                    success=False,
                    data={"conflict": {"start_time": _minutes_to_str(bs), "end_time": _minutes_to_str(be)}},
                    message=(
                        f"Ce créneau ({_minutes_to_str(s)}-{_minutes_to_str(e)}) chevauche une "
                        f"occupation existante ({_minutes_to_str(bs)}-{_minutes_to_str(be)}). "
                        f"Choisis un autre horaire libre."
                    ),
                )

        # Sleep is a protected flexible block: a one-off urgent event may move
        # ordinary flexible habits, but it must not silently eat sleep.
        for placement in place_day(user, target_date):
            if placement["skipped"] or placement["block_type"] != "sleep":
                continue
            ps = placement["start_min"]
            pe = placement["end_min"]
            if ps is None or pe is None:
                continue
            sleep_pieces = [(ps, pe)] if pe > ps else [(ps, 24 * 60), (0, pe)]
            for bs, be in sleep_pieces:
                if s < be and bs < e:
                    conflict_start, conflict_end = bs, be
                    for os, oe in occupied_intervals(user, target_date, 0, 24 * 60):
                        if s < oe and os < e:
                            conflict_start, conflict_end = os, oe
                            break
                    return ToolResult(
                        success=False,
                        data={
                            "conflict": {
                                "start_time": _minutes_to_str(conflict_start),
                                "end_time": _minutes_to_str(conflict_end),
                            }
                        },
                        message=(
                            f"Ce créneau ({_minutes_to_str(s)}-{_minutes_to_str(e)}) "
                            f"chevauche le sommeil protégé "
                            f"({_minutes_to_str(conflict_start)}-"
                            f"{_minutes_to_str(conflict_end)}). "
                            f"Choisis un autre horaire libre."
                        ),
                    )

        # Réutilise une tâche active du même titre (idempotence), sinon la crée.
        task = Task.objects.filter(user=user, completed=False, title__iexact=title).first()
        if task is None:
            task = Task.objects.create(
                user=user,
                title=title,
                task_type=kwargs.get("task_type", "shallow"),
                priority=kwargs.get("priority", 5),
                description=kwargs.get("description", ""),
            )

        sb = ScheduledBlock.objects.create(
            user=user, task=task, date=target_date,
            start_time=start_t, end_time=end_t, locked=True,
        )
        return ToolResult(
            success=True,
            data={"scheduled_block": {
                "id": sb.id,
                "title": task.title,
                "date": target_date.isoformat(),
                "start_time": start_t.strftime("%H:%M"),
                "end_time": end_t.strftime("%H:%M"),
            }},
            message=(
                f"'{task.title}' planifié le {DAY_NAMES[target_date.weekday()]} "
                f"{target_date.isoformat()} de {start_t.strftime('%H:%M')} à {end_t.strftime('%H:%M')}."
            ),
        )
