"""
Schedule query tools for the Planner AI agent.
"""
from datetime import date, datetime, time, timedelta

from django.contrib.auth.models import User
from django.utils import timezone

from core.models import RecurringBlock, ScheduledBlock, Task
from .base import BaseTool, ToolResult

DAY_NAMES = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']


def _time_to_minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def _minutes_to_str(minutes: int) -> str:
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


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

        # Recurring blocks for this day
        recurring = RecurringBlock.objects.filter(
            user=user, day_of_week=day_of_week, active=True
        ).order_by("start_time")

        blocks = []
        for b in recurring:
            blocks.append({
                "id": b.id,
                "title": b.title,
                "type": "recurring",
                "block_type": b.block_type,
                "start_time": b.start_time.strftime("%H:%M"),
                "end_time": b.end_time.strftime("%H:%M"),
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

        # Find free slots (between 7:00 and 23:00)
        free_slots = self._find_free_slots(blocks, 7 * 60, 23 * 60)

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

    def _find_free_slots(self, blocks, day_start_min, day_end_min):
        """Find free time slots between blocks."""
        occupied = []
        for b in blocks:
            parts = b["start_time"].split(":")
            start = int(parts[0]) * 60 + int(parts[1])
            parts = b["end_time"].split(":")
            end = int(parts[0]) * 60 + int(parts[1])
            if end > start:  # Skip overnight blocks
                occupied.append((start, end))

        occupied.sort()

        # Merge overlapping
        merged = []
        for start, end in occupied:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Find gaps
        free = []
        current = day_start_min
        for start, end in merged:
            if start > current:
                duration = start - current
                if duration >= 30:  # Only slots >= 30 min
                    free.append({
                        "start_time": _minutes_to_str(current),
                        "end_time": _minutes_to_str(start),
                        "duration_minutes": duration,
                    })
            current = max(current, end)

        if current < day_end_min:
            duration = day_end_min - current
            if duration >= 30:
                free.append({
                    "start_time": _minutes_to_str(current),
                    "end_time": _minutes_to_str(day_end_min),
                    "duration_minutes": duration,
                })

        return free


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
            )

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

        # Get all blocks for this day
        blocks = RecurringBlock.objects.filter(
            user=user, day_of_week=day_of_week, active=True
        ).order_by("start_time")

        occupied = []
        for b in blocks:
            start = _time_to_minutes(b.start_time)
            end = _time_to_minutes(b.end_time)
            if end > start:
                occupied.append((start, end))

        # Merge overlapping
        occupied.sort()
        merged = []
        for start, end in occupied:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Find gaps (7:00 - 23:00)
        free = []
        current = 7 * 60
        day_end = 23 * 60

        for start, end in merged:
            if start > current:
                duration = start - current
                if duration >= min_duration:
                    free.append({
                        "start_time": _minutes_to_str(current),
                        "end_time": _minutes_to_str(start),
                        "duration_minutes": duration,
                    })
            current = max(current, end)

        if current < day_end:
            duration = day_end - current
            if duration >= min_duration:
                free.append({
                    "start_time": _minutes_to_str(current),
                    "end_time": _minutes_to_str(day_end),
                    "duration_minutes": duration,
                })

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
