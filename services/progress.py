"""
Positive weekly summary (progress principle — Amabile).

Surfaces only what was ACCOMPLISHED, in the user's own terms, tied to their real
goal. Never a raw score, never the misses (peak-end: end on the win). Counters
the 'slow churn' of not seeing results.
"""
from datetime import timedelta

from django.utils import timezone


def _completed_minutes(user, start, end):
    from core.models import ScheduledBlock
    total = 0
    for sb in ScheduledBlock.objects.filter(
        user=user, date__gte=start, date__lt=end, actually_completed=True
    ):
        mins = sb.actual_duration_minutes
        if mins is None:
            s = sb.start_time.hour * 60 + sb.start_time.minute
            e = sb.end_time.hour * 60 + sb.end_time.minute
            mins = max(0, e - s)
        total += mins
    return total


def weekly_summary(user) -> dict:
    from core.models import RecurringBlockCompletion, Task

    today = timezone.localdate()
    week_start = today - timedelta(days=today.weekday())
    prev_start = week_start - timedelta(days=7)

    this_week = _completed_minutes(user, week_start, week_start + timedelta(days=7))
    last_week = _completed_minutes(user, prev_start, week_start)
    delta_pct = None
    if last_week > 0:
        delta_pct = round((this_week - last_week) / last_week * 100)

    blocks_done = RecurringBlockCompletion.objects.filter(
        user=user, date__gte=week_start
    ).count()
    tasks_done = sum(
        1
        for dt in Task.objects.filter(
            user=user, completed=True, completed_at__isnull=False
        ).values_list('completed_at', flat=True)
        if timezone.localtime(dt).date() >= week_start
    )

    # Tie to the nearest real stake (upcoming deadline) if any.
    now = timezone.now()
    next_task = (
        Task.objects.filter(user=user, completed=False, deadline__gte=now)
        .order_by('deadline')
        .first()
    )
    stake = None
    if next_task and next_task.deadline:
        days = (timezone.localtime(next_task.deadline).date() - today).days
        stake = f"« {next_task.title} » dans {days} jour(s)"

    hours = round(this_week / 60, 1)
    parts = [f"{hours} h de travail bouclées cette semaine"]
    if delta_pct is not None and delta_pct > 0:
        parts.append(f"+{delta_pct}% vs la semaine passée")
    if tasks_done:
        parts.append(f"{tasks_done} tâche(s) terminée(s)")
    message = ", ".join(parts) + "."
    if stake:
        message += f" Prochain enjeu : {stake}."

    return {
        'completed_minutes': this_week,
        'completed_hours': hours,
        'delta_pct': delta_pct,
        'blocks_completed': blocks_done,
        'tasks_completed': tasks_done,
        'next_stake': stake,
        'message': message,
    }
