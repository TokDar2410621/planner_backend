"""
Forgiving, elastic streak (le streak parfait tue au premier manque).

A day 'counts' if the user did ANYTHING that mattered: completed a block/task, OR
adjusted their plan (a SchedulePlanChange = they engaged, they didn't abandon).
Up to FREEZE_BUDGET inactive days are tolerated inside the current run (retroactive
freeze), so real life (a guest, a swapped shift, an absent prof) never snaps the
streak. Tied to engagement, not perfection.
"""
from datetime import timedelta

from django.utils import timezone

FREEZE_BUDGET = 2  # inactive days tolerated inside the current streak
LOOKBACK_DAYS = 120


def _active_dates(user, since):
    """Set of local dates on which the user did or adjusted something."""
    from core.models import (
        RecurringBlockCompletion, ScheduledBlock, Task, SchedulePlanChange,
    )
    dates = set()
    dates |= set(
        RecurringBlockCompletion.objects.filter(user=user, date__gte=since)
        .values_list('date', flat=True)
    )
    dates |= set(
        ScheduledBlock.objects.filter(
            user=user, actually_completed=True, date__gte=since
        ).values_list('date', flat=True)
    )
    # Task completions + plan adjustments (datetimes -> local date).
    for dt in Task.objects.filter(
        user=user, completed=True, completed_at__isnull=False
    ).values_list('completed_at', flat=True):
        d = timezone.localtime(dt).date()
        if d >= since:
            dates.add(d)
    for dt in SchedulePlanChange.objects.filter(user=user).values_list('created_at', flat=True):
        d = timezone.localtime(dt).date()
        if d >= since:
            dates.add(d)
    return dates


def compute_streak(user) -> dict:
    """Current forgiving streak (active days), today counted as in-progress."""
    today = timezone.localdate()
    since = today - timedelta(days=LOOKBACK_DAYS)
    active = _active_dates(user, since)

    # Walk back from today. `pending` = inactive days since the last active day;
    # a run breaks once more than FREEZE_BUDGET consecutive days are missed.
    # Only INTERIOR gaps (between two active days) are counted as freezes used.
    streak = 0
    freezes_used = 0
    leading_gap = None
    pending = 0
    d = today
    while d >= since:
        if d in active:
            if leading_gap is None:
                leading_gap = pending  # gaps between today and the first active day
            streak += 1
            freezes_used += pending
            pending = 0
        elif d != today:
            pending += 1
            if pending > FREEZE_BUDGET:
                break  # too many consecutive misses -> streak ends here
        d -= timedelta(days=1)
    if leading_gap is None:
        leading_gap = pending
    misses = freezes_used

    if streak == 0:
        message = "Nouveau départ — un seul bloc aujourd'hui et le streak démarre."
    else:
        message = f"{streak} jour(s) où tu as tenu ou ajusté ton plan. On continue."

    return {
        'current_streak': streak,
        'active_today': today in active,
        'freezes_used': misses,
        'freezes_available': max(0, FREEZE_BUDGET - leading_gap),
        'message': message,
    }
