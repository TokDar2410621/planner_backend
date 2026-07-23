"""
Social co-presence — Planner's structural moat.

Because Planner knows several users' REAL schedules, it can compute their common
FREE time on a given day (fin de cours, retour maison) and propose body-doubling
/ study groups on the actual overlap. Non-punitive accountability: the social
stake replaces app guilt (spec / recherche nature humaine).

Interval math in minutes-since-midnight, clipped to a waking window.
"""
from datetime import timedelta

from services.scheduling.overlap import MINUTES_PER_DAY
from services.scheduling.exceptions import skipped_block_ids

DAY_START_MIN = 8 * 60   # default waking window for co-presence suggestions
DAY_END_MIN = 22 * 60


def _to_min(t) -> int:
    return t.hour * 60 + t.minute


def _merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(e, out[-1][1]))
        else:
            out.append((s, e))
    return out


def busy_intervals(user, date, day_start=DAY_START_MIN, day_end=DAY_END_MIN):
    """Occupied (start, end) minute-intervals for `user` on `date`, clipped."""
    from core.models import RecurringBlock, ScheduledBlock

    dow = date.weekday()
    prev = (dow - 1) % 7
    raw = []
    skipped_today = skipped_block_ids(user, date)
    skipped_prev = skipped_block_ids(user, date - timedelta(days=1))

    for b in RecurringBlock.objects.filter(
        user=user, active=True, day_of_week=dow
    ).exclude(id__in=skipped_today):
        s, e = _to_min(b.start_time), _to_min(b.end_time)
        raw.append((s, MINUTES_PER_DAY) if (b.is_night_shift or e <= s) else (s, e))
    # Previous-day overnight block spilling into this morning (skip if that
    # previous-day occurrence was cancelled).
    for b in RecurringBlock.objects.filter(
        user=user, active=True, day_of_week=prev
    ).exclude(id__in=skipped_prev):
        s, e = _to_min(b.start_time), _to_min(b.end_time)
        if b.is_night_shift or e <= s:
            raw.append((0, e))
    for sb in ScheduledBlock.objects.filter(user=user, date=date):
        s, e = _to_min(sb.start_time), _to_min(sb.end_time)
        raw.append((s, MINUTES_PER_DAY) if e <= s else (s, e))

    clipped = []
    for s, e in raw:
        s, e = max(s, day_start), min(e, day_end)
        if e > s:
            clipped.append((s, e))
    return _merge(clipped)


def free_intervals(user, date, day_start=DAY_START_MIN, day_end=DAY_END_MIN):
    """Free (start, end) minute-intervals for `user` on `date`."""
    busy = busy_intervals(user, date, day_start, day_end)
    free = []
    cur = day_start
    for s, e in busy:
        if s > cur:
            free.append((cur, s))
        cur = max(cur, e)
    if cur < day_end:
        free.append((cur, day_end))
    return free


def _intersect(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _fmt(m):
    return f"{m // 60:02d}:{m % 60:02d}"


def common_free(user_a, user_b, date, min_minutes=30):
    """Shared free slots >= min_minutes between two users on a date."""
    slots = _intersect(free_intervals(user_a, date), free_intervals(user_b, date))
    return [
        {'start': _fmt(s), 'end': _fmt(e), 'minutes': e - s}
        for s, e in slots
        if e - s >= min_minutes
    ]
