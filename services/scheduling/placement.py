"""Pure placement engine for flexible recurring blocks.

This module decides where flexible recurring blocks sit on one concrete date,
using fixed commitments as hard walls. It performs ORM reads only and never
writes schedule state.
"""
from __future__ import annotations

from datetime import timedelta

from services.scheduling.exceptions import skipped_block_ids
from services.scheduling.overlap import (
    MINUTES_PER_DAY,
    is_overnight,
    time_to_min,
)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _clip_interval(
    start: int,
    end: int,
    day_start: int = 0,
    day_end: int = MINUTES_PER_DAY,
) -> tuple[int, int] | None:
    start = max(start, day_start)
    end = min(end, day_end)
    if end <= start:
        return None
    return start, end


def _fmt(minutes: int | None) -> str | None:
    if minutes is None:
        return None
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _overlaps(interval: tuple[int, int], intervals: list[tuple[int, int]]) -> bool:
    start, end = interval
    for other_start, other_end in intervals:
        if start < other_end and other_start < end:
            return True
    return False


def _result(
    block,
    *,
    start_min: int | None,
    end_min: int | None,
    preferred: bool,
    shrunk: bool,
    skipped: bool,
    overnight_kept: bool,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict:
    return {
        "block_id": block.id,
        "title": block.title,
        "block_type": block.block_type,
        "start_min": start_min,
        "end_min": end_min,
        "start_time": start_time if start_time is not None else _fmt(start_min),
        "end_time": end_time if end_time is not None else _fmt(end_min),
        "preferred": preferred,
        "shrunk": shrunk,
        "skipped": skipped,
        "overnight_kept": overnight_kept,
    }


def fixed_busy_intervals(user, date) -> list[tuple[int, int]]:
    """Return fixed hard-wall intervals for ``user`` on ``date``.

    Intervals are clipped to ``[0, 1440]``, merged, and sorted. Fixed recurring
    blocks count on their own start date, previous-day fixed overnight blocks
    spill into the current morning, and all scheduled blocks on the date count
    as walls. Flexible recurring blocks are intentionally excluded.
    """
    from core.models import RecurringBlock, ScheduledBlock

    dow = date.weekday()
    prev_dow = (dow - 1) % 7
    prev_date = date - timedelta(days=1)
    skipped_today = skipped_block_ids(user, date)
    skipped_prev = skipped_block_ids(user, prev_date)

    raw: list[tuple[int, int]] = []

    today_blocks = RecurringBlock.objects.filter(
        user=user,
        active=True,
        day_of_week=dow,
    ).exclude(id__in=skipped_today)
    for block in today_blocks:
        if block.is_flexible:
            continue
        start = time_to_min(block.start_time)
        end = time_to_min(block.end_time)
        if is_overnight(block.start_time, block.end_time, block.is_night_shift):
            raw.append((start, MINUTES_PER_DAY))
        else:
            raw.append((start, end))

    previous_blocks = RecurringBlock.objects.filter(
        user=user,
        active=True,
        day_of_week=prev_dow,
    ).exclude(id__in=skipped_prev)
    for block in previous_blocks:
        if block.is_flexible:
            continue
        if is_overnight(block.start_time, block.end_time, block.is_night_shift):
            raw.append((0, time_to_min(block.end_time)))

    for block in ScheduledBlock.objects.filter(user=user, date=date):
        start = time_to_min(block.start_time)
        end = time_to_min(block.end_time)
        raw.append((start, MINUTES_PER_DAY) if end <= start else (start, end))

    clipped = [
        clipped_interval
        for start, end in raw
        if (clipped_interval := _clip_interval(start, end)) is not None
    ]
    return _merge_intervals(clipped)


def free_gaps(
    busy: list[tuple[int, int]],
    day_start: int,
    day_end: int,
) -> list[tuple[int, int]]:
    """Return the complement of ``busy`` within ``[day_start, day_end]``."""
    clipped_busy = [
        clipped_interval
        for start, end in busy
        if (clipped_interval := _clip_interval(start, end, day_start, day_end))
        is not None
    ]
    gaps: list[tuple[int, int]] = []
    cursor = day_start
    for start, end in _merge_intervals(clipped_busy):
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < day_end:
        gaps.append((cursor, day_end))
    return gaps


def _sort_key(result: dict) -> tuple[int, int, int]:
    if result["skipped"]:
        return 1, MINUTES_PER_DAY + 1, result["block_id"]
    return 0, result["start_min"], result["block_id"]


def place_day(user, date, day_start=0, day_end=MINUTES_PER_DAY) -> list[dict]:
    """Place flexible recurring blocks for one user/date.

    Algorithm:
    1. Compute fixed hard walls with :func:`fixed_busy_intervals`.
    2. Collect active, non-skipped flexible recurring blocks for the date's
       weekday.
    3. Keep v1 scope deliberately intra-day: flexible blocks whose stored
       preferred slot crosses midnight are not relocated. They are emitted at
       their stored time with ``preferred=True`` and ``overnight_kept=True``.
       This means sleep stored as 23:00-07:00 stays overnight for now, while the
       common 00:00-07:00 sleep shape is movable because it fits inside the day.
    4. Place the remaining intra-day flexible blocks by duration descending,
       then preferred start ascending, so large blocks reserve space first.
    5. For each block, try its preferred interval. If that is unavailable, use
       the fitting free gap whose start is nearest to the preferred start, with
       an earliest-gap tiebreak, and place the interval inside that gap as close
       to the preferred start as possible. If no gap fully fits, fill the
       largest gap and mark the placement as shrunk. If no gap exists, mark the
       block skipped.

    The function is deterministic and pure apart from ORM reads. It does not
    create, update, or delete schedule rows.
    """
    from core.models import RecurringBlock

    fixed = fixed_busy_intervals(user, date)
    taken = list(fixed)
    skipped_today = skipped_block_ids(user, date)

    flexible_blocks = [
        block
        for block in RecurringBlock.objects.filter(
            user=user,
            active=True,
            day_of_week=date.weekday(),
        ).exclude(id__in=skipped_today)
        if block.is_flexible
    ]

    overnight_results: list[dict] = []
    intraday_blocks = []
    for block in flexible_blocks:
        start = time_to_min(block.start_time)
        end = time_to_min(block.end_time)
        duration = block.effective_duration_minutes()
        if is_overnight(block.start_time, block.end_time, block.is_night_shift):
            overnight_results.append(
                _result(
                    block,
                    start_min=start,
                    end_min=end,
                    start_time=block.start_time.strftime("%H:%M"),
                    end_time=block.end_time.strftime("%H:%M"),
                    preferred=True,
                    shrunk=False,
                    skipped=False,
                    overnight_kept=True,
                )
            )
            todays_piece = _clip_interval(start, MINUTES_PER_DAY, day_start, day_end)
            if todays_piece is not None:
                taken.append(todays_piece)
            continue

        if duration <= 0 or duration > MINUTES_PER_DAY or start + duration > MINUTES_PER_DAY:
            overnight_results.append(
                _result(
                    block,
                    start_min=None,
                    end_min=None,
                    preferred=False,
                    shrunk=False,
                    skipped=True,
                    overnight_kept=False,
                )
            )
            continue

        intraday_blocks.append((block, duration, start))

    intraday_blocks.sort(key=lambda item: (-item[1], item[2], item[0].id))

    placed_results: list[dict] = []
    for block, duration, preferred_start in intraday_blocks:
        preferred_interval = (preferred_start, preferred_start + duration)
        if (
            preferred_interval[0] >= day_start
            and preferred_interval[1] <= day_end
            and not _overlaps(preferred_interval, taken)
        ):
            start, end = preferred_interval
            placed_results.append(
                _result(
                    block,
                    start_min=start,
                    end_min=end,
                    preferred=True,
                    shrunk=False,
                    skipped=False,
                    overnight_kept=False,
                )
            )
            taken.append((start, end))
            continue

        gaps = free_gaps(_merge_intervals(taken), day_start, day_end)
        fitting_gaps = [
            (start, end)
            for start, end in gaps
            if end - start >= duration
        ]
        if fitting_gaps:
            gap_start, gap_end = min(
                fitting_gaps,
                key=lambda gap: (abs(gap[0] - preferred_start), gap[0]),
            )
            start = max(
                gap_start,
                min(preferred_start, gap_end - duration),
            )
            end = start + duration
            placed_results.append(
                _result(
                    block,
                    start_min=start,
                    end_min=end,
                    preferred=False,
                    shrunk=False,
                    skipped=False,
                    overnight_kept=False,
                )
            )
            taken.append((start, end))
            continue

        if gaps:
            start, end = max(gaps, key=lambda gap: (gap[1] - gap[0], -gap[0]))
            placed_results.append(
                _result(
                    block,
                    start_min=start,
                    end_min=end,
                    preferred=False,
                    shrunk=True,
                    skipped=False,
                    overnight_kept=False,
                )
            )
            taken.append((start, end))
            continue

        placed_results.append(
            _result(
                block,
                start_min=None,
                end_min=None,
                preferred=False,
                shrunk=False,
                skipped=True,
                overnight_kept=False,
            )
        )

    return sorted(overnight_results + placed_results, key=_sort_key)
