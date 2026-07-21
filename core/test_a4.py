"""
Regression tests for group A4 (deterministic AI scheduler).

Covers:
- B10: day boundaries / deadline buckets computed in local time (Europe/Paris),
       and a reference date that stays consistent within a single run.
- B11: the deadline TIME-of-day is respected, not only the date.
- B12: profile.max_deep_work_hours_per_day is enforced by the deterministic
       placement, not only in the (dead) Gemini prompt.

All offline: the AIScheduler is exercised directly, no LLM call.
"""
from datetime import datetime, time, timedelta

import pytest
from django.utils import timezone

from core.models import RecurringBlock, ScheduledBlock, Task
from services.ai_scheduler import AIScheduler


def _local_dt(d, hour, minute=0):
    """Aware datetime at the given local date/time (Europe/Paris)."""
    naive = datetime.combine(d, time(hour, minute))
    return timezone.make_aware(naive, timezone.get_current_timezone())


def _overlaps(a_start, a_end, b_start, b_end):
    return not (a_end <= b_start or a_start >= b_end)


@pytest.mark.django_db
def test_deep_work_task_due_today_not_placed_after_deadline_time(user):
    """B11: a task due today at 09:00 must not land after 09:00 the same day.

    We block the morning after 09:15 with a recurring block and expose a tempting
    high-energy afternoon window. The buggy scheduler (date-only deadline) would
    pick the afternoon slot; the fixed one keeps the placement before 09:00.
    """
    today = timezone.localdate()
    # Afternoon = high energy 13-17, low energy at 08:00 -> afternoon slot is
    # more attractive to the energy scorer, which is exactly the trap for B11.
    profile = user.profile
    profile.peak_productivity_time = 'afternoon'
    profile.max_deep_work_hours_per_day = 8
    profile.transport_time_minutes = 0
    profile.save()

    # Blocks the day from 09:15 to 13:00 -> free slots: 08:00-09:15 and 13:00-22:00
    RecurringBlock.objects.create(
        user=user,
        title='Cours',
        block_type='course',
        day_of_week=today.weekday(),
        start_time=time(9, 15),
        end_time=time(13, 0),
        active=True,
    )

    task = Task.objects.create(
        user=user,
        title='Deep work due at 9',
        task_type='deep_work',
        priority=8,
        estimated_duration_minutes=60,
        deadline=_local_dt(today, 9, 0),
    )

    scheduler = AIScheduler()
    blocks = scheduler.generate_schedule(user, tasks=[task], start_date=today, num_days=3)

    assert blocks, "task should be scheduled"
    today_blocks = [b for b in blocks if b.date == today]
    assert today_blocks, "on-time slot before the deadline should be used"
    for b in today_blocks:
        assert b.start_time <= time(9, 0), (
            f"block for a task due today 09:00 placed at {b.start_time} (after deadline)"
        )


@pytest.mark.django_db
def test_deep_work_minutes_per_day_capped(user):
    """B12: total deep-work minutes placed on any day never exceed the cap."""
    today = timezone.localdate()
    profile = user.profile
    profile.max_deep_work_hours_per_day = 2  # 120 minutes / day
    profile.peak_productivity_time = 'morning'
    profile.transport_time_minutes = 0
    profile.save()

    # 6 deep-work tasks of 90 min each, no deadline. Only 1 fits per day under a
    # 120-min cap (a 2nd would total 180 > 120).
    tasks = []
    for i in range(6):
        tasks.append(Task.objects.create(
            user=user,
            title=f'Deep {i}',
            task_type='deep_work',
            priority=5,
            estimated_duration_minutes=90,
        ))

    scheduler = AIScheduler()
    blocks = scheduler.generate_schedule(user, tasks=tasks, start_date=today, num_days=7)

    assert blocks, "some tasks should be scheduled"
    minutes_by_date = {}
    for b in blocks:
        placed = (b.end_time.hour * 60 + b.end_time.minute) - (
            b.start_time.hour * 60 + b.start_time.minute
        )
        minutes_by_date[b.date] = minutes_by_date.get(b.date, 0) + placed

    for d, mins in minutes_by_date.items():
        assert mins <= 120, f"{mins} deep-work minutes placed on {d}, cap is 120"


@pytest.mark.django_db
def test_generated_blocks_never_overlap_recurring_block(user):
    """A generated block must never overlap an active recurring block."""
    today = timezone.localdate()
    profile = user.profile
    profile.peak_productivity_time = 'morning'
    profile.transport_time_minutes = 0
    profile.max_deep_work_hours_per_day = 8
    profile.save()

    # A recurring block on every day of the week 10:00-14:00.
    for dow in range(7):
        RecurringBlock.objects.create(
            user=user,
            title='Work',
            block_type='work',
            day_of_week=dow,
            start_time=time(10, 0),
            end_time=time(14, 0),
            active=True,
        )

    tasks = [
        Task.objects.create(
            user=user,
            title=f'Task {i}',
            task_type='deep_work' if i % 2 == 0 else 'shallow',
            priority=5,
            estimated_duration_minutes=120,
        )
        for i in range(5)
    ]

    scheduler = AIScheduler()
    blocks = scheduler.generate_schedule(user, tasks=tasks, start_date=today, num_days=7)

    assert blocks, "tasks should be scheduled around the recurring blocks"
    for b in blocks:
        rec = RecurringBlock.objects.filter(user=user, day_of_week=b.date.weekday())
        for r in rec:
            assert not _overlaps(
                b.start_time, b.end_time, r.start_time, r.end_time
            ), (
                f"generated block {b.date} {b.start_time}-{b.end_time} overlaps "
                f"recurring {r.start_time}-{r.end_time}"
            )


@pytest.mark.django_db
def test_reference_date_consistent_within_run(user):
    """B10: start_date defaults to the local date and drives scoring/placement.

    With an explicit start_date, every generated block falls within the
    requested horizon (no off-by-one day from a UTC vs local mismatch).
    """
    today = timezone.localdate()
    profile = user.profile
    profile.transport_time_minutes = 0
    profile.save()

    tasks = [
        Task.objects.create(
            user=user,
            title=f'T{i}',
            task_type='shallow',
            priority=5,
            estimated_duration_minutes=60,
        )
        for i in range(3)
    ]

    scheduler = AIScheduler()
    blocks = scheduler.generate_schedule(user, tasks=tasks, start_date=today, num_days=3)

    assert blocks
    for b in blocks:
        assert today <= b.date < today + timedelta(days=3)
