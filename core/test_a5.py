"""
Regression tests for AI Insights bugs B13-B16 (group A5).

All tests are offline: they never hit the Gemini API. The AIInsightsService
constructor only builds a genai client when GEMINI_API_KEY is set, and none of
these tests exercise the LLM code paths.
"""
import datetime as dt

import pytest
from django.utils import timezone

from core.models import Task, ScheduledBlock, TaskHistory
from services.ai_insights import AIInsightsService


@pytest.fixture
def service():
    return AIInsightsService()


def _make_history(user, **overrides):
    defaults = dict(
        user=user,
        task_title='Etude maths',
        task_type='deep_work',
        estimated_duration_minutes=60,
        actual_duration_minutes=55,
        scheduled_start_time=dt.time(9, 0),
        day_of_week=0,
        completed_at=timezone.now(),
        energy_level='high',
    )
    defaults.update(overrides)
    return TaskHistory.objects.create(**defaults)


# ==================== B15: predict_duration on whitespace title ====================

@pytest.mark.django_db
def test_predict_duration_whitespace_title_does_not_raise(service, user):
    """A whitespace-only title is truthy but split() is empty -> must not IndexError."""
    result = service.predict_duration(user, '   ', 'deep_work')
    assert isinstance(result, dict)
    assert 'predicted_minutes' in result
    # No history -> default estimate for deep_work
    assert result['predicted_minutes'] == 90


@pytest.mark.django_db
def test_predict_duration_empty_title_does_not_raise(service, user):
    result = service.predict_duration(user, '', 'shallow')
    assert result['predicted_minutes'] == 30


@pytest.mark.django_db
def test_predict_duration_tab_only_title_does_not_raise(service, user):
    result = service.predict_duration(user, '\t\n ', 'errand')
    assert result['predicted_minutes'] == 45


# ==================== B15: patterns ZeroDivision when actual=0 ====================

@pytest.mark.django_db
def test_best_time_for_task_type_actual_zero_does_not_raise(service, user):
    """actual_duration_minutes == 0 must not trigger ZeroDivisionError."""
    _make_history(
        user,
        task_type='deep_work',
        estimated_duration_minutes=60,
        actual_duration_minutes=0,
        scheduled_start_time=dt.time(10, 0),
    )
    history = TaskHistory.objects.filter(user=user)
    result = service._get_best_time_for_task_type(history, 'deep_work')
    assert isinstance(result, dict)
    assert result.get('hour') == 10


@pytest.mark.django_db
def test_analyze_patterns_with_zero_actual_row_does_not_raise(service, user):
    """Full pattern analysis with an actual=0 row present must not raise."""
    # Need >= 5 history rows for analyze_user_patterns to run.
    _make_history(user, actual_duration_minutes=0, estimated_duration_minutes=60,
                  scheduled_start_time=dt.time(9, 0))
    for i in range(4):
        _make_history(user, actual_duration_minutes=50 + i,
                      scheduled_start_time=dt.time(9, 0))
    patterns = service.analyze_user_patterns(user)
    assert patterns.get('status') != 'insufficient_data'
    assert 'best_time_for_deep_work' in patterns


# ==================== B13/B14: smart_reschedule persistence + midnight ====================

@pytest.mark.django_db
def test_smart_reschedule_persists_shift(service, user):
    """A within-day shift must be persisted to the DB (block.save)."""
    task = Task.objects.create(user=user, title='Task', task_type='shallow')
    today = timezone.localdate()
    overflowed = ScheduledBlock.objects.create(
        user=user, task=task, date=today,
        start_time=dt.time(10, 0), end_time=dt.time(11, 0),
    )
    affected = ScheduledBlock.objects.create(
        user=user, task=task, date=today,
        start_time=dt.time(11, 0), end_time=dt.time(12, 0),
    )

    result = service.smart_reschedule(user, overflowed.id, dt.time(11, 30))
    assert result['status'] == 'rescheduled'

    affected.refresh_from_db()
    assert affected.start_time == dt.time(11, 30)
    assert affected.end_time == dt.time(12, 30)


@pytest.mark.django_db
def test_smart_reschedule_move_to_next_day_persists(service, user):
    """B13: 'move_to_next_day' must actually persist the new date, not just report it."""
    task = Task.objects.create(user=user, title='LateTask', task_type='shallow')
    today = timezone.localdate()
    overflowed = ScheduledBlock.objects.create(
        user=user, task=task, date=today,
        start_time=dt.time(22, 0), end_time=dt.time(22, 30),
    )
    # Affected block ends late enough that a shift pushes it past midnight.
    affected = ScheduledBlock.objects.create(
        user=user, task=task, date=today,
        start_time=dt.time(22, 45), end_time=dt.time(23, 45),
    )

    result = service.smart_reschedule(user, overflowed.id, dt.time(23, 0))
    assert result['status'] == 'rescheduled'
    actions = [r['action'] for r in result['rescheduled']]
    assert 'move_to_next_day' in actions

    affected.refresh_from_db()
    # Persisted onto the next calendar day.
    assert affected.date == today + dt.timedelta(days=1)


@pytest.mark.django_db
def test_smart_reschedule_never_saves_end_before_start(service, user):
    """B14: no persisted block may have end_time < start_time (midnight rollover)."""
    task = Task.objects.create(user=user, title='RollTask', task_type='shallow')
    today = timezone.localdate()
    overflowed = ScheduledBlock.objects.create(
        user=user, task=task, date=today,
        start_time=dt.time(22, 0), end_time=dt.time(22, 30),
    )
    affected = ScheduledBlock.objects.create(
        user=user, task=task, date=today,
        start_time=dt.time(22, 45), end_time=dt.time(23, 45),
    )

    service.smart_reschedule(user, overflowed.id, dt.time(23, 0))

    for block in ScheduledBlock.objects.all():
        assert block.end_time >= block.start_time, (
            f"Block {block.id} saved with end {block.end_time} < start {block.start_time}"
        )


# ==================== B16: detect_conflicts clamps days_ahead ====================

@pytest.mark.django_db
def test_detect_conflicts_clamps_days_ahead(service, user):
    """A huge days_ahead must be clamped so a far-future deadline is excluded."""
    now = timezone.now()
    near = Task.objects.create(
        user=user, title='NearDeadline', task_type='shallow',
        completed=False, deadline=now + dt.timedelta(days=89),
    )
    far = Task.objects.create(
        user=user, title='FarDeadline', task_type='shallow',
        completed=False, deadline=now + dt.timedelta(days=200),
    )

    conflicts = service.detect_conflicts(user, days_ahead=100000)
    messages = ' '.join(c.message for c in conflicts)

    # Within the clamp (90 days): near deadline surfaces as a risk.
    assert 'NearDeadline' in messages
    # Beyond the clamp: far deadline must NOT surface (proves days_ahead was clamped).
    assert 'FarDeadline' not in messages


@pytest.mark.django_db
def test_detect_conflicts_negative_days_ahead_is_safe(service, user):
    """A non-positive / non-int days_ahead must not crash."""
    assert isinstance(service.detect_conflicts(user, days_ahead=0), list)
    assert isinstance(service.detect_conflicts(user, days_ahead=-5), list)
