"""
Regression tests for GROUP T1 (models + migration).

Covers:
  - B23: UserProfile.energy_levels / notification_preferences default to {}
    and persist arbitrary JSON.
  - D1: composite indexes exist on hot query paths; CheckConstraints exist for
    Task.priority (1..10) and Goal.progress (0..100) and are enforced at the DB
    level when the save()-clamp is bypassed.
  - D9: Task.mark_completed / ScheduledBlock.mark_completed are the single
    source of truth reconciling completed / completed_at / actually_completed /
    TaskHistory, and are idempotent.
"""
import datetime

import pytest
from django.db import IntegrityError, transaction
from django.utils import timezone

from core.models import (
    ConversationMessage,
    Goal,
    RecurringBlock,
    ScheduledBlock,
    Task,
    TaskHistory,
    UserProfile,
)


# ---------------------------------------------------------------------------
# B23 - UserProfile JSON fields
# ---------------------------------------------------------------------------

def test_energy_and_notification_defaults_empty_dict(user):
    profile = user.profile
    assert profile.energy_levels == {}
    assert profile.notification_preferences == {}


def test_energy_and_notification_persist(user):
    profile = user.profile
    profile.energy_levels = {'morning': 'high', 'evening': 'low'}
    profile.notification_preferences = {'email': True, 'quiet_hours': [22, 7]}
    profile.save()

    reloaded = UserProfile.objects.get(pk=profile.pk)
    assert reloaded.energy_levels == {'morning': 'high', 'evening': 'low'}
    assert reloaded.notification_preferences == {'email': True, 'quiet_hours': [22, 7]}


# ---------------------------------------------------------------------------
# D1 - composite indexes
# ---------------------------------------------------------------------------

def _index_names(model):
    return {idx.name for idx in model._meta.indexes}


def test_hot_path_indexes_exist():
    assert 'schedblock_user_date_idx' in _index_names(ScheduledBlock)
    assert 'recurblock_user_day_active_idx' in _index_names(RecurringBlock)
    assert 'convmsg_user_created_idx' in _index_names(ConversationMessage)

    sched = next(i for i in ScheduledBlock._meta.indexes if i.name == 'schedblock_user_date_idx')
    assert sched.fields == ['user', 'date']

    recur = next(i for i in RecurringBlock._meta.indexes if i.name == 'recurblock_user_day_active_idx')
    assert recur.fields == ['user', 'day_of_week', 'active']

    conv = next(i for i in ConversationMessage._meta.indexes if i.name == 'convmsg_user_created_idx')
    assert conv.fields == ['user', 'created_at']


# ---------------------------------------------------------------------------
# D1 - CheckConstraints
# ---------------------------------------------------------------------------

def _constraint_names(model):
    return {c.name for c in model._meta.constraints}


def test_priority_and_progress_constraints_declared():
    assert 'task_priority_between_1_and_10' in _constraint_names(Task)
    assert 'goal_progress_between_0_and_100' in _constraint_names(Goal)


def test_task_priority_constraint_enforced_at_db(user):
    # bulk_create bypasses Task.save()'s clamp, so an out-of-range priority
    # reaches the database and must be rejected by the CheckConstraint.
    with pytest.raises(IntegrityError):
        with transaction.atomic():
            Task.objects.bulk_create([Task(user=user, title='bad', priority=999)])


def test_task_priority_zero_rejected_at_db(user):
    with pytest.raises(IntegrityError):
        with transaction.atomic():
            Task.objects.bulk_create([Task(user=user, title='bad-low', priority=0)])


def test_goal_progress_constraint_enforced_at_db(user):
    with pytest.raises(IntegrityError):
        with transaction.atomic():
            Goal.objects.bulk_create([Goal(user=user, title='bad', progress=500)])


def test_valid_priority_and_progress_accepted(user):
    Task.objects.bulk_create([Task(user=user, title='ok', priority=7)])
    Goal.objects.bulk_create([Goal(user=user, title='ok', progress=42)])
    assert Task.objects.filter(title='ok', priority=7).exists()
    assert Goal.objects.filter(title='ok', progress=42).exists()


# ---------------------------------------------------------------------------
# D9 - mark_completed reconciliation
# ---------------------------------------------------------------------------

def _make_task_with_block(user, **task_kwargs):
    task = Task.objects.create(
        user=user,
        title=task_kwargs.pop('title', 'Write report'),
        task_type=task_kwargs.pop('task_type', 'deep_work'),
        estimated_duration_minutes=task_kwargs.pop('estimated_duration_minutes', 60),
        **task_kwargs,
    )
    block = ScheduledBlock.objects.create(
        user=user,
        task=task,
        date=datetime.date(2026, 7, 20),  # a Monday
        start_time=datetime.time(9, 0),
        end_time=datetime.time(10, 0),
    )
    return task, block


def test_task_mark_completed_syncs_block_and_history(user):
    task, block = _make_task_with_block(user)

    task.mark_completed(actual_minutes=75)

    task.refresh_from_db()
    block.refresh_from_db()

    assert task.completed is True
    assert task.completed_at is not None

    assert block.actually_completed is True
    assert block.actual_duration_minutes == 75

    histories = TaskHistory.objects.filter(user=user, task_title='Write report')
    assert histories.count() == 1
    h = histories.first()
    assert h.actual_duration_minutes == 75
    assert h.task_type == 'deep_work'
    assert h.estimated_duration_minutes == 60
    assert h.day_of_week == 0  # Monday
    assert h.scheduled_start_time == datetime.time(9, 0)
    assert h.completed_at == task.completed_at


def test_task_mark_completed_is_idempotent(user):
    task, block = _make_task_with_block(user)

    task.mark_completed(actual_minutes=50)
    first_completed_at = Task.objects.get(pk=task.pk).completed_at

    # call again on a fresh instance
    Task.objects.get(pk=task.pk).mark_completed(actual_minutes=999)

    assert TaskHistory.objects.filter(user=user, task_title='Write report').count() == 1
    # completed_at is preserved, not overwritten
    assert Task.objects.get(pk=task.pk).completed_at == first_completed_at


def test_scheduledblock_mark_completed_reconciles_task(user):
    task, block = _make_task_with_block(user, title='Study math')

    block.mark_completed(actual_minutes=40)

    task.refresh_from_db()
    block.refresh_from_db()

    assert block.actually_completed is True
    assert block.actual_duration_minutes == 40
    assert task.completed is True
    assert task.completed_at is not None
    assert TaskHistory.objects.filter(user=user, task_title='Study math').count() == 1


def test_mark_completed_without_actual_minutes_falls_back_to_estimate(user):
    task, block = _make_task_with_block(user, title='Read paper', estimated_duration_minutes=30)

    task.mark_completed()

    h = TaskHistory.objects.get(user=user, task_title='Read paper')
    assert h.actual_duration_minutes == 30
    block.refresh_from_db()
    assert block.actually_completed is True


def test_mark_completed_without_block_uses_completed_at_weekday(user):
    task = Task.objects.create(user=user, title='Standalone', task_type='shallow')
    task.mark_completed(actual_minutes=15)

    h = TaskHistory.objects.get(user=user, task_title='Standalone')
    assert h.actual_duration_minutes == 15
    assert h.scheduled_start_time is None
    expected_dow = timezone.localtime(task.completed_at).weekday()
    assert h.day_of_week == expected_dow
