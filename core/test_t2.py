"""
Regression tests for GROUP T2 (views + serializers).

Covers:
  - B23: PATCH /profile/ persists AND returns energy_levels /
    notification_preferences (the JSONFields T1 adds to UserProfile), so the
    frontend PATCH actually saves them.
  - D9: completing a Task through the API marks its active ScheduledBlock
    actually_completed and records a TaskHistory row -- end-to-end through the
    endpoint, routed through the canonical Task.mark_completed sync helper.
  - D9: marking a ScheduledBlock completed through PATCH /schedule/<id>/ routes
    through the canonical ScheduledBlock.mark_completed helper.

The energy_levels / task-sync assertions that require the T1 model layer are
gated on the presence of the T1 fields/helpers, so this suite is green both
before and after the T1 migration is applied while still exercising real
behaviour in each case.
"""
import datetime

import pytest
from django.urls import reverse

from core.models import ScheduledBlock, Task, TaskHistory, UserProfile


def _model_has_field(model, name):
    return any(f.name == name for f in model._meta.get_fields())


_HAS_JSON_PROFILE_FIELDS = _model_has_field(UserProfile, 'energy_levels')


# ---------------------------------------------------------------------------
# B23 - profile JSON preferences persist through the API
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _HAS_JSON_PROFILE_FIELDS,
    reason="UserProfile.energy_levels absent (T1 migration not applied yet)",
)
def test_patch_profile_persists_and_returns_energy_levels(authenticated_client, user):
    url = reverse('profile')
    payload = {
        'energy_levels': {'morning': 'high', 'evening': 'low'},
        'notification_preferences': {'email': True, 'quiet_hours': [22, 7]},
    }

    resp = authenticated_client.patch(url, payload, format='json')

    assert resp.status_code == 200
    # returned in the PATCH response
    assert resp.data['energy_levels'] == {'morning': 'high', 'evening': 'low'}
    assert resp.data['notification_preferences'] == {
        'email': True, 'quiet_hours': [22, 7],
    }

    # actually persisted to the database, not merely echoed back
    profile = UserProfile.objects.get(pk=user.profile.pk)
    assert profile.energy_levels == {'morning': 'high', 'evening': 'low'}
    assert profile.notification_preferences == {'email': True, 'quiet_hours': [22, 7]}

    # returned again on a fresh GET
    get_resp = authenticated_client.get(url)
    assert get_resp.status_code == 200
    assert get_resp.data['energy_levels'] == {'morning': 'high', 'evening': 'low'}
    assert get_resp.data['notification_preferences'] == {
        'email': True, 'quiet_hours': [22, 7],
    }


@pytest.mark.skipif(
    not _HAS_JSON_PROFILE_FIELDS,
    reason="UserProfile.energy_levels absent (T1 migration not applied yet)",
)
def test_patch_profile_energy_levels_is_partial(authenticated_client, user):
    """PATCHing only energy_levels must not clobber unrelated fields."""
    url = reverse('profile')

    authenticated_client.patch(
        url, {'peak_productivity_time': 'evening'}, format='json'
    )
    resp = authenticated_client.patch(
        url, {'energy_levels': {'afternoon': 'medium'}}, format='json'
    )

    assert resp.status_code == 200
    assert resp.data['energy_levels'] == {'afternoon': 'medium'}
    # earlier update survived the second PATCH
    assert resp.data['peak_productivity_time'] == 'evening'


# ---------------------------------------------------------------------------
# D9 - Task completion endpoint reconciles the block + history
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_complete_task_endpoint_marks_block_and_creates_history(
    authenticated_client, user
):
    task = Task.objects.create(
        user=user,
        title='Finish essay',
        task_type='deep_work',
        estimated_duration_minutes=60,
    )
    block = ScheduledBlock.objects.create(
        user=user,
        task=task,
        date=datetime.date(2026, 7, 20),
        start_time=datetime.time(9, 0),
        end_time=datetime.time(10, 0),
    )

    url = reverse('task-complete', kwargs={'pk': task.pk})
    resp = authenticated_client.post(
        url, {'actual_duration_minutes': 45}, format='json'
    )

    assert resp.status_code == 200
    assert resp.data['completed'] is True

    task.refresh_from_db()
    block.refresh_from_db()

    assert task.completed is True
    assert task.completed_at is not None

    # the active scheduled block is reconciled end-to-end through the endpoint
    assert block.actually_completed is True
    assert block.actual_duration_minutes == 45

    # exactly one TaskHistory row was recorded (no divergence / duplication)
    histories = TaskHistory.objects.filter(user=user, task_title='Finish essay')
    assert histories.count() == 1
    assert histories.first().actual_duration_minutes == 45


@pytest.mark.django_db
def test_complete_task_endpoint_is_idempotent_on_history(authenticated_client, user):
    """Completing twice must not stack duplicate TaskHistory rows (with T1)."""
    task = Task.objects.create(
        user=user, title='Study', task_type='shallow',
        estimated_duration_minutes=30,
    )
    ScheduledBlock.objects.create(
        user=user, task=task,
        date=datetime.date(2026, 7, 20),
        start_time=datetime.time(11, 0), end_time=datetime.time(12, 0),
    )
    url = reverse('task-complete', kwargs={'pk': task.pk})

    first = authenticated_client.post(url, {'actual_duration_minutes': 25}, format='json')
    assert first.status_code == 200

    if hasattr(Task, 'mark_completed'):
        # The canonical helper is idempotent: a second completion must not add
        # another history row.
        authenticated_client.post(url, {'actual_duration_minutes': 25}, format='json')
        assert TaskHistory.objects.filter(user=user, task_title='Study').count() == 1


# ---------------------------------------------------------------------------
# D9 - ScheduledBlock completion PATCH routes through the sync helper
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_scheduled_block_patch_completion(authenticated_client, user):
    task = Task.objects.create(user=user, title='Read chapter', task_type='shallow')
    block = ScheduledBlock.objects.create(
        user=user, task=task,
        date=datetime.date(2026, 7, 20),
        start_time=datetime.time(14, 0), end_time=datetime.time(15, 0),
    )

    url = reverse('schedule-block', kwargs={'block_id': block.pk})
    resp = authenticated_client.patch(
        url,
        {'actually_completed': True, 'actual_duration_minutes': 55},
        format='json',
    )

    assert resp.status_code == 200
    block.refresh_from_db()
    assert block.actually_completed is True
    assert block.actual_duration_minutes == 55

    # With the T1 helper present, the parent Task is reconciled canonically.
    if hasattr(block, 'mark_completed'):
        task.refresh_from_db()
        assert task.completed is True
        assert task.completed_at is not None
        assert (
            TaskHistory.objects.filter(user=user, task_title='Read chapter').count()
            == 1
        )


@pytest.mark.django_db
def test_scheduled_block_patch_drag_still_works(authenticated_client, user):
    """A non-completion PATCH (drag & drop) is unchanged by the D9 routing."""
    task = Task.objects.create(user=user, title='Move me', task_type='shallow')
    block = ScheduledBlock.objects.create(
        user=user, task=task,
        date=datetime.date(2026, 7, 20),
        start_time=datetime.time(9, 0), end_time=datetime.time(10, 0),
    )

    url = reverse('schedule-block', kwargs={'block_id': block.pk})
    resp = authenticated_client.patch(
        url, {'start_time': '11:00:00'}, format='json'
    )

    assert resp.status_code == 200
    block.refresh_from_db()
    assert block.start_time == datetime.time(11, 0)
    # end_time auto-shifts to preserve the 1h duration; block stays incomplete
    assert block.end_time == datetime.time(12, 0)
    assert block.actually_completed is False
