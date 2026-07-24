from datetime import date, time, timedelta

import pytest
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APIClient

from core.models import RecurringBlock, ScheduledBlock, Task, TaskHistory, UserPlace
from services.ai_scheduler import AIScheduler
from services.agent.tools.tasks import CompleteTaskTool
from services.agent.tools.schedule import ScheduleTaskAtTool
from services.ai_insights import AIInsightsService
from services.replan import replan_after_delay
from services.scheduling.placement import fixed_busy_intervals, place_day


MONDAY = date(2026, 7, 27)


def _profile(user, *, prep=15, margin=10):
    profile = user.profile
    profile.prep_time_minutes = prep
    profile.safety_margin_minutes = margin
    profile.transport_time_minutes = 0
    profile.min_sleep_hours = 8
    profile.automation_mode = "automatic"
    profile.save()
    return profile


def _block(user, title, block_type, dow, start, end, **kwargs):
    return RecurringBlock.objects.create(
        user=user,
        title=title,
        block_type=block_type,
        day_of_week=dow,
        start_time=start,
        end_time=end,
        **kwargs,
    )


def _placement_by_title(placements, title):
    return next(p for p in placements if p["title"] == title)


@pytest.mark.django_db
def test_flexible_recurring_placement_reserves_fixed_block_commute_windows():
    user = User.objects.create_user("qa-commute-placement", password="pw")
    _profile(user)
    work = UserPlace.objects.create(
        user=user, name="Travail", kind="work", travel_minutes=32
    )
    _block(user, "Cours", "course", MONDAY.weekday(), time(8), time(15, 30))
    _block(
        user,
        "Travail",
        "work",
        MONDAY.weekday(),
        time(18),
        time(21),
        place=work,
    )
    _block(user, "Lecture", "other", MONDAY.weekday(), time(16, 30), time(17, 30))
    _block(user, "Sport", "sport", MONDAY.weekday(), time(21), time(22))

    placements = place_day(user, MONDAY)
    lecture = _placement_by_title(placements, "Lecture")
    sport = _placement_by_title(placements, "Sport")

    assert lecture["end_time"] == "17:03"
    assert sport["start_time"] == "21:32"


@pytest.mark.django_db
def test_scheduled_task_place_commute_counts_as_busy_time():
    user = User.objects.create_user("qa-task-commute", password="pw")
    _profile(user, prep=10, margin=5)
    place = UserPlace.objects.create(
        user=user, name="Clinique", kind="other", travel_minutes=20
    )
    task = Task.objects.create(user=user, title="Rdv", place=place)
    ScheduledBlock.objects.create(
        user=user,
        task=task,
        date=MONDAY,
        start_time=time(12),
        end_time=time(13),
    )

    assert fixed_busy_intervals(user, MONDAY) == [(11 * 60 + 25, 13 * 60 + 20)]


@pytest.mark.django_db
def test_overnight_sleep_does_not_trigger_post_night_shift_recovery():
    user = User.objects.create_user("qa-sleep-not-night-work", password="pw")
    _profile(user)
    _block(
        user,
        "Sommeil",
        "sleep",
        MONDAY.weekday(),
        time(23),
        time(7),
        is_night_shift=True,
    )
    tuesday = MONDAY + timedelta(days=1)

    slots = AIScheduler()._get_available_slots(user, tuesday, 1)

    assert slots
    assert slots[0].start_time == time(8)


@pytest.mark.django_db
def test_recurring_block_rest_api_rejects_overlapping_fixed_create():
    user = User.objects.create_user("qa-api-fixed-overlap", password="pw")
    client = APIClient()
    client.force_authenticate(user)

    first = client.post(
        reverse("recurring-block-list"),
        {
            "title": "Cours",
            "block_type": "course",
            "day_of_week": 0,
            "start_time": "09:00",
            "end_time": "11:00",
        },
        format="json",
    )
    assert first.status_code == status.HTTP_201_CREATED

    second = client.post(
        reverse("recurring-block-list"),
        {
            "title": "Travail",
            "block_type": "work",
            "day_of_week": 0,
            "start_time": "10:00",
            "end_time": "12:00",
        },
        format="json",
    )

    assert second.status_code == status.HTTP_400_BAD_REQUEST
    assert RecurringBlock.objects.filter(user=user).count() == 1


@pytest.mark.django_db
def test_recurring_block_rest_api_rejects_overlapping_fixed_update():
    user = User.objects.create_user("qa-api-fixed-update", password="pw")
    client = APIClient()
    client.force_authenticate(user)
    _block(user, "Cours", "course", 0, time(9), time(11))
    work = _block(user, "Travail", "work", 0, time(12), time(13))

    response = client.patch(
        reverse("recurring-block-detail", args=[work.id]),
        {"start_time": "10:00", "end_time": "12:00"},
        format="json",
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    work.refresh_from_db()
    assert work.start_time == time(12)


@pytest.mark.django_db
def test_scheduled_block_patch_rejects_overlap_with_recurring_wall():
    user = User.objects.create_user("qa-patch-overlap", password="pw")
    client = APIClient()
    client.force_authenticate(user)
    _block(user, "Cours", "course", MONDAY.weekday(), time(9), time(11))
    task = Task.objects.create(user=user, title="Lecture")
    scheduled = ScheduledBlock.objects.create(
        user=user,
        task=task,
        date=MONDAY,
        start_time=time(12),
        end_time=time(13),
    )

    response = client.patch(
        reverse("schedule-block", args=[scheduled.id]),
        {"start_time": "10:00", "end_time": "11:00"},
        format="json",
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    scheduled.refresh_from_db()
    assert scheduled.start_time == time(12)


@pytest.mark.django_db
def test_complete_task_tool_reconciles_scheduled_block_and_history():
    user = User.objects.create_user("qa-complete-tool", password="pw")
    task = Task.objects.create(user=user, title="Lecture")
    scheduled = ScheduledBlock.objects.create(
        user=user,
        task=task,
        date=timezone.localdate(),
        start_time=time(9),
        end_time=time(10),
    )

    result = CompleteTaskTool().execute(user, task_id=task.id)

    assert result.success
    task.refresh_from_db()
    scheduled.refresh_from_db()
    assert task.completed
    assert scheduled.actually_completed
    assert TaskHistory.objects.filter(user=user, task_title="Lecture").count() == 1


@pytest.mark.django_db
def test_replan_does_not_resurrect_task_completed_with_stale_block_flag():
    user = User.objects.create_user("qa-stale-completion-replan", password="pw")
    _profile(user)
    today = timezone.localdate()
    task = Task.objects.create(user=user, title="Déjà fait", completed=True)
    ScheduledBlock.objects.create(
        user=user,
        task=task,
        date=today,
        start_time=time(9),
        end_time=time(10),
        actually_completed=False,
    )

    result = replan_after_delay(user, resume_time="11:00")

    assert result["moved"] == []
    assert ScheduledBlock.objects.filter(
        user=user, task=task, start_time=time(9)
    ).exists()


@pytest.mark.django_db
def test_scheduler_does_not_shrink_indivisible_task_into_partial_gap():
    user = User.objects.create_user("qa-no-partial-task", password="pw")
    _profile(user)
    _block(user, "Wall 1", "work", MONDAY.weekday(), time(8, 30), time(9))
    _block(user, "Wall 2", "work", MONDAY.weekday(), time(9, 30), time(22))
    task = Task.objects.create(
        user=user,
        title="Indivisible 1h",
        estimated_duration_minutes=60,
    )

    scheduler = AIScheduler()
    created = scheduler.generate_schedule(
        user, tasks=[task], start_date=MONDAY, num_days=1
    )

    assert created == []
    assert scheduler.last_unplaced
    assert scheduler.last_unplaced[0]["largest_free_slot_minutes"] == 30


@pytest.mark.django_db
def test_sleep_minimum_is_not_shrunk_into_shorter_gap():
    user = User.objects.create_user("qa-no-shrunk-sleep", password="pw")
    _profile(user)
    _block(user, "Long wall", "work", MONDAY.weekday(), time(0), time(17))
    _block(user, "Sommeil", "sleep", MONDAY.weekday(), time(0), time(8))

    sleep = _placement_by_title(place_day(user, MONDAY), "Sommeil")

    assert sleep["skipped"]
    assert sleep["start_time"] is None
    assert sleep["end_time"] is None


@pytest.mark.django_db
def test_schedule_task_at_can_displace_flexible_recurring_block():
    user = User.objects.create_user("qa-urgent-over-flex", password="pw")
    _profile(user)
    _block(user, "Sport", "sport", MONDAY.weekday(), time(10), time(12))

    result = ScheduleTaskAtTool().execute(
        user,
        title="Urgence fixe",
        date=MONDAY.isoformat(),
        start_time="10:30",
        end_time="11:00",
    )

    assert result.success, result.message
    sport = _placement_by_title(place_day(user, MONDAY), "Sport")
    assert not (sport["start_min"] < 11 * 60 and 10 * 60 + 30 < sport["end_min"])


@pytest.mark.django_db
def test_natural_language_contradictory_preference_reports_unplaced_reason():
    user = User.objects.create_user("qa-nl-contradiction", password="pw")
    _profile(user)
    _block(user, "Travail soir", "work", 0, time(18), time(22))

    result = AIInsightsService().execute_scheduling_request(user, {
        "action": "schedule",
        "task_title": "Sport soir",
        "duration_minutes": 60,
        "days": [0],
        "preferred_time": "evening",
    })

    assert result["status"] == "unplaced"
    assert result["unplaced"]
    assert "evening" in result["unplaced"][0]["reason"]


@pytest.mark.django_db
def test_natural_language_missing_duration_reports_visible_assumption():
    user = User.objects.create_user("qa-nl-assumption", password="pw")
    _profile(user)

    result = AIInsightsService().execute_scheduling_request(user, {
        "action": "schedule",
        "task_title": "Lire",
        "days": [MONDAY.weekday()],
    })

    assert result["assumptions"]
    assert result["assumptions"][0]["field"] == "duration_minutes"
