"""Tests for act-first scheduling: one-off dated placement + overnight free slots.

- ScheduleTaskAtTool places a one-off dated timed ScheduledBlock (locked),
  rejects overlaps (incl. night-shift work) and end<=start.
- Free-slot tools are overnight-aware: a Sat 19:00-07:00 shift occupies the
  evening (no more false "libre 7h-23h").
"""
from datetime import date, time, timedelta

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient

from core.models import RecurringBlock, ScheduledBlock, Task
from services.agent.tools.schedule import (
    ScheduleTaskAtTool,
    FindFreeSlotsTool,
    GetTodayScheduleTool,
)

_MONDAY = date(2026, 7, 20)  # Monday


def _weekday_on_or_after(dow: int) -> date:
    return _MONDAY + timedelta(days=(dow - _MONDAY.weekday()) % 7)


class NightShiftFreeSlotsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="night", password="x")
        # Saturday (day_of_week=5) night shift 19:00 -> 07:00.
        RecurringBlock.objects.create(
            user=self.user, title="Travail", block_type="work",
            day_of_week=5, start_time=time(19, 0), end_time=time(7, 0),
            is_night_shift=True,
        )
        self.saturday = _weekday_on_or_after(5)

    def test_find_free_slots_respects_night_shift(self):
        res = FindFreeSlotsTool().execute(self.user, date=self.saturday.isoformat())
        slots = res.data["free_slots"]
        # The 19:00-23:00 evening is busy (working), so no free slot ends after 19:00.
        self.assertTrue(slots)
        self.assertTrue(all(s["end_time"] <= "19:00" for s in slots), slots)
        # Free window is 07:00-19:00 = 720 min, not the whole 07:00-23:00 (960).
        self.assertEqual(res.data["total_free_minutes"], 720)

    def test_get_today_schedule_free_slots_respect_night_shift(self):
        res = GetTodayScheduleTool().execute(self.user, date=self.saturday.isoformat())
        self.assertEqual(res.data["total_free_minutes"], 720)
        self.assertTrue(all(s["end_time"] <= "19:00" for s in res.data["free_slots"]))


class FlexiblePlacementReadToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="placed", password="x")
        self.saturday = _weekday_on_or_after(5)
        self.sunday = self.saturday + timedelta(days=1)
        RecurringBlock.objects.create(
            user=self.user, title="Travail", block_type="work",
            day_of_week=self.saturday.weekday(), start_time=time(19, 0),
            end_time=time(7, 0), is_night_shift=True,
        )
        RecurringBlock.objects.create(
            user=self.user, title="Sommeil", block_type="sleep",
            day_of_week=self.sunday.weekday(), start_time=time(0, 0),
            end_time=time(7, 0),
        )

    def test_get_today_schedule_shows_placed_flexible_sleep(self):
        res = GetTodayScheduleTool().execute(self.user, date=self.sunday.isoformat())

        self.assertTrue(res.success, res.message)
        sleep = next(b for b in res.data["blocks"] if b["title"] == "Sommeil")
        self.assertEqual(sleep["start_time"], "07:00")
        self.assertEqual(sleep["end_time"], "14:00")

    def test_find_free_slots_excludes_placed_sleep_window(self):
        res = FindFreeSlotsTool().execute(self.user, date=self.sunday.isoformat())

        self.assertTrue(res.success, res.message)
        self.assertEqual(
            res.data["free_slots"],
            [{"start_time": "14:00", "end_time": "23:00", "duration_minutes": 540}],
        )

    def test_schedule_task_rejects_overlap_with_placed_sleep(self):
        res = ScheduleTaskAtTool().execute(
            self.user, title="Lecture", date=self.sunday.isoformat(),
            start_time="08:00", end_time="09:00",
        )

        self.assertFalse(res.success)
        self.assertEqual(
            res.data["conflict"],
            {"start_time": "00:00", "end_time": "14:00"},
        )
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 0)


class ScheduleTaskAtTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="oneoff", password="x")
        RecurringBlock.objects.create(
            user=self.user, title="Travail", block_type="work",
            day_of_week=5, start_time=time(19, 0), end_time=time(7, 0),
            is_night_shift=True,
        )
        self.saturday = _weekday_on_or_after(5)

    def test_places_one_off_locked_block(self):
        res = ScheduleTaskAtTool().execute(
            self.user, title="Lecture", date=self.saturday.isoformat(),
            start_time="09:00", end_time="11:00",
        )
        self.assertTrue(res.success, res.message)
        sb = ScheduledBlock.objects.get(user=self.user, date=self.saturday)
        self.assertEqual(sb.task.title, "Lecture")
        self.assertTrue(sb.locked)
        self.assertEqual(sb.start_time, time(9, 0))
        self.assertEqual(sb.end_time, time(11, 0))

    def test_rejects_overlap_with_night_shift(self):
        res = ScheduleTaskAtTool().execute(
            self.user, title="Lecture", date=self.saturday.isoformat(),
            start_time="20:00", end_time="21:00",
        )
        self.assertFalse(res.success)
        self.assertIn("conflict", res.data)
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 0)

    def test_rejects_end_before_start(self):
        res = ScheduleTaskAtTool().execute(
            self.user, title="Lecture", date=self.saturday.isoformat(),
            start_time="11:00", end_time="09:00",
        )
        self.assertFalse(res.success)

    def test_reuses_existing_task_no_duplicate(self):
        Task.objects.create(user=self.user, title="Lecture")
        ScheduleTaskAtTool().execute(
            self.user, title="Lecture", date=self.saturday.isoformat(),
            start_time="09:00", end_time="10:00",
        )
        self.assertEqual(Task.objects.filter(user=self.user, title="Lecture").count(), 1)

    def test_invalid_date(self):
        res = ScheduleTaskAtTool().execute(
            self.user, title="X", date="pas-une-date", start_time="09:00", end_time="10:00",
        )
        self.assertFalse(res.success)

    def test_tool_takes_no_block_id(self):
        self.assertNotIn("block_id", ScheduleTaskAtTool().parameters["properties"])


class SchedulePayloadPlacementTests(TestCase):
    """The week payload (/api/schedule/) exposes daily placement of flexible blocks."""

    def setUp(self):
        self.user = User.objects.create_user(username="payload", password="x")
        self.client = APIClient()
        self.client.force_authenticate(self.user)
        self.saturday = _weekday_on_or_after(5)
        self.sunday = self.saturday + timedelta(days=1)
        RecurringBlock.objects.create(
            user=self.user, title="Travail", block_type="work",
            day_of_week=self.saturday.weekday(), start_time=time(19, 0),
            end_time=time(7, 0), is_night_shift=True,
        )
        RecurringBlock.objects.create(
            user=self.user, title="Sommeil", block_type="sleep",
            day_of_week=self.sunday.weekday(), start_time=time(0, 0),
            end_time=time(7, 0),
        )

    def test_week_payload_places_flexible_sleep_in_daytime(self):
        r = self.client.get(reverse("schedule"), {"start_date": self.saturday.isoformat()})
        self.assertEqual(r.status_code, 200)
        self.assertIn("recurring_placements", r.data)
        sunday_sleep = [
            p for p in r.data["recurring_placements"]
            if p["date"] == self.sunday.isoformat() and not p["skipped"]
        ]
        self.assertTrue(sunday_sleep, "sleep placement missing for Sunday")
        self.assertEqual(sunday_sleep[0]["start_time"], "07:00")
        self.assertEqual(sunday_sleep[0]["end_time"], "14:00")
