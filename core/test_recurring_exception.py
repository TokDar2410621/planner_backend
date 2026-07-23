"""Tests for single-occurrence skip of recurring blocks (RecurringBlockException).

Covers the agent tools (skip/restore resolve by date+type, never an ID),
the schedule expanders hiding a skipped occurrence, and the REST + week payload.
"""
from datetime import date, time, timedelta

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient

from core.models import RecurringBlock, RecurringBlockException
from services.agent.tools.blocks import (
    SkipBlockOccurrenceTool,
    RestoreBlockOccurrenceTool,
)
from services.agent.tools.schedule import GetTodayScheduleTool, FindFreeSlotsTool

# 2026-07-20 is a Monday (weekday 0); deterministic base for weekday math.
_MONDAY = date(2026, 7, 20)


def _weekday_on_or_after(dow: int) -> date:
    return _MONDAY + timedelta(days=(dow - _MONDAY.weekday()) % 7)


class SkipToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="skipper", password="x")
        # Friday (day_of_week=4) night-shift work 19:00 -> 07:00.
        self.work = RecurringBlock.objects.create(
            user=self.user, title="Travail", block_type="work",
            day_of_week=4, start_time=time(19, 0), end_time=time(7, 0),
            is_night_shift=True,
        )
        self.friday = _weekday_on_or_after(4)

    def test_skip_creates_exception_by_date_and_type(self):
        res = SkipBlockOccurrenceTool().execute(
            self.user, date=self.friday.isoformat(), block_type="work"
        )
        self.assertTrue(res.success, res.message)
        self.assertTrue(
            RecurringBlockException.objects.filter(
                user=self.user, recurring_block=self.work, date=self.friday
            ).exists()
        )

    def test_skip_is_idempotent(self):
        tool = SkipBlockOccurrenceTool()
        tool.execute(self.user, date=self.friday.isoformat(), block_type="work")
        tool.execute(self.user, date=self.friday.isoformat(), block_type="work")
        self.assertEqual(
            RecurringBlockException.objects.filter(user=self.user).count(), 1
        )

    def test_skip_zero_match_is_failure(self):
        res = SkipBlockOccurrenceTool().execute(
            self.user, date=self.friday.isoformat(), block_type="sport"
        )
        self.assertFalse(res.success)
        self.assertEqual(RecurringBlockException.objects.count(), 0)

    def test_skip_ambiguous_returns_candidates_without_writing(self):
        RecurringBlock.objects.create(
            user=self.user, title="Réunion", block_type="work",
            day_of_week=4, start_time=time(8, 0), end_time=time(9, 0),
        )
        res = SkipBlockOccurrenceTool().execute(
            self.user, date=self.friday.isoformat(), block_type="work"
        )
        self.assertFalse(res.success)
        self.assertEqual(len(res.data.get("candidates", [])), 2)
        self.assertEqual(RecurringBlockException.objects.count(), 0)

    def test_skip_ambiguous_resolved_by_title(self):
        RecurringBlock.objects.create(
            user=self.user, title="Réunion", block_type="work",
            day_of_week=4, start_time=time(8, 0), end_time=time(9, 0),
        )
        res = SkipBlockOccurrenceTool().execute(
            self.user, date=self.friday.isoformat(), block_type="work", title="Travail"
        )
        self.assertTrue(res.success, res.message)
        self.assertEqual(RecurringBlockException.objects.count(), 1)

    def test_restore_removes_exception(self):
        RecurringBlockException.objects.create(
            user=self.user, recurring_block=self.work, date=self.friday
        )
        res = RestoreBlockOccurrenceTool().execute(
            self.user, date=self.friday.isoformat(), block_type="work"
        )
        self.assertTrue(res.success, res.message)
        self.assertEqual(RecurringBlockException.objects.count(), 0)

    def test_invalid_date(self):
        res = SkipBlockOccurrenceTool().execute(self.user, date="pas-une-date")
        self.assertFalse(res.success)

    def test_skip_never_needs_block_id(self):
        # The tool schema exposes no block_id: it must resolve server-side.
        self.assertNotIn("block_id", SkipBlockOccurrenceTool().parameters["properties"])


class ScheduleExclusionTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="sched", password="x")
        self.course = RecurringBlock.objects.create(
            user=self.user, title="Cours", block_type="course",
            day_of_week=4, start_time=time(9, 0), end_time=time(11, 0),
        )
        self.friday = _weekday_on_or_after(4)
        self.next_friday = self.friday + timedelta(days=7)

    def test_get_today_hides_skipped_but_keeps_other_weeks(self):
        RecurringBlockException.objects.create(
            user=self.user, recurring_block=self.course, date=self.friday
        )
        res = GetTodayScheduleTool().execute(self.user, date=self.friday.isoformat())
        self.assertNotIn("Cours", [b["title"] for b in res.data["blocks"]])

        res_next = GetTodayScheduleTool().execute(
            self.user, date=self.next_friday.isoformat()
        )
        self.assertIn("Cours", [b["title"] for b in res_next.data["blocks"]])

    def test_free_slots_open_up_on_skipped_date(self):
        RecurringBlockException.objects.create(
            user=self.user, recurring_block=self.course, date=self.friday
        )
        res = FindFreeSlotsTool().execute(self.user, date=self.friday.isoformat())
        # No blocks left that day: the whole 07:00-23:00 window (16h) is free.
        self.assertEqual(res.data["total_free_minutes"], 16 * 60)


class RestApiTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="rest", password="x")
        self.block = RecurringBlock.objects.create(
            user=self.user, title="Travail", block_type="work",
            day_of_week=4, start_time=time(19, 0), end_time=time(7, 0),
            is_night_shift=True,
        )
        self.client = APIClient()
        self.client.force_authenticate(self.user)
        self.friday = _weekday_on_or_after(4)

    def test_create_is_idempotent(self):
        url = reverse("recurring-exception-list")
        payload = {"recurring_block": self.block.id, "date": self.friday.isoformat()}
        r1 = self.client.post(url, payload, format="json")
        self.assertIn(r1.status_code, (200, 201), r1.content)
        r2 = self.client.post(url, payload, format="json")
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(RecurringBlockException.objects.count(), 1)

    def test_week_payload_includes_exceptions(self):
        RecurringBlockException.objects.create(
            user=self.user, recurring_block=self.block, date=self.friday
        )
        r = self.client.get(reverse("schedule"), {"start_date": self.friday.isoformat()})
        self.assertEqual(r.status_code, 200)
        self.assertIn("recurring_exceptions", r.data)
        self.assertEqual(len(r.data["recurring_exceptions"]), 1)

    def test_cannot_skip_another_users_block(self):
        other = User.objects.create_user(username="other", password="x")
        other_block = RecurringBlock.objects.create(
            user=other, title="Autre", block_type="work",
            day_of_week=4, start_time=time(10, 0), end_time=time(11, 0),
        )
        url = reverse("recurring-exception-list")
        r = self.client.post(
            url,
            {"recurring_block": other_block.id, "date": self.friday.isoformat()},
            format="json",
        )
        self.assertEqual(r.status_code, 400)
        self.assertEqual(RecurringBlockException.objects.count(), 0)
