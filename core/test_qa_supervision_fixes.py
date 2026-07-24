"""Regression tests for the SUPERVISOR fixes to Codex's QA patches.

Codex's original drag/drop and recurring-block overlap validators had bugs that
their own tests masked (they exercised payloads the frontend never sends). These
tests exercise the REAL paths:
- drag/drop sends {date, start_time} ONLY (end is re-derived, duration-preserving);
- a metadata-only PATCH must not be blocked by a pre-existing overlap;
- retyping a block flips which overlaps count (fixed conflicts, flexible yields).
"""
from datetime import date, time

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient

from core.models import UserProfile, RecurringBlock, Task, ScheduledBlock

MONDAY = date(2026, 7, 27)  # weekday 0


class DragDropOverlapRealPayloadTests(TestCase):
    """PATCH /schedule/<id>/ with the real {start_time}-only drag payload."""

    def setUp(self):
        self.user = User.objects.create_user(username="drag", password="x")
        UserProfile.objects.update_or_create(user=self.user, defaults={})
        self.client = APIClient()
        self.client.force_authenticate(self.user)
        self.task = Task.objects.create(user=self.user, title="Lecture")

    def _sb(self, start, end):
        return ScheduledBlock.objects.create(
            user=self.user, task=self.task, date=MONDAY, start_time=start, end_time=end
        )

    def _wall(self, start, end):
        RecurringBlock.objects.create(
            user=self.user, title="Cours", block_type="course",
            day_of_week=MONDAY.weekday(), start_time=start, end_time=end,
        )

    def test_later_drag_onto_wall_is_rejected(self):
        # block 12:00-13:00, fixed wall 13:00-15:00. Drag start_time=13:00 only ->
        # end re-derived to 14:00 -> overlaps the wall -> must 400 (Codex accepted it).
        sb = self._sb(time(12, 0), time(13, 0))
        self._wall(time(13, 0), time(15, 0))
        url = reverse("schedule-block", kwargs={"block_id": sb.id})
        r = self.client.patch(url, {"start_time": "13:00"}, format="json")
        self.assertEqual(r.status_code, 400, r.content)
        sb.refresh_from_db()
        self.assertEqual(sb.start_time, time(12, 0))  # unchanged

    def test_earlier_drag_into_free_space_is_allowed(self):
        # block 14:00-15:00, wall 11:30-12:30. Drag start_time=10:00 only -> real
        # 10:00-11:00 is free -> must 200 (Codex falsely rejected it).
        sb = self._sb(time(14, 0), time(15, 0))
        self._wall(time(11, 30), time(12, 30))
        url = reverse("schedule-block", kwargs={"block_id": sb.id})
        r = self.client.patch(url, {"start_time": "10:00"}, format="json")
        self.assertEqual(r.status_code, 200, r.content)
        sb.refresh_from_db()
        self.assertEqual(sb.start_time, time(10, 0))
        self.assertEqual(sb.end_time, time(11, 0))  # duration preserved

    def test_small_self_move_is_allowed(self):
        sb = self._sb(time(9, 0), time(10, 0))
        url = reverse("schedule-block", kwargs={"block_id": sb.id})
        r = self.client.patch(url, {"start_time": "09:15"}, format="json")
        self.assertEqual(r.status_code, 200, r.content)

    def test_drag_onto_flexible_block_is_allowed(self):
        # A movable flexible block must NOT wall off a drag (flexible yields).
        RecurringBlock.objects.create(
            user=self.user, title="Sport", block_type="sport",
            day_of_week=MONDAY.weekday(), start_time=time(10, 0), end_time=time(11, 0),
        )
        sb = self._sb(time(14, 0), time(15, 0))
        url = reverse("schedule-block", kwargs={"block_id": sb.id})
        r = self.client.patch(url, {"start_time": "10:00"}, format="json")
        self.assertEqual(r.status_code, 200, r.content)


class RecurringOverlapUpdateTests(TestCase):
    """RecurringBlockSerializer.validate on update: metadata edits + retyping."""

    def setUp(self):
        self.user = User.objects.create_user(username="rec", password="x")
        UserProfile.objects.update_or_create(user=self.user, defaults={})
        self.client = APIClient()
        self.client.force_authenticate(self.user)

    def _block(self, title, block_type, start, end, day=0):
        return RecurringBlock.objects.create(
            user=self.user, title=title, block_type=block_type,
            day_of_week=day, start_time=start, end_time=end,
        )

    def test_metadata_only_edit_on_preexisting_overlap_is_allowed(self):
        # Two overlapping FIXED blocks exist (created via ORM, as extraction does).
        a = self._block("Work A", "work", time(9, 0), time(11, 0))
        self._block("Work B", "work", time(10, 0), time(12, 0))
        url = reverse("recurring-block-detail", kwargs={"pk": a.id})
        # Deactivate A: pure metadata, must NOT be blocked by the overlap.
        r = self.client.patch(url, {"active": False}, format="json")
        self.assertEqual(r.status_code, 200, r.content)
        # Rename A: same.
        r2 = self.client.patch(url, {"title": "Renamed"}, format="json")
        self.assertEqual(r2.status_code, 200, r2.content)

    def test_retype_flexible_to_fixed_with_overlap_is_rejected(self):
        self._block("Work", "work", time(10, 0), time(12, 0))
        sport = self._block("Sport", "sport", time(14, 0), time(15, 0))
        url = reverse("recurring-block-detail", kwargs={"pk": sport.id})
        # Sport (flexible) -> work (fixed) onto the work block -> now fixed vs fixed.
        r = self.client.patch(
            url, {"block_type": "work", "start_time": "10:30", "end_time": "11:30"},
            format="json",
        )
        self.assertEqual(r.status_code, 400, r.content)

    def test_retype_fixed_to_flexible_with_overlap_is_allowed(self):
        self._block("Cours", "course", time(10, 0), time(12, 0))
        work = self._block("Work", "work", time(14, 0), time(15, 0))
        url = reverse("recurring-block-detail", kwargs={"pk": work.id})
        # Work (fixed) -> sport (flexible) onto the course -> flexible yields -> OK.
        r = self.client.patch(
            url, {"block_type": "sport", "start_time": "10:30", "end_time": "11:30"},
            format="json",
        )
        self.assertEqual(r.status_code, 200, r.content)

    def test_create_fixed_over_fixed_is_still_rejected(self):
        self._block("Work", "work", time(9, 0), time(11, 0))
        url = reverse("recurring-block-list")
        r = self.client.post(
            url,
            {"title": "Cours", "block_type": "course", "day_of_week": 0,
             "start_time": "10:00", "end_time": "12:00"},
            format="json",
        )
        self.assertEqual(r.status_code, 400, r.content)
