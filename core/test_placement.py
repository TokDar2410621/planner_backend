from datetime import date, time

from django.contrib.auth.models import User
from django.test import TestCase

from core.models import RecurringBlock, RecurringBlockException, ScheduledBlock, Task
from services.scheduling.placement import fixed_busy_intervals, place_day


MONDAY = date(2026, 7, 20)
SATURDAY = date(2026, 7, 25)
SUNDAY = date(2026, 7, 26)


class PlacementEngineTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("placement", password="pw")

    def _recurring(self, title, block_type, dow, start, end, **overrides):
        return RecurringBlock.objects.create(
            user=self.user,
            title=title,
            block_type=block_type,
            day_of_week=dow,
            start_time=start,
            end_time=end,
            **overrides,
        )

    def _scheduled(self, target_date, start, end, title="Task"):
        task = Task.objects.create(user=self.user, title=title)
        return ScheduledBlock.objects.create(
            user=self.user,
            task=task,
            date=target_date,
            start_time=start,
            end_time=end,
        )

    def _by_title(self, placements, title):
        return next(block for block in placements if block["title"] == title)

    def test_no_fixed_blocks_flexible_sleep_stays_preferred(self):
        self._recurring(
            "Sleep",
            "sleep",
            MONDAY.weekday(),
            time(0, 0),
            time(7, 0),
        )

        sleep = self._by_title(place_day(self.user, MONDAY), "Sleep")

        self.assertEqual(sleep["start_min"], 0)
        self.assertEqual(sleep["end_min"], 7 * 60)
        self.assertEqual(sleep["start_time"], "00:00")
        self.assertEqual(sleep["end_time"], "07:00")
        self.assertTrue(sleep["preferred"])
        self.assertFalse(sleep["shrunk"])
        self.assertFalse(sleep["skipped"])
        self.assertFalse(sleep["overnight_kept"])

    def test_night_worker_saturday_shift_spills_into_sunday_morning(self):
        self._recurring(
            "Night Work",
            "work",
            SATURDAY.weekday(),
            time(19, 0),
            time(7, 0),
            is_night_shift=True,
        )
        self._recurring(
            "Saturday Sleep",
            "sleep",
            SATURDAY.weekday(),
            time(0, 0),
            time(7, 0),
        )
        self._recurring(
            "Sunday Sleep",
            "sleep",
            SUNDAY.weekday(),
            time(0, 0),
            time(7, 0),
        )

        saturday_sleep = self._by_title(place_day(self.user, SATURDAY), "Saturday Sleep")
        sunday_sleep = self._by_title(place_day(self.user, SUNDAY), "Sunday Sleep")

        self.assertEqual(saturday_sleep["start_min"], 0)
        self.assertEqual(saturday_sleep["end_min"], 7 * 60)
        self.assertTrue(saturday_sleep["preferred"])

        self.assertEqual(fixed_busy_intervals(self.user, SUNDAY), [(0, 7 * 60)])
        self.assertFalse(sunday_sleep["preferred"])
        self.assertEqual(sunday_sleep["end_min"] - sunday_sleep["start_min"], 7 * 60)
        self.assertFalse(
            sunday_sleep["start_min"] < 7 * 60 and sunday_sleep["end_min"] > 0
        )

    def test_two_flexible_blocks_do_not_overlap(self):
        self._recurring(
            "Sleep",
            "sleep",
            MONDAY.weekday(),
            time(0, 0),
            time(7, 0),
        )
        self._recurring(
            "Sport",
            "sport",
            MONDAY.weekday(),
            time(8, 0),
            time(9, 0),
        )

        placements = place_day(self.user, MONDAY)
        sleep = self._by_title(placements, "Sleep")
        sport = self._by_title(placements, "Sport")

        self.assertFalse(sleep["skipped"])
        self.assertFalse(sport["skipped"])
        self.assertFalse(
            sleep["start_min"] < sport["end_min"]
            and sport["start_min"] < sleep["end_min"]
        )

    def test_flexible_block_with_no_room_is_skipped(self):
        self._recurring(
            "Full Wall",
            "work",
            MONDAY.weekday(),
            time(0, 0),
            time(0, 0),
        )
        self._recurring(
            "Sleep",
            "sleep",
            MONDAY.weekday(),
            time(0, 0),
            time(7, 0),
        )

        sleep = self._by_title(place_day(self.user, MONDAY), "Sleep")

        self.assertTrue(sleep["skipped"])
        self.assertIsNone(sleep["start_min"])
        self.assertIsNone(sleep["end_min"])
        self.assertFalse(sleep["preferred"])

    def test_fixed_busy_excludes_flexible_and_includes_scheduled_and_spill(self):
        self._recurring(
            "Saturday Night Work",
            "work",
            SATURDAY.weekday(),
            time(19, 0),
            time(7, 0),
            is_night_shift=True,
        )
        self._recurring(
            "Flexible Sleep",
            "sleep",
            SUNDAY.weekday(),
            time(0, 0),
            time(7, 0),
        )
        self._scheduled(SUNDAY, time(10, 0), time(11, 0), title="Appointment")

        self.assertEqual(
            fixed_busy_intervals(self.user, SUNDAY),
            [(0, 7 * 60), (10 * 60, 11 * 60)],
        )

    def test_skipped_blocks_are_absent_from_walls_and_placements(self):
        work = self._recurring(
            "Skipped Night Work",
            "work",
            SATURDAY.weekday(),
            time(19, 0),
            time(7, 0),
            is_night_shift=True,
        )
        sleep = self._recurring(
            "Skipped Sleep",
            "sleep",
            SUNDAY.weekday(),
            time(0, 0),
            time(7, 0),
        )
        RecurringBlockException.objects.create(
            user=self.user,
            recurring_block=work,
            date=SATURDAY,
        )
        RecurringBlockException.objects.create(
            user=self.user,
            recurring_block=sleep,
            date=SUNDAY,
        )

        self.assertEqual(fixed_busy_intervals(self.user, SUNDAY), [])
        self.assertEqual(place_day(self.user, SUNDAY), [])

    def test_overnight_flexible_block_is_kept_not_relocated(self):
        self._recurring(
            "Overnight Sleep",
            "sleep",
            MONDAY.weekday(),
            time(23, 0),
            time(7, 0),
        )

        sleep = self._by_title(place_day(self.user, MONDAY), "Overnight Sleep")

        self.assertEqual(sleep["start_min"], 23 * 60)
        self.assertEqual(sleep["end_min"], 7 * 60)
        self.assertEqual(sleep["start_time"], "23:00")
        self.assertEqual(sleep["end_time"], "07:00")
        self.assertTrue(sleep["preferred"])
        self.assertFalse(sleep["shrunk"])
        self.assertFalse(sleep["skipped"])
        self.assertTrue(sleep["overnight_kept"])
