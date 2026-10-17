"""Contrat des occurrences effectives, sans fournisseur ni calendrier externe."""
from datetime import date, time, timedelta

from django.contrib.auth.models import User
from django.test import TestCase

from core.models import RecurringBlock, RecurringBlockException, ScheduledBlock, Task
from services.scheduling.day_view import effective_day_blocks
from services.scheduling.placement import fixed_busy_intervals, open_intervals, place_day
from services.scheduling.solve_day import solve_day, solve_placement
from services.agent.tools import get_today_schedule, get_week_schedule, organize_day, harnais


MONDAY = date(2099, 10, 12)
# Le contrat porte sur un lundi sans dépendre de l'horloge du runner.
MONDAY -= timedelta(days=MONDAY.weekday())


class EffectiveOccurrenceTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("refonte-planning")
        self.other = User.objects.create_user("refonte-other")

    def block(self, title="Cours", **kwargs):
        fields = dict(user=self.user, title=title, block_type="course",
                      day_of_week=MONDAY.weekday(), start_time=time(9), end_time=time(10))
        fields.update(kwargs)
        return RecurringBlock.objects.create(**fields)

    def scheduled(self, **kwargs):
        user = kwargs.pop("user", self.user)
        task = Task.objects.create(user=user, title=kwargs.pop("title", "Rendez-vous daté"))
        fields = dict(user=user, task=task, date=MONDAY, start_time=time(14),
                      end_time=time(15), locked=True)
        fields.update(kwargs)
        return ScheduledBlock.objects.create(**fields)

    def day(self):
        return harnais.appeler(get_today_schedule, self.user, dict(date=MONDAY.isoformat())).data

    def week_day(self):
        return harnais.appeler(get_week_schedule, self.user, dict(start_date=MONDAY.isoformat())).data["days"][0]

    def test_day_excludes_expired_and_future_fixed_recurrences(self):
        self.block("Expiré", end_date=MONDAY - timedelta(days=1))
        self.block("Futur", start_date=MONDAY + timedelta(days=1))
        self.assertEqual(self.day()["blocks"], [])

    def test_week_excludes_expired_and_future_fixed_recurrences(self):
        self.block("Expiré", end_date=MONDAY - timedelta(days=1))
        self.block("Futur", start_date=MONDAY + timedelta(days=1))
        self.assertEqual(self.week_day()["detail"], [])

    def test_expired_fixed_wall_does_not_hide_free_slot(self):
        self.block(end_date=MONDAY - timedelta(days=1))
        self.assertEqual(open_intervals(self.user, MONDAY, 540, 600), [(540, 600)])

    def test_expired_fixed_wall_does_not_make_solver_infeasible(self):
        block = self.block(end_date=MONDAY - timedelta(days=1))
        extra = [{"title": "Une heure", "duration_minutes": 60}]
        self.assertTrue(solve_day(self.user, MONDAY, extra=extra,
                                  day_start=540, day_end=600)["feasible"])
        # Témoin négatif : exactement le même mur, dans sa période valide.
        block.end_date = MONDAY
        block.save(update_fields=["end_date"])
        self.assertFalse(solve_day(self.user, MONDAY, extra=extra,
                                   day_start=540, day_end=600)["feasible"])

    def test_inactive_flexible_occurrences_do_not_enter_either_solver(self):
        self.block("Futur", block_type="sport", start_date=MONDAY + timedelta(days=1))
        self.block("Expiré", block_type="sport", end_date=MONDAY - timedelta(days=1))
        self.assertEqual(place_day(self.user, MONDAY), [])
        self.assertEqual(solve_placement(self.user, MONDAY), [])
        solved = solve_day(self.user, MONDAY, extra=[{"title": "Une heure", "duration_minutes": 60}],
                           day_start=540, day_end=600)
        self.assertTrue(solved["feasible"])
        self.assertEqual([p["kind"] for p in solved["placements"]], ["extra"])

    def test_inclusive_bounds_skips_pending_inactive_and_user_isolation(self):
        self.block("Présent", start_date=MONDAY, end_date=MONDAY)
        skipped = self.block("Sauté")
        RecurringBlockException.objects.create(user=self.user, recurring_block=skipped, date=MONDAY)
        self.block("Autre compte", user=self.other)
        self.scheduled(user=self.other)
        self.block("Désactivé", active=False)
        self.block("Non confirmé", status="pending")
        self.assertEqual([b["title"] for b in self.day()["blocks"]], ["Présent"])
        self.assertEqual([b["title"] for b in self.week_day()["detail"]], ["Présent"])
        self.assertEqual([b["title"] for b in effective_day_blocks(self.user, MONDAY)], ["Présent"])

    def test_week_includes_dated_appointments_with_readable_identity(self):
        scheduled = self.scheduled()
        detail = self.week_day()["detail"]
        self.assertEqual(len(detail), 1)
        self.assertEqual(detail[0]["id"], scheduled.id)
        self.assertEqual(detail[0]["type"], "scheduled_task")
        self.assertEqual(detail[0]["start_time"], "14:00")
        self.assertEqual(self.week_day()["occupied_hours"], 1)
        self.assertEqual(self.week_day()["block_count"], 1)

    def test_day_week_and_effective_reader_share_placed_flexible_hours(self):
        self.block()
        self.block("Sport", block_type="sport")
        rows = [self.day()["blocks"], self.week_day()["detail"], effective_day_blocks(self.user, MONDAY)]
        slots = []
        for row in rows:
            sport = next(b for b in row if b["title"] == "Sport")
            slots.append((sport["start_time"], sport["end_time"]))
        self.assertEqual(slots, [("10:00", "11:00")] * 3)

    def test_query_window_does_not_move_a_flexible_occurrence(self):
        self.block("Sport matinal", block_type="sport", start_time=time(6), end_time=time(7))
        self.assertEqual(self.day()["blocks"][0]["start_time"], "06:00")
        self.assertEqual(open_intervals(self.user, MONDAY, 420, 480), [(420, 480)])

    def test_previous_overnight_uses_its_start_date_for_bounds_and_skip(self):
        previous = MONDAY - timedelta(days=1)
        night = self.block("Quart", day_of_week=previous.weekday(), start_time=time(22),
                           end_time=time(8), start_date=previous, end_date=previous)
        self.assertEqual(fixed_busy_intervals(self.user, MONDAY), [(0, 480)])
        night.end_date = previous - timedelta(days=1)
        night.save(update_fields=["end_date"])
        self.assertEqual(fixed_busy_intervals(self.user, MONDAY), [])
        night.end_date = previous
        night.save(update_fields=["end_date"])
        RecurringBlockException.objects.create(user=self.user, recurring_block=night, date=previous)
        self.assertEqual(fixed_busy_intervals(self.user, MONDAY), [])

    def test_skipped_occurrence_does_not_change_the_next_week(self):
        block = self.block()
        RecurringBlockException.objects.create(user=self.user, recurring_block=block, date=MONDAY)
        self.assertEqual(fixed_busy_intervals(self.user, MONDAY), [])
        self.assertEqual(fixed_busy_intervals(self.user, MONDAY + timedelta(days=7)), [(540, 600)])

    def test_completed_scheduled_item_is_visible_but_frees_its_slot(self):
        self.scheduled(actually_completed=True)
        self.assertEqual(len(self.day()["blocks"]), 1)
        self.assertEqual(len(self.week_day()["detail"]), 1)
        self.assertEqual(open_intervals(self.user, MONDAY, 840, 900), [(840, 900)])

    def test_overnight_sleep_morning_is_protected_in_free_slots(self):
        self.block("Sommeil", block_type="sleep", start_time=time(23), end_time=time(8))
        self.assertEqual(open_intervals(self.user, MONDAY, 420, 480), [])

    def test_organize_day_applies_occurrence_without_changing_series(self):
        self.block()
        self.block("Sport", block_type="sport")
        before = list(RecurringBlock.objects.filter(user=self.user).values())
        result = harnais.appeler(organize_day, self.user, dict(date=MONDAY.isoformat(), apply=True))
        self.assertTrue(result.success, result.message)
        self.assertTrue(result.data["applied"])
        self.assertEqual(list(RecurringBlock.objects.filter(user=self.user).values()), before)
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 0)
        self.assertEqual(Task.objects.filter(user=self.user).count(), 0)
        self.assertEqual(RecurringBlockException.objects.filter(user=self.user).count(), 0)
        proposal = harnais.appeler(organize_day, self.user, dict(date=MONDAY.isoformat(), apply=False))
        self.assertTrue(proposal.success)
        self.assertEqual(result.data["placed"], proposal.data["placed"])
