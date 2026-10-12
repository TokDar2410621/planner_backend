"""Applications datées: preuve ORM, sans toucher aux séries."""
from datetime import date, time, timedelta
from unittest import mock

from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import TestCase, TransactionTestCase

from core.models import RecurringBlock, RecurringBlockException, RecurringBlockOverride
from core.test_agent_v2_gardes import HarnaisGardes, puces
from services.agent.tools.schedule import (
    OrganizeDayTool, OptimizeWeekTool, GetTodayScheduleTool, GetWeekScheduleTool,
    _window_conflict,
)
from services.scheduling.day_view import effective_day_entries
from services.scheduling.occurrences import recurring_occurrences
from services.scheduling.placement import fixed_busy_intervals, open_intervals, place_day
from services.scheduling.solve_day import (
    empreinte_entrees_jour, empreinte_entrees_semaine, solve_placement, solve_day,
)


MONDAY = date(2026, 10, 19)


class DatedOptimizationTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user("dated-optimization")
        RecurringBlock.objects.create(
            user=self.user, title="Cours", block_type="course", day_of_week=0,
            start_time=time(9), end_time=time(12), flexibility="fixed")
        self.block = RecurringBlock.objects.create(
            user=self.user, title="Sport", block_type="sport", day_of_week=0,
            start_time=time(10), end_time=time(11), flexibility="flexible")

    def test_day_applies_date_and_preserves_series(self):
        result = OrganizeDayTool().execute(self.user, date=MONDAY.isoformat(), apply=True)
        self.assertTrue(result.success, result.message)
        self.assertTrue(result.data["applied"])
        self.block.refresh_from_db()
        self.assertEqual(self.block.start_time, time(10))
        observed = {b["id"]: b for b in effective_day_entries(self.user, MONDAY)}
        for placement in result.data["placed"]:
            self.assertEqual(observed[placement["block_id"]]["start_time"], placement["start_time"])

    def test_week_applies_dates_and_preserves_series(self):
        before = list(RecurringBlock.objects.filter(user=self.user).values())
        result = OptimizeWeekTool().execute(self.user, start_date=MONDAY.isoformat(), apply=True)
        self.assertTrue(result.success, result.message)
        self.assertTrue(result.data["applied"])
        self.assertEqual(list(RecurringBlock.objects.filter(user=self.user).values()), before)

    def arrangement(self, start=780, end=840):
        return [{"block_id": self.block.pk, "title": self.block.title, "block_type": "sport",
                 "start_min": start, "end_min": end,
                 "start_time": f"{start // 60:02d}:{start % 60:02d}",
                 "end_time": f"{end // 60:02d}:{end % 60:02d}",
                 "skipped": False, "overnight_kept": False, "preferred": False, "shrunk": False}]

    def apply(self, arrangement=None, **kwargs):
        return OrganizeDayTool().execute(
            self.user, date=MONDAY.isoformat(), apply=True,
            _arrangement=self.arrangement() if arrangement is None else arrangement, **kwargs)

    def test_proposal_writes_nothing_and_bad_date_is_rejected(self):
        for tool, kwargs in [(OrganizeDayTool(), {"date": MONDAY.isoformat()}),
                             (OptimizeWeekTool(), {"start_date": MONDAY.isoformat()})]:
            self.assertTrue(tool.execute(self.user, apply=False, **kwargs).success)
        self.assertFalse(OrganizeDayTool().execute(self.user, date="invalid", apply=True).success)
        self.assertFalse(RecurringBlockOverride.objects.exists())

    def test_day_week_availability_and_solver_share_confirmed_hours(self):
        self.assertTrue(self.apply().success)
        self.assertEqual(RecurringBlockOverride.objects.get().date, MONDAY)
        current = GetTodayScheduleTool().execute(self.user, date=MONDAY.isoformat()).data
        week = GetWeekScheduleTool().execute(self.user, start_date=MONDAY.isoformat()).data
        self.assertEqual(current["blocks"], week["days"][0]["detail"])
        sport = next(b for b in current["blocks"] if b["id"] == self.block.pk)
        self.assertEqual((sport["start_time"], sport["end_time"]), ("13:00", "14:00"))
        self.assertEqual(open_intervals(self.user, MONDAY, 780, 840), [])
        self.assertIsNotNone(_window_conflict(self.user, MONDAY, 790, 800))
        solved = solve_placement(self.user, MONDAY)
        self.assertEqual((solved[0]["start_time"], solved[0]["end_time"]), ("13:00", "14:00"))
        self.assertFalse(solve_day(self.user, MONDAY, extra=[{"title": "Test", "duration_minutes": 30}],
                                   day_start=780, day_end=840)["feasible"])
        later = MONDAY + timedelta(days=7)
        self.assertEqual(open_intervals(self.user, later, 780, 840), [(780, 840)])
        self.assertEqual(recurring_occurrences(self.user, later)[1].start_time, time(10))

    def test_fixed_conflict_rolls_back(self):
        result = self.apply(self.arrangement(600, 660))
        self.assertFalse(result.success)
        self.assertEqual(result.data["reason"], "occurrence_conflict")
        self.assertFalse(RecurringBlockOverride.objects.exists())

    def test_foreign_block_id_is_rejected_without_writes(self):
        other = User.objects.create_user("foreign-owner")
        foreign = RecurringBlock.objects.create(user=other, title="Sport", block_type="sport",
                                                day_of_week=0, start_time=time(13), end_time=time(14))
        arrangement = self.arrangement()
        arrangement[0]["block_id"] = foreign.pk
        self.assertFalse(self.apply(arrangement).success)
        self.assertFalse(RecurringBlockOverride.objects.exists())

    def test_skipped_or_out_of_bounds_occurrence_cannot_be_applied(self):
        RecurringBlockException.objects.create(user=self.user, recurring_block=self.block, date=MONDAY)
        self.assertFalse(self.apply().success)
        RecurringBlockException.objects.all().delete()
        self.block.end_date = MONDAY - timedelta(days=1)
        self.block.save(update_fields=["end_date"])
        self.assertFalse(self.apply().success)
        self.assertFalse(RecurringBlockOverride.objects.exists())

    def test_duplicate_occurrence_and_changed_duration_rejected(self):
        row = self.arrangement()[0]
        self.assertFalse(self.apply([row, row.copy()]).success)
        self.assertFalse(self.apply(self.arrangement(780, 810)).success)
        self.assertFalse(RecurringBlockOverride.objects.exists())

    def test_stale_inputs_rejected_before_any_write(self):
        expected = empreinte_entrees_jour(self.user, MONDAY)
        self.block.start_time = time(8)
        self.block.end_time = time(9)
        self.block.save(update_fields=["start_time", "end_time"])
        result = self.apply(_expected_entries=expected)
        self.assertFalse(result.success)
        self.assertEqual(result.data["reason"], "stale_plan")
        self.assertFalse(RecurringBlockOverride.objects.exists())

    def test_renamed_block_invalidates_cached_inputs(self):
        from services.agent_v2.outils import _plan_propose_jour
        first, before, _ = _plan_propose_jour(OrganizeDayTool(), self.user, {"date": MONDAY.isoformat()})
        self.block.title = "Natation"
        self.block.save(update_fields=["title"])
        second, after, _ = _plan_propose_jour(OrganizeDayTool(), self.user, {"date": MONDAY.isoformat()})
        self.assertNotEqual(before, after)
        self.assertEqual(first.data["placed"][0]["title"], "Sport")
        self.assertEqual(second.data["placed"][0]["title"], "Natation")

    def test_confirmation_does_not_add_new_transport_margins_to_soft_blocks(self):
        self.user.profile.transport_time_minutes = 30
        self.user.profile.save(update_fields=["transport_time_minutes"])
        self.assertTrue(self.apply().success)
        self.assertEqual(open_intervals(self.user, MONDAY, 750, 780), [(750, 780)])
        self.assertEqual(open_intervals(self.user, MONDAY, 840, 870), [(840, 870)])

    def test_week_rejects_invalid_later_day_without_partial_write(self):
        result = OptimizeWeekTool().execute(
            self.user, start_date=MONDAY.isoformat(), apply=True,
            _arrangements=[self.arrangement(), self.arrangement(), [], [], [], [], []])
        self.assertFalse(result.success)
        self.assertFalse(RecurringBlockOverride.objects.exists())

    def test_failed_postcondition_rolls_back_actual_override(self):
        from services.agent.tools import schedule
        original = schedule.effective_day_entries

        def read(user, day):
            # La première lecture observe le vrai planning. Après l'écriture,
            # le contrôle relit une projection divergente pour tester le rollback.
            return [] if RecurringBlockOverride.objects.exists() else original(user, day)

        with mock.patch.object(schedule, "effective_day_entries", side_effect=read):
            result = self.apply()
        self.assertEqual(result.data["reason"], "occurrence_postcondition_failed")
        self.assertFalse(RecurringBlockOverride.objects.exists())

    def test_end_at_midnight_is_stored_and_read_as_zero(self):
        result = self.apply(self.arrangement(1380, 1440))
        self.assertTrue(result.success, result.message)
        override = RecurringBlockOverride.objects.get()
        self.assertEqual(override.end_time, time(0))
        self.assertEqual(result.data["placed"][0]["end_time"], "00:00")
        self.assertEqual(fixed_busy_intervals(self.user, MONDAY + timedelta(days=1)), [])

    def test_override_overnight_occupies_its_start_date_and_next_morning(self):
        RecurringBlockOverride.objects.create(user=self.user, recurring_block=self.block,
                                             date=MONDAY, start_time=time(23), end_time=time(7))
        self.assertEqual(open_intervals(self.user, MONDAY, 0, 420), [(0, 420)])
        self.assertEqual(open_intervals(self.user, MONDAY, 1380, 1440), [])
        self.assertEqual(open_intervals(self.user, MONDAY + timedelta(days=1), 0, 420), [])
        self.assertEqual(place_day(self.user, MONDAY)[0]["end_time"], "07:00")

    def test_override_and_skip_change_cache_fingerprints(self):
        day_before = empreinte_entrees_jour(self.user, MONDAY)
        week_before = empreinte_entrees_semaine(self.user, MONDAY)
        self.assertTrue(self.apply().success)
        self.assertNotEqual(day_before, empreinte_entrees_jour(self.user, MONDAY))
        self.assertNotEqual(week_before, empreinte_entrees_semaine(self.user, MONDAY))
        RecurringBlockException.objects.create(user=self.user, recurring_block=self.block, date=MONDAY)
        self.assertEqual([b.title for b in recurring_occurrences(self.user, MONDAY)], ["Cours"])
        self.assertEqual(open_intervals(self.user, MONDAY, 780, 840), [(780, 840)])


class DayConfirmationTests(HarnaisGardes, TransactionTestCase):
    def setUp(self):
        super().setUp()
        cache.clear()
        self.sport = self.bloc("Sport", 0, "06:00", "07:00", block_type="sport", flexibility="flexible")

    def demand(self):
        action = self.premier_tour("Réorganise lundi", "organize_day", date="2026-09-14", apply=True)
        self.assertFalse(action.succes)
        self.assertFalse(RecurringBlockOverride.objects.exists())
        return action.donnees["demande"]

    def test_day_requires_confirmation_then_applies_the_same_plan(self):
        demande = self.demand()
        self.assertEqual(demande["cle"], "organize_day:apply")
        self.attendre([puces(demande)], "Oui, applique le plan.")
        registre, tools = self.outils("Oui, applique le plan.", tache="u:2")
        self.appeler(tools, "organize_day", date="2026-09-14", apply=True)
        self.assertTrue(registre.actions[-1].succes, registre.actions[-1])
        self.assertTrue(RecurringBlockOverride.objects.filter(recurring_block=self.sport).exists())

    def test_confirmation_does_not_apply_another_date(self):
        demande = self.demand()
        self.attendre([puces(demande)], "Oui, applique le plan.")
        registre, tools = self.outils("Oui, applique le plan.", tache="u:2")
        self.appeler(tools, "organize_day", date="2026-09-21", apply=True)
        self.assertFalse(registre.actions[-1].succes)
        self.assertFalse(RecurringBlockOverride.objects.exists())

    def test_confirmation_rejected_after_changed_inputs_even_same_placement(self):
        demande = self.demand()
        self.bloc("Cours", 0, "16:00", "17:00")
        self.attendre([puces(demande)], "Oui, applique le plan.")
        registre, tools = self.outils("Oui, applique le plan.", tache="u:2")
        self.appeler(tools, "organize_day", date="2026-09-14", apply=True)
        self.assertFalse(registre.actions[-1].succes)
        self.assertFalse(RecurringBlockOverride.objects.exists())

    def test_confirmation_chip_executes_the_dated_plan(self):
        from services.agent_v2.outils import appliquer_choix_en_attente
        from services.agent_v2.registre import Registre
        demande = self.demand()
        self.attendre([puces(demande)], "Oui, applique le plan.")
        registre = Registre()
        appliquer_choix_en_attente(self.user, registre, "Oui, applique le plan.", "u:2")
        self.assertTrue(registre.actions[-1].succes)
        self.assertTrue(RecurringBlockOverride.objects.filter(recurring_block=self.sport).exists())

    def test_confirmation_chip_does_not_apply_a_stale_plan(self):
        from services.agent_v2.outils import appliquer_choix_en_attente
        from services.agent_v2.registre import Registre
        demande = self.demand()
        self.bloc("Cours", 0, "16:00", "17:00")
        self.attendre([puces(demande)], "Oui, applique le plan.")
        registre = Registre()
        appliquer_choix_en_attente(self.user, registre, "Oui, applique le plan.", "u:2")
        self.assertFalse(registre.actions[-1].succes)
        self.assertFalse(RecurringBlockOverride.objects.exists())
