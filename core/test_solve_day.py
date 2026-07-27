"""Tests du solveur exact de faisabilité (services/scheduling/solve_day.py).

Ces cas verrouillent la réponse EXACTE d'OR2: une journée dont le temps libre est
fragmenté (3x1h) ne peut PAS accueillir un bloc de 2h, même si "3h sont libres".
"""
from datetime import date, time

from django.contrib.auth.models import User
from django.test import TestCase

from core.models import RecurringBlock
from services.scheduling.solve_day import solve_day, solve_placement

MONDAY = date(2026, 7, 27)


class SolvePlacementTests(TestCase):
    """solve_placement (candidat au remplacement du glouton, pour une action
    'optimise' opt-in): place les souples à l'optimum, contrat identique à place_day."""

    def test_returns_place_day_contract_keys(self):
        u = User.objects.create_user("sp_contract", password="pw")
        RecurringBlock.objects.create(
            user=u, title="Sport", block_type="sport", day_of_week=0,
            start_time=time(18, 0), end_time=time(19, 0))
        res = solve_placement(u, MONDAY)
        self.assertEqual(len(res), 1)
        for k in ("block_id", "title", "block_type", "start_min", "end_min",
                  "start_time", "end_time", "preferred", "shrunk", "skipped",
                  "overnight_kept"):
            self.assertIn(k, res[0])
        self.assertFalse(res[0]["skipped"])
        self.assertEqual(res[0]["shrunk"], False)  # le solveur ne rétrécit jamais

    def test_packs_both_where_greedy_would_split(self):
        # Trou unique 07-10 (3h) après murs; Révision 2h (préf 07:30) + Sport 1h
        # (préf 07:00). Le solveur cale les DEUX à pleine durée.
        u = User.objects.create_user("sp_pack", password="pw")
        for d in range(7):
            RecurringBlock.objects.create(
                user=u, title="Sommeil", block_type="sleep", day_of_week=d,
                start_time=time(23, 0), end_time=time(7, 0))
        RecurringBlock.objects.create(
            user=u, title="Cours", block_type="course", day_of_week=0,
            start_time=time(10, 0), end_time=time(23, 0))
        RecurringBlock.objects.create(
            user=u, title="Révision", block_type="revision", day_of_week=0,
            start_time=time(7, 30), end_time=time(9, 30))
        RecurringBlock.objects.create(
            user=u, title="Sport", block_type="sport", day_of_week=0,
            start_time=time(7, 0), end_time=time(8, 0))

        res = solve_placement(u, MONDAY)
        placed = {r["title"]: r for r in res if not r["skipped"] and not r["overnight_kept"]}
        self.assertIn("Révision", placed)
        self.assertIn("Sport", placed)
        # aucun placé avant 07:00 (sommeil du matin muré)
        for r in placed.values():
            self.assertGreaterEqual(r["start_min"], 7 * 60)


class SolveDayFeasibilityTests(TestCase):
    def _saturated_week(self, user):
        # Lundi: cours 8-17 (fixe) + travail 18-22 (fixe). Sommeil 23-07 tous les
        # jours (souple overnight) -> lundi libre = 07-08, 17-18, 22-23 = 3x1h.
        RecurringBlock.objects.create(
            user=user, title="Cours", block_type="course", day_of_week=0,
            start_time=time(8, 0), end_time=time(17, 0))
        RecurringBlock.objects.create(
            user=user, title="Travail", block_type="work", day_of_week=0,
            start_time=time(18, 0), end_time=time(22, 0))
        for d in range(7):
            RecurringBlock.objects.create(
                user=user, title="Sommeil", block_type="sleep", day_of_week=d,
                start_time=time(23, 0), end_time=time(7, 0))

    def _extra_placed(self, r):
        return sorted(p["title"] for p in r["placements"] if p["kind"] == "extra")

    def _extra_unplaced(self, r):
        return sorted(x["title"] for x in r["unplaced"] if x["key"].startswith("extra"))

    def test_empty_day_single_activity_fits(self):
        u = User.objects.create_user("sd_empty", password="pw")
        r = solve_day(u, MONDAY, extra=[{"title": "Lecture", "duration_minutes": 60}])
        self.assertTrue(r["feasible"])
        self.assertEqual(self._extra_placed(r), ["Lecture"])

    def test_saturated_two_hour_block_does_not_fit_fragmented_free_time(self):
        u = User.objects.create_user("sd_or2", password="pw")
        self._saturated_week(u)
        r = solve_day(u, MONDAY, extra=[
            {"title": "Étude", "duration_minutes": 120},
            {"title": "Lecture", "duration_minutes": 60},
        ])
        self.assertFalse(r["feasible"])
        self.assertIn("Étude", self._extra_unplaced(r))   # 2h dans aucun trou de 1h
        self.assertIn("Lecture", self._extra_placed(r))   # 1h rentre

    def test_saturated_three_one_hour_activities_all_fit(self):
        u = User.objects.create_user("sd_or2b", password="pw")
        self._saturated_week(u)
        r = solve_day(u, MONDAY, extra=[
            {"title": "A", "duration_minutes": 60},
            {"title": "B", "duration_minutes": 60},
            {"title": "C", "duration_minutes": 60},
        ])
        self.assertTrue(r["feasible"])
        self.assertEqual(self._extra_placed(r), ["A", "B", "C"])

    def test_saturated_four_one_hour_activities_overflow(self):
        u = User.objects.create_user("sd_or2c", password="pw")
        self._saturated_week(u)
        r = solve_day(u, MONDAY, extra=[
            {"title": f"X{i}", "duration_minutes": 60} for i in range(4)
        ])
        self.assertFalse(r["feasible"])  # 4h demandées > 3h libres

    def test_extra_never_evicts_an_existing_flexible_block(self):
        # Un extra ne doit JAMAIS évincer un engagement souple existant (sinon
        # feasible=True serait trompeur = OR2 recréé). Seul créneau libre 17-19,
        # pris par une séance de sport souple -> une étude de 2h ne rentre pas.
        u = User.objects.create_user("sd_evict", password="pw")
        RecurringBlock.objects.create(
            user=u, title="Cours", block_type="course", day_of_week=0,
            start_time=time(7, 0), end_time=time(17, 0))
        RecurringBlock.objects.create(
            user=u, title="Travail", block_type="work", day_of_week=0,
            start_time=time(19, 0), end_time=time(23, 0))
        for d in range(7):
            RecurringBlock.objects.create(
                user=u, title="Sommeil", block_type="sleep", day_of_week=d,
                start_time=time(23, 0), end_time=time(7, 0))
        RecurringBlock.objects.create(  # souple, occupe le seul trou 17-19
            user=u, title="Sport", block_type="sport", day_of_week=0,
            start_time=time(17, 0), end_time=time(19, 0))

        r = solve_day(u, MONDAY, extra=[{"title": "Étude", "duration_minutes": 120}])

        self.assertFalse(r["feasible"])
        self.assertIn("Sport", [p["title"] for p in r["placements"] if p["kind"] == "flexible"])
        self.assertIn("Étude", self._extra_unplaced(r))

    def test_overnight_sleep_from_previous_day_blocks_the_morning(self):
        # Le sommeil 23-07 de dimanche déborde sur lundi 00:00-07:00: une activité
        # préférant minuit doit être poussée à 07:00 au plus tôt (sommeil protégé).
        u = User.objects.create_user("sd_night", password="pw")
        for d in range(7):
            RecurringBlock.objects.create(
                user=u, title="Sommeil", block_type="sleep", day_of_week=d,
                start_time=time(23, 0), end_time=time(7, 0))
        r = solve_day(u, MONDAY, extra=[
            {"title": "Bloc", "duration_minutes": 120, "preferred_start": 0}
        ])
        self.assertTrue(r["feasible"])
        placed = [p for p in r["placements"] if p["kind"] == "extra"][0]
        self.assertGreaterEqual(placed["start_min"], 7 * 60)
