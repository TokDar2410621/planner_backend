"""optimize_week — l'optimiseur de SEMAINE (A8): solveur exact jour par jour.

apply=false PROPOSE sans muter; apply=true déplace les blocs souples. Ne touche
jamais aux fixes. Chaque bloc récurrent vit sur son jour de semaine, la passe
hebdo le traite exactement une fois.
"""
from datetime import date, time, timedelta

from django.contrib.auth.models import User
from django.test import TestCase

from core.models import RecurringBlock
from services.agent.tools.schedule import OptimizeWeekTool


MONDAY = date(2026, 8, 3)  # un lundi


def _rb(user, title, bt, dow, s, e, flex):
    return RecurringBlock.objects.create(
        user=user, title=title, block_type=bt, day_of_week=dow,
        start_time=s, end_time=e, flexibility=flex,
    )


class OptimizeWeekToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("weekopt", password="pw-123456")
        self.tool = OptimizeWeekTool()

    def test_propose_does_not_mutate(self):
        _rb(self.user, "Travail", "work", 0, time(9, 0), time(17, 0), "fixed")
        sport = _rb(self.user, "Sport", "sport", 0, time(16, 0), time(17, 0), "flexible")

        result = self.tool.execute(
            self.user, start_date=MONDAY.isoformat(), apply=False)

        self.assertTrue(result.success)
        self.assertFalse(result.data["applied"])
        sport.refresh_from_db()
        # PROPOSE seulement: le bloc n'a pas bougé en base.
        self.assertEqual(sport.start_time, time(16, 0))
        # mais la proposition le replace HORS du mur 9-17
        monday = result.data["days"][0]
        placed = {p["title"]: p for p in monday["placed"]}
        self.assertIn("Sport", placed)
        s = placed["Sport"]["start_time"]
        self.assertTrue(s >= "17:00" or s <= "08:00", s)

    def test_apply_moves_flexible_outside_fixed_wall(self):
        _rb(self.user, "Travail", "work", 0, time(9, 0), time(17, 0), "fixed")
        sport = _rb(self.user, "Sport", "sport", 0, time(16, 0), time(17, 0), "flexible")

        result = self.tool.execute(
            self.user, start_date=MONDAY.isoformat(), apply=True)

        self.assertTrue(result.success)
        self.assertTrue(result.data["applied"])
        self.assertGreaterEqual(result.data["moved_count"], 1)
        sport.refresh_from_db()
        # le souple est sorti du mur fixe
        self.assertFalse(time(9, 0) <= sport.start_time < time(17, 0))

    def test_fixed_blocks_never_move(self):
        work = _rb(self.user, "Travail", "work", 0, time(9, 0), time(17, 0), "fixed")
        _rb(self.user, "Sport", "sport", 0, time(10, 0), time(11, 0), "flexible")

        self.tool.execute(self.user, start_date=MONDAY.isoformat(), apply=True)

        work.refresh_from_db()
        self.assertEqual(work.start_time, time(9, 0))
        self.assertEqual(work.end_time, time(17, 0))

    def test_covers_seven_days_each_dow_once(self):
        # deux jours distincts avec des souples en conflit avec un mur
        for dow in (1, 4):
            _rb(self.user, f"Cours{dow}", "course", dow, time(9, 0), time(12, 0), "fixed")
            _rb(self.user, f"Lecture{dow}", "study", dow, time(10, 0), time(11, 0), "flexible")

        result = self.tool.execute(
            self.user, start_date=MONDAY.isoformat(), apply=False)

        self.assertEqual(len(result.data["days"]), 7)
        dates = [d["date"] for d in result.data["days"]]
        self.assertEqual(len(set(dates)), 7)
        self.assertEqual(dates[0], MONDAY.isoformat())
        self.assertEqual(dates[-1], (MONDAY + timedelta(days=6)).isoformat())
        # les 2 jours à replacer apparaissent bien
        with_placed = [d for d in result.data["days"] if d["placed"]]
        self.assertGreaterEqual(len(with_placed), 2)

    def test_invalid_date_is_rejected(self):
        result = self.tool.execute(self.user, start_date="pas-une-date")
        self.assertFalse(result.success)

    def test_default_start_is_today_and_empty_week_is_honest(self):
        result = self.tool.execute(self.user)
        self.assertTrue(result.success)
        self.assertIn("rien à replacer", result.message)
