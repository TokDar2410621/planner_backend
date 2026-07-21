"""Schedule-intent gate: scores schedule-like text, filters chit-chat."""
from django.test import SimpleTestCase

from services.schedule_intent import intent_score, has_schedule_signal, signals


class ScheduleIntentTest(SimpleTestCase):
    def test_full_schedule_line_scores_high(self):
        txt = "Cours de maths lundi 9h à 11h, salle A-101"
        self.assertGreaterEqual(intent_score(txt), 0.8)
        self.assertTrue(has_schedule_signal(txt))

    def test_chitchat_scores_low(self):
        txt = "Salut, ça va bien ? On se voit plus tard j'espère."
        self.assertLess(intent_score(txt), 0.4)
        self.assertFalse(has_schedule_signal(txt))

    def test_accented_keywords_match(self):
        # 'réunion' / 'matière' must match despite accents.
        txt = "Réunion mardi à 14h pour la matière de projet"
        self.assertTrue(has_schedule_signal(txt))

    def test_time_formats_detected(self):
        for t in ["9h", "9h30", "09:00", "14 h 30"]:
            self.assertGreaterEqual(signals(f"un truc {t}")['times'], 1, t)

    def test_bare_numbers_do_not_trigger_times(self):
        # Years / prices must not count as clock times.
        self.assertEqual(signals("budget 2024 pour 500 dollars")['times'], 0)

    def test_empty_text_is_zero(self):
        self.assertEqual(intent_score(''), 0.0)
        self.assertFalse(has_schedule_signal(''))

    def test_days_alone_below_threshold(self):
        # A day name with no time and no vocab should not clear the gate.
        self.assertFalse(has_schedule_signal("lundi"))
