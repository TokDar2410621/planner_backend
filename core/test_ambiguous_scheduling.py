"""
Chips de créneaux DÉTERMINISTES quand une planification est bloquée par un
conflit (demande Darius 2026-08-15 : « toute planification ambiguë propose
2-3 créneaux en chips », garanti par le code comme la question de fin de
récurrence).
"""
from datetime import time

from django.contrib.auth.models import User
from django.test import TestCase

from core.models import RecurringBlock
from services.agent.agent import _ambiguous_scheduling_chips


def _blocked_call(title='Café client', date='2026-09-07',
                  start='14:15', end='14:45'):
    return {
        'tool': 'schedule_task_at',
        'args': {'title': title, 'date': date,
                 'start_time': start, 'end_time': end},
        'result': {'success': False, 'message': 'conflit',
                   'data': {'conflict': {'start_time': '14:00', 'end_time': '15:00'}}},
    }


class AmbiguousSchedulingChipsTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('chips', password='pw-chips-1')
        # lundi 2026-09-07 : 14:00-15:00 occupé (fixe)
        RecurringBlock.objects.create(
            user=self.user, title='Réunion', block_type='work',
            day_of_week=0, start_time=time(14, 0), end_time=time(15, 0),
            flexibility='fixed')

    def test_conflict_yields_free_slot_chips(self):
        out = _ambiguous_scheduling_chips(self.user, [_blocked_call()])
        self.assertIsNotNone(out)
        phrase, chips = out
        self.assertIn('créneau est pris', phrase)
        self.assertTrue(1 <= len(chips) <= 3)
        for c in chips:
            self.assertIn('Café client', c['value'])
            self.assertIn('2026-09-07', c['value'])
            self.assertTrue(c['label'].startswith('🕐'))
            # aucun chip ne recouvre l'occupation 14:00-15:00
            self.assertNotIn('de 14:', c['value'])

    def _retire_test_agent_own_chips_are_respected(self):
        calls = [_blocked_call(), {
            'tool': 'present_quick_replies', 'args': {},
            'result': {'success': True, 'data': {}},
        }]
        self.assertIsNone(_ambiguous_scheduling_chips(self.user, calls))

    def test_eventual_success_disarms(self):
        ok = dict(_blocked_call())
        ok['result'] = {'success': True, 'data': {}}
        self.assertIsNone(_ambiguous_scheduling_chips(self.user, [_blocked_call(), ok]))

    def test_full_day_falls_back_to_tomorrow(self):
        RecurringBlock.objects.create(
            user=self.user, title='Marathon', block_type='work',
            day_of_week=0, start_time=time(7, 0), end_time=time(23, 0),
            flexibility='fixed')
        out = _ambiguous_scheduling_chips(self.user, [_blocked_call()])
        self.assertIsNotNone(out)
        _, chips = out
        self.assertIn('demain', chips[0]['label'])
        self.assertIn('2026-09-08', chips[0]['value'])

    def test_scheduled_block_conflict_also_yields_chips(self):
        # Cas prod: l'occupation est un ScheduledBlock (pas un récurrent).
        from datetime import date, time as t
        from core.models import Task, ScheduledBlock
        task = Task.objects.create(user=self.user, title='Occupation', task_type='shallow')
        ScheduledBlock.objects.create(
            user=self.user, task=task, date=date(2026, 9, 7),
            start_time=t(14, 0), end_time=t(15, 0), locked=True)
        out = _ambiguous_scheduling_chips(self.user, [_blocked_call()])
        self.assertIsNotNone(out)

    def test_agent_chips_do_not_disarm_conflict_case(self):
        # Les chips de l'agent sont vagues (« cherche un autre créneau »); les
        # nôtres portent des heures CONCRÈTES: sur un conflit, on force.
        calls = [_blocked_call(), {
            'tool': 'present_quick_replies', 'args': {},
            'result': {'success': True, 'data': {}},
        }]
        self.assertIsNotNone(_ambiguous_scheduling_chips(self.user, calls))

    def test_no_scheduling_calls_no_chips(self):
        self.assertIsNone(_ambiguous_scheduling_chips(self.user, []))

    def test_today_slots_never_in_the_past(self):
        from unittest import mock
        from datetime import datetime
        from zoneinfo import ZoneInfo
        import services.agent.agent as agent_mod
        # aujourd'hui simulé = le jour visé (lundi 2026-09-07), à 16:00 :
        # le trou du matin ne doit plus être offert tel quel.
        fake_now = datetime(2026, 9, 7, 16, 0, tzinfo=ZoneInfo('America/Toronto'))
        with mock.patch('django.utils.timezone.localtime', return_value=fake_now):
            out = _ambiguous_scheduling_chips(self.user, [_blocked_call()])
        self.assertIsNotNone(out)
        _, chips = out
        for c in chips:
            start = c['value'].split(' de ')[-1].split(' à ')[0]
            h = int(start.split(':')[0])
            self.assertGreaterEqual(h, 16, c)

    def test_consult_only_turn_yields_slot_chips(self):
        # L'agent consulte find_free_slots sans jamais tenter l'écriture:
        # la garantie doit quand même produire des chips de créneaux.
        calls = [{
            'tool': 'find_free_slots',
            'args': {'date': '2026-09-07', 'min_duration_minutes': 30},
            'result': {'success': True, 'data': {
                'date': '2026-09-07',
                'free_slots': [
                    {'start_time': '07:00', 'end_time': '14:00', 'duration_minutes': 420},
                    {'start_time': '15:00', 'end_time': '23:00', 'duration_minutes': 480},
                ],
            }},
        }]
        out = _ambiguous_scheduling_chips(self.user, calls)
        self.assertIsNotNone(out)
        phrase, chips = out
        self.assertIn('créneaux libres', phrase)
        self.assertEqual(chips[0]['label'], '🕐 07:00–07:30')
        self.assertIn('le 2026-09-07', chips[0]['value'])

    def test_consult_then_write_success_no_chips(self):
        calls = [
            {'tool': 'find_free_slots', 'args': {'date': '2026-09-07'},
             'result': {'success': True, 'data': {'date': '2026-09-07',
                        'free_slots': [{'start_time': '15:00', 'end_time': '16:00'}]}}},
            {'tool': 'schedule_task_at', 'args': {},
             'result': {'success': True, 'data': {}}},
        ]
        self.assertIsNone(_ambiguous_scheduling_chips(self.user, calls))

    def test_message_only_turn_yields_chips_when_window_busy(self):
        # L'agent répond depuis l'historique sans appeler d'outil: le message
        # explicite (« Planifie « X » aujourd'hui de 14:05 à 14:35 ») suffit.
        from unittest import mock
        from datetime import datetime
        from zoneinfo import ZoneInfo
        fake_now = datetime(2026, 9, 7, 9, 0, tzinfo=ZoneInfo('America/Toronto'))
        with mock.patch('django.utils.timezone.localdate',
                        return_value=fake_now.date()), \
             mock.patch('django.utils.timezone.localtime', return_value=fake_now):
            out = _ambiguous_scheduling_chips(
                self.user, [],
                "Planifie « Essai final » aujourd'hui de 14:05 à 14:35")
        self.assertIsNotNone(out)
        _, chips = out
        self.assertIn('Essai final', chips[0]['value'])

    def test_message_with_free_window_no_chips(self):
        from unittest import mock
        from datetime import datetime
        from zoneinfo import ZoneInfo
        fake_now = datetime(2026, 9, 7, 9, 0, tzinfo=ZoneInfo('America/Toronto'))
        with mock.patch('django.utils.timezone.localdate',
                        return_value=fake_now.date()), \
             mock.patch('django.utils.timezone.localtime', return_value=fake_now):
            out = _ambiguous_scheduling_chips(
                self.user, [],
                "Planifie « Essai libre » aujourd'hui de 16:00 à 16:30")
        self.assertIsNone(out)

    def test_message_without_times_no_chips(self):
        self.assertIsNone(_ambiguous_scheduling_chips(
            self.user, [], "Planifie ma lecture bientôt"))
