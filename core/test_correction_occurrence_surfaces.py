"""Une occurrence déplacée garde la même heure dans API, iCal et rappels."""
from datetime import datetime, time, timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from core.models import RecurringBlock, RecurringBlockException, RecurringBlockOverride
from services.ical import build_calendar


class OccurrenceSurfacesTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user('surface-datee')
        self.day = timezone.localdate() + timedelta(days=1)
        self.block = RecurringBlock.objects.create(
            user=self.user, title='Révision', day_of_week=self.day.weekday(),
            block_type='revision', flexibility='flexible',
            start_time=time(9), end_time=time(10),
        )
        self.override = RecurringBlockOverride.objects.create(
            user=self.user, recurring_block=self.block, date=self.day,
            start_time=time(15), end_time=time(16),
        )

    def test_api_datee_et_semaine_suivante_ne_changent_pas_la_serie(self):
        client = APIClient()
        client.force_authenticate(self.user)
        response = client.get('/api/schedule/', {'start_date': self.day.isoformat()})
        self.assertEqual(response.status_code, 200)
        slots = [p for p in response.data['recurring_placements']
                 if p['recurring_block'] == self.block.pk and p['date'] == self.day.isoformat()]
        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0]['start_time'], '15:00')
        self.block.refresh_from_db()
        self.assertEqual(self.block.start_time, time(9))
        from services.scheduling.occurrences import recurring_occurrences
        following = recurring_occurrences(self.user, self.day + timedelta(days=7))
        self.assertEqual(next(b for b in following if b.pk == self.block.pk).start_time, time(9))

    def test_ical_override_recurrence_id_garde_l_horaire_original(self):
        calendar = build_calendar(self.user)
        stamp = self.day.strftime('%Y%m%d')
        self.assertIn(f'RECURRENCE-ID:{stamp}T090000', calendar)
        self.assertIn(f'DTSTART:{stamp}T150000', calendar)
        self.assertEqual(calendar.count(f'UID:recblock-{self.block.pk}@planner-ai'), 2)

    def test_ical_nuit_datee_et_skip_prioritaire(self):
        self.override.start_time, self.override.end_time = time(23), time(1)
        self.override.save()
        calendar = build_calendar(self.user)
        self.assertIn(f'DTEND:{(self.day + timedelta(days=1)):%Y%m%d}T010000', calendar)
        RecurringBlockException.objects.create(user=self.user, recurring_block=self.block, date=self.day)
        self.assertNotIn('RECURRENCE-ID:', build_calendar(self.user))

    def _rappels(self, heure):
        now = timezone.make_aware(datetime.combine(self.day, heure))
        with patch('core.management.commands.send_reminders.timezone.localtime', return_value=now), \
             patch('core.management.commands.send_reminders.push_configured', return_value=True), \
             patch('core.management.commands.send_reminders.PushSubscription.objects') as subscriptions, \
             patch('core.management.commands.send_reminders.send_to_user', return_value=1) as send, \
             patch('core.management.commands.send_reminders.rattraper_tous_les_reveils'), \
             patch('services.webhooks.dispatch'):
            subscriptions.values_list.return_value.distinct.return_value = [self.user.pk]
            call_command('send_reminders', lead=15)
            return send.call_args_list

    def test_rappel_n_utilise_plus_l_ancienne_heure(self):
        self.assertEqual(self._rappels(time(8, 50)), [])

    def test_rappel_utilise_l_heure_datee(self):
        calls = self._rappels(time(14, 50))
        self.assertEqual(len(calls), 1)
        self.assertIn('15:00', calls[0].args[2])

    def test_occurrence_ignoree_ne_declenche_pas_de_rappel(self):
        RecurringBlockException.objects.create(user=self.user, recurring_block=self.block, date=self.day)
        self.assertEqual(self._rappels(time(14, 50)), [])
