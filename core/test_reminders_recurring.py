"""Parcours metier des rappels, occurrences et alertes visibles."""
from datetime import datetime, time, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

from django.contrib.auth.models import User
from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APITestCase

from core.models import RecurrenceSeries, Reminder, Task
from services.agent.tools import create_task, manage_reminder, manage_recurrence
from services.agent_v2.reconciliation import reconcilier
from services.agent_v2.registre import Registre
from services.reminder_series import materialize

TZ = ZoneInfo('America/Toronto')


class RecurrenceTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('series-owner', password='test-password')

    def test_monthly_31_clamps_and_each_task_keeps_own_completion(self):
        series = RecurrenceSeries.objects.create(
            user=self.user, kind='task', title='Payer le loyer',
            start_date=datetime(2027, 1, 31).date(), frequency='monthly',
            day_of_month=31, due_time=time(17, 0),
        )
        self.assertEqual(materialize(series, datetime(2027, 3, 31).date()), 3)
        days = list(series.tasks.order_by('occurrence_date').values_list('occurrence_date', flat=True))
        self.assertEqual([str(day) for day in days], ['2027-01-31', '2027-02-28', '2027-03-31'])
        first = series.tasks.order_by('occurrence_date').first()
        first.mark_completed()
        self.assertEqual(series.tasks.filter(completed=False).count(), 2)
        self.assertEqual(materialize(series, datetime(2027, 3, 31).date()), 0)

    def test_reminder_requires_hour_and_task_date_only_has_no_alert(self):
        result = manage_reminder.execute(self.user, action='create', title='Lait',
                                         due_at=None)
        self.assertFalse(result.success)
        day = (timezone.localdate() + timedelta(days=2)).isoformat()
        result = create_task.execute(self.user, title='Acheter du lait', deadline=day)
        self.assertTrue(result.success)
        task = Task.objects.get(pk=result.data['task']['id'])
        self.assertIsNone(task.alert_at)
        result = create_task.execute(self.user, title='Acheter du lait',
                                     deadline=f'{day}T17:00')
        self.assertTrue(result.success)
        self.assertEqual(Task.objects.filter(user=self.user).count(), 2)
        self.assertEqual(timezone.localtime(Task.objects.get(pk=result.data['task']['id']).alert_at).hour, 17)

    def test_reminder_receipt_compares_the_same_instant_across_timezones(self):
        due_at = datetime(2027, 1, 5, 10, 40, tzinfo=TZ)
        result = manage_reminder.execute(
            self.user, action='create', title='Acheter du lait', due_at=due_at.isoformat())
        self.assertTrue(result.success)

        registre = Registre()
        registre.ajouter('manage_reminder', {
            'action': 'create', 'title': 'Acheter du lait', 'due_at': due_at.isoformat(),
        }, result)
        reconcilier(self.user, registre)
        self.assertTrue(registre.par_id('a1').etat_verifie)

        Reminder.objects.filter(pk=result.data['reminder']['id']).update(title='Autre titre')
        registre = Registre()
        registre.ajouter('manage_reminder', {
            'action': 'create', 'title': 'Acheter du lait', 'due_at': due_at.isoformat(),
        }, result)
        reconcilier(self.user, registre)
        self.assertFalse(registre.par_id('a1').etat_verifie)

    def test_timed_task_and_alert_receipt_compares_instants(self):
        result = create_task.execute(
            self.user, title='Sortir les poubelles', deadline='2027-01-05T20:00')
        self.assertTrue(result.success)

        registre = Registre()
        registre.ajouter('create_task', {
            'title': 'Sortir les poubelles', 'deadline': '2027-01-05T20:00',
        }, result)
        reconcilier(self.user, registre)
        self.assertTrue(registre.par_id('a1').etat_verifie)

    def test_series_tool_creates_independent_reminders(self):
        day = timezone.localdate() + timedelta(days=2)
        result = manage_recurrence.execute(
            self.user, action='create', kind='reminder', title='Sortir les bacs',
            start_date=day, frequency='weekly', weekdays=[day.weekday()],
            due_time=time(20, 0))
        self.assertTrue(result.success)
        self.assertGreaterEqual(Reminder.objects.filter(user=self.user).count(), 4)

    def test_cancel_and_stop_require_code_confirmation_with_current_target(self):
        from services.agent_v2 import outils
        from services.agent_v2.registre import Registre
        from services.agent_v2.rendu import rendre_demandes

        due = timezone.now() + timedelta(days=2)
        reminder = Reminder.objects.create(user=self.user, title='Lait', due_at=due)
        series = RecurrenceSeries.objects.create(
            user=self.user, kind='reminder', title='Bacs',
            start_date=timezone.localdate() + timedelta(days=2),
            frequency='weekly', due_time=time(20, 0),
        )
        context = outils._Contexte(self.user, Registre(), 'test:confirmation',
                                   'Annule le rappel et arrête la répétition')
        for name, params in (
            ('manage_reminder', {'action': 'cancel', 'reminder_id': reminder.pk}),
            ('manage_recurrence', {'action': 'stop', 'series_id': series.pk}),
        ):
            with self.subTest(name=name):
                guard = outils._analyser(context, name, params)
                self.assertTrue(guard.actif)
                self.assertEqual(guard.motif, 'destructif')
                request = guard.demande()
                question, chips, _ = rendre_demandes([request], timezone.localdate())
                self.assertIn('rappel' if name == 'manage_reminder' else 'répétition', question)
                self.assertEqual([chip['option'] for chip in chips], ['confirmer', 'annuler'])
                confirm = next(option for option in request['options']
                               if option['id'] == 'confirmer')
                self.assertTrue(outils._effet_valide(request, 'confirmer', confirm['effet']))
                self.assertFalse(outils._cible_changee(self.user, request))
                if name == 'manage_reminder':
                    Reminder.objects.filter(pk=reminder.pk).update(title='Lait changé')
                else:
                    RecurrenceSeries.objects.filter(pk=series.pk).update(title='Bacs changés')
                self.assertTrue(outils._cible_changee(self.user, request))


class DispatchTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('dispatch-owner', password='test-password')
        self.now = datetime(2027, 1, 5, 17, 0, tzinfo=TZ)

    def test_no_device_marks_reminder_missed_without_late_delivery(self):
        reminder = Reminder.objects.create(user=self.user, title='Lait', due_at=self.now)
        with patch('core.management.commands.dispatch_due_alerts.timezone.now', return_value=self.now), \
             patch('core.management.commands.dispatch_due_alerts.has_device', return_value=False), \
             patch('core.management.commands.dispatch_due_alerts.send_visible') as send:
            call_command('dispatch_due_alerts')
            call_command('dispatch_due_alerts')
        reminder.refresh_from_db()
        self.assertEqual(reminder.status, 'missed')
        send.assert_not_called()

    def test_due_notification_once_and_completed_task_skipped(self):
        reminder = Reminder.objects.create(user=self.user, title='Lait', due_at=self.now)
        Task.objects.create(user=self.user, title='Déjà fait', deadline=self.now,
                            alert_at=self.now, completed=True)
        with patch('core.management.commands.dispatch_due_alerts.timezone.now', return_value=self.now), \
             patch('core.management.commands.dispatch_due_alerts.has_device', return_value=True), \
             patch('core.management.commands.dispatch_due_alerts.send_visible', return_value=1) as send:
            call_command('dispatch_due_alerts')
            call_command('dispatch_due_alerts')
        reminder.refresh_from_db()
        self.assertEqual(reminder.status, 'sent')
        self.assertEqual(reminder.delivered_count, 1)
        self.assertEqual(send.call_count, 1)

    def test_temporary_delivery_failure_retries_but_never_sends_late(self):
        reminder = Reminder.objects.create(user=self.user, title='Lait', due_at=self.now)
        task = Task.objects.create(user=self.user, title='Rapport', deadline=self.now,
                                   alert_at=self.now)
        with patch('core.management.commands.dispatch_due_alerts.timezone.now', return_value=self.now), \
             patch('core.management.commands.dispatch_due_alerts.has_device', return_value=True), \
             patch('core.management.commands.dispatch_due_alerts.send_visible', side_effect=[0, 0, 1, 1]) as send:
            call_command('dispatch_due_alerts')
            reminder.refresh_from_db()
            self.assertEqual(reminder.status, 'pending')
            call_command('dispatch_due_alerts')
            reminder.refresh_from_db()
            self.assertEqual(reminder.status, 'sent')
            self.assertEqual(send.call_count, 4)
        self.assertEqual(Task.objects.get(pk=task.pk).completed, False)
        late = Reminder.objects.create(user=self.user, title='Trop tard', due_at=self.now)
        with patch('core.management.commands.dispatch_due_alerts.timezone.now',
                   return_value=self.now + timedelta(seconds=91)), \
             patch('core.management.commands.dispatch_due_alerts.send_visible') as send:
            call_command('dispatch_due_alerts')
        late.refresh_from_db()
        self.assertEqual(late.status, 'missed')
        send.assert_not_called()


class ReminderApiTests(APITestCase):
    def setUp(self):
        self.owner = User.objects.create_user('rem-owner', password='test-password')
        self.other = User.objects.create_user('rem-other', password='test-password')
        self.client.force_authenticate(self.owner)

    def test_own_reminders_only_and_cancel_does_not_delete_history(self):
        future = timezone.now() + timedelta(days=3)
        own = Reminder.objects.create(user=self.owner, title='Moi', due_at=future)
        Reminder.objects.create(user=self.other, title='Autre', due_at=future)
        response = self.client.get('/api/reminders/')
        items = response.data['results'] if isinstance(response.data, dict) else response.data
        self.assertEqual([item['title'] for item in items], ['Moi'])
        self.assertEqual(self.client.delete(f'/api/reminders/{own.pk}/').status_code, 204)
        own.refresh_from_db()
        self.assertEqual(own.status, 'cancelled')
        self.assertEqual(self.client.patch(f'/api/reminders/{own.pk}/',
                                          {'title': 'Trop tard'}, format='json').status_code, 404)

    def test_cannot_create_past_reminder(self):
        response = self.client.post('/api/reminders/', {
            'title': 'Passé', 'due_at': (timezone.now() - timedelta(minutes=1)).isoformat(),
        }, format='json')
        self.assertEqual(response.status_code, 400)
