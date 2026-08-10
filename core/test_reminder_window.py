"""
Fenêtre à rétro-regard + anti-doublon du worker de rappels.

Vécu 2026-08-09 : ticks dérivants (sleep 900 + durée d'exécution) + fenêtre
[now, now+15] collée au présent = un bloc démarrant à l'heure pile tombait
dans la fissure entre deux fenêtres et n'était JAMAIS notifié.
"""
from datetime import datetime, time
from unittest import mock
from zoneinfo import ZoneInfo

from django.contrib.auth.models import User
from django.core.management import call_command
from django.test import TestCase

from core.models import PushSendLog, PushSubscription, RecurringBlock

TZ = ZoneInfo('America/Toronto')


def _at(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=TZ)


class ReminderWindowTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('remindme', password='pw-rem-1')
        PushSubscription.objects.create(
            user=self.user, endpoint='https://ex/fake', p256dh='k', auth='a')
        # lundi 10:00 (2026-08-10 est un lundi, dow0)
        self.block = RecurringBlock.objects.create(
            user=self.user, title='Publiar build', block_type='project',
            day_of_week=0, start_time=time(10, 0), end_time=time(12, 0))

    def _run(self, now):
        sends = []
        with mock.patch('core.management.commands.send_reminders.push_configured', return_value=True), \
             mock.patch('core.management.commands.send_reminders.send_to_user',
                        side_effect=lambda u, t, b, **k: sends.append(t) or 1), \
             mock.patch('core.management.commands.send_reminders.timezone.localtime',
                        return_value=now):
            call_command('send_reminders', '--lead', '15')
        return sends

    def test_tick_just_after_start_still_notifies(self):
        # tick à 10:02: l'ancienne fenêtre [10:02, 10:17] ratait le bloc de
        # 10:00 pour toujours; le rétro-regard le rattrape.
        sends = self._run(_at(2026, 8, 10, 10, 2))
        self.assertEqual(sends, ['Bientôt : Publiar build'])

    def test_overlapping_ticks_send_once(self):
        self._run(_at(2026, 8, 10, 9, 50))
        sends2 = self._run(_at(2026, 8, 10, 10, 2))
        self.assertEqual(sends2, [])  # déjà journalisé
        self.assertEqual(PushSendLog.objects.filter(user=self.user, kind='block_soon').count(), 1)

    def test_midnight_crossing_not_skipped(self):
        # bloc lundi 00:05; tick dimanche 23:55: l'ancien code sautait tout
        # tick dont la fenêtre traversait minuit.
        RecurringBlock.objects.create(
            user=self.user, title='Nuit', block_type='other',
            day_of_week=0, start_time=time(0, 5), end_time=time(1, 0))
        sends = self._run(_at(2026, 8, 9, 23, 55))
        self.assertEqual(sends, ['Bientôt : Nuit'])

    def test_out_of_window_not_notified(self):
        sends = self._run(_at(2026, 8, 10, 9, 30))
        self.assertEqual(sends, [])
