"""
Phase 2 du moteur: replanification partielle (spec §7) + notifications de
départ (spec §9).
"""
from datetime import datetime, time
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase

from core.models import PushSubscription, RecurringBlock, ScheduledBlock, Task, UserPlace
from services.replan import replan_after_delay, apply_proposal, undo_change


class ReplanPartialTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('replanuser', password='pw-rpl-123456')
        # These tests exercise the APPLIED mechanics -> force automatic mode.
        self.user.profile.automation_mode = 'automatic'
        self.user.profile.save()
        self.today = timezone.localdate()

    def _scheduled(self, task, start, end, completed=False):
        return ScheduledBlock.objects.create(
            user=self.user, task=task, date=self.today,
            start_time=start, end_time=end, actually_completed=completed,
        )

    def test_delayed_block_is_moved_after_resume(self):
        task = Task.objects.create(
            user=self.user, title='Lecture', estimated_duration_minutes=60,
        )
        self._scheduled(task, time(9, 0), time(10, 0))  # avant la reprise

        result = replan_after_delay(self.user, resume_time='11:00')

        self.assertEqual(result['resume_time'], '11:00')
        self.assertEqual(len(result['moved']), 1)
        moved = result['moved'][0]
        self.assertEqual(moved['title'], 'Lecture')
        self.assertEqual(moved['was'], '09:00')
        # Re-placé après 11:00.
        self.assertGreaterEqual(moved['now'], '11:00')
        # L'ancien bloc 09:00 n'existe plus, un nouveau (>= 11:00) le remplace.
        blocks = ScheduledBlock.objects.filter(user=self.user, task=task)
        self.assertEqual(blocks.count(), 1)
        self.assertGreaterEqual(blocks.first().start_time, time(11, 0))

    def test_completed_block_is_not_touched(self):
        task = Task.objects.create(
            user=self.user, title='Déjà fait', estimated_duration_minutes=60,
        )
        done = self._scheduled(task, time(8, 0), time(9, 0), completed=True)
        replan_after_delay(self.user, resume_time='11:00')
        # Toujours là, inchangé.
        done.refresh_from_db()
        self.assertTrue(done.actually_completed)
        self.assertEqual(done.start_time, time(8, 0))

    def test_nothing_to_move_message(self):
        task = Task.objects.create(
            user=self.user, title='Plus tard', estimated_duration_minutes=60,
        )
        self._scheduled(task, time(15, 0), time(16, 0))  # après la reprise 11:00
        result = replan_after_delay(self.user, resume_time='11:00')
        self.assertEqual(result['moved'], [])
        self.assertIn('tient toujours', result['message'])


class ReplanEndpointTest(APITestCase):
    def test_replan_endpoint(self):
        user = User.objects.create_user('replanapi', password='pw-rpa-123456')
        self.client.force_authenticate(user)
        task = Task.objects.create(user=user, title='Sport', estimated_duration_minutes=60)
        ScheduledBlock.objects.create(
            user=user, task=task, date=timezone.localdate(),
            start_time=time(9, 0), end_time=time(10, 0),
        )
        r = self.client.post(reverse('schedule-replan'), {'resume_time': '12:00'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertEqual(r.data['resume_time'], '12:00')
        self.assertEqual(len(r.data['moved']), 1)

    def test_replan_rejects_bad_delay(self):
        user = User.objects.create_user('replanbad', password='pw-rpb-123456')
        self.client.force_authenticate(user)
        r = self.client.post(reverse('schedule-replan'), {'delay_minutes': 'oops'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_400_BAD_REQUEST)


class AutomationModeTest(TestCase):
    """spec §8: suggestion propose, automatique applique + annulable."""

    def setUp(self):
        self.user = User.objects.create_user('automode', password='pw-aut-123456')
        self.today = timezone.localdate()
        self.task = Task.objects.create(
            user=self.user, title='Sport', estimated_duration_minutes=60,
        )
        ScheduledBlock.objects.create(
            user=self.user, task=self.task, date=self.today,
            start_time=time(9, 0), end_time=time(10, 0),
        )

    def _set_mode(self, mode):
        self.user.profile.automation_mode = mode
        self.user.profile.save()

    def _current_start(self):
        b = ScheduledBlock.objects.filter(user=self.user, task=self.task).first()
        return b.start_time if b else None

    def test_suggestion_proposes_and_does_not_apply(self):
        self._set_mode('suggestion')
        result = replan_after_delay(self.user, resume_time='11:00')
        self.assertFalse(result['applied'])
        self.assertTrue(result['token'])
        self.assertIn('propose', result['message'])
        # DB unchanged: block still at 09:00.
        self.assertEqual(self._current_start(), time(9, 0))

    def test_proposal_can_be_applied(self):
        self._set_mode('suggestion')
        token = replan_after_delay(self.user, resume_time='11:00')['token']
        applied = apply_proposal(self.user, token)
        self.assertTrue(applied['applied'])
        self.assertGreaterEqual(self._current_start(), time(11, 0))
        # Re-applying the same proposal is now a no-op (already applied -> None).
        self.assertIsNone(apply_proposal(self.user, token))

    def test_automatic_applies_and_undo_restores(self):
        self._set_mode('automatic')
        result = replan_after_delay(self.user, resume_time='11:00')
        self.assertTrue(result['applied'])
        self.assertGreaterEqual(self._current_start(), time(11, 0))
        undo = undo_change(self.user, result['token'])
        self.assertTrue(undo['undone'])
        # Restored to the original 09:00.
        self.assertEqual(self._current_start(), time(9, 0))

    def test_semi_auto_small_change_auto_applies(self):
        self._set_mode('semi_auto')
        # 09:00 -> 09:30 = 30 min shift <= threshold(60) -> small -> applied.
        result = replan_after_delay(self.user, resume_time='09:30')
        self.assertTrue(result['applied'])

    def test_semi_auto_important_change_is_proposed(self):
        self._set_mode('semi_auto')
        # 09:00 -> 11:00 = 120 min shift > threshold(60) -> important -> proposed.
        result = replan_after_delay(self.user, resume_time='11:00')
        self.assertFalse(result['applied'])
        self.assertEqual(self._current_start(), time(9, 0))

    def test_undo_is_user_scoped(self):
        self._set_mode('automatic')
        token = replan_after_delay(self.user, resume_time='11:00')['token']
        other = User.objects.create_user('automode-other', password='pw-oth-123456')
        self.assertIsNone(undo_change(other, token))


class ProposalUndoEndpointTest(APITestCase):
    def test_apply_and_undo_endpoints(self):
        user = User.objects.create_user('propapi', password='pw-pra-123456')
        user.profile.automation_mode = 'suggestion'
        user.profile.save()
        self.client.force_authenticate(user)
        task = Task.objects.create(user=user, title='Lecture', estimated_duration_minutes=60)
        ScheduledBlock.objects.create(
            user=user, task=task, date=timezone.localdate(),
            start_time=time(9, 0), end_time=time(10, 0),
        )
        # Propose (suggestion mode).
        replan = self.client.post(reverse('schedule-replan'), {'resume_time': '12:00'}, format='json')
        self.assertFalse(replan.data['applied'])
        token = replan.data['token']
        # Apply.
        applied = self.client.post(reverse('schedule-proposal-apply'), {'token': token}, format='json')
        self.assertEqual(applied.status_code, status.HTTP_200_OK)
        # Undo.
        undo = self.client.post(reverse('schedule-undo'), {'token': token}, format='json')
        self.assertEqual(undo.status_code, status.HTTP_200_OK)
        self.assertTrue(undo.data['undone'])

    def test_bad_token_is_404(self):
        user = User.objects.create_user('badtok', password='pw-bad-123456')
        self.client.force_authenticate(user)
        r = self.client.post(reverse('schedule-undo'), {'token': 'not-a-uuid'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_404_NOT_FOUND)


class ReplanRegressionFixesTest(TestCase):
    """Bugs trouvés par la revue adversariale — ne doivent plus se reproduire."""

    def setUp(self):
        self.user = User.objects.create_user('regfix', password='pw-reg-123456')
        self.user.profile.automation_mode = 'automatic'
        self.user.profile.save()
        self.today = timezone.localdate()

    def _block(self, task, hh, completed=False):
        return ScheduledBlock.objects.create(
            user=self.user, task=task, date=self.today,
            start_time=time(hh, 0), end_time=time(hh + 1, 0),
            actually_completed=completed,
        )

    def test_interleaved_other_task_block_survives_undo(self):
        # Lost-update fix: un bloc d'une AUTRE tâche créé entre apply et undo
        # ne doit PAS être effacé par le restore (scope = tâches du changement).
        a = Task.objects.create(user=self.user, title='A', estimated_duration_minutes=60)
        self._block(a, 9)
        res = replan_after_delay(self.user, resume_time='11:00')  # A 09->11, applied
        z = Task.objects.create(user=self.user, title='Z', estimated_duration_minutes=60)
        self._block(z, 16)  # interleaved write for another task
        undo_change(self.user, res['token'])
        # Z toujours là, A restauré à 09:00.
        self.assertTrue(ScheduledBlock.objects.filter(user=self.user, task=z, start_time=time(16, 0)).exists())
        self.assertTrue(ScheduledBlock.objects.filter(user=self.user, task=a, start_time=time(9, 0)).exists())

    def test_completed_task_not_desynced_by_undo(self):
        # Completion-aware restore: si la tâche a été complétée après l'apply,
        # l'undo ne la ressuscite pas 'à faire' -> pas de désync Task.completed.
        a = Task.objects.create(user=self.user, title='A', estimated_duration_minutes=60)
        self._block(a, 9)
        res = replan_after_delay(self.user, resume_time='11:00')  # A 09->11
        moved_block = ScheduledBlock.objects.get(user=self.user, task=a)
        moved_block.mark_completed(actual_minutes=60)  # complète au nouveau créneau
        undo_change(self.user, res['token'])
        a.refresh_from_db()
        self.assertTrue(a.completed)  # reste complétée
        # Pas de bloc incomplet ressuscité à 09:00.
        self.assertFalse(
            ScheduledBlock.objects.filter(
                user=self.user, task=a, start_time=time(9, 0), actually_completed=False
            ).exists()
        )

    def test_duplicate_blocks_replanned_once(self):
        # Over-scheduling fix: une tâche avec 2 blocs avant la reprise est
        # re-planifiée UNE fois, pas deux.
        a = Task.objects.create(user=self.user, title='A', estimated_duration_minutes=60)
        self._block(a, 9)
        self._block(a, 10)
        replan_after_delay(self.user, resume_time='12:00')
        blocks = ScheduledBlock.objects.filter(user=self.user, task=a)
        self.assertEqual(blocks.count(), 1)
        self.assertGreaterEqual(blocks.first().start_time, time(12, 0))

    def test_noop_replan_records_no_token(self):
        # Rien à déplacer -> pas de SchedulePlanChange, pas de token.
        a = Task.objects.create(user=self.user, title='Tard', estimated_duration_minutes=60)
        self._block(a, 15)  # après la reprise
        res = replan_after_delay(self.user, resume_time='11:00')
        self.assertTrue(res['applied'])
        self.assertIsNone(res['token'])
        from core.models import SchedulePlanChange
        self.assertEqual(SchedulePlanChange.objects.filter(user=self.user).count(), 0)


@patch('core.management.commands.send_reminders.push_configured', return_value=True)
@patch('core.management.commands.send_reminders.send_to_user', return_value=1)
class DepartureAlertsTest(TestCase):
    """spec §9: 'prépare-toi' à l'indisponibilité, 'pars maintenant' au départ.

    Bloc 18:00, trajet 32, marge 10, prépa 15 -> départ 17:18, prépa dès 17:03.
    """

    def _setup_user(self):
        user = User.objects.create_user('depart', password='pw-dep-123456')
        p = user.profile
        p.prep_time_minutes = 15
        p.safety_margin_minutes = 10
        p.save()
        PushSubscription.objects.create(
            user=user, endpoint='https://push.example/dep', p256dh='k', auth='a'
        )
        place = UserPlace.objects.create(
            user=user, name='Travail', kind='work', travel_minutes=32
        )
        return user, place

    def _fake_now(self, hh, mm, weekday_ref):
        naive = datetime(2026, 7, 20, hh, mm)  # base date; weekday set below
        aware = timezone.make_aware(naive)
        return aware

    def test_prep_alert_fires_at_1703(self, mock_send, mock_cfg):
        user, place = self._setup_user()
        # now = 17:00, fenêtre [17:00, 17:15] -> prépa-start 17:03 dedans.
        fake = timezone.make_aware(datetime(2026, 7, 20, 17, 0))
        RecurringBlock.objects.create(
            user=user, title='Job', block_type='work', day_of_week=fake.weekday(),
            start_time=time(18, 0), end_time=time(22, 0), place=place,
        )
        with patch('core.management.commands.send_reminders.timezone.localtime', return_value=fake):
            call_command('send_reminders', '--lead', '15')
        titles = [c.args[1] for c in mock_send.call_args_list]
        self.assertIn('Prépare-toi', titles)
        self.assertNotIn('Pars maintenant', titles)

    def test_leave_alert_fires_at_1715(self, mock_send, mock_cfg):
        user, place = self._setup_user()
        # now = 17:15, fenêtre [17:15, 17:30] -> départ 17:18 dedans.
        fake = timezone.make_aware(datetime(2026, 7, 20, 17, 15))
        RecurringBlock.objects.create(
            user=user, title='Job', block_type='work', day_of_week=fake.weekday(),
            start_time=time(18, 0), end_time=time(22, 0), place=place,
        )
        with patch('core.management.commands.send_reminders.timezone.localtime', return_value=fake):
            call_command('send_reminders', '--lead', '15')
        titles = [c.args[1] for c in mock_send.call_args_list]
        self.assertIn('Pars maintenant', titles)

    def test_no_place_no_departure_alert(self, mock_send, mock_cfg):
        user, _ = self._setup_user()
        fake = timezone.make_aware(datetime(2026, 7, 20, 17, 0))
        RecurringBlock.objects.create(
            user=user, title='Sans lieu', block_type='course', day_of_week=fake.weekday(),
            start_time=time(18, 0), end_time=time(22, 0),  # pas de place
        )
        with patch('core.management.commands.send_reminders.timezone.localtime', return_value=fake):
            call_command('send_reminders', '--lead', '15')
        titles = [c.args[1] for c in mock_send.call_args_list]
        self.assertNotIn('Prépare-toi', titles)
        self.assertNotIn('Pars maintenant', titles)
