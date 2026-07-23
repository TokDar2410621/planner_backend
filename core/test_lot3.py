"""
Lot 3: ancres/anti-thrash (replan) + social co-présence (disponibilité commune).
"""
from datetime import time, timedelta

from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase
from django.test import TestCase

from core.models import Connection, RecurringBlock, ScheduledBlock, Task, SchedulePlanChange
from services.replan import replan_after_delay, MIN_SHIFT_MINUTES, MAX_AUTO_CHANGES_PER_DAY
from services.social import free_intervals, common_free


class AntiThrashTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('thrash', password='pw-thr-123456')
        self.user.profile.automation_mode = 'automatic'
        self.user.profile.save()
        self.today = timezone.localdate()

    def _block(self, task, hh, locked=False):
        return ScheduledBlock.objects.create(
            user=self.user, task=task, date=self.today,
            start_time=time(hh, 0), end_time=time(hh + 1, 0), locked=locked,
        )

    def test_trivial_delay_is_noop(self):
        res = replan_after_delay(self.user, delay_minutes=MIN_SHIFT_MINUTES - 1)
        self.assertTrue(res['applied'])
        self.assertEqual(res['moved'], [])
        self.assertIsNone(res['token'])

    def test_locked_block_is_never_displaced(self):
        task = Task.objects.create(user=self.user, title='Sacré', estimated_duration_minutes=60)
        self._block(task, 9, locked=True)  # locked, before resume
        res = replan_after_delay(self.user, resume_time='11:00')
        # Locked block not moved -> nothing to replan.
        self.assertEqual(res['moved'], [])
        self.assertTrue(ScheduledBlock.objects.filter(user=self.user, task=task, start_time=time(9, 0)).exists())

    def test_change_budget_downgrades_auto_to_proposal(self):
        # Saturate today's applied-change budget.
        for _ in range(MAX_AUTO_CHANGES_PER_DAY):
            SchedulePlanChange.objects.create(
                user=self.user, date=self.today, before=[], after=[], status='applied'
            )
        task = Task.objects.create(user=self.user, title='Sport', estimated_duration_minutes=60)
        self._block(task, 9)
        res = replan_after_delay(self.user, resume_time='11:00')
        # Auto but over budget -> proposed, not applied (anti-thrash).
        self.assertFalse(res['applied'])


class SocialAvailabilityUnitTest(TestCase):
    def setUp(self):
        self.a = User.objects.create_user('alice', password='pw-ali-123456')
        self.b = User.objects.create_user('bob', password='pw-bob-123456')
        # Monday
        self.date = timezone.localdate()
        while self.date.weekday() != 0:
            self.date += timedelta(days=1)

    def _course(self, u, s, e):
        RecurringBlock.objects.create(
            user=u, title='Cours', block_type='course', day_of_week=self.date.weekday(),
            start_time=s, end_time=e,
        )

    def test_common_free_excludes_busy_overlap(self):
        # Alice busy 9-11; Bob busy 10-12. Common free (8-22) excludes 9-12.
        self._course(self.a, time(9, 0), time(11, 0))
        self._course(self.b, time(10, 0), time(12, 0))
        slots = common_free(self.a, self.b, self.date, min_minutes=30)
        # Free before 9 (8-9) and after 12 (12-22) are common.
        starts = {s['start'] for s in slots}
        self.assertIn('08:00', starts)
        self.assertIn('12:00', starts)
        # 09:00-12:00 must not appear as free for both.
        for s in slots:
            self.assertFalse(s['start'] < '12:00' and s['end'] > '09:00' and s['start'] >= '09:00')

    def test_free_intervals_basic(self):
        self._course(self.a, time(13, 0), time(15, 0))
        free = free_intervals(self.a, self.date)
        # Should include a slot ending at 13:00 and one starting at 15:00.
        mins = [(s, e) for s, e in free]
        self.assertTrue(any(e == 13 * 60 for s, e in mins))
        self.assertTrue(any(s == 15 * 60 for s, e in mins))


class SocialEndpointsTest(APITestCase):
    def setUp(self):
        self.alice = User.objects.create_user('alice2', password='pw-al2-123456')
        self.bob = User.objects.create_user('bob2', password='pw-bo2-123456')

    def test_connect_accept_and_availability_flow(self):
        # Alice sends request to Bob.
        self.client.force_authenticate(self.alice)
        r = self.client.post(reverse('social-connect'), {'username': 'bob2'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_201_CREATED)
        cid = r.data['connection_id']

        # Availability before acceptance -> 403 (not friends yet).
        r = self.client.get(reverse('social-availability'), {'friend': self.bob.id})
        self.assertEqual(r.status_code, status.HTTP_403_FORBIDDEN)

        # Bob accepts.
        self.client.force_authenticate(self.bob)
        r = self.client.post(reverse('social-accept', kwargs={'connection_id': cid}))
        self.assertEqual(r.status_code, status.HTTP_200_OK)

        # Now Alice sees Bob as a friend + can query availability.
        self.client.force_authenticate(self.alice)
        r = self.client.get(reverse('social-connections'))
        self.assertEqual(len(r.data['friends']), 1)
        self.assertEqual(r.data['friends'][0]['username'], 'bob2')
        r = self.client.get(reverse('social-availability'), {'friend': self.bob.id})
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertIn('common_free', r.data)

    def test_reverse_request_auto_accepts(self):
        # Bob requests Alice, then Alice requests Bob -> auto-accepted (mutual).
        self.client.force_authenticate(self.bob)
        self.client.post(reverse('social-connect'), {'username': 'alice2'}, format='json')
        self.client.force_authenticate(self.alice)
        r = self.client.post(reverse('social-connect'), {'username': 'bob2'}, format='json')
        self.assertEqual(r.data['status'], 'accepted')

    def test_cannot_connect_to_self(self):
        self.client.force_authenticate(self.alice)
        r = self.client.post(reverse('social-connect'), {'username': 'alice2'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_400_BAD_REQUEST)
