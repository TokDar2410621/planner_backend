"""
Lot 1 rétention: pardon (roll-over), streak élastique, reset-day, bilan hebdo.
"""
from datetime import time, timedelta

from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase
from django.test import TestCase

from core.models import (
    RecurringBlockCompletion, RecurringBlock, ScheduledBlock, Task, SchedulePlanChange,
)
from services.rollover import roll_over_missed
from services.streak import compute_streak, FREEZE_BUDGET
from services.progress import weekly_summary


class RollOverTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('roll', password='pw-rol-123456')
        self.today = timezone.localdate()

    def test_missed_block_is_reprogrammed_not_left_in_past(self):
        task = Task.objects.create(user=self.user, title='TP', estimated_duration_minutes=60)
        # A block 3 days ago, never done.
        ScheduledBlock.objects.create(
            user=self.user, task=task, date=self.today - timedelta(days=3),
            start_time=time(9, 0), end_time=time(10, 0), actually_completed=False,
        )
        report = roll_over_missed(self.user)
        self.assertEqual(report['rolled'], 1)
        # Old past block gone; a new one exists today or later (no red debt).
        self.assertFalse(
            ScheduledBlock.objects.filter(
                user=self.user, date__lt=self.today, actually_completed=False
            ).exists()
        )
        self.assertTrue(ScheduledBlock.objects.filter(user=self.user, task=task, date__gte=self.today).exists())

    def test_completed_task_not_resurrected(self):
        task = Task.objects.create(user=self.user, title='Fait', completed=True, completed_at=timezone.now())
        ScheduledBlock.objects.create(
            user=self.user, task=task, date=self.today - timedelta(days=1),
            start_time=time(9, 0), end_time=time(10, 0), actually_completed=False,
        )
        report = roll_over_missed(self.user)
        self.assertEqual(report['rolled'], 0)


class StreakTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('streak', password='pw-stk-123456')
        self.today = timezone.localdate()

    def _complete_on(self, d):
        block = RecurringBlock.objects.create(
            user=self.user, title='C', block_type='course', day_of_week=d.weekday(),
            start_time=time(9, 0), end_time=time(10, 0),
        )
        RecurringBlockCompletion.objects.create(user=self.user, recurring_block=block, date=d)

    def test_consecutive_days_build_streak(self):
        for i in range(1, 4):  # yesterday, -2, -3
            self._complete_on(self.today - timedelta(days=i))
        s = compute_streak(self.user)
        self.assertGreaterEqual(s['current_streak'], 3)

    def test_gap_within_freeze_budget_does_not_break(self):
        # active -1 and -3, gap at -2 (1 miss <= budget) -> streak spans.
        self._complete_on(self.today - timedelta(days=1))
        self._complete_on(self.today - timedelta(days=3))
        s = compute_streak(self.user)
        self.assertEqual(s['current_streak'], 2)
        self.assertEqual(s['freezes_used'], 1)

    def test_plan_adjustment_counts_as_active(self):
        SchedulePlanChange.objects.create(
            user=self.user, date=self.today - timedelta(days=1),
            before=[], after=[], status='applied',
        )
        s = compute_streak(self.user)
        self.assertGreaterEqual(s['current_streak'], 1)

    def test_too_many_gaps_break_streak(self):
        # active only -1, then nothing (misses at -2,-3,-4 > budget 2).
        self._complete_on(self.today - timedelta(days=1))
        self._complete_on(self.today - timedelta(days=5))  # beyond the break
        s = compute_streak(self.user)
        self.assertEqual(s['current_streak'], 1)  # -5 not reached (broke after budget)


class WeeklySummaryTest(TestCase):
    def test_summary_counts_completed_work_positively(self):
        user = User.objects.create_user('sum', password='pw-sum-123456')
        today = timezone.localdate()
        task = Task.objects.create(user=user, title='Rev')
        ScheduledBlock.objects.create(
            user=user, task=task, date=today,
            start_time=time(9, 0), end_time=time(11, 0),
            actually_completed=True, actual_duration_minutes=120,
        )
        r = weekly_summary(user)
        self.assertEqual(r['completed_minutes'], 120)
        self.assertEqual(r['completed_hours'], 2.0)
        self.assertIn('cette semaine', r['message'])


class RetentionEndpointsTest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user('reten', password='pw-ret-123456')
        self.client.force_authenticate(self.user)

    def test_rollover_endpoint(self):
        r = self.client.post(reverse('schedule-rollover'))
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertIn('rolled', r.data)

    def test_reset_day_endpoint(self):
        r = self.client.post(reverse('schedule-reset-day'))
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertIn('count', r.data)

    def test_streak_endpoint(self):
        r = self.client.get(reverse('streak'))
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertIn('current_streak', r.data)

    def test_weekly_summary_endpoint(self):
        r = self.client.get(reverse('weekly-summary'))
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertIn('message', r.data)
