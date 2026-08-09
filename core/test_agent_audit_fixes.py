"""Fixes issued from the capability audit:
- A3: create_goal/update_goal must accept a 'YYYY-MM-DD' deadline (used to crash
  because _goal_to_dict called .isoformat() on a raw string).
- A5: the destructive-tool guard (_user_authorized_destructive) gates
  requires_confirmation tools on an explicit user authorization.
- A6: cancel_scheduled_block cancels a one-off event (by date + optional title),
  handles ambiguity, not-found, overnight tail, and orphan-task cleanup.
"""
from datetime import date, time, timedelta

from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone

from core.models import Goal, ScheduledBlock, Task
from services.agent.agent import _user_authorized_destructive
from services.agent.tools.goals import CreateGoalTool, UpdateGoalTool
from services.agent.tools.schedule import CancelScheduledBlockTool


class GoalDeadlineTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("goaler", password="pw-123456")

    def test_create_goal_with_deadline_string(self):
        r = CreateGoalTool().execute(self.user, title="Diplôme", goal_type="long_term",
                                     deadline="2026-09-01")
        self.assertTrue(r.success, r.message)
        self.assertEqual(r.data["goal"]["deadline"], "2026-09-01")
        g = Goal.objects.get(id=r.data["goal"]["id"])
        self.assertEqual(g.deadline, date(2026, 9, 1))

    def test_create_goal_without_deadline(self):
        r = CreateGoalTool().execute(self.user, title="Sport", goal_type="short_term")
        self.assertTrue(r.success, r.message)
        self.assertIsNone(r.data["goal"]["deadline"])

    def test_create_goal_invalid_deadline_rejected(self):
        r = CreateGoalTool().execute(self.user, title="X", goal_type="short_term",
                                     deadline="pas-une-date")
        self.assertFalse(r.success)

    def test_update_goal_deadline(self):
        g = Goal.objects.create(user=self.user, title="Y", goal_type="short_term")
        r = UpdateGoalTool().execute(self.user, goal_id=g.id, deadline="2026-12-31")
        self.assertTrue(r.success, r.message)
        g.refresh_from_db()
        self.assertEqual(g.deadline, date(2026, 12, 31))


class DestructiveGuardTests(TestCase):
    def test_authorization_detected(self):
        for msg in ["efface tout", "supprime tout mon planning", "oui vas-y",
                    "je confirme", "d'accord supprime", "remets a zero"]:
            self.assertTrue(_user_authorized_destructive(msg), msg)

    def test_no_authorization(self):
        for msg in ["organise ma semaine", "j'ai quoi demain ?",
                    "ajoute du sport lundi", "c'est quoi mon prochain cours"]:
            self.assertFalse(_user_authorized_destructive(msg), msg)


class CancelScheduledBlockTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("canceller", password="pw-123456")
        self.tool = CancelScheduledBlockTool()
        self.day = timezone.localdate() + timedelta(days=1)

    def _sched(self, title, s, e, d=None):
        task = Task.objects.create(user=self.user, title=title)
        return ScheduledBlock.objects.create(
            user=self.user, task=task, date=d or self.day,
            start_time=time.fromisoformat(s), end_time=time.fromisoformat(e))

    def test_cancel_single_event_removes_block_and_orphan_task(self):
        sb = self._sched("Dentiste", "10:00", "11:00")
        r = self.tool.execute(self.user, date=self.day.isoformat())
        self.assertTrue(r.success, r.message)
        self.assertFalse(ScheduledBlock.objects.filter(id=sb.id).exists())
        self.assertFalse(Task.objects.filter(id=sb.task_id).exists())  # orphan cleaned

    def test_not_found(self):
        r = self.tool.execute(self.user, date=self.day.isoformat())
        self.assertFalse(r.success)
        self.assertIn("Aucun", r.message)

    def test_ambiguous_asks_which(self):
        self._sched("Dentiste", "10:00", "11:00")
        self._sched("Médecin", "14:00", "15:00")
        r = self.tool.execute(self.user, date=self.day.isoformat())
        self.assertFalse(r.success)
        self.assertIn("Lequel", r.message)
        # both still present (nothing deleted on ambiguity)
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 2)

    def test_title_disambiguates(self):
        self._sched("Dentiste", "10:00", "11:00")
        m = self._sched("Médecin", "14:00", "15:00")
        r = self.tool.execute(self.user, date=self.day.isoformat(), title="Dentiste")
        self.assertTrue(r.success, r.message)
        self.assertTrue(ScheduledBlock.objects.filter(id=m.id).exists())  # Médecin kept
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 1)

    def test_overnight_tail_removed(self):
        task = Task.objects.create(user=self.user, title="Quart nuit")
        ScheduledBlock.objects.create(user=self.user, task=task, date=self.day,
                                      start_time=time(22, 0), end_time=time(23, 59))
        ScheduledBlock.objects.create(user=self.user, task=task, date=self.day + timedelta(days=1),
                                      start_time=time(0, 0), end_time=time(6, 0))
        r = self.tool.execute(self.user, date=self.day.isoformat())
        self.assertTrue(r.success, r.message)
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 0)  # both pieces gone
