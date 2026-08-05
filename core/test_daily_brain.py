"""Cerveau quotidien (daily_brain): brief 1x/jour/user, signaux réels, push gated.

Couvre: échéances (tâches 72h + objectifs 7j), conflits fixe/fixe, idempotence
(cache par (user, date)), commande (garde horaire + batch), endpoint GET
/insights/daily-brief/, et le push (uniquement si actionnable).
"""
from datetime import date, time, timedelta
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone
from io import StringIO
from rest_framework.test import APIClient

from core.models import DailyBrainReport, Goal, RecurringBlock, Task
from services.daily_brain import build_daily_brief, push_daily_brief


class BuildDailyBriefTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("brainy", password="pw-123456")

    def test_deadline_task_within_72h_is_reported(self):
        Task.objects.create(
            user=self.user, title="Rendre le rapport",
            deadline=timezone.now() + timedelta(days=1))
        payload = build_daily_brief(self.user)
        texts = [i["text"] for i in payload["items"]]
        self.assertTrue(any("Rendre le rapport" in t for t in texts), texts)

    def test_completed_or_far_tasks_are_ignored(self):
        Task.objects.create(
            user=self.user, title="Déjà fait", completed=True,
            deadline=timezone.now() + timedelta(days=1))
        Task.objects.create(
            user=self.user, title="Dans un mois",
            deadline=timezone.now() + timedelta(days=30))
        payload = build_daily_brief(self.user)
        texts = " ".join(i["text"] for i in payload["items"])
        self.assertNotIn("Déjà fait", texts)
        self.assertNotIn("Dans un mois", texts)

    def test_active_goal_deadline_within_7d_is_reported(self):
        Goal.objects.create(
            user=self.user, title="Finir le mémoire",
            deadline=timezone.localdate() + timedelta(days=5), progress=40)
        payload = build_daily_brief(self.user)
        texts = [i["text"] for i in payload["items"]]
        self.assertTrue(any("mémoire" in t for t in texts), texts)

    def test_fixed_fixed_conflict_is_reported(self):
        # deux FIXES qui se chevauchent (créés en ORM direct, la garde REST
        # l'empêcherait) demain -> le détecteur produit doit le remonter.
        dow = (timezone.localdate() + timedelta(days=1)).weekday()
        RecurringBlock.objects.create(
            user=self.user, title="Cours A", block_type="course", day_of_week=dow,
            start_time=time(9, 0), end_time=time(11, 0), flexibility="fixed")
        RecurringBlock.objects.create(
            user=self.user, title="Cours B", block_type="course", day_of_week=dow,
            start_time=time(10, 0), end_time=time(12, 0), flexibility="fixed")
        payload = build_daily_brief(self.user)
        self.assertTrue(
            any(i["type"] == "conflict" for i in payload["items"]),
            payload["items"])

    def test_idempotent_per_user_and_day(self):
        Task.objects.create(
            user=self.user, title="Unique",
            deadline=timezone.now() + timedelta(days=1))
        p1 = build_daily_brief(self.user)
        # un 2e signal apparaît APRÈS le premier calcul: le brief du jour est
        # figé (cache), pas recalculé.
        Task.objects.create(
            user=self.user, title="Arrivé après",
            deadline=timezone.now() + timedelta(days=1))
        p2 = build_daily_brief(self.user)
        self.assertEqual(p1, p2)
        self.assertEqual(
            DailyBrainReport.objects.filter(user=self.user).count(), 1)

    def test_quiet_user_gets_empty_items_and_no_push(self):
        payload = build_daily_brief(self.user)
        self.assertEqual(payload["items"], [])
        self.assertEqual(push_daily_brief(self.user, payload), 0)

    def test_push_sent_only_when_actionable_and_configured(self):
        payload = {"items": [{"type": "deadline", "text": "Échéance X"}]}
        with patch("services.daily_brain.push_configured", return_value=True), \
             patch("services.daily_brain.send_to_user", return_value=2) as send:
            n = push_daily_brief(self.user, payload)
        self.assertEqual(n, 2)
        kwargs = send.call_args.kwargs
        args = send.call_args.args
        # (user, title, body, url=...)
        self.assertIn("Échéance X", args[2] if len(args) > 2 else kwargs.get("body", ""))


class DailyBrainCommandTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("cmdbrain", password="pw-123456")
        Task.objects.create(
            user=self.user, title="Examen",
            deadline=timezone.now() + timedelta(days=2))

    def test_force_processes_user_and_creates_report(self):
        out = StringIO()
        call_command("daily_brain", "--force", "--user", str(self.user.id), stdout=out)
        self.assertTrue(
            DailyBrainReport.objects.filter(
                user=self.user, date=timezone.localdate()).exists())
        self.assertIn("1 user(s)", out.getvalue())

    def test_hour_gate_skips_early(self):
        out = StringIO()
        with patch("core.management.commands.daily_brain.timezone.localtime") as lt:
            lt.return_value = timezone.now().replace(hour=3, minute=0)
            call_command("daily_brain", stdout=out)
        self.assertIn("rien à faire", out.getvalue())
        self.assertFalse(DailyBrainReport.objects.exists())

    def test_batch_skips_already_processed_users(self):
        call_command("daily_brain", "--force", stdout=StringIO())
        self.assertEqual(DailyBrainReport.objects.count(), 1)
        # 2e passe du worker le même jour: personne à retraiter
        out = StringIO()
        call_command("daily_brain", "--force", stdout=out)
        self.assertIn("0 user(s)", out.getvalue())
        self.assertEqual(DailyBrainReport.objects.count(), 1)


class DailyBriefEndpointTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("apibrain", password="pw-123456")
        self.client_api = APIClient()
        self.client_api.force_authenticate(self.user)

    def test_get_returns_todays_brief(self):
        Task.objects.create(
            user=self.user, title="Rendu API",
            deadline=timezone.now() + timedelta(days=1))
        resp = self.client_api.get("/api/insights/daily-brief/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["date"], timezone.localdate().isoformat())
        self.assertTrue(
            any("Rendu API" in i["text"] for i in resp.data["items"]))

    def test_requires_auth(self):
        resp = APIClient().get("/api/insights/daily-brief/")
        self.assertEqual(resp.status_code, 401)
