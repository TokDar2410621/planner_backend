"""
Audit test suite (Phase 2 of the remise-a-niveau).

Adds coverage for critical paths the original suite left untested:
tenant isolation / object-level authorization, the public share endpoints,
the AI chat endpoint wiring, LLM provider configuration, schedule-generation
correctness, and deployment config.

Tests decorated with ``@unittest.expectedFailure`` REPRODUCE a confirmed bug
found during the audit. They are expected to fail until the bug is fixed
(Phase 4). When a fix lands, the test flips to an "unexpected pass" (xpass),
which is the signal to remove the decorator. This keeps the suite green while
still documenting every defect with an executable repro.

Run:  python -m pytest core/test_audit.py -v
"""
import unittest
from datetime import time, timedelta
from unittest.mock import patch

from django.conf import settings
from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase

from .models import RecurringBlock, RecurringBlockCompletion, Task


# ---------------------------------------------------------------------------
# Tenant isolation / object-level authorization (IDOR)
# ---------------------------------------------------------------------------
class TenantIsolationTest(APITestCase):
    """A user must never read or mutate another user's data."""

    def setUp(self):
        self.alice = User.objects.create_user('alice', password='pw-alice-123')
        self.bob = User.objects.create_user('bob', password='pw-bob-12345')

        self.alice_task = Task.objects.create(user=self.alice, title="Alice private task")
        self.alice_block = RecurringBlock.objects.create(
            user=self.alice, title="Alice private class", block_type='course',
            day_of_week=0, start_time=time(9, 0), end_time=time(10, 30),
            location="Room A-101",
        )

    def _auth(self, user):
        self.client.force_authenticate(user=user)

    def test_user_cannot_read_others_task(self):
        self._auth(self.bob)
        resp = self.client.get(reverse('task-detail', kwargs={'pk': self.alice_task.id}))
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_user_cannot_modify_others_task(self):
        self._auth(self.bob)
        resp = self.client.patch(
            reverse('task-detail', kwargs={'pk': self.alice_task.id}),
            {'title': 'hijacked'},
        )
        self.assertIn(resp.status_code, (status.HTTP_403_FORBIDDEN, status.HTTP_404_NOT_FOUND))
        self.alice_task.refresh_from_db()
        self.assertEqual(self.alice_task.title, "Alice private task")

    def test_user_cannot_read_others_recurring_block(self):
        self._auth(self.bob)
        resp = self.client.get(
            reverse('recurring-block-detail', kwargs={'pk': self.alice_block.id})
        )
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)

    def test_user_cannot_complete_others_recurring_block(self):
        """REPRODUCES BUG: recurring-completions IDOR (views.py:456-477).

        The RecurringBlockCompletionSerializer.recurring_block field uses the
        default (unfiltered) queryset, so Bob can create a completion that
        references Alice's block. Expected: rejected (400/403/404).
        """
        self._auth(self.bob)
        resp = self.client.post(reverse('recurring-completion-list'), {
            'recurring_block': self.alice_block.id,
            'date': timezone.now().date().isoformat(),
        })
        self.assertNotIn(
            resp.status_code,
            (status.HTTP_200_OK, status.HTTP_201_CREATED),
            msg="Bob was able to write a completion against Alice's block (IDOR)",
        )


# ---------------------------------------------------------------------------
# Public / unauthenticated endpoints
# ---------------------------------------------------------------------------
class PublicEndpointTest(APITestCase):

    def setUp(self):
        self.alice = User.objects.create_user('alice', password='pw-alice-123')
        RecurringBlock.objects.create(
            user=self.alice, title="Secret night shift", block_type='course',
            day_of_week=0, start_time=time(9, 0), end_time=time(10, 30),
            location="Building B, Room 205",
        )

    def test_planning_by_username_is_not_public_without_optin(self):
        """REPRODUCES BUG: PublicPlanningByUsernameView PII leak (views.py:1011).

        Any user's full recurring schedule (titles, room locations, times) is
        returned to an anonymous caller keyed only on the (guessable) username,
        with no opt-in / share-token check. Expected: not exposed unless the
        user explicitly opted into public sharing.
        """
        resp = self.client.get(reverse('public-planning', kwargs={'username': 'alice'}))
        leaked = resp.status_code == status.HTTP_200_OK and resp.data.get('recurring_blocks')
        self.assertFalse(
            leaked,
            msg="Anonymous caller obtained Alice's schedule + room locations with no consent",
        )

    def test_unknown_username_behaviour(self):
        """Unknown username should 404 (documents current behaviour)."""
        resp = self.client.get(reverse('public-planning', kwargs={'username': 'nobody'}))
        self.assertEqual(resp.status_code, status.HTTP_404_NOT_FOUND)


# ---------------------------------------------------------------------------
# AI chat endpoint wiring (LLM mocked — no network / API key needed)
# ---------------------------------------------------------------------------
class ChatEndpointTest(APITestCase):

    def setUp(self):
        self.user = User.objects.create_user('chatuser', password='pw-chat-123')

    def test_chat_requires_authentication(self):
        resp = self.client.post(reverse('chat'), {'message': 'hi'})
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_chat_empty_payload_rejected(self):
        self.client.force_authenticate(user=self.user)
        resp = self.client.post(reverse('chat'), {})
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_chat_returns_agent_response(self):
        """View wiring works when the agent succeeds (agent mocked)."""
        self.client.force_authenticate(user=self.user)
        fake = {'response': 'Bonjour !', 'quick_replies': [], 'blocks_created': [], 'tasks_created': []}
        with patch('core.views.PlannerAgent') as MockAgent:
            MockAgent.return_value.process_message.return_value = fake
            resp = self.client.post(reverse('chat'), {'message': 'salut'})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data['response'], 'Bonjour !')


# ---------------------------------------------------------------------------
# LLM provider configuration
# ---------------------------------------------------------------------------
class LLMProviderConfigTest(SimpleTestCase):

    def test_claude_default_model_is_not_retired(self):
        """REPRODUCES BUG: claude.py:29 pins a retired model id.

        'claude-sonnet-4-20250514' returns 404 not_found_error from the API,
        so every Claude chat turn fails with "Erreur ... IA". Expected: a
        current model id (e.g. claude-sonnet-5 / claude-opus-4-8).
        """
        from services.llm.claude import ClaudeProvider
        self.assertNotEqual(ClaudeProvider.DEFAULT_MODEL, 'claude-sonnet-4-20250514')


class AgentProviderSelectionTest(TestCase):

    def test_agent_respects_configured_provider(self):
        """REPRODUCES BUG: agent.py:38 hardcodes ClaudeProvider().

        PlannerAgent ignores settings.LLM_PROVIDER (default 'gemini') and the
        user's profile.preferred_llm. Expected: with the gemini provider
        configured, the agent uses GeminiProvider.
        """
        from services.agent.agent import PlannerAgent
        from services.llm.gemini import GeminiProvider
        with self.settings(LLM_PROVIDER='gemini'):
            agent = PlannerAgent()
            self.assertIsInstance(agent.llm, GeminiProvider)


# ---------------------------------------------------------------------------
# Schedule-generation correctness (the original suite only smoke-tests HTTP 200)
# ---------------------------------------------------------------------------
class ScheduleGenerationCorrectnessTest(APITestCase):

    def setUp(self):
        self.user = User.objects.create_user('planner', password='pw-plan-123')
        self.client.force_authenticate(user=self.user)
        self.today = timezone.now().date()
        # A busy block on today's weekday, 09:00-11:00.
        self.block = RecurringBlock.objects.create(
            user=self.user, title="Busy block", block_type='course',
            day_of_week=self.today.weekday(),
            start_time=time(9, 0), end_time=time(11, 0),
        )
        Task.objects.create(
            user=self.user, title="Deep work", estimated_duration_minutes=60,
            task_type='deep_work', priority=8,
        )

    @staticmethod
    def _overlap(a_start, a_end, b_start, b_end):
        return a_start < b_end and b_start < a_end

    def test_generated_tasks_do_not_overlap_recurring_block(self):
        """The 'no task on task' guarantee: no generated block may collide with
        the recurring busy block on the same day."""
        resp = self.client.post(reverse('schedule-generate'), {'force': True})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        created = resp.data.get('created_blocks', [])
        for b in created:
            if str(b.get('date')) != self.today.isoformat():
                continue
            self.assertFalse(
                self._overlap(
                    str(b['start_time']), str(b['end_time']),
                    self.block.start_time.isoformat(), self.block.end_time.isoformat(),
                ),
                msg=f"Generated block {b['start_time']}-{b['end_time']} overlaps the recurring block",
            )

    def test_generated_tasks_within_working_hours(self):
        """Generated blocks stay within the 08:00-22:00 working window."""
        resp = self.client.post(reverse('schedule-generate'), {'force': True})
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        for b in resp.data.get('created_blocks', []):
            self.assertGreaterEqual(str(b['start_time']), '08:00:00')
            self.assertLessEqual(str(b['end_time']), '22:00:00')


# ---------------------------------------------------------------------------
# Deployment configuration
# ---------------------------------------------------------------------------
class DeploymentConfigTest(SimpleTestCase):

    def test_token_blacklist_app_installed(self):
        """REPRODUCES BUG: settings enable BLACKLIST_AFTER_ROTATION but the
        token_blacklist app is not installed (settings.py:124), so rotated
        refresh tokens are never revoked and logout cannot invalidate them.
        """
        self.assertIn('rest_framework_simplejwt.token_blacklist', settings.INSTALLED_APPS)
