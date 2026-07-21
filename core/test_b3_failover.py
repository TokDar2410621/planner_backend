"""
B3 regression: the agent must fail over to the alternate LLM provider and must
never persist an LLM-failure message as a real assistant turn.
"""
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase

from core.models import ConversationMessage
from services.llm.base import LLMResponse


class _FakeProvider:
    def __init__(self, name, response):
        self._name = name
        self._response = response

    def is_available(self):
        return True

    def generate_with_history(self, messages, tools, system_prompt):
        return self._response

    @property
    def name(self):
        return self._name


class AgentFailoverTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('failover_user', password='pw-failover-1')
        # Primary = claude, so the alternate is gemini.
        self.user.profile.preferred_llm = 'claude'
        self.user.profile.save()

    def _run_with(self, claude_resp, gemini_resp):
        from services.agent.agent import PlannerAgent

        def fake_get_provider(name=None, user=None):
            if name == 'claude':
                return _FakeProvider('claude', claude_resp)
            return _FakeProvider('gemini', gemini_resp)

        with patch('services.agent.agent.get_provider', side_effect=fake_get_provider):
            agent = PlannerAgent()
            return agent.process_message(self.user, "planifie ma journée")

    def test_falls_over_to_alternate_provider(self):
        err = LLMResponse(text="Erreur lors de la communication avec l'IA.", is_error=True)
        ok = LLMResponse(text="Réponse de secours", is_error=False)

        result = self._run_with(err, ok)

        self.assertEqual(result['response'], "Réponse de secours")
        # The successful fallback reply IS persisted...
        self.assertTrue(
            ConversationMessage.objects.filter(
                user=self.user, role='assistant', content="Réponse de secours"
            ).exists()
        )
        # ...and the primary error string is NEVER persisted as an assistant turn.
        self.assertFalse(
            ConversationMessage.objects.filter(
                user=self.user, role='assistant', content__icontains="Erreur"
            ).exists()
        )

    def test_both_providers_fail_does_not_persist_error_turn(self):
        err1 = LLMResponse(text="Erreur lors de la communication avec l'IA.", is_error=True)
        err2 = LLMResponse(text="Erreur Gemini.", is_error=True)

        result = self._run_with(err1, err2)

        # The user still sees an error message in the response...
        self.assertTrue(result['response'])
        # ...but no assistant turn is persisted (would pollute future context).
        self.assertEqual(
            ConversationMessage.objects.filter(user=self.user, role='assistant').count(), 0
        )
        # The user's own message is still recorded.
        self.assertEqual(
            ConversationMessage.objects.filter(user=self.user, role='user').count(), 1
        )
