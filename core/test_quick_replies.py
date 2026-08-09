"""quick_replies are now LLM-generated (contextual next steps) with the old
rule-based generator as a deterministic fallback. Tests cover the JSON parsing,
the fallback path, and the had_error short-circuit. No real LLM is called.
"""
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase

from services.agent.agent import PlannerAgent


def _fake_llm(text=None, available=True, raise_on_generate=False):
    class _LLM:
        def is_available(self):
            return available

        def generate(self, prompt, tools=None, system_prompt=None):
            if raise_on_generate:
                raise RuntimeError("boom")
            return SimpleNamespace(text=text, is_error=False)
    return _LLM()


class ParseQuickRepliesTests(TestCase):
    def test_valid_json_array(self):
        out = PlannerAgent._parse_quick_replies(
            '[{"label": "📅 Voir", "value": "Montre mon planning"}, '
            '{"label": "➕ Cours", "value": "Ajoute un autre cours"}]')
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], {"label": "📅 Voir", "value": "Montre mon planning"})

    def test_json_embedded_in_prose(self):
        out = PlannerAgent._parse_quick_replies(
            'Voici: [{"label":"a","value":"b"}] merci')
        self.assertEqual(out, [{"label": "a", "value": "b"}])

    def test_invalid_json_returns_empty(self):
        self.assertEqual(PlannerAgent._parse_quick_replies("pas de json ici"), [])
        self.assertEqual(PlannerAgent._parse_quick_replies("[oops"), [])

    def test_dedup_and_cap_three(self):
        out = PlannerAgent._parse_quick_replies(
            '[{"label":"l1","value":"same"},{"label":"l2","value":"same"},'
            '{"label":"l3","value":"x"},{"label":"l4","value":"y"},{"label":"l5","value":"z"}]')
        self.assertEqual(len(out), 3)  # capped
        values = [o["value"] for o in out]
        self.assertEqual(values, ["same", "x", "y"])  # duplicate "same" dropped

    def test_skips_incomplete_items(self):
        out = PlannerAgent._parse_quick_replies(
            '[{"label":"","value":"v"},{"label":"ok","value":""},{"label":"ok","value":"v"}]')
        self.assertEqual(out, [{"label": "ok", "value": "v"}])


class GenerateQuickRepliesTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("qr", password="pw-123456")
        self.agent = PlannerAgent()

    def test_had_error_returns_empty(self):
        self.agent.llm = _fake_llm(text='[{"label":"a","value":"b"}]')
        out = self.agent._generate_quick_replies("hi", "reply", [], {}, had_error=True)
        self.assertEqual(out, [])

    def test_llm_path_used_when_it_returns_replies(self):
        self.agent.llm = _fake_llm(text='[{"label":"➕ Cours","value":"Ajoute un cours"}]')
        out = self.agent._generate_quick_replies("hi", "J'ai ajouté ton cours.", [], {})
        self.assertEqual(out, [{"label": "➕ Cours", "value": "Ajoute un cours"}])

    def test_falls_back_to_rules_when_llm_raises(self):
        self.agent.llm = _fake_llm(raise_on_generate=True)
        ctx = {"total_blocks": 3, "tasks": {"pending_count": 0}}
        tool_calls = [{"tool": "create_block", "args": {}, "result": {}}]
        out = self.agent._generate_quick_replies("ajoute un cours", "C'est fait.", tool_calls, ctx)
        # rule-based create_block suggestions
        self.assertTrue(any("planning" in r["value"].lower() for r in out))

    def test_falls_back_when_llm_returns_nothing(self):
        self.agent.llm = _fake_llm(text="pas de json")
        ctx = {"total_blocks": 0, "tasks": {"pending_count": 0}}
        out = self.agent._generate_quick_replies("salut", "Bonjour !", [], ctx)
        # rule-based: 0 blocks -> "Configurer mon planning"
        self.assertTrue(any("configurer" in r["label"].lower() for r in out))


class QuickRepliesForTests(TestCase):
    """Deferred endpoint path: quick_replies_for(user, msg, response)."""

    def setUp(self):
        self.user = User.objects.create_user("qrf", password="pw-123456")
        self.agent = PlannerAgent()

    def test_returns_llm_replies(self):
        with patch.object(
            self.agent, "_build_provider",
            return_value=_fake_llm(text='[{"label":"➕ Cours","value":"Ajoute un cours"}]'),
        ):
            out = self.agent.quick_replies_for(self.user, "hi", "J'ai ajouté ton cours.")
        self.assertEqual(out, [{"label": "➕ Cours", "value": "Ajoute un cours"}])

    def test_empty_inputs_return_empty(self):
        # Court-circuit AVANT tout appel LLM (aucun provider construit).
        self.assertEqual(self.agent.quick_replies_for(self.user, "", "reply"), [])
        self.assertEqual(self.agent.quick_replies_for(self.user, "hi", ""), [])

    def test_unavailable_provider_returns_empty(self):
        with patch.object(
            self.agent, "_build_provider", return_value=_fake_llm(available=False),
        ):
            out = self.agent.quick_replies_for(self.user, "hi", "reply")
        self.assertEqual(out, [])
