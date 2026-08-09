"""
Regression: the agent's tool definitions must convert into a valid google-genai
tool config. Passing the raw Claude-format dicts broke prod on google-genai 1.7x
(pydantic 'Input should be callable' / 'Extra inputs are not permitted'), which
killed the whole chat. This test builds the config on the INSTALLED google-genai
version, so a future version skew is caught before deploy.
"""
from django.test import SimpleTestCase


class GeminiToolConversionTest(SimpleTestCase):
    def test_all_tools_convert_and_config_builds(self):
        from google.genai import types
        from services.llm.gemini import GeminiProvider
        from services.agent.tools import get_tools_for_claude

        tools = get_tools_for_claude()
        converted = GeminiProvider()._to_gemini_tools(tools)

        self.assertIsNotNone(converted)
        self.assertEqual(len(converted), 1)
        self.assertEqual(len(converted[0].function_declarations), len(tools))

        # This exact call raised 92 validation errors in prod with raw dicts.
        types.GenerateContentConfig(
            tools=converted,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            ),
        )

    def test_empty_tools_returns_none(self):
        from services.llm.gemini import GeminiProvider
        self.assertIsNone(GeminiProvider()._to_gemini_tools(None))
        self.assertIsNone(GeminiProvider()._to_gemini_tools([]))


class GeminiThinkingConfigTest(SimpleTestCase):
    """Regression: un thinking_budget EXPLICITE est requis. En mode dynamique
    (défaut -1), gemini-2.5-flash + outils + gros system prompt renvoie par
    intermittence un candidat VIDE (STOP, 0 part) et l'agent reste muet. Un
    budget fixe supprime ce comportement (reproduit: dynamique = vide,
    0/512/1024/2048 = OK)."""

    def test_build_config_sets_explicit_non_dynamic_thinking_budget(self):
        from services.llm.gemini import GeminiProvider
        cfg = GeminiProvider()._build_config(gemini_tools=None)
        self.assertIsNotNone(getattr(cfg, "thinking_config", None))
        self.assertEqual(cfg.thinking_config.thinking_budget, GeminiProvider.THINKING_BUDGET)
        self.assertGreaterEqual(GeminiProvider.THINKING_BUDGET, 0)  # jamais -1 (dynamique)

    def test_build_config_keeps_tools_and_budget_together(self):
        from services.llm.gemini import GeminiProvider
        from services.agent.tools import get_tools_for_claude
        p = GeminiProvider()
        cfg = p._build_config(p._to_gemini_tools(get_tools_for_claude()))
        self.assertIsNotNone(getattr(cfg, "thinking_config", None))
        self.assertTrue(cfg.tools)


class GeminiParseResponseRobustnessTest(SimpleTestCase):
    """Regression: a candidate that exposes no parts (SAFETY / RECITATION /
    MAX_TOKENS with empty content, or content=None) yields response.parts=None.
    Iterating it raised "'NoneType' object is not iterable"; with the Claude
    failover capped, that crash surfaced to users as "Erreur ... avec l'IA"."""

    def _make_response(self, content):
        from types import SimpleNamespace
        candidate = SimpleNamespace(content=content, finish_reason=SimpleNamespace(name="SAFETY"))
        return SimpleNamespace(candidates=[candidate], usage_metadata=None)

    def test_candidate_without_content_does_not_crash(self):
        from services.llm.gemini import GeminiProvider
        parsed = GeminiProvider()._parse_response(self._make_response(content=None))
        self.assertEqual(parsed.text, "")
        self.assertEqual(parsed.function_calls, [])
        self.assertEqual(parsed.stop_reason, "SAFETY")

    def test_candidate_with_none_parts_does_not_crash(self):
        from types import SimpleNamespace
        from services.llm.gemini import GeminiProvider
        resp = self._make_response(content=SimpleNamespace(parts=None))
        parsed = GeminiProvider()._parse_response(resp)
        self.assertEqual(parsed.text, "")
        self.assertEqual(parsed.function_calls, [])
