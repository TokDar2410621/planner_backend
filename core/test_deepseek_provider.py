"""DeepSeek provider: OpenAI-compatible translation, thinking/tool wiring, and the
mandatory reasoning_content echo across tool turns (without it DeepSeek returns
HTTP 400). The DeepSeek API itself is never called; the client is mocked.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from django.test import TestCase, override_settings

from services.llm.deepseek import DeepSeekProvider


def _resp(content=None, reasoning=None, tool_calls=None, finish="stop"):
    """Build a fake OpenAI-shaped chat completion response object."""
    msg = SimpleNamespace(content=content, reasoning_content=reasoning,
                          tool_calls=tool_calls or None)
    choice = SimpleNamespace(message=msg, finish_reason=finish)
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
    return SimpleNamespace(choices=[choice], usage=usage)


def _tool_call(call_id, name, args_obj):
    return SimpleNamespace(
        id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args_obj)),
    )


@override_settings(DEEPSEEK_API_KEY="sk-test", DEEPSEEK_THINKING=True,
                   DEEPSEEK_REASONING_EFFORT="high", DEEPSEEK_MODEL="deepseek-v4-pro")
class DeepSeekProviderTests(TestCase):
    def _provider(self):
        p = DeepSeekProvider()
        p._client = MagicMock()
        return p

    def test_is_available(self):
        self.assertTrue(self._provider().is_available())

    @override_settings(DEEPSEEK_API_KEY="")
    def test_unavailable_without_key(self):
        p = DeepSeekProvider()
        self.assertFalse(p.is_available())
        r = p.generate_with_history([{"role": "user", "content": "hi"}])
        self.assertTrue(r.is_error)

    def test_convert_tools_to_openai_shape(self):
        anthropic = [{"name": "list_blocks", "description": "liste",
                      "input_schema": {"type": "object", "properties": {"d": {"type": "integer"}}}}]
        out = DeepSeekProvider._convert_tools(anthropic)
        self.assertEqual(out[0]["type"], "function")
        self.assertEqual(out[0]["function"]["name"], "list_blocks")
        self.assertEqual(out[0]["function"]["parameters"]["properties"]["d"]["type"], "integer")

    def test_tool_results_become_tool_role_messages(self):
        p = self._provider()
        messages = [
            {"role": "user", "content": "planifie"},
            {"role": "assistant", "content": {"role": "assistant", "content": None,
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "x", "arguments": "{}"}}]}},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "c1", "name": "x", "content": "ok"}]},
        ]
        out = p._to_openai_messages(messages, "SYS")
        self.assertEqual(out[0], {"role": "system", "content": "SYS"})
        # the assistant dict is passed through verbatim (the echo)
        self.assertEqual(out[2]["tool_calls"][0]["id"], "c1")
        # tool result mapped to role:tool keyed by tool_call_id
        self.assertEqual(out[3], {"role": "tool", "tool_call_id": "c1", "content": "ok"})

    def test_parse_response_tool_call_keeps_reasoning_for_echo(self):
        p = self._provider()
        resp = _resp(content="", reasoning="je réfléchis...",
                     tool_calls=[_tool_call("c9", "create_block", {"title": "Sport"})],
                     finish="tool_calls")
        r = p._parse_response(resp)
        self.assertTrue(r.has_function_calls)
        self.assertEqual(r.function_calls[0].name, "create_block")
        self.assertEqual(r.function_calls[0].args, {"title": "Sport"})
        self.assertEqual(r.function_calls[0].call_id, "c9")
        # raw_content MUST carry reasoning_content + tool_calls (anti-400 echo)
        self.assertEqual(r.raw_content["reasoning_content"], "je réfléchis...")
        self.assertEqual(r.raw_content["tool_calls"][0]["id"], "c9")
        self.assertIsNone(r.raw_content["content"])  # no text -> null content
        self.assertEqual(r.usage["input_tokens"], 100)

    def test_parse_response_plain_text(self):
        p = self._provider()
        r = p._parse_response(_resp(content="J'ai ajouté ton cours.", reasoning="..."))
        self.assertEqual(r.text, "J'ai ajouté ton cours.")
        self.assertFalse(r.has_function_calls)
        self.assertEqual(r.raw_content["content"], "J'ai ajouté ton cours.")

    def test_reasoning_content_is_echoed_when_replayed(self):
        """The parsed tool-call raw_content, fed back into the next request, must
        still carry reasoning_content (else DeepSeek 400s)."""
        p = self._provider()
        first = p._parse_response(_resp(
            content="", reasoning="chain-of-thought",
            tool_calls=[_tool_call("c1", "list_blocks", {})], finish="tool_calls"))
        replay = p._to_openai_messages(
            [{"role": "assistant", "content": first.raw_content}], None)
        self.assertEqual(replay[0]["reasoning_content"], "chain-of-thought")
        self.assertEqual(replay[0]["tool_calls"][0]["function"]["name"], "list_blocks")

    def test_generate_with_history_enables_thinking_and_tools(self):
        p = self._provider()
        p._client.chat.completions.create.return_value = _resp(content="ok")
        tools = [{"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}]
        r = p.generate_with_history([{"role": "user", "content": "hi"}], tools=tools, system_prompt="S")
        self.assertEqual(r.text, "ok")
        kwargs = p._client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "deepseek-v4-pro")
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertEqual(kwargs["reasoning_effort"], "high")
        self.assertEqual(kwargs["tools"][0]["function"]["name"], "t")

    @override_settings(DEEPSEEK_THINKING=False)
    def test_thinking_disabled_omits_reasoning_params(self):
        p = self._provider()
        p._client.chat.completions.create.return_value = _resp(content="ok")
        p.generate_with_history([{"role": "user", "content": "hi"}])
        kwargs = p._client.chat.completions.create.call_args.kwargs
        self.assertNotIn("extra_body", kwargs)
        self.assertNotIn("reasoning_effort", kwargs)
