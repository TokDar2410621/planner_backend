"""
DeepSeek LLM provider.

DeepSeek's API is OpenAI-compatible (base_url https://api.deepseek.com), so we
drive it with the openai SDK. The default model (deepseek-v4-pro) supports
THINKING mode together with tool calling: the model reasons across multiple tool
turns before its final answer. Two hard requirements from DeepSeek's docs shape
this provider:

  1. In thinking mode with tools, the `reasoning_content` returned alongside the
     assistant message MUST be echoed back on every subsequent request or the API
     returns HTTP 400. So `raw_content` here IS a ready-to-replay OpenAI assistant
     message carrying `reasoning_content` + `tool_calls`, which the agent appends
     to history verbatim and we replay unchanged.
  2. Tool results come back to us in the agent's canonical Anthropic shape
     ({"type":"tool_result","tool_use_id",...}); we translate them to OpenAI
     `role:"tool"` messages keyed by tool_call_id.
"""
import json
import logging
import time
from typing import Optional

from django.conf import settings

from .base import LLMProvider, LLMResponse, FunctionCall

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when the dep is absent
    _OPENAI_AVAILABLE = False


class DeepSeekProvider(LLMProvider):
    """DeepSeek provider (thinking + tool calling) via the OpenAI-compatible API."""

    BASE_URL = "https://api.deepseek.com"
    # L'agent route vers stream_with_history quand le client veut du SSE.
    supports_streaming = True
    # Pendant le reasoning (et le streaming d'arguments d'outils), AUCUN texte ne
    # part au client: sans battement, un flux SSE muet plusieurs minutes se fait
    # couper par les idle-timeouts (edge Railway, proxys mobiles) et le client
    # croit à une panne. On ré-émet un événement "thinking" au plus toutes les
    # KEEPALIVE_SECONDS tant que des chunks silencieux arrivent.
    KEEPALIVE_SECONDS = 8.0

    def __init__(self):
        self.api_key = getattr(settings, "DEEPSEEK_API_KEY", "") or ""
        self.model = getattr(settings, "DEEPSEEK_MODEL", "deepseek-v4-pro")
        self.reasoning_effort = getattr(settings, "DEEPSEEK_REASONING_EFFORT", "high")
        self.thinking = getattr(settings, "DEEPSEEK_THINKING", True)
        self.max_tokens = int(getattr(settings, "DEEPSEEK_MAX_TOKENS", 8192))
        self._client = None
        if _OPENAI_AVAILABLE and self.api_key:
            self._client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL)
        logger.info("DeepSeekProvider initialized with model: %s (thinking=%s)",
                    self.model, self.thinking)

    @property
    def name(self) -> str:
        return "deepseek"

    def is_available(self) -> bool:
        return bool(_OPENAI_AVAILABLE and self.api_key and self._client is not None)

    # ---- tool + message translation -------------------------------------

    @staticmethod
    def _convert_tools(tools: Optional[list]) -> Optional[list]:
        """Anthropic tool shape ({name, description, input_schema}) -> OpenAI."""
        if not tools:
            return None
        out = []
        for t in tools:
            out.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        return out

    def _to_openai_messages(self, messages: list, system_prompt: Optional[str]) -> list:
        out = []
        if system_prompt:
            out.append({"role": "system", "content": system_prompt})
        for msg in messages:
            role = msg.get("role", "user")
            if role == "model":  # Gemini compatibility
                role = "assistant"
            content = msg.get("content")

            if isinstance(content, str):
                out.append({"role": role, "content": content})
            elif isinstance(content, dict):
                # Our own raw_content: an already OpenAI-shaped assistant message
                # (carries reasoning_content + tool_calls for the mandatory echo).
                out.append(content)
            elif isinstance(content, list):
                if content and isinstance(content[0], dict) and content[0].get("type") == "tool_result":
                    # Agent's canonical tool-results -> OpenAI tool messages.
                    for block in content:
                        out.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id") or block.get("id") or "",
                            "content": block.get("content", ""),
                        })
                else:
                    # Anthropic-shaped assistant blocks (only reachable via a
                    # cross-provider failover mid tool-loop). Best-effort, no
                    # reasoning_content to replay.
                    texts, tool_calls = [], []
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "text":
                            texts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                                },
                            })
                    m = {"role": "assistant", "content": ("\n".join(texts) or None)}
                    if tool_calls:
                        m["tool_calls"] = tool_calls
                    out.append(m)
        return out

    # ---- API call -------------------------------------------------------

    def _request_kwargs(
        self,
        openai_messages: list,
        openai_tools: Optional[list],
        stream: bool = False,
    ) -> dict:
        kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": self.max_tokens,
            # Parsing d'horaires: temperature basse pour la fidelite jours/heures.
            "temperature": 0.3,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools
        if self.thinking:
            # DeepSeek-specific toggle + effort. Reasoning is on by default but we
            # set it explicitly so behaviour is stable across model aliases.
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            # Disable reasoning EXPLICITLY. deepseek-v4-pro reasons by DEFAULT when
            # the field is omitted, so callers that toggle thinking off (e.g. the
            # quick-replies suggestion call, which needs no reasoning) were still
            # paying full reasoning latency. Measured: omit -> 6.7s (reasoning on)
            # vs explicit disabled -> 2.6s (reasoning off).
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if stream:
            kwargs["stream"] = True
        return kwargs

    def _create(self, openai_messages: list, openai_tools: Optional[list]) -> LLMResponse:
        resp = self._client.chat.completions.create(
            **self._request_kwargs(openai_messages, openai_tools)
        )
        return self._parse_response(resp)

    @staticmethod
    def _build_raw_content(text: str, reasoning: Optional[str], tool_calls_raw: list) -> dict:
        """Assistant turn to replay verbatim next request. It MUST carry
        reasoning_content when tool_calls are present (thinking mode), else
        DeepSeek rejects the follow-up with HTTP 400."""
        raw_content = {"role": "assistant", "content": text or ""}
        if reasoning:
            raw_content["reasoning_content"] = reasoning
        if tool_calls_raw:
            raw_content["tool_calls"] = tool_calls_raw
            if not (text or "").strip():
                raw_content["content"] = None
        return raw_content

    def _parse_response(self, resp) -> LLMResponse:
        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None)

        function_calls, tool_calls_raw = [], []
        for tc in (getattr(msg, "tool_calls", None) or []):
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                args = {}
            function_calls.append(FunctionCall(name=tc.function.name, args=args, call_id=tc.id))
            tool_calls_raw.append({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": raw_args},
            })

        raw_content = self._build_raw_content(text, reasoning, tool_calls_raw)

        usage = None
        if getattr(resp, "usage", None):
            usage = {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            }

        return LLMResponse(
            text=text,
            function_calls=function_calls,
            raw_response=resp,
            raw_content=raw_content,
            stop_reason=choice.finish_reason,
            usage=usage,
        )

    # ---- public interface ----------------------------------------------

    def generate_with_history(
        self,
        messages: list[dict],
        tools: Optional[list] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(
                text="Service IA non disponible.",
                is_error=True,
                error="provider_unavailable",
            )
        try:
            openai_messages = self._to_openai_messages(messages, system_prompt)
            openai_tools = self._convert_tools(tools)
            return self._create(openai_messages, openai_tools)
        except Exception as e:  # noqa: BLE001 - surface as an error response, don't crash the loop
            logger.error(f"DeepSeek API error with history: {e}", exc_info=True)
            return LLMResponse(
                text="Erreur lors de la communication avec l'IA.",
                is_error=True,
                error=str(e),
            )

    def stream_with_history(
        self,
        messages: list[dict],
        tools: Optional[list] = None,
        system_prompt: Optional[str] = None,
    ):
        """Streaming variant of generate_with_history.

        Yields {"type": "thinking"} once when reasoning starts, then
        {"type": "text_delta", "text": ...} per content chunk, and ALWAYS a
        terminal {"type": "final", "response": LLMResponse} whose shape matches
        generate_with_history exactly (including the reasoning_content echo in
        raw_content) so the agent's tool loop replays history identically.
        Tool-call fragments are accumulated by index (OpenAI convention: name
        and id arrive early, arguments arrive in pieces to concatenate).
        """
        if not self.is_available():
            yield {"type": "final", "response": LLMResponse(
                text="Service IA non disponible.",
                is_error=True, error="provider_unavailable")}
            return
        try:
            openai_messages = self._to_openai_messages(messages, system_prompt)
            openai_tools = self._convert_tools(tools)
            stream = self._client.chat.completions.create(
                **self._request_kwargs(openai_messages, openai_tools, stream=True)
            )

            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            acc: dict[int, dict] = {}
            finish_reason = None
            # 0.0 => le PREMIER chunk silencieux (reasoning/outil) émet un
            # battement immédiat (feedback UI instantané), puis throttle.
            last_yield = 0.0

            for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                reasoning_piece = getattr(delta, "reasoning_content", None)
                if reasoning_piece:
                    reasoning_parts.append(reasoning_piece)
                    # La chaîne de raisonnement elle-même n'est JAMAIS envoyée
                    # au client (tokens + privacy) — seulement un battement
                    # throttlé qui garde le socket vivant et l'UI informée.
                    if time.monotonic() - last_yield >= self.KEEPALIVE_SECONDS:
                        last_yield = time.monotonic()
                        yield {"type": "thinking"}

                content_piece = getattr(delta, "content", None)
                if content_piece:
                    text_parts.append(content_piece)
                    last_yield = time.monotonic()
                    yield {"type": "text_delta", "text": content_piece}

                # Les fragments d'arguments d'outils sont eux aussi silencieux:
                # même battement pour ne pas laisser mourir le flux.
                if not content_piece and getattr(delta, "tool_calls", None):
                    if time.monotonic() - last_yield >= self.KEEPALIVE_SECONDS:
                        last_yield = time.monotonic()
                        yield {"type": "thinking"}

                for tc in (getattr(delta, "tool_calls", None) or []):
                    idx = tc.index if getattr(tc, "index", None) is not None else 0
                    slot = acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if getattr(tc, "id", None):
                        slot["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            slot["name"] += fn.name
                        if getattr(fn, "arguments", None):
                            slot["arguments"] += fn.arguments

                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason

            text = "".join(text_parts)
            reasoning = "".join(reasoning_parts) or None
            function_calls, tool_calls_raw = [], []
            for idx in sorted(acc):
                slot = acc[idx]
                raw_args = slot["arguments"] or "{}"
                try:
                    args = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                function_calls.append(FunctionCall(
                    name=slot["name"], args=args, call_id=slot["id"]))
                tool_calls_raw.append({
                    "id": slot["id"], "type": "function",
                    "function": {"name": slot["name"], "arguments": raw_args},
                })

            yield {"type": "final", "response": LLMResponse(
                text=text,
                function_calls=function_calls,
                raw_response=None,
                raw_content=self._build_raw_content(text, reasoning, tool_calls_raw),
                stop_reason=finish_reason,
            )}
        except Exception as e:  # noqa: BLE001 - surface as an error final, never crash the SSE
            logger.error(f"DeepSeek streaming error: {e}", exc_info=True)
            yield {"type": "final", "response": LLMResponse(
                text="Erreur lors de la communication avec l'IA.",
                is_error=True, error=str(e))}

    def generate(
        self,
        prompt: str,
        tools: Optional[list] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        return self.generate_with_history(
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            system_prompt=system_prompt,
        )
