"""
T4 regression: LLM cost/usage observability helpers.

These are pure unit tests for `format_llm_usage` / `log_llm_usage` — no network,
no DB. They assert the helper formats a response that HAS usage, silently skips
one that does NOT, never raises on odd shapes, and never leaks message bodies.
"""
import logging

from services.llm.base import LLMResponse
from utils.helpers import format_llm_usage, log_llm_usage


class _Bare:
    """A minimal object that merely quacks like a response (no LLMResponse)."""

    def __init__(self, usage=None, stop_reason=None):
        self.usage = usage
        self.stop_reason = stop_reason


def test_format_returns_none_without_usage():
    resp = LLMResponse(text="hello")
    assert format_llm_usage(resp) is None


def test_format_includes_tokens_and_stop_reason():
    resp = LLMResponse(
        text="hi",
        stop_reason="end_turn",
        usage={"input_tokens": 120, "output_tokens": 45},
    )
    line = format_llm_usage(resp, provider_name="Claude (test)")

    assert line is not None
    assert "input_tokens=120" in line
    assert "output_tokens=45" in line
    assert "stop_reason=end_turn" in line
    assert "[Claude (test)]" in line
    # Never leak the message body into usage logs.
    assert "hi" not in line


def test_format_without_provider_name_has_no_prefix():
    resp = LLMResponse(usage={"input_tokens": 1, "output_tokens": 2})
    line = format_llm_usage(resp)
    assert line is not None
    assert not line.startswith("[")


def test_format_handles_object_style_usage():
    """usage exposed as attributes rather than a dict must still work."""

    class _Usage:
        input_tokens = 7
        output_tokens = 3

    resp = _Bare(usage=_Usage(), stop_reason="STOP")
    line = format_llm_usage(resp, provider_name="Gemini")
    assert "input_tokens=7" in line
    assert "output_tokens=3" in line
    assert "stop_reason=STOP" in line


def test_format_tolerates_missing_token_keys():
    resp = LLMResponse(usage={"input_tokens": 10})  # no output_tokens key
    line = format_llm_usage(resp)
    assert "input_tokens=10" in line
    assert "output_tokens=None" in line


def test_log_with_usage_emits_info(caplog):
    resp = LLMResponse(
        stop_reason="max_tokens",
        usage={"input_tokens": 500, "output_tokens": 4096},
    )
    with caplog.at_level(logging.INFO, logger="utils.helpers"):
        log_llm_usage(resp, provider_name="Claude")

    records = [r for r in caplog.records if "LLM usage" in r.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert "input_tokens=500" in records[0].getMessage()


def test_log_without_usage_emits_nothing(caplog):
    resp = LLMResponse(text="no usage here")
    with caplog.at_level(logging.INFO, logger="utils.helpers"):
        log_llm_usage(resp)
    assert not [r for r in caplog.records if "LLM usage" in r.getMessage()]


def test_log_never_raises_on_garbage_response():
    # None, a plain string, and an object whose .usage access explodes must all
    # be swallowed — observability must never break a request.
    class _Explosive:
        @property
        def usage(self):
            raise RuntimeError("boom")

    log_llm_usage(None)
    log_llm_usage("not a response")
    log_llm_usage(_Explosive())
    log_llm_usage(_Bare(usage={"input_tokens": 1, "output_tokens": 1}))


def test_log_accepts_custom_logger():
    custom = logging.getLogger("t4.custom")
    resp = LLMResponse(usage={"input_tokens": 2, "output_tokens": 2})
    # Should route through the provided logger without raising.
    log_llm_usage(resp, provider_name="X", log=custom)
