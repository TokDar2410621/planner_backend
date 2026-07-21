"""
Regression tests for GROUP A2 (LLM layer + infra helpers).

Covers:
  B1  - ClaudeProvider.DEFAULT_MODEL is a current model id (not the retired one).
  B3  - LLMResponse carries an is_error flag; provider errors set it.
  B18 - retry classifier decides by exception type / status_code, not substring.
  D4  - LLMResponse exposes stop_reason + usage and detects truncation.
  factory - get_provider() resolves name -> profile -> settings -> 'gemini'.

All tests run offline: no network, no real API keys.
"""
import pytest

from services.llm import (
    get_provider,
    GeminiProvider,
    ClaudeProvider,
    LLMResponse,
)
from utils.helpers import (
    is_retryable_error,
    DEFAULT_RETRYABLE_STATUS_CODES,
)


# --------------------------------------------------------------------------- #
# B1 - default Claude model must not be the retired id
# --------------------------------------------------------------------------- #
def test_claude_default_model_not_retired():
    assert ClaudeProvider.DEFAULT_MODEL != 'claude-sonnet-4-20250514'
    assert ClaudeProvider.DEFAULT_MODEL == 'claude-sonnet-5'
    # No invented date suffix (e.g. -20260101).
    assert '-2025' not in ClaudeProvider.DEFAULT_MODEL
    assert '-2026' not in ClaudeProvider.DEFAULT_MODEL


# --------------------------------------------------------------------------- #
# Provider factory
# --------------------------------------------------------------------------- #
def test_get_provider_explicit_names():
    assert isinstance(get_provider('gemini'), GeminiProvider)
    assert isinstance(get_provider('claude'), ClaudeProvider)


def test_get_provider_case_insensitive():
    assert isinstance(get_provider('CLAUDE'), ClaudeProvider)
    assert isinstance(get_provider(' Gemini '), GeminiProvider)


def test_get_provider_default_is_gemini():
    # No name, no user, settings has no LLM_PROVIDER by default -> gemini.
    assert isinstance(get_provider(), GeminiProvider)


def test_get_provider_unknown_falls_back_to_gemini():
    assert isinstance(get_provider('openai-gpt-99'), GeminiProvider)


@pytest.mark.django_db
def test_get_provider_uses_user_preference():
    from django.contrib.auth.models import User

    u = User.objects.create_user(username='pref', password='x')
    u.profile.preferred_llm = 'claude'
    u.profile.save()

    assert isinstance(get_provider(user=u), ClaudeProvider)


@pytest.mark.django_db
def test_get_provider_explicit_name_beats_user_preference():
    from django.contrib.auth.models import User

    u = User.objects.create_user(username='pref2', password='x')
    u.profile.preferred_llm = 'claude'
    u.profile.save()

    # Explicit name wins over the stored preference.
    assert isinstance(get_provider('gemini', user=u), GeminiProvider)


def test_get_provider_bad_user_object_does_not_raise():
    # An object without .profile must not blow up the factory.
    class Dummy:
        pass

    assert isinstance(get_provider(user=Dummy()), GeminiProvider)


# --------------------------------------------------------------------------- #
# B18 - retry classification by TYPE / status_code, never by substring
# --------------------------------------------------------------------------- #
class _StatusError(Exception):
    """Fake typed API error exposing a status_code attribute."""
    def __init__(self, status_code, message=""):
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(Exception):
    """Name-matched typed error (mirrors anthropic.RateLimitError)."""


def test_typed_529_is_retryable():
    assert is_retryable_error(_StatusError(529, "overloaded")) is True


def test_typed_429_is_retryable():
    assert is_retryable_error(_StatusError(429, "slow down")) is True


def test_529_in_default_status_codes():
    assert 529 in DEFAULT_RETRYABLE_STATUS_CODES


def test_typed_error_by_class_name_is_retryable():
    # No status_code attribute at all: classified by class name.
    assert is_retryable_error(RateLimitError("boom")) is True


def test_plain_value_error_with_500_substring_is_not_retryable():
    # The old substring classifier retried this; the type-based one must not.
    assert is_retryable_error(ValueError("req_500_something_went_wrong")) is False


def test_400_status_is_not_retryable():
    assert is_retryable_error(_StatusError(400, "bad request")) is False


def test_404_status_is_not_retryable():
    # A retired-model 404 must NOT be retried 3x.
    assert is_retryable_error(_StatusError(404, "model not_found")) is False


def test_timeout_error_is_retryable():
    assert is_retryable_error(TimeoutError("timed out")) is True


def test_connection_error_is_retryable():
    assert is_retryable_error(ConnectionError("reset")) is True


def test_retry_default_exceptions_is_not_bare_exception():
    # The retry decorator must NOT default to catching bare Exception.
    from utils.helpers import _build_default_retryable_exceptions
    excs = _build_default_retryable_exceptions()
    assert Exception not in excs
    assert excs  # non-empty


# --------------------------------------------------------------------------- #
# B3 - LLMResponse error flag
# --------------------------------------------------------------------------- #
def test_llmresponse_default_is_not_error():
    r = LLMResponse(text="hello")
    assert r.is_error is False
    assert r.error is None


def test_llmresponse_error_flag_set():
    r = LLMResponse(text="Erreur...", is_error=True, error="404 not_found")
    assert r.is_error is True
    assert r.error == "404 not_found"


def test_claude_unavailable_returns_error_flagged_response():
    # No API key configured in tests -> provider unavailable -> is_error True,
    # not a normal-looking assistant reply.
    p = ClaudeProvider()
    if p.is_available():
        pytest.skip("Claude configured in this env; skipping unavailable path")
    resp = p.generate("hi")
    assert resp.is_error is True
    assert resp.text  # still a human string


# --------------------------------------------------------------------------- #
# D4 - stop_reason / usage / truncation
# --------------------------------------------------------------------------- #
def test_llmresponse_truncation_detection():
    assert LLMResponse(stop_reason="max_tokens").is_truncated is True
    assert LLMResponse(stop_reason="MAX_TOKENS").is_truncated is True
    assert LLMResponse(stop_reason="end_turn").is_truncated is False
    assert LLMResponse(stop_reason=None).is_truncated is False


def test_llmresponse_usage_field():
    r = LLMResponse(usage={'input_tokens': 10, 'output_tokens': 5})
    assert r.usage['input_tokens'] == 10
    assert r.usage['output_tokens'] == 5


# --------------------------------------------------------------------------- #
# B19 - Gemini tool round-trip (offline; exercises the block converter)
# --------------------------------------------------------------------------- #
def _gemini_available():
    try:
        from google.genai import types  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gemini_available(), reason="google-genai not installed")
def test_gemini_roundtrips_function_call_block():
    from google.genai import types

    p = GeminiProvider()  # no client needed; converter is pure
    msg = {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "name": "create_task", "input": {"title": "Buy milk"}},
        ],
    }
    content = p._message_to_content(msg)
    assert content.role == "model"
    assert len(content.parts) == 1
    fc = content.parts[0].function_call
    assert fc is not None
    assert fc.name == "create_task"
    assert dict(fc.args) == {"title": "Buy milk"}


@pytest.mark.skipif(not _gemini_available(), reason="google-genai not installed")
def test_gemini_roundtrips_function_response_block():
    p = GeminiProvider()
    msg = {
        "role": "user",
        "content": [
            {"type": "tool_result", "name": "create_task", "content": {"ok": True}},
        ],
    }
    content = p._message_to_content(msg)
    assert content.role == "user"
    fr = content.parts[0].function_response
    assert fr is not None
    assert fr.name == "create_task"
    assert dict(fr.response) == {"ok": True}


@pytest.mark.skipif(not _gemini_available(), reason="google-genai not installed")
def test_gemini_plain_text_message_still_works():
    p = GeminiProvider()
    content = p._message_to_content({"role": "user", "content": "hello"})
    assert content.role == "user"
    assert content.parts[0].text == "hello"


# --------------------------------------------------------------------------- #
# B17 - run_in_background closes DB connections + records errors
# --------------------------------------------------------------------------- #
@pytest.mark.django_db
def test_run_in_background_invokes_on_error():
    import threading
    from utils.helpers import run_in_background

    done = threading.Event()
    captured = {}

    def failing():
        raise ValueError("boom")

    def on_error(exc):
        captured['exc'] = exc
        done.set()

    run_in_background(failing, on_error=on_error)
    assert done.wait(timeout=5), "background task never reported error"
    assert isinstance(captured['exc'], ValueError)


def test_run_in_background_runs_work():
    import threading
    from utils.helpers import run_in_background

    done = threading.Event()

    def work():
        done.set()

    run_in_background(work)
    assert done.wait(timeout=5)
