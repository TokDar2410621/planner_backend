"""
Utility functions for Planner AI backend.
"""
import logging
import random
import time as time_module
from datetime import datetime, time, timedelta
from functools import wraps
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Default set of HTTP status codes worth retrying. 529 = Anthropic "Overloaded"
# (must be retried); 429 = rate limit; 5xx = transient server errors.
DEFAULT_RETRYABLE_STATUS_CODES: tuple = (408, 409, 429, 500, 502, 503, 504, 529)

# Exception TYPE names (not string contents) that represent transient failures.
# Matched by class name to avoid importing optional SDKs (anthropic / google-genai)
# at module import time. Covers both provider SDKs plus stdlib transport errors.
_RETRYABLE_EXCEPTION_NAMES = frozenset({
    # anthropic
    'RateLimitError', 'InternalServerError', 'APITimeoutError',
    'APIConnectionError', 'OverloadedError',
    # google-genai
    'ServerError', 'ServiceUnavailable', 'ResourceExhausted', 'DeadlineExceeded',
    # generic httpx / requests transport
    'ConnectTimeout', 'ReadTimeout', 'ConnectionError',
})


def _build_default_retryable_exceptions() -> tuple:
    """
    Build the default tuple of catchable exception types.

    Deliberately NOT `(Exception,)`: a bare Exception would swallow genuine
    programming errors (ValueError, KeyError, ...) and funnel them through the
    retry path. We restrict to transport-level errors plus, when available, the
    provider SDK base error classes.
    """
    excs: list = [TimeoutError, ConnectionError, OSError]
    try:
        import anthropic
        excs.append(anthropic.APIError)
    except Exception:  # pragma: no cover - SDK optional
        pass
    try:
        from google.genai import errors as _genai_errors
        excs.append(_genai_errors.APIError)
    except Exception:  # pragma: no cover - SDK optional
        pass
    return tuple(excs)


def _extract_status_code(exc: BaseException) -> Optional[int]:
    """Pull an HTTP status code off an exception, checking common attributes."""
    for attr in ('status_code', 'code', 'http_status', 'status'):
        val = getattr(exc, attr, None)
        if isinstance(val, bool):  # bool is an int subclass; ignore
            continue
        if isinstance(val, int):
            return val
    response = getattr(exc, 'response', None)
    if response is not None:
        val = getattr(response, 'status_code', None)
        if isinstance(val, int) and not isinstance(val, bool):
            return val
    return None


def is_retryable_error(
    exc: BaseException,
    retryable_status_codes: tuple = DEFAULT_RETRYABLE_STATUS_CODES,
) -> bool:
    """
    Classify whether an exception should be retried, by TYPE and status_code.

    This intentionally does NOT inspect ``str(exc)`` for substrings like '500'
    or 'rate limit' (the old behaviour), which retried a ValueError('req_500')
    and missed a typed 429/529 whose message lacked the magic token.

    Args:
        exc: The raised exception instance.
        retryable_status_codes: Status codes considered transient.

    Returns:
        True if the error is a transient/retryable failure.
    """
    # Transport-level failures are always transient.
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    # Typed provider/transport errors, matched by class name.
    if type(exc).__name__ in _RETRYABLE_EXCEPTION_NAMES:
        return True

    # Fall back to the HTTP status code carried by the exception, if any.
    code = _extract_status_code(exc)
    if code is not None and code in retryable_status_codes:
        return True

    return False


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[tuple] = None,
    retryable_status_codes: tuple = DEFAULT_RETRYABLE_STATUS_CODES,
) -> Callable:
    """
    Decorator for retry with exponential backoff.

    Retryability is decided by :func:`is_retryable_error` (exception type /
    status code), never by substring matching on the error message.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Add random jitter to prevent thundering herd
        retryable_exceptions: Exception types to *catch*. Defaults to a curated
            tuple of transport/provider errors (NOT bare Exception), so genuine
            bugs surface immediately instead of being retried.
        retryable_status_codes: HTTP status codes to retry on (includes 529).

    Usage:
        @retry_with_backoff(max_retries=3)
        def call_api():
            return client.generate_content(...)
    """
    if retryable_exceptions is None:
        retryable_exceptions = _build_default_retryable_exceptions()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if not is_retryable_error(e, retryable_status_codes) or attempt == max_retries:
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)

                    # Add jitter (0-25% random addition)
                    if jitter:
                        delay += random.uniform(0, delay * 0.25)

                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {delay:.1f}s (error: {type(e).__name__}: {e})"
                    )
                    time_module.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


def time_to_minutes(t: time) -> int:
    """Convert time to minutes since midnight."""
    return t.hour * 60 + t.minute


def minutes_to_time(minutes: int) -> time:
    """Convert minutes since midnight to time."""
    hours = minutes // 60
    mins = minutes % 60
    return time(hour=hours % 24, minute=mins)


def time_range_overlap(
    start1: time,
    end1: time,
    start2: time,
    end2: time
) -> bool:
    """Check if two time ranges overlap."""
    return not (end1 <= start2 or end2 <= start1)


def format_duration(minutes: int) -> str:
    """Format duration in minutes to human readable string."""
    if minutes < 60:
        return f"{minutes} min"
    hours = minutes // 60
    mins = minutes % 60
    if mins == 0:
        return f"{hours}h"
    return f"{hours}h{mins:02d}"


def parse_time(time_str: str) -> Optional[time]:
    """Parse time string to time object."""
    formats = ['%H:%M', '%H:%M:%S', '%I:%M %p', '%I:%M%p']
    for fmt in formats:
        try:
            return datetime.strptime(time_str.strip(), fmt).time()
        except ValueError:
            continue
    return None


def get_week_dates(start_date: datetime.date, num_weeks: int = 1) -> list:
    """Get list of dates for given number of weeks starting from start_date."""
    dates = []
    current = start_date
    for _ in range(num_weeks * 7):
        dates.append(current)
        current += timedelta(days=1)
    return dates


def clamp(value: int, min_val: int, max_val: int) -> int:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def run_in_background(func: Callable, *args, **kwargs) -> None:
    """
    Run a function in a background greenlet (gevent) or thread (fallback).

    Args:
        func: Function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Usage:
        run_in_background(process_document, document_id=123)
    """
    # Log by function name only — never the args, which may carry user content
    # or credentials.
    logger.debug("run_in_background scheduling %s", func.__name__)

    on_error = kwargs.pop('on_error', None)

    def wrapper():
        logger.debug("Background task %s starting", func.__name__)
        # A daemon thread runs outside Django's request/response cycle, so it
        # gets no automatic connection cleanup. Close any stale connections up
        # front (the thread may inherit a connection opened in the request) and,
        # critically, again in `finally` so we never leak a DB connection when
        # the task raises or the process is torn down mid-run (B17).
        from django.db import close_old_connections
        close_old_connections()
        try:
            func(*args, **kwargs)
            logger.debug("Background task %s completed successfully", func.__name__)
        except Exception as e:
            logger.exception(f"Background task {func.__name__} failed: {e}")
            # Record the failure via the caller-supplied hook so a killed/failed
            # task can be marked (e.g. document.processing_error) instead of
            # silently staying stuck in a pending state.
            if on_error is not None:
                try:
                    on_error(e)
                except Exception:
                    logger.exception(
                        f"on_error hook for background task {func.__name__} itself failed"
                    )
        finally:
            close_old_connections()

    # Always use threading for background tasks
    # Threading works reliably with both sync and gevent gunicorn workers
    # Gevent greenlets only work properly with gevent workers, which Railway doesn't use
    import threading
    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
    logger.info(f"Started background thread: {func.__name__}")


def format_llm_usage(response, provider_name: Optional[str] = None) -> Optional[str]:
    """
    Build a compact one-line summary of an LLM response's token usage.

    Reads ``response.usage`` ({'input_tokens', 'output_tokens'}) and
    ``response.stop_reason`` defensively so it works with an
    :class:`~services.llm.base.LLMResponse`, a bare object, or anything that
    merely quacks like one. Returns ``None`` when no usage is present so the
    caller can skip logging entirely.

    Never logs message bodies or secrets — only token counts and the stop
    reason, which are safe at INFO for cost/usage observability.
    """
    usage = getattr(response, 'usage', None)
    if not usage:
        return None

    if isinstance(usage, dict):
        input_tokens = usage.get('input_tokens')
        output_tokens = usage.get('output_tokens')
    else:
        input_tokens = getattr(usage, 'input_tokens', None)
        output_tokens = getattr(usage, 'output_tokens', None)

    stop_reason = getattr(response, 'stop_reason', None)
    prefix = f"[{provider_name}] " if provider_name else ""
    return (
        f"{prefix}LLM usage: input_tokens={input_tokens} "
        f"output_tokens={output_tokens} stop_reason={stop_reason}"
    )


def log_llm_usage(
    response,
    provider_name: Optional[str] = None,
    log: Optional[logging.Logger] = None,
) -> None:
    """
    Log LLM token usage at INFO when the response carries it.

    Safe to call unconditionally on any response (with or without ``.usage``):
    it silently does nothing when usage is absent and never raises, so
    observability logging can never break a request.
    """
    target = log or logger
    try:
        line = format_llm_usage(response, provider_name)
        if line:
            target.info(line)
    except Exception:  # pragma: no cover - observability must never break a request
        pass
