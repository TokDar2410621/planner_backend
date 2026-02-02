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


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
    retryable_status_codes: tuple = (429, 500, 502, 503, 504),
) -> Callable:
    """
    Decorator for retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Add random jitter to prevent thundering herd
        retryable_exceptions: Tuple of exceptions to retry on
        retryable_status_codes: HTTP status codes to retry on (for API errors)

    Usage:
        @retry_with_backoff(max_retries=3)
        def call_api():
            return client.generate_content(...)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    # Check if it's a rate limit or retryable error
                    is_retryable = False
                    error_str = str(e).lower()

                    # Check for rate limit (429)
                    if '429' in error_str or 'rate limit' in error_str or 'quota' in error_str:
                        is_retryable = True
                    # Check for server errors
                    elif any(str(code) in error_str for code in retryable_status_codes):
                        is_retryable = True
                    # Check for timeout
                    elif 'timeout' in error_str or 'timed out' in error_str:
                        is_retryable = True

                    if not is_retryable or attempt == max_retries:
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)

                    # Add jitter (0-25% random addition)
                    if jitter:
                        delay += random.uniform(0, delay * 0.25)

                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {delay:.1f}s (error: {e})"
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
    print(f"[BACKGROUND] run_in_background called for {func.__name__} with args={args}", flush=True)

    def wrapper():
        print(f"[BACKGROUND] wrapper starting for {func.__name__}", flush=True)
        try:
            func(*args, **kwargs)
            print(f"[BACKGROUND] {func.__name__} completed successfully", flush=True)
        except Exception as e:
            print(f"[BACKGROUND] {func.__name__} FAILED with error: {e}", flush=True)
            logger.error(f"Background task {func.__name__} failed: {e}")

    try:
        import gevent
        greenlet = gevent.spawn(wrapper)
        # Yield to allow the greenlet to start executing
        # This is critical for gevent's cooperative scheduling
        gevent.sleep(0)
        print(f"[BACKGROUND] Spawned gevent greenlet for {func.__name__}: {greenlet}", flush=True)
        logger.info(f"Started background greenlet: {func.__name__}")
    except ImportError:
        print(f"[BACKGROUND] gevent not available, using threading", flush=True)
        # Fallback to threading if gevent not available
        import threading
        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()
        print(f"[BACKGROUND] Started thread for {func.__name__}", flush=True)
        logger.info(f"Started background thread: {func.__name__}")
