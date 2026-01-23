"""
Utility functions for Planner AI backend.
"""
from datetime import datetime, time, timedelta
from typing import Optional


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
