"""
Builds rich context for the Planner AI agent's system prompt.
"""
from datetime import timedelta

from django.contrib.auth.models import User
from django.utils import timezone

from core.models import RecurringBlock, Task, Goal

DAY_NAMES = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']


def build_context(user: User) -> dict:
    """
    Build a rich context dict for the system prompt.

    Returns a dict with keys: profile, today_schedule, pending_tasks, active_goals, stats.
    """
    profile = user.profile
    today = timezone.localdate()
    day_of_week = today.weekday()

    # Profile summary
    profile_data = {
        "name": user.first_name or user.username or user.email.split("@")[0],
        "min_sleep_hours": profile.min_sleep_hours,
        "peak_productivity_time": profile.peak_productivity_time,
        "transport_time_minutes": profile.transport_time_minutes,
        "max_deep_work_hours": profile.max_deep_work_hours_per_day,
        "onboarding_completed": profile.onboarding_completed,
    }

    # Today's blocks
    today_blocks = RecurringBlock.objects.filter(
        user=user, day_of_week=day_of_week, active=True
    ).order_by("start_time")

    today_schedule = []
    for b in today_blocks:
        today_schedule.append(
            f"  {b.start_time.strftime('%H:%M')}-{b.end_time.strftime('%H:%M')} {b.title} ({b.get_block_type_display()})"
        )

    # Total blocks count
    total_blocks = RecurringBlock.objects.filter(user=user, active=True).count()

    # Pending tasks (top 5 by priority)
    pending_tasks = Task.objects.filter(user=user, completed=False)[:5]
    tasks_list = []
    for t in pending_tasks:
        deadline_str = f" (deadline: {t.deadline.strftime('%d/%m')})" if t.deadline else ""
        tasks_list.append(f"  - [{t.priority}/10] {t.title}{deadline_str}")

    pending_count = Task.objects.filter(user=user, completed=False).count()

    # Active goals
    active_goals = Goal.objects.filter(user=user, status="active")[:5]
    goals_list = []
    for g in active_goals:
        goals_list.append(f"  - {g.title} ({g.get_goal_type_display()}, {g.progress}%)")

    return {
        "profile": profile_data,
        "today": {
            "date": today.isoformat(),
            "day_name": DAY_NAMES[day_of_week],
            "blocks": today_schedule,
        },
        "total_blocks": total_blocks,
        "tasks": {
            "list": tasks_list,
            "pending_count": pending_count,
        },
        "goals": goals_list,
    }
