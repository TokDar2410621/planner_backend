"""
AI-powered scheduling service.
"""
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Optional

from django.conf import settings
from django.contrib.auth.models import User
from django.utils import timezone

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from core.models import (
    UserProfile,
    RecurringBlock,
    Task,
    ScheduledBlock,
)
from utils.helpers import retry_with_backoff

logger = logging.getLogger(__name__)


@dataclass
class TimeSlot:
    """Represents an available time slot."""
    date: date
    start_time: time
    end_time: time
    energy_level: str  # 'high', 'medium', 'low'
    duration_minutes: int

    def overlaps(self, start: time, end: time) -> bool:
        """Check if this slot overlaps with given time range."""
        return not (end <= self.start_time or start >= self.end_time)


class AIScheduler:
    """
    Generates optimal schedules using AI-powered analysis.

    Takes into account user preferences, recurring blocks, task priorities,
    and energy levels throughout the day.
    """

    # Default work hours if no blocks defined
    DEFAULT_START_HOUR = 8
    DEFAULT_END_HOUR = 22

    # Energy levels by time of day (default, adjusted by user preference)
    ENERGY_LEVELS = {
        'morning': {
            'high': [(8, 12)],
            'medium': [(12, 14), (16, 18)],
            'low': [(14, 16), (18, 22)],
        },
        'afternoon': {
            'high': [(13, 17)],
            'medium': [(10, 13), (17, 19)],
            'low': [(8, 10), (19, 22)],
        },
        'evening': {
            'high': [(18, 22)],
            'medium': [(14, 18)],
            'low': [(8, 14)],
        },
    }

    def __init__(self):
        """Initialize the scheduler."""
        if GEMINI_AVAILABLE and settings.GEMINI_API_KEY:
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            self.model_name = 'gemini-2.5-flash'
        else:
            self.client = None
            self.model_name = None

    @retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=30.0)
    def _call_gemini(self, contents):
        """Call Gemini API with retry logic for rate limits."""
        return self.client.models.generate_content(
            model=self.model_name,
            contents=contents
        )

    def generate_schedule(
        self,
        user: User,
        tasks: Optional[list] = None,
        start_date: Optional[date] = None,
        num_days: int = 7,
        force: bool = False
    ) -> list:
        """
        Generate an optimal schedule for the user's tasks.

        Args:
            user: The user to generate schedule for
            tasks: List of tasks to schedule (defaults to all incomplete tasks)
            start_date: Start date for scheduling (defaults to today)
            num_days: Number of days to schedule
            force: If True, regenerate even if blocks exist

        Returns:
            list: List of ScheduledBlock instances created
        """
        if start_date is None:
            start_date = timezone.now().date()

        if tasks is None:
            tasks = list(Task.objects.filter(
                user=user,
                completed=False
            ).order_by('deadline', '-priority'))

        if not tasks:
            return []

        # Clear existing scheduled blocks if forcing
        if force:
            ScheduledBlock.objects.filter(
                user=user,
                date__gte=start_date,
                date__lt=start_date + timedelta(days=num_days)
            ).delete()

        # Get available time slots
        available_slots = self._get_available_slots(user, start_date, num_days)

        # Score and sort tasks
        scored_tasks = [
            (task, self._calculate_task_priority(task, start_date))
            for task in tasks
        ]
        scored_tasks.sort(key=lambda x: x[1], reverse=True)

        # Schedule tasks
        created_blocks = []
        for task, score in scored_tasks:
            block = self._match_task_to_slot(task, available_slots, user.profile)
            if block:
                block.save()
                created_blocks.append(block)

                # Remove used slot time
                self._update_available_slots(available_slots, block)

        return created_blocks

    def _get_available_slots(
        self,
        user: User,
        start_date: date,
        num_days: int
    ) -> list:
        """
        Get all available time slots for the user.

        Args:
            user: The user
            start_date: Start date
            num_days: Number of days to check

        Returns:
            list: List of TimeSlot objects
        """
        profile = user.profile
        slots = []

        for day_offset in range(num_days):
            current_date = start_date + timedelta(days=day_offset)
            day_of_week = current_date.weekday()

            # Get blocked times for this day
            blocked_times = self._get_blocked_times(user, day_of_week)

            # Add sleep/recovery block after night shift
            # Check if there was a night shift the day before
            yesterday_blocks = RecurringBlock.objects.filter(
                user=user,
                day_of_week=(day_of_week - 1) % 7,
                is_night_shift=True,
                active=True
            )
            if yesterday_blocks.exists():
                # Find the latest end time of night shifts
                latest_end = time(7, 0)  # Default assumption
                for night_block in yesterday_blocks:
                    if night_block.end_time and night_block.end_time > time(0, 0) and night_block.end_time < time(12, 0):
                        if night_block.end_time > latest_end:
                            latest_end = night_block.end_time

                # Calculate wake time: end_time + min_sleep_hours
                # Use configured wake time or calculate from shift end + sleep needed
                if profile.post_night_shift_wake_time:
                    wake_time = profile.post_night_shift_wake_time
                else:
                    # Default: wake up min_sleep_hours after shift ends
                    sleep_hours = profile.min_sleep_hours or 7
                    wake_hour = (latest_end.hour + sleep_hours) % 24
                    wake_time = time(wake_hour, 0)

                # Block from midnight to wake time
                blocked_times.append((
                    time(0, 0),
                    wake_time
                ))
                logger.info(f"Blocked recovery time after night shift until {wake_time}")

            # Generate available slots
            day_slots = self._find_free_slots(
                current_date,
                blocked_times,
                profile.peak_productivity_time
            )
            slots.extend(day_slots)

        return slots

    def _get_blocked_times(self, user: User, day_of_week: int) -> list:
        """
        Get all blocked time ranges for a specific day.

        Args:
            user: The user
            day_of_week: Day of week (0=Monday)

        Returns:
            list: List of (start_time, end_time) tuples
        """
        blocks = RecurringBlock.objects.filter(
            user=user,
            day_of_week=day_of_week,
            active=True
        )

        blocked = []
        for block in blocks:
            # Add transport buffer
            transport_minutes = user.profile.transport_time_minutes
            if transport_minutes > 0:
                buffer_start = datetime.combine(date.today(), block.start_time) - timedelta(minutes=transport_minutes)
                buffer_end = datetime.combine(date.today(), block.end_time) + timedelta(minutes=transport_minutes)
                blocked.append((buffer_start.time(), buffer_end.time()))
            else:
                blocked.append((block.start_time, block.end_time))

        return blocked

    def _find_free_slots(
        self,
        current_date: date,
        blocked_times: list,
        productivity_preference: str
    ) -> list:
        """
        Find free time slots in a day.

        Args:
            current_date: The date to find slots for
            blocked_times: List of blocked time ranges
            productivity_preference: User's peak productivity time

        Returns:
            list: List of TimeSlot objects
        """
        slots = []

        # Merge and sort blocked times
        blocked_times = sorted(blocked_times, key=lambda x: x[0])
        merged_blocked = []
        for start, end in blocked_times:
            if merged_blocked and start <= merged_blocked[-1][1]:
                merged_blocked[-1] = (merged_blocked[-1][0], max(end, merged_blocked[-1][1]))
            else:
                merged_blocked.append((start, end))

        # Find gaps
        current_time = time(self.DEFAULT_START_HOUR, 0)
        end_of_day = time(self.DEFAULT_END_HOUR, 0)

        for block_start, block_end in merged_blocked:
            if current_time < block_start:
                # There's a gap
                duration = self._time_diff_minutes(current_time, block_start)
                if duration >= 30:  # Minimum 30 minutes slot
                    energy = self._get_energy_level(current_time, productivity_preference)
                    slots.append(TimeSlot(
                        date=current_date,
                        start_time=current_time,
                        end_time=block_start,
                        energy_level=energy,
                        duration_minutes=duration
                    ))
            current_time = max(current_time, block_end)

        # Check for time after last block
        if current_time < end_of_day:
            duration = self._time_diff_minutes(current_time, end_of_day)
            if duration >= 30:
                energy = self._get_energy_level(current_time, productivity_preference)
                slots.append(TimeSlot(
                    date=current_date,
                    start_time=current_time,
                    end_time=end_of_day,
                    energy_level=energy,
                    duration_minutes=duration
                ))

        return slots

    def _get_energy_level(self, slot_time: time, preference: str) -> str:
        """
        Determine energy level for a given time.

        Args:
            slot_time: Time to check
            preference: User's productivity preference

        Returns:
            str: 'high', 'medium', or 'low'
        """
        hour = slot_time.hour
        levels = self.ENERGY_LEVELS.get(preference, self.ENERGY_LEVELS['morning'])

        for level, ranges in levels.items():
            for start_hour, end_hour in ranges:
                if start_hour <= hour < end_hour:
                    return level

        return 'low'

    def _time_diff_minutes(self, start: time, end: time) -> int:
        """Calculate difference between two times in minutes."""
        start_minutes = start.hour * 60 + start.minute
        end_minutes = end.hour * 60 + end.minute
        return end_minutes - start_minutes

    def _calculate_task_priority(self, task: Task, reference_date: date) -> float:
        """
        Calculate a priority score for a task.

        Args:
            task: The task to score
            reference_date: Reference date for deadline calculation

        Returns:
            float: Priority score (higher = more urgent/important)
        """
        score = task.priority * 10  # Base score from priority (10-100)

        # Deadline urgency
        if task.deadline:
            days_until = (task.deadline.date() - reference_date).days
            if days_until <= 0:
                score += 100  # Overdue
            elif days_until <= 1:
                score += 80  # Due today/tomorrow
            elif days_until <= 3:
                score += 50  # Due soon
            elif days_until <= 7:
                score += 20  # Due this week

        # Boost for deep work (should be scheduled in high-energy slots)
        if task.task_type == 'deep_work':
            score += 15

        return score

    def _match_task_to_slot(
        self,
        task: Task,
        slots: list,
        profile: UserProfile
    ) -> Optional[ScheduledBlock]:
        """
        Find the best slot for a task and create a ScheduledBlock.

        Args:
            task: Task to schedule
            slots: Available time slots
            profile: User profile

        Returns:
            ScheduledBlock or None if no suitable slot found
        """
        if not slots:
            return None

        duration = task.estimated_duration_minutes or 60  # Default 1 hour

        # Filter slots that can fit the task
        fitting_slots = [s for s in slots if s.duration_minutes >= duration]
        if not fitting_slots:
            # Try to find partial slot
            fitting_slots = [s for s in slots if s.duration_minutes >= 30]
            if not fitting_slots:
                return None
            duration = min(duration, max(s.duration_minutes for s in fitting_slots))

        # Score slots
        def slot_score(slot: TimeSlot) -> float:
            score = 0

            # Match task type with energy level
            if task.task_type == 'deep_work':
                if slot.energy_level == 'high':
                    score += 50
                elif slot.energy_level == 'medium':
                    score += 20
            else:  # shallow or errand
                if slot.energy_level == 'low':
                    score += 30
                elif slot.energy_level == 'medium':
                    score += 20

            # Prefer earlier dates
            days_from_start = (slot.date - timezone.now().date()).days
            score -= days_from_start * 5

            # If task has deadline, prefer slots closer to but before deadline
            if task.deadline:
                days_to_deadline = (task.deadline.date() - slot.date).days
                if days_to_deadline < 0:
                    score -= 100  # After deadline is bad
                elif days_to_deadline <= 1:
                    score += 30  # Day before deadline
                elif days_to_deadline <= 3:
                    score += 20

            return score

        # Sort slots by score
        fitting_slots.sort(key=slot_score, reverse=True)
        best_slot = fitting_slots[0]

        # Calculate end time
        start_datetime = datetime.combine(best_slot.date, best_slot.start_time)
        end_datetime = start_datetime + timedelta(minutes=duration)
        end_time = end_datetime.time()

        # Don't exceed slot end time
        if end_time > best_slot.end_time:
            end_time = best_slot.end_time

        return ScheduledBlock(
            user=task.user,
            task=task,
            date=best_slot.date,
            start_time=best_slot.start_time,
            end_time=end_time,
        )

    def _update_available_slots(self, slots: list, scheduled_block: ScheduledBlock):
        """
        Update available slots after scheduling a block.

        Args:
            slots: List of TimeSlot objects
            scheduled_block: The newly scheduled block
        """
        to_remove = []
        to_add = []

        for i, slot in enumerate(slots):
            if slot.date != scheduled_block.date:
                continue

            if slot.overlaps(scheduled_block.start_time, scheduled_block.end_time):
                to_remove.append(i)

                # Create new slots for remaining time
                if slot.start_time < scheduled_block.start_time:
                    duration = self._time_diff_minutes(slot.start_time, scheduled_block.start_time)
                    if duration >= 30:
                        to_add.append(TimeSlot(
                            date=slot.date,
                            start_time=slot.start_time,
                            end_time=scheduled_block.start_time,
                            energy_level=slot.energy_level,
                            duration_minutes=duration
                        ))

                if slot.end_time > scheduled_block.end_time:
                    duration = self._time_diff_minutes(scheduled_block.end_time, slot.end_time)
                    if duration >= 30:
                        to_add.append(TimeSlot(
                            date=slot.date,
                            start_time=scheduled_block.end_time,
                            end_time=slot.end_time,
                            energy_level=slot.energy_level,
                            duration_minutes=duration
                        ))

        # Remove used slots (in reverse order to preserve indices)
        for i in reversed(to_remove):
            slots.pop(i)

        # Add new partial slots
        slots.extend(to_add)

    def optimize_with_ai(self, user: User, schedule: list) -> list:
        """
        Use AI to further optimize the schedule.

        Args:
            user: The user
            schedule: Current schedule

        Returns:
            list: Optimized schedule suggestions
        """
        if not self.client or not schedule:
            return schedule

        # Prepare schedule data for AI
        schedule_data = []
        for block in schedule:
            schedule_data.append({
                'task': block.task.title,
                'type': block.task.task_type,
                'date': str(block.date),
                'start': str(block.start_time),
                'end': str(block.end_time),
                'deadline': str(block.task.deadline) if block.task.deadline else None,
            })

        prompt = f"""Analyse ce planning et suggère des améliorations (JSON):

Planning actuel:
{json.dumps(schedule_data, indent=2, ensure_ascii=False)}

Préférences utilisateur:
- Pic de productivité: {user.profile.get_peak_productivity_time_display()}
- Max deep work/jour: {user.profile.max_deep_work_hours_per_day}h

Réponds avec:
{{
    "suggestions": [
        {{"type": "move|swap|warning", "task": "...", "reason": "..."}}
    ],
    "overall_score": 1-10
}}"""

        try:
            response = self._call_gemini(prompt)
            # For now, just log the suggestions
            logger.info(f"AI optimization suggestions: {response.text}")
        except Exception as e:
            logger.error(f"AI optimization error: {e}")

        return schedule