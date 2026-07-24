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
from django.db import transaction
from django.utils import timezone

from services.scheduling.exceptions import skipped_block_ids

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
from services.scheduling.overlap import MINUTES_PER_DAY
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
        # Populated by generate_schedule: tasks that could not fit + why.
        self.last_unplaced = []

    @retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=30.0)
    def _call_gemini(self, contents):
        """Call Gemini API with retry logic for rate limits."""
        return self.client.models.generate_content(
            model=self.model_name,
            contents=contents
        )

    @transaction.atomic
    def generate_schedule(
        self,
        user: User,
        tasks: Optional[list] = None,
        start_date: Optional[date] = None,
        num_days: int = 7,
        force: bool = False,
        earliest_start: Optional[dict] = None,
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
            # Use local date (Europe/Paris) so the schedule starts on the
            # user's "today", not the UTC day which rolls over at 01:00/02:00.
            start_date = timezone.localdate()

        # Reference date pinned once per run so every scoring/placement
        # decision within this call uses the same "today".
        reference_date = start_date

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

        # Get available time slots (earliest_start floors out already-elapsed
        # or in-progress time on a partial replan: "rien avant la reprise").
        available_slots = self._get_available_slots(
            user, start_date, num_days, earliest_start=earliest_start
        )

        # Score and sort tasks
        scored_tasks = [
            (task, self._calculate_task_priority(task, reference_date))
            for task in tasks
        ]
        scored_tasks.sort(key=lambda x: x[1], reverse=True)

        # Deep-work minutes already placed per date, so we never exceed the
        # user's max_deep_work_hours_per_day in the deterministic placement.
        deep_minutes_by_date: dict = {}
        max_deep_minutes = (user.profile.max_deep_work_hours_per_day or 0) * 60

        # Honest-conflict report (spec §10): a task that cannot fit is never
        # silently dropped; it lands here with the real remaining capacity so
        # the API/UI can surface options instead of inventing an impossible plan.
        self.last_unplaced = []

        # Schedule tasks
        created_blocks = []
        for task, score in scored_tasks:
            block = self._match_task_to_slot(
                task,
                available_slots,
                user.profile,
                reference_date,
                deep_minutes_by_date,
                max_deep_minutes,
            )
            if block is None:
                needed = task.estimated_duration_minutes or 60
                biggest = max((s.duration_minutes for s in available_slots), default=0)
                # Diagnose the REAL blocker so the explanation is accurate
                # (spec §10): deep-work daily cap vs. no room in the days.
                if (
                    task.task_type == 'deep_work'
                    and max_deep_minutes > 0
                    and needed > max_deep_minutes
                ):
                    reason = (
                        f'dépasse le plafond quotidien de travail intense '
                        f'({max_deep_minutes} min/jour) : fractionner la tâche '
                        f'ou augmenter le plafond'
                    )
                elif biggest < 30:
                    reason = 'aucun créneau disponible'
                else:
                    reason = f'demande {needed} min, plus grand créneau restant {biggest} min'
                self.last_unplaced.append({
                    'task_id': task.id,
                    'title': task.title,
                    'needed_minutes': needed,
                    'largest_free_slot_minutes': biggest,
                    'reason': reason,
                })
            if block:
                block.save()
                created_blocks.append(block)

                # Track deep-work load for the day the block landed on.
                if task.task_type == 'deep_work':
                    placed = self._time_diff_minutes(block.start_time, block.end_time)
                    deep_minutes_by_date[block.date] = (
                        deep_minutes_by_date.get(block.date, 0) + placed
                    )

                # Remove used slot time
                self._update_available_slots(available_slots, block)

        return created_blocks

    def _get_available_slots(
        self,
        user: User,
        start_date: date,
        num_days: int,
        earliest_start: Optional[dict] = None,
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

            # Get blocked times for this day (récurrents + overnight + déjà planifiés)
            blocked_times = self._get_blocked_times(user, current_date)

            skipped_prev = skipped_block_ids(user, current_date - timedelta(days=1))

            # Add sleep/recovery block after a real work night shift. Overnight
            # sleep uses the same storage flag to cross midnight, but it must
            # not create an extra post-shift recovery window.
            yesterday_blocks = RecurringBlock.objects.filter(
                user=user,
                day_of_week=(day_of_week - 1) % 7,
                is_night_shift=True,
                block_type='work',
                active=True,
            ).exclude(id__in=skipped_prev)
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

            # Partial-replan floor: block everything before the resume time on
            # this date so no flexible activity is placed in already-elapsed or
            # in-progress time. earliest_start maps date -> time.
            if earliest_start:
                floor = earliest_start.get(current_date)
                if floor and floor > time(0, 0):
                    blocked_times.append((time(0, 0), floor))

            # Generate available slots
            day_slots = self._find_free_slots(
                current_date,
                blocked_times,
                profile.peak_productivity_time
            )
            slots.extend(day_slots)

        return slots

    @staticmethod
    def _min_to_time(m: int) -> time:
        """Convertit des minutes (0..1440) en time, en bornant à 23:59:59."""
        if m >= MINUTES_PER_DAY:
            return time(23, 59, 59)
        m = max(0, m)
        return time(m // 60, m % 60)

    def _get_blocked_times(self, user: User, current_date: date) -> list:
        """
        Retourne TOUTES les plages occupées (dans la journée) pour une DATE donnée.

        Inclut, correctement clippés à la journée :
        - les blocs récurrents du jour (avec buffer transport), overnight géré ;
        - la portion matinale d'un bloc de nuit de la VEILLE qui déborde ;
        - les ScheduledBlock DÉJÀ planifiés à cette date (évite "tâche sur tâche").

        Returns:
            list: liste de tuples (start_time, end_time) avec end > start.
        """
        day_of_week = current_date.weekday()
        prev_day = (day_of_week - 1) % 7
        profile = user.profile
        skipped_today = skipped_block_ids(user, current_date)
        skipped_prev = skipped_block_ids(user, current_date - timedelta(days=1))

        blocked = []  # (start_min, end_min) bornés à [0, 1440]

        def add(s: int, e: int):
            s = max(0, s)
            e = min(MINUTES_PER_DAY, e)
            if e > s:
                blocked.append((s, e))

        def to_min(t: time) -> int:
            return t.hour * 60 + t.minute

        # Blocs récurrents du jour courant. La fenêtre d'indisponibilité AVANT
        # un bloc rattaché à un lieu suit la formule de la spec (préparation +
        # trajet + marge de sécurité -> "début_indisponibilité"), et le retour
        # bloque `trajet` minutes APRÈS. Sans lieu: buffer transport plat (legacy).
        from services.commute import block_commute_minutes

        for b in RecurringBlock.objects.filter(
            user=user, day_of_week=day_of_week, active=True
        ).exclude(id__in=skipped_today).select_related('place'):
            before, after = block_commute_minutes(b, profile)
            s, e = to_min(b.start_time), to_min(b.end_time)
            if b.is_night_shift or e <= s:
                add(s - before, MINUTES_PER_DAY)  # part du jour J
            else:
                add(s - before, e + after)

        # Portion matinale d'un bloc de nuit commencé la VEILLE (le retour
        # bloque `trajet` minutes après la fin du shift)
        for b in RecurringBlock.objects.filter(
            user=user, day_of_week=prev_day, active=True
        ).exclude(id__in=skipped_prev).select_related('place'):
            _, after = block_commute_minutes(b, profile)
            s, e = to_min(b.start_time), to_min(b.end_time)
            if b.is_night_shift or e <= s:
                add(0, e + after)

        # Blocs DÉJÀ planifiés à cette date -> ne jamais empiler dessus
        from services.commute import task_commute_minutes

        for sb in ScheduledBlock.objects.filter(
            user=user, date=current_date
        ).select_related('task', 'task__place'):
            before, after = task_commute_minutes(sb.task, profile)
            s, e = to_min(sb.start_time), to_min(sb.end_time)
            if e <= s:
                add(s - before, MINUTES_PER_DAY)
            else:
                add(s - before, e + after)

        return [(self._min_to_time(s), self._min_to_time(e)) for s, e in blocked]

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
            # Compare in local time so the deadline bucket matches the user's
            # calendar day, not the UTC date.
            deadline_local_date = timezone.localtime(task.deadline).date()
            days_until = (deadline_local_date - reference_date).days
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
        profile: UserProfile,
        reference_date: Optional[date] = None,
        deep_minutes_by_date: Optional[dict] = None,
        max_deep_minutes: int = 0,
    ) -> Optional[ScheduledBlock]:
        """
        Find the best slot for a task and create a ScheduledBlock.

        Args:
            task: Task to schedule
            slots: Available time slots
            profile: User profile
            reference_date: The run's pinned "today" (local date)
            deep_minutes_by_date: Deep-work minutes already placed per date
            max_deep_minutes: Cap of deep-work minutes allowed per day (0 = off)

        Returns:
            ScheduledBlock or None if no suitable slot found
        """
        if not slots:
            return None

        if reference_date is None:
            reference_date = timezone.localdate()
        if deep_minutes_by_date is None:
            deep_minutes_by_date = {}

        duration = task.estimated_duration_minutes or 60  # Default 1 hour

        # Filter slots that can fit the task
        fitting_slots = [s for s in slots if s.duration_minutes >= duration]
        if not fitting_slots:
            return None

        is_deep_work = task.task_type == 'deep_work'

        # B12: enforce the daily deep-work cap in the deterministic placement.
        # Drop slots whose date has no remaining deep-work budget for this task.
        if is_deep_work and max_deep_minutes > 0:
            capped = [
                s for s in fitting_slots
                if deep_minutes_by_date.get(s.date, 0) + duration <= max_deep_minutes
            ]
            if not capped:
                # No day can host this deep-work task without exceeding the cap.
                return None
            fitting_slots = capped

        # B11: honor the deadline TIME-of-day, not only the date. A slot that
        # starts after the deadline instant places the task late; prefer slots
        # that start before the deadline whenever any exist.
        deadline_local = None
        if task.deadline:
            deadline_local = timezone.localtime(task.deadline)

        def slot_start_dt(slot: TimeSlot) -> datetime:
            # Slot start as an aware datetime in the same tz as the deadline,
            # so the comparison respects the deadline's time-of-day.
            naive = datetime.combine(slot.date, slot.start_time)
            return timezone.make_aware(naive, deadline_local.tzinfo)

        if deadline_local is not None:
            on_time = [s for s in fitting_slots if slot_start_dt(s) < deadline_local]
            if on_time:
                fitting_slots = on_time

        # Score slots
        def slot_score(slot: TimeSlot) -> float:
            score = 0

            # Match task type with energy level
            if is_deep_work:
                if slot.energy_level == 'high':
                    score += 50
                elif slot.energy_level == 'medium':
                    score += 20
            else:  # shallow or errand
                if slot.energy_level == 'low':
                    score += 30
                elif slot.energy_level == 'medium':
                    score += 20

            # Prefer earlier dates (relative to the run's pinned reference date)
            days_from_start = (slot.date - reference_date).days
            score -= days_from_start * 5

            # If task has deadline, prefer slots closer to but before deadline.
            # A slot that starts after the deadline instant is penalized hard.
            if deadline_local is not None:
                if slot_start_dt(slot) >= deadline_local:
                    score -= 1000  # Late placement: last resort only
                days_to_deadline = (deadline_local.date() - slot.date).days
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
