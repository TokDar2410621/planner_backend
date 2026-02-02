"""
AI Insights Service - Intelligent scheduling features.

Features:
- Proactive suggestions ("Tu as 2h libre, veux-tu avancer sur X?")
- Pattern analysis (detect habits and optimize)
- Duration prediction (estimate based on history)
- Natural language scheduling ("Planifie ma révision de maths cette semaine")
- Conflict detection (alert on overlaps)
- Smart rescheduling (auto-reorganize when task overflows)
"""
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Optional, List, Dict, Tuple

from django.conf import settings
from django.contrib.auth.models import User
from django.db.models import Avg, Count, Q
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
    TaskHistory,
)
from utils.helpers import retry_with_backoff

logger = logging.getLogger(__name__)


@dataclass
class TimeGap:
    """Represents a free time gap in the schedule."""
    date: date
    start_time: time
    end_time: time
    duration_minutes: int
    energy_level: str  # 'high', 'medium', 'low'


@dataclass
class Suggestion:
    """A proactive suggestion for the user."""
    type: str  # 'free_time', 'reschedule', 'pattern', 'conflict', 'reminder'
    message: str
    task_id: Optional[int] = None
    action: Optional[str] = None  # 'schedule', 'move', 'split', 'remind'
    metadata: Optional[dict] = None


@dataclass
class Conflict:
    """Represents a scheduling conflict."""
    type: str  # 'overlap', 'overload', 'deadline_risk', 'energy_mismatch'
    severity: str  # 'high', 'medium', 'low'
    message: str
    blocks_involved: List[int]
    suggested_resolution: Optional[str] = None


class AIInsightsService:
    """
    AI-powered insights and suggestions for scheduling.
    """

    # Energy levels by time of day
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
        """Initialize the AI Insights service."""
        if GEMINI_AVAILABLE and settings.GEMINI_API_KEY:
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            self.model_name = 'gemini-2.5-flash'
        else:
            self.client = None
            self.model_name = None

    @retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=30.0)
    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API with retry logic."""
        if not self.client:
            raise RuntimeError("Gemini API not configured")
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text

    # ==================== PROACTIVE SUGGESTIONS ====================

    def get_proactive_suggestions(
        self,
        user: User,
        target_date: Optional[date] = None,
        limit: int = 5
    ) -> List[Suggestion]:
        """
        Generate proactive suggestions for the user.

        Args:
            user: The user to generate suggestions for
            target_date: Date to analyze (defaults to today)
            limit: Maximum number of suggestions

        Returns:
            List of Suggestion objects
        """
        if target_date is None:
            target_date = timezone.now().date()

        suggestions = []

        # 1. Find free time gaps
        gaps = self._find_free_time_gaps(user, target_date)
        pending_tasks = Task.objects.filter(user=user, completed=False).order_by('-priority', 'deadline')

        for gap in gaps[:3]:  # Top 3 gaps
            if gap.duration_minutes >= 30 and pending_tasks.exists():
                # Find matching task for this gap
                matching_task = self._find_best_task_for_gap(gap, pending_tasks, user.profile)
                if matching_task:
                    suggestions.append(Suggestion(
                        type='free_time',
                        message=f"Tu as {gap.duration_minutes} minutes de libre de {gap.start_time.strftime('%H:%M')} a {gap.end_time.strftime('%H:%M')}. Veux-tu avancer sur '{matching_task.title}'?",
                        task_id=matching_task.id,
                        action='schedule',
                        metadata={
                            'gap_start': gap.start_time.isoformat(),
                            'gap_end': gap.end_time.isoformat(),
                            'gap_date': gap.date.isoformat(),
                            'energy_level': gap.energy_level,
                        }
                    ))

        # 2. Check for approaching deadlines
        deadline_tasks = pending_tasks.filter(
            deadline__isnull=False,
            deadline__lte=timezone.now() + timedelta(days=3)
        )
        for task in deadline_tasks[:2]:
            days_left = (task.deadline.date() - target_date).days
            if days_left <= 0:
                suggestions.append(Suggestion(
                    type='reminder',
                    message=f"'{task.title}' est en retard! Deadline: {task.deadline.strftime('%d/%m %H:%M')}",
                    task_id=task.id,
                    action='schedule',
                    metadata={'urgency': 'critical'}
                ))
            elif days_left <= 1:
                suggestions.append(Suggestion(
                    type='reminder',
                    message=f"'{task.title}' est due demain. Planifie-la maintenant!",
                    task_id=task.id,
                    action='schedule',
                    metadata={'urgency': 'high'}
                ))

        # 3. Detect patterns and suggest optimizations
        pattern_suggestions = self._analyze_patterns_for_suggestions(user)
        suggestions.extend(pattern_suggestions[:2])

        return suggestions[:limit]

    def _find_free_time_gaps(
        self,
        user: User,
        target_date: date
    ) -> List[TimeGap]:
        """Find free time gaps in the user's schedule."""
        profile = user.profile
        day_of_week = target_date.weekday()

        # Get all blocks for this day
        recurring_blocks = RecurringBlock.objects.filter(
            user=user,
            day_of_week=day_of_week,
            active=True
        ).values_list('start_time', 'end_time')

        scheduled_blocks = ScheduledBlock.objects.filter(
            user=user,
            date=target_date
        ).values_list('start_time', 'end_time')

        # Merge all blocked times
        blocked_times = list(recurring_blocks) + list(scheduled_blocks)
        blocked_times = sorted(blocked_times, key=lambda x: x[0])

        # Merge overlapping blocks
        merged = []
        for start, end in blocked_times:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))

        # Find gaps
        gaps = []
        current_time = time(8, 0)  # Start of day
        end_of_day = time(22, 0)  # End of day

        for block_start, block_end in merged:
            if current_time < block_start:
                duration = self._time_diff_minutes(current_time, block_start)
                if duration >= 30:
                    energy = self._get_energy_level(current_time, profile.peak_productivity_time)
                    gaps.append(TimeGap(
                        date=target_date,
                        start_time=current_time,
                        end_time=block_start,
                        duration_minutes=duration,
                        energy_level=energy
                    ))
            current_time = max(current_time, block_end)

        # Check for time after last block
        if current_time < end_of_day:
            duration = self._time_diff_minutes(current_time, end_of_day)
            if duration >= 30:
                energy = self._get_energy_level(current_time, profile.peak_productivity_time)
                gaps.append(TimeGap(
                    date=target_date,
                    start_time=current_time,
                    end_time=end_of_day,
                    duration_minutes=duration,
                    energy_level=energy
                ))

        return gaps

    def _find_best_task_for_gap(
        self,
        gap: TimeGap,
        tasks,
        profile: UserProfile
    ) -> Optional[Task]:
        """Find the best task to suggest for a time gap."""
        for task in tasks:
            # Check if task fits in gap
            estimated = task.estimated_duration_minutes or 60
            if estimated > gap.duration_minutes:
                continue

            # Match energy level with task type
            if task.task_type == 'deep_work' and gap.energy_level == 'high':
                return task
            elif task.task_type in ['shallow', 'errand'] and gap.energy_level in ['medium', 'low']:
                return task

        # If no perfect match, return first fitting task
        for task in tasks:
            estimated = task.estimated_duration_minutes or 60
            if estimated <= gap.duration_minutes:
                return task

        return None

    # ==================== PATTERN ANALYSIS ====================

    def analyze_user_patterns(self, user: User) -> Dict:
        """
        Analyze user's scheduling patterns.

        Returns:
            Dict with pattern insights
        """
        history = TaskHistory.objects.filter(user=user)

        if history.count() < 5:
            return {'status': 'insufficient_data', 'message': 'Pas assez de donnees pour analyser les patterns'}

        patterns = {
            'most_productive_day': self._get_most_productive_day(history),
            'best_time_for_deep_work': self._get_best_time_for_task_type(history, 'deep_work'),
            'average_accuracy': self._get_estimation_accuracy(history),
            'completion_rate_by_day': self._get_completion_rate_by_day(history),
            'common_reschedule_reasons': self._get_reschedule_patterns(history),
            'energy_patterns': self._get_energy_patterns(history),
        }

        return patterns

    def _get_most_productive_day(self, history) -> Dict:
        """Find the day with most completed tasks."""
        by_day = history.values('day_of_week').annotate(
            count=Count('id'),
            avg_duration=Avg('actual_duration_minutes')
        ).order_by('-count')

        if by_day:
            day_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
            best = by_day[0]
            return {
                'day': day_names[best['day_of_week']],
                'day_index': best['day_of_week'],
                'tasks_completed': best['count'],
                'avg_duration': round(best['avg_duration'] or 0)
            }
        return {}

    def _get_best_time_for_task_type(self, history, task_type: str) -> Dict:
        """Find the best time for a specific task type."""
        type_history = history.filter(task_type=task_type, scheduled_start_time__isnull=False)

        if not type_history.exists():
            return {}

        # Group by hour
        by_hour = defaultdict(list)
        for h in type_history:
            hour = h.scheduled_start_time.hour
            efficiency = h.estimated_duration_minutes / h.actual_duration_minutes if h.estimated_duration_minutes else 1
            by_hour[hour].append(efficiency)

        # Find hour with best efficiency
        best_hour = max(by_hour.keys(), key=lambda h: sum(by_hour[h]) / len(by_hour[h]))
        return {
            'hour': best_hour,
            'time_range': f"{best_hour}h-{best_hour + 1}h",
            'efficiency': round(sum(by_hour[best_hour]) / len(by_hour[best_hour]), 2)
        }

    def _get_estimation_accuracy(self, history) -> Dict:
        """Calculate how accurate the user's time estimates are."""
        with_estimates = history.filter(estimated_duration_minutes__isnull=False)

        if not with_estimates.exists():
            return {'accuracy': None}

        total_diff = 0
        underestimates = 0
        overestimates = 0

        for h in with_estimates:
            diff = h.actual_duration_minutes - h.estimated_duration_minutes
            total_diff += abs(diff)
            if diff > 0:
                underestimates += 1
            elif diff < 0:
                overestimates += 1

        avg_diff = total_diff / with_estimates.count()
        return {
            'average_error_minutes': round(avg_diff),
            'tendency': 'underestimate' if underestimates > overestimates else 'overestimate',
            'underestimate_rate': round(underestimates / with_estimates.count() * 100),
        }

    def _get_completion_rate_by_day(self, history) -> Dict:
        """Get completion rate by day of week."""
        day_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        by_day = history.values('day_of_week').annotate(count=Count('id'))

        result = {}
        for entry in by_day:
            result[day_names[entry['day_of_week']]] = entry['count']

        return result

    def _get_reschedule_patterns(self, history) -> List[Dict]:
        """Analyze reschedule patterns."""
        rescheduled = history.filter(was_rescheduled=True)

        if not rescheduled.exists():
            return []

        by_type = rescheduled.values('task_type').annotate(count=Count('id'))
        return list(by_type)

    def _get_energy_patterns(self, history) -> Dict:
        """Analyze energy level patterns."""
        with_energy = history.exclude(energy_level='')

        if not with_energy.exists():
            return {}

        by_energy = with_energy.values('energy_level').annotate(
            count=Count('id'),
            avg_duration=Avg('actual_duration_minutes')
        )

        return {e['energy_level']: {
            'count': e['count'],
            'avg_duration': round(e['avg_duration'] or 0)
        } for e in by_energy}

    def _analyze_patterns_for_suggestions(self, user: User) -> List[Suggestion]:
        """Generate suggestions based on pattern analysis."""
        suggestions = []
        patterns = self.analyze_user_patterns(user)

        if patterns.get('status') == 'insufficient_data':
            return suggestions

        # Suggestion based on estimation accuracy
        accuracy = patterns.get('average_accuracy', {})
        if accuracy.get('tendency') == 'underestimate' and accuracy.get('average_error_minutes', 0) > 15:
            suggestions.append(Suggestion(
                type='pattern',
                message=f"Tu sous-estimes souvent tes taches de ~{accuracy['average_error_minutes']} min. Ajoute un buffer de temps!",
                action=None,
                metadata={'pattern_type': 'estimation'}
            ))

        # Suggestion for best deep work time
        best_time = patterns.get('best_time_for_deep_work', {})
        if best_time.get('hour'):
            suggestions.append(Suggestion(
                type='pattern',
                message=f"Tu es plus efficace pour le travail en profondeur vers {best_time['time_range']}. Planifie tes taches importantes a ce moment!",
                action=None,
                metadata={'pattern_type': 'timing', 'best_hour': best_time['hour']}
            ))

        return suggestions

    # ==================== DURATION PREDICTION ====================

    def predict_duration(
        self,
        user: User,
        task_title: str,
        task_type: str,
        scheduled_time: Optional[time] = None
    ) -> Dict:
        """
        Predict task duration based on historical data.

        Args:
            user: The user
            task_title: Title of the task
            task_type: Type of task (deep_work, shallow, errand)
            scheduled_time: Planned start time

        Returns:
            Dict with prediction and confidence
        """
        history = TaskHistory.objects.filter(user=user)

        # 1. Look for similar task titles
        similar_tasks = history.filter(task_title__icontains=task_title.split()[0] if task_title else '')

        if similar_tasks.count() >= 3:
            avg_duration = similar_tasks.aggregate(avg=Avg('actual_duration_minutes'))['avg']
            return {
                'predicted_minutes': round(avg_duration),
                'confidence': 'high',
                'based_on': f'{similar_tasks.count()} taches similaires',
                'range': {
                    'min': round(avg_duration * 0.8),
                    'max': round(avg_duration * 1.3)
                }
            }

        # 2. Look for same task type
        type_tasks = history.filter(task_type=task_type)

        if type_tasks.count() >= 5:
            avg_duration = type_tasks.aggregate(avg=Avg('actual_duration_minutes'))['avg']
            return {
                'predicted_minutes': round(avg_duration),
                'confidence': 'medium',
                'based_on': f'moyenne des taches {task_type}',
                'range': {
                    'min': round(avg_duration * 0.7),
                    'max': round(avg_duration * 1.5)
                }
            }

        # 3. Default estimates by task type
        defaults = {
            'deep_work': 90,
            'shallow': 30,
            'errand': 45,
        }

        return {
            'predicted_minutes': defaults.get(task_type, 60),
            'confidence': 'low',
            'based_on': 'estimation par defaut',
            'range': {
                'min': defaults.get(task_type, 60) // 2,
                'max': defaults.get(task_type, 60) * 2
            }
        }

    # ==================== CONFLICT DETECTION ====================

    def detect_conflicts(
        self,
        user: User,
        target_date: Optional[date] = None,
        days_ahead: int = 7
    ) -> List[Conflict]:
        """
        Detect scheduling conflicts.

        Args:
            user: The user
            target_date: Start date for conflict check
            days_ahead: Number of days to check

        Returns:
            List of Conflict objects
        """
        if target_date is None:
            target_date = timezone.now().date()

        conflicts = []
        profile = user.profile
        transport_minutes = profile.transport_time_minutes or 0

        for day_offset in range(days_ahead):
            current_date = target_date + timedelta(days=day_offset)
            day_of_week = current_date.weekday()
            previous_day_of_week = (day_of_week - 1) % 7

            # Get all blocks for this day
            recurring = list(RecurringBlock.objects.filter(
                user=user, day_of_week=day_of_week, active=True
            ))
            scheduled = list(ScheduledBlock.objects.filter(
                user=user, date=current_date
            ))

            # Also get night shift blocks from previous day (they end today)
            previous_night_blocks = list(RecurringBlock.objects.filter(
                user=user, day_of_week=previous_day_of_week, active=True, is_night_shift=True
            ))

            # Check for overlaps
            all_blocks = []
            for rb in recurring:
                is_overnight = self._is_overnight_block(rb.start_time, rb.end_time)
                all_blocks.append({
                    'id': rb.id,
                    'type': 'recurring',
                    'title': rb.title,
                    'start': rb.start_time,
                    'end': rb.end_time,
                    'date': current_date,
                    'is_overnight': is_overnight,
                    'is_night_shift': rb.is_night_shift,
                    'block_type': rb.block_type,
                })

            # Add previous day's night shift blocks (they end today morning)
            for rb in previous_night_blocks:
                # These blocks started yesterday but end today
                all_blocks.append({
                    'id': rb.id,
                    'type': 'recurring_from_yesterday',
                    'title': rb.title,
                    'start': time(0, 0),  # Starts at midnight (continuation)
                    'end': rb.end_time,   # Ends in the morning
                    'date': current_date,
                    'is_overnight': False,  # We're only looking at the morning part
                    'is_night_shift': True,
                    'block_type': rb.block_type,
                    'real_start': rb.start_time,  # Original start from yesterday
                })

            for sb in scheduled:
                is_overnight = self._is_overnight_block(sb.start_time, sb.end_time)
                all_blocks.append({
                    'id': sb.id,
                    'type': 'scheduled',
                    'title': sb.task.title,
                    'start': sb.start_time,
                    'end': sb.end_time,
                    'date': current_date,
                    'is_overnight': is_overnight,
                    'is_night_shift': False,
                    'block_type': 'task',
                })

            # Sort by start time (but overnight blocks that START today go at their start time)
            all_blocks.sort(key=lambda x: x['start'])

            # Detect overlaps using the new overlap detection
            for i, block_a in enumerate(all_blocks):
                for j, block_b in enumerate(all_blocks):
                    if i >= j:
                        continue  # Avoid duplicate checks

                    overlaps, overlap_minutes = self._blocks_overlap(
                        block_a['start'], block_a['end'],
                        block_b['start'], block_b['end'],
                        block_a['is_overnight'], block_b['is_overnight']
                    )

                    if overlaps and overlap_minutes > 0:
                        # Special handling for sleep vs work conflicts
                        is_sleep_work_conflict = (
                            (block_a.get('block_type') == 'sleep' and block_b.get('is_night_shift')) or
                            (block_b.get('block_type') == 'sleep' and block_a.get('is_night_shift'))
                        )

                        if is_sleep_work_conflict:
                            work_block = block_a if block_a.get('is_night_shift') else block_b
                            sleep_block = block_b if block_a.get('is_night_shift') else block_a
                            work_end = work_block['end']

                            # Calculate suggested sleep time (after work + transport)
                            suggested_start = (datetime.combine(date.today(), work_end) +
                                             timedelta(minutes=transport_minutes)).time()

                            conflicts.append(Conflict(
                                type='overlap',
                                severity='high',
                                message=f"'{sleep_block['title']}' chevauche '{work_block['title']}'. Pour un travail de nuit, le sommeil doit etre apres la fin du travail.",
                                blocks_involved=[block_a['id'], block_b['id']],
                                suggested_resolution=f"Deplacer '{sleep_block['title']}' a {suggested_start.strftime('%H:%M')} (apres travail{' + ' + str(transport_minutes) + 'min trajet' if transport_minutes else ''})"
                            ))
                        else:
                            conflicts.append(Conflict(
                                type='overlap',
                                severity='high' if overlap_minutes > 30 else 'medium',
                                message=f"Chevauchement de {overlap_minutes}min entre '{block_a['title']}' et '{block_b['title']}' le {current_date.strftime('%d/%m')}",
                                blocks_involved=[block_a['id'], block_b['id']],
                                suggested_resolution=f"Deplacer '{block_b['title']}' apres {block_a['end'].strftime('%H:%M')}"
                            ))

            # Check for deep work overload
            deep_work_minutes = 0
            for sb in scheduled:
                if sb.task.task_type == 'deep_work':
                    deep_work_minutes += self._time_diff_minutes(sb.start_time, sb.end_time)

            max_deep_work = user.profile.max_deep_work_hours_per_day * 60
            if deep_work_minutes > max_deep_work:
                conflicts.append(Conflict(
                    type='overload',
                    severity='medium',
                    message=f"Trop de deep work le {current_date.strftime('%d/%m')}: {deep_work_minutes}min (max: {max_deep_work}min)",
                    blocks_involved=[],
                    suggested_resolution="Deplacer certaines taches deep work a un autre jour"
                ))

        # Check for deadline risks
        pending_tasks = Task.objects.filter(
            user=user,
            completed=False,
            deadline__isnull=False,
            deadline__lte=timezone.now() + timedelta(days=days_ahead)
        )

        for task in pending_tasks:
            scheduled_count = ScheduledBlock.objects.filter(
                task=task,
                date__lte=task.deadline.date()
            ).count()

            if scheduled_count == 0:
                days_until = (task.deadline.date() - target_date).days
                conflicts.append(Conflict(
                    type='deadline_risk',
                    severity='high' if days_until <= 1 else 'medium',
                    message=f"'{task.title}' n'est pas planifie mais due dans {days_until} jour(s)!",
                    blocks_involved=[],
                    suggested_resolution="Planifier cette tache immediatement"
                ))

        return conflicts

    # ==================== SMART RESCHEDULING ====================

    def smart_reschedule(
        self,
        user: User,
        overflowed_block_id: int,
        actual_end_time: time
    ) -> Dict:
        """
        Intelligently reschedule blocks when a task overflows.

        Args:
            user: The user
            overflowed_block_id: ID of the block that overflowed
            actual_end_time: When the task actually ended

        Returns:
            Dict with rescheduling actions
        """
        try:
            overflowed = ScheduledBlock.objects.get(id=overflowed_block_id, user=user)
        except ScheduledBlock.DoesNotExist:
            return {'error': 'Block not found'}

        original_end = overflowed.end_time
        overflow_minutes = self._time_diff_minutes(original_end, actual_end_time)

        if overflow_minutes <= 0:
            return {'status': 'no_overflow', 'message': 'Pas de debordement detecte'}

        # Find affected blocks (those that start after original end but before actual end)
        affected_blocks = ScheduledBlock.objects.filter(
            user=user,
            date=overflowed.date,
            start_time__gte=original_end,
            start_time__lt=actual_end_time
        ).order_by('start_time')

        rescheduled = []
        current_shift = timedelta(minutes=overflow_minutes)

        for block in affected_blocks:
            new_start = (datetime.combine(date.today(), block.start_time) + current_shift).time()
            new_end = (datetime.combine(date.today(), block.end_time) + current_shift).time()

            # Check if new end exceeds end of day
            if new_end > time(23, 0):
                # Need to move to next day
                next_day = overflowed.date + timedelta(days=1)
                rescheduled.append({
                    'block_id': block.id,
                    'task_title': block.task.title,
                    'action': 'move_to_next_day',
                    'new_date': next_day.isoformat(),
                    'message': f"'{block.task.title}' deplace au {next_day.strftime('%d/%m')}"
                })
            else:
                # Just shift within the day
                old_start = block.start_time
                block.start_time = new_start
                block.end_time = new_end
                block.save()

                rescheduled.append({
                    'block_id': block.id,
                    'task_title': block.task.title,
                    'action': 'shifted',
                    'old_start': old_start.isoformat(),
                    'new_start': new_start.isoformat(),
                    'shift_minutes': overflow_minutes,
                    'message': f"'{block.task.title}' decale de {overflow_minutes}min"
                })

        # Update the overflowed block's actual duration
        overflowed.end_time = actual_end_time
        actual_duration = self._time_diff_minutes(overflowed.start_time, actual_end_time)
        overflowed.actual_duration_minutes = actual_duration
        overflowed.save()

        return {
            'status': 'rescheduled',
            'overflow_minutes': overflow_minutes,
            'blocks_affected': len(rescheduled),
            'rescheduled': rescheduled,
            'message': f"Debordement de {overflow_minutes}min. {len(rescheduled)} bloc(s) replanifie(s)."
        }

    # ==================== NATURAL LANGUAGE SCHEDULING ====================

    def parse_scheduling_request(self, user: User, message: str) -> Dict:
        """
        Parse a natural language scheduling request.

        Examples:
            "Planifie ma révision de maths cette semaine"
            "Ajoute 2h de sport mardi et jeudi"
            "Je veux étudier 3h par jour cette semaine"

        Args:
            user: The user
            message: Natural language request

        Returns:
            Dict with parsed scheduling intent
        """
        if not self.client:
            return self._parse_scheduling_local(message)

        prompt = f"""Analyse cette demande de planification et extrais les informations en JSON:

Demande: "{message}"

Reponds UNIQUEMENT avec un JSON valide:
{{
    "action": "schedule" | "create_recurring" | "modify" | "delete",
    "task_title": "titre de la tache",
    "task_type": "deep_work" | "shallow" | "errand",
    "duration_minutes": nombre ou null,
    "frequency": "once" | "daily" | "weekly" | "specific_days",
    "days": [0, 1, 2, 3, 4, 5, 6] ou null,
    "preferred_time": "morning" | "afternoon" | "evening" | null,
    "deadline": "YYYY-MM-DD" ou null,
    "priority": 1-10 ou null,
    "constraints": ["avant 18h", "apres le travail", etc] ou []
}}

Jours: 0=Lundi, 1=Mardi, 2=Mercredi, 3=Jeudi, 4=Vendredi, 5=Samedi, 6=Dimanche
"""

        try:
            response = self._call_gemini(prompt)
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                parsed['raw_request'] = message
                parsed['parsed_by'] = 'ai'
                return parsed
        except Exception as e:
            logger.error(f"AI parsing failed: {e}")

        return self._parse_scheduling_local(message)

    def _parse_scheduling_local(self, message: str) -> Dict:
        """Fallback local parsing without AI."""
        message_lower = message.lower()

        result = {
            'action': 'schedule',
            'task_title': None,
            'task_type': 'shallow',
            'duration_minutes': None,
            'frequency': 'once',
            'days': None,
            'preferred_time': None,
            'deadline': None,
            'priority': 5,
            'constraints': [],
            'raw_request': message,
            'parsed_by': 'local'
        }

        # Detect task type
        if any(word in message_lower for word in ['revision', 'etude', 'travail', 'projet', 'rediger']):
            result['task_type'] = 'deep_work'
        elif any(word in message_lower for word in ['course', 'rdv', 'rendez-vous', 'appel']):
            result['task_type'] = 'errand'

        # Detect duration
        duration_match = re.search(r'(\d+)\s*h(?:eure)?s?', message_lower)
        if duration_match:
            result['duration_minutes'] = int(duration_match.group(1)) * 60

        minutes_match = re.search(r'(\d+)\s*min(?:ute)?s?', message_lower)
        if minutes_match:
            result['duration_minutes'] = (result['duration_minutes'] or 0) + int(minutes_match.group(1))

        # Detect days
        day_patterns = {
            'lundi': 0, 'mardi': 1, 'mercredi': 2, 'jeudi': 3,
            'vendredi': 4, 'samedi': 5, 'dimanche': 6
        }
        detected_days = []
        for day_name, day_num in day_patterns.items():
            if day_name in message_lower:
                detected_days.append(day_num)

        if 'semaine' in message_lower and not detected_days:
            detected_days = [0, 1, 2, 3, 4]  # Weekdays
            result['frequency'] = 'weekly'

        if detected_days:
            result['days'] = detected_days

        # Detect time preference
        if any(word in message_lower for word in ['matin', 'tot']):
            result['preferred_time'] = 'morning'
        elif any(word in message_lower for word in ['apres-midi', 'aprem']):
            result['preferred_time'] = 'afternoon'
        elif any(word in message_lower for word in ['soir', 'soiree']):
            result['preferred_time'] = 'evening'

        # Extract title (simplified)
        # Remove common words and take the rest as title
        title_words = message.split()
        stop_words = ['planifie', 'ajoute', 'je', 'veux', 'ma', 'mon', 'mes', 'de', 'du', 'la', 'le', 'les', 'cette', 'ce', 'semaine', 'jour', 'jours']
        title_words = [w for w in title_words if w.lower() not in stop_words and not re.match(r'^\d+h?$', w.lower())]
        if title_words:
            result['task_title'] = ' '.join(title_words[:5]).capitalize()

        return result

    def execute_scheduling_request(self, user: User, parsed_request: Dict) -> Dict:
        """
        Execute a parsed scheduling request.

        Args:
            user: The user
            parsed_request: Parsed request from parse_scheduling_request

        Returns:
            Dict with execution result
        """
        action = parsed_request.get('action', 'schedule')

        if action == 'schedule':
            # Create task and schedule it
            task_title = parsed_request.get('task_title') or 'Nouvelle tache'
            task_type = parsed_request.get('task_type', 'shallow')
            duration = parsed_request.get('duration_minutes') or 60
            days = parsed_request.get('days') or [timezone.now().weekday()]

            # Create the task
            task = Task.objects.create(
                user=user,
                title=task_title,
                task_type=task_type,
                estimated_duration_minutes=duration,
                priority=parsed_request.get('priority') or 5
            )

            # Schedule for each specified day
            scheduled_blocks = []
            today = timezone.now().date()

            for day in days:
                # Find next occurrence of this day
                days_ahead = (day - today.weekday()) % 7
                if days_ahead == 0 and parsed_request.get('frequency') == 'weekly':
                    days_ahead = 7
                target_date = today + timedelta(days=days_ahead)

                # Find a suitable slot
                gaps = self._find_free_time_gaps(user, target_date)
                preferred_time = parsed_request.get('preferred_time')

                suitable_gap = None
                for gap in gaps:
                    if gap.duration_minutes >= duration:
                        if preferred_time:
                            gap_hour = gap.start_time.hour
                            if preferred_time == 'morning' and 6 <= gap_hour < 12:
                                suitable_gap = gap
                                break
                            elif preferred_time == 'afternoon' and 12 <= gap_hour < 18:
                                suitable_gap = gap
                                break
                            elif preferred_time == 'evening' and 18 <= gap_hour < 23:
                                suitable_gap = gap
                                break
                        else:
                            suitable_gap = gap
                            break

                if suitable_gap:
                    end_time = (datetime.combine(date.today(), suitable_gap.start_time) + timedelta(minutes=duration)).time()
                    block = ScheduledBlock.objects.create(
                        user=user,
                        task=task,
                        date=target_date,
                        start_time=suitable_gap.start_time,
                        end_time=end_time
                    )
                    scheduled_blocks.append({
                        'date': target_date.isoformat(),
                        'start': suitable_gap.start_time.isoformat(),
                        'end': end_time.isoformat()
                    })

            return {
                'status': 'success',
                'task_id': task.id,
                'task_title': task_title,
                'scheduled_blocks': scheduled_blocks,
                'message': f"'{task_title}' planifie pour {len(scheduled_blocks)} creneau(x)"
            }

        elif action == 'create_recurring':
            # Create recurring blocks
            # Implementation for recurring block creation
            pass

        return {'status': 'unknown_action', 'action': action}

    # ==================== UTILITY METHODS ====================

    def _time_diff_minutes(self, start: time, end: time, allow_overnight: bool = False) -> int:
        """
        Calculate difference between two times in minutes.

        Args:
            start: Start time
            end: End time
            allow_overnight: If True, handles cases where end < start (crosses midnight)

        Returns:
            Duration in minutes (always positive if allow_overnight=True)
        """
        start_minutes = start.hour * 60 + start.minute
        end_minutes = end.hour * 60 + end.minute

        diff = end_minutes - start_minutes

        # If negative and overnight allowed, add 24 hours
        if diff < 0 and allow_overnight:
            diff += 24 * 60  # Add 24 hours in minutes

        return diff

    def _is_overnight_block(self, start: time, end: time) -> bool:
        """Check if a block crosses midnight."""
        return end < start

    def _blocks_overlap(self, a_start: time, a_end: time, b_start: time, b_end: time,
                        a_is_overnight: bool = False, b_is_overnight: bool = False) -> Tuple[bool, int]:
        """
        Check if two blocks overlap, handling overnight blocks.

        Returns:
            Tuple of (overlaps: bool, overlap_minutes: int)
        """
        # Convert to minutes from midnight, handling overnight
        def to_range(start: time, end: time, is_overnight: bool) -> List[Tuple[int, int]]:
            start_min = start.hour * 60 + start.minute
            end_min = end.hour * 60 + end.minute

            if is_overnight:
                # Split into two ranges: start->midnight and midnight->end
                return [(start_min, 24 * 60), (0, end_min)]
            else:
                return [(start_min, end_min)]

        ranges_a = to_range(a_start, a_end, a_is_overnight)
        ranges_b = to_range(b_start, b_end, b_is_overnight)

        total_overlap = 0
        for (a1, a2) in ranges_a:
            for (b1, b2) in ranges_b:
                overlap_start = max(a1, b1)
                overlap_end = min(a2, b2)
                if overlap_end > overlap_start:
                    total_overlap += overlap_end - overlap_start

        return (total_overlap > 0, total_overlap)

    def _get_energy_level(self, slot_time: time, preference: str) -> str:
        """Determine energy level for a given time."""
        hour = slot_time.hour
        levels = self.ENERGY_LEVELS.get(preference, self.ENERGY_LEVELS['morning'])

        for level, ranges in levels.items():
            for start_hour, end_hour in ranges:
                if start_hour <= hour < end_hour:
                    return level

        return 'low'

    def record_task_completion(
        self,
        user: User,
        task: Task,
        actual_duration: int,
        scheduled_time: Optional[time] = None,
        was_rescheduled: bool = False,
        reschedule_count: int = 0
    ):
        """
        Record task completion for future predictions.

        Args:
            user: The user
            task: The completed task
            actual_duration: Actual duration in minutes
            scheduled_time: When the task was scheduled
            was_rescheduled: Whether the task was rescheduled
            reschedule_count: Number of times rescheduled
        """
        TaskHistory.objects.create(
            user=user,
            task_title=task.title,
            task_type=task.task_type,
            estimated_duration_minutes=task.estimated_duration_minutes,
            actual_duration_minutes=actual_duration,
            scheduled_start_time=scheduled_time,
            day_of_week=timezone.now().weekday(),
            completed_at=timezone.now(),
            energy_level=self._get_energy_level(
                scheduled_time or timezone.now().time(),
                user.profile.peak_productivity_time
            ),
            was_rescheduled=was_rescheduled,
            reschedule_count=reschedule_count
        )
