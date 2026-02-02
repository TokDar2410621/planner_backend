"""
Security service for input validation and rate limiting.

Handles:
- Input sanitization (prompt injection prevention)
- Rate limiting
- Resource limits validation
- Argument validation for blocks and tasks
"""
import logging
import re
from datetime import datetime
from typing import Tuple

from django.contrib.auth.models import User
from django.core.cache import cache

from core.models import RecurringBlock, Task

logger = logging.getLogger(__name__)

# Security constants
MAX_MESSAGE_LENGTH = 5000  # Maximum characters per message
MAX_REQUESTS_PER_MINUTE = 20  # Rate limit per user
RATE_LIMIT_WINDOW = 60  # seconds
MAX_BLOCKS_PER_DAY = 15  # Max recurring blocks per day per user
MAX_TASKS_PER_USER = 100  # Max pending tasks per user


class SecurityService:
    """
    Centralized security validation service.

    Provides input sanitization, rate limiting, and resource validation.
    """

    # Prompt injection patterns to filter
    INJECTION_PATTERNS = [
        r'ignore\s+(all\s+)?(previous|above|prior)\s+instructions?',
        r'disregard\s+(all\s+)?(previous|above|prior)',
        r'forget\s+(everything|all)',
        r'you\s+are\s+now\s+',
        r'new\s+instructions?:',
        r'system\s*:\s*',
        r'<\s*system\s*>',
        r'\]\s*\[\s*system',
        r'###\s*(system|instruction)',
        r'IMPORTANT:\s*ignore',
    ]

    # Valid block and task types
    VALID_BLOCK_TYPES = ['course', 'work', 'sleep', 'meal', 'sport', 'project', 'revision', 'other']
    VALID_TASK_TYPES = ['deep_work', 'shallow', 'errand']

    def sanitize_input(self, message: str) -> Tuple[str, bool]:
        """
        Sanitize user input to prevent prompt injection attacks.

        Note: These filters are symbolic and don't provide complete security.
        Real security comes from never trusting AI output and validating
        all actions server-side.

        Args:
            message: Raw user message

        Returns:
            Tuple of (sanitized_message, is_safe)
        """
        if not message:
            return "", True

        # Truncate to max length
        if len(message) > MAX_MESSAGE_LENGTH:
            message = message[:MAX_MESSAGE_LENGTH]
            logger.warning(f"Message truncated to {MAX_MESSAGE_LENGTH} characters")

        # Detect and filter injection patterns
        message_lower = message.lower()
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                logger.warning(f"Potential prompt injection detected: {pattern}")
                message = re.sub(pattern, '[filtered]', message, flags=re.IGNORECASE)

        # Remove control characters (except newlines and tabs)
        message = ''.join(char for char in message if char.isprintable() or char in '\n\t')

        return message, True

    def check_rate_limit(self, user: User) -> Tuple[bool, str]:
        """
        Check if user has exceeded rate limit.

        Args:
            user: The user making the request

        Returns:
            Tuple of (is_allowed, error_message)
        """
        cache_key = f"chat_rate_limit:{user.id}"
        request_count = cache.get(cache_key, 0)

        if request_count >= MAX_REQUESTS_PER_MINUTE:
            logger.warning(f"Rate limit exceeded for user {user.id}")
            return False, "Tu envoies trop de messages. Attends un moment avant de réessayer."

        # Increment counter
        cache.set(cache_key, request_count + 1, RATE_LIMIT_WINDOW)
        return True, ""

    def validate_block_limits(self, user: User, day: int) -> Tuple[bool, str]:
        """
        Validate that user hasn't exceeded block limits for a given day.

        Args:
            user: The user
            day: Day of week (0-6)

        Returns:
            Tuple of (is_allowed, error_message)
        """
        blocks_count = RecurringBlock.objects.filter(
            user=user, day_of_week=day, active=True
        ).count()

        if blocks_count >= MAX_BLOCKS_PER_DAY:
            return False, f"Tu as déjà {blocks_count} blocs ce jour-là. Supprime-en d'abord."
        return True, ""

    def validate_task_limits(self, user: User) -> Tuple[bool, str]:
        """
        Validate that user hasn't exceeded task limits.

        Args:
            user: The user

        Returns:
            Tuple of (is_allowed, error_message)
        """
        tasks_count = Task.objects.filter(user=user, completed=False).count()

        if tasks_count >= MAX_TASKS_PER_USER:
            return False, f"Tu as déjà {tasks_count} tâches en attente. Termine-en d'abord."
        return True, ""

    def validate_block_args(self, args: dict) -> Tuple[bool, str, dict]:
        """
        Validate and sanitize block creation arguments.

        Args:
            args: Arguments from AI function call

        Returns:
            Tuple of (is_valid, error_message, sanitized_args)
        """
        sanitized = {}

        # Validate title (max 100 chars, printable only)
        title = str(args.get('title', 'Bloc'))[:100]
        title = ''.join(c for c in title if c.isprintable())
        sanitized['title'] = title or 'Bloc'

        # Validate block_type
        block_type = str(args.get('block_type', 'other')).lower()
        sanitized['block_type'] = block_type if block_type in self.VALID_BLOCK_TYPES else 'other'

        # Validate days
        days = args.get('days', [])
        if isinstance(days, (int, float)):
            days = [int(days)]
        elif isinstance(days, str):
            try:
                import ast
                days = ast.literal_eval(days)
            except:
                days = []

        sanitized['days'] = [
            int(d) for d in days
            if isinstance(d, (int, float)) and 0 <= int(d) <= 6
        ]
        if not sanitized['days']:
            sanitized['days'] = [0, 1, 2, 3, 4]  # Default to weekdays

        # Validate times
        sanitized['start_time'] = self._parse_time(args.get('start_time'), '09:00')
        sanitized['end_time'] = self._parse_time(args.get('end_time'), '10:00')

        # Validate location (max 200 chars)
        location = str(args.get('location', ''))[:200]
        sanitized['location'] = ''.join(c for c in location if c.isprintable())

        # Night shift flag
        sanitized['is_night_shift'] = bool(args.get('is_night_shift', False))

        return True, "", sanitized

    def validate_task_args(self, args: dict) -> Tuple[bool, str, dict]:
        """
        Validate and sanitize task creation arguments.

        Args:
            args: Arguments from AI function call

        Returns:
            Tuple of (is_valid, error_message, sanitized_args)
        """
        sanitized = {}

        # Validate title (max 200 chars)
        title = str(args.get('title', 'Tâche'))[:200]
        title = ''.join(c for c in title if c.isprintable())
        sanitized['title'] = title or 'Tâche'

        # Validate description (max 1000 chars)
        description = str(args.get('description', ''))[:1000]
        sanitized['description'] = ''.join(
            c for c in description if c.isprintable() or c in '\n'
        )

        # Validate priority (1-10)
        try:
            priority = int(args.get('priority', 5))
            sanitized['priority'] = max(1, min(10, priority))
        except (ValueError, TypeError):
            sanitized['priority'] = 5

        # Validate task_type
        task_type = str(args.get('task_type', 'shallow')).lower()
        sanitized['task_type'] = task_type if task_type in self.VALID_TASK_TYPES else 'shallow'

        # Validate deadline (if present)
        deadline = args.get('deadline')
        if deadline and deadline != 'null':
            sanitized['deadline'] = self._parse_deadline(deadline)
        else:
            sanitized['deadline'] = None

        # Validate estimated duration
        try:
            duration = args.get('estimated_duration_minutes')
            if duration:
                sanitized['estimated_duration_minutes'] = max(5, min(480, int(duration)))
            else:
                sanitized['estimated_duration_minutes'] = None
        except (ValueError, TypeError):
            sanitized['estimated_duration_minutes'] = None

        return True, "", sanitized

    def _parse_time(self, time_str: str, default: str) -> str:
        """Parse and validate a time string."""
        if not time_str:
            return default

        match = re.match(r'^(\d{1,2}):(\d{2})$', str(time_str))
        if match:
            hour, minute = int(match.group(1)), int(match.group(2))
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return f"{hour:02d}:{minute:02d}"

        return default

    def _parse_deadline(self, deadline: str) -> datetime:
        """Parse deadline from various formats."""
        date_formats = [
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d',
            '%d/%m/%Y %H:%M',
            '%d/%m/%Y',
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(str(deadline), fmt)
            except ValueError:
                continue

        return None
