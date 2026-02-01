"""
Chat engine service for conversational AI interactions.
"""
import json
import logging
import re
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Tuple

from django.conf import settings
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.cache import cache

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from core.models import (
    UserProfile,
    UploadedDocument,
    ConversationMessage,
    Task,
    RecurringBlock,
)
from services.ai_scheduler import AIScheduler
from utils.helpers import retry_with_backoff

logger = logging.getLogger(__name__)

# Security constants
MAX_MESSAGE_LENGTH = 5000  # Maximum characters per message
MAX_REQUESTS_PER_MINUTE = 20  # Rate limit per user
RATE_LIMIT_WINDOW = 60  # seconds
MAX_BLOCKS_PER_DAY = 15  # Max recurring blocks per day per user
MAX_TASKS_PER_USER = 100  # Max pending tasks per user


class ChatEngine:
    """
    Manages conversation with the user using Gemini AI.

    Handles onboarding, task extraction, and general assistance.
    """

    ONBOARDING_STEPS = [
        "upload_schedule",  # Step 0: Ask for course/work schedule
        "confirm_schedule",  # Step 1: Confirm extracted data
        "preferences",      # Step 2: Ask about productivity preferences
        "completed",        # Step 3: Onboarding done
    ]

    def __init__(self):
        """Initialize the chat engine."""
        if GEMINI_AVAILABLE and settings.GEMINI_API_KEY:
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            self.model_name = 'gemini-2.5-flash'
        else:
            self.client = None
            self.model_name = None
            logger.warning("Gemini API not configured. Chat will be limited.")

    @retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=30.0)
    def _call_gemini(self, contents, config=None):
        """
        Call Gemini API with retry logic for rate limits (429).

        Args:
            contents: Content to send to Gemini
            config: Optional GenerateContentConfig

        Returns:
            Response from Gemini
        """
        if config:
            return self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
        return self.client.models.generate_content(
            model=self.model_name,
            contents=contents
        )

    # ==================== SECURITY LAYER ====================

    def _sanitize_input(self, message: str) -> Tuple[str, bool]:
        """
        Sanitize user input to prevent prompt injection attacks.

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

        # Detect prompt injection patterns
        injection_patterns = [
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

        message_lower = message.lower()
        for pattern in injection_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                logger.warning(f"Potential prompt injection detected: {pattern}")
                # Don't block, but flag and sanitize
                message = re.sub(pattern, '[filtered]', message, flags=re.IGNORECASE)

        # Remove potential control characters (except newlines and tabs)
        message = ''.join(char for char in message if char.isprintable() or char in '\n\t')

        return message, True

    def _check_rate_limit(self, user: User) -> Tuple[bool, str]:
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

    def _validate_block_limits(self, user: User, day: int) -> Tuple[bool, str]:
        """
        Validate that user hasn't exceeded block limits.

        Args:
            user: The user
            day: Day of week (0-6)

        Returns:
            Tuple of (is_allowed, error_message)
        """
        blocks_count = RecurringBlock.objects.filter(user=user, day_of_week=day, active=True).count()
        if blocks_count >= MAX_BLOCKS_PER_DAY:
            return False, f"Tu as déjà {blocks_count} blocs ce jour-là. Supprime-en d'abord."
        return True, ""

    def _validate_task_limits(self, user: User) -> Tuple[bool, str]:
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

    def _validate_block_args(self, args: dict) -> Tuple[bool, str, dict]:
        """
        Validate and sanitize block creation arguments.

        Args:
            args: Arguments from AI function call

        Returns:
            Tuple of (is_valid, error_message, sanitized_args)
        """
        sanitized = {}

        # Validate title
        title = str(args.get('title', 'Bloc'))[:100]  # Max 100 chars
        title = ''.join(c for c in title if c.isprintable())
        sanitized['title'] = title or 'Bloc'

        # Validate block_type
        valid_types = ['course', 'work', 'sleep', 'meal', 'sport', 'project', 'other']
        block_type = str(args.get('block_type', 'other')).lower()
        sanitized['block_type'] = block_type if block_type in valid_types else 'other'

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
        sanitized['days'] = [int(d) for d in days if isinstance(d, (int, float)) and 0 <= int(d) <= 6]
        if not sanitized['days']:
            sanitized['days'] = [0, 1, 2, 3, 4]  # Default to weekdays

        # Validate times
        def parse_time(time_str: str, default: str) -> str:
            if not time_str:
                return default
            match = re.match(r'^(\d{1,2}):(\d{2})$', str(time_str))
            if match:
                hour, minute = int(match.group(1)), int(match.group(2))
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    return f"{hour:02d}:{minute:02d}"
            return default

        sanitized['start_time'] = parse_time(args.get('start_time'), '09:00')
        sanitized['end_time'] = parse_time(args.get('end_time'), '10:00')

        # Validate location
        location = str(args.get('location', ''))[:200]  # Max 200 chars
        sanitized['location'] = ''.join(c for c in location if c.isprintable())

        return True, "", sanitized

    def _validate_task_args(self, args: dict) -> Tuple[bool, str, dict]:
        """
        Validate and sanitize task creation arguments.

        Args:
            args: Arguments from AI function call

        Returns:
            Tuple of (is_valid, error_message, sanitized_args)
        """
        sanitized = {}

        # Validate title
        title = str(args.get('title', 'Tâche'))[:200]
        title = ''.join(c for c in title if c.isprintable())
        sanitized['title'] = title or 'Tâche'

        # Validate description
        description = str(args.get('description', ''))[:1000]
        sanitized['description'] = ''.join(c for c in description if c.isprintable() or c in '\n')

        # Validate priority
        try:
            priority = int(args.get('priority', 5))
            sanitized['priority'] = max(1, min(10, priority))
        except (ValueError, TypeError):
            sanitized['priority'] = 5

        # Validate task_type
        valid_types = ['deep_work', 'shallow', 'errand']
        task_type = str(args.get('task_type', 'shallow')).lower()
        sanitized['task_type'] = task_type if task_type in valid_types else 'shallow'

        # Validate deadline (if present)
        deadline = args.get('deadline')
        if deadline and deadline != 'null':
            try:
                # Try parsing various formats
                for fmt in ['%Y-%m-%d %H:%M', '%Y-%m-%d', '%d/%m/%Y %H:%M', '%d/%m/%Y']:
                    try:
                        sanitized['deadline'] = datetime.strptime(str(deadline), fmt)
                        break
                    except ValueError:
                        continue
            except:
                sanitized['deadline'] = None
        else:
            sanitized['deadline'] = None

        return True, "", sanitized

    def _ensure_user_isolation(self, user: User, object_user_id: int) -> bool:
        """
        Ensure that operations only affect the authenticated user's data.

        Args:
            user: The authenticated user
            object_user_id: The user_id of the object being accessed

        Returns:
            True if user owns the object, False otherwise
        """
        if user.id != object_user_id:
            logger.error(f"User isolation violation: user {user.id} tried to access data of user {object_user_id}")
            return False
        return True

    # ==================== END SECURITY LAYER ====================

    def generate_response(
        self,
        user: User,
        message: str,
        attachment: Optional[UploadedDocument] = None
    ) -> dict:
        """
        Generate a response to the user's message.

        Args:
            user: The user sending the message
            message: The user's message content
            attachment: Optional uploaded document

        Returns:
            dict: Response containing 'response', optionally 'extracted_data', 'tasks_created', and 'quick_replies'
        """
        result = {
            'response': '',
            'extracted_data': None,
            'tasks_created': [],
            'quick_replies': [],
            'interactive_inputs': [],
        }

        # ========== SECURITY CHECKS ==========

        # 1. Rate limiting
        is_allowed, rate_error = self._check_rate_limit(user)
        if not is_allowed:
            result['response'] = rate_error
            return result

        # 2. Sanitize input (anti-prompt injection)
        message, is_safe = self._sanitize_input(message)
        if not message and not attachment:
            result['response'] = "Message vide. Que puis-je faire pour toi?"
            return result

        # ========== END SECURITY CHECKS ==========

        # Save user message (after sanitization)
        user_msg = ConversationMessage.objects.create(
            user=user,
            role='user',
            content=message,
            attachment=attachment,
        )

        # Build context
        system_prompt = self._build_system_prompt(user)
        history = self._get_conversation_history(user)

        # If there's an attachment with extracted data, include it in result
        if attachment and attachment.processed and attachment.extracted_data:
            result['extracted_data'] = attachment.extracted_data

        try:
            # Handle document uploads with special processing
            if attachment:
                doc_response = self._handle_document_upload(user, attachment)
                result['response'] = doc_response['text']
                result['quick_replies'] = doc_response.get('quick_replies', [])

            # Let AI handle ALL conversations - no keyword detection
            # The AI has tools to: create blocks, create tasks, show proposals,
            # accept proposals, add document data, update preferences, etc.
            else:
                gemini_response = self._smart_conversation(user, message, system_prompt, history)
                result['response'] = gemini_response.get('text', '')
                result['quick_replies'] = gemini_response.get('quick_replies', [])
                if gemini_response.get('tasks_created'):
                    result['tasks_created'] = gemini_response['tasks_created']

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            result['response'] = "Désolé, j'ai eu un problème. Peux-tu reformuler ta demande?"

        # Save assistant response
        ConversationMessage.objects.create(
            user=user,
            role='assistant',
            content=result['response'],
            metadata={
                'tasks_created': [t.id for t in result['tasks_created']] if result['tasks_created'] else [],
            }
        )

        return result

    def _build_system_prompt(self, user: User) -> str:
        """
        Build the system prompt with user context including full schedule.

        Args:
            user: The current user

        Returns:
            str: The system prompt
        """
        profile = user.profile

        # Get all recurring blocks
        blocks = RecurringBlock.objects.filter(user=user, active=True).order_by('day_of_week', 'start_time')
        pending_tasks = Task.objects.filter(user=user, completed=False).order_by('deadline', '-priority')[:10]

        # Build schedule summary by day
        days_fr = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        schedule_by_day = {i: [] for i in range(7)}
        for block in blocks:
            schedule_by_day[block.day_of_week].append(
                f"  - {block.start_time.strftime('%H:%M')}-{block.end_time.strftime('%H:%M')}: {block.title} ({block.block_type})"
            )

        schedule_text = ""
        for day_num, day_name in enumerate(days_fr):
            if schedule_by_day[day_num]:
                schedule_text += f"\n{day_name}:\n" + "\n".join(schedule_by_day[day_num])

        if not schedule_text:
            schedule_text = "\nAucun bloc récurrent configuré."

        # Build tasks summary
        tasks_text = ""
        if pending_tasks:
            tasks_text = "\n\nTâches en attente:"
            for task in pending_tasks:
                deadline = f" (deadline: {task.deadline.strftime('%d/%m')})" if task.deadline else ""
                tasks_text += f"\n- {task.title}{deadline} [priorité {task.priority}/10]"

        return f"""Tu es un assistant de planification intelligent. Tu aides {user.first_name or user.username} à organiser son temps.

PROFIL UTILISATEUR:
- Sommeil minimum: {profile.min_sleep_hours}h
- Pic de productivité: {profile.get_peak_productivity_time_display()}
- Max travail profond/jour: {profile.max_deep_work_hours_per_day}h
- Temps de transport: {profile.transport_time_minutes} min

PLANNING ACTUEL:{schedule_text}{tasks_text}

TES CAPACITÉS:
1. Tu peux CRÉER des blocs récurrents (sommeil, projets, sport, etc.)
2. Tu peux AJOUTER des tâches avec deadlines
3. Tu peux ANALYSER le planning et suggérer des optimisations
4. Tu connais les créneaux libres de l'utilisateur

RÈGLES IMPORTANTES:
- Réponds en français, sois amical et naturel
- Tu peux avoir des conversations normales! Pas besoin de toujours parler de planning
- Quand l'utilisateur demande une action de planning, sois proactif et agis immédiatement
- Utilise des VALEURS PAR DÉFAUT quand les détails manquent:
  * Jours non précisés → tous les jours de la semaine (0-6)
  * Horaires non précisés → trouve un créneau libre de 1-2h le matin ou l'après-midi
  * Durée non précisée → 1h pour les tâches, 2h pour les projets
- Si l'utilisateur dit juste "salut", "ça va?", etc. → réponds naturellement, ne demande pas son emploi du temps"""

    def _get_conversation_history(self, user: User, limit: int = 10) -> list:
        """
        Get recent conversation history.

        Args:
            user: The current user
            limit: Maximum number of messages to retrieve

        Returns:
            list: List of message dictionaries
        """
        messages = ConversationMessage.objects.filter(user=user).order_by('-created_at')[:limit]
        return [
            {'role': msg.role, 'content': msg.content}
            for msg in reversed(messages)
        ]

    def _extract_task_from_message(self, user: User, message: str) -> Optional[dict]:
        """
        Extract task details from natural language message.

        Args:
            user: The current user
            message: The user's message

        Returns:
            dict or None: Task data if extraction successful
        """
        if not self.client:
            return None

        prompt = f"""Analyse ce message et extrait les informations de tâche au format JSON:
Message: "{message}"

Format de réponse (JSON uniquement):
{{
    "title": "Titre court de la tâche",
    "description": "Description détaillée si présente",
    "deadline": "YYYY-MM-DD HH:MM ou null si pas de deadline",
    "estimated_duration_minutes": nombre ou null,
    "task_type": "deep_work|shallow|errand",
    "priority": 1-10,
    "related_course": "Nom du cours si applicable"
}}

Règles:
- deep_work: étude, projet complexe, révision
- shallow: emails, petites tâches rapides
- errand: courses, démarches administratives
- Si pas de deadline explicite, utilise null
- La date d'aujourd'hui est {timezone.now().strftime('%Y-%m-%d')}

Retourne UNIQUEMENT le JSON."""

        try:
            response = self._call_gemini(prompt)
            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response.text)
            if json_match:
                data = json.loads(json_match.group(0))

                # Convert deadline string to datetime
                if data.get('deadline'):
                    try:
                        data['deadline'] = datetime.strptime(
                            data['deadline'], '%Y-%m-%d %H:%M'
                        ).replace(tzinfo=timezone.get_current_timezone())
                    except ValueError:
                        try:
                            data['deadline'] = datetime.strptime(
                                data['deadline'], '%Y-%m-%d'
                            ).replace(tzinfo=timezone.get_current_timezone())
                        except ValueError:
                            data['deadline'] = None

                # Ensure valid task_type
                if data.get('task_type') not in ['deep_work', 'shallow', 'errand']:
                    data['task_type'] = 'shallow'

                # Clamp priority
                data['priority'] = max(1, min(10, data.get('priority', 5)))

                return data

        except Exception as e:
            logger.error(f"Error extracting task: {e}")

        return None

    def _generate_task_confirmation(self, task: Task) -> str:
        """
        Generate a confirmation message for created task.

        Args:
            task: The created task

        Returns:
            str: Confirmation message
        """
        msg = f"J'ai ajouté la tâche: **{task.title}**"

        if task.deadline:
            msg += f"\n📅 Deadline: {task.deadline.strftime('%d/%m/%Y à %H:%M')}"

        if task.estimated_duration_minutes:
            hours = task.estimated_duration_minutes // 60
            minutes = task.estimated_duration_minutes % 60
            if hours > 0:
                msg += f"\n⏱️ Durée estimée: {hours}h{minutes:02d}"
            else:
                msg += f"\n⏱️ Durée estimée: {minutes} minutes"

        msg += f"\n🎯 Priorité: {task.priority}/10"

        return msg

    def _generate_schedule_summary(self, user: User) -> str:
        """
        Generate a summary of the user's schedule.

        Args:
            user: The current user

        Returns:
            str: Schedule summary
        """
        today = timezone.now().date()
        tomorrow = today + timedelta(days=1)

        # Get today's recurring blocks
        today_blocks = RecurringBlock.objects.filter(
            user=user,
            active=True,
            day_of_week=today.weekday()
        ).order_by('start_time')

        # Get pending tasks
        pending_tasks = Task.objects.filter(
            user=user,
            completed=False
        ).order_by('deadline', '-priority')[:5]

        msg = f"📋 **Ton planning**\n\n"

        if today_blocks.exists():
            msg += f"**Aujourd'hui ({today.strftime('%A %d/%m')}):**\n"
            for block in today_blocks:
                msg += f"• {block.start_time.strftime('%H:%M')}-{block.end_time.strftime('%H:%M')}: {block.title}"
                if block.location:
                    msg += f" ({block.location})"
                msg += "\n"
        else:
            msg += f"Pas de cours ou travail prévu aujourd'hui.\n"

        msg += "\n"

        if pending_tasks.exists():
            msg += "**Tâches à faire:**\n"
            for task in pending_tasks:
                deadline_str = ""
                if task.deadline:
                    days_until = (task.deadline.date() - today).days
                    if days_until == 0:
                        deadline_str = " ⚠️ Aujourd'hui!"
                    elif days_until == 1:
                        deadline_str = " (demain)"
                    elif days_until < 0:
                        deadline_str = " 🔴 En retard!"
                    else:
                        deadline_str = f" ({task.deadline.strftime('%d/%m')})"
                msg += f"• {task.title}{deadline_str}\n"
        else:
            msg += "Aucune tâche en attente. 🎉"

        return msg

    def _should_show_onboarding(
        self,
        user: User,
        message: str,
        attachment: Optional[UploadedDocument]
    ) -> bool:
        """
        Determine if we should show onboarding prompts or allow normal conversation.

        Returns True if:
        - User uploaded a document (attachment)
        - It's the user's first message (no conversation history)
        - User is at step 1 or 2 (confirming schedule or setting preferences)
        - User wants to skip or continue onboarding

        Returns False to allow normal Gemini conversation.
        """
        # Always handle document uploads in onboarding flow
        if attachment:
            return True

        profile = user.profile
        step = profile.onboarding_step
        message_lower = message.lower()

        # Check if user is responding to an onboarding quick reply
        onboarding_keywords = ['skip', 'passer', 'upload', 'matin', 'soir', 'après-midi', 'apres-midi', 'oui', 'correct']
        is_onboarding_response = any(kw in message_lower for kw in onboarding_keywords)

        # Steps 1 and 2 need specific responses (confirmation, preferences)
        if step in [1, 2]:
            return True

        # For step 0
        if step == 0:
            # If user is responding to onboarding options
            if is_onboarding_response:
                return True
            # Show onboarding only for first 2 messages (user + assistant)
            message_count = ConversationMessage.objects.filter(user=user).count()
            if message_count <= 2:
                return True

        return False

    def _handle_onboarding(
        self,
        user: User,
        message: str,
        attachment: Optional[UploadedDocument]
    ) -> dict:
        """
        Handle onboarding conversation flow - SIMPLIFIED.

        Flow:
        - Step 0: Welcome + upload OR use Gemini for free conversation
        - Step 1: Confirm extracted data + ask productivity preference
        - Step 2: Show proposal, handle feedback, complete onboarding

        Args:
            user: The current user
            message: The user's message
            attachment: Optional uploaded document

        Returns:
            dict: Response dict with 'text' and optional 'quick_replies'
        """
        profile = user.profile
        step = profile.onboarding_step
        message_lower = message.lower().strip()

        # Count user messages to know if this is their first interaction
        message_count = ConversationMessage.objects.filter(user=user, role='user').count()

        if step == 0:  # Welcome + Upload
            if attachment:
                # Document uploaded - process and move to step 1
                return self._process_onboarding_document(user, attachment, profile)

            elif message_count <= 1:
                # First message - show welcome
                return {
                    'text': "Salut! 👋 Je suis ton assistant de planification.\n\nEnvoie-moi ton emploi du temps (cours ou travail) en photo/PDF, ou décris-moi simplement tes contraintes!",
                    'quick_replies': [
                        {'label': "📷 J'uploade mon emploi du temps", 'value': 'upload'},
                        {'label': "💬 Je préfère décrire", 'value': 'Je vais te décrire mon emploi du temps'},
                        {'label': "⏭️ Passer", 'value': 'skip'},
                    ]
                }

            elif 'skip' in message_lower or 'passer' in message_lower:
                # Skip to proposal with productivity question
                profile.onboarding_step = 2
                profile.save()
                return {
                    'text': "Pas de souci! Avant de te proposer un planning, dis-moi: à quel moment de la journée es-tu le plus productif?",
                    'quick_replies': [
                        {'label': "🌅 Le matin", 'value': 'Je suis plus productif le matin'},
                        {'label': "☀️ L'après-midi", 'value': "Je suis plus productif l'après-midi"},
                        {'label': "🌙 Le soir", 'value': 'Je suis plus productif le soir'},
                    ]
                }

            elif 'upload' in message_lower:
                # User wants to upload
                return {
                    'text': "Envoie-moi ton emploi du temps en photo ou PDF!",
                    'quick_replies': []
                }

            else:
                # User is describing their schedule - use Gemini to understand
                return self._smart_onboarding_conversation(user, message, profile)

        elif step == 1:  # Confirm data + productivity preference
            if attachment:
                # New document at step 1
                return self._process_onboarding_document(user, attachment, profile)

            elif 'oui' in message_lower or 'correct' in message_lower or 'ok' in message_lower or "c'est bon" in message_lower:
                # User confirmed - ask productivity preference then show proposal
                profile.onboarding_step = 2
                profile.save()
                return {
                    'text': "Super! Maintenant, dis-moi: à quel moment de la journée es-tu le plus productif pour travailler/étudier?",
                    'quick_replies': [
                        {'label': "🌅 Le matin", 'value': 'Je suis plus productif le matin'},
                        {'label': "☀️ L'après-midi", 'value': "Je suis plus productif l'après-midi"},
                        {'label': "🌙 Le soir", 'value': 'Je suis plus productif le soir'},
                    ]
                }

            elif 'modifier' in message_lower or 'changer' in message_lower or 'corriger' in message_lower:
                # User wants to modify - let them describe what's wrong
                return {
                    'text': "Qu'est-ce qui n'est pas correct? Décris-moi les modifications à faire.",
                    'quick_replies': [
                        {'label': "📷 Renvoyer le document", 'value': 'upload'},
                    ]
                }

            else:
                # Use Gemini to understand what user wants
                return self._smart_onboarding_conversation(user, message, profile)

        elif step == 2:  # Proposal + finish
            # Check if user is answering productivity preference
            productivity_keywords = {
                'matin': 'morning',
                'morning': 'morning',
                'après-midi': 'afternoon',
                'apres-midi': 'afternoon',
                'afternoon': 'afternoon',
                'soir': 'evening',
                'evening': 'evening',
                'nuit': 'evening',
            }

            for keyword, value in productivity_keywords.items():
                if keyword in message_lower:
                    profile.peak_productivity_time = value
                    profile.save()
                    # Now show the proposal
                    proposal = self._generate_smart_planning_proposal(user)
                    return proposal

            # Check if user is responding to a proposal
            recent_messages = ConversationMessage.objects.filter(user=user).order_by('-created_at')[:3]
            recent_proposal = any('proposition de planning' in msg.content.lower() for msg in recent_messages if msg.role == 'assistant')

            if recent_proposal:
                # Let Gemini handle feedback
                feedback_response = self._handle_proposal_feedback(user, message, list(recent_messages))
                if feedback_response:
                    return feedback_response

            # If no proposal shown yet, show it
            blocks = RecurringBlock.objects.filter(user=user, active=True)
            if blocks.exists():
                proposal = self._generate_smart_planning_proposal(user)
                return proposal

            # No blocks yet - use Gemini to understand
            return self._smart_onboarding_conversation(user, message, profile)

        return {'text': "Je suis là pour t'aider! Dis-moi tes horaires ou contraintes.", 'quick_replies': []}

    def _process_onboarding_document(self, user: User, attachment: UploadedDocument, profile: UserProfile) -> dict:
        """Process document during onboarding and show summary."""
        if attachment.processed and attachment.extracted_data:
            extracted = attachment.extracted_data
            summary = self._build_extraction_summary(extracted)

            profile.onboarding_step = 1
            profile.save()

            return {
                'text': f"J'ai analysé ton document! Voici ce que j'ai trouvé:\n\n{summary}\n\nEst-ce correct?",
                'quick_replies': [
                    {'label': "✅ C'est correct", 'value': 'oui'},
                    {'label': "✏️ Modifier", 'value': 'modifier'},
                ]
            }
        else:
            error_msg = getattr(attachment, 'processing_error', None) or "en cours d'analyse"
            profile.onboarding_step = 1
            profile.save()
            return {
                'text': f"Document reçu! Traitement: {error_msg}. Tu peux continuer ou renvoyer.",
                'quick_replies': [
                    {'label': "Continuer ➡️", 'value': 'oui'},
                    {'label': "📷 Renvoyer", 'value': 'upload'},
                ]
            }

    def _build_extraction_summary(self, extracted: dict) -> str:
        """Build a human-readable summary of extracted data."""
        summary_lines = []

        if 'courses' in extracted and extracted['courses']:
            summary_lines.append(f"📚 **{len(extracted['courses'])} cours:**")
            for course in extracted['courses'][:4]:
                name = course.get('name', 'Cours')
                day = course.get('day', '?')
                time_str = f"{course.get('start_time', '?')}-{course.get('end_time', '?')}"
                summary_lines.append(f"  • {name} ({day} {time_str})")
            if len(extracted['courses']) > 4:
                summary_lines.append(f"  ... +{len(extracted['courses']) - 4} autres")

        if 'shifts' in extracted and extracted['shifts']:
            summary_lines.append(f"\n💼 **{len(extracted['shifts'])} créneaux de travail:**")
            for shift in extracted['shifts'][:4]:
                day = shift.get('day', '?')
                time_str = f"{shift.get('start_time', '?')}-{shift.get('end_time', '?')}"
                role = shift.get('role', '')
                role_str = f" ({role})" if role else ""
                summary_lines.append(f"  • {day} {time_str}{role_str}")
            if len(extracted['shifts']) > 4:
                summary_lines.append(f"  ... +{len(extracted['shifts']) - 4} autres")

        if 'events' in extracted and extracted['events']:
            summary_lines.append(f"\n📅 **{len(extracted['events'])} événements:**")
            for event in extracted['events'][:3]:
                title = event.get('title', 'Événement')
                day = event.get('day', '')
                summary_lines.append(f"  • {title} ({day})" if day else f"  • {title}")
            if len(extracted['events']) > 3:
                summary_lines.append(f"  ... +{len(extracted['events']) - 3} autres")

        return "\n".join(summary_lines) if summary_lines else "Données extraites avec succès!"

    def _smart_onboarding_conversation(self, user: User, message: str, profile: UserProfile) -> dict:
        """Use Gemini to understand user input during onboarding."""
        if not self.client:
            return {
                'text': "Je n'ai pas compris. Tu peux m'envoyer ton emploi du temps ou décrire tes horaires.",
                'quick_replies': [
                    {'label': "📷 Envoyer un document", 'value': 'upload'},
                ]
            }

        # Get existing blocks
        blocks = RecurringBlock.objects.filter(user=user, active=True)
        days_fr = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']

        blocks_info = ""
        if blocks.exists():
            blocks_list = [f"- {b.title}: {days_fr[b.day_of_week]} {b.start_time.strftime('%H:%M')}-{b.end_time.strftime('%H:%M')}" for b in blocks[:10]]
            blocks_info = f"\n\nBLOCS EXISTANTS:\n" + "\n".join(blocks_list)

        prompt = f"""Tu es un assistant de planification en phase d'onboarding. L'utilisateur décrit son emploi du temps.
{blocks_info}

MESSAGE DE L'UTILISATEUR:
"{message}"

ANALYSE et détermine:
1. L'utilisateur décrit-il des HORAIRES DE COURS? (ex: "j'ai cours le lundi de 8h à 12h")
2. L'utilisateur décrit-il des HORAIRES DE TRAVAIL/JOB? (ex: "je travaille le soir", "je bosse le weekend", "je travaille de 23h à 7h")
3. L'utilisateur indique-t-il une PRÉFÉRENCE de productivité? (ex: "je suis du matin")
4. L'utilisateur veut-il AUTRE CHOSE?

RÈGLES IMPORTANTES:
- SHIFTS DE NUIT: Si l'heure de fin est AVANT l'heure de début (ex: 23h à 7h), c'est un shift de nuit
  Ex: "de 23h à 7h" → start_time: "23:00", end_time: "07:00", is_night_shift: true
- Jours: vendredi=4, samedi=5, dimanche=6, lundi=0, mardi=1, mercredi=2, jeudi=3
- TOUJOURS extraire les heures exactes mentionnées par l'utilisateur
- Si l'utilisateur mentionne un lieu (ex: "à l'UQAC"), le capturer dans "location"

Réponds en JSON:
{{
    "understood": true/false,
    "action": "create_blocks|set_preference|need_more_info|other",
    "blocks_to_create": [
        {{
            "title": "Titre",
            "type": "course|work|other",
            "days": [4, 5],
            "start_time": "HH:MM",
            "end_time": "HH:MM",
            "is_night_shift": true/false,
            "location": "Lieu si mentionné"
        }}
    ],
    "preference": {{
        "peak_productivity_time": "morning|afternoon|evening"
    }},
    "response_to_user": "Message naturel à afficher",
    "next_step": "show_proposal|ask_productivity|continue_onboarding"
}}

IMPORTANT: Si l'utilisateur donne des infos de planning, crée les blocs. Sois proactif."""

        try:
            response = self._call_gemini(prompt)
            json_match = re.search(r'\{[\s\S]*\}', response.text)
            if not json_match:
                return self._default_onboarding_response(profile)

            data = json.loads(json_match.group(0))

            # Create blocks if specified
            blocks_created = []
            if data.get('blocks_to_create'):
                from datetime import time as dt_time
                day_mapping = {'lundi': 0, 'mardi': 1, 'mercredi': 2, 'jeudi': 3, 'vendredi': 4, 'samedi': 5, 'dimanche': 6}

                for block_info in data['blocks_to_create']:
                    days = block_info.get('days', [])
                    if isinstance(days, list):
                        for day in days:
                            # Handle both int days and string days
                            if isinstance(day, str):
                                day = day_mapping.get(day.lower(), None)
                            if day is not None and 0 <= day <= 6:
                                try:
                                    # Handle None values explicitly
                                    start_time_str = block_info.get('start_time') or '09:00'
                                    end_time_str = block_info.get('end_time') or '10:00'
                                    start_parts = start_time_str.split(':')
                                    end_parts = end_time_str.split(':')

                                    start_hour = int(start_parts[0])
                                    end_hour = int(end_parts[0])
                                    start_minute = int(start_parts[1]) if len(start_parts) > 1 else 0
                                    end_minute = int(end_parts[1]) if len(end_parts) > 1 else 0

                                    # Auto-detect night shift
                                    is_night_shift = block_info.get('is_night_shift', False)
                                    if end_hour < start_hour or (end_hour == start_hour and end_minute < start_minute):
                                        is_night_shift = True

                                    block = RecurringBlock.objects.create(
                                        user=user,
                                        title=block_info.get('title') or 'Bloc',
                                        block_type=block_info.get('type') or 'other',
                                        day_of_week=day,
                                        start_time=dt_time(start_hour, start_minute),
                                        end_time=dt_time(end_hour, end_minute),
                                        is_night_shift=is_night_shift,
                                        location=block_info.get('location') or None,
                                    )
                                    blocks_created.append(block)
                                except Exception as e:
                                    logger.error(f"Error creating onboarding block: {e}")

            # Update preference if specified
            if data.get('preference', {}).get('peak_productivity_time'):
                pref = data['preference']['peak_productivity_time']
                if pref in ['morning', 'afternoon', 'evening']:
                    profile.peak_productivity_time = pref
                    profile.save()

            # Build response
            response_text = data.get('response_to_user', '')
            if blocks_created:
                # Group blocks by title and time for better display
                day_names = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']
                block_groups = {}
                for b in blocks_created:
                    key = (b.title, b.start_time.strftime('%H:%M'), b.end_time.strftime('%H:%M'))
                    if key not in block_groups:
                        block_groups[key] = []
                    block_groups[key].append(day_names[b.day_of_week])

                block_descriptions = []
                for (title, start, end), days in block_groups.items():
                    block_descriptions.append(f"{title} ({start}-{end}) les {', '.join(days)}")

                response_text += f"\n\n✅ J'ai créé {len(blocks_created)} bloc(s): {'; '.join(block_descriptions)}"

            # Determine next action
            next_step = data.get('next_step', 'continue_onboarding')

            if next_step == 'show_proposal' or (blocks_created and len(blocks_created) >= 2):
                profile.onboarding_step = 2
                profile.save()
                # Add productivity question if not set
                if profile.peak_productivity_time == 'morning':  # default value
                    return {
                        'text': response_text + "\n\nMaintenant, à quel moment es-tu le plus productif?",
                        'quick_replies': [
                            {'label': "🌅 Matin", 'value': 'Je suis plus productif le matin'},
                            {'label': "☀️ Après-midi", 'value': "Je suis plus productif l'après-midi"},
                            {'label': "🌙 Soir", 'value': 'Je suis plus productif le soir'},
                        ]
                    }
                else:
                    proposal = self._generate_smart_planning_proposal(user)
                    return {
                        'text': response_text + "\n\n" + proposal['text'],
                        'quick_replies': proposal.get('quick_replies', [])
                    }

            elif next_step == 'ask_productivity':
                profile.onboarding_step = 2
                profile.save()
                return {
                    'text': response_text + "\n\nÀ quel moment de la journée es-tu le plus productif?",
                    'quick_replies': [
                        {'label': "🌅 Matin", 'value': 'Je suis plus productif le matin'},
                        {'label': "☀️ Après-midi", 'value': "Je suis plus productif l'après-midi"},
                        {'label': "🌙 Soir", 'value': 'Je suis plus productif le soir'},
                    ]
                }

            return {
                'text': response_text if response_text else "Compris! Tu peux continuer à me décrire ton emploi du temps ou m'envoyer un document.",
                'quick_replies': [
                    {'label': "📷 Envoyer un document", 'value': 'upload'},
                    {'label': "✅ C'est tout", 'value': 'oui'},
                ]
            }

        except Exception as e:
            logger.error(f"Smart onboarding error: {e}")
            return self._default_onboarding_response(profile)

    def _default_onboarding_response(self, profile: UserProfile) -> dict:
        """Default response when Gemini fails during onboarding."""
        return {
            'text': "Je n'ai pas tout compris. Tu peux m'envoyer ton emploi du temps en photo/PDF, ou me décrire tes horaires de cours/travail.",
            'quick_replies': [
                {'label': "📷 Envoyer un document", 'value': 'upload'},
                {'label': "⏭️ Passer", 'value': 'skip'},
            ]
        }

    def _handle_proposal_feedback(self, user: User, message: str, recent_messages: list) -> Optional[dict]:
        """
        Use Gemini to intelligently understand and handle user feedback on a planning proposal.

        Args:
            user: The current user
            message: The user's feedback message
            recent_messages: Recent conversation messages for context

        Returns:
            dict with 'text' and 'quick_replies', or None if couldn't process
        """
        try:
            profile = user.profile
        except Exception as e:
            logger.error(f"Error accessing user profile in _handle_proposal_feedback: {e}")
            return None

        message_lower = message.lower().strip()
        logger.debug(f"_handle_proposal_feedback called with message: '{message_lower}'")

        # DIRECT DETECTION: Handle obvious acceptance phrases without Gemini
        # This ensures button clicks always work reliably
        acceptance_phrases = [
            'oui, crée ce planning',
            'crée ce planning',
            'crée le planning',
            'créer le planning',
            'créer ce planning',
            'valide le planning',
            'valider le planning',
            "c'est bon",
            'c\'est parfait',
            'ok crée',
            'oui crée',
            'go pour le planning',
            'accepte le planning',
            'accepter le planning',
        ]

        if any(phrase in message_lower for phrase in acceptance_phrases):
            logger.info(f"Direct acceptance detected: '{message}'")
            try:
                created_blocks = self._create_proposed_blocks(user)
                blocks_count = len(created_blocks) if created_blocks else 0
                logger.info(f"Created {blocks_count} proposed blocks for user {user.id}")
            except Exception as e:
                logger.error(f"Error creating proposed blocks: {e}")
                created_blocks = []
                blocks_count = 0

            # Mark onboarding as completed regardless of block creation
            profile.onboarding_completed = True
            profile.onboarding_step = 3
            profile.save()
            logger.info(f"Onboarding completed for user {user.id}")

            if blocks_count > 0:
                response_text = f"✅ Parfait! J'ai créé {blocks_count} blocs dans ton planning.\n\nTon planning est prêt!"
            else:
                response_text = "✅ Parfait! Ton planning est configuré."

            return {
                'text': response_text,
                'quick_replies': [
                    {'label': "📋 Voir mon planning", 'value': 'Montre-moi mon planning'},
                    {'label': "➕ Ajouter une tâche", 'value': "J'ai une tâche à ajouter"},
                ]
            }

        if not self.client:
            return None

        # Get current blocks for context
        blocks = RecurringBlock.objects.filter(user=user, active=True).order_by('day_of_week', 'start_time')
        days_fr = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']

        blocks_summary = []
        for block in blocks[:20]:  # Limit for prompt size
            day_name = days_fr[block.day_of_week]
            blocks_summary.append(f"- {block.title} ({block.block_type}): {day_name} {block.start_time.strftime('%H:%M')}-{block.end_time.strftime('%H:%M')}")

        blocks_text = "\n".join(blocks_summary) if blocks_summary else "Aucun bloc"

        # Get the last proposal shown
        last_proposal = ""
        for msg in recent_messages:
            if msg.role == 'assistant' and 'proposition de planning' in msg.content.lower():
                last_proposal = msg.content
                break

        prompt = f"""Tu es un assistant de planification intelligent. L'utilisateur a reçu une proposition de planning et donne un feedback.

PLANNING ACTUEL DE L'UTILISATEUR:
{blocks_text}

PRÉFÉRENCES ACTUELLES:
- Pic de productivité: {profile.get_peak_productivity_time_display()}
- Sommeil minimum: {profile.min_sleep_hours}h

DERNIÈRE PROPOSITION MONTRÉE:
{last_proposal[:1000] if last_proposal else "Non disponible"}

MESSAGE DE L'UTILISATEUR:
"{message}"

ANALYSE ce message et détermine:
1. L'utilisateur veut-il ACCEPTER la proposition? (mots comme "oui", "ok", "crée", "parfait", "c'est bon", "valide")
2. L'utilisateur indique-t-il une CONTRAINTE DE TRAVAIL/JOB? (ex: "je travaille le soir" = il a un job le soir, donc NON DISPONIBLE le soir)
3. L'utilisateur indique-t-il une PRÉFÉRENCE DE PRODUCTIVITÉ? (ex: "je suis plus productif le soir", "je préfère étudier le matin")
4. L'utilisateur veut-il MODIFIER quelque chose de spécifique? (heures de coucher, réveil, repas, etc.)
5. L'utilisateur demande-t-il un NOMBRE D'HEURES D'ÉTUDE spécifique? (ex: "je veux 15-20h de projet", "je veux plus d'heures pour réviser")

IMPORTANT:
- "Je travaille le soir" = JOB le soir = l'utilisateur n'est PAS disponible le soir
- "Je préfère le soir" = PRÉFÉRENCE = l'utilisateur est plus productif le soir
- Si l'utilisateur mentionne des heures spécifiques (coucher, réveil, repas), utilise "modify_specific"
- SHIFTS DE NUIT: Si l'heure de fin est AVANT l'heure de début (ex: 23h à 7h), c'est un shift de nuit
  Ex: "de 23h à 7h" → start_time: "23:00", end_time: "07:00", is_night_shift: true
- TOUJOURS extraire les heures exactes mentionnées par l'utilisateur
- Si l'utilisateur mentionne un lieu de travail (ex: "à l'UQAC"), le capturer dans "location"
- Si l'utilisateur demande des heures d'étude/projet (ex: "15-20h de travail sur mes projets"), utilise "request_study_hours"

Réponds en JSON:
{{
    "understood": true/false,
    "action": "accept_proposal|add_work_block|set_productivity_preference|ask_clarification|modify_specific|request_study_hours",
    "details": {{
        "productivity_time": "morning|afternoon|evening" (si applicable),
        "work_block": {{
            "days": ["vendredi", "samedi", ...],
            "start_time": "HH:MM",
            "end_time": "HH:MM",
            "title": "Travail (nuit)" ou "Travail" ou "Travail - [employeur]" si mentionné,
            "is_night_shift": true/false,
            "location": "Lieu de travail si mentionné"
        }} (si applicable),
        "schedule_overrides": {{
            "bedtime": "HH:MM" (heure de coucher si mentionnée),
            "wake_time": "HH:MM" (heure de réveil si mentionnée),
            "breakfast_time": "HH:MM" (petit-déjeuner si mentionné),
            "lunch_time": "HH:MM" (déjeuner si mentionné)
        }} (si applicable),
        "study_hours_request": {{
            "min_hours": 15,
            "max_hours": 20,
            "type": "project|revision|both"
        }} (si applicable)
    }},
    "response_to_user": "Message à afficher à l'utilisateur",
    "needs_more_info": true/false,
    "clarification_question": "Question si besoin de plus d'infos"
}}

Retourne UNIQUEMENT le JSON."""

        try:
            response = self._call_gemini(prompt)
            json_match = re.search(r'\{[\s\S]*\}', response.text)
            if not json_match:
                return None

            data = json.loads(json_match.group(0))

            if not data.get('understood', False):
                return None

            action = data.get('action', '')
            details = data.get('details', {})

            # Execute the action
            if action == 'accept_proposal':
                created_blocks = self._create_proposed_blocks(user)
                profile.onboarding_completed = True
                profile.onboarding_step = 3
                profile.save()
                return {
                    'text': f"✅ Parfait! J'ai créé {len(created_blocks)} blocs dans ton planning.\n\nTon planning est prêt!",
                    'quick_replies': [
                        {'label': "📋 Voir mon planning", 'value': 'Montre-moi mon planning'},
                        {'label': "➕ Ajouter autre chose", 'value': "J'ai autre chose à ajouter"},
                    ]
                }

            elif action == 'add_work_block':
                # User has a job - add work blocks
                work_info = details.get('work_block', {})
                days = work_info.get('days', [])
                # Handle None values explicitly (not just missing keys)
                start_time = work_info.get('start_time') or '18:00'
                end_time = work_info.get('end_time') or '23:00'
                title = work_info.get('title') or 'Travail'
                is_night_shift = work_info.get('is_night_shift', False)
                location = work_info.get('location', '')

                from datetime import time as dt_time
                day_mapping = {'lundi': 0, 'mardi': 1, 'mercredi': 2, 'jeudi': 3, 'vendredi': 4, 'samedi': 5, 'dimanche': 6}

                created_count = 0
                for day_name in days:
                    day_num = day_mapping.get(day_name.lower())
                    if day_num is not None:
                        try:
                            start_parts = start_time.split(':')
                            end_parts = end_time.split(':')
                            start_hour = int(start_parts[0])
                            end_hour = int(end_parts[0])

                            # Detect night shift (end time is "earlier" than start time)
                            if end_hour < start_hour or (end_hour == start_hour and int(end_parts[1]) < int(start_parts[0])):
                                is_night_shift = True

                            RecurringBlock.objects.create(
                                user=user,
                                title=title,
                                block_type='work',
                                day_of_week=day_num,
                                start_time=dt_time(int(start_parts[0]), int(start_parts[1]) if len(start_parts) > 1 else 0),
                                end_time=dt_time(int(end_parts[0]), int(end_parts[1]) if len(end_parts) > 1 else 0),
                                is_night_shift=is_night_shift,
                                location=location or '',
                            )
                            created_count += 1
                        except Exception as e:
                            logger.error(f"Error creating work block: {e}")

                # Regenerate proposal with new constraints
                proposal = self._generate_smart_planning_proposal(user)
                return {
                    'text': f"J'ai noté que tu travailles {', '.join(days)} de {start_time} à {end_time}. J'ai ajouté {created_count} blocs de travail.\n\n{proposal['text']}",
                    'quick_replies': proposal.get('quick_replies', [])
                }

            elif action == 'set_productivity_preference':
                prod_time = details.get('productivity_time', 'morning')
                if prod_time in ['morning', 'afternoon', 'evening']:
                    profile.peak_productivity_time = prod_time
                    profile.save()

                time_labels = {'morning': 'le matin', 'afternoon': "l'après-midi", 'evening': 'le soir'}
                proposal = self._generate_smart_planning_proposal(user)
                return {
                    'text': f"J'ai noté que tu es plus productif {time_labels.get(prod_time, prod_time)}!\n\n{proposal['text']}",
                    'quick_replies': proposal.get('quick_replies', [])
                }

            elif action == 'modify_specific':
                # User wants to modify specific times (bedtime, wake, meals)
                overrides = details.get('schedule_overrides', {})
                changes_made = []

                if overrides.get('bedtime'):
                    changes_made.append(f"coucher à {overrides['bedtime']}")
                if overrides.get('wake_time'):
                    changes_made.append(f"réveil à {overrides['wake_time']}")
                if overrides.get('breakfast_time'):
                    changes_made.append(f"petit-déjeuner à {overrides['breakfast_time']}")
                if overrides.get('lunch_time'):
                    changes_made.append(f"déjeuner à {overrides['lunch_time']}")

                # Generate proposal with overrides
                proposal = self._generate_smart_planning_proposal(user, overrides=overrides)

                if changes_made:
                    changes_text = ", ".join(changes_made)
                    return {
                        'text': f"J'ai pris en compte tes préférences ({changes_text}).\n\n{proposal['text']}",
                        'quick_replies': proposal.get('quick_replies', [])
                    }
                else:
                    return {
                        'text': proposal['text'],
                        'quick_replies': proposal.get('quick_replies', [])
                    }

            elif action == 'request_study_hours':
                # User wants specific study/project hours
                study_request = details.get('study_hours_request', {})
                min_hours = study_request.get('min_hours', 15)
                max_hours = study_request.get('max_hours', 20)
                study_type = study_request.get('type', 'both')

                # Generate proposal with study hours focus
                proposal = self._generate_smart_planning_proposal(user)

                type_labels = {
                    'project': 'projets',
                    'revision': 'révisions/devoirs',
                    'both': 'projets et révisions'
                }
                type_text = type_labels.get(study_type, 'études')

                response_text = f"J'ai noté que tu veux {min_hours}-{max_hours}h de {type_text} par semaine.\n\n"
                response_text += "Voici les créneaux disponibles dans ton planning:\n\n"
                response_text += proposal['text']

                return {
                    'text': response_text,
                    'quick_replies': proposal.get('quick_replies', [])
                }

            elif action == 'ask_clarification' or data.get('needs_more_info'):
                question = data.get('clarification_question', "Peux-tu préciser ce que tu veux changer?")
                return {
                    'text': question,
                    'quick_replies': [
                        {'label': "💼 J'ai un job", 'value': "J'ai un travail/job"},
                        {'label': "📚 Préférence d'étude", 'value': "Je préfère étudier à un certain moment"},
                        {'label': "✅ C'est bon", 'value': "Crée ce planning"},
                    ]
                }

            else:
                # Default: show the response from Gemini
                response_text = data.get('response_to_user', '')
                if response_text:
                    return {
                        'text': response_text,
                        'quick_replies': [
                            {'label': "✅ Crée le planning", 'value': 'Crée ce planning'},
                            {'label': "🔧 Ajuster", 'value': 'Je veux modifier quelque chose'},
                        ]
                    }

        except Exception as e:
            logger.error(f"Error handling proposal feedback: {e}")

        return None

    def _add_extracted_to_planning(self, user: User) -> dict:
        """
        Add extracted data from the most recent document to the planning.

        Args:
            user: The current user

        Returns:
            dict: Response with text and quick_replies
        """
        from services.document_processor import DocumentProcessor

        # Get the most recent processed document
        recent_doc = UploadedDocument.objects.filter(
            user=user,
            processed=True
        ).order_by('-uploaded_at').first()

        if not recent_doc or not recent_doc.extracted_data:
            return {
                'text': "Je n'ai pas trouvé de document récent avec des données à ajouter. Envoie-moi ton emploi du temps!",
                'quick_replies': [
                    {'label': "📚 Envoyer mon emploi du temps", 'value': 'upload'},
                ]
            }

        # Check if blocks already exist for this document
        existing_blocks = RecurringBlock.objects.filter(source_document=recent_doc).count()
        if existing_blocks > 0:
            return {
                'text': f"Les données de ce document ont déjà été ajoutées ({existing_blocks} blocs créés). Veux-tu voir ton planning?",
                'quick_replies': [
                    {'label': "📋 Voir mon planning", 'value': 'Montre-moi mon planning'},
                    {'label': "📊 Voir la proposition", 'value': 'Montre-moi la proposition de planning'},
                ]
            }

        # Create blocks from extracted data
        processor = DocumentProcessor()
        created_blocks = processor._create_recurring_blocks(recent_doc, recent_doc.extracted_data)

        if created_blocks:
            block_summary = []
            courses = [b for b in created_blocks if b.block_type == 'course']
            shifts = [b for b in created_blocks if b.block_type == 'work']
            others = [b for b in created_blocks if b.block_type == 'other']

            if courses:
                block_summary.append(f"📚 {len(courses)} cours")
            if shifts:
                block_summary.append(f"💼 {len(shifts)} créneaux de travail")
            if others:
                block_summary.append(f"📅 {len(others)} événements")

            summary = ", ".join(block_summary)

            return {
                'text': f"J'ai ajouté {len(created_blocks)} blocs à ton planning:\n{summary}\n\nJe peux maintenant te proposer un planning optimisé!",
                'quick_replies': [
                    {'label': "📊 Voir la proposition", 'value': 'Montre-moi la proposition de planning'},
                    {'label': "📋 Voir mon planning", 'value': 'Montre-moi mon planning'},
                ]
            }
        else:
            return {
                'text': "Je n'ai pas pu créer de blocs à partir des données extraites. Les jours ou heures ne sont peut-être pas au bon format.",
                'quick_replies': [
                    {'label': "📝 Décrire mon emploi du temps", 'value': "Je vais te décrire mon emploi du temps"},
                ]
            }

    def _handle_document_upload(self, user: User, attachment: UploadedDocument) -> dict:
        """
        Handle document upload at any time (after onboarding completed).

        Args:
            user: The current user
            attachment: The uploaded document

        Returns:
            dict: Response with text and quick_replies
        """
        if not attachment.processed:
            # Document processing failed or still in progress
            error = attachment.processing_error if hasattr(attachment, 'processing_error') else None
            if error:
                return {
                    'text': f"Désolé, je n'ai pas pu analyser ce document: {error}\n\nEssaie avec un autre fichier ou décris-moi ton emploi du temps.",
                    'quick_replies': [
                        {'label': "📝 Décrire mon emploi du temps", 'value': "Je vais te décrire mon emploi du temps"},
                    ]
                }
            else:
                return {
                    'text': "Le document est en cours de traitement. Réessaie dans quelques secondes.",
                    'quick_replies': []
                }

        # Document processed successfully
        extracted = attachment.extracted_data or {}

        # Build summary of extracted data
        summary_lines = []
        total_items = 0

        if 'courses' in extracted and extracted['courses']:
            count = len(extracted['courses'])
            total_items += count
            summary_lines.append(f"📚 **{count} cours détectés:**")
            for course in extracted['courses'][:5]:
                name = course.get('name', 'Cours')
                day = course.get('day', '?')
                time_str = f"{course.get('start_time', '?')} - {course.get('end_time', '?')}"
                summary_lines.append(f"  • {name} ({day} {time_str})")
            if count > 5:
                summary_lines.append(f"  ... et {count - 5} autres")

        if 'shifts' in extracted and extracted['shifts']:
            count = len(extracted['shifts'])
            total_items += count
            summary_lines.append(f"\n💼 **{count} créneaux de travail détectés:**")
            for shift in extracted['shifts'][:5]:
                day = shift.get('day', '?')
                time_str = f"{shift.get('start_time', '?')} - {shift.get('end_time', '?')}"
                night = " (nuit)" if shift.get('is_night_shift') else ""
                summary_lines.append(f"  • {day} {time_str}{night}")
            if count > 5:
                summary_lines.append(f"  ... et {count - 5} autres")

        if 'events' in extracted and extracted['events']:
            count = len(extracted['events'])
            total_items += count
            summary_lines.append(f"\n📅 **{count} événements détectés:**")
            for event in extracted['events'][:5]:
                title = event.get('title', 'Événement')
                day = event.get('day', '')
                time_str = f"{event.get('start_time', '')} - {event.get('end_time', '')}" if event.get('start_time') else ''
                summary_lines.append(f"  • {title} {f'({day} {time_str})' if day or time_str else ''}")
            if count > 5:
                summary_lines.append(f"  ... et {count - 5} autres")

        if summary_lines:
            summary = "\n".join(summary_lines)

            # Check if blocks were created
            blocks_created = RecurringBlock.objects.filter(source_document=attachment).count()

            if blocks_created > 0:
                text = f"Document analysé! {total_items} éléments extraits et {blocks_created} blocs créés.\n\n{summary}\n\nJe peux maintenant te proposer un planning optimisé!"
                quick_replies = [
                    {'label': "📋 Voir la proposition", 'value': 'Montre-moi la proposition de planning'},
                    {'label': "📅 Voir mon planning", 'value': 'Montre-moi mon planning'},
                ]
            else:
                text = f"Document analysé! Voici ce que j'ai trouvé:\n\n{summary}\n\nVeux-tu que j'ajoute ces éléments à ton planning?"
                quick_replies = [
                    {'label': "✅ Ajouter au planning", 'value': 'Ajoute ces données à mon planning'},
                    {'label': "📋 Voir le planning actuel", 'value': 'Montre-moi mon planning'},
                ]

            return {
                'text': text,
                'quick_replies': quick_replies
            }
        else:
            return {
                'text': "J'ai analysé le document mais je n'ai pas trouvé d'informations d'emploi du temps. Peux-tu m'envoyer un planning de cours ou de travail?",
                'quick_replies': [
                    {'label': "📝 Décrire mon emploi du temps", 'value': "Je vais te décrire mon emploi du temps"},
                ]
            }

    def _create_proposed_blocks(self, user: User) -> list:
        """
        Create the proposed blocks (sleep, meals) in the database.

        Returns:
            list: List of created block info dicts
        """
        from datetime import time as dt_time

        profile = user.profile
        created = []
        day_names = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

        # Get existing blocks to find earliest start
        blocks = RecurringBlock.objects.filter(user=user, active=True)

        # Build schedule by day for free slot calculation
        schedule_by_day = {i: [] for i in range(7)}
        earliest_weekday_start = None

        for block in blocks:
            schedule_by_day[block.day_of_week].append({
                'start': block.start_time,
                'end': block.end_time,
            })
            if block.day_of_week < 5:
                if earliest_weekday_start is None or block.start_time < earliest_weekday_start:
                    earliest_weekday_start = block.start_time

        # Calculate sleep times
        min_sleep = profile.min_sleep_hours or 7
        transport_time = profile.transport_time_minutes or 30

        if earliest_weekday_start:
            wake_minutes = earliest_weekday_start.hour * 60 + earliest_weekday_start.minute
            wake_minutes -= transport_time + 60
            wake_hour = max(5, wake_minutes // 60)
            wake_minute = wake_minutes % 60 if wake_minutes > 0 else 0

            bed_hour = (wake_hour - min_sleep) % 24
            if bed_hour < 12:
                bed_hour = 24 - min_sleep + wake_hour
                bed_hour = bed_hour % 24
        else:
            wake_hour, wake_minute = 7, 0
            bed_hour = 23

        wake_time = dt_time(wake_hour, wake_minute)
        bed_time = dt_time(bed_hour % 24, 0)

        # Create sleep blocks for each day
        for day in range(7):
            existing = RecurringBlock.objects.filter(
                user=user, day_of_week=day, block_type='sleep'
            ).exists()

            if not existing:
                RecurringBlock.objects.create(
                    user=user,
                    title="Sommeil",
                    block_type='sleep',
                    day_of_week=day,
                    start_time=bed_time,
                    end_time=wake_time,
                )
                created.append({'title': f'Sommeil ({day_names[day]})'})

        # Helper to find free slots in a day
        def find_free_slots(day_blocks, day_start=dt_time(7, 0), day_end=dt_time(23, 0)):
            if not day_blocks:
                return [(day_start, day_end)]

            sorted_blocks = sorted(day_blocks, key=lambda x: x['start'])
            free_slots = []
            current_time = day_start

            for block in sorted_blocks:
                if block['start'] > current_time:
                    free_slots.append((current_time, block['start']))
                if block['end'] > current_time:
                    current_time = block['end']

            if current_time < day_end:
                free_slots.append((current_time, day_end))

            return free_slots

        # Create meal blocks for weekdays (Monday-Friday)
        for day in range(5):
            # Check if meal blocks already exist
            existing_meals = RecurringBlock.objects.filter(
                user=user, day_of_week=day, block_type='other',
                title__in=['Petit-déjeuner', 'Déjeuner', 'Dîner']
            ).values_list('title', flat=True)
            existing_meals = list(existing_meals)

            free_slots = find_free_slots(schedule_by_day[day])

            for slot_start, slot_end in free_slots:
                slot_start_minutes = slot_start.hour * 60 + slot_start.minute
                slot_end_minutes = slot_end.hour * 60 + slot_end.minute
                duration = slot_end_minutes - slot_start_minutes

                # Breakfast (6h-9h, 30min duration)
                if 'Petit-déjeuner' not in existing_meals and 360 <= slot_start_minutes < 540 and duration >= 30:
                    RecurringBlock.objects.create(
                        user=user,
                        title='Petit-déjeuner',
                        block_type='other',
                        day_of_week=day,
                        start_time=slot_start,
                        end_time=dt_time(slot_start.hour, slot_start.minute + 30) if slot_start.minute + 30 < 60 else dt_time(slot_start.hour + 1, (slot_start.minute + 30) % 60),
                    )
                    created.append({'title': f'Petit-déjeuner ({day_names[day]})'})
                    existing_meals.append('Petit-déjeuner')

                # Lunch (11h30-14h, 45min duration)
                elif 'Déjeuner' not in existing_meals and 690 <= slot_start_minutes < 840 and duration >= 45:
                    end_minute = slot_start.minute + 45
                    end_hour = slot_start.hour + (end_minute // 60)
                    end_minute = end_minute % 60
                    RecurringBlock.objects.create(
                        user=user,
                        title='Déjeuner',
                        block_type='other',
                        day_of_week=day,
                        start_time=slot_start,
                        end_time=dt_time(end_hour, end_minute),
                    )
                    created.append({'title': f'Déjeuner ({day_names[day]})'})
                    existing_meals.append('Déjeuner')

                # Dinner (18h-21h, 45min duration)
                elif 'Dîner' not in existing_meals and 1080 <= slot_start_minutes < 1260 and duration >= 45:
                    end_minute = slot_start.minute + 45
                    end_hour = slot_start.hour + (end_minute // 60)
                    end_minute = end_minute % 60
                    RecurringBlock.objects.create(
                        user=user,
                        title='Dîner',
                        block_type='other',
                        day_of_week=day,
                        start_time=slot_start,
                        end_time=dt_time(end_hour, end_minute),
                    )
                    created.append({'title': f'Dîner ({day_names[day]})'})
                    existing_meals.append('Dîner')

        return created

    def _handle_smart_planning(
        self,
        user: User,
        message: str,
        system_prompt: str,
        history: list
    ) -> dict:
        """
        Handle complex planning requests using Gemini AI.

        This handles requests like:
        - "Ajoute mes heures de sommeil"
        - "Crée des blocs pour mes projets"
        - "Organise mon temps de travail personnel"

        Args:
            user: The current user
            message: The user's message
            system_prompt: The system context with schedule
            history: Conversation history

        Returns:
            dict: Response with 'text', 'quick_replies', and 'interactive_inputs'
        """
        if not self.client:
            return {
                'text': "Le service de planification n'est pas disponible.",
                'quick_replies': [],
                'interactive_inputs': []
            }

        # Define tools that Gemini can use (like function calling)
        tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="ask_time_range",
                        description="Demander une plage horaire à l'utilisateur (ex: heures de sommeil, horaires de travail)",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "id": types.Schema(type="STRING", description="Identifiant unique"),
                                "label": types.Schema(type="STRING", description="Label court"),
                                "question": types.Schema(type="STRING", description="Question à poser"),
                                "default_start": types.Schema(type="STRING", description="Heure de début par défaut (HH:MM)"),
                                "default_end": types.Schema(type="STRING", description="Heure de fin par défaut (HH:MM)"),
                            },
                            required=["id", "label", "question"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="ask_days",
                        description="Demander à l'utilisateur de sélectionner des jours de la semaine",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "id": types.Schema(type="STRING", description="Identifiant unique"),
                                "label": types.Schema(type="STRING", description="Label court"),
                                "question": types.Schema(type="STRING", description="Question à poser"),
                            },
                            required=["id", "label", "question"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="ask_number",
                        description="Demander un nombre à l'utilisateur (heures, durée en minutes, etc.)",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "id": types.Schema(type="STRING", description="Identifiant unique"),
                                "label": types.Schema(type="STRING", description="Label court"),
                                "question": types.Schema(type="STRING", description="Question à poser"),
                                "min_value": types.Schema(type="INTEGER", description="Valeur minimum"),
                                "max_value": types.Schema(type="INTEGER", description="Valeur maximum"),
                                "default_value": types.Schema(type="INTEGER", description="Valeur par défaut"),
                            },
                            required=["id", "label", "question"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="ask_time",
                        description="Demander une heure simple à l'utilisateur",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "id": types.Schema(type="STRING", description="Identifiant unique"),
                                "label": types.Schema(type="STRING", description="Label court"),
                                "question": types.Schema(type="STRING", description="Question à poser"),
                                "default_time": types.Schema(type="STRING", description="Heure par défaut (HH:MM)"),
                            },
                            required=["id", "label", "question"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="ask_choice",
                        description="Demander à l'utilisateur de choisir parmi des options",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "id": types.Schema(type="STRING", description="Identifiant unique"),
                                "label": types.Schema(type="STRING", description="Label court"),
                                "question": types.Schema(type="STRING", description="Question à poser"),
                                "options": types.Schema(
                                    type="ARRAY",
                                    items=types.Schema(type="STRING"),
                                    description="Liste des options (ex: ['Blocs longs (2-4h)', 'Blocs courts (30min-1h)'])"
                                ),
                            },
                            required=["id", "label", "question", "options"]
                        )
                    ),
                ]
            )
        ]

        # Build the prompt with context
        prompt = f"""{system_prompt}

Historique récent:
"""
        for msg in history[-3:]:
            role = "Utilisateur" if msg['role'] == 'user' else "Assistant"
            prompt += f"{role}: {msg['content']}\n"

        prompt += f"""
Utilisateur: {message}

Tu as accès à des outils pour collecter des informations de l'utilisateur via des formulaires interactifs.
Utilise-les quand tu as besoin d'informations spécifiques (horaires, jours, durées, préférences).
Tu peux appeler PLUSIEURS outils en même temps si nécessaire.
Réponds en français."""

        try:
            response = self._call_gemini(
                prompt,
                config=types.GenerateContentConfig(tools=tools)
            )

            # Check if Gemini called any tools
            interactive_inputs = []
            response_text = ""

            for part in response.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    args = dict(fc.args)

                    if fc.name == "ask_time_range":
                        interactive_inputs.append({
                            'id': args.get('id', 'time_range'),
                            'type': 'time_range',
                            'label': args.get('label', 'Horaires'),
                            'question': args.get('question', 'Quels horaires ?'),
                            'default': {
                                'start': args.get('default_start', '09:00'),
                                'end': args.get('default_end', '18:00')
                            }
                        })

                    elif fc.name == "ask_days":
                        interactive_inputs.append({
                            'id': args.get('id', 'days'),
                            'type': 'checkbox',
                            'label': args.get('label', 'Jours'),
                            'question': args.get('question', 'Quels jours ?'),
                            'options': [
                                {'value': '0', 'label': 'Lundi'},
                                {'value': '1', 'label': 'Mardi'},
                                {'value': '2', 'label': 'Mercredi'},
                                {'value': '3', 'label': 'Jeudi'},
                                {'value': '4', 'label': 'Vendredi'},
                                {'value': '5', 'label': 'Samedi'},
                                {'value': '6', 'label': 'Dimanche'},
                            ]
                        })

                    elif fc.name == "ask_number":
                        interactive_inputs.append({
                            'id': args.get('id', 'number'),
                            'type': 'number',
                            'label': args.get('label', 'Nombre'),
                            'question': args.get('question', 'Combien ?'),
                            'min': args.get('min_value', 1),
                            'max': args.get('max_value', 100),
                            'default': args.get('default_value', 10)
                        })

                    elif fc.name == "ask_time":
                        interactive_inputs.append({
                            'id': args.get('id', 'time'),
                            'type': 'time',
                            'label': args.get('label', 'Heure'),
                            'question': args.get('question', 'À quelle heure ?'),
                            'default': args.get('default_time', '18:00')
                        })

                    elif fc.name == "ask_choice":
                        options = args.get('options', ['Option 1', 'Option 2'])
                        interactive_inputs.append({
                            'id': args.get('id', 'choice'),
                            'type': 'select',
                            'label': args.get('label', 'Choix'),
                            'question': args.get('question', 'Que préfères-tu ?'),
                            'options': [{'value': str(i), 'label': opt} for i, opt in enumerate(options)]
                        })

                elif hasattr(part, 'text') and part.text:
                    response_text += part.text

            if interactive_inputs:
                return {
                    'text': response_text.strip() if response_text else "J'ai besoin de quelques informations :",
                    'quick_replies': [],
                    'interactive_inputs': interactive_inputs
                }
            else:
                return {
                    'text': response_text.strip() if response_text else "C'est noté! Tu veux ajouter autre chose?",
                    'quick_replies': [
                        {'label': "✅ Ça me va!", 'value': "oui c'est bon, crée les blocs"},
                        {'label': "📋 Voir mon planning", 'value': 'Montre-moi mon planning'},
                    ],
                    'interactive_inputs': []
                }

        except Exception as e:
            logger.error(f"Smart planning error: {e}")
            return {
                'text': "J'ai eu un souci pour analyser ta demande. Peux-tu reformuler?",
                'quick_replies': [],
                'interactive_inputs': []
            }

    def _call_gemini_conversation(self, system_prompt: str, history: list, message: str) -> str:
        """
        Call Gemini API for general conversation (builds context from history).

        Args:
            system_prompt: The system context
            history: Conversation history
            message: Current user message

        Returns:
            str: Generated response
        """
        if not self.client:
            return "Désolé, le service de conversation n'est pas disponible pour le moment."

        # Build conversation context
        context = f"{system_prompt}\n\nHistorique de conversation:\n"
        for msg in history[-5:]:  # Last 5 messages for context
            role = "Utilisateur" if msg['role'] == 'user' else "Assistant"
            context += f"{role}: {msg['content']}\n"

        context += f"\nUtilisateur: {message}\nAssistant:"

        try:
            response = self._call_gemini(context)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return "Désolé, j'ai eu un problème technique. Réessaie dans un moment."

    def _smart_conversation(
        self,
        user: User,
        message: str,
        system_prompt: str,
        history: list
    ) -> dict:
        """
        Use Gemini to intelligently handle any user message.

        Gemini decides what action to take based on the message:
        - Add tasks
        - Create recurring blocks
        - Show schedule
        - Answer questions
        - General conversation

        Args:
            user: The current user
            message: The user's message
            system_prompt: System context with schedule info
            history: Conversation history

        Returns:
            dict: Response with 'text', 'quick_replies', 'tasks_created'
        """
        if not self.client:
            return {
                'text': "Le service de conversation n'est pas disponible.",
                'quick_replies': [],
                'tasks_created': []
            }

        # Define functions Gemini can call
        tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="create_recurring_block",
                        description="Créer un bloc récurrent (cours, travail, sport, projet, sommeil, repas, etc.)",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "title": types.Schema(type="STRING", description="Titre du bloc"),
                                "block_type": types.Schema(type="STRING", description="Type: course|work|sleep|meal|sport|project|other"),
                                "days": types.Schema(
                                    type="ARRAY",
                                    items=types.Schema(type="INTEGER"),
                                    description="Jours de la semaine (0=Lundi, 6=Dimanche)"
                                ),
                                "start_time": types.Schema(type="STRING", description="Heure de début (HH:MM)"),
                                "end_time": types.Schema(type="STRING", description="Heure de fin (HH:MM)"),
                                "location": types.Schema(type="STRING", description="Lieu (optionnel)"),
                            },
                            required=["title", "block_type", "days", "start_time", "end_time"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="create_task",
                        description="Créer une tâche avec deadline optionnelle",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "title": types.Schema(type="STRING", description="Titre de la tâche"),
                                "description": types.Schema(type="STRING", description="Description"),
                                "deadline": types.Schema(type="STRING", description="Deadline (YYYY-MM-DD HH:MM ou null)"),
                                "priority": types.Schema(type="INTEGER", description="Priorité 1-10"),
                                "task_type": types.Schema(type="STRING", description="Type: deep_work|shallow|errand"),
                            },
                            required=["title"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="update_preference",
                        description="Mettre à jour une préférence utilisateur APRÈS confirmation. Utilise cette fonction UNIQUEMENT après avoir demandé confirmation avec ask_preference_confirmation.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "preference": types.Schema(type="STRING", description="Nom: peak_productivity_time|min_sleep_hours|max_deep_work_hours_per_day|transport_time_minutes"),
                                "value": types.Schema(type="STRING", description="Nouvelle valeur"),
                            },
                            required=["preference", "value"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="ask_preference_confirmation",
                        description="Demander confirmation à l'utilisateur pour une préférence détectée. UTILISE CETTE FONCTION quand tu détectes une information comme le temps de trajet, les heures de sommeil, etc. Affiche des options radio pour confirmer/modifier.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "preference_type": types.Schema(type="STRING", description="Type: transport_time|sleep_hours|productivity_time|deep_work_hours"),
                                "detected_value": types.Schema(type="STRING", description="Valeur détectée dans le message (ex: '20' pour 20 min)"),
                                "question": types.Schema(type="STRING", description="Question à poser à l'utilisateur"),
                            },
                            required=["preference_type", "detected_value", "question"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="show_planning_proposal",
                        description="Générer et afficher une proposition de planning intelligent basée sur les disponibilités de l'utilisateur. Utilise cette fonction quand l'utilisateur veut voir sa proposition de planning.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={},
                            required=[]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="accept_planning_proposal",
                        description="Accepter la proposition de planning et créer tous les blocs proposés. Utilise cette fonction quand l'utilisateur dit 'oui', 'crée ce planning', 'valide', 'c'est bon', etc.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={},
                            required=[]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="add_document_data",
                        description="Ajouter les données extraites du dernier document uploadé au planning. Utilise cette fonction quand l'utilisateur veut ajouter les données d'un document.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={},
                            required=[]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="complete_onboarding",
                        description="Marquer l'onboarding comme terminé. Utilise cette fonction après avoir configuré le planning initial de l'utilisateur.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={},
                            required=[]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="ask_clarification",
                        description="Demander une clarification quand le message de l'utilisateur n'est pas clair ou ambigu. Affiche des options radio pour guider l'utilisateur.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "question": types.Schema(type="STRING", description="Question de clarification à poser"),
                                "options": types.Schema(
                                    type="ARRAY",
                                    items=types.Schema(type="STRING"),
                                    description="Liste des options possibles (2-4 options)"
                                ),
                                "allow_other": types.Schema(type="BOOLEAN", description="Permettre une réponse personnalisée"),
                            },
                            required=["question", "options"]
                        )
                    ),
                ]
            )
        ]

        # Build conversation context
        prompt = f"""{system_prompt}

Historique récent:
"""
        for msg in history[-5:]:
            role = "Utilisateur" if msg['role'] == 'user' else "Assistant"
            prompt += f"{role}: {msg['content']}\n"

        # Get onboarding status for context
        profile = user.profile
        onboarding_status = "terminé" if profile.onboarding_completed else f"en cours (étape {profile.onboarding_step})"

        # Analyze session context from recent history
        session_context = self._analyze_session_context(history, message)

        prompt += f"""
Utilisateur: {message}

CONTEXTE DE SESSION:
- Dernier sujet: {session_context.get('last_topic', 'aucun')}
- Action en attente: {session_context.get('pending_action', 'aucune')}
- Humeur conversation: {session_context.get('mood', 'neutre')}
- Clarification demandée: {'oui' if session_context.get('clarification_asked') else 'non'}

STATUT ONBOARDING: {onboarding_status}
DATE: {timezone.now().strftime('%Y-%m-%d %H:%M')}

══════════════════════════════════════════════════════════════
CLASSIFICATION DES MESSAGES - ANALYSE D'ABORD LE TYPE:
══════════════════════════════════════════════════════════════

1. CASUAL (salut, ça va, merci, ok, cool, etc.)
   → Réponds naturellement et chaleureusement SANS appeler de fonction
   → Exemples: "Salut! Comment ça va aujourd'hui?", "De rien, avec plaisir!"

2. PREFERENCE (temps trajet, sommeil, productivité, etc.)
   → UTILISE ask_preference_confirmation avec options radio
   → "20 min de route" → ask_preference_confirmation(preference_type="transport_time", detected_value="20", question="Je note 20 minutes de trajet. C'est bien ça?")
   → "je dors 7h" → ask_preference_confirmation(preference_type="sleep_hours", detected_value="7", question="...")

3. ACTION (créer bloc, ajouter tâche, modifier planning)
   → Exécute l'action avec la fonction appropriée
   → "je travaille de 19h à 7h" → create_recurring_block directement

4. QUESTION (c'est quoi mon planning, j'ai quoi demain, comment ça marche)
   → Réponds à la question en utilisant le contexte fourni
   → Utilise show_planning_proposal si pertinent

5. CONFIRMATION (oui, d'accord, c'est bon, valide, confirme)
   → Si confirmation d'une préférence (ex: "transport_time_minutes: 20") → update_preference
   → Si confirmation de planning → accept_planning_proposal

6. HORS_SUJET (météo, capitale, recette, etc.)
   → "Je suis spécialisé dans la planification! Je peux t'aider à organiser ton emploi du temps, ajouter des blocs, gérer tes tâches..."

7. AMBIGU (pas clair ce que l'utilisateur veut)
   → UTILISE ask_clarification pour proposer des options
   → ask_clarification(question="Je n'ai pas bien compris. Tu voulais:", options=["Ajouter un bloc à ton planning", "Modifier une préférence", "Voir ton planning"], allow_other=true)

══════════════════════════════════════════════════════════════
FONCTIONS DISPONIBLES:
══════════════════════════════════════════════════════════════
- create_recurring_block: Créer bloc récurrent (cours, travail, sport, etc.)
- create_task: Créer tâche avec deadline
- ask_preference_confirmation: Demander confirmation préférence (AVANT update_preference)
- update_preference: Sauvegarder préférence (APRÈS confirmation)
- ask_clarification: Demander clarification si message ambigu
- show_planning_proposal: Montrer proposition de planning
- accept_planning_proposal: Valider planning proposé
- add_document_data: Ajouter données document
- complete_onboarding: Terminer onboarding

══════════════════════════════════════════════════════════════
RÈGLES IMPORTANTES:
══════════════════════════════════════════════════════════════
1. JAMAIS répondre "Comment puis-je t'aider?" de façon robotique
2. Pour CASUAL → Réponds naturellement, sois amical
3. Pour PREFERENCE → TOUJOURS ask_preference_confirmation d'abord
4. Pour AMBIGU → UTILISE ask_clarification avec options
5. Pour HORS_SUJET → Explique poliment ton domaine (planification)
6. Sois concis et naturel, pas robotique

══════════════════════════════════════════════════════════════
UTILISE LE CONTEXTE DE SESSION:
══════════════════════════════════════════════════════════════
- Si "confirmation_attendue" → L'utilisateur répond probablement à ta question précédente
- Si "réponse_attendue" → Interprète la réponse dans le contexte de ta question
- Si "clarification_asked" → L'utilisateur clarifie sa demande précédente
- Si "last_topic" est défini → Le sujet est probablement lié à ce thème
- Si "mood: frustré" → Sois plus empathique et propose de l'aide
- Si "mood: positif" → Continue sur cette lancée positive"""

        try:
            response = self._call_gemini(
                prompt,
                config=types.GenerateContentConfig(
                    tools=tools,
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode='AUTO')
                    )
                )
            )

            logger.info(f"Gemini response candidates: {len(response.candidates)}")

            response_text = ""
            tasks_created = []
            blocks_created = []
            skipped_days_all = []
            function_calls_made = []
            preferences_updated = []  # Track which preferences were updated

            for part in response.parts:
                logger.info(f"Response part type: {type(part)}, has function_call: {hasattr(part, 'function_call')}")

                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    args = dict(fc.args)
                    function_calls_made.append(fc.name)
                    logger.info(f"Function call: {fc.name} with args: {args}")

                    if fc.name == "create_recurring_block":
                        result = self._execute_create_block(user, args)
                        if result:
                            blocks_created.extend(result.get('blocks', []))
                            skipped_days_all.extend(result.get('skipped_days', []))
                            if result.get('blocks'):
                                logger.info(f"Block(s) created: {[b.title for b in result['blocks']]}")
                            if result.get('skipped_days'):
                                logger.info(f"Days skipped due to overlap: {result['skipped_days']}")
                        else:
                            logger.error(f"Failed to create block with args: {args}")

                    elif fc.name == "create_task":
                        task = self._execute_create_task(user, args)
                        if task:
                            tasks_created.append(task)
                            logger.info(f"Task created: {task.title} (id={task.id})")
                        else:
                            logger.error(f"Failed to create task with args: {args}")

                    elif fc.name == "ask_preference_confirmation":
                        # Generate interactive radio inputs for preference confirmation
                        pref_type = args.get('preference_type', '')
                        detected_value = args.get('detected_value', '')
                        question = args.get('question', 'Confirme cette information:')

                        interactive_input = self._generate_preference_radio_input(pref_type, detected_value, question)
                        if interactive_input:
                            # Return immediately with interactive inputs
                            return {
                                'text': question,
                                'quick_replies': [],
                                'tasks_created': [],
                                'interactive_inputs': [interactive_input]
                            }
                        logger.info(f"Asked preference confirmation: {pref_type}={detected_value}")

                    elif fc.name == "ask_clarification":
                        # Generate clarification radio inputs when message is ambiguous
                        question = args.get('question', "Je n'ai pas bien compris. Tu voulais:")
                        options = args.get('options', [])
                        allow_other = args.get('allow_other', True)

                        if options:
                            interactive_input = {
                                'id': 'clarification',
                                'type': 'radio',
                                'label': 'Clarification',
                                'question': question,
                                'options': [{'value': opt, 'label': opt} for opt in options],
                                'allowOther': allow_other,
                                'otherPlaceholder': 'Autre chose...',
                            }
                            return {
                                'text': question,
                                'quick_replies': [],
                                'tasks_created': [],
                                'interactive_inputs': [interactive_input]
                            }
                        logger.info(f"Asked clarification with {len(options)} options")

                    elif fc.name == "update_preference":
                        pref_result = self._execute_update_preference(user, args)
                        if pref_result:
                            pref_name = args.get('preference', '')
                            pref_value = args.get('value', '')
                            preferences_updated.append({'name': pref_name, 'value': pref_value})
                            # Add confirmation if AI didn't provide text
                            pref_confirmations = {
                                'transport_time_minutes': f"✅ J'ai noté {pref_value} minutes de trajet.",
                                'min_sleep_hours': f"✅ J'ai noté {pref_value}h de sommeil minimum.",
                                'peak_productivity_time': f"✅ J'ai noté que tu es plus productif le {pref_value.replace('morning', 'matin').replace('afternoon', 'après-midi').replace('evening', 'soir')}.",
                                'max_deep_work_hours_per_day': f"✅ J'ai noté {pref_value}h de travail profond max par jour.",
                            }
                            if pref_name in pref_confirmations and not response_text:
                                response_text = pref_confirmations[pref_name]
                            logger.info(f"Preference updated: {pref_name}={pref_value}")

                    elif fc.name == "show_planning_proposal":
                        proposal = self._generate_smart_planning_proposal(user)
                        response_text = proposal.get('text', '')
                        logger.info("Generated planning proposal via function call")

                    elif fc.name == "accept_planning_proposal":
                        created_blocks = self._create_proposed_blocks(user)
                        blocks_count = len(created_blocks) if created_blocks else 0
                        logger.info(f"Accepted proposal - created {blocks_count} blocks")
                        # Mark onboarding as completed
                        profile = user.profile
                        profile.onboarding_completed = True
                        profile.onboarding_step = 3
                        profile.save()
                        if blocks_count > 0:
                            response_text = f"✅ Parfait! J'ai créé {blocks_count} blocs dans ton planning.\n\nTon planning est prêt! Tu veux ajouter autre chose?"
                        else:
                            response_text = "✅ Ton planning est configuré! Tu veux ajouter des blocs?"

                    elif fc.name == "add_document_data":
                        add_result = self._add_extracted_to_planning(user)
                        response_text = add_result.get('text', '')
                        logger.info("Added document data to planning via function call")

                    elif fc.name == "complete_onboarding":
                        profile = user.profile
                        profile.onboarding_completed = True
                        profile.onboarding_step = 3
                        profile.save()
                        response_text = "✅ Configuration terminée! Tu peux maintenant ajouter des tâches ou des blocs à ton planning."
                        logger.info(f"Onboarding completed for user {user.id}")

                elif hasattr(part, 'text') and part.text:
                    response_text += part.text

            logger.info(f"Function calls made: {function_calls_made}, blocks created: {len(blocks_created)}, tasks created: {len(tasks_created)}")

            # Build response
            result_text = response_text.strip() if response_text else ""

            # Add confirmation for created items
            if blocks_created:
                # Group blocks by title and time for better display
                day_names = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']
                block_groups = {}
                for b in blocks_created:
                    key = (b.title, b.start_time.strftime('%H:%M'), b.end_time.strftime('%H:%M'))
                    if key not in block_groups:
                        block_groups[key] = []
                    block_groups[key].append(day_names[b.day_of_week])

                block_descriptions = []
                for (title, start, end), days in block_groups.items():
                    block_descriptions.append(f"{title} ({start}-{end}) les {', '.join(days)}")

                confirmation = f"✅ J'ai créé {len(blocks_created)} bloc(s): {'; '.join(block_descriptions)}"
                if result_text:
                    result_text += f"\n\n{confirmation}"
                else:
                    result_text = confirmation

            # Inform about skipped days due to overlapping blocks
            if skipped_days_all:
                unique_skipped = list(set(skipped_days_all))
                result_text += f"\n\n⚠️ Certains jours n'ont pas été ajoutés car il y a déjà des blocs sur cette plage horaire: {', '.join(unique_skipped)}"
            elif not blocks_created and function_calls_made and 'create_recurring_block' in function_calls_made:
                # All blocks were skipped or failed
                result_text = "⚠️ Je n'ai pas pu créer ce bloc car il y a déjà des activités sur cette plage horaire. Choisis un autre créneau."

            if tasks_created:
                task_names = [t.title for t in tasks_created]
                # Check if tasks were scheduled
                scheduled_count = sum(1 for t in tasks_created if t.scheduled_blocks.exists())

                if result_text:
                    result_text += f"\n\n📝 Tâche(s) créée(s): {', '.join(task_names)}"
                else:
                    result_text = f"📝 J'ai créé {len(tasks_created)} tâche(s): {', '.join(task_names)}"

                if scheduled_count > 0:
                    result_text += f"\n📅 Planifié automatiquement dans ton emploi du temps!"

            if not result_text:
                # Intelligent fallback based on context
                result_text = self._get_intelligent_fallback(user, message, function_calls_made)

            # Check if this is a casual conversation (no quick replies needed)
            is_casual = self._is_casual_message(message)

            # Build contextual quick replies (skip for casual conversation)
            if is_casual and not blocks_created and not tasks_created and not preferences_updated:
                quick_replies = []  # Natural conversation without buttons
            else:
                quick_replies = self._get_contextual_quick_replies(user, blocks_created, tasks_created, preferences_updated)

            return {
                'text': result_text,
                'quick_replies': quick_replies,
                'tasks_created': tasks_created,
                'interactive_inputs': []
            }

        except Exception as e:
            logger.error(f"Smart conversation error: {e}")
            return {
                'text': "J'ai eu un souci. Peux-tu reformuler?",
                'quick_replies': [],
                'tasks_created': [],
                'interactive_inputs': []
            }

    def _execute_create_block(self, user: User, args: dict) -> Optional[RecurringBlock]:
        """Execute block creation from Gemini function call with security validation."""
        from datetime import time as dt_time

        logger.info(f"_execute_create_block called with args: {args}")

        try:
            # ========== SECURITY: Validate and sanitize args ==========
            is_valid, error_msg, sanitized = self._validate_block_args(args)
            if not is_valid:
                logger.warning(f"Block validation failed: {error_msg}")
                return None

            title = sanitized['title']
            block_type = sanitized['block_type']
            days = sanitized['days']
            start_time_str = sanitized['start_time']
            end_time_str = sanitized['end_time']
            location = sanitized['location']

            logger.info(f"Sanitized values - title: {title}, type: {block_type}, days: {days}, times: {start_time_str}-{end_time_str}")

            # Parse times
            start_parts = start_time_str.split(':')
            end_parts = end_time_str.split(':')
            start_time = dt_time(int(start_parts[0]), int(start_parts[1]))
            end_time = dt_time(int(end_parts[0]), int(end_parts[1]))

            # Create block for each day (check for overlaps first)
            created_blocks = []
            skipped_days = []

            for day in days:
                if 0 <= day <= 6:
                    # Check for existing overlapping blocks on this day
                    existing_blocks = RecurringBlock.objects.filter(
                        user=user,
                        day_of_week=day
                    )

                    has_overlap = False
                    for existing in existing_blocks:
                        # Check if time ranges overlap
                        # Two ranges overlap if: start1 < end2 AND start2 < end1
                        existing_start = existing.start_time
                        existing_end = existing.end_time

                        # Handle overnight blocks (end < start means it goes past midnight)
                        if existing_end <= existing_start:
                            # Overnight block: overlaps if new block starts before existing ends (next day)
                            # or if new block is within the evening portion
                            if start_time >= existing_start or end_time <= existing_end:
                                has_overlap = True
                                break
                        elif end_time <= start_time:
                            # New block is overnight
                            if existing_start >= start_time or existing_end <= end_time:
                                has_overlap = True
                                break
                        else:
                            # Normal case: both blocks are within same day
                            if start_time < existing_end and existing_start < end_time:
                                has_overlap = True
                                break

                    if has_overlap:
                        logger.warning(f"Skipping day {day}: overlapping block exists at {start_time}-{end_time}")
                        skipped_days.append(day)
                        continue

                    # ========== SECURITY: Check block limits ==========
                    is_allowed, limit_error = self._validate_block_limits(user, day)
                    if not is_allowed:
                        logger.warning(f"Block limit exceeded for day {day}: {limit_error}")
                        skipped_days.append(day)
                        continue

                    block = RecurringBlock.objects.create(
                        user=user,
                        title=title,
                        block_type=block_type,
                        day_of_week=day,
                        start_time=start_time,
                        end_time=end_time,
                        location=location or '',
                    )
                    created_blocks.append(block)
                    logger.info(f"Created block id={block.id} for day {day}")

            skipped_names = []
            if skipped_days:
                day_names = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']
                skipped_names = [day_names[d] for d in skipped_days]
                logger.info(f"Skipped {len(skipped_days)} days due to overlapping blocks: {skipped_names}")

            if created_blocks:
                logger.info(f"Successfully created {len(created_blocks)} blocks")
                return {
                    'blocks': created_blocks,
                    'skipped_days': skipped_names
                }
            else:
                if skipped_days:
                    # All days were skipped due to overlaps
                    return {
                        'blocks': [],
                        'skipped_days': skipped_names
                    }
                logger.error(f"No blocks created - days list was empty or invalid: {days}")
                return None

        except Exception as e:
            logger.error(f"Error creating block: {e}", exc_info=True)
            return None

    def _execute_create_task(self, user: User, args: dict) -> Optional[Task]:
        """Execute task creation from Gemini function call with security validation."""
        try:
            # ========== SECURITY: Validate task limits ==========
            is_allowed, limit_error = self._validate_task_limits(user)
            if not is_allowed:
                logger.warning(f"Task limit exceeded for user {user.id}: {limit_error}")
                return None

            # ========== SECURITY: Validate and sanitize args ==========
            is_valid, error_msg, sanitized = self._validate_task_args(args)
            if not is_valid:
                logger.warning(f"Task validation failed: {error_msg}")
                return None

            title = sanitized['title']
            description = sanitized['description']
            deadline = sanitized.get('deadline')
            priority = sanitized['priority']
            task_type = sanitized['task_type']

            # Add timezone if deadline exists
            if deadline:
                deadline = deadline.replace(tzinfo=timezone.get_current_timezone())

            task = Task.objects.create(
                user=user,
                title=title,
                description=description,
                deadline=deadline,
                priority=priority,
                task_type=task_type,
            )

            # Auto-schedule the task
            try:
                scheduler = AIScheduler()
                scheduled_blocks = scheduler.generate_schedule(
                    user=user,
                    tasks=[task],
                    num_days=7
                )
                if scheduled_blocks:
                    logger.info(f"Auto-scheduled task '{task.title}' from chat into {len(scheduled_blocks)} block(s)")
            except Exception as e:
                logger.warning(f"Could not auto-schedule task '{task.title}': {e}")

            return task

        except Exception as e:
            logger.error(f"Error creating task: {e}")
            return None

    def _analyze_session_context(self, history: list, current_message: str) -> dict:
        """
        Analyze conversation history to build session context.
        This helps the bot understand the flow of conversation.
        """
        import re

        context = {
            'last_topic': None,
            'pending_action': None,
            'mood': 'neutre',
            'clarification_asked': False,
            'preference_being_discussed': None,
        }

        if not history:
            return context

        # Analyze last few messages
        recent_messages = history[-4:]  # Last 4 messages (2 exchanges)

        # Topic detection patterns
        topic_patterns = {
            'transport': r'(trajet|route|transport|voiture|métro|bus|vélo)',
            'sommeil': r'(dors|sommeil|couche|réveil|lever|nuit)',
            'travail': r'(travail|boulot|job|bureau|entreprise|horaires)',
            'cours': r'(cours|école|université|fac|études|exam)',
            'sport': r'(sport|gym|fitness|entraînement|course|foot)',
            'productivité': r'(productif|concentration|focus|efficace)',
            'planning': r'(planning|emploi du temps|agenda|calendrier)',
        }

        # Check recent messages for topics
        for msg in reversed(recent_messages):
            content = msg.get('content', '').lower()
            for topic, pattern in topic_patterns.items():
                if re.search(pattern, content):
                    context['last_topic'] = topic
                    break
            if context['last_topic']:
                break

        # Check for pending actions (questions asked by bot)
        if recent_messages:
            last_assistant_msg = None
            for msg in reversed(recent_messages):
                if msg.get('role') == 'assistant':
                    last_assistant_msg = msg.get('content', '')
                    break

            if last_assistant_msg:
                # Check if bot asked a question
                if '?' in last_assistant_msg:
                    if any(word in last_assistant_msg.lower() for word in ['confirme', 'c\'est bien', 'correct']):
                        context['pending_action'] = 'confirmation_attendue'
                    elif any(word in last_assistant_msg.lower() for word in ['quel', 'quelle', 'combien', 'quand']):
                        context['pending_action'] = 'réponse_attendue'

                # Check if clarification was asked
                if any(phrase in last_assistant_msg.lower() for phrase in ['je n\'ai pas compris', 'tu voulais', 'peux-tu préciser']):
                    context['clarification_asked'] = True

        # Detect conversation mood
        current_lower = current_message.lower()
        if any(word in current_lower for word in ['merci', 'super', 'génial', 'parfait', 'cool', 'top']):
            context['mood'] = 'positif'
        elif any(word in current_lower for word in ['non', 'pas ça', 'erreur', 'problème', 'bug']):
            context['mood'] = 'frustré'
        elif self._is_casual_message(current_message):
            context['mood'] = 'casual'

        # Check if preference is being discussed
        pref_patterns = {
            'transport_time': r'(\d+)\s*(min|minutes?)\s*(de\s*)?(route|trajet)',
            'sleep_hours': r'(dors|dort)\s*(\d+)\s*h',
            'productivity_time': r'(productif|efficace).*(matin|soir|après-midi)',
        }

        for pref, pattern in pref_patterns.items():
            if re.search(pattern, current_lower):
                context['preference_being_discussed'] = pref
                break

        return context

    def _is_casual_message(self, message: str) -> bool:
        """
        Check if a message is casual conversation (greetings, thanks, ok, etc.)
        These messages don't need quick replies - keep the conversation natural.
        """
        import re
        message_lower = message.lower().strip()

        casual_patterns = [
            r'^(salut|hello|hi|hey|coucou|bonjour|bonsoir|yo)[\s!?]*$',
            r'^(ça va|ca va|comment ça va|comment vas-tu|quoi de neuf)[\s!?]*$',
            r'^(ça va bien|ca va bien|bien et toi|super et toi).*$',
            r'^(merci|thanks|thx|merci beaucoup)[\s!?]*$',
            r'^(ok|d\'accord|cool|super|nice|top|parfait|génial|genial)[\s!?]*$',
            r'^(oui|non|ouais|nope|yep|nan)[\s!?]*$',
            r'^(de rien|pas de quoi|avec plaisir)[\s!?]*$',
            r'^(à plus|a plus|bye|ciao|à bientôt|a bientot)[\s!?]*$',
        ]

        for pattern in casual_patterns:
            if re.search(pattern, message_lower):
                return True
        return False

    def _get_intelligent_fallback(self, user: User, message: str, function_calls_made: list) -> str:
        """
        Generate an intelligent fallback response based on context.

        Instead of generic "Comment puis-je t'aider?", provide contextual responses.
        """
        import re

        message_lower = message.lower().strip()

        # Casual greetings - respond naturally
        casual_patterns = {
            r'^(salut|hello|hi|hey|coucou|bonjour|bonsoir)': [
                "Salut! Comment ça va?",
                "Hey! Qu'est-ce qui t'amène?",
                "Coucou! Je suis là pour t'aider avec ton planning.",
            ],
            r'^(ça va|ca va|comment ça va|comment vas-tu)': [
                "Ça va bien, merci! Et toi, comment je peux t'aider?",
                "Super! Qu'est-ce que je peux faire pour toi?",
            ],
            r'^(merci|thanks|thx)': [
                "De rien! N'hésite pas si tu as besoin d'autre chose.",
                "Avec plaisir! Je suis là si tu as besoin.",
            ],
            r'^(ok|d\'accord|cool|super|nice|top)': [
                "Parfait! Autre chose?",
                "Super! Tu as besoin d'autre chose?",
            ],
            r'^(oui|non|ouais|nope)$': [
                "D'accord! Qu'est-ce que tu veux faire maintenant?",
            ],
        }

        import random
        for pattern, responses in casual_patterns.items():
            if re.search(pattern, message_lower):
                return random.choice(responses)

        # Check user's profile completion
        profile = user.profile
        missing_prefs = []
        if not profile.transport_time_minutes:
            missing_prefs.append("temps de trajet")
        if not profile.min_sleep_hours:
            missing_prefs.append("heures de sommeil")

        # Check if they have blocks
        from core.models import RecurringBlock
        blocks_count = RecurringBlock.objects.filter(user=user).count()

        # Contextual suggestion based on what's missing
        if not profile.onboarding_completed:
            if blocks_count == 0:
                return "Dis-moi tes contraintes! Par exemple: 'je travaille de 9h à 17h' ou 'j'ai cours le lundi matin'."
            elif missing_prefs:
                return f"Pour optimiser ton planning, dis-moi ton {missing_prefs[0]}. Par exemple: 'je prends 20 min de route'."

        # If we made function calls but got no response, something went wrong
        if function_calls_made:
            return "C'est noté! Autre chose à ajouter?"

        # Default contextual response
        if blocks_count > 0:
            return "Je suis là! Tu veux ajouter quelque chose à ton planning ou modifier une préférence?"
        else:
            return "Je suis ton assistant de planification! Dis-moi tes horaires de travail, tes cours, ou tes activités."

    def _generate_preference_radio_input(self, pref_type: str, detected_value: str, question: str) -> dict:
        """
        Generate a radio input configuration for preference confirmation.

        Args:
            pref_type: Type of preference (transport_time, sleep_hours, etc.)
            detected_value: The value detected from the user's message
            question: The question to ask

        Returns:
            dict: Radio input configuration for the frontend
        """
        # Define options based on preference type
        options_map = {
            'transport_time': {
                'id': 'transport_time_minutes',
                'label': 'Temps de trajet',
                'options': [
                    {'value': '10', 'label': '10 minutes'},
                    {'value': '15', 'label': '15 minutes'},
                    {'value': '20', 'label': '20 minutes'},
                    {'value': '30', 'label': '30 minutes'},
                    {'value': '45', 'label': '45 minutes'},
                    {'value': '60', 'label': '1 heure'},
                ],
                'other_placeholder': 'Nombre de minutes...',
            },
            'sleep_hours': {
                'id': 'min_sleep_hours',
                'label': 'Heures de sommeil',
                'options': [
                    {'value': '6', 'label': '6 heures'},
                    {'value': '7', 'label': '7 heures'},
                    {'value': '8', 'label': '8 heures'},
                    {'value': '9', 'label': '9 heures'},
                ],
                'other_placeholder': 'Nombre d\'heures...',
            },
            'productivity_time': {
                'id': 'peak_productivity_time',
                'label': 'Moment de productivité',
                'options': [
                    {'value': 'morning', 'label': 'Le matin'},
                    {'value': 'afternoon', 'label': 'L\'après-midi'},
                    {'value': 'evening', 'label': 'Le soir'},
                ],
                'allow_other': False,
            },
            'deep_work_hours': {
                'id': 'max_deep_work_hours_per_day',
                'label': 'Heures de travail profond',
                'options': [
                    {'value': '2', 'label': '2 heures'},
                    {'value': '3', 'label': '3 heures'},
                    {'value': '4', 'label': '4 heures'},
                    {'value': '5', 'label': '5 heures'},
                    {'value': '6', 'label': '6 heures'},
                ],
                'other_placeholder': 'Nombre d\'heures...',
            },
        }

        config = options_map.get(pref_type)
        if not config:
            return None

        # Try to match detected_value to an option, or use it as default
        default_value = detected_value
        for opt in config['options']:
            if opt['value'] == detected_value or opt['label'].lower().startswith(detected_value.lower()):
                default_value = opt['value']
                break

        return {
            'id': config['id'],
            'type': 'radio',
            'label': config['label'],
            'question': question,
            'options': config['options'],
            'default': default_value,
            'allowOther': config.get('allow_other', True),
            'otherPlaceholder': config.get('other_placeholder', 'Précisez...'),
        }

    def _execute_update_preference(self, user: User, args: dict) -> bool:
        """Execute preference update from Gemini function call."""
        try:
            preference = args.get('preference', '')
            value = args.get('value', '')
            profile = user.profile

            if preference == 'peak_productivity_time':
                if value in ['morning', 'afternoon', 'evening']:
                    profile.peak_productivity_time = value
                elif value in ['matin', 'le matin']:
                    profile.peak_productivity_time = 'morning'
                elif value in ['après-midi', 'apres-midi', "l'après-midi"]:
                    profile.peak_productivity_time = 'afternoon'
                elif value in ['soir', 'le soir']:
                    profile.peak_productivity_time = 'evening'

            elif preference == 'min_sleep_hours':
                try:
                    hours = int(value)
                    if 4 <= hours <= 12:
                        profile.min_sleep_hours = hours
                except ValueError:
                    pass

            elif preference == 'max_deep_work_hours_per_day':
                try:
                    hours = int(value)
                    if 1 <= hours <= 12:
                        profile.max_deep_work_hours_per_day = hours
                except ValueError:
                    pass

            elif preference == 'transport_time_minutes':
                try:
                    minutes = int(value)
                    if 0 <= minutes <= 180:
                        profile.transport_time_minutes = minutes
                except ValueError:
                    pass

            profile.save()
            return True

        except Exception as e:
            logger.error(f"Error updating preference: {e}")
            return False

    def _get_contextual_quick_replies(self, user: User, blocks_created: list = None, tasks_created: list = None, preferences_updated: list = None) -> list:
        """
        Generate contextual quick reply buttons based on user state.

        Args:
            user: The current user
            blocks_created: List of blocks just created (if any)
            tasks_created: List of tasks just created (if any)
            preferences_updated: List of preferences just updated (if any)

        Returns:
            list: Contextual quick reply buttons
        """
        quick_replies = []

        # Get user state
        pending_tasks = Task.objects.filter(user=user, completed=False).count()
        total_blocks = RecurringBlock.objects.filter(user=user, active=True).count()
        today = timezone.now()
        is_weekend = today.weekday() >= 5

        # If preferences were updated, offer relevant follow-up actions
        if preferences_updated:
            pref_names = [p['name'] for p in preferences_updated]

            if 'transport_time_minutes' in pref_names:
                quick_replies.append({'label': "🚗 Ajouter déplacement", 'value': "Ajoute des blocs de déplacement dans mon planning"})
                quick_replies.append({'label': "📅 Voir planning", 'value': 'Montre-moi mon planning'})

            elif 'min_sleep_hours' in pref_names:
                quick_replies.append({'label': "😴 Ajouter sommeil", 'value': "Ajoute des blocs de sommeil dans mon planning"})
                quick_replies.append({'label': "📅 Voir planning", 'value': 'Montre-moi mon planning'})

            elif 'peak_productivity_time' in pref_names:
                quick_replies.append({'label': "📅 Optimiser planning", 'value': "Optimise mon planning selon ma productivité"})
                quick_replies.append({'label': "➕ Ajouter tâche", 'value': "Je veux ajouter une tâche"})

            else:
                quick_replies.append({'label': "📅 Voir planning", 'value': 'Montre-moi mon planning'})
                quick_replies.append({'label': "➕ Ajouter bloc", 'value': "Je veux ajouter un bloc"})

        # If blocks were just created, offer to see planning
        elif blocks_created:
            quick_replies.append({'label': "📅 Voir le planning", 'value': 'Montre-moi mon planning de la semaine'})
            quick_replies.append({'label': "➕ Ajouter autre chose", 'value': "Je veux ajouter un autre bloc"})

        # If tasks were just created, offer task-related actions
        elif tasks_created:
            quick_replies.append({'label': "📋 Voir mes tâches", 'value': 'Quelles sont mes tâches à faire?'})
            if total_blocks > 0:
                quick_replies.append({'label': "⏰ Quand travailler dessus?", 'value': 'Quand puis-je travailler sur cette tâche?'})

        # Default contextual buttons based on user state
        else:
            # Always offer planning view if user has blocks
            if total_blocks > 0:
                if is_weekend:
                    quick_replies.append({'label': "📅 Planning du weekend", 'value': 'Montre-moi mon planning du weekend'})
                else:
                    quick_replies.append({'label': "📅 Planning d'aujourd'hui", 'value': "Qu'est-ce que j'ai aujourd'hui?"})

            # Offer task view if user has pending tasks
            if pending_tasks > 0:
                quick_replies.append({'label': f"📋 {pending_tasks} tâche(s)", 'value': 'Montre-moi mes tâches à faire'})

            # Always offer to add something
            if len(quick_replies) < 2:
                quick_replies.append({'label': "➕ Ajouter", 'value': "Je veux ajouter quelque chose à mon planning"})

        # Limit to 3 buttons max
        return quick_replies[:3]

    def _generate_smart_planning_proposal(self, user: User, overrides: dict = None) -> dict:
        """
        Analyze user's schedule and generate an intelligent planning proposal.

        Instead of asking many questions, this method:
        1. Analyzes existing blocks (courses, work)
        2. Identifies free time slots
        3. Proposes a complete optimized schedule

        Args:
            user: The current user
            overrides: Optional dict with custom times:
                - bedtime: "HH:MM"
                - wake_time: "HH:MM"
                - breakfast_time: "HH:MM"
                - lunch_time: "HH:MM"

        Returns:
            dict: Response with proposal text and quick_replies
        """
        from datetime import time as dt_time

        overrides = overrides or {}
        profile = user.profile
        blocks = RecurringBlock.objects.filter(user=user, active=True).order_by('day_of_week', 'start_time')

        if not blocks.exists():
            return {
                'text': "Je n'ai pas encore assez d'informations sur ton emploi du temps. Envoie-moi ton planning de cours ou de travail!",
                'quick_replies': [
                    {'label': "📚 Envoyer mon emploi du temps", 'value': 'upload'},
                ]
            }

        # Analyze schedule by day
        days_fr = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        schedule_by_day = {i: [] for i in range(7)}

        earliest_start_by_day = {}
        latest_end_by_day = {}

        for block in blocks:
            day = block.day_of_week
            schedule_by_day[day].append({
                'start': block.start_time,
                'end': block.end_time,
                'title': block.title,
                'type': block.block_type,
            })

            # Track earliest start and latest end per day
            if day not in earliest_start_by_day or block.start_time < earliest_start_by_day[day]:
                earliest_start_by_day[day] = block.start_time
            if day not in latest_end_by_day or block.end_time > latest_end_by_day[day]:
                latest_end_by_day[day] = block.end_time

        # Find the earliest start across the week (for sleep calculation)
        earliest_weekday_start = None
        for day in range(5):  # Monday to Friday
            if day in earliest_start_by_day:
                if earliest_weekday_start is None or earliest_start_by_day[day] < earliest_weekday_start:
                    earliest_weekday_start = earliest_start_by_day[day]

        # Calculate proposed sleep schedule
        min_sleep = profile.min_sleep_hours or 7
        transport_time = profile.transport_time_minutes or 30

        if earliest_weekday_start:
            # Wake up = earliest start - transport time - 1h (breakfast/prep)
            wake_minutes = earliest_weekday_start.hour * 60 + earliest_weekday_start.minute
            wake_minutes -= transport_time + 60  # transport + prep time
            wake_hour = max(5, wake_minutes // 60)  # Not earlier than 5am
            wake_minute = wake_minutes % 60 if wake_minutes > 0 else 0

            # Bedtime = wake time - sleep hours
            bed_hour = (wake_hour - min_sleep) % 24
            if bed_hour < 12:  # Should be evening
                bed_hour += 24 - min_sleep

            proposed_wake = f"{wake_hour:02d}:{wake_minute:02d}"
            proposed_bed = f"{bed_hour % 24:02d}:00"
        else:
            proposed_wake = "07:00"
            proposed_bed = "23:00"

        # Apply overrides if provided
        if overrides.get('bedtime'):
            proposed_bed = overrides['bedtime']
        if overrides.get('wake_time'):
            proposed_wake = overrides['wake_time']

        # Calculate free slots and propose meals
        def find_free_slots(day_blocks, day_start=dt_time(7, 0), day_end=dt_time(23, 0)):
            """Find free time slots in a day."""
            if not day_blocks:
                return [(day_start, day_end)]

            sorted_blocks = sorted(day_blocks, key=lambda x: x['start'])
            free_slots = []

            current_time = day_start
            for block in sorted_blocks:
                if block['start'] > current_time:
                    free_slots.append((current_time, block['start']))
                current_time = max(current_time, block['end'])

            if current_time < day_end:
                free_slots.append((current_time, day_end))

            return free_slots

        # Build the proposal
        proposal_lines = ["**Voici ma proposition de planning optimisé:**\n"]

        # Sleep schedule
        proposal_lines.append(f"**🛏️ Sommeil ({min_sleep}h minimum)**")
        proposal_lines.append(f"  Coucher: {proposed_bed} → Réveil: {proposed_wake}")
        proposal_lines.append("")

        # Analyze a typical weekday for meal proposals
        typical_day = None
        for day in range(5):
            if schedule_by_day[day]:
                typical_day = day
                break

        # Build meal times - use overrides or calculate from free slots
        proposed_meals = []

        # Check for overrides first
        if overrides.get('breakfast_time'):
            proposed_meals.append(('🍳 Petit-déjeuner', overrides['breakfast_time']))
        if overrides.get('lunch_time'):
            proposed_meals.append(('🍽️ Déjeuner', overrides['lunch_time']))

        # Calculate remaining meals from free slots if not all overridden
        if typical_day is not None and len(proposed_meals) < 3:
            free_slots = find_free_slots(schedule_by_day[typical_day])
            has_breakfast = any('Petit-déjeuner' in m[0] for m in proposed_meals)
            has_lunch = any('Déjeuner' in m[0] for m in proposed_meals)

            for slot_start, slot_end in free_slots:
                slot_start_minutes = slot_start.hour * 60 + slot_start.minute
                slot_end_minutes = slot_end.hour * 60 + slot_end.minute
                duration = slot_end_minutes - slot_start_minutes

                # Breakfast (6h-9h)
                if not has_breakfast and 360 <= slot_start_minutes < 540 and duration >= 30:
                    proposed_meals.append(('🍳 Petit-déjeuner', f"{slot_start.hour:02d}:{slot_start.minute:02d}"))
                    has_breakfast = True
                # Lunch (11h30-14h)
                elif not has_lunch and 690 <= slot_start_minutes < 840 and duration >= 45:
                    proposed_meals.append(('🍽️ Déjeuner', f"{slot_start.hour:02d}:{slot_start.minute:02d}"))
                    has_lunch = True
                # Dinner (18h-21h)
                elif 1080 <= slot_start_minutes < 1260 and duration >= 45:
                    proposed_meals.append(('🍝 Dîner', f"{slot_start.hour:02d}:{slot_start.minute:02d}"))

        if proposed_meals:
            proposal_lines.append("**🍴 Repas (jours de semaine)**")
            # Sort meals by time
            meal_order = {'Petit-déjeuner': 0, 'Déjeuner': 1, 'Dîner': 2}
            proposed_meals.sort(key=lambda x: meal_order.get(x[0].split()[-1], 9))
            for meal_name, meal_time in proposed_meals:
                proposal_lines.append(f"  {meal_name}: {meal_time}")
            proposal_lines.append("")

        # Propose study/project time based on productivity preference
        peak_time = profile.peak_productivity_time or 'morning'

        # Find ALL study slots (not just during peak time)
        all_study_slots = []
        peak_study_slots = []

        for day in range(7):
            day_blocks = schedule_by_day.get(day, [])
            free_slots = find_free_slots(day_blocks)

            for slot_start, slot_end in free_slots:
                duration = (slot_end.hour * 60 + slot_end.minute) - (slot_start.hour * 60 + slot_start.minute)

                # Skip meal times and very short slots
                slot_start_minutes = slot_start.hour * 60 + slot_start.minute

                # Skip breakfast time (7h-8h30), lunch time (12h-13h30), dinner time (18h30-20h)
                is_meal_time = (420 <= slot_start_minutes < 510) or \
                               (720 <= slot_start_minutes < 810) or \
                               (1110 <= slot_start_minutes < 1200)

                if duration >= 60 and not is_meal_time:  # At least 1h
                    is_peak = False
                    if peak_time == 'morning' and slot_start.hour < 12:
                        is_peak = True
                    elif peak_time == 'afternoon' and 12 <= slot_start.hour < 18:
                        is_peak = True
                    elif peak_time == 'evening' and slot_start.hour >= 18:
                        is_peak = True

                    slot_info = (day, slot_start, slot_end, duration, is_peak)
                    all_study_slots.append(slot_info)
                    if is_peak:
                        peak_study_slots.append(slot_info)

        # Calculate total available study time
        total_available_minutes = sum(slot[3] for slot in all_study_slots)
        peak_available_minutes = sum(slot[3] for slot in peak_study_slots)

        proposal_lines.append(f"**📚 Travail personnel & Révisions (pic: {profile.get_peak_productivity_time_display()})**")
        proposal_lines.append(f"  📊 Temps disponible/semaine: ~{total_available_minutes // 60}h")

        if peak_available_minutes > 0:
            proposal_lines.append(f"  🎯 Créneaux optimaux (pic productivité): ~{peak_available_minutes // 60}h")

        proposal_lines.append("")

        # Show study slots by category
        # Helper function to break down large slots into reasonable chunks (max 3h)
        def chunk_slot(day, start, end, duration, max_duration=180):
            """Break a large slot into smaller chunks of max_duration minutes."""
            chunks = []
            current_start = start
            remaining = duration

            while remaining > 0:
                chunk_dur = min(remaining, max_duration)
                # Calculate chunk end time
                chunk_end_minutes = current_start.hour * 60 + current_start.minute + chunk_dur
                chunk_end = dt_time(chunk_end_minutes // 60, chunk_end_minutes % 60)

                chunks.append((day, current_start, chunk_end, chunk_dur))

                # Move to next chunk start (add 30min break between chunks)
                next_start_minutes = chunk_end_minutes + 30
                if next_start_minutes >= 24 * 60:
                    break
                current_start = dt_time(next_start_minutes // 60, next_start_minutes % 60)
                remaining = remaining - chunk_dur - 30

            return chunks

        # 1. Project work (deep work) - peak time slots, 2h+ preferred
        # Break down large slots into 2-3h chunks
        project_slots = []
        for d, s, e, dur, is_peak in all_study_slots:
            if is_peak and dur >= 90:
                if dur > 180:  # More than 3h -> chunk it
                    project_slots.extend(chunk_slot(d, s, e, dur, max_duration=150))  # 2h30 chunks
                else:
                    project_slots.append((d, s, e, dur))

        # Sort by duration (descending) and take best slots
        project_slots.sort(key=lambda x: -x[3])

        if project_slots:
            proposal_lines.append("  **💻 Projets (travail profond):**")
            total_project_hours = 0
            for day, start, end, duration in project_slots[:5]:  # Show up to 5 project slots
                hours = duration // 60
                mins = duration % 60
                dur_str = f"{hours}h{mins:02d}" if mins else f"{hours}h"
                proposal_lines.append(f"    {days_fr[day]}: {start.hour:02d}:{start.minute:02d} - {end.hour:02d}:{end.minute:02d} ({dur_str})")
                total_project_hours += duration / 60
            proposal_lines.append(f"    → Total projets: ~{int(total_project_hours)}h/semaine")
            proposal_lines.append("")

        # 2. Revision/homework slots - shorter slots (1h-2h)
        revision_slots = [(d, s, e, dur) for d, s, e, dur, is_peak in all_study_slots if 60 <= dur < 120]
        revision_slots.sort(key=lambda x: x[0])  # Sort by day

        if revision_slots:
            proposal_lines.append("  **📖 Révisions & Devoirs:**")
            total_revision_hours = 0
            for day, start, end, duration in revision_slots[:5]:
                hours = duration // 60
                mins = duration % 60
                dur_str = f"{hours}h{mins:02d}" if mins else f"{hours}h"
                proposal_lines.append(f"    {days_fr[day]}: {start.hour:02d}:{start.minute:02d} - {end.hour:02d}:{end.minute:02d} ({dur_str})")
                total_revision_hours += duration / 60
            proposal_lines.append(f"    → Total révisions: ~{int(total_revision_hours)}h/semaine")
            proposal_lines.append("")

        if not project_slots and not revision_slots:
            proposal_lines.append("  (créneaux à définir selon tes disponibilités)")
            proposal_lines.append("")

        # Summary
        proposal_lines.append("**📈 Résumé de la semaine:**")
        total_study = (sum(s[3] for s in project_slots[:5]) + sum(s[3] for s in revision_slots[:5])) / 60
        proposal_lines.append(f"  Temps d'étude proposé: ~{int(total_study)}h/semaine")

        proposal_lines.append("")
        proposal_lines.append("**Cette proposition est basée sur ton emploi du temps actuel.**")
        proposal_lines.append("Je peux créer ces blocs automatiquement ou les ajuster.")

        proposal_text = "\n".join(proposal_lines)

        # Build study proposals from project and revision slots
        study_proposals = []
        for day, start, end, duration in project_slots[:3]:
            study_proposals.append({
                'day': day,
                'start': f"{start.hour:02d}:{start.minute:02d}",
                'end': f"{end.hour:02d}:{end.minute:02d}",
                'type': 'project'
            })
        for day, start, end, duration in revision_slots[:3]:
            study_proposals.append({
                'day': day,
                'start': f"{start.hour:02d}:{start.minute:02d}",
                'end': f"{end.hour:02d}:{end.minute:02d}",
                'type': 'revision'
            })

        return {
            'text': proposal_text,
            'quick_replies': [
                {'label': "✅ Crée ce planning!", 'value': 'Oui, crée ce planning!'},
                {'label': "🔧 Ajuster", 'value': 'Je voudrais ajuster quelque chose'},
                {'label': "💬 J'ai des questions", 'value': "J'ai des questions sur cette proposition"},
            ],
            'proposed_schedule': {
                'sleep': {'bedtime': proposed_bed, 'wake': proposed_wake},
                'study_slots': study_proposals[:6] if study_proposals else [],
            }
        }
