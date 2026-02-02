"""
Chat Orchestrator - Lightweight coordinator for chat operations.

This is the refactored version of ChatEngine that delegates to specialized services:
- SecurityService: Input validation, rate limiting
- OnboardingService: New user setup flow
- PlanningService: Schedule management
- LLMProvider: AI interactions

Usage:
    orchestrator = ChatOrchestrator()
    response = orchestrator.process_message(user, message, attachment)
"""
import logging
from typing import Optional

from django.contrib.auth.models import User
from django.conf import settings
from django.utils import timezone

from core.models import UploadedDocument, ConversationMessage
from services.llm import GeminiProvider, ClaudeProvider, LLMProvider
from services.security import SecurityService
from services.onboarding import OnboardingService
from services.planning import PlanningService
from services.chat_helpers import (
    analyze_session_context,
    is_casual_message,
    get_intelligent_fallback,
    generate_preference_radio_input,
)
from services.chat_planning import (
    generate_smart_planning_proposal,
    create_proposed_blocks,
    add_extracted_to_planning,
    check_block_overlap,
)

logger = logging.getLogger(__name__)


def get_llm_provider(provider_name: str = None) -> LLMProvider:
    """
    Get an LLM provider by name.

    Args:
        provider_name: 'gemini' or 'claude'. If None, uses settings.LLM_PROVIDER

    Returns:
        LLMProvider instance
    """
    if provider_name is None:
        provider_name = getattr(settings, 'LLM_PROVIDER', 'gemini')

    provider_name = provider_name.lower()

    if provider_name == 'claude':
        provider = ClaudeProvider()
        if provider.is_available():
            return provider
        logger.warning("Claude not available, falling back to Gemini")

    # Default to Gemini
    return GeminiProvider()


def get_user_llm_provider(user) -> LLMProvider:
    """
    Get the LLM provider based on user's preference.

    Args:
        user: Django User object

    Returns:
        LLMProvider instance based on user.profile.preferred_llm
    """
    try:
        preferred = user.profile.preferred_llm
        return get_llm_provider(preferred)
    except Exception:
        return get_llm_provider()


class ChatOrchestrator:
    """
    Lightweight orchestrator for chat interactions.

    Coordinates between specialized services to handle user messages.
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """
        Initialize the orchestrator with all required services.

        Args:
            llm_provider: Optional LLM provider override. If not provided,
                         uses the default from settings (LLM_PROVIDER).
        """
        # Initialize LLM provider (from arg, settings, or default)
        self.llm_provider = llm_provider or get_llm_provider()

        # Initialize specialized services
        self.security = SecurityService()
        self.planning = PlanningService(llm_provider=self.llm_provider)
        self.onboarding = OnboardingService(
            llm_provider=self.llm_provider,
            planning_service=self.planning
        )

        logger.info(f"ChatOrchestrator initialized (LLM: {self.llm_provider.name})")

    def generate_response(
        self,
        user: User,
        message: str,
        attachment: Optional[UploadedDocument] = None
    ) -> dict:
        """
        Generate a response - COMPATIBLE with ChatEngine API.

        This method provides backward compatibility with the old ChatEngine.
        Use this in views.py without changing any other code.

        Args:
            user: The authenticated user
            message: The user's message
            attachment: Optional uploaded document

        Returns:
            dict with 'response', 'quick_replies', 'tasks_created', etc.
        """
        result = self.process_message(user, message, attachment)

        # Transform to ChatEngine format
        return {
            'response': result.get('text', ''),
            'quick_replies': result.get('quick_replies', []),
            'tasks_created': result.get('tasks_created', []),
            'interactive_inputs': result.get('interactive_inputs', []),
            'extracted_data': result.get('extracted_data'),
        }

    def process_message(
        self,
        user: User,
        message: str,
        attachment: Optional[UploadedDocument] = None
    ) -> dict:
        """
        Process a user message and return a response.

        This is the main entry point for chat interactions.

        Args:
            user: The authenticated user
            message: The user's message
            attachment: Optional uploaded document

        Returns:
            dict with 'text', 'quick_replies', etc.
        """
        # Step 0: Use user's preferred LLM
        user_llm = get_user_llm_provider(user)
        if user_llm.name != self.llm_provider.name:
            self.llm_provider = user_llm
            self.planning.llm_provider = user_llm
            self.onboarding.llm_provider = user_llm
            logger.info(f"Switched to user's preferred LLM: {user_llm.name}")

        # Step 1: Security checks
        allowed, error_msg = self.security.check_rate_limit(user)
        if not allowed:
            return self._error_response(error_msg)

        sanitized_message, _ = self.security.sanitize_input(message)

        # Step 2: Save user message
        self._save_message(user, sanitized_message, 'user', attachment)

        # Step 3: Always use LLM-powered chat (like old ChatEngine)
        # The AI decides when to complete onboarding via function calls
        try:
            response = self._handle_regular_chat(user, sanitized_message, attachment)

            # Step 4: Save assistant response
            self._save_message(user, response.get('text', ''), 'assistant')

            return response

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return self._error_response("Désolé, une erreur s'est produite. Réessaie!")

    def _handle_onboarding(
        self,
        user: User,
        message: str,
        attachment: Optional[UploadedDocument]
    ) -> dict:
        """Handle messages during onboarding flow."""
        return self.onboarding.handle_message(user, message, attachment)

    def _handle_regular_chat(
        self,
        user: User,
        message: str,
        attachment: Optional[UploadedDocument]
    ) -> dict:
        """
        Handle regular chat messages (post-onboarding).

        Uses the LLM with function calling for complex interactions.
        """
        # Handle document upload first if there's an attachment
        if attachment:
            return self._handle_document_upload(user, attachment)

        # Check for recently processed documents the user hasn't seen
        recent_doc = self._check_recently_processed_document(user)
        if recent_doc:
            return self._handle_document_upload(user, recent_doc)

        if not self.llm_provider.is_available():
            logger.warning("LLM provider not available, using fallback")
            return self._fallback_response(message)

        # Build conversation context
        context = self._build_context(user)
        logger.debug(f"Context built: blocks={context['blocks_count']}, tasks={context['tasks_count']}")

        # Get conversation history
        history = self._get_conversation_history(user)
        logger.debug(f"Conversation history: {len(history)} messages")

        # Analyze session context for better understanding
        session_context = self._analyze_session_context(history, message)
        logger.debug(f"Session context: topic={session_context.get('last_topic')}, mood={session_context.get('mood')}, pending={session_context.get('pending_action')}")

        # Build system prompt with session context
        system_prompt = self._build_system_prompt(user, context, session_context)
        tools = self._get_function_tools()
        logger.debug(f"System prompt length: {len(system_prompt)} chars, tools: {len(tools)} tools")

        # Add current message to history
        history.append({"role": "user", "content": message})

        # Get LLM response with history
        try:
            response = self.llm_provider.generate_with_history(
                messages=history,
                tools=tools,
                system_prompt=system_prompt
            )
            logger.info(f"LLM response: text_length={len(response.text or '')}, function_calls={len(response.function_calls)}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return self._fallback_response(message)

        # Process function calls if any
        if response.has_function_calls:
            logger.info(f"Processing {len(response.function_calls)} function calls: {[fc.name for fc in response.function_calls]}")
            return self._execute_function_calls(user, response.function_calls, response.text, message)

        # Check if this is a casual conversation (no quick replies needed)
        is_casual = self._is_casual_message(message)
        logger.debug(f"Message casual: {is_casual}")

        # Return text response
        response_text = response.text or self._get_intelligent_fallback(user, message, [])
        if not response.text:
            logger.warning(f"LLM returned empty text, using fallback for message: '{message[:50]}...'")

        return {
            'text': response_text,
            'quick_replies': [] if is_casual else self._get_contextual_quick_replies(user, message, response_text)
        }

    def _execute_function_calls(
        self,
        user: User,
        function_calls: list,
        ai_text: str,
        original_message: str = ""
    ) -> dict:
        """Execute function calls from the LLM response."""
        response_text = ai_text
        blocks_created = []
        tasks_created = []
        skipped_days_all = []
        interactive_inputs = []
        preferences_updated = []
        function_calls_made = []

        for fc in function_calls:
            logger.info(f"Executing function: {fc.name} with args: {fc.args}")
            function_calls_made.append(fc.name)

            if fc.name == "create_recurring_block":
                created, skipped = self._execute_create_block(user, fc.args)
                if created:
                    blocks_created.extend(created)
                if skipped:
                    skipped_days_all.extend(skipped)

            elif fc.name == "create_task":
                task = self._execute_create_task(user, fc.args)
                if task:
                    tasks_created.append(task)

            elif fc.name == "show_planning_proposal":
                proposal = self._generate_smart_planning_proposal(user)
                response_text = proposal.get('text', '')

            elif fc.name == "update_preference":
                pref_name = fc.args.get('preference', '')
                pref_value = fc.args.get('value', '')
                if self._execute_update_preference(user, fc.args):
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

            elif fc.name == "complete_onboarding":
                profile = user.profile
                profile.onboarding_completed = True
                profile.onboarding_step = 3
                profile.save()
                if not response_text:
                    response_text = "✅ Configuration terminée! Ton planning est prêt."

            elif fc.name == "ask_preference_confirmation":
                # Generate interactive radio inputs for preference confirmation
                pref_type = fc.args.get('preference_type', '')
                detected_value = fc.args.get('detected_value', '')
                question = fc.args.get('question', 'Confirme cette information:')

                interactive_input = self._generate_preference_radio_input(pref_type, detected_value, question)
                if interactive_input:
                    return {
                        'text': question,
                        'quick_replies': [],
                        'tasks_created': [],
                        'interactive_inputs': [interactive_input]
                    }

            elif fc.name == "ask_clarification":
                # Generate clarification radio inputs
                question = fc.args.get('question', "Je n'ai pas bien compris. Tu voulais:")
                options = fc.args.get('options', [])
                allow_other = fc.args.get('allow_other', True)

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

            elif fc.name == "accept_planning_proposal":
                created_blocks = self._create_proposed_blocks(user)
                blocks_count = len(created_blocks) if created_blocks else 0
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

        # Build final response
        if blocks_created:
            day_names = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']
            # Group blocks by title and time
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
            if response_text:
                response_text += f"\n\n{confirmation}"
            else:
                response_text = confirmation

        # Inform about skipped days
        if skipped_days_all:
            unique_skipped = list(set(skipped_days_all))
            response_text += f"\n\n⚠️ Certains jours n'ont pas été ajoutés car il y a déjà des blocs sur cette plage horaire: {', '.join(unique_skipped)}"

        if tasks_created:
            task_names = [t.title for t in tasks_created]
            if response_text:
                response_text += f"\n\n📝 Tâche(s) créée(s): {', '.join(task_names)}"
            else:
                response_text = f"📝 J'ai créé {len(tasks_created)} tâche(s): {', '.join(task_names)}"

        # Use intelligent fallback if no response text
        if not response_text:
            response_text = self._get_intelligent_fallback(user, original_message, function_calls_made)

        # Check if this is a casual conversation (no quick replies needed)
        is_casual = self._is_casual_message(original_message)

        # Build contextual quick replies (skip for casual conversation without actions)
        if is_casual and not blocks_created and not tasks_created and not preferences_updated:
            quick_replies = []  # Natural conversation without buttons
        else:
            quick_replies = self._get_contextual_quick_replies(
                user, original_message, response_text,
                blocks_created=blocks_created,
                tasks_created=tasks_created,
                preferences_updated=preferences_updated
            )

        return {
            'text': response_text,
            'quick_replies': quick_replies,
            'blocks_created': [b.id for b in blocks_created],
            'tasks_created': [{'id': t.id, 'title': t.title} for t in tasks_created],
            'interactive_inputs': interactive_inputs
        }

    def _execute_create_block(self, user: User, args: dict) -> tuple:
        """Execute block creation with validation and overlap detection."""
        # Validate args
        valid, error, sanitized = self.security.validate_block_args(args)
        if not valid:
            logger.warning(f"Invalid block args: {error}")
            return [], []

        created = []
        skipped_days = []
        day_names = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']

        for day in sanitized['days']:
            # Check day limits
            allowed, _ = self.security.validate_block_limits(user, day)
            if not allowed:
                skipped_days.append(day_names[day])
                continue

            # Check for overlaps
            has_overlap, overlapping_title = self._check_block_overlap(
                user, day, sanitized['start_time'], sanitized['end_time']
            )
            if has_overlap:
                logger.info(f"Skipping day {day} due to overlap with '{overlapping_title}'")
                skipped_days.append(day_names[day])
                continue

            block = self.planning.create_block(
                user=user,
                title=sanitized['title'],
                block_type=sanitized['block_type'],
                day=day,
                start_time=sanitized['start_time'],
                end_time=sanitized['end_time'],
                location=sanitized.get('location', ''),
                is_night_shift=sanitized.get('is_night_shift', False)
            )
            if block:
                created.append(block)

        return created, skipped_days

    def _execute_create_task(self, user: User, args: dict) -> Optional:
        """Execute task creation with validation."""
        # Validate limits
        allowed, error = self.security.validate_task_limits(user)
        if not allowed:
            logger.warning(f"Task limit reached for user {user.id}")
            return None

        # Validate args
        valid, error, sanitized = self.security.validate_task_args(args)
        if not valid:
            logger.warning(f"Invalid task args: {error}")
            return None

        return self.planning.create_task(
            user=user,
            title=sanitized['title'],
            task_type=sanitized['task_type'],
            priority=sanitized['priority'],
            description=sanitized.get('description', ''),
            deadline=sanitized.get('deadline'),
            estimated_duration_minutes=sanitized.get('estimated_duration_minutes')
        )

    def _check_recently_processed_document(self, user: User) -> Optional[UploadedDocument]:
        """Check for recently processed documents the user hasn't been notified about."""
        from datetime import timedelta

        # Find documents processed in the last 5 minutes that haven't been acknowledged
        recent_cutoff = timezone.now() - timedelta(minutes=5)

        recent_doc = UploadedDocument.objects.filter(
            user=user,
            processed=True,
            uploaded_at__gte=recent_cutoff,
        ).exclude(
            extracted_data__isnull=True
        ).order_by('-uploaded_at').first()

        if recent_doc and recent_doc.extracted_data:
            # Check if we already notified about this document
            from core.models import ConversationMessage
            recent_messages = ConversationMessage.objects.filter(
                user=user,
                created_at__gte=recent_cutoff,
                content__icontains='document analysé'
            ).exists()

            if not recent_messages:
                return recent_doc

        return None

    def _handle_document_upload(self, _user: User, attachment: UploadedDocument) -> dict:
        """Handle document upload and show extracted data."""
        from core.models import RecurringBlock

        if not attachment.processed:
            error = getattr(attachment, 'processing_error', None)
            if error:
                return {
                    'text': f"Désolé, je n'ai pas pu analyser ce document: {error}\n\nEssaie avec un autre fichier ou décris-moi ton emploi du temps.",
                    'quick_replies': [
                        {'label': "📝 Décrire mes horaires", 'value': "Je vais te décrire mon emploi du temps"},
                    ]
                }
            return {
                'text': "Le document est en cours de traitement. Réessaie dans quelques secondes.",
                'quick_replies': []
            }

        extracted = attachment.extracted_data or {}
        summary_lines = []
        total_items = 0

        # Courses
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

        # Work shifts
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

        # Events
        if 'events' in extracted and extracted['events']:
            count = len(extracted['events'])
            total_items += count
            summary_lines.append(f"\n📅 **{count} événements détectés:**")
            for event in extracted['events'][:5]:
                title = event.get('title', 'Événement')
                day = event.get('day', '')
                summary_lines.append(f"  • {title} ({day})" if day else f"  • {title}")
            if count > 5:
                summary_lines.append(f"  ... et {count - 5} autres")

        if summary_lines:
            summary = "\n".join(summary_lines)
            blocks_created = RecurringBlock.objects.filter(source_document=attachment).count()

            if blocks_created > 0:
                text = f"✅ Document analysé! {total_items} éléments extraits et {blocks_created} blocs créés.\n\n{summary}\n\nJe peux te proposer un planning optimisé!"
                quick_replies = [
                    {'label': "📋 Voir la proposition", 'value': 'Montre-moi la proposition de planning'},
                    {'label': "📅 Voir mon planning", 'value': 'Montre-moi mon planning'},
                ]
            else:
                text = f"✅ Document analysé! Voici ce que j'ai trouvé:\n\n{summary}\n\nVeux-tu que j'ajoute ces éléments à ton planning?"
                quick_replies = [
                    {'label': "✅ Ajouter au planning", 'value': 'Ajoute ces données à mon planning'},
                    {'label': "📋 Voir le planning actuel", 'value': 'Montre-moi mon planning'},
                ]

            return {'text': text, 'quick_replies': quick_replies}

        return {
            'text': "J'ai analysé le document mais je n'ai pas trouvé d'emploi du temps. Peux-tu me décrire tes horaires?",
            'quick_replies': [
                {'label': "📝 Décrire mes horaires", 'value': "Je vais te décrire mon emploi du temps"},
            ]
        }

    def _execute_update_preference(self, user: User, args: dict) -> bool:
        """Execute preference update."""
        preference = args.get('preference', '')
        value = args.get('value', '')

        if not preference or not value:
            return False

        profile = user.profile

        try:
            if preference == 'peak_productivity_time':
                if value in ['morning', 'afternoon', 'evening']:
                    profile.peak_productivity_time = value
            elif preference == 'min_sleep_hours':
                profile.min_sleep_hours = int(value)
            elif preference == 'transport_time_minutes':
                profile.transport_time_minutes = int(value)
            elif preference == 'max_deep_work_hours_per_day':
                profile.max_deep_work_hours_per_day = int(value)

            profile.save()
            logger.info(f"Updated preference {preference}={value} for user {user.id}")
            return True
        except (ValueError, TypeError) as e:
            logger.error(f"Error updating preference: {e}")
            return False

    def _build_context(self, user: User) -> dict:
        """Build context dictionary for the user."""
        from core.models import RecurringBlock, Task

        blocks_count = RecurringBlock.objects.filter(user=user, active=True).count()
        tasks_count = Task.objects.filter(user=user, completed=False).count()

        # Get recent blocks summary
        blocks_summary = ""
        if blocks_count > 0:
            blocks = RecurringBlock.objects.filter(user=user, active=True).order_by('day_of_week', 'start_time')[:10]
            day_names = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']
            blocks_summary = "\n".join([
                f"  - {b.title} ({day_names[b.day_of_week]} {b.start_time.strftime('%H:%M')}-{b.end_time.strftime('%H:%M')})"
                for b in blocks
            ])

        return {
            'blocks_count': blocks_count,
            'tasks_count': tasks_count,
            'blocks_summary': blocks_summary,
            'productivity_time': user.profile.peak_productivity_time,
            'onboarding_completed': user.profile.onboarding_completed,
        }

    def _get_conversation_history(self, user: User, limit: int = 10) -> list:
        """Get recent conversation history for context."""
        messages = ConversationMessage.objects.filter(user=user).order_by('-created_at')[:limit]

        # Convert to list and reverse to get chronological order
        history = []
        for msg in reversed(messages):
            history.append({
                "role": msg.role,
                "content": msg.content
            })

        return history

    def _build_system_prompt(self, user: User, context: dict, session_context: dict = None) -> str:
        """Build the system prompt for the LLM."""
        profile = user.profile
        session_context = session_context or {}

        # Build full schedule by day
        schedule_text = ""
        if context['blocks_summary']:
            schedule_text = f"\n{context['blocks_summary']}"
        else:
            schedule_text = "\nAucun bloc récurrent configuré."

        # Tasks summary
        tasks_text = ""
        if context['tasks_count'] > 0:
            from core.models import Task
            pending_tasks = Task.objects.filter(user=user, completed=False).order_by('deadline', '-priority')[:5]
            if pending_tasks:
                tasks_text = "\n\nTÂCHES EN ATTENTE:"
                for task in pending_tasks:
                    deadline = f" (deadline: {task.deadline.strftime('%d/%m')})" if task.deadline else ""
                    tasks_text += f"\n- {task.title}{deadline} [priorité {task.priority}/10]"

        onboarding_status = "terminé" if profile.onboarding_completed else f"en cours (étape {profile.onboarding_step})"

        # Build session context section
        session_section = f"""
CONTEXTE DE SESSION:
- Dernier sujet: {session_context.get('last_topic', 'aucun')}
- Action en attente: {session_context.get('pending_action', 'aucune')}
- Humeur conversation: {session_context.get('mood', 'neutre')}
- Clarification demandée: {'oui' if session_context.get('clarification_asked') else 'non'}
- Préférence en discussion: {session_context.get('preference_being_discussed', 'aucune')}"""

        return f"""Tu es un assistant de planification intelligent pour Planner AI. Tu aides {user.first_name or user.username} à organiser son temps.

PROFIL UTILISATEUR:
- Sommeil minimum: {profile.min_sleep_hours}h
- Pic de productivité: {profile.get_peak_productivity_time_display()}
- Temps de transport: {profile.transport_time_minutes} min
- Onboarding: {onboarding_status}

PLANNING ACTUEL:{schedule_text}{tasks_text}
{session_section}

DATE: {timezone.now().strftime('%d/%m/%Y %H:%M')}

══════════════════════════════════════════════════════════════
CLASSIFICATION DES MESSAGES - ANALYSE D'ABORD LE TYPE:
══════════════════════════════════════════════════════════════

1. CASUAL (salut, ça va, merci, ok, cool, etc.)
   → Réponds naturellement et chaleureusement SANS appeler de fonction
   → Exemples: "Salut! Comment ça va?", "De rien, avec plaisir!"

2. PREFERENCE (temps trajet, sommeil, productivité, etc.)
   → UTILISE ask_preference_confirmation avec options radio
   → "20 min de route" → ask_preference_confirmation(preference_type="transport_time", detected_value="20", question="Je note 20 minutes de trajet. C'est bien ça?")
   → "je dors 7h" → ask_preference_confirmation(preference_type="sleep_hours", detected_value="7", question="...")

3. ACTION (créer bloc, ajouter tâche, j'ai cours, je travaille, etc.)
   → Exécute l'action avec create_recurring_block ou create_task
   → "je travaille de 19h à 7h vendredi et samedi" → create_recurring_block

4. QUESTION (c'est quoi mon planning, j'ai quoi demain, comment ça marche)
   → Réponds avec les infos du contexte ci-dessus
   → Utilise show_planning_proposal si pertinent

5. CONFIRMATION (oui, d'accord, c'est bon, valide, confirme)
   → Si confirmation d'une préférence → update_preference
   → Si confirmation de planning → accept_planning_proposal
   → Si validation globale → complete_onboarding

6. HORS_SUJET (météo, capitale, recette, etc.)
   → "Je suis spécialisé dans la planification! Je peux t'aider à organiser ton emploi du temps, ajouter des blocs, gérer tes tâches..."

7. AMBIGU (pas clair ce que l'utilisateur veut)
   → UTILISE ask_clarification pour proposer des options
   → ask_clarification(question="Je n'ai pas bien compris. Tu voulais:", options=["Ajouter un bloc à ton planning", "Modifier une préférence", "Voir ton planning"], allow_other=true)

══════════════════════════════════════════════════════════════
UTILISE LE CONTEXTE DE SESSION:
══════════════════════════════════════════════════════════════
- Si "confirmation_attendue" → L'utilisateur répond probablement à ta question précédente
- Si "réponse_attendue" → Interprète la réponse dans le contexte de ta question
- Si "clarification_asked" → L'utilisateur clarifie sa demande précédente
- Si "last_topic" est défini → Le sujet est probablement lié à ce thème
- Si "mood: frustré" → Sois plus empathique et propose de l'aide
- Si "mood: positif" → Continue sur cette lancée positive

══════════════════════════════════════════════════════════════
RÈGLES IMPORTANTES:
══════════════════════════════════════════════════════════════
- Réponds en français, sois amical et naturel
- Quand l'utilisateur décrit ses horaires → crée les blocs IMMÉDIATEMENT avec create_recurring_block
- Pour PREFERENCE → TOUJOURS ask_preference_confirmation d'abord
- Pour AMBIGU → UTILISE ask_clarification avec options
- Pour HORS_SUJET → Explique poliment ton domaine (planification)
- JAMAIS répondre "Comment puis-je t'aider?" de façon robotique
- Sois concis (2-3 phrases max)"""

    def _get_function_tools(self) -> list:
        """Get the function tools for the LLM."""
        try:
            from google.genai import types
        except ImportError:
            return []

        return [types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="create_recurring_block",
                description="Créer un ou plusieurs blocs récurrents (cours, travail, sport, sommeil, repas, etc.). Utilise cette fonction dès que l'utilisateur mentionne ses horaires.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "title": types.Schema(type="STRING", description="Titre du bloc"),
                        "block_type": types.Schema(
                            type="STRING",
                            description="Type: course|work|sleep|meal|sport|project|revision|other"
                        ),
                        "days": types.Schema(
                            type="ARRAY",
                            items=types.Schema(type="INTEGER"),
                            description="Jours de la semaine (0=Lundi, 6=Dimanche). Ex: [0,1,2,3,4] pour lun-ven"
                        ),
                        "start_time": types.Schema(type="STRING", description="Heure de début (HH:MM)"),
                        "end_time": types.Schema(type="STRING", description="Heure de fin (HH:MM)"),
                        "location": types.Schema(type="STRING", description="Lieu (optionnel)"),
                        "is_night_shift": types.Schema(type="BOOLEAN", description="True si le bloc passe minuit (ex: 22h-6h)"),
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
                        "description": types.Schema(type="STRING", description="Description (optionnel)"),
                        "deadline": types.Schema(type="STRING", description="Deadline format YYYY-MM-DD (optionnel)"),
                        "priority": types.Schema(type="INTEGER", description="Priorité 1-10 (défaut: 5)"),
                        "task_type": types.Schema(type="STRING", description="Type: deep_work|shallow|errand"),
                        "estimated_duration_minutes": types.Schema(type="INTEGER", description="Durée estimée en minutes"),
                    },
                    required=["title"]
                )
            ),
            types.FunctionDeclaration(
                name="update_preference",
                description="Mettre à jour une préférence utilisateur",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "preference": types.Schema(
                            type="STRING",
                            description="Nom: peak_productivity_time|min_sleep_hours|transport_time_minutes"
                        ),
                        "value": types.Schema(type="STRING", description="Nouvelle valeur"),
                    },
                    required=["preference", "value"]
                )
            ),
            types.FunctionDeclaration(
                name="show_planning_proposal",
                description="Générer et afficher une proposition de planning intelligent basée sur les disponibilités de l'utilisateur.",
                parameters=types.Schema(type="OBJECT", properties={}, required=[])
            ),
            types.FunctionDeclaration(
                name="complete_onboarding",
                description="Marquer l'onboarding comme terminé. Utilise cette fonction quand l'utilisateur valide sa configuration ou dit que c'est bon.",
                parameters=types.Schema(type="OBJECT", properties={}, required=[])
            ),
            types.FunctionDeclaration(
                name="ask_preference_confirmation",
                description="Demander confirmation à l'utilisateur pour une préférence détectée (temps de trajet, heures de sommeil, etc.). Affiche des options radio pour confirmer/modifier.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "preference_type": types.Schema(
                            type="STRING",
                            description="Type: transport_time|sleep_hours|productivity_time|deep_work_hours"
                        ),
                        "detected_value": types.Schema(
                            type="STRING",
                            description="Valeur détectée dans le message (ex: '20' pour 20 min)"
                        ),
                        "question": types.Schema(
                            type="STRING",
                            description="Question à poser à l'utilisateur"
                        ),
                    },
                    required=["preference_type", "detected_value", "question"]
                )
            ),
            types.FunctionDeclaration(
                name="ask_clarification",
                description="Demander une clarification quand le message est ambigu. Affiche des options radio pour guider l'utilisateur.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "question": types.Schema(type="STRING", description="Question de clarification"),
                        "options": types.Schema(
                            type="ARRAY",
                            items=types.Schema(type="STRING"),
                            description="Liste des options possibles (2-4 options)"
                        ),
                        "allow_other": types.Schema(
                            type="BOOLEAN",
                            description="Permettre une réponse personnalisée"
                        ),
                    },
                    required=["question", "options"]
                )
            ),
            types.FunctionDeclaration(
                name="accept_planning_proposal",
                description="Accepter la proposition de planning et créer tous les blocs proposés. Utilise quand l'utilisateur dit 'oui', 'crée ce planning', 'valide', etc.",
                parameters=types.Schema(type="OBJECT", properties={}, required=[])
            ),
            types.FunctionDeclaration(
                name="add_document_data",
                description="Ajouter les données extraites du dernier document uploadé au planning.",
                parameters=types.Schema(type="OBJECT", properties={}, required=[])
            ),
        ])]

    def _get_contextual_quick_replies(
        self,
        user: User,
        _message: str,
        response_text: str = "",
        blocks_created: list = None,
        tasks_created: list = None,
        preferences_updated: list = None
    ) -> list:
        """
        Get contextual quick reply suggestions based on conversation state.

        Args:
            user: The current user
            _message: The user's message (unused but kept for API compatibility)
            response_text: The assistant's response text
            blocks_created: List of blocks just created (if any)
            tasks_created: List of tasks just created (if any)
            preferences_updated: List of preferences just updated (if any)

        Returns:
            list: Contextual quick reply buttons (max 3)
        """
        from core.models import RecurringBlock, Task

        quick_replies = []
        blocks_created = blocks_created or []
        tasks_created = tasks_created or []
        preferences_updated = preferences_updated or []

        # Get user state
        pending_tasks = Task.objects.filter(user=user, completed=False).count()
        total_blocks = RecurringBlock.objects.filter(user=user, active=True).count()
        today = timezone.now()
        is_weekend = today.weekday() >= 5

        response_lower = response_text.lower() if response_text else ""

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

        # If bot is asking for schedule info
        elif any(word in response_lower for word in ['horaires', 'emploi du temps', 'contraintes']):
            quick_replies.append({'label': "💬 Décrire mes horaires", 'value': "Je vais te décrire mon emploi du temps"})
            quick_replies.append({'label': "📎 Joindre un fichier", 'value': "Je veux joindre un document avec mes horaires"})

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

            # Always offer to add something if no blocks yet
            if total_blocks == 0:
                quick_replies.append({'label': "💬 Décrire mes horaires", 'value': "Je vais te décrire mon emploi du temps"})
                quick_replies.append({'label': "📎 Joindre un fichier", 'value': "Je veux joindre un document avec mes horaires"})
            elif len(quick_replies) < 2:
                quick_replies.append({'label': "➕ Ajouter", 'value': "Je veux ajouter quelque chose à mon planning"})

        # Limit to 3 buttons max
        return quick_replies[:3]

    def _save_message(
        self,
        user: User,
        content: str,
        role: str,
        attachment: Optional[UploadedDocument] = None
    ) -> ConversationMessage:
        """Save a message to the conversation history."""
        return ConversationMessage.objects.create(
            user=user,
            role=role,
            content=content,
            attachment=attachment
        )

    def _error_response(self, message: str) -> dict:
        """Return a standardized error response."""
        return {
            'text': message,
            'quick_replies': [],
            'error': True
        }

    def _fallback_response(self, _message: str) -> dict:
        """Fallback response when LLM is not available."""
        return {
            'text': "Je suis là pour t'aider! Que veux-tu faire avec ton planning?",
            'quick_replies': [
                {'label': "📷 Envoyer un document", 'value': 'upload'},
                {'label': "📋 Voir mon planning", 'value': 'Montre-moi mon planning'},
            ]
        }

    # ==================== WRAPPER METHODS FOR HELPER FUNCTIONS ====================

    def _analyze_session_context(self, history: list, current_message: str) -> dict:
        """Analyze conversation history to build session context."""
        return analyze_session_context(history, current_message, is_casual_message)

    def _is_casual_message(self, message: str) -> bool:
        """Check if a message is casual conversation."""
        return is_casual_message(message)

    def _get_intelligent_fallback(self, user: User, message: str, function_calls_made: list = None) -> str:
        """Generate an intelligent fallback response based on context."""
        return get_intelligent_fallback(user, message, function_calls_made)

    def _generate_preference_radio_input(self, pref_type: str, detected_value: str, question: str):
        """Generate a radio input configuration for preference confirmation."""
        return generate_preference_radio_input(pref_type, detected_value, question)

    def _generate_smart_planning_proposal(self, user: User, overrides: dict = None) -> dict:
        """Analyze user's schedule and generate an intelligent planning proposal."""
        return generate_smart_planning_proposal(user, overrides)

    def _create_proposed_blocks(self, user: User) -> list:
        """Create the proposed blocks (sleep, meals) in the database."""
        return create_proposed_blocks(user)

    def _add_extracted_to_planning(self, user: User) -> dict:
        """Add extracted data from the most recent document to the planning."""
        return add_extracted_to_planning(user)

    def _check_block_overlap(self, user: User, day: int, start_time, end_time) -> tuple:
        """Check if a block would overlap with existing blocks."""
        return check_block_overlap(user, day, start_time, end_time)
