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

from core.models import UploadedDocument, ConversationMessage
from services.llm import GeminiProvider, ClaudeProvider, LLMProvider
from services.security import SecurityService
from services.onboarding import OnboardingService
from services.planning import PlanningService

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

        # Step 3: Route to appropriate handler
        try:
            if not self.onboarding.is_onboarding_complete(user):
                response = self._handle_onboarding(user, sanitized_message, attachment)
            else:
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
        if not self.llm_provider.is_available():
            return self._fallback_response(message)

        # Build conversation context
        context = self._build_context(user)

        # Build prompt with tools
        prompt = self._build_chat_prompt(user, message, context)
        tools = self._get_function_tools()

        # Get LLM response
        response = self.llm_provider.generate(prompt, tools=tools)

        # Process function calls if any
        if response.has_function_calls:
            return self._execute_function_calls(user, response.function_calls, response.text)

        # Return text response
        return {
            'text': response.text or "Je suis là pour t'aider! Que veux-tu faire?",
            'quick_replies': self._get_contextual_quick_replies(user, message)
        }

    def _execute_function_calls(
        self,
        user: User,
        function_calls: list,
        ai_text: str
    ) -> dict:
        """Execute function calls from the LLM response."""
        response_text = ai_text
        blocks_created = []
        tasks_created = []

        for fc in function_calls:
            logger.info(f"Executing function: {fc.name} with args: {fc.args}")

            if fc.name == "create_recurring_block":
                result = self._execute_create_block(user, fc.args)
                if result:
                    blocks_created.extend(result)

            elif fc.name == "create_task":
                task = self._execute_create_task(user, fc.args)
                if task:
                    tasks_created.append(task)

            elif fc.name == "show_planning_proposal":
                proposal = self.planning.generate_proposal(user)
                response_text = proposal.get('text', '')

        # Build final response
        if blocks_created:
            block_names = [b.title for b in blocks_created]
            response_text += f"\n\n✅ Blocs créés: {', '.join(block_names)}"

        if tasks_created:
            task_names = [t.title for t in tasks_created]
            response_text += f"\n\n✅ Tâches créées: {', '.join(task_names)}"

        return {
            'text': response_text,
            'quick_replies': [],
            'blocks_created': [b.id for b in blocks_created],
            'tasks_created': [{'id': t.id, 'title': t.title} for t in tasks_created]
        }

    def _execute_create_block(self, user: User, args: dict) -> list:
        """Execute block creation with validation."""
        # Validate args
        valid, error, sanitized = self.security.validate_block_args(args)
        if not valid:
            logger.warning(f"Invalid block args: {error}")
            return []

        created = []
        for day in sanitized['days']:
            # Check day limits
            allowed, _ = self.security.validate_block_limits(user, day)
            if not allowed:
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

        return created

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

    def _build_context(self, user: User) -> dict:
        """Build context dictionary for the user."""
        from core.models import RecurringBlock, Task

        blocks_count = RecurringBlock.objects.filter(user=user, active=True).count()
        tasks_count = Task.objects.filter(user=user, completed=False).count()

        return {
            'blocks_count': blocks_count,
            'tasks_count': tasks_count,
            'productivity_time': user.profile.peak_productivity_time,
            'onboarding_completed': user.profile.onboarding_completed,
        }

    def _build_chat_prompt(self, user: User, message: str, context: dict) -> str:
        """Build the chat prompt for the LLM."""
        return f"""Tu es un assistant de planification intelligent et amical.

CONTEXTE UTILISATEUR:
- Blocs récurrents: {context['blocks_count']}
- Tâches en cours: {context['tasks_count']}
- Pic de productivité: {context['productivity_time']}

MESSAGE DE L'UTILISATEUR:
"{message}"

INSTRUCTIONS:
- Sois concis et naturel
- Si l'utilisateur veut créer un bloc ou une tâche, utilise les fonctions appropriées
- Si l'utilisateur pose une question, réponds directement
- Si c'est du small talk, réponds chaleureusement

Réponds en français."""

    def _get_function_tools(self) -> list:
        """Get the function tools for the LLM."""
        try:
            from google.genai import types
        except ImportError:
            return []

        return [types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="create_recurring_block",
                description="Créer un bloc récurrent dans le planning",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "title": types.Schema(type="STRING", description="Nom du bloc"),
                        "block_type": types.Schema(
                            type="STRING",
                            description="Type: course, work, sleep, meal, sport, project, other"
                        ),
                        "days": types.Schema(
                            type="ARRAY",
                            items=types.Schema(type="INTEGER"),
                            description="Jours (0=lundi, 6=dimanche)"
                        ),
                        "start_time": types.Schema(type="STRING", description="Heure début HH:MM"),
                        "end_time": types.Schema(type="STRING", description="Heure fin HH:MM"),
                    },
                    required=["title", "days", "start_time", "end_time"]
                )
            ),
            types.FunctionDeclaration(
                name="create_task",
                description="Créer une tâche à planifier",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "title": types.Schema(type="STRING", description="Nom de la tâche"),
                        "task_type": types.Schema(
                            type="STRING",
                            description="Type: deep_work, shallow, errand"
                        ),
                        "priority": types.Schema(type="INTEGER", description="Priorité 1-10"),
                    },
                    required=["title"]
                )
            ),
            types.FunctionDeclaration(
                name="show_planning_proposal",
                description="Afficher une proposition de planning",
                parameters=types.Schema(type="OBJECT", properties={}, required=[])
            ),
        ])]

    def _get_contextual_quick_replies(self, user: User, message: str) -> list:
        """Get contextual quick reply suggestions."""
        from core.models import RecurringBlock

        has_blocks = RecurringBlock.objects.filter(user=user, active=True).exists()

        if has_blocks:
            return [
                {'label': "📋 Voir mon planning", 'value': 'Montre-moi mon planning'},
                {'label': "➕ Ajouter un bloc", 'value': 'Je veux ajouter un bloc'},
            ]
        else:
            return [
                {'label': "📷 Envoyer un document", 'value': 'upload'},
                {'label': "💬 Décrire mes horaires", 'value': 'Je vais te décrire mon emploi du temps'},
            ]

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

    def _fallback_response(self, message: str) -> dict:
        """Fallback response when LLM is not available."""
        return {
            'text': "Je suis là pour t'aider! Que veux-tu faire avec ton planning?",
            'quick_replies': [
                {'label': "📷 Envoyer un document", 'value': 'upload'},
                {'label': "📋 Voir mon planning", 'value': 'Montre-moi mon planning'},
            ]
        }
