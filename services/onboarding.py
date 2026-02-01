"""
Onboarding service for new user setup flow.

Handles the guided conversation to collect user preferences
and set up their initial planning configuration.
"""
import logging
from typing import Optional

from django.contrib.auth.models import User

from core.models import (
    UserProfile,
    UploadedDocument,
    ConversationMessage,
    RecurringBlock,
)

logger = logging.getLogger(__name__)


class OnboardingService:
    """
    Manages the user onboarding conversation flow.

    Steps:
    - Step 0: Welcome + upload schedule OR describe constraints
    - Step 1: Confirm extracted data + ask productivity preference
    - Step 2: Show proposal, handle feedback, complete onboarding
    """

    STEPS = [
        "upload_schedule",   # Step 0
        "confirm_schedule",  # Step 1
        "preferences",       # Step 2
        "completed",         # Step 3
    ]

    PRODUCTIVITY_KEYWORDS = {
        'matin': 'morning',
        'morning': 'morning',
        'après-midi': 'afternoon',
        'apres-midi': 'afternoon',
        'afternoon': 'afternoon',
        'soir': 'evening',
        'evening': 'evening',
        'nuit': 'evening',
    }

    def __init__(self, llm_provider=None, planning_service=None):
        """
        Initialize the onboarding service.

        Args:
            llm_provider: LLM provider for AI conversations
            planning_service: Service for generating planning proposals
        """
        self.llm_provider = llm_provider
        self.planning_service = planning_service

    def is_onboarding_complete(self, user: User) -> bool:
        """Check if user has completed onboarding."""
        try:
            return user.profile.onboarding_completed
        except UserProfile.DoesNotExist:
            return False

    def get_current_step(self, user: User) -> int:
        """Get the current onboarding step for a user."""
        try:
            return user.profile.onboarding_step
        except UserProfile.DoesNotExist:
            return 0

    def handle_message(
        self,
        user: User,
        message: str,
        attachment: Optional[UploadedDocument] = None
    ) -> dict:
        """
        Handle a message during onboarding.

        Args:
            user: The current user
            message: The user's message
            attachment: Optional uploaded document

        Returns:
            dict: Response with 'text' and optional 'quick_replies'
        """
        profile = user.profile
        step = profile.onboarding_step
        message_lower = message.lower().strip()

        if step == 0:
            return self._handle_step_0(user, message, message_lower, attachment, profile)
        elif step == 1:
            return self._handle_step_1(user, message, message_lower, attachment, profile)
        elif step == 2:
            return self._handle_step_2(user, message, message_lower, profile)

        return self._default_response()

    def _handle_step_0(
        self,
        user: User,
        message: str,
        message_lower: str,
        attachment: Optional[UploadedDocument],
        profile: UserProfile
    ) -> dict:
        """Handle step 0: Welcome + Upload."""
        # Document uploaded
        if attachment:
            return self._process_document(user, attachment, profile)

        # Count messages to detect first interaction
        message_count = ConversationMessage.objects.filter(
            user=user, role='user'
        ).count()

        if message_count <= 1:
            return self._welcome_response()

        if 'skip' in message_lower or 'passer' in message_lower:
            return self._skip_to_preferences(profile)

        if 'upload' in message_lower:
            return {
                'text': "Envoie-moi ton emploi du temps en photo ou PDF!",
                'quick_replies': []
            }

        # User is describing schedule - delegate to LLM
        return self._handle_free_conversation(user, message, profile)

    def _handle_step_1(
        self,
        user: User,
        message: str,
        message_lower: str,
        attachment: Optional[UploadedDocument],
        profile: UserProfile
    ) -> dict:
        """Handle step 1: Confirm data + productivity preference."""
        if attachment:
            return self._process_document(user, attachment, profile)

        # User confirms
        confirmation_words = ['oui', 'correct', 'ok', "c'est bon", 'yes']
        if any(word in message_lower for word in confirmation_words):
            return self._ask_productivity_preference(profile)

        # User wants to modify
        modification_words = ['modifier', 'changer', 'corriger', 'non', 'pas correct']
        if any(word in message_lower for word in modification_words):
            return {
                'text': "Qu'est-ce qui n'est pas correct? Décris-moi les modifications à faire.",
                'quick_replies': [
                    {'label': "📷 Renvoyer le document", 'value': 'upload'},
                ]
            }

        # Free conversation
        return self._handle_free_conversation(user, message, profile)

    def _handle_step_2(
        self,
        user: User,
        message: str,
        message_lower: str,
        profile: UserProfile
    ) -> dict:
        """Handle step 2: Proposal + finish."""
        # Check for productivity preference
        for keyword, value in self.PRODUCTIVITY_KEYWORDS.items():
            if keyword in message_lower:
                profile.peak_productivity_time = value
                profile.save()
                return self._generate_proposal(user)

        # Check if responding to a proposal
        recent_messages = ConversationMessage.objects.filter(
            user=user
        ).order_by('-created_at')[:3]

        has_recent_proposal = any(
            'proposition de planning' in msg.content.lower()
            for msg in recent_messages
            if msg.role == 'assistant'
        )

        if has_recent_proposal:
            return self._handle_proposal_feedback(user, message, list(recent_messages))

        # Show proposal if user has blocks
        if RecurringBlock.objects.filter(user=user, active=True).exists():
            return self._generate_proposal(user)

        return self._handle_free_conversation(user, message, profile)

    def _welcome_response(self) -> dict:
        """Return the welcome message for new users."""
        return {
            'text': "Salut! 👋 Je suis ton assistant de planification.\n\nEnvoie-moi ton emploi du temps (cours ou travail) en photo/PDF, ou décris-moi simplement tes contraintes!",
            'quick_replies': [
                {'label': "📷 J'uploade mon emploi du temps", 'value': 'upload'},
                {'label': "💬 Je préfère décrire", 'value': 'Je vais te décrire mon emploi du temps'},
                {'label': "⏭️ Passer", 'value': 'skip'},
            ]
        }

    def _skip_to_preferences(self, profile: UserProfile) -> dict:
        """Skip upload and go to productivity preferences."""
        profile.onboarding_step = 2
        profile.save()
        return self._productivity_question()

    def _ask_productivity_preference(self, profile: UserProfile) -> dict:
        """Ask about productivity preference and advance to step 2."""
        profile.onboarding_step = 2
        profile.save()
        return self._productivity_question()

    def _productivity_question(self) -> dict:
        """Return the productivity preference question."""
        return {
            'text': "Super! Maintenant, dis-moi: à quel moment de la journée es-tu le plus productif pour travailler/étudier?",
            'quick_replies': [
                {'label': "🌅 Le matin", 'value': 'Je suis plus productif le matin'},
                {'label': "☀️ L'après-midi", 'value': "Je suis plus productif l'après-midi"},
                {'label': "🌙 Le soir", 'value': 'Je suis plus productif le soir'},
            ]
        }

    def _process_document(
        self,
        user: User,
        attachment: UploadedDocument,
        profile: UserProfile
    ) -> dict:
        """Process an uploaded document during onboarding."""
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

        if not summary_lines:
            return "Aucune donnée structurée trouvée dans le document."

        return "\n".join(summary_lines)

    def _generate_proposal(self, user: User) -> dict:
        """Generate a planning proposal for the user."""
        if self.planning_service:
            return self.planning_service.generate_proposal(user)

        # Fallback if no planning service
        return {
            'text': "Je n'ai pas pu générer de proposition. Essaie de décrire tes contraintes.",
            'quick_replies': []
        }

    def _handle_proposal_feedback(
        self,
        user: User,
        message: str,
        recent_messages: list
    ) -> dict:
        """Handle user feedback on a planning proposal."""
        if self.planning_service:
            return self.planning_service.handle_feedback(user, message, recent_messages)

        return self._default_response()

    def _handle_free_conversation(
        self,
        user: User,
        message: str,
        profile: UserProfile
    ) -> dict:
        """Handle free-form conversation using LLM."""
        if self.llm_provider and self.llm_provider.is_available():
            # This would delegate to the main chat engine or a specialized handler
            # For now, return a prompt to be more specific
            pass

        return {
            'text': "Je n'ai pas tout compris. Tu peux m'envoyer ton emploi du temps en photo/PDF, ou me décrire tes horaires.",
            'quick_replies': [
                {'label': "📷 Envoyer un document", 'value': 'upload'},
                {'label': "⏭️ Passer", 'value': 'skip'},
            ]
        }

    def _default_response(self) -> dict:
        """Return a default response."""
        return {
            'text': "Je suis là pour t'aider! Dis-moi tes horaires ou contraintes.",
            'quick_replies': []
        }

    def complete_onboarding(self, user: User) -> None:
        """Mark onboarding as complete for a user."""
        profile = user.profile
        profile.onboarding_completed = True
        profile.onboarding_step = 3
        profile.save()
        logger.info(f"Onboarding completed for user {user.id}")
