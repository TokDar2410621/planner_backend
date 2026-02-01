"""
Planning service for schedule management.

Handles:
- Planning proposal generation
- Proposal feedback handling
- Block and task creation
"""
import json
import logging
import re
from datetime import time as dt_time
from typing import Optional

from django.contrib.auth.models import User
from django.utils import timezone

from core.models import RecurringBlock, Task, UserProfile
from services.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class PlanningService:
    """
    Service for managing user planning operations.

    Generates planning proposals, handles feedback, and creates blocks/tasks.
    """

    DAY_NAMES_FR = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']
    DAY_MAPPING = {
        'lundi': 0, 'mardi': 1, 'mercredi': 2, 'jeudi': 3,
        'vendredi': 4, 'samedi': 5, 'dimanche': 6
    }

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """
        Initialize the planning service.

        Args:
            llm_provider: LLM provider for AI-assisted operations
        """
        self.llm_provider = llm_provider

    def generate_proposal(self, user: User, overrides: Optional[dict] = None) -> dict:
        """
        Generate a smart planning proposal for the user.

        Args:
            user: The user to generate proposal for
            overrides: Optional schedule overrides (bedtime, wake_time, etc.)

        Returns:
            dict with 'text' and 'quick_replies'
        """
        profile = user.profile
        blocks = RecurringBlock.objects.filter(user=user, active=True).order_by('day_of_week', 'start_time')

        if not blocks.exists():
            return {
                'text': "Tu n'as pas encore de blocs dans ton planning. Décris-moi tes contraintes ou envoie un document!",
                'quick_replies': [
                    {'label': "📷 Envoyer un document", 'value': 'upload'},
                ]
            }

        # Build current planning summary
        summary = self._build_planning_summary(blocks)

        # Build proposal text
        proposal_text = f"""📋 **Proposition de planning**

Voici ton planning actuel:

{summary}

**Tes préférences:**
- Pic de productivité: {self._get_productivity_label(profile.peak_productivity_time)}
- Sommeil minimum: {profile.min_sleep_hours}h

Est-ce que ça te convient?"""

        return {
            'text': proposal_text,
            'quick_replies': [
                {'label': "✅ Oui, crée ce planning", 'value': 'Oui, crée ce planning'},
                {'label': "✏️ Modifier", 'value': 'Je veux modifier quelque chose'},
            ]
        }

    def handle_feedback(self, user: User, message: str, recent_messages: list) -> dict:
        """
        Handle user feedback on a planning proposal using AI.

        Args:
            user: The current user
            message: The user's feedback message
            recent_messages: Recent conversation for context

        Returns:
            dict with 'text' and 'quick_replies'
        """
        if not self.llm_provider or not self.llm_provider.is_available():
            return self._handle_feedback_fallback(user, message)

        profile = user.profile
        blocks = RecurringBlock.objects.filter(user=user, active=True).order_by('day_of_week', 'start_time')

        # Build context for AI
        blocks_text = self._build_blocks_context(blocks)
        last_proposal = self._extract_last_proposal(recent_messages)

        prompt = self._build_feedback_prompt(
            message=message,
            blocks_text=blocks_text,
            profile=profile,
            last_proposal=last_proposal
        )

        try:
            response = self.llm_provider.generate(prompt)

            # Parse JSON response
            json_match = re.search(r'\{[\s\S]*\}', response.text)
            if not json_match:
                return self._handle_feedback_fallback(user, message)

            data = json.loads(json_match.group(0))
            return self._execute_feedback_action(user, data, profile)

        except Exception as e:
            logger.error(f"Error handling feedback: {e}")
            return self._handle_feedback_fallback(user, message)

    def create_block(
        self,
        user: User,
        title: str,
        block_type: str,
        day: int,
        start_time: str,
        end_time: str,
        location: str = "",
        is_night_shift: bool = False
    ) -> Optional[RecurringBlock]:
        """
        Create a recurring block for a user.

        Args:
            user: The user
            title: Block title
            block_type: Type of block (course, work, etc.)
            day: Day of week (0-6)
            start_time: Start time as "HH:MM"
            end_time: End time as "HH:MM"
            location: Optional location
            is_night_shift: Whether this is a night shift

        Returns:
            Created RecurringBlock or None on error
        """
        try:
            start_parts = start_time.split(':')
            end_parts = end_time.split(':')

            block = RecurringBlock.objects.create(
                user=user,
                title=title,
                block_type=block_type,
                day_of_week=day,
                start_time=dt_time(int(start_parts[0]), int(start_parts[1]) if len(start_parts) > 1 else 0),
                end_time=dt_time(int(end_parts[0]), int(end_parts[1]) if len(end_parts) > 1 else 0),
                location=location or '',
                is_night_shift=is_night_shift,
            )
            logger.info(f"Created block: {block.title} for user {user.id}")
            return block

        except Exception as e:
            logger.error(f"Error creating block: {e}")
            return None

    def create_task(
        self,
        user: User,
        title: str,
        task_type: str = 'shallow',
        priority: int = 5,
        description: str = "",
        deadline=None,
        estimated_duration_minutes: int = None
    ) -> Optional[Task]:
        """
        Create a task for a user.

        Args:
            user: The user
            title: Task title
            task_type: Type of task
            priority: Priority 1-10
            description: Optional description
            deadline: Optional deadline datetime
            estimated_duration_minutes: Optional duration estimate

        Returns:
            Created Task or None on error
        """
        try:
            task = Task.objects.create(
                user=user,
                title=title,
                task_type=task_type,
                priority=priority,
                description=description,
                deadline=deadline,
                estimated_duration_minutes=estimated_duration_minutes,
            )
            logger.info(f"Created task: {task.title} for user {user.id}")
            return task

        except Exception as e:
            logger.error(f"Error creating task: {e}")
            return None

    def _build_planning_summary(self, blocks) -> str:
        """Build a human-readable planning summary."""
        by_day = {}
        for block in blocks:
            day_name = self.DAY_NAMES_FR[block.day_of_week]
            if day_name not in by_day:
                by_day[day_name] = []
            by_day[day_name].append(
                f"  • {block.title} ({block.start_time.strftime('%H:%M')}-{block.end_time.strftime('%H:%M')})"
            )

        lines = []
        for day in self.DAY_NAMES_FR:
            if day in by_day:
                lines.append(f"**{day.capitalize()}:**")
                lines.extend(by_day[day])

        return "\n".join(lines) if lines else "Aucun bloc configuré"

    def _build_blocks_context(self, blocks) -> str:
        """Build blocks context for AI prompt."""
        lines = []
        for block in blocks[:20]:
            day_name = self.DAY_NAMES_FR[block.day_of_week]
            lines.append(
                f"- {block.title} ({block.block_type}): {day_name} "
                f"{block.start_time.strftime('%H:%M')}-{block.end_time.strftime('%H:%M')}"
            )
        return "\n".join(lines) if lines else "Aucun bloc"

    def _extract_last_proposal(self, recent_messages: list) -> str:
        """Extract the last planning proposal from messages."""
        for msg in recent_messages:
            if msg.role == 'assistant' and 'proposition de planning' in msg.content.lower():
                return msg.content[:1000]
        return ""

    def _get_productivity_label(self, value: str) -> str:
        """Get French label for productivity time."""
        labels = {
            'morning': 'le matin',
            'afternoon': "l'après-midi",
            'evening': 'le soir'
        }
        return labels.get(value, value)

    def _build_feedback_prompt(
        self,
        message: str,
        blocks_text: str,
        profile: UserProfile,
        last_proposal: str
    ) -> str:
        """Build the AI prompt for handling feedback."""
        return f"""Tu es un assistant de planification. L'utilisateur donne un feedback sur sa proposition de planning.

PLANNING ACTUEL:
{blocks_text}

PRÉFÉRENCES:
- Pic de productivité: {profile.get_peak_productivity_time_display()}
- Sommeil minimum: {profile.min_sleep_hours}h

DERNIÈRE PROPOSITION:
{last_proposal if last_proposal else "Non disponible"}

MESSAGE DE L'UTILISATEUR:
"{message}"

ANALYSE et détermine:
1. L'utilisateur veut-il ACCEPTER? (oui, ok, crée, parfait, c'est bon)
2. Indique-t-il une CONTRAINTE DE TRAVAIL? (ex: "je travaille le soir")
3. Indique-t-il une PRÉFÉRENCE DE PRODUCTIVITÉ? (ex: "je préfère le soir")
4. Veut-il MODIFIER quelque chose? (heures de coucher, réveil, etc.)

Réponds en JSON:
{{
    "action": "accept_proposal|add_work_block|set_productivity_preference|ask_clarification|modify_specific",
    "details": {{
        "productivity_time": "morning|afternoon|evening",
        "work_block": {{
            "days": ["vendredi", "samedi"],
            "start_time": "HH:MM",
            "end_time": "HH:MM",
            "title": "Travail",
            "is_night_shift": true/false
        }}
    }},
    "response_to_user": "Message à afficher"
}}

Retourne UNIQUEMENT le JSON."""

    def _execute_feedback_action(self, user: User, data: dict, profile: UserProfile) -> dict:
        """Execute the action determined by AI."""
        action = data.get('action', '')
        details = data.get('details', {})
        response_text = data.get('response_to_user', '')

        if action == 'accept_proposal':
            profile.onboarding_completed = True
            profile.onboarding_step = 3
            profile.save()

            blocks_count = RecurringBlock.objects.filter(user=user, active=True).count()
            return {
                'text': f"✅ Parfait! Ton planning avec {blocks_count} blocs est prêt!",
                'quick_replies': [
                    {'label': "📋 Voir mon planning", 'value': 'Montre-moi mon planning'},
                    {'label': "➕ Ajouter autre chose", 'value': "J'ai autre chose à ajouter"},
                ]
            }

        elif action == 'add_work_block':
            work_info = details.get('work_block', {})
            created_count = self._create_work_blocks(user, work_info)

            proposal = self.generate_proposal(user)
            return {
                'text': f"J'ai ajouté {created_count} bloc(s) de travail.\n\n{proposal['text']}",
                'quick_replies': proposal.get('quick_replies', [])
            }

        elif action == 'set_productivity_preference':
            prod_time = details.get('productivity_time', 'morning')
            if prod_time in ['morning', 'afternoon', 'evening']:
                profile.peak_productivity_time = prod_time
                profile.save()

            proposal = self.generate_proposal(user)
            return {
                'text': f"J'ai noté ta préférence!\n\n{proposal['text']}",
                'quick_replies': proposal.get('quick_replies', [])
            }

        # Default: return AI response
        return {
            'text': response_text or "Je n'ai pas compris. Tu peux reformuler?",
            'quick_replies': []
        }

    def _create_work_blocks(self, user: User, work_info: dict) -> int:
        """Create work blocks from AI-extracted info."""
        days = work_info.get('days', [])
        start_time = work_info.get('start_time') or '18:00'
        end_time = work_info.get('end_time') or '23:00'
        title = work_info.get('title') or 'Travail'
        is_night_shift = work_info.get('is_night_shift', False)
        location = work_info.get('location', '')

        created_count = 0
        for day_name in days:
            day_num = self.DAY_MAPPING.get(day_name.lower())
            if day_num is not None:
                block = self.create_block(
                    user=user,
                    title=title,
                    block_type='work',
                    day=day_num,
                    start_time=start_time,
                    end_time=end_time,
                    location=location,
                    is_night_shift=is_night_shift
                )
                if block:
                    created_count += 1

        return created_count

    def _handle_feedback_fallback(self, user: User, message: str) -> dict:
        """Handle feedback when AI is not available."""
        message_lower = message.lower()

        # Simple keyword matching
        acceptance_words = ['oui', 'ok', 'parfait', "c'est bon", 'crée', 'valide']
        if any(word in message_lower for word in acceptance_words):
            profile = user.profile
            profile.onboarding_completed = True
            profile.onboarding_step = 3
            profile.save()

            return {
                'text': "✅ Ton planning est configuré!",
                'quick_replies': [
                    {'label': "📋 Voir mon planning", 'value': 'Montre-moi mon planning'},
                ]
            }

        return {
            'text': "Je n'ai pas compris. Tu veux accepter le planning ou le modifier?",
            'quick_replies': [
                {'label': "✅ Accepter", 'value': 'Oui, crée ce planning'},
                {'label': "✏️ Modifier", 'value': 'Je veux modifier'},
            ]
        }
