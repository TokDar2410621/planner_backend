"""
Chat Helper Functions - Utility functions for ChatOrchestrator.

Contains:
- Session context analysis (topic detection, mood, pending actions)
- Casual message detection
- Intelligent fallback responses
- UI-related helpers (radio inputs generation)
"""
import logging
import re
import random
from typing import Optional

logger = logging.getLogger(__name__)


def analyze_session_context(history: list, current_message: str, is_casual_func=None) -> dict:
    """
    Analyze conversation history to build session context.
    This helps the bot understand the flow of conversation.

    Args:
        history: List of conversation messages [{'role': 'user'/'assistant', 'content': '...'}]
        current_message: The current user message
        is_casual_func: Optional function to check if message is casual

    Returns:
        dict with last_topic, pending_action, mood, clarification_asked, preference_being_discussed
    """
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
        'transport': r'(trajet|route|transport|voiture|métro|bus|vélo|metro)',
        'sommeil': r'(dors|dort|sommeil|couche|réveil|reveil|lever|nuit|dormir)',
        'travail': r'(travail|boulot|job|bureau|entreprise|horaires|bosse|bosser)',
        'cours': r'(cours|école|ecole|université|universite|fac|études|etudes|exam|examen)',
        'sport': r'(sport|gym|fitness|entraînement|entrainement|course|foot|muscu)',
        'productivité': r'(productif|concentration|focus|efficace|productivite)',
        'planning': r'(planning|emploi du temps|agenda|calendrier|semaine)',
        'repas': r'(repas|manger|déjeuner|dejeuner|dîner|diner|petit-déjeuner|cuisine)',
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
            last_lower = last_assistant_msg.lower()

            # Check if bot asked a question
            if '?' in last_assistant_msg:
                if any(word in last_lower for word in ['confirme', "c'est bien", 'correct', 'est-ce que']):
                    context['pending_action'] = 'confirmation_attendue'
                elif any(word in last_lower for word in ['quel', 'quelle', 'combien', 'quand', 'où', 'ou']):
                    context['pending_action'] = 'réponse_attendue'

            # Check if clarification was asked
            if any(phrase in last_lower for phrase in ["je n'ai pas compris", 'tu voulais', 'peux-tu préciser', 'précise', 'clarifi']):
                context['clarification_asked'] = True

    # Detect conversation mood from current message
    current_lower = current_message.lower()
    if any(word in current_lower for word in ['merci', 'super', 'génial', 'genial', 'parfait', 'cool', 'top', 'excellent', 'nickel']):
        context['mood'] = 'positif'
    elif any(word in current_lower for word in ['non', 'pas ça', 'pas ca', 'erreur', 'problème', 'probleme', 'bug', 'marche pas', 'fonctionne pas']):
        context['mood'] = 'frustré'
    elif is_casual_func and is_casual_func(current_message):
        context['mood'] = 'casual'
    elif is_casual_message(current_message):
        context['mood'] = 'casual'

    # Check if preference is being discussed
    pref_patterns = {
        'transport_time': r'(\d+)\s*(min|minutes?)\s*(de\s*)?(route|trajet|transport)',
        'sleep_hours': r'(dors|dort|dormir)\s*(\d+)\s*h',
        'productivity_time': r'(productif|efficace|concentr).*(matin|soir|après-midi|apres-midi)',
    }

    for pref, pattern in pref_patterns.items():
        if re.search(pattern, current_lower):
            context['preference_being_discussed'] = pref
            break

    return context


def is_casual_message(message: str) -> bool:
    """
    Check if a message is casual conversation (greetings, thanks, ok, etc.)
    These messages don't need quick replies - keep the conversation natural.

    Args:
        message: The user's message

    Returns:
        True if the message is casual conversation
    """
    message_lower = message.lower().strip()

    casual_patterns = [
        r'^(salut|hello|hi|hey|coucou|bonjour|bonsoir|yo)[\s!?\.]*$',
        r'^(ça va|ca va|comment ça va|comment ca va|comment vas-tu|quoi de neuf)[\s!?\.,]*$',
        r'^(ça va bien|ca va bien|bien et toi|super et toi|bien merci).*$',
        r'^(merci|thanks|thx|merci beaucoup|merci bien)[\s!?\.,]*$',
        r'^(ok|okay|d\'accord|daccord|cool|super|nice|top|parfait|génial|genial|nickel)[\s!?\.,]*$',
        r'^(oui|non|ouais|nope|yep|nan|ouep|nah)[\s!?\.,]*$',
        r'^(de rien|pas de quoi|avec plaisir|je t\'en prie)[\s!?\.,]*$',
        r'^(à plus|a plus|bye|ciao|à bientôt|a bientot|salut|tchao)[\s!?\.,]*$',
        r'^(lol|mdr|haha|hihi|xd|ptdr)[\s!?\.,]*$',
    ]

    for pattern in casual_patterns:
        if re.search(pattern, message_lower):
            return True
    return False


def get_intelligent_fallback(user, message: str, function_calls_made: list = None) -> str:
    """
    Generate an intelligent fallback response based on context.
    Instead of generic "Comment puis-je t'aider?", provide contextual responses.

    Args:
        user: Django User object
        message: The user's message
        function_calls_made: List of function names that were called

    Returns:
        A contextual fallback response string
    """
    message_lower = message.lower().strip()
    function_calls_made = function_calls_made or []

    # Casual greetings - respond naturally with variety
    casual_responses = {
        r'^(salut|hello|hi|hey|coucou|bonjour|bonsoir)': [
            "Salut! Comment ça va?",
            "Hey! Qu'est-ce qui t'amène?",
            "Coucou! Je suis là pour t'aider avec ton planning.",
            "Salut! Quoi de neuf?",
        ],
        r'^(ça va|ca va|comment ça va|comment vas-tu)': [
            "Ça va bien, merci! Et toi, comment je peux t'aider?",
            "Super! Qu'est-ce que je peux faire pour toi?",
            "Nickel! Tu as besoin de quelque chose?",
        ],
        r'^(merci|thanks|thx)': [
            "De rien! N'hésite pas si tu as besoin d'autre chose.",
            "Avec plaisir! Je suis là si tu as besoin.",
            "Pas de quoi! Autre chose?",
        ],
        r'^(ok|okay|d\'accord|cool|super|nice|top|parfait)': [
            "Parfait! Autre chose?",
            "Super! Tu as besoin d'autre chose?",
            "Nickel! Je peux t'aider avec autre chose?",
        ],
        r'^(oui|non|ouais|nope)$': [
            "D'accord! Qu'est-ce que tu veux faire maintenant?",
            "OK! Dis-moi ce dont tu as besoin.",
        ],
    }

    for pattern, responses in casual_responses.items():
        if re.search(pattern, message_lower):
            return random.choice(responses)

    # If we made function calls but got no response, acknowledge the action
    if function_calls_made:
        return "C'est noté! Autre chose à ajouter?"

    # Check user's profile completion
    try:
        profile = user.profile
        from core.models import RecurringBlock
        blocks_count = RecurringBlock.objects.filter(user=user, active=True).count()

        # Contextual suggestion based on what's missing
        if not profile.onboarding_completed:
            if blocks_count == 0:
                return "Dis-moi tes contraintes! Par exemple: 'je travaille de 9h à 17h' ou 'j'ai cours le lundi matin'."

            missing_prefs = []
            if not profile.transport_time_minutes or profile.transport_time_minutes == 30:
                missing_prefs.append("temps de trajet")
            if not profile.min_sleep_hours or profile.min_sleep_hours == 7:
                missing_prefs.append("heures de sommeil")

            if missing_prefs:
                return f"Pour optimiser ton planning, dis-moi ton {missing_prefs[0]}. Par exemple: 'je prends 20 min de route'."

        # Default contextual response
        if blocks_count > 0:
            return "Je suis là! Tu veux ajouter quelque chose à ton planning ou modifier une préférence?"
        else:
            return "Je suis ton assistant de planification! Dis-moi tes horaires de travail, tes cours, ou tes activités."

    except Exception as e:
        logger.warning(f"Error getting user context in fallback: {e}")
        return "Je suis là pour t'aider avec ton planning! Que veux-tu faire?"


def generate_preference_radio_input(pref_type: str, detected_value: str, question: str) -> Optional[dict]:
    """
    Generate a radio input configuration for preference confirmation.

    Args:
        pref_type: Type of preference (transport_time, sleep_hours, etc.)
        detected_value: The value detected from the user's message
        question: The question to ask

    Returns:
        dict: Radio input configuration for the frontend, or None if invalid pref_type
    """
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
            'other_placeholder': "Nombre d'heures...",
        },
        'productivity_time': {
            'id': 'peak_productivity_time',
            'label': 'Moment de productivité',
            'options': [
                {'value': 'morning', 'label': 'Le matin'},
                {'value': 'afternoon', 'label': "L'après-midi"},
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
            'other_placeholder': "Nombre d'heures...",
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
