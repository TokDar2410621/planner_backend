"""
Consentement IA (Apple 5.1.2(i)).

L'app envoie le contenu utilisateur (messages du chat, documents d'horaire)
à des IA tierces: Google Gemini, Anthropic Claude, DeepSeek et Hugging Face.
Apple exige un consentement EXPLICITE avant le premier envoi. La vérité vit
sur UserProfile.ai_consent_at (null = jamais consenti ou révoqué); chaque vue
qui déclenche un appel LLM appelle ai_consent_denied() en première ligne, et
le cron de documents saute les utilisateurs non consentants.
"""
from rest_framework import status
from rest_framework.response import Response

AI_PROVIDERS = "Google Gemini, Anthropic Claude, DeepSeek et Hugging Face"


def has_ai_consent(user) -> bool:
    profile = getattr(user, 'profile', None)
    return bool(profile and profile.ai_consent_at)


def ai_consent_denied(user):
    """Response 403 `code: ai_consent_required` si non consenti, sinon None."""
    if has_ai_consent(user):
        return None
    return Response(
        {
            "error": (
                "L'assistant IA a besoin de ton consentement: tes messages et "
                f"documents sont traités par {AI_PROVIDERS}. "
                "Tu peux l'activer (et le révoquer) dans Profil."
            ),
            "code": "ai_consent_required",
        },
        status=status.HTTP_403_FORBIDDEN,
    )
