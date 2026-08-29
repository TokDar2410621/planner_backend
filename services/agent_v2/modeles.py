"""
Les modeles de v2 et leur chaine de secours.

AGIR raisonne et outille. DIRE redige, sans outil et SANS raisonnement: verifie
par sonde, DeepSeek refuse tool_choice=required en mode thinking, or c'est
ainsi que PydanticAI force une sortie structuree. Sans le reglage, DIRE echoue
systematiquement.
"""
from __future__ import annotations

from django.conf import settings
from openai import AsyncOpenAI
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

# Coupe le raisonnement pour la phase DIRE. Voir docs/sonde-agent-v2-2026-08-24.md
REGLAGES_DIRE = {"extra_body": {"thinking": {"type": "disabled"}}}


# Au-dela, on considere que le fournisseur ne repond plus assez vite pour un
# usage conversationnel, et la chaine passe au suivant.
#
# Mesure du 2026-08-29. En production ce matin: mediane de 57 s par tour, un
# tour a 397 s, jusqu'a 99 s pour un seul appel. Le meme code l'apres-midi:
# mediane de 8,2 s, maximum 16,7 s, avec 15 a 350 jetons de raisonnement par
# etape. Rien n'avait change chez nous: DeepSeek etait lent, voila tout.
#
# Or le client par defaut attend 600 SECONDES en lecture et reessaie deux
# fois. Personne n'arretait donc le tour de 397 s. `FallbackModel` bascule sur
# ModelAPIError, et un depassement de delai en leve un (verifie par sonde):
# borner suffit a rendre la main a Gemini.
#
# `max_retries=0` est deliberé: rejouer un appel qui vient d'expirer chez un
# fournisseur qui rame, c'est attendre trois fois. La resilience est assuree
# par la chaine de repli, pas par l'acharnement.
DELAI_MODELE = 60.0


def _deepseek():
    cle = getattr(settings, "DEEPSEEK_API_KEY", "")
    if not cle:
        return None
    return OpenAIChatModel(
        getattr(settings, "DEEPSEEK_MODEL", "deepseek-v4-pro"),
        provider=DeepSeekProvider(
            openai_client=AsyncOpenAI(
                api_key=cle,
                base_url="https://api.deepseek.com",
                timeout=DELAI_MODELE,
                max_retries=0,
            )
        ),
    )


def _gemini():
    cle = getattr(settings, "GEMINI_API_KEY", "")
    if not cle:
        return None
    try:
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider
    except ImportError:
        return None
    return GoogleModel("gemini-2.5-flash", provider=GoogleProvider(api_key=cle))


def _claude():
    cle = getattr(settings, "ANTHROPIC_API_KEY", "")
    if not cle:
        return None
    try:
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
    except ImportError:
        return None
    return AnthropicModel("claude-sonnet-4-5", provider=AnthropicProvider(api_key=cle))


def _chaine():
    vivants = [m for m in (_deepseek(), _gemini(), _claude()) if m is not None]
    if not vivants:
        raise RuntimeError(
            "Aucun fournisseur configure: renseigner DEEPSEEK_API_KEY, "
            "GEMINI_API_KEY ou ANTHROPIC_API_KEY."
        )
    return vivants[0] if len(vivants) == 1 else FallbackModel(*vivants)


def modele_agir():
    return _chaine()


def modele_dire():
    return _chaine()


def noms_de(modele) -> list[str]:
    membres = getattr(modele, "models", None)
    if membres is None:
        return [getattr(modele, "model_name", str(modele))]
    return [getattr(m, "model_name", str(m)) for m in membres]
