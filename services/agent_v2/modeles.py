"""
Les modeles de v2 et leur chaine de secours.

AGIR raisonne et outille. DIRE redige, sans outil et SANS raisonnement: verifie
par sonde, DeepSeek refuse tool_choice=required en mode thinking, or c'est
ainsi que PydanticAI force une sortie structuree. Sans le reglage, DIRE echoue
systematiquement.
"""
from __future__ import annotations

from django.conf import settings
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

# Coupe le raisonnement pour la phase DIRE. Voir docs/sonde-agent-v2-2026-08-24.md
REGLAGES_DIRE = {"extra_body": {"thinking": {"type": "disabled"}}}


def _deepseek():
    cle = getattr(settings, "DEEPSEEK_API_KEY", "")
    if not cle:
        return None
    return OpenAIChatModel(
        getattr(settings, "DEEPSEEK_MODEL", "deepseek-v4-pro"),
        provider=DeepSeekProvider(api_key=cle),
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
