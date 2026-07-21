"""
LLM Provider abstraction layer.

Allows switching between different LLM providers (Gemini, Claude, OpenAI, etc.)
without changing the rest of the codebase.

Usage:
    from services.llm import GeminiProvider, ClaudeProvider

    # Use Gemini
    llm = GeminiProvider()

    # Use Claude
    llm = ClaudeProvider()

    # Both have the same API
    response = llm.generate("Hello!")
"""
import logging

from .base import LLMProvider, LLMResponse, FunctionCall
from .gemini import GeminiProvider
from .claude import ClaudeProvider

logger = logging.getLogger(__name__)

# Canonical provider registry. Keys are the accepted `name` values.
_PROVIDERS = {
    'gemini': GeminiProvider,
    'claude': ClaudeProvider,
}

DEFAULT_PROVIDER = 'gemini'


def get_provider(name: str | None = None, user=None) -> LLMProvider:
    """
    Resolve and instantiate an LLM provider.

    Resolution order (first match wins):
        1. explicit ``name`` argument
        2. ``user.profile.preferred_llm`` (guarded; ignored if missing/invalid)
        3. ``settings.LLM_PROVIDER``
        4. ``'gemini'`` (default that already works)

    Unknown names fall back to the default provider rather than raising, so a
    bad stored preference can never break the chat.

    Args:
        name: Explicit provider name ('gemini' | 'claude'), case-insensitive.
        user: Optional Django user; its ``profile.preferred_llm`` is consulted.

    Returns:
        An instantiated :class:`LLMProvider` (GeminiProvider or ClaudeProvider).
    """
    resolved = None

    # 1. Explicit name.
    if name:
        resolved = name

    # 2. User preference (guarded — profile / attribute may not exist).
    if not resolved and user is not None:
        try:
            profile = getattr(user, 'profile', None)
            preferred = getattr(profile, 'preferred_llm', None) if profile else None
            if preferred:
                resolved = preferred
        except Exception:
            logger.debug("Could not read user.profile.preferred_llm", exc_info=True)

    # 3. Settings default.
    if not resolved:
        try:
            from django.conf import settings
            resolved = getattr(settings, 'LLM_PROVIDER', None)
        except Exception:
            resolved = None

    # 4. Hard default.
    if not resolved:
        resolved = DEFAULT_PROVIDER

    key = str(resolved).strip().lower()
    provider_cls = _PROVIDERS.get(key)
    if provider_cls is None:
        logger.warning(
            "Unknown LLM provider '%s'; falling back to '%s'.", resolved, DEFAULT_PROVIDER
        )
        provider_cls = _PROVIDERS[DEFAULT_PROVIDER]

    return provider_cls()


__all__ = [
    'LLMProvider',
    'LLMResponse',
    'FunctionCall',
    'GeminiProvider',
    'ClaudeProvider',
    'get_provider',
]
