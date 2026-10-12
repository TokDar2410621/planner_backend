"""
Les modeles de la boucle unique et leur chaine de secours.

La boucle raisonne, outille et rend une reponse structuree (ReponseDire).
DeepSeek refuse tool_choice=required en mode thinking (verifie par sonde),
or c'est ainsi que PydanticAI force une sortie structuree: REGLAGES_BOUCLE_
SANS_RAISONNEMENT existe pour ce cas.
"""
from __future__ import annotations

import inspect
import logging

from django.conf import settings
from openai import AsyncOpenAI
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

# Coupe le raisonnement pour la boucle unique quand le fournisseur l'exige.
# (DeepSeek refuse tool_choice=required en mode thinking, or c'est ainsi que
# PydanticAI force une sortie structuree.) Reserve aux modeles qui en ont
# besoin; la boucle par defaut raisonne normalement.
REGLAGES_BOUCLE_SANS_RAISONNEMENT = {"extra_body": {"thinking": {"type": "disabled"}}}


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


async def fermer_modele(modele):
    """Ferme les clients de ce tour avant que sa boucle asyncio disparaisse."""
    from pydantic_ai.models.wrapper import WrapperModel
    if isinstance(modele, WrapperModel):
        await fermer_modele(modele.wrapped)
        return
    if isinstance(modele, FallbackModel):
        for membre in modele.models:
            await fermer_modele(membre)
        return
    client = getattr(modele, 'client', None)
    if client is None:
        return
    try:
        aio = getattr(client, 'aio', None)
        if aio is not None and hasattr(aio, 'aclose'):
            await aio.aclose()
        fermeture = client.close()
        if inspect.isawaitable(fermeture):
            await fermeture
    except Exception as exc:
        logging.getLogger(__name__).warning('fermeture client modèle type=%s', type(exc).__name__)


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
        from google import genai
        from google.genai import types
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider
    except ImportError:
        return None
    client = genai.Client(api_key=cle, http_options=types.HttpOptions(
        timeout=int(DELAI_MODELE * 1000),
        retry_options=types.HttpRetryOptions(attempts=1)))
    return GoogleModel("gemini-2.5-flash", provider=GoogleProvider(client=client))


def _claude():
    cle = getattr(settings, "ANTHROPIC_API_KEY", "")
    if not cle:
        return None
    try:
        from anthropic import AsyncAnthropic
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
    except ImportError:
        return None
    return AnthropicModel("claude-sonnet-4-5", provider=AnthropicProvider(
        anthropic_client=AsyncAnthropic(api_key=cle, timeout=DELAI_MODELE, max_retries=0)))


def _chaine():
    vivants = [m for m in (_deepseek(), _gemini(), _claude()) if m is not None]
    if not vivants:
        raise RuntimeError(
            "Aucun fournisseur configure: renseigner DEEPSEEK_API_KEY, "
            "GEMINI_API_KEY ou ANTHROPIC_API_KEY."
        )
    return vivants[0] if len(vivants) == 1 else FallbackModel(*vivants)


def modele_agir():
    from services.agent_v2.budget_tour import modele_borne
    return modele_borne(_chaine())


def noms_de(modele) -> list[str]:
    membres = getattr(modele, "models", None)
    if membres is None:
        return [getattr(modele, "model_name", str(modele))]
    return [getattr(m, "model_name", str(m)) for m in membres]
