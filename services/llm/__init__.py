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
from .base import LLMProvider, LLMResponse, FunctionCall
from .gemini import GeminiProvider
from .claude import ClaudeProvider

__all__ = [
    'LLMProvider',
    'LLMResponse',
    'FunctionCall',
    'GeminiProvider',
    'ClaudeProvider',
]
