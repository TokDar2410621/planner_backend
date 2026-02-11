"""
Base LLM Provider interface.

All LLM providers should implement this interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FunctionCall:
    """Represents a function call from the LLM."""
    name: str
    args: dict = field(default_factory=dict)
    call_id: str = ""  # Tool use ID (required for Claude multi-turn)


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    text: str = ""
    function_calls: list[FunctionCall] = field(default_factory=list)
    raw_response: Any = None
    raw_content: list = field(default_factory=list)  # Raw content blocks for multi-turn

    @property
    def has_function_calls(self) -> bool:
        return len(self.function_calls) > 0


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implement this interface to add support for new LLM backends
    (OpenAI, Anthropic, local models, etc.)
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        tools: Optional[list] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt/message
            tools: Optional list of function declarations for function calling
            system_prompt: Optional system prompt to set context

        Returns:
            LLMResponse with text and/or function calls
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is properly configured and available."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name for logging."""
        pass
