"""
Claude (Anthropic) LLM Provider implementation.
"""
import logging
from typing import Optional

from django.conf import settings

from .base import LLMProvider, LLMResponse, FunctionCall
from utils.helpers import retry_with_backoff

logger = logging.getLogger(__name__)

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    anthropic = None


class ClaudeProvider(LLMProvider):
    """
    Anthropic Claude LLM provider.

    Supports tool use (function calling) and retries with exponential backoff.
    """

    DEFAULT_MODEL = 'claude-sonnet-4-20250514'

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the Claude provider.

        Args:
            model_name: Optional model name override
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.client = None

        api_key = getattr(settings, 'ANTHROPIC_API_KEY', None)
        if CLAUDE_AVAILABLE and api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"ClaudeProvider initialized with model: {self.model_name}")
        else:
            if not CLAUDE_AVAILABLE:
                logger.warning("anthropic package not installed. Run: pip install anthropic")
            else:
                logger.warning("ANTHROPIC_API_KEY not configured in settings")

    def is_available(self) -> bool:
        """Check if Claude is properly configured."""
        return self.client is not None

    @property
    def name(self) -> str:
        return f"Claude ({self.model_name})"

    @retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=30.0)
    def _call_api(self, messages: list, system: str = None, tools: list = None):
        """
        Call Claude API with retry logic for rate limits.

        Args:
            messages: List of message dicts
            system: Optional system prompt
            tools: Optional list of tool definitions

        Returns:
            Raw response from Claude
        """
        kwargs = {
            "model": self.model_name,
            "max_tokens": 4096,
            "messages": messages,
        }

        if system:
            kwargs["system"] = system

        if tools:
            kwargs["tools"] = tools

        return self.client.messages.create(**kwargs)

    def generate(
        self,
        prompt: str,
        tools: Optional[list] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a response from Claude.

        Args:
            prompt: The user prompt/message
            tools: Optional list of tool definitions (Claude format)
            system_prompt: Optional system prompt

        Returns:
            LLMResponse with text and/or function calls
        """
        if not self.is_available():
            logger.error("Claude provider not available")
            return LLMResponse(text="Service IA non disponible.")

        try:
            messages = [{"role": "user", "content": prompt}]

            # Convert Gemini-style tools to Claude format if needed
            claude_tools = self._convert_tools(tools) if tools else None

            response = self._call_api(
                messages=messages,
                system=system_prompt,
                tools=claude_tools
            )

            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return LLMResponse(text="Erreur lors de la communication avec l'IA.")

    def _convert_tools(self, tools: list) -> list:
        """
        Convert Gemini-style tools to Claude format.

        Claude expects:
        {
            "name": "function_name",
            "description": "...",
            "input_schema": { JSON Schema }
        }
        """
        claude_tools = []

        for tool in tools:
            # Handle Gemini Tool objects
            if hasattr(tool, 'function_declarations'):
                for func in tool.function_declarations:
                    claude_tool = {
                        "name": func.name,
                        "description": func.description or "",
                        "input_schema": self._convert_schema(func.parameters) if func.parameters else {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                    claude_tools.append(claude_tool)
            # Handle dict format
            elif isinstance(tool, dict):
                claude_tools.append(tool)

        return claude_tools

    def _convert_schema(self, gemini_schema) -> dict:
        """Convert Gemini Schema to JSON Schema for Claude."""
        if hasattr(gemini_schema, 'type'):
            schema = {"type": gemini_schema.type.lower()}

            if hasattr(gemini_schema, 'properties') and gemini_schema.properties:
                schema["properties"] = {}
                for name, prop in gemini_schema.properties.items():
                    schema["properties"][name] = self._convert_schema(prop)

            if hasattr(gemini_schema, 'required') and gemini_schema.required:
                schema["required"] = list(gemini_schema.required)

            if hasattr(gemini_schema, 'description') and gemini_schema.description:
                schema["description"] = gemini_schema.description

            if hasattr(gemini_schema, 'items') and gemini_schema.items:
                schema["items"] = self._convert_schema(gemini_schema.items)

            return schema

        return {"type": "string"}

    def _parse_response(self, response) -> LLMResponse:
        """
        Parse Claude response into standardized LLMResponse.
        Preserves raw content blocks for multi-turn tool use.
        """
        text_parts = []
        function_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                function_calls.append(FunctionCall(
                    name=block.name,
                    args=block.input if block.input else {},
                    call_id=block.id,
                ))
                logger.info(f"Tool use detected: {block.name} (id={block.id}) with args: {block.input}")

        return LLMResponse(
            text="".join(text_parts),
            function_calls=function_calls,
            raw_response=response,
            raw_content=response.content,
        )

    def generate_with_history(
        self,
        messages: list[dict],
        tools: Optional[list] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a response with conversation history.

        Supports multi-turn tool use with these message formats:
        - {"role": "user", "content": "text"}
        - {"role": "assistant", "content": [content_blocks]}  (raw Claude blocks)
        - {"role": "user", "content": [tool_result_blocks]}  (tool results)
        """
        if not self.is_available():
            return LLMResponse(text="Service IA non disponible.")

        try:
            claude_messages = []
            for msg in messages:
                role = msg["role"]
                if role == "model":  # Gemini compatibility
                    role = "assistant"

                content = msg["content"]

                # If content is already a list (raw blocks or tool results), pass directly
                if isinstance(content, list):
                    claude_messages.append({"role": role, "content": content})
                else:
                    claude_messages.append({"role": role, "content": content})

            claude_tools = self._convert_tools(tools) if tools else None

            response = self._call_api(
                messages=claude_messages,
                system=system_prompt,
                tools=claude_tools
            )

            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Claude API error with history: {e}", exc_info=True)
            return LLMResponse(text="Erreur lors de la communication avec l'IA.")
