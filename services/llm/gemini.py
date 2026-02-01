"""
Gemini LLM Provider implementation.
"""
import logging
from typing import Optional

from django.conf import settings

from .base import LLMProvider, LLMResponse, FunctionCall
from utils.helpers import retry_with_backoff

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    types = None


class GeminiProvider(LLMProvider):
    """
    Google Gemini LLM provider.

    Supports function calling and retries with exponential backoff.
    """

    DEFAULT_MODEL = 'gemini-2.5-flash'

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the Gemini provider.

        Args:
            model_name: Optional model name override
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.client = None

        if GEMINI_AVAILABLE and settings.GEMINI_API_KEY:
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            logger.info(f"GeminiProvider initialized with model: {self.model_name}")
        else:
            logger.warning("Gemini API not configured or google-genai not installed")

    def is_available(self) -> bool:
        """Check if Gemini is properly configured."""
        return self.client is not None

    @property
    def name(self) -> str:
        return f"Gemini ({self.model_name})"

    @retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=30.0)
    def _call_api(self, contents, config=None):
        """
        Call Gemini API with retry logic for rate limits (429).

        Args:
            contents: Content to send to Gemini
            config: Optional GenerateContentConfig

        Returns:
            Raw response from Gemini
        """
        if config:
            return self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
        return self.client.models.generate_content(
            model=self.model_name,
            contents=contents
        )

    def generate(
        self,
        prompt: str,
        tools: Optional[list] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a response from Gemini.

        Args:
            prompt: The user prompt/message
            tools: Optional list of Gemini FunctionDeclaration objects
            system_prompt: Optional system prompt (prepended to prompt)

        Returns:
            LLMResponse with text and/or function calls
        """
        if not self.is_available():
            logger.error("Gemini provider not available")
            return LLMResponse(text="Service IA non disponible.")

        # Combine system prompt with user prompt if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            # Build config if tools are provided
            config = None
            if tools:
                config = types.GenerateContentConfig(
                    tools=tools,
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode='AUTO')
                    )
                )

            # Call the API
            response = self._call_api(full_prompt, config)

            # Parse the response
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return LLMResponse(text="Erreur lors de la communication avec l'IA.")

    def _parse_response(self, response) -> LLMResponse:
        """
        Parse Gemini response into standardized LLMResponse.

        Args:
            response: Raw Gemini response

        Returns:
            LLMResponse with extracted text and function calls
        """
        text_parts = []
        function_calls = []

        if not response.candidates:
            logger.warning("Gemini returned no candidates")
            return LLMResponse(text="", raw_response=response)

        for part in response.parts:
            # Check for function calls
            fc = getattr(part, 'function_call', None)
            if fc:
                function_calls.append(FunctionCall(
                    name=fc.name,
                    args=dict(fc.args) if fc.args else {}
                ))
                logger.info(f"Function call detected: {fc.name} with args: {dict(fc.args) if fc.args else {}}")

            # Check for text
            if hasattr(part, 'text') and part.text:
                text_parts.append(part.text)

        return LLMResponse(
            text="".join(text_parts),
            function_calls=function_calls,
            raw_response=response
        )

    def generate_with_history(
        self,
        messages: list[dict],
        tools: Optional[list] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a response with conversation history.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str}
            tools: Optional list of function declarations
            system_prompt: Optional system prompt

        Returns:
            LLMResponse with text and/or function calls
        """
        if not self.is_available():
            return LLMResponse(text="Service IA non disponible.")

        try:
            # Build conversation content
            contents = []
            if system_prompt:
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=f"[SYSTEM]\n{system_prompt}")]
                ))
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text="Compris, je suivrai ces instructions.")]
                ))

            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])]
                ))

            # Build config
            config = None
            if tools:
                config = types.GenerateContentConfig(
                    tools=tools,
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode='AUTO')
                    )
                )

            response = self._call_api(contents, config)
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Gemini API error with history: {e}")
            return LLMResponse(text="Erreur lors de la communication avec l'IA.")
