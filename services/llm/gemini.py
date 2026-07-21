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
            return LLMResponse(
                text="Service IA non disponible.",
                is_error=True,
                error="provider_unavailable",
            )

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
            return LLMResponse(
                text="Erreur lors de la communication avec l'IA.",
                is_error=True,
                error=str(e),
            )

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

        # D4: capture finish_reason + token usage. Gemini exposes finish_reason
        # on the candidate (e.g. STOP, MAX_TOKENS, SAFETY) and usage on
        # usage_metadata.
        stop_reason = None
        candidate = response.candidates[0]
        fr = getattr(candidate, 'finish_reason', None)
        if fr is not None:
            stop_reason = getattr(fr, 'name', None) or str(fr)

        usage = None
        um = getattr(response, 'usage_metadata', None)
        if um is not None:
            usage = {
                'input_tokens': getattr(um, 'prompt_token_count', None),
                'output_tokens': getattr(um, 'candidates_token_count', None),
            }

        if stop_reason and str(stop_reason).lower() in ('max_tokens', 'length'):
            logger.warning(
                "Gemini response truncated (finish_reason=%s); output is incomplete.",
                stop_reason,
            )

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
            raw_response=response,
            stop_reason=stop_reason,
            usage=usage,
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
            logger.error("Gemini provider not available (client is None)")
            return LLMResponse(
                text="Service IA non disponible.",
                is_error=True,
                error="provider_unavailable",
            )

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
                contents.append(self._message_to_content(msg))

            logger.debug(f"Gemini request: {len(contents)} content parts, tools={'yes' if tools else 'no'}")

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
            parsed = self._parse_response(response)

            logger.debug(f"Gemini response parsed: text_len={len(parsed.text)}, function_calls={len(parsed.function_calls)}")
            return parsed

        except Exception as e:
            logger.error(f"Gemini API error with history: {e}", exc_info=True)
            return LLMResponse(
                text="Erreur lors de la communication avec l'IA.",
                is_error=True,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Multi-turn / tool round-trip helpers (B19)
    # ------------------------------------------------------------------
    def _message_to_content(self, msg: dict) -> "types.Content":
        """
        Convert one history message into a Gemini ``Content``.

        Handles plain text as well as tool round-trip blocks so that
        function-call (assistant) and function-response (tool-result) turns
        survive a full request cycle instead of being flattened to text.

        Accepted ``content`` shapes:
          - ``str`` -> a single text part
          - ``list`` -> a list of blocks, each of which may be a genai
            ``Part``, a ``FunctionCall`` dataclass, or a dict describing a
            text / function_call / function_response / Claude-style tool block.
        """
        role = "model" if msg.get("role") in ("assistant", "model") else "user"
        content = msg.get("content", "")

        if isinstance(content, str):
            return types.Content(role=role, parts=[types.Part(text=content)])

        if isinstance(content, list):
            parts = []
            for block in content:
                part = self._block_to_part(block)
                if part is not None:
                    parts.append(part)
            if not parts:
                parts = [types.Part(text="")]
            return types.Content(role=role, parts=parts)

        # Unknown shape: stringify defensively.
        return types.Content(role=role, parts=[types.Part(text=str(content))])

    def _block_to_part(self, block) -> Optional["types.Part"]:
        """Convert a single content block into a Gemini ``Part`` (or None)."""
        # Already a genai Part.
        if types is not None and isinstance(block, types.Part):
            return block

        # FunctionCall dataclass from base.py.
        if isinstance(block, FunctionCall):
            return types.Part.from_function_call(name=block.name, args=block.args or {})

        if isinstance(block, str):
            return types.Part(text=block)

        if isinstance(block, dict):
            # Explicit function_call block.
            fc = block.get("function_call")
            if fc:
                return types.Part.from_function_call(
                    name=fc.get("name", ""),
                    args=fc.get("args") or fc.get("arguments") or {},
                )

            # Explicit function_response block.
            fr = block.get("function_response")
            if fr:
                resp = fr.get("response")
                if not isinstance(resp, dict):
                    resp = {"result": resp}
                return types.Part.from_function_response(
                    name=fr.get("name", ""),
                    response=resp,
                )

            # Claude-style blocks (raw_content round-trip).
            btype = block.get("type")
            if btype == "tool_use":
                return types.Part.from_function_call(
                    name=block.get("name", ""),
                    args=block.get("input") or {},
                )
            if btype == "tool_result":
                raw = block.get("content")
                if isinstance(raw, dict):
                    resp = raw
                else:
                    resp = {"result": raw}
                return types.Part.from_function_response(
                    name=block.get("name") or block.get("tool_use_id", ""),
                    response=resp,
                )
            if btype == "text" or "text" in block:
                return types.Part(text=block.get("text", ""))

        # Fallback: stringify.
        return types.Part(text=str(block))
