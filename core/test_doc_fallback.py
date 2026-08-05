"""
Fallback DeepSeek pour la STRUCTURATION TEXTE des documents: quand Gemini
tombe, un PDF texte doit quand meme etre structure (DeepSeek est texte-only
mais cette etape n'exige pas la vision).
"""
from unittest.mock import patch, MagicMock

from django.test import TestCase

from services.document_processor import DocumentProcessor
from services.llm.base import LLMResponse


class _FakeDeepSeek:
    def __init__(self, response):
        self._response = response

    def is_available(self):
        return True

    def generate_with_history(self, messages, tools, system_prompt):
        return self._response


class DocTextFallbackTest(TestCase):
    def _processor_with_failing_gemini(self):
        proc = DocumentProcessor.__new__(DocumentProcessor)
        proc.gemini_client = MagicMock()
        proc.hf_client = None
        proc.hf_model = None
        proc._call_gemini = MagicMock(side_effect=RuntimeError("gemini down"))
        return proc

    def test_deepseek_rescues_text_structuring_when_gemini_fails(self):
        proc = self._processor_with_failing_gemini()
        ok = LLMResponse(text='{"courses": []}', is_error=False)
        with patch('services.llm.get_provider', return_value=_FakeDeepSeek(ok)):
            out = proc._structure_text_with_llm("Lundi 9h cours de maths")
        self.assertEqual(out, '{"courses": []}')

    def test_raises_when_both_fail(self):
        proc = self._processor_with_failing_gemini()
        err = LLMResponse(text="", is_error=True)
        with patch('services.llm.get_provider', return_value=_FakeDeepSeek(err)):
            with self.assertRaises(RuntimeError):
                proc._structure_text_with_llm("Lundi 9h cours de maths")

    def test_deepseek_used_directly_when_no_gemini(self):
        proc = DocumentProcessor.__new__(DocumentProcessor)
        proc.gemini_client = None
        proc.hf_client = None
        proc.hf_model = None
        ok = LLMResponse(text='{"shifts": []}', is_error=False)
        with patch('services.llm.get_provider', return_value=_FakeDeepSeek(ok)):
            out = proc._structure_text_with_llm("vendredi 18h resto")
        self.assertEqual(out, '{"shifts": []}')
