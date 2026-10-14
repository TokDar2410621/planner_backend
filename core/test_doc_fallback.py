"""
Fallback DeepSeek pour la STRUCTURATION TEXTE des documents: quand Gemini
tombe, un PDF texte doit quand meme etre structure (DeepSeek est texte-only
mais cette etape n'exige pas la vision).
"""
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from django.contrib.auth import get_user_model
from django.test import TestCase

from services.document_processor import DocumentProcessor
from services.llm.deepseek import DeepSeekProvider
from services.agent_v2.budget_tour import ouvrir_trace, fermer_trace
from services.agent_v2.trace_tour import trace_courante


def _provider(text):
    provider = DeepSeekProvider()
    provider.api_key = 'fixture'
    provider._client = MagicMock()
    provider._client.with_options.return_value = provider._client
    provider._client.chat.completions.create.return_value = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8, total_tokens=20),
        choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
            content=text, tool_calls=[], reasoning_content=None))])
    return provider


class DocTextFallbackTest(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username='fallback-doc')
        self.trace = ouvrir_trace(self.user)
        self.token = trace_courante.set(self.trace)

    def tearDown(self):
        trace_courante.reset(self.token)
        fermer_trace(self.trace)

    def _processor_with_failing_gemini(self):
        proc = DocumentProcessor.__new__(DocumentProcessor)
        proc.gemini_client = MagicMock()
        proc.hf_client = None
        proc.hf_model = None
        proc._call_gemini = MagicMock(side_effect=RuntimeError("gemini down"))
        return proc

    def test_deepseek_rescues_text_structuring_when_gemini_fails(self):
        proc = self._processor_with_failing_gemini()
        provider = _provider('{"courses": []}')
        with patch('services.llm.get_provider', return_value=provider):
            out = proc._structure_text_with_llm("Lundi 9h cours de maths")
        self.assertEqual(out, '{"courses": []}')
        self.assertEqual(self.trace.usages[-1]['total'], 20)
        self.assertEqual(provider._client.with_options.call_args.kwargs['max_retries'], 0)

    def test_raises_when_both_fail(self):
        proc = self._processor_with_failing_gemini()
        provider = _provider('')
        with patch('services.llm.get_provider', return_value=provider):
            with self.assertRaises(RuntimeError):
                proc._structure_text_with_llm("Lundi 9h cours de maths")

    def test_deepseek_used_directly_when_no_gemini(self):
        proc = DocumentProcessor.__new__(DocumentProcessor)
        proc.gemini_client = None
        proc.hf_client = None
        proc.hf_model = None
        with patch('services.llm.get_provider', return_value=_provider('{"shifts": []}')):
            out = proc._structure_text_with_llm("vendredi 18h resto")
        self.assertEqual(out, '{"shifts": []}')

    def test_deepseek_retried_when_gemini_json_is_malformed(self):
        """Incident 2026-10-13: Gemini a repondu un JSON invalide (pas
        d'exception), l'ancien code ne retentait jamais DeepSeek."""
        proc = DocumentProcessor.__new__(DocumentProcessor)
        proc.gemini_client = MagicMock()
        proc.hf_client = None
        proc.hf_model = None
        bad_json = '```json\n{"courses": [{"title": "Maths",}\n```'
        proc._call_gemini = MagicMock(
            return_value=SimpleNamespace(text=bad_json))
        provider = _provider('{"courses": []}')
        with patch('services.llm.get_provider', return_value=provider):
            out = proc._structure_text_with_llm("Lundi 9h cours de maths")
        self.assertEqual(out, '{"courses": []}')

    def test_returns_last_raw_when_both_providers_malformed(self):
        proc = DocumentProcessor.__new__(DocumentProcessor)
        proc.gemini_client = MagicMock()
        proc.hf_client = None
        proc.hf_model = None
        bad_json = '```json\n{"courses": [{"title": "Maths",}\n```'
        proc._call_gemini = MagicMock(
            return_value=SimpleNamespace(text=bad_json))
        provider = _provider(bad_json)
        with patch('services.llm.get_provider', return_value=provider):
            out = proc._structure_text_with_llm("Lundi 9h cours de maths")
        self.assertEqual(out, bad_json)

    def test_gemini_json_mode_sets_response_mime_type(self):
        proc = DocumentProcessor.__new__(DocumentProcessor)
        proc.gemini_model = 'fixture'
        proc.gemini_client = MagicMock()
        proc.gemini_client.models.generate_content.return_value = SimpleNamespace(
            text='{}', usage_metadata=None)
        proc._verifier_echeance = lambda: None
        # La trace est deja ouverte par setUp.
        proc._call_gemini('texte', json_mode=True)
        config = proc.gemini_client.models.generate_content.call_args.kwargs['config']
        self.assertEqual(config.response_mime_type, 'application/json')

    def test_parse_response_repairs_trailing_comma(self):
        proc = DocumentProcessor.__new__(DocumentProcessor)
        out = proc._parse_response('{"courses": [{"name": "Maths",}], "shifts": []}')
        self.assertNotIn('parse_error', out)
        self.assertEqual(out['courses'][0]['name'], 'Maths')
