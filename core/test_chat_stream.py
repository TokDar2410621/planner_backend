"""Streaming chat (SSE) — Phase 2 latence.

Couvre les 3 étages:
  1. process_message_stream (agent): ordre des événements delta/status/
     turn_discard/done, autorité de done.response (garde anti faux succès),
     équivalence du wrapper process_message.
  2. ChatStreamView: frames SSE `data: {json}` + content-type.
  3. Providers: assemblage des chunks streamés DeepSeek (tool_calls fragmentés,
     écho reasoning_content) et Gemini (deltas + fallback candidat vide).
"""
import json

from core.lecture_flux import corps_du_flux
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework.test import APIClient

from services.agent.agent import PlannerAgent
from services.agent_v2 import PlannerAgentV2
from services.llm.base import FunctionCall, LLMResponse
from services.llm.deepseek import DeepSeekProvider
from services.llm.gemini import GeminiProvider


class FakeStreamLLM:
    """Provider factice: rejoue une liste d'événements par tour."""

    supports_streaming = True

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = 0

    def is_available(self):
        return True

    def stream_with_history(self, messages, tools=None, system_prompt=None):
        events = self.turns[min(self.calls, len(self.turns) - 1)]
        self.calls += 1
        yield from events

    def generate_with_history(self, messages, tools=None, system_prompt=None):
        return LLMResponse(text="(fallback non-streamé)")

    def generate(self, prompt, tools=None, system_prompt=None):
        return LLMResponse(text="[]")


def _final(text="", function_calls=None, raw_content=None):
    return {"type": "final", "response": LLMResponse(
        text=text,
        function_calls=function_calls or [],
        raw_content=raw_content if raw_content is not None else {
            "role": "assistant", "content": text},
    )}


class AgentStreamTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("streamer", password="pw-123456")
        self.agent = PlannerAgent()

    def _events(self, fake):
        with patch.object(self.agent, "_build_provider", return_value=fake):
            return list(self.agent.process_message_stream(self.user, "salut"))

    def test_simple_answer_streams_deltas_then_done(self):
        fake = FakeStreamLLM([[
            {"type": "text_delta", "text": "Bonjour "},
            {"type": "text_delta", "text": "toi."},
            _final("Bonjour toi."),
        ]])
        events = self._events(fake)
        deltas = [e for e in events if e["type"] == "delta"]
        dones = [e for e in events if e["type"] == "done"]
        self.assertEqual([d["text"] for d in deltas], ["Bonjour ", "toi."])
        self.assertEqual(len(dones), 1)
        self.assertEqual(dones[0]["response"], "Bonjour toi.")
        # done est TERMINAL
        self.assertEqual(events[-1]["type"], "done")

    def test_tool_turn_emits_status_then_final_deltas(self):
        tool_turn = [_final("", function_calls=[
            FunctionCall(name="list_blocks", args={}, call_id="c1")])]
        text_turn = [
            {"type": "text_delta", "text": "Tu n'as aucun bloc."},
            _final("Tu n'as aucun bloc."),
        ]
        fake = FakeStreamLLM([tool_turn, text_turn])
        events = self._events(fake)
        types = [e["type"] for e in events]
        self.assertIn("status", types)
        status_idx = types.index("status")
        delta_idx = types.index("delta")
        self.assertLess(status_idx, delta_idx)  # statut AVANT le texte final
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["response"], "Tu n'as aucun bloc.")

    def test_streamed_text_on_tool_turn_is_discarded(self):
        # Le modèle narre ("Je regarde...") PUIS appelle un outil: le client
        # doit vider la bulle (turn_discard) avant le statut d'outil.
        tool_turn = [
            {"type": "text_delta", "text": "Je regarde ça..."},
            _final("Je regarde ça...", function_calls=[
                FunctionCall(name="list_blocks", args={}, call_id="c1")]),
        ]
        text_turn = [
            {"type": "text_delta", "text": "Aucun bloc."},
            _final("Aucun bloc."),
        ]
        fake = FakeStreamLLM([tool_turn, text_turn])
        events = self._events(fake)
        types = [e["type"] for e in events]
        self.assertIn("turn_discard", types)
        self.assertLess(types.index("turn_discard"), types.index("status"))

    def test_done_response_overrides_streamed_lie(self):
        # Mutation TENTÉE (create_block args invalides -> échec) + texte final
        # streamé qui prétend le succès: done.response doit porter le démenti
        # honnête, PAS le texte streamé.
        tool_turn = [_final("", function_calls=[
            FunctionCall(name="create_block", args={}, call_id="c1")])]
        lie_turn = [
            {"type": "text_delta", "text": "C'est fait ! Ton bloc est créé."},
            _final("C'est fait ! Ton bloc est créé."),
        ]
        fake = FakeStreamLLM([tool_turn, lie_turn])
        events = self._events(fake)
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertIn("pas pu appliquer", done["response"])

    def test_stream_error_falls_back_to_non_streamed(self):
        class BrokenStream(FakeStreamLLM):
            def stream_with_history(self, messages, tools=None, system_prompt=None):
                yield {"type": "text_delta", "text": "déb"}
                raise RuntimeError("boom")

        fake = BrokenStream([[]])
        events = self._events(fake)
        types = [e["type"] for e in events]
        # le partiel est jeté puis le chemin non-streamé répond
        self.assertIn("turn_discard", types)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["response"], "(fallback non-streamé)")

    def test_wrapper_process_message_equals_done_payload(self):
        # process_message force use_streaming=False: c'est le chemin
        # generate_with_history (legacy exact) qui répond, jamais le stream.
        fake = FakeStreamLLM([[]])
        fake.generate_with_history = lambda **kw: LLMResponse(text="Ok.")
        with patch.object(self.agent, "_build_provider", return_value=fake):
            result = self.agent.process_message(
                self.user, "salut", generate_quick_replies=False)
        self.assertEqual(result["response"], "Ok.")
        self.assertNotIn("type", result)
        self.assertIn("quick_replies", result)


class ChatStreamViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("sse", password="pw-123456")
        # Gate Apple 5.1.2(i): sans consentement IA, la vue répond 403 avant
        # de streamer (couvert par core.test_ai_consent).
        from django.utils import timezone
        self.user.profile.ai_consent_at = timezone.now()
        self.user.profile.save(update_fields=['ai_consent_at'])
        self.client_api = APIClient()
        self.client_api.force_authenticate(self.user)

    def test_sse_frames_and_content_type(self):
        events = [
            {"type": "status", "text": "Je consulte ton planning…"},
            {"type": "delta", "text": "Voi"},
            {"type": "delta", "text": "là."},
            {"type": "done", "response": "Voilà.", "quick_replies": []},
        ]
        with patch.object(PlannerAgentV2, "process_message_stream",
                          return_value=iter(events)):
            resp = self.client_api.post("/api/chat/stream/", {"message": "salut"})
            # streaming_content est PARESSEUX: consommer DANS le patch, sinon
            # c'est le vrai agent (LLM live) qui s'exécute à l'itération.
            body = corps_du_flux(resp)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Content-Type"], "text/event-stream")
        frames = [json.loads(line[len("data: "):])
                  for line in body.split("\n\n") if line.startswith("data: ")]
        self.assertEqual(frames, events)

    def test_requires_message_or_file(self):
        resp = self.client_api.post("/api/chat/stream/", {})
        self.assertEqual(resp.status_code, 400)

    def test_agent_crash_yields_error_frame(self):
        with patch.object(PlannerAgentV2, "process_message_stream",
                          side_effect=RuntimeError("boom")):
            resp = self.client_api.post("/api/chat/stream/", {"message": "salut"})
            body = corps_du_flux(resp)
        self.assertIn('"type": "error"', body)


def _ds_chunk(content=None, reasoning=None, tool_calls=None, finish=None):
    delta = SimpleNamespace(
        content=content, reasoning_content=reasoning, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)])


class DeepSeekStreamTests(TestCase):
    def _provider(self):
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        p.api_key = "k"
        p.model = "deepseek-v4-pro"
        p.reasoning_effort = "high"
        p.thinking = True
        p.max_tokens = 512
        p._client = MagicMock()
        return p

    def test_stream_accumulates_fragmented_tool_call_and_reasoning(self):
        p = self._provider()
        tc1 = SimpleNamespace(index=0, id="call_1", function=SimpleNamespace(
            name="create_block", arguments='{"title":'))
        tc2 = SimpleNamespace(index=0, id=None, function=SimpleNamespace(
            name=None, arguments='"X"}'))
        p._client.chat.completions.create.return_value = iter([
            _ds_chunk(reasoning="hmm"),
            _ds_chunk(tool_calls=[tc1]),
            _ds_chunk(tool_calls=[tc2], finish="tool_calls"),
        ])
        events = list(p.stream_with_history([{"role": "user", "content": "hi"}]))
        self.assertEqual(events[0], {"type": "thinking"})
        final = events[-1]["response"]
        self.assertEqual(len(final.function_calls), 1)
        self.assertEqual(final.function_calls[0].name, "create_block")
        self.assertEqual(final.function_calls[0].args, {"title": "X"})
        self.assertEqual(final.function_calls[0].call_id, "call_1")
        # écho obligatoire: reasoning_content + tool_calls dans raw_content
        self.assertEqual(final.raw_content["reasoning_content"], "hmm")
        self.assertEqual(final.raw_content["tool_calls"][0]["function"]["name"],
                         "create_block")
        self.assertIsNone(final.raw_content["content"])  # pas de texte
        # le mode stream est bien demandé
        kwargs = p._client.chat.completions.create.call_args.kwargs
        self.assertTrue(kwargs["stream"])

    def test_stream_text_deltas(self):
        p = self._provider()
        p._client.chat.completions.create.return_value = iter([
            _ds_chunk(content="Bon"),
            _ds_chunk(content="jour", finish="stop"),
        ])
        events = list(p.stream_with_history([{"role": "user", "content": "hi"}]))
        self.assertEqual(
            [e["text"] for e in events if e["type"] == "text_delta"],
            ["Bon", "jour"])
        self.assertEqual(events[-1]["response"].text, "Bonjour")
        self.assertFalse(events[-1]["response"].has_function_calls)

    def test_stream_exception_yields_error_final(self):
        p = self._provider()
        p._client.chat.completions.create.side_effect = RuntimeError("boom")
        events = list(p.stream_with_history([{"role": "user", "content": "hi"}]))
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["response"].is_error)

    def test_reasoning_keepalive_is_throttled_not_oneshot(self):
        # Revue adversariale: le reasoning est une fenêtre SSE muette (minutes en
        # effort high) -> les proxies idle-timeout coupaient le flux. Le battement
        # "thinking" doit se RÉPÉTER (throttlé), pas partir une seule fois.
        p = self._provider()
        p.KEEPALIVE_SECONDS = 0.0  # chaque chunk silencieux bat
        p._client.chat.completions.create.return_value = iter([
            _ds_chunk(reasoning="a"),
            _ds_chunk(reasoning="b"),
            _ds_chunk(reasoning="c"),
            _ds_chunk(content="fin", finish="stop"),
        ])
        events = list(p.stream_with_history([{"role": "user", "content": "hi"}]))
        thinkings = [e for e in events if e["type"] == "thinking"]
        self.assertGreaterEqual(len(thinkings), 3)
        # et le reasoning complet reste dans l'écho
        self.assertEqual(events[-1]["response"].raw_content.get("reasoning_content"),
                         "abc")


def _gm_chunk(parts):
    return SimpleNamespace(candidates=[SimpleNamespace(
        finish_reason=None,
        content=SimpleNamespace(parts=parts))])


class GeminiStreamTests(TestCase):
    def _provider(self):
        p = GeminiProvider.__new__(GeminiProvider)
        p.model_name = "gemini-2.5-flash"
        p.client = MagicMock()
        return p

    def test_stream_text_deltas_and_final(self):
        p = self._provider()
        p.client.models.generate_content_stream.return_value = iter([
            _gm_chunk([SimpleNamespace(text="Sal")]),
            _gm_chunk([SimpleNamespace(text="ut !")]),
        ])
        events = list(p.stream_with_history([{"role": "user", "content": "hi"}]))
        self.assertEqual(
            [e["text"] for e in events if e["type"] == "text_delta"],
            ["Sal", "ut !"])
        final = events[-1]["response"]
        self.assertEqual(final.text, "Salut !")
        # raw_content consolidé en un seul bloc texte
        self.assertEqual(final.raw_content, [{"type": "text", "text": "Salut !"}])

    def test_empty_stream_falls_back_to_retried_path(self):
        p = self._provider()
        p.client.models.generate_content_stream.return_value = iter([])
        with patch.object(GeminiProvider, "generate_with_history",
                          return_value=LLMResponse(text="repêché")) as gen:
            events = list(p.stream_with_history([{"role": "user", "content": "hi"}]))
        gen.assert_called_once()
        self.assertEqual(
            [e["text"] for e in events if e["type"] == "text_delta"],
            ["repêché"])
        self.assertEqual(events[-1]["response"].text, "repêché")

    def test_stream_exception_falls_back(self):
        p = self._provider()
        p.client.models.generate_content_stream.side_effect = RuntimeError("boom")
        with patch.object(GeminiProvider, "generate_with_history",
                          return_value=LLMResponse(text="repêché")):
            events = list(p.stream_with_history([{"role": "user", "content": "hi"}]))
        self.assertEqual(events[-1]["response"].text, "repêché")

    def test_midstream_exception_emits_stream_reset_before_fallback(self):
        # Revue adversariale: exception APRÈS des deltas déjà émis -> sans reset,
        # le client concatène « partiel + texte complet du fallback ».
        def broken_stream():
            yield _gm_chunk([SimpleNamespace(text="Voici ton pl")])
            raise RuntimeError("503 mid-flux")

        p = self._provider()
        p.client.models.generate_content_stream.return_value = broken_stream()
        with patch.object(GeminiProvider, "generate_with_history",
                          return_value=LLMResponse(text="Voici ton planning complet.")):
            events = list(p.stream_with_history([{"role": "user", "content": "hi"}]))
        types = [e["type"] for e in events]
        self.assertIn("stream_reset", types)
        # le reset arrive APRÈS le delta partiel et AVANT le delta du fallback
        self.assertLess(types.index("text_delta"), types.index("stream_reset"))
        self.assertLess(types.index("stream_reset"),
                        len(types) - 1 - types[::-1].index("text_delta"))


class AgentEmptyStreamTurnTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("emptier", password="pw-123456")
        self.agent = PlannerAgent()

    def test_empty_streamed_turn_retries_non_streamed(self):
        # Candidat vide Gemini (flakiness connue, systématique en streaming sur
        # les tours à outil): un final VIDE non-erreur doit être rejoué via
        # _generate_with_failover au lieu de finir sur le filet « reformule ».
        fake = FakeStreamLLM([[_final("")]])  # stream « réussi » mais vide
        fake.generate_with_history = lambda **kw: LLMResponse(
            text="Réponse repêchée.")
        with patch.object(self.agent, "_build_provider", return_value=fake):
            events = list(self.agent.process_message_stream(self.user, "salut"))
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["response"], "Réponse repêchée.")

    def test_zero_tool_mutation_claim_is_replayed_non_streamed(self):
        # Mensonge zéro-outil (spécifique au stream Gemini, vérifié prod A/B):
        # le tour streamé PRÉTEND « j'ai ajouté » sans qu'AUCUN outil n'ait
        # tourné -> le texte est jeté (turn_discard) et le tour rejoué en
        # non-streamé, dont la réponse fait foi.
        fake = FakeStreamLLM([[
            {"type": "text_delta", "text": "J'ai ajouté ton cours de biologie."},
            _final("J'ai ajouté ton cours de biologie."),
        ]])
        fake.generate_with_history = lambda **kw: LLMResponse(
            text="Peux-tu préciser l'horaire du cours ?")
        with patch.object(self.agent, "_build_provider", return_value=fake):
            events = list(self.agent.process_message_stream(
                self.user, "ajoute mon cours de biologie le jeudi"))
        types = [e["type"] for e in events]
        self.assertIn("turn_discard", types)  # le texte suspect est jeté
        self.assertEqual(events[-1]["response"], "Peux-tu préciser l'horaire du cours ?")

    def test_true_recap_after_real_tool_is_not_replayed(self):
        # Après un VRAI appel d'outil dans la requête, un texte « c'est fait »
        # est légitime: pas de rerun, pas de discard du tour final.
        tool_turn = [_final("", function_calls=[
            FunctionCall(name="list_blocks", args={}, call_id="c1")])]
        confirm_turn = [
            {"type": "text_delta", "text": "C'est fait, j'ai vérifié tes blocs."},
            _final("C'est fait, j'ai vérifié tes blocs."),
        ]
        fake = FakeStreamLLM([tool_turn, confirm_turn])
        with patch.object(self.agent, "_build_provider", return_value=fake):
            events = list(self.agent.process_message_stream(self.user, "vérifie mes blocs"))
        self.assertEqual(events[-1]["response"], "C'est fait, j'ai vérifié tes blocs.")
        # un seul turn_discard maximum n'est pas attendu ici (aucun texte
        # streamé sur le tour outil, et le tour final est accepté tel quel)
        self.assertNotIn("turn_discard", [e["type"] for e in events])


class AgentStreamResetTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("resetter", password="pw-123456")
        self.agent = PlannerAgent()

    def test_stream_reset_becomes_turn_discard(self):
        fake = FakeStreamLLM([[
            {"type": "text_delta", "text": "partiel"},
            {"type": "stream_reset"},
            {"type": "text_delta", "text": "Réponse complète."},
            _final("Réponse complète."),
        ]])
        with patch.object(self.agent, "_build_provider", return_value=fake):
            events = list(self.agent.process_message_stream(self.user, "salut"))
        types = [e["type"] for e in events]
        self.assertIn("turn_discard", types)
        self.assertLess(types.index("turn_discard"),
                        len(types) - 1 - types[::-1].index("delta"))
        self.assertEqual(events[-1]["response"], "Réponse complète.")
