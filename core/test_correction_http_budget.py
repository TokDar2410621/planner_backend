"""Réservation avant upload et même échéance dans le tour détaché."""
import asyncio
import time
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TransactionTestCase, override_settings
from django.utils import timezone
from rest_framework.test import APIClient

from core.models import BudgetJetonsJournalier, ReservationBudgetJetons
from services.agent_v2.trace_tour import trace_courante


class HttpBudgetTests(TransactionTestCase):
    def setUp(self):
        cache.clear()
        self.user = get_user_model().objects.create_user('budget-http')
        self.user.profile.ai_consent_at = timezone.now()
        self.user.profile.save(update_fields=['ai_consent_at'])
        self.client = APIClient()
        self.client.force_authenticate(self.user)

    @override_settings(AGENT_V2_BUDGET_JETONS_JOUR=100)
    def test_budget_refuse_avant_upload_et_rejoue_sans_reserver(self):
        BudgetJetonsJournalier.objects.create(user=self.user, jour=timezone.localdate(), jetons=100)
        identity = str(uuid4())
        with patch('core.views._create_chat_attachment') as upload, \
             patch('core.views._agent_pour') as agent:
            for endpoint in ('/api/chat/', '/api/chat/stream/'):
                response = self.client.post(endpoint, {'message': 'Bonjour', 'turn_id': identity})
                self.assertEqual(response.status_code, 429)
        upload.assert_not_called()
        agent.assert_not_called()
        self.assertEqual(ReservationBudgetJetons.objects.count(), 0)

    def test_upload_et_agent_partagent_reservation_puis_liberent(self):
        traces = []
        def upload(*args):
            traces.append(trace_courante.get())
            return None, None
        def process(*args, **kwargs):
            traces.append(trace_courante.get())
            self.assertGreater(BudgetJetonsJournalier.objects.get(user=self.user).jetons_reserves, 0)
            return {'response': 'Salut'}
        agent = Mock()
        agent.process_message.side_effect = process
        with patch('core.views._create_chat_attachment', side_effect=upload), \
             patch('core.views._agent_pour', return_value=agent):
            response = self.client.post('/api/chat/', {'message': 'Bonjour'})
        self.assertEqual(response.status_code, 200)
        self.assertIs(traces[0], traces[1])
        self.assertEqual(traces[0].reservation.user_id, self.user.pk)
        self.assertEqual(BudgetJetonsJournalier.objects.get(user=self.user).jetons_reserves, 0)
        self.assertTrue(ReservationBudgetJetons.objects.get().cloturee)
        self.assertIsNone(trace_courante.get())

    def test_erreur_upload_libere_reservation(self):
        with patch('core.views._create_chat_attachment', side_effect=ValueError('upload')):
            response = self.client.post('/api/chat/', {'message': 'Bonjour'})
        self.assertEqual(response.status_code, 500)
        self.assertTrue(ReservationBudgetJetons.objects.get().cloturee)
        self.assertEqual(BudgetJetonsJournalier.objects.get(user=self.user).jetons_reserves, 0)

    def test_sse_transporte_trace_dans_worker_et_la_ferme(self):
        traces = []
        def upload(*args):
            traces.append(trace_courante.get())
            return None, None
        def stream(*args, **kwargs):
            traces.append(trace_courante.get())
            yield {'type': 'done', 'response': 'Salut'}
        agent = Mock()
        agent.process_message_stream.side_effect = stream
        with patch('core.views._create_chat_attachment', side_effect=upload), \
             patch('core.views._agent_pour', return_value=agent):
            response = self.client.post('/api/chat/stream/', {'message': 'Bonjour'})
            async def consume():
                return [part async for part in response.streaming_content]
            parts = asyncio.run(consume())
        self.assertTrue(parts)
        self.assertIs(traces[0], traces[1])
        self.assertTrue(ReservationBudgetJetons.objects.get().cloturee)
        self.assertIsNone(trace_courante.get())

    def test_document_standalone_garde_reservation_jusqua_future(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        future = Future()
        with patch('core.views.DocumentProcessor.process_document_async', return_value=future):
            response = self.client.post('/api/documents/', {
                'file': SimpleUploadedFile('test.pdf', b'%PDF-1.4\n%%EOF', content_type='application/pdf'),
                'document_type': 'other',
            })
        self.assertEqual(response.status_code, 201)
        self.assertFalse(ReservationBudgetJetons.objects.get().cloturee)
        future.set_result({})
        self.assertTrue(ReservationBudgetJetons.objects.get().cloturee)
        self.assertEqual(BudgetJetonsJournalier.objects.get(user=self.user).jetons_reserves, 0)

    def test_chat_termine_garde_la_reservation_du_document_encore_actif(self):
        from services.document_processor import DocumentProcessor
        future = Future()
        traces = []
        self.addCleanup(lambda: future.set_result({}) if not future.done() else None)
        def upload(*args):
            traces.append(trace_courante.get())
            return SimpleNamespace(_processing_future=future), None
        agent = Mock()
        agent.process_message.return_value = {'response': 'Analyse encore en cours'}
        with patch('core.views._create_chat_attachment', side_effect=upload), \
             patch('core.views._agent_pour', return_value=agent):
            response = self.client.post('/api/chat/', {'message': 'Analyse le document'})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(future.done())
        self.assertFalse(ReservationBudgetJetons.objects.get().cloturee)
        self.assertGreater(traces[0].restant(), 0)
        token = trace_courante.set(traces[0])
        try:
            DocumentProcessor._verifier_echeance()
        finally:
            trace_courante.reset(token)
        future.set_result({})
        self.assertTrue(ReservationBudgetJetons.objects.get().cloturee)
        self.assertEqual(BudgetJetonsJournalier.objects.get(user=self.user).jetons_reserves, 0)

    def pending_document(self):
        future = Future()
        self.addCleanup(lambda: future.set_result({}) if not future.done() else None)
        return future, SimpleNamespace(_processing_future=future)

    def assert_budget_closed(self):
        self.assertTrue(ReservationBudgetJetons.objects.get().cloturee)
        self.assertEqual(BudgetJetonsJournalier.objects.get(user=self.user).jetons_reserves, 0)

    def test_document_fini_ne_ferme_pas_le_chat_encore_actif(self):
        future, attachment = self.pending_document()
        def process(*args, **kwargs):
            future.set_result({})
            self.assertFalse(ReservationBudgetJetons.objects.get().cloturee)
            self.assertFalse(trace_courante.get().reservation.fermee)
            return {'response': 'Document traité'}
        agent = Mock()
        agent.process_message.side_effect = process
        with patch('core.views._create_chat_attachment', return_value=(attachment, None)), \
             patch('core.views._agent_pour', return_value=agent):
            response = self.client.post('/api/chat/', {'message': 'Document'})
        self.assertEqual(response.status_code, 200)
        self.assert_budget_closed()

    def test_chat_en_erreur_attend_la_fin_du_document_puis_libere(self):
        future, attachment = self.pending_document()
        agent = Mock()
        agent.process_message.side_effect = RuntimeError('agent local en panne')
        with patch('core.views._create_chat_attachment', return_value=(attachment, None)), \
             patch('core.views._agent_pour', return_value=agent):
            response = self.client.post('/api/chat/', {'message': 'Document'})
        self.assertEqual(response.status_code, 500)
        self.assertFalse(ReservationBudgetJetons.objects.get().cloturee)
        future.set_exception(RuntimeError('extraction locale en panne'))
        self.assert_budget_closed()

    def test_sse_termine_garde_le_document_actif_sans_bloquer_la_reponse(self):
        future, attachment = self.pending_document()
        traces = []
        def upload(*args):
            traces.append(trace_courante.get())
            return attachment, None
        def stream(*args, **kwargs):
            traces.append(trace_courante.get())
            yield {'type': 'done', 'response': 'Analyse en cours'}
        agent = Mock()
        agent.process_message_stream.side_effect = stream
        with patch('core.views._create_chat_attachment', side_effect=upload), \
             patch('core.views._agent_pour', return_value=agent):
            response = self.client.post('/api/chat/stream/', {'message': 'Document'})
            async def consume():
                return [part async for part in response.streaming_content]
            parts = asyncio.run(consume())
        self.assertTrue(parts)
        self.assertIs(traces[0], traces[1])
        self.assertFalse(future.done())
        self.assertFalse(ReservationBudgetJetons.objects.get().cloturee)
        future.set_result({})
        self.assert_budget_closed()

    def test_lancement_sse_en_erreur_garde_document_jusqua_annulation(self):
        future, attachment = self.pending_document()
        with patch('core.views._create_chat_attachment', return_value=(attachment, None)), \
             patch('core.views.lancer_flux', side_effect=RuntimeError('pool indisponible')):
            response = self.client.post('/api/chat/stream/', {'message': 'Document'})
        self.assertEqual(response.status_code, 500)
        self.assertFalse(ReservationBudgetJetons.objects.get().cloturee)
        self.assertTrue(future.cancel())
        self.assert_budget_closed()

    def test_drain_sse_annule_avant_demarrage_libere_sans_document(self):
        drain = Future()
        async def empty():
            if False:
                yield ''
        with patch('core.views.lancer_flux', return_value=(empty(), drain)):
            response = self.client.post('/api/chat/stream/', {'message': 'Bonjour'})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(ReservationBudgetJetons.objects.get().cloturee)
        self.assertTrue(drain.cancel())
        self.assert_budget_closed()

    def test_echeance_commune_libere_un_document_bloque_sans_double_remboursement(self):
        from services.document_processor import DocumentProcessor
        future, attachment = self.pending_document()
        traces = []
        def upload(*args):
            traces.append(trace_courante.get())
            return attachment, None
        agent = Mock()
        agent.process_message.return_value = {'response': 'Analyse en cours'}
        with patch('core.views.threading.Timer') as timer, \
             patch('core.views._create_chat_attachment', side_effect=upload), \
             patch('core.views._agent_pour', return_value=agent):
            response = self.client.post('/api/chat/', {'message': 'Document'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(timer.call_count, 1)
        delay, callback = timer.call_args.args
        self.assertGreater(delay, 0)
        self.assertLessEqual(delay, 180)
        self.assertFalse(ReservationBudgetJetons.objects.get().cloturee)
        traces[0].echeance = time.monotonic() - 1
        callback()
        self.assertFalse(future.done())
        self.assert_budget_closed()
        token = trace_courante.set(traces[0])
        try:
            with self.assertRaises(TimeoutError):
                DocumentProcessor._verifier_echeance()
        finally:
            trace_courante.reset(token)
        future.set_result({})
        callback()
        self.assert_budget_closed()
        self.assertEqual(BudgetJetonsJournalier.objects.get(user=self.user).jetons, 0)

    def test_document_standalone_ferme_a_lecheance_si_future_ne_revient_pas(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        future, _ = self.pending_document()
        with patch('core.views.threading.Timer') as timer, \
             patch('core.views.DocumentProcessor.process_document_async', return_value=future):
            response = self.client.post('/api/documents/', {
                'file': SimpleUploadedFile('test.pdf', b'%PDF-1.4\n%%EOF', content_type='application/pdf'),
                'document_type': 'other',
            })
        self.assertEqual(response.status_code, 201)
        self.assertFalse(ReservationBudgetJetons.objects.get().cloturee)
        timer.call_args.args[1]()
        self.assert_budget_closed()
