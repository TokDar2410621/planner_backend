"""
Consentement IA (Apple 5.1.2(i)): aucun contenu utilisateur ne part vers une
IA tierce (Gemini, Claude, DeepSeek, Hugging Face) sans consentement explicite,
et le consentement est révocable.
"""
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import TestCase
from rest_framework.test import APIClient

from core.models import UploadedDocument
from services.agent.agent import PlannerAgent


class AIConsentEndpointTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='u', password='x')
        self.client = APIClient()
        self.client.force_authenticate(self.user)

    def test_etat_par_defaut_non_consenti(self):
        res = self.client.get('/api/ai/consent/')
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()['granted'])
        self.assertIsNone(res.json()['granted_at'])

    def test_consentir_puis_revoquer(self):
        res = self.client.post('/api/ai/consent/', {'granted': True}, format='json')
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()['granted'])
        self.assertIsNotNone(res.json()['granted_at'])

        self.user.profile.refresh_from_db()
        self.assertIsNotNone(self.user.profile.ai_consent_at)

        res = self.client.post('/api/ai/consent/', {'granted': False}, format='json')
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()['granted'])
        self.user.profile.refresh_from_db()
        self.assertIsNone(self.user.profile.ai_consent_at)

    def test_granted_non_booleen_refuse(self):
        for mauvais in ('oui', 1, None):
            res = self.client.post('/api/ai/consent/', {'granted': mauvais}, format='json')
            self.assertEqual(res.status_code, 400, f'granted={mauvais!r} doit faire 400')

    def test_anonyme_refuse(self):
        res = APIClient().get('/api/ai/consent/')
        self.assertEqual(res.status_code, 401)


class AIGateTests(TestCase):
    """Sans consentement, TOUTES les portes vers un LLM tiers renvoient 403."""

    ENDPOINTS_POST = [
        ('/api/chat/', {'message': 'salut'}),
        ('/api/chat/stream/', {'message': 'salut'}),
        ('/api/chat/quick-replies/', {'message': 'salut', 'response': 'ok'}),
        ('/api/insights/natural-language/', {'message': 'planifie du sport demain'}),
    ]

    def setUp(self):
        self.user = User.objects.create_user(username='u2', password='x')
        self.client = APIClient()
        self.client.force_authenticate(self.user)

    def _consent(self):
        self.client.post('/api/ai/consent/', {'granted': True}, format='json')

    def test_sans_consentement_403_partout_et_agent_jamais_appele(self):
        with patch.object(PlannerAgent, '__init__', side_effect=AssertionError(
            'PlannerAgent instancié malgré le refus de consentement'
        )):
            for url, payload in self.ENDPOINTS_POST:
                res = self.client.post(url, payload, format='multipart'
                                       if 'chat/' in url and 'quick' not in url else 'json')
                self.assertEqual(res.status_code, 403, f'{url} doit être gaté')
                self.assertEqual(res.json()['code'], 'ai_consent_required', url)

    def test_upload_document_gate(self):
        f = SimpleUploadedFile('horaire.pdf', b'%PDF-1.4 fake', content_type='application/pdf')
        with patch('core.views.DocumentProcessor') as proc:
            res = self.client.post(
                '/api/documents/', {'file': f, 'document_type': 'course_schedule'},
                format='multipart',
            )
        self.assertEqual(res.status_code, 403)
        self.assertEqual(res.json()['code'], 'ai_consent_required')
        proc.assert_not_called()
        self.assertEqual(UploadedDocument.objects.count(), 0,
                         'aucun document ne doit être créé sans consentement')

    def test_avec_consentement_le_chat_passe(self):
        self._consent()
        with patch.object(PlannerAgent, 'process_message', return_value={
            'response': 'Bonjour !', 'quick_replies': [],
        }):
            res = self.client.post('/api/chat/', {'message': 'salut'}, format='multipart')
        self.assertEqual(res.status_code, 200)

    def test_la_revocation_referme_la_porte(self):
        self._consent()
        self.client.post('/api/ai/consent/', {'granted': False}, format='json')
        res = self.client.post('/api/chat/', {'message': 'salut'}, format='multipart')
        self.assertEqual(res.status_code, 403)


class CronConsentTests(TestCase):
    """Le worker de documents ne traite jamais un utilisateur non consentant."""

    def _doc(self, user):
        return UploadedDocument.objects.create(
            user=user,
            file=SimpleUploadedFile('h.pdf', b'%PDF-1.4', content_type='application/pdf'),
            document_type='course_schedule',
            processed=False,
        )

    def test_pending_sans_consentement_saute(self):
        user = User.objects.create_user(username='u3', password='x')
        doc = self._doc(user)
        with patch('services.document_processor.DocumentProcessor.process_document',
                   side_effect=AssertionError('LLM appelé sans consentement')):
            call_command('process_pending_documents')
        doc.refresh_from_db()
        self.assertFalse(doc.processed)

    def test_pending_avec_consentement_traite(self):
        from django.utils import timezone
        user = User.objects.create_user(username='u4', password='x')
        user.profile.ai_consent_at = timezone.now()
        user.profile.save(update_fields=['ai_consent_at'])
        doc = self._doc(user)
        with patch('services.document_processor.DocumentProcessor.process_document',
                   return_value=True) as proc:
            call_command('process_pending_documents')
        proc.assert_called_once()
