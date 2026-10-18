"""Chat document batches, durable history, and owner-scoped previews."""
from io import BytesIO
from concurrent.futures import Future
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zipfile import ZipFile

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, TransactionTestCase, override_settings
from django.utils import timezone
from rest_framework.test import APIClient
from unittest.mock import patch

from core.chat_turns import _fingerprint
from core.models import ChatTurn, Conversation, ConversationMessage, UploadedDocument
from core.views import _validate_chat_files
from core.lecture_flux import corps_du_flux
from services.agent_v2.conversation import AgentConversation


def pdf(name='one.pdf', size=32):
    return SimpleUploadedFile(name, b'%PDF-' + b'x' * size, content_type='application/pdf')


class ChatAttachmentTests(TestCase):
    def setUp(self):
        media = TemporaryDirectory()
        self.addCleanup(media.cleanup)
        settings_override = override_settings(MEDIA_ROOT=media.name)
        settings_override.enable()
        self.addCleanup(settings_override.disable)
        self.user = User.objects.create_user('owner', password='x')
        self.other = User.objects.create_user('other', password='x')
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    def test_batch_validation_rejects_one_bad_file_before_storage(self):
        valid = pdf()
        bad = SimpleUploadedFile('bad.pdf', b'<script>')
        result = _validate_chat_files([valid, bad])
        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.data['file_index'], 1)
        self.assertFalse(UploadedDocument.objects.exists())
        self.assertIsNone(_validate_chat_files([valid]))
        self.assertEqual(_validate_chat_files([pdf(str(i) + '.pdf') for i in range(6)]).status_code, 400)
        huge_batch = [SimpleNamespace(size=8 * 1024 * 1024) for _ in range(3)]
        self.assertEqual(_validate_chat_files(huge_batch).status_code, 400)

    def test_word_validation_checks_container_and_signature(self):
        data = BytesIO()
        with ZipFile(data, 'w') as archive:
            archive.writestr('[Content_Types].xml', '<Types/>')
            archive.writestr('word/document.xml', '<w:document/>')
        good = SimpleUploadedFile('notes.docx', data.getvalue())
        self.assertIsNone(_validate_chat_files([good]))
        self.assertEqual(_validate_chat_files([SimpleUploadedFile('notes.docx', b'PK\x03\x04bad')]).status_code, 400)

    def test_history_keeps_each_document_and_status(self):
        conversation = Conversation.objects.create(user=self.user)
        message = ConversationMessage.objects.create(user=self.user, conversation=conversation,
                                                     role='user', content='Voici les documents')
        first = UploadedDocument.objects.create(user=self.user, file=pdf('a.pdf'),
                                                file_name='a.pdf', processed=True)
        second = UploadedDocument.objects.create(user=self.user, file=pdf('b.pdf'),
                                                 file_name='b.pdf', processing_error='Extraction impossible.')
        message.attachments.set([first, second])
        body = self.client.get(f'/api/conversations/{conversation.pk}/messages/').json()
        docs = body['messages'][0]['attachments']
        self.assertEqual([item['file_name'] for item in docs], ['a.pdf', 'b.pdf'])
        self.assertEqual([item['status'] for item in docs], ['ready', 'failed'])
        self.assertEqual(docs[1]['processing_error'], 'Extraction impossible.')

    def test_document_status_and_original_are_owner_scoped(self):
        doc = UploadedDocument.objects.create(user=self.user, file=pdf(), file_name='one.pdf')
        status = self.client.get(f'/api/chat/documents/{doc.pk}/')
        self.assertEqual(status.status_code, 200)
        original = self.client.get(f'/api/chat/documents/{doc.pk}/file/')
        self.assertEqual(original.status_code, 200)
        self.assertEqual(b''.join(original.streaming_content)[:5], b'%PDF-')
        self.client.force_authenticate(user=self.other)
        self.assertEqual(self.client.get(f'/api/chat/documents/{doc.pk}/').status_code, 404)
        self.assertEqual(self.client.get(f'/api/chat/documents/{doc.pk}/file/').status_code, 404)
        self.client.force_authenticate(user=None)
        self.assertIn(self.client.get(f'/api/chat/documents/{doc.pk}/').status_code, (401, 403))

    def test_fingerprint_includes_order_and_conversation(self):
        def request(files, conversation):
            return SimpleNamespace(FILES=SimpleNamespace(getlist=lambda key: files),
                                   data={'message': 'analyse', 'conversation_id': conversation})
        one, two = pdf('one.pdf'), pdf('two.pdf')
        self.assertNotEqual(_fingerprint(request([one, two], '1')),
                            _fingerprint(request([two, one], '1')))
        self.assertNotEqual(_fingerprint(request([one, two], '1')),
                            _fingerprint(request([one, two], '2')))


class ChatBatchTransportTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user('batch', password='x')
        self.user.profile.ai_consent_at = timezone.now()
        self.user.profile.save(update_fields=['ai_consent_at'])
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        media = TemporaryDirectory()
        self.addCleanup(media.cleanup)
        settings_override = override_settings(MEDIA_ROOT=media.name)
        settings_override.enable()
        self.addCleanup(settings_override.disable)

    def test_stream_receives_two_files_and_returns_individual_receipts(self):
        seen = {}
        future = Future()
        future.set_result(None)

        def answer(agent, user, message, attachment, **kwargs):
            seen['files'] = [doc.file_name for doc in agent._chat_attachments]
            seen['first'] = attachment.file_name
            yield {'type': 'done', 'response': 'Reçus.', 'quick_replies': []}

        with patch.object(AgentConversation, 'process_message_stream', answer), patch(
            'core.views.DocumentProcessor.process_document_async', return_value=future,
        ):
            response = self.client.post('/api/chat/stream/', {
                'message': 'Analyse tout', 'attachment': [pdf('a.pdf'), pdf('b.pdf')],
            }, format='multipart')
            body = corps_du_flux(response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(seen['files'], ['a.pdf', 'b.pdf'])
        self.assertEqual(seen['first'], 'a.pdf')
        self.assertIn('"file_name": "a.pdf"', body)
        self.assertIn('"file_name": "b.pdf"', body)
        self.assertEqual(UploadedDocument.objects.filter(user=self.user).count(), 2)

    def test_one_invalid_file_rejects_whole_batch_before_turn(self):
        response = self.client.post('/api/chat/stream/', {
            'message': 'Analyse tout',
            'attachment': [pdf('a.pdf'), SimpleUploadedFile('bad.pdf', b'<script>')],
        }, format='multipart')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()['file_index'], 1)
        self.assertFalse(UploadedDocument.objects.filter(user=self.user).exists())
        self.assertFalse(ChatTurn.objects.filter(user=self.user).exists())
