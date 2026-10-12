"""Contrat HTTP du tour durable, sans fournisseur ni stockage externe."""
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from datetime import timedelta
from uuid import uuid4

from django.contrib.auth.models import User
from django.test import TestCase, TransactionTestCase, override_settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.cache import cache
from django.db import IntegrityError, connection, close_old_connections, transaction
from django.utils import timezone
from rest_framework.test import APIClient
from core.models import ChatTurn


class DurableTransportTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(username='transport')
        self.user.profile.ai_consent_at = timezone.now()
        self.user.profile.save(update_fields=['ai_consent_at'])
        self.client = APIClient()
        self.client.force_authenticate(self.user)
        self.agent = Mock()
        self.agent.process_message.return_value = {'response': 'Réponse sauvée'}
        self.agent_patch = patch('core.views._agent_pour', return_value=self.agent)
        self.agent_patch.start()
        self.addCleanup(self.agent_patch.stop)

    def post(self, turn_id, message='Même instruction'):
        return self.client.post('/api/chat/', {'turn_id': str(turn_id), 'message': message})

    def test_retry_replays_saved_result_once(self):
        turn_id = uuid4()
        first = self.post(turn_id)
        second = self.post(turn_id)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.data, first.data)
        self.assertEqual(self.agent.process_message.call_count, 1)
        self.assertEqual(first.data['turn_id'], str(turn_id))

    def test_same_text_new_identity_is_a_new_turn(self):
        first_id, second_id = uuid4(), uuid4()
        self.agent.process_message.side_effect = [
            {'response': 'Première réponse', 'assistant_message_id': 11},
            {'response': 'Deuxième réponse', 'assistant_message_id': 22},
        ]
        self.post(first_id)
        self.post(second_id)
        self.assertEqual(self.agent.process_message.call_count, 2)
        second = self.client.get(f'/api/chat/turns/{second_id}/').data['result']
        self.assertEqual(second['response'], 'Deuxième réponse')
        self.assertEqual(second['assistant_message_id'], 22)
        self.assertEqual(self.post(first_id).data['response'], 'Première réponse')
        self.assertEqual(self.agent.process_message.call_count, 2)

    def test_payload_mismatch_rejected_without_execution(self):
        turn_id = uuid4()
        self.post(turn_id)
        changed = self.post(turn_id, 'Une autre instruction')
        self.assertEqual(changed.status_code, 409)
        self.assertEqual(changed.data['code'], 'turn_payload_mismatch')
        self.assertEqual(self.agent.process_message.call_count, 1)

    def test_result_recovered_by_id_and_isolated_per_user(self):
        turn_id = uuid4()
        self.post(turn_id)
        result = self.client.get(f'/api/chat/turns/{turn_id}/')
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.data['result']['response'], 'Réponse sauvée')
        other = User.objects.create_user(username='other-transport')
        self.client.force_authenticate(other)
        self.assertEqual(self.client.get(f'/api/chat/turns/{turn_id}/').status_code, 404)

    def test_invalid_id_never_invokes_agent(self):
        response = self.post('wrong-uuid')
        self.assertEqual(response.status_code, 400)
        self.agent.process_message.assert_not_called()

    def test_old_client_remains_compatible_without_deduplication(self):
        self.client.post('/api/chat/', {'message': 'bonjour'})
        self.client.post('/api/chat/', {'message': 'bonjour'})
        self.assertEqual(self.agent.process_message.call_count, 2)

    def test_same_id_other_user_is_independent(self):
        turn_id = uuid4()
        self.post(turn_id)
        other = User.objects.create_user(username='separate-transport')
        other.profile.ai_consent_at = timezone.now()
        other.profile.save(update_fields=['ai_consent_at'])
        self.client.force_authenticate(other)
        self.post(turn_id)
        self.assertEqual(self.agent.process_message.call_count, 2)

    def test_unique_constraint_is_enforced(self):
        turn_id = uuid4()
        self.post(turn_id)
        with self.assertRaises(IntegrityError), transaction.atomic():
            ChatTurn.objects.create(user=self.user, turn_id=turn_id, payload_hash='x')

    def test_upload_retry_is_deduplicated_before_storage(self):
        turn_id = str(uuid4())
        with patch('core.views._create_chat_attachment', return_value=(None, None)) as upload:
            for _ in range(2):
                response = self.client.post('/api/chat/', {
                    'turn_id': turn_id, 'message': 'importe',
                    'attachment': SimpleUploadedFile('horaire.txt', b'same bytes', content_type='text/plain'),
                })
                self.assertEqual(response.status_code, 200)
            self.assertEqual(upload.call_count, 1)
            mismatch = self.client.post('/api/chat/', {
                'turn_id': turn_id, 'message': 'importe',
                'attachment': SimpleUploadedFile('horaire.txt', b'different', content_type='text/plain'),
            })
            self.assertEqual(mismatch.status_code, 409)
            self.assertEqual(upload.call_count, 1)

    @override_settings(CHAT_TURN_STALE_SECONDS=10)
    def test_abandoned_worker_is_reported_without_blind_retry(self):
        turn_id = uuid4()
        self.post(turn_id)
        ChatTurn.objects.filter(turn_id=turn_id).update(
            status='running', result={}, updated_at=timezone.now() - timedelta(seconds=11),
        )
        status_response = self.client.get(f'/api/chat/turns/{turn_id}/')
        self.assertEqual(status_response.data['status'], 'interrupted')
        retried = self.post(turn_id)
        self.assertEqual(retried.status_code, 409)
        self.assertEqual(retried.data['code'], 'turn_interrupted')
        self.assertEqual(self.agent.process_message.call_count, 1)

    def test_failed_turn_never_reexecutes(self):
        turn_id = uuid4()
        self.agent.process_message.side_effect = RuntimeError('local failure')
        self.assertEqual(self.post(turn_id).status_code, 500)
        self.assertEqual(self.post(turn_id).status_code, 500)
        self.assertEqual(self.agent.process_message.call_count, 1)

    def test_durable_result_never_persists_private_reasoning(self):
        from core.chat_turns import finish
        turn = ChatTurn.objects.create(user=self.user, turn_id=uuid4(), payload_hash='x')
        live = finish(turn, {'response': 'ok', 'raisonnement': 'Éphémère'})
        turn.refresh_from_db()
        self.assertEqual(live['raisonnement'], 'Éphémère')
        self.assertNotIn('raisonnement', turn.result)

    def test_migration_matches_current_models(self):
        from django.apps import apps
        from django.db.migrations.autodetector import MigrationAutodetector
        from django.db.migrations.loader import MigrationLoader
        from django.db.migrations.state import ProjectState
        loader = MigrationLoader(None)
        changes = MigrationAutodetector(loader.project_state(), ProjectState.from_apps(apps)).changes(graph=loader.graph)
        self.assertEqual(changes.get('core', []), [])

    @override_settings(CHAT_TURN_STALE_SECONDS=10)
    def test_worker_queued_past_deadline_does_not_start_agent(self):
        held = []
        async def empty():
            if False:
                yield ''
        def enqueue(events, _user_id):
            held.append(events)
            return empty(), None
        turn_id = uuid4()
        with patch('core.views.lancer_flux', side_effect=enqueue):
            response = self.client.post('/api/chat/stream/', {'message': 'attente', 'turn_id': str(turn_id)})
        self.assertEqual(response.status_code, 200)
        ChatTurn.objects.filter(turn_id=turn_id).update(updated_at=timezone.now() - timedelta(seconds=11))
        events = list(held[0])
        self.assertEqual(events[0]['type'], 'error')
        self.agent.process_message_stream.assert_not_called()
        self.assertEqual(ChatTurn.objects.get(turn_id=turn_id).status, 'interrupted')

    def test_stream_then_post_share_admission_and_saved_terminal(self):
        from asgiref.sync import async_to_sync

        turn_id = uuid4()
        captured = []
        self.agent.process_message_stream.return_value = iter([
            {'type': 'delta', 'text': 'partiel'},
            {'type': 'done', 'response': 'final', 'question_posee': True},
        ])

        def drain(events, _user_id):
            captured.extend(events)
            async def reader():
                for event in captured:
                    yield str(event)
            return reader(), None

        with patch('core.views.lancer_flux', side_effect=drain):
            streamed = self.client.post('/api/chat/stream/', {'turn_id': str(turn_id), 'message': 'Même instruction'})
        self.assertEqual(streamed.status_code, 200)
        retried = self.post(turn_id)
        self.assertEqual(retried.data['response'], 'final')
        self.assertTrue(retried.data['question_posee'])
        self.assertEqual(captured[-1]['turn_id'], str(turn_id))
        self.agent.process_message.assert_not_called()
        self.agent.process_message_stream.assert_called_once()


class ConcurrentAdmissionTests(TransactionTestCase):
    def test_overlapping_requests_execute_only_once_without_long_db_lock(self):
        cache.clear()
        user = User.objects.create_user(username='concurrent-transport')
        user.profile.ai_consent_at = timezone.now()
        user.profile.save(update_fields=['ai_consent_at'])
        entered, finish = Event(), Event()
        turn_id = str(uuid4())

        def process(*_args, **_kwargs):
            self.assertFalse(connection.in_atomic_block)
            entered.set()
            if not finish.wait(5):
                raise AssertionError('Concurrent control never released')
            return {'response': 'une fois'}

        def post():
            close_old_connections()
            client = APIClient()
            client.force_authenticate(user)
            try:
                return client.post('/api/chat/', {'turn_id': turn_id, 'message': 'instruction'})
            finally:
                close_old_connections()

        agent = Mock(process_message=Mock(side_effect=process))
        with patch('core.views._agent_pour', return_value=agent), ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(post)
            try:
                self.assertTrue(entered.wait(5))
                second = post()
                self.assertEqual(second.status_code, 202)
                self.assertEqual(second.data['code'], 'turn_pending')
                self.assertEqual(agent.process_message.call_count, 1)
            finally:
                finish.set()
            self.assertEqual(first.result(timeout=5).status_code, 200)
            self.assertEqual(post().data['response'], 'une fois')
            self.assertEqual(agent.process_message.call_count, 1)
