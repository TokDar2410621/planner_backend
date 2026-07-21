"""Outbound webhooks: signing, dispatch filtering, endpoints."""
import hashlib
import hmac
import json
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from core.models import WebhookEndpoint
from services import webhooks


class WebhookSigningTest(APITestCase):
    def test_sign_is_hmac_sha256(self):
        body = b'{"event":"x"}'
        sig = webhooks._sign('secret', body)
        expected = 'sha256=' + hmac.new(b'secret', body, hashlib.sha256).hexdigest()
        self.assertEqual(sig, expected)


class WebhookWantsTest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user('hookuser', password='pw-hook-12345')

    def test_empty_events_means_all(self):
        h = WebhookEndpoint.objects.create(user=self.user, url='https://x.test', events=[])
        self.assertTrue(h.wants('task.created'))
        self.assertTrue(h.wants('task.completed'))

    def test_specific_events_filter(self):
        h = WebhookEndpoint.objects.create(
            user=self.user, url='https://x.test', events=['task.completed']
        )
        self.assertFalse(h.wants('task.created'))
        self.assertTrue(h.wants('task.completed'))

    def test_inactive_never_wants(self):
        h = WebhookEndpoint.objects.create(
            user=self.user, url='https://x.test', events=[], active=False
        )
        self.assertFalse(h.wants('task.created'))


class WebhookDispatchTest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user('dispatcher', password='pw-disp-12345')

    def test_dispatch_sync_posts_signed_payload(self):
        WebhookEndpoint.objects.create(
            user=self.user, url='https://hook.test/in', secret='s3cr3t', events=[]
        )
        with patch.object(webhooks, 'requests') as mock_requests:
            mock_requests.post.return_value = MagicMock(status_code=200)
            mock_requests.RequestException = Exception
            results = webhooks.dispatch_sync(
                self.user, 'task.completed', {'task': {'id': 1}}
            )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], '200')
        # Verify body + signature header.
        _, kwargs = mock_requests.post.call_args
        body = kwargs['data']
        payload = json.loads(body)
        self.assertEqual(payload['event'], 'task.completed')
        self.assertEqual(payload['data']['task']['id'], 1)
        sig = kwargs['headers']['X-Planner-Signature']
        expected = 'sha256=' + hmac.new(b's3cr3t', body, hashlib.sha256).hexdigest()
        self.assertEqual(sig, expected)

    def test_dispatch_skips_non_matching_event(self):
        WebhookEndpoint.objects.create(
            user=self.user, url='https://hook.test/in', events=['reminder.sent']
        )
        with patch.object(webhooks, 'requests') as mock_requests:
            n = webhooks.dispatch(self.user, 'task.created', {'x': 1})
        self.assertEqual(n, 0)
        mock_requests.post.assert_not_called()


class WebhookEndpointAPITest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user('apiuser', password='pw-api-123456')
        self.client.force_authenticate(self.user)

    def test_create_requires_valid_url(self):
        r = self.client.post(reverse('webhooks'), {'url': 'ftp://nope'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_400_BAD_REQUEST)

    def test_create_rejects_unknown_event(self):
        r = self.client.post(
            reverse('webhooks'),
            {'url': 'https://n8n.test/webhook', 'events': ['nope.event']},
            format='json',
        )
        self.assertEqual(r.status_code, status.HTTP_400_BAD_REQUEST)

    def test_create_returns_secret_once(self):
        r = self.client.post(
            reverse('webhooks'),
            {'url': 'https://n8n.test/webhook', 'events': ['task.created']},
            format='json',
        )
        self.assertEqual(r.status_code, status.HTTP_201_CREATED)
        self.assertIn('secret', r.data)  # revealed at creation
        self.assertTrue(r.data['secret'])
        # List view must NOT reveal the secret.
        lst = self.client.get(reverse('webhooks'))
        self.assertNotIn('secret', lst.data[0])
        self.assertTrue(lst.data[0]['has_secret'])

    def test_delete_webhook(self):
        h = WebhookEndpoint.objects.create(user=self.user, url='https://x.test')
        r = self.client.delete(reverse('webhooks-detail', kwargs={'hook_id': h.id}))
        self.assertEqual(r.data['deleted'], 1)
        self.assertEqual(WebhookEndpoint.objects.count(), 0)

    def test_cannot_touch_other_users_webhook(self):
        other = User.objects.create_user('other', password='pw-other-12345')
        h = WebhookEndpoint.objects.create(user=other, url='https://x.test')
        r = self.client.delete(reverse('webhooks-detail', kwargs={'hook_id': h.id}))
        self.assertEqual(r.status_code, status.HTTP_404_NOT_FOUND)
        self.assertEqual(WebhookEndpoint.objects.count(), 1)

    def test_test_endpoint_fires_sample(self):
        WebhookEndpoint.objects.create(user=self.user, url='https://hook.test/in', events=[])
        with patch.object(webhooks, 'requests') as mock_requests:
            mock_requests.post.return_value = MagicMock(status_code=200)
            mock_requests.RequestException = Exception
            r = self.client.post(reverse('webhooks-test'))
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertEqual(r.data['delivered_to'], 1)
