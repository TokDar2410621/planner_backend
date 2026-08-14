"""POST /api/push/send/ (envoi push pour l'agent IA / MCP) + outil send_notification."""
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from core.models import PushSubscription

# VAPID factice: push_configured() doit répondre vrai pendant les tests,
# l'envoi réel est toujours mocké.
VAPID = dict(VAPID_PUBLIC_KEY='pub', VAPID_PRIVATE_KEY='priv', VAPID_SUBJECT='mailto:x@y.z')


@override_settings(**VAPID)
class PushSendEndpointTest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user('notifuser', password='pw-notif-12345')

    def _subscribe(self, n=1):
        for i in range(n):
            PushSubscription.objects.create(
                user=self.user, endpoint=f'https://push.example/{i}', p256dh='k', auth='a'
            )

    def test_anonymous_is_rejected(self):
        r = self.client.post(reverse('push-send'), {'body': 'x'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_send_with_subscriptions(self):
        self._subscribe(2)
        self.client.force_authenticate(self.user)
        from services import push
        with patch.object(push, 'webpush') as mock_wp:
            r = self.client.post(
                reverse('push-send'),
                {'title': 'Rappel', 'body': 'Ton bloc commence bientôt.'},
                format='json',
            )
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertEqual(r.data['sent'], 2)
        self.assertEqual(mock_wp.call_count, 2)

    def test_default_title_is_planner_ai(self):
        self.client.force_authenticate(self.user)
        with patch('services.push.send_to_user', return_value=1) as mock_send:
            r = self.client.post(reverse('push-send'), {'body': 'Hello'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        args, _ = mock_send.call_args
        self.assertEqual(args[1], 'Planner AI')

    def test_no_subscription_returns_zero(self):
        self.client.force_authenticate(self.user)
        from services import push
        with patch.object(push, 'webpush') as mock_wp:
            r = self.client.post(reverse('push-send'), {'body': 'Hello'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertEqual(r.data['sent'], 0)
        mock_wp.assert_not_called()

    def test_missing_body_is_400(self):
        self.client.force_authenticate(self.user)
        r = self.client.post(reverse('push-send'), {'title': 'Sans corps'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_400_BAD_REQUEST)

    def test_length_limits(self):
        self.client.force_authenticate(self.user)
        r = self.client.post(
            reverse('push-send'), {'title': 'T' * 101, 'body': 'ok'}, format='json'
        )
        self.assertEqual(r.status_code, status.HTTP_400_BAD_REQUEST)
        r2 = self.client.post(reverse('push-send'), {'body': 'B' * 501}, format='json')
        self.assertEqual(r2.status_code, status.HTTP_400_BAD_REQUEST)


@override_settings(**VAPID)
class SendNotificationToolTest(TestCase):
    def setUp(self):
        from services.agent.tools.notify import SendNotificationTool
        self.user = User.objects.create_user('tooluser', password='pw-tool-12345')
        self.tool = SendNotificationTool()

    def test_tool_is_registered(self):
        from services.agent.tools import TOOL_MAP
        self.assertIn('send_notification', TOOL_MAP)

    def test_no_subscription_gives_honest_message(self):
        result = self.tool.execute(self.user, body='Hello')
        self.assertFalse(result.success)
        self.assertIn('Aucun appareil abonné', result.message)

    def test_send_counts_devices(self):
        PushSubscription.objects.create(
            user=self.user, endpoint='https://push.example/t', p256dh='k', auth='a'
        )
        from services import push
        with patch.object(push, 'webpush'):
            result = self.tool.execute(self.user, body='Ton créneau démarre.')
        self.assertTrue(result.success)
        self.assertEqual(result.data['sent'], 1)

    def test_missing_body_fails(self):
        result = self.tool.execute(self.user, body='  ')
        self.assertFalse(result.success)
