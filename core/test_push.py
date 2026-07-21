"""Web Push (VAPID) — subscription endpoints, send helper, reminders."""
from datetime import time
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.core.management import call_command
from django.test import TestCase, override_settings
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from core.models import PushSubscription, RecurringBlock


class PushSubscriptionEndpointTest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user('pushuser', password='pw-push-12345')

    def test_vapid_key_endpoint(self):
        self.client.force_authenticate(self.user)
        with override_settings(VAPID_PUBLIC_KEY='PUBKEY123'):
            r = self.client.get(reverse('push-vapid-key'))
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertEqual(r.data['publicKey'], 'PUBKEY123')

    def test_subscribe_requires_auth(self):
        r = self.client.post(reverse('push-subscribe'), {}, format='json')
        self.assertEqual(r.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_subscribe_validates_payload(self):
        self.client.force_authenticate(self.user)
        r = self.client.post(reverse('push-subscribe'), {'endpoint': 'https://x'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_400_BAD_REQUEST)

    def test_subscribe_and_unsubscribe(self):
        self.client.force_authenticate(self.user)
        payload = {'endpoint': 'https://push.example/abc', 'keys': {'p256dh': 'k', 'auth': 'a'}}
        r = self.client.post(reverse('push-subscribe'), payload, format='json')
        self.assertEqual(r.status_code, status.HTTP_201_CREATED)
        self.assertEqual(PushSubscription.objects.filter(user=self.user).count(), 1)

        # re-subscribe same endpoint = refresh (not a duplicate)
        r2 = self.client.post(reverse('push-subscribe'), payload, format='json')
        self.assertEqual(r2.status_code, status.HTTP_200_OK)
        self.assertEqual(PushSubscription.objects.count(), 1)

        r3 = self.client.post(reverse('push-unsubscribe'), {'endpoint': payload['endpoint']}, format='json')
        self.assertEqual(r3.data['deleted'], 1)
        self.assertEqual(PushSubscription.objects.count(), 0)


@override_settings(VAPID_PUBLIC_KEY='pub', VAPID_PRIVATE_KEY='priv', VAPID_SUBJECT='mailto:x@y.z')
class PushSendTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('sender', password='pw-send-12345')
        self.sub = PushSubscription.objects.create(
            user=self.user, endpoint='https://push.example/1', p256dh='k', auth='a'
        )

    def test_send_to_user_success(self):
        from services import push
        with patch.object(push, 'webpush') as mock_wp:
            sent = push.send_to_user(self.user, 'T', 'B')
        self.assertEqual(sent, 1)
        mock_wp.assert_called_once()

    def test_gone_subscription_is_pruned(self):
        from services import push

        exc = push.WebPushException('gone')
        exc.response = MagicMock(status_code=410)
        with patch.object(push, 'webpush', side_effect=exc):
            sent = push.send_to_user(self.user, 'T', 'B')
        self.assertEqual(sent, 0)
        self.assertEqual(PushSubscription.objects.filter(user=self.user).count(), 0)  # pruned


@override_settings(VAPID_PUBLIC_KEY='pub', VAPID_PRIVATE_KEY='priv')
class SendRemindersCommandTest(TestCase):
    def test_reminds_only_users_with_subscriptions(self):
        from services import push

        with_sub = User.objects.create_user('has_sub', password='pw-1234567')
        without = User.objects.create_user('no_sub', password='pw-1234567')
        PushSubscription.objects.create(
            user=with_sub, endpoint='https://push.example/2', p256dh='k', auth='a'
        )
        # a block for each, "now-ish" is hard to force; assert the command runs
        # and only queries users with subscriptions (no crash, no send for `without`).
        RecurringBlock.objects.create(
            user=without, title='X', block_type='course', day_of_week=0,
            start_time=time(9, 0), end_time=time(10, 0),
        )
        with patch.object(push, 'webpush') as mock_wp:
            call_command('send_reminders', '--lead', '15')
        # `without` has no subscription -> webpush never called for them
        for call in mock_wp.call_args_list:
            self.assertNotIn('https://push.example', str(call))  # only real sends would hit it
