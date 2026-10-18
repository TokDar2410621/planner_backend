from unittest.mock import patch

from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from core.models import AppareilPush, PushSubscription


User = get_user_model()


class PushDeviceStatusTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='push-status', password='x')
        self.other = User.objects.create_user(username='push-other', password='x')
        self.url = '/api/push/device-status/'

    def test_auth_and_input(self):
        self.assertEqual(self.client.post(self.url, {'platform': 'ios'}, format='json').status_code, 401)
        self.client.force_authenticate(self.user)
        for data in ({}, {'platform': 'ios'}, {'platform': 'web'}):
            self.assertEqual(self.client.post(self.url, data, format='json').status_code, 400)

    def test_ios_checks_this_user_permission_and_apns_configuration(self):
        token = 'a' * 64
        self.client.force_authenticate(self.user)
        with patch('services.apns.apns_configured', return_value=True):
            response = self.client.post(self.url, {'platform': 'ios', 'token': token}, format='json')
            self.assertEqual(response.data, {'active': False, 'reason': 'not_registered'})
            device = AppareilPush.objects.create(user=self.other, token=token, notifications_autorisees=True)
            response = self.client.post(self.url, {'platform': 'ios', 'token': token}, format='json')
            self.assertEqual(response.data['reason'], 'not_registered')
            device.user = self.user
            device.notifications_autorisees = False
            device.save()
            response = self.client.post(self.url, {'platform': 'ios', 'token': token}, format='json')
            self.assertEqual(response.data['reason'], 'permission_missing')
            device.notifications_autorisees = True
            device.save()
            response = self.client.post(self.url, {'platform': 'ios', 'token': token}, format='json')
            self.assertEqual(response.data, {'active': True, 'reason': 'ready'})
        with patch('services.apns.apns_configured', return_value=False):
            response = self.client.post(self.url, {'platform': 'ios', 'token': token}, format='json')
            self.assertEqual(response.data, {'active': False, 'reason': 'server_unconfigured'})

    def test_web_checks_this_browser_and_vapid_configuration(self):
        endpoint = 'https://push.example.com/subscription/one'
        self.client.force_authenticate(self.user)
        with patch('services.push.push_configured', return_value=True):
            response = self.client.post(self.url, {'platform': 'web', 'endpoint': endpoint}, format='json')
            self.assertEqual(response.data['reason'], 'not_registered')
            sub = PushSubscription.objects.create(user=self.other, endpoint=endpoint, p256dh='p', auth='a')
            response = self.client.post(self.url, {'platform': 'web', 'endpoint': endpoint}, format='json')
            self.assertEqual(response.data['reason'], 'not_registered')
            sub.user = self.user
            sub.save()
            response = self.client.post(self.url, {'platform': 'web', 'endpoint': endpoint}, format='json')
            self.assertEqual(response.data, {'active': True, 'reason': 'ready'})
        with patch('services.push.push_configured', return_value=False):
            response = self.client.post(self.url, {'platform': 'web', 'endpoint': endpoint}, format='json')
            self.assertEqual(response.data['reason'], 'server_unconfigured')

    def test_registration_reports_whether_notifications_can_be_delivered(self):
        self.client.force_authenticate(self.user)
        with patch('services.apns.apns_configured', return_value=True):
            response = self.client.post('/api/push/appareil/', {
                'token': 'b' * 64, 'notifications_autorisees': True,
            }, format='json')
            self.assertTrue(response.data['active'])
        with patch('services.push.push_configured', return_value=False):
            response = self.client.post('/api/push/subscribe/', {
                'endpoint': 'https://push.example.com/two',
                'keys': {'p256dh': 'p', 'auth': 'a'},
            }, format='json')
            self.assertFalse(response.data['active'])
