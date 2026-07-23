"""Sign in with Apple: account logic (token verification is mocked)."""
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import override_settings
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from core.models import RecurringBlock
from services.social_login import resolve_social_user


def _claims(email='u@example.com', verified='true', **extra):
    return {'email': email, 'email_verified': verified, 'sub': 'apple-sub-1', **extra}


@override_settings(APPLE_CLIENT_ID='com.planner.web')
class AppleAuthViewTest(APITestCase):
    def test_new_user_created(self):
        with patch('services.apple_auth.verify_apple_identity_token', return_value=_claims('new@example.com')):
            r = self.client.post(reverse('apple-auth'),
                                 {'id_token': 'x', 'name': {'firstName': 'New', 'lastName': 'User'}},
                                 format='json')
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertTrue(r.data['created'])
        u = User.objects.get(email='new@example.com')
        self.assertEqual(u.first_name, 'New')

    def test_existing_user_login(self):
        User.objects.create_user('existing', email='dup@example.com', password='pw-1234567')
        with patch('services.apple_auth.verify_apple_identity_token', return_value=_claims('dup@example.com')):
            r = self.client.post(reverse('apple-auth'), {'id_token': 'x'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertFalse(r.data['created'])

    def test_unverified_email_rejected(self):
        with patch('services.apple_auth.verify_apple_identity_token', return_value=_claims(verified='false')):
            r = self.client.post(reverse('apple-auth'), {'id_token': 'x'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_missing_email_rejected(self):
        with patch('services.apple_auth.verify_apple_identity_token', return_value={'sub': 's', 'email_verified': 'true'}):
            r = self.client.post(reverse('apple-auth'), {'id_token': 'x'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_400_BAD_REQUEST)

    def test_invalid_token_rejected(self):
        with patch('services.apple_auth.verify_apple_identity_token', side_effect=ValueError('bad')):
            r = self.client.post(reverse('apple-auth'), {'id_token': 'x'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_missing_id_token(self):
        r = self.client.post(reverse('apple-auth'), {}, format='json')
        self.assertEqual(r.status_code, status.HTTP_400_BAD_REQUEST)


@override_settings(APPLE_CLIENT_ID='')
class AppleNotConfiguredTest(APITestCase):
    def test_returns_503_when_unconfigured(self):
        r = self.client.post(reverse('apple-auth'), {'id_token': 'x'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)


class ResolveSocialUserTest(APITestCase):
    def test_picks_account_with_most_data(self):
        empty = User.objects.create_user('empty', email='shared@example.com', password='pw-1234567')
        data = User.objects.create_user('data', email='shared@example.com', password='pw-1234567')
        RecurringBlock.objects.create(
            user=data, title='C', block_type='course', day_of_week=0,
            start_time='09:00', end_time='10:00',
        )
        user, created = resolve_social_user('shared@example.com')
        self.assertFalse(created)
        self.assertEqual(user.id, data.id)

    def test_creates_unique_username(self):
        User.objects.create_user('taken', email='other@x.com', password='pw-1234567')
        # New email whose local part collides with an existing username.
        User.objects.create_user('collide', email='z@x.com', password='pw-1234567')
        User.objects.filter(username='collide').update(username='collide')
        user, created = resolve_social_user('collide@example.com')
        self.assertTrue(created)
        self.assertNotEqual(user.username, 'collide')  # de-duplicated
