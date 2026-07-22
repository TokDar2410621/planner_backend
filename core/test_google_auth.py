"""
Google login + email uniqueness (fix du 409 'plusieurs comptes partagent cet
email' constaté en prod).
"""
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.test import override_settings
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from core.models import RecurringBlock


def _google_response(email, aud, verified='true', **extra):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {'email': email, 'aud': aud, 'email_verified': verified, **extra}
    return resp


@override_settings(GOOGLE_CLIENT_ID='test-client-id')
class GoogleAuthDuplicateTest(APITestCase):
    def test_duplicate_email_signs_into_data_account_not_409(self):
        # Reproduit la prod: 2 comptes sur le même email, l'un a les données.
        User.objects.create_user('empty', email='dup@example.com', password='pw-1234567')
        data_user = User.objects.create_user('mdarius', email='dup@example.com', password='pw-1234567')
        RecurringBlock.objects.create(
            user=data_user, title='Cours', block_type='course', day_of_week=0,
            start_time='09:00', end_time='10:00',
        )
        with patch('requests.get', return_value=_google_response('dup@example.com', 'test-client-id')):
            r = self.client.post(reverse('google-auth'), {'credential': 'tok'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        # Connecté au compte qui a les données, pas de 409.
        self.assertEqual(r.data['user']['username'], 'mdarius')
        self.assertFalse(r.data['created'])

    def test_unverified_email_still_rejected(self):
        with patch('requests.get', return_value=_google_response('x@example.com', 'test-client-id', verified='false')):
            r = self.client.post(reverse('google-auth'), {'credential': 'tok'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_wrong_aud_rejected(self):
        with patch('requests.get', return_value=_google_response('x@example.com', 'someone-else')):
            r = self.client.post(reverse('google-auth'), {'credential': 'tok'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_new_google_user_is_created(self):
        with patch('requests.get', return_value=_google_response('new@example.com', 'test-client-id', given_name='New')):
            r = self.client.post(reverse('google-auth'), {'credential': 'tok'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertTrue(r.data['created'])
        self.assertTrue(User.objects.filter(email='new@example.com').exists())


class RegistrationEmailUniquenessTest(APITestCase):
    def test_register_rejects_existing_email(self):
        User.objects.create_user('first', email='taken@example.com', password='pw-1234567')
        r = self.client.post(reverse('register'), {
            'username': 'second', 'email': 'taken@example.com',
            'password': 'Str0ng-pw-xyz-9', 'password_confirm': 'Str0ng-pw-xyz-9',
        }, format='json')
        self.assertEqual(r.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('email', r.data)

    def test_register_rejects_case_insensitive_duplicate(self):
        User.objects.create_user('first', email='Taken@Example.com', password='pw-1234567')
        r = self.client.post(reverse('register'), {
            'username': 'second', 'email': 'taken@example.com',
            'password': 'Str0ng-pw-xyz-9', 'password_confirm': 'Str0ng-pw-xyz-9',
        }, format='json')
        self.assertEqual(r.status_code, status.HTTP_400_BAD_REQUEST)

    def test_register_fresh_email_ok(self):
        r = self.client.post(reverse('register'), {
            'username': 'fresh', 'email': 'fresh@example.com',
            'password': 'Str0ng-pw-xyz-9', 'password_confirm': 'Str0ng-pw-xyz-9',
        }, format='json')
        self.assertEqual(r.status_code, status.HTTP_201_CREATED)
