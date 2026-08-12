"""
Suppression de compte depuis l'app — exigence App Store 5.1.1(v).

Une app qui permet de creer un compte doit permettre de le SUPPRIMER, dans
l'app. Le reviewer Apple le teste systematiquement; l'absence de ce chemin est
un rejet mecanique. Ces tests fixent le contrat: tout part (cascade), personne
d'autre n'est touche, et le garde-fou type-to-confirm tient.
"""
from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken

from core.models import Goal, RecurringBlock, Task, UserProfile


def _client(user):
    api = APIClient()
    api.credentials(HTTP_AUTHORIZATION=f'Bearer {RefreshToken.for_user(user).access_token}')
    return api


class DeleteAccountTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='partant', password='x')
        self.other = User.objects.create_user(username='restant', password='x')
        self.client_ = _client(self.user)

    def _peupler(self, user):
        UserProfile.objects.get_or_create(user=user)
        Task.objects.create(user=user, title='Tache')
        RecurringBlock.objects.create(
            user=user, title='Bloc', block_type='work',
            day_of_week=0, start_time='09:00', end_time='10:00',
        )
        Goal.objects.create(user=user, title='Objectif', goal_type='short_term')
        Token.objects.get_or_create(user=user)  # jeton MCP

    def test_le_compte_et_toutes_ses_donnees_partent(self):
        self._peupler(self.user)

        r = self.client_.post('/api/auth/delete-account/', {'confirmation': 'SUPPRIMER'}, format='json')

        self.assertEqual(r.status_code, 204, r.content)
        self.assertFalse(User.objects.filter(username='partant').exists())
        self.assertEqual(Task.objects.filter(user__username='partant').count(), 0)
        self.assertEqual(Goal.objects.count(), 0)
        self.assertEqual(RecurringBlock.objects.count(), 0)
        self.assertEqual(Token.objects.count(), 0)

    def test_les_autres_comptes_ne_bougent_pas(self):
        self._peupler(self.user)
        self._peupler(self.other)

        self.client_.post('/api/auth/delete-account/', {'confirmation': 'SUPPRIMER'}, format='json')

        self.assertTrue(User.objects.filter(username='restant').exists())
        self.assertEqual(Task.objects.filter(user=self.other).count(), 1)
        self.assertEqual(Goal.objects.filter(user=self.other).count(), 1)

    def test_sans_confirmation_rien_ne_part(self):
        for payload in ({}, {'confirmation': 'supprimer'}, {'confirmation': 'OUI'}):
            r = self.client_.post('/api/auth/delete-account/', payload, format='json')
            self.assertEqual(r.status_code, 400, payload)
        self.assertTrue(User.objects.filter(username='partant').exists())

    def test_anonyme_refuse(self):
        r = APIClient().post('/api/auth/delete-account/', {'confirmation': 'SUPPRIMER'}, format='json')
        self.assertEqual(r.status_code, 401)

    def test_le_jeton_devient_inutilisable_apres_suppression(self):
        """Le JWT encore en main du client ne rouvre pas un compte mort."""
        self.client_.post('/api/auth/delete-account/', {'confirmation': 'SUPPRIMER'}, format='json')
        r = self.client_.get('/api/auth/me/')
        self.assertEqual(r.status_code, 401)
