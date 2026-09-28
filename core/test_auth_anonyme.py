"""Comptes anonymes : creation, reprise, conversion, quotas, suppression.

Modele Firebase Anonymous Auth : signInAnonymously (POST /auth/anonymous/)
puis linkWithCredential (conversion via Google/Apple avec le JWT anonyme).
Aucun appel reseau reel : tokeninfo Google et verification Apple moques.
"""
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.test import override_settings
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase
from rest_framework_simplejwt.tokens import RefreshToken

from core.anonyme import device_id_valide, est_anonyme
from core.models import BudgetJetonsJournalier
from core.views import AnonymousChatThrottle
from services.agent_v2 import agent as module_agent
from services.agent_v2.agent import PlannerAgentV2


def _google_response(email, aud='test-client-id', verified='true', **extra):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {'email': email, 'aud': aud,
                              'email_verified': verified, **extra}
    return resp


def _claims(email='u@example.com', verified='true', **extra):
    return {'email': email, 'email_verified': verified,
            'sub': 'apple-sub-1', **extra}


def _creer_anonyme(device_id='device-test-01'):
    user = User.objects.create_user(username=f'anon_{device_id[:8]}', email='')
    profil = user.profile
    profil.is_anonymous = True
    profil.device_id = device_id
    profil.save(update_fields=['is_anonymous', 'device_id'])
    return user


def _bearer(user):
    return f"Bearer {RefreshToken.for_user(user).access_token}"


class HelpersAnonymeTest(APITestCase):
    def test_device_id_valide(self):
        self.assertTrue(device_id_valide('abc-123_XYZ'))
        self.assertTrue(device_id_valide('a' * 8))
        self.assertTrue(device_id_valide('a' * 64))
        self.assertFalse(device_id_valide(None))
        self.assertFalse(device_id_valide('court'))
        self.assertFalse(device_id_valide('a' * 65))
        self.assertFalse(device_id_valide('avec espace'))
        self.assertFalse(device_id_valide('avec.point'))

    def test_est_anonyme_defensif(self):
        self.assertFalse(est_anonyme(None))
        user = User.objects.create_user('normal', password='pw-1234567')
        self.assertFalse(est_anonyme(user))
        anon = _creer_anonyme('device-def-01')
        self.assertTrue(est_anonyme(anon))
        # Profil manquant (donnees anciennes) : jamais anonyme par defaut.
        user.profile.delete()
        self.assertFalse(est_anonyme(user))


class CreationAnonymeTest(APITestCase):
    url = reverse('anonymous-auth')

    def test_creation(self):
        r = self.client.post(self.url, {'device_id': 'device-creation-01'},
                             format='json')
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertTrue(r.data['created'])
        self.assertFalse(r.data['converted'])
        self.assertIn('refresh', r.data['tokens'])
        self.assertIn('access', r.data['tokens'])
        user = User.objects.get(username=r.data['user']['username'])
        self.assertTrue(user.username.startswith('anon_'))
        self.assertEqual(user.email, '')
        self.assertFalse(user.has_usable_password())
        self.assertTrue(user.profile.is_anonymous)
        self.assertEqual(user.profile.device_id, 'device-creation-01')
        # Expose en top-level pour que le client sache proposer le compte.
        self.assertTrue(r.data['user']['is_anonymous'])

    def test_reprise_meme_device_id(self):
        r1 = self.client.post(self.url, {'device_id': 'device-reprise-02'},
                              format='json')
        r2 = self.client.post(self.url, {'device_id': 'device-reprise-02'},
                              format='json')
        self.assertTrue(r1.data['created'])
        self.assertFalse(r2.data['created'])
        self.assertEqual(r1.data['user']['id'], r2.data['user']['id'])
        self.assertEqual(User.objects.filter(
            profile__device_id='device-reprise-02').count(), 1)

    def test_device_id_invalide_400(self):
        for mauvais in (None, '', 'court', 'a' * 65, 'avec espace',
                        'avec.point', 123):
            r = self.client.post(self.url, {'device_id': mauvais},
                                 format='json')
            self.assertEqual(r.status_code, status.HTTP_400_BAD_REQUEST,
                             f"device_id={mauvais!r}")

    def test_throttle_creation_10_par_heure(self):
        # Scope auth_anon : 11e creation dans l'heure -> 429.
        with patch('core.views.ScopedRateThrottle.get_rate',
                    return_value='10/hour'):
            codes = set()
            for i in range(11):
                r = self.client.post(
                    self.url, {'device_id': f'device-throttle-{i:02d}'},
                    format='json')
                codes.add(r.status_code)
        self.assertIn(status.HTTP_429_TOO_MANY_REQUESTS, codes)


@override_settings(GOOGLE_CLIENT_ID='test-client-id',
                   GOOGLE_ALLOWED_CLIENT_IDS=['test-client-id'])
class ConversionGoogleTest(APITestCase):
    def test_conversion_google(self):
        anon = _creer_anonyme('device-conv-ggl')
        self.client.credentials(HTTP_AUTHORIZATION=_bearer(anon))
        with patch('requests.get', return_value=_google_response(
                'nouveau@example.com', given_name='Jean',
                picture='http://img/x.png')):
            r = self.client.post(reverse('google-auth'),
                                 {'credential': 'tok'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertTrue(r.data['converted'])
        self.assertFalse(r.data['created'])
        anon.refresh_from_db()
        self.assertEqual(anon.email, 'nouveau@example.com')
        self.assertEqual(anon.first_name, 'Jean')
        self.assertFalse(anon.profile.is_anonymous)
        self.assertIsNone(anon.profile.device_id)
        self.assertEqual(anon.profile.avatar_url, 'http://img/x.png')
        self.assertFalse(r.data['user']['is_anonymous'])

    def test_conversion_google_409_email_deja_utilise(self):
        User.objects.create_user('reel', email='pris@example.com',
                                 password='pw-1234567')
        anon = _creer_anonyme('device-conv-409')
        self.client.credentials(HTTP_AUTHORIZATION=_bearer(anon))
        with patch('requests.get', return_value=_google_response(
                'pris@example.com')):
            r = self.client.post(reverse('google-auth'),
                                 {'credential': 'tok'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_409_CONFLICT)
        self.assertEqual(r.data['code'], 'email_deja_utilise')
        # L'anonyme est intact : pas d'email vole, pas de conversion.
        anon.refresh_from_db()
        self.assertEqual(anon.email, '')
        self.assertTrue(anon.profile.is_anonymous)

    def test_sans_jwt_anonyme_chemin_normal(self):
        # Sans JWT : pas de conversion, comportement historique inchange.
        with patch('requests.get', return_value=_google_response(
                'frais@example.com')):
            r = self.client.post(reverse('google-auth'),
                                 {'credential': 'tok'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertFalse(r.data['converted'])
        self.assertTrue(r.data['created'])


@override_settings(APPLE_CLIENT_ID='com.planner.web')
class ConversionAppleTest(APITestCase):
    def test_conversion_apple(self):
        anon = _creer_anonyme('device-conv-apl')
        self.client.credentials(HTTP_AUTHORIZATION=_bearer(anon))
        with patch('services.apple_auth.verify_apple_identity_token',
                    return_value=_claims('apple@example.com')):
            r = self.client.post(reverse('apple-auth'),
                                 {'id_token': 'x',
                                  'name': {'firstName': 'Marie'}},
                                 format='json')
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertTrue(r.data['converted'])
        anon.refresh_from_db()
        self.assertEqual(anon.email, 'apple@example.com')
        self.assertEqual(anon.first_name, 'Marie')
        self.assertFalse(anon.profile.is_anonymous)
        self.assertIsNone(anon.profile.device_id)

    def test_conversion_apple_409(self):
        User.objects.create_user('reel2', email='pris2@example.com',
                                 password='pw-1234567')
        anon = _creer_anonyme('device-conv-apl2')
        self.client.credentials(HTTP_AUTHORIZATION=_bearer(anon))
        with patch('services.apple_auth.verify_apple_identity_token',
                    return_value=_claims('pris2@example.com')):
            r = self.client.post(reverse('apple-auth'), {'id_token': 'x'},
                                 format='json')
        self.assertEqual(r.status_code, status.HTTP_409_CONFLICT)
        self.assertEqual(r.data['code'], 'email_deja_utilise')


class ThrottleChatAnonymeTest(APITestCase):
    url = reverse('chat')

    def _poster(self, user):
        self.client.force_authenticate(user=user)
        # Sans consentement IA la vue repondrait 403 : le throttle (429)
        # passe AVANT le handler, c'est lui qu'on mesure.
        return self.client.post(self.url, {}, format='json')

    def test_quota_anonyme_independant_et_resserre(self):
        anon = _creer_anonyme('device-throttle-chat')
        normal = User.objects.create_user('normal-chat', password='pw-1234567')
        with patch.object(AnonymousChatThrottle, 'get_rate',
                          return_value='2/min'):
            # L'anonyme epuise SON quota en 2 requetes...
            self.assertNotEqual(self._poster(anon).status_code, 429)
            self.assertNotEqual(self._poster(anon).status_code, 429)
            self.assertEqual(self._poster(anon).status_code, 429)
            # ...sans toucher a celui du compte inscrit (scope `chat`).
            self.assertNotEqual(self._poster(normal).status_code, 429)
            self.assertNotEqual(self._poster(normal).status_code, 429)
            self.assertEqual(self._poster(normal).status_code, 429)

    def test_scope_choisi_selon_le_compte(self):
        anon = _creer_anonyme('device-throttle-scope')
        normal = User.objects.create_user('normal-scope', password='pw-1234567')
        throttle = AnonymousChatThrottle()
        requete = MagicMock()
        requete.user = anon
        throttle.allow_request(requete, MagicMock())
        self.assertEqual(throttle.scope, 'chat_anon')
        requete.user = normal
        throttle.allow_request(requete, MagicMock())
        self.assertEqual(throttle.scope, 'chat')


class BudgetAnonymeTest(APITestCase):
    def test_plafond_anonyme_400k(self):
        anon = _creer_anonyme('device-budget-01')
        normal = User.objects.create_user('normal-budget', password='pw-1234567')
        self.assertEqual(module_agent._budget_jetons_jour(anon), 400000)
        self.assertEqual(module_agent._budget_jetons_jour(normal), 2000000)
        self.assertEqual(module_agent._budget_jetons_jour(), 2000000)
        self.assertEqual(module_agent._budget_jetons_jour(None), 2000000)

    def test_epuise_selon_le_bon_plafond(self):
        anon = _creer_anonyme('device-budget-02')
        normal = User.objects.create_user('normal-budget2', password='pw-1234567')
        BudgetJetonsJournalier.objects.create(
            user=anon, jour=timezone.localdate(), jetons=400000)
        BudgetJetonsJournalier.objects.create(
            user=normal, jour=timezone.localdate(), jetons=400000)
        self.assertTrue(module_agent._budget_jour_epuise(anon))
        self.assertFalse(module_agent._budget_jour_epuise(normal))

    def test_plafond_anonyme_zero_desactive(self):
        anon = _creer_anonyme('device-budget-03')
        BudgetJetonsJournalier.objects.create(
            user=anon, jour=timezone.localdate(), jetons=999999999)
        with override_settings(AGENT_V2_BUDGET_JETONS_JOUR_ANON=0):
            self.assertFalse(module_agent._budget_jour_epuise(anon))

    def test_message_suggere_le_compte(self):
        """Budget anonyme epuise : le tour propose de creer un compte."""
        anon = _creer_anonyme('device-budget-04')
        BudgetJetonsJournalier.objects.create(
            user=anon, jour=timezone.localdate(), jetons=400000)
        with patch.object(PlannerAgentV2, '_agir',
                          side_effect=AssertionError("AGIR ne doit pas tourner")), \
             patch.object(PlannerAgentV2, '_dire',
                          side_effect=AssertionError("DIRE ne doit pas tourner")):
            res = PlannerAgentV2().process_message(anon, "bonjour")
        self.assertIn("Crée un compte", res['response'])


class SuppressionAnonymeTest(APITestCase):
    def test_suppression_compte_anonyme(self):
        anon = _creer_anonyme('device-delete-01')
        anon_id = anon.pk
        self.client.force_authenticate(user=anon)
        r = self.client.post(reverse('delete-account'),
                             {'confirmation': 'SUPPRIMER'}, format='json')
        self.assertEqual(r.status_code, status.HTTP_204_NO_CONTENT)
        self.assertFalse(User.objects.filter(pk=anon_id).exists())
