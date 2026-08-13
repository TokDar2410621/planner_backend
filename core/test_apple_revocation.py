"""
Révocation Sign in with Apple à la suppression de compte.

Exigence Apple depuis juin 2022 (prolongement de 5.1.1(v)): une app qui offre
SIWA et la suppression de compte doit révoquer les jetons Apple au moment de
la suppression. Chaîne testée: le login stocke le refresh token (échange du
code d'autorisation), la suppression le révoque, et AUCUN échec Apple ne
bloque jamais ni la connexion ni la suppression.
"""
import time
from unittest.mock import MagicMock, patch

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken

WEB_AUD = 'com.tokamdarius.planner.web'
NATIVE_AUD = 'com.tokamdarius.planner'


def _ec_private_pem():
    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


TEST_KEY = _ec_private_pem()

REVOCATION_SETTINGS = dict(
    APPLE_SIGNIN_PRIVATE_KEY=TEST_KEY,
    APPLE_SIGNIN_KEY_ID='CLETEST123',
    APPLE_TEAM_ID='EQUIPE1234',
)


@override_settings(**REVOCATION_SETTINGS)
class ClientSecretTests(TestCase):
    def test_le_client_secret_est_un_jwt_es256_aux_bons_claims(self):
        from services.apple_revocation import make_client_secret

        secret = make_client_secret(NATIVE_AUD)
        headers = jwt.get_unverified_header(secret)
        self.assertEqual(headers['alg'], 'ES256')
        self.assertEqual(headers['kid'], 'CLETEST123')

        claims = jwt.decode(secret, options={'verify_signature': False}, audience='https://appleid.apple.com')
        self.assertEqual(claims['iss'], 'EQUIPE1234')
        self.assertEqual(claims['sub'], NATIVE_AUD)
        self.assertLessEqual(claims['exp'] - claims['iat'], 6 * 30 * 24 * 3600)


@override_settings(**REVOCATION_SETTINGS, APPLE_CLIENT_ID=f'{WEB_AUD},{NATIVE_AUD}')
class AppleLoginStoresTokenTests(TestCase):
    def _login(self, aud=NATIVE_AUD, code='code-abc', exchange_response=None):
        claims = {
            'aud': aud,
            'sub': 'apple-user-1',
            'email': 'personne@example.com',
            'email_verified': True,
        }
        exchange = exchange_response if exchange_response is not None else MagicMock(
            status_code=200, json=lambda: {'refresh_token': 'jeton-refresh'}
        )
        with patch('services.apple_auth.verify_apple_identity_token', return_value=claims), \
             patch('services.apple_revocation.requests.post', return_value=exchange):
            payload = {'id_token': 'jeton', 'authorization_code': code}
            return APIClient().post('/api/auth/apple/', payload, format='json')

    def test_le_refresh_token_est_stocke_avec_l_audience_du_flux(self):
        r = self._login(aud=NATIVE_AUD)
        self.assertEqual(r.status_code, 200, r.content)
        profile = User.objects.get(email='personne@example.com').profile
        self.assertEqual(profile.apple_refresh_token, 'jeton-refresh')
        self.assertEqual(profile.apple_client_id, NATIVE_AUD)

    def test_un_echange_refuse_ne_bloque_pas_la_connexion(self):
        refus = MagicMock(status_code=400, text='invalid_grant', json=lambda: {})
        r = self._login(exchange_response=refus)
        self.assertEqual(r.status_code, 200, r.content)
        profile = User.objects.get(email='personne@example.com').profile
        self.assertEqual(profile.apple_refresh_token, '')

    def test_sans_code_d_autorisation_rien_n_est_stocke_et_le_login_passe(self):
        claims = {'aud': WEB_AUD, 'sub': 'x', 'email': 'personne@example.com', 'email_verified': True}
        with patch('services.apple_auth.verify_apple_identity_token', return_value=claims):
            r = APIClient().post('/api/auth/apple/', {'id_token': 'jeton'}, format='json')
        self.assertEqual(r.status_code, 200, r.content)

    def test_le_refresh_token_ne_sort_jamais_par_l_api_profil(self):
        self._login()
        user = User.objects.get(email='personne@example.com')
        api = APIClient()
        api.credentials(HTTP_AUTHORIZATION=f'Bearer {RefreshToken.for_user(user).access_token}')
        body = api.get('/api/profile/').json()
        self.assertNotIn('apple_refresh_token', body)
        self.assertNotIn('apple_client_id', body)


@override_settings(**REVOCATION_SETTINGS)
class DeleteRevokesTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='partant', password='x')
        profile = self.user.profile
        profile.apple_refresh_token = 'jeton-refresh'
        profile.apple_client_id = NATIVE_AUD
        profile.save()
        self.client_ = APIClient()
        self.client_.credentials(HTTP_AUTHORIZATION=f'Bearer {RefreshToken.for_user(self.user).access_token}')

    def _delete(self):
        return self.client_.post('/api/auth/delete-account/', {'confirmation': 'SUPPRIMER'}, format='json')

    def test_la_suppression_revoque_chez_apple(self):
        ok = MagicMock(status_code=200)
        with patch('services.apple_revocation.requests.post', return_value=ok) as poste:
            r = self._delete()
        self.assertEqual(r.status_code, 204)
        self.assertEqual(poste.call_count, 1)
        appel = poste.call_args
        self.assertIn('revoke', appel.args[0])
        self.assertEqual(appel.kwargs['data']['token'], 'jeton-refresh')
        self.assertEqual(appel.kwargs['data']['client_id'], NATIVE_AUD)
        self.assertFalse(User.objects.filter(username='partant').exists())

    def test_apple_en_panne_n_empeche_pas_la_suppression(self):
        with patch('services.apple_revocation.requests.post', side_effect=Exception('reseau mort')):
            r = self._delete()
        self.assertEqual(r.status_code, 204)
        self.assertFalse(User.objects.filter(username='partant').exists())

    def test_sans_jeton_apple_aucun_appel_sortant(self):
        profile = self.user.profile
        profile.apple_refresh_token = ''
        profile.save()
        with patch('services.apple_revocation.requests.post') as poste:
            r = self._delete()
        self.assertEqual(r.status_code, 204)
        poste.assert_not_called()
