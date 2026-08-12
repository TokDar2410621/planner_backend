"""
Audience de l'ID token Google : une LISTE, plus une valeur unique.

Le web envoie aud = client web. L'app iOS native (GoogleSignIn SDK) envoie en
principe aussi le client web via serverClientID, mais peut envoyer le client
iOS selon la configuration. Refuser tout ce qui n'est pas la valeur unique
aurait rendu la connexion Google impossible dans la coque native.
"""
from unittest.mock import patch, MagicMock

from django.test import TestCase, override_settings
from rest_framework.test import APIClient

WEB_ID = 'web-123.apps.googleusercontent.com'
IOS_ID = 'ios-456.apps.googleusercontent.com'


def _google_ok(aud):
    """Réponse tokeninfo simulée: token valide portant l'audience donnée."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        'aud': aud,
        'email': 'personne@example.com',
        'email_verified': 'true',
        'given_name': 'Test',
        'family_name': 'Personne',
    }
    return resp


@override_settings(GOOGLE_ALLOWED_CLIENT_IDS=[WEB_ID, IOS_ID])
class GoogleAudienceTests(TestCase):
    def _post(self):
        return APIClient().post('/api/auth/google/', {'credential': 'jeton'}, format='json')

    def test_audience_web_acceptee(self):
        with patch('requests.get', return_value=_google_ok(WEB_ID)):
            r = self._post()
        self.assertIn(r.status_code, (200, 201), r.content)

    def test_audience_ios_acceptee(self):
        with patch('requests.get', return_value=_google_ok(IOS_ID)):
            r = self._post()
        self.assertIn(r.status_code, (200, 201), r.content)

    def test_audience_inconnue_refusee(self):
        with patch('requests.get', return_value=_google_ok('autre-app.apps.googleusercontent.com')):
            r = self._post()
        self.assertEqual(r.status_code, 401)


class GoogleAudienceSettingsTests(TestCase):
    def test_la_liste_derive_de_l_environnement(self):
        """GOOGLE_CLIENT_ID + GOOGLE_EXTRA_CLIENT_IDS, vides filtrés."""
        import importlib
        import os
        from unittest.mock import patch as env_patch

        with env_patch.dict(os.environ, {
            'GOOGLE_CLIENT_ID': WEB_ID,
            'GOOGLE_EXTRA_CLIENT_IDS': f'{IOS_ID}, ,',
        }):
            import planner.settings as s
            importlib.reload(s)
            self.assertEqual(s.GOOGLE_ALLOWED_CLIENT_IDS, [WEB_ID, IOS_ID])
        # Remet le module dans l'état de l'environnement réel pour les autres tests.
        import planner.settings as s
        importlib.reload(s)
