"""
Flux OAuth MCP « comme Gridar » : approve (utilisateur connecté) frappe un code
éphémère; lookup/redeem (secret partagé serveur MCP) valident PKCE et rendent
le token DRF. Un code = un usage, 5 minutes.
"""
from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient

SECRET = 'test-mcp-oauth-secret'


@override_settings(MCP_OAUTH_SECRET=SECRET)
class McpOAuthFlowTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('oauth_user', password='pw-oauth-1')
        self.client = APIClient()
        self.client.force_authenticate(self.user)
        self.mcp = APIClient()  # non authentifié: parle avec le secret

    def _approve(self, **overrides):
        payload = {
            'client_id': 'client-abc',
            'redirect_uri': 'https://claude.ai/api/mcp/auth_callback',
            'code_challenge': 'challenge-xyz',
            'state': 'state-123',
            'scopes': ['planner'],
        }
        payload.update(overrides)
        return self.client.post('/api/auth/mcp-oauth/approve/', payload, format='json')

    def test_approve_returns_redirect_with_code_and_state(self):
        r = self._approve()
        self.assertEqual(r.status_code, 200)
        self.assertIn('code=', r.data['redirect'])
        self.assertIn('state=state-123', r.data['redirect'])
        self.assertTrue(r.data['redirect'].startswith('https://claude.ai/'))

    def test_approve_rejects_shady_redirect(self):
        r = self._approve(redirect_uri='ftp://evil.example')
        self.assertEqual(r.status_code, 400)

    def test_approve_requires_auth(self):
        anon = APIClient()
        r = anon.post('/api/auth/mcp-oauth/approve/', {}, format='json')
        self.assertIn(r.status_code, (401, 403))

    def _extract_code(self, redirect):
        from urllib.parse import parse_qs, urlparse
        return parse_qs(urlparse(redirect).query)['code'][0]

    def test_lookup_then_redeem_returns_drf_token_once(self):
        code = self._extract_code(self._approve().data['redirect'])

        look = self.mcp.post('/api/auth/mcp-oauth/lookup/', {'code': code},
                             format='json', HTTP_X_MCP_SECRET=SECRET)
        self.assertEqual(look.status_code, 200)
        self.assertEqual(look.data['code_challenge'], 'challenge-xyz')
        self.assertEqual(look.data['client_id'], 'client-abc')

        redeem = self.mcp.post('/api/auth/mcp-oauth/redeem/', {'code': code},
                               format='json', HTTP_X_MCP_SECRET=SECRET)
        self.assertEqual(redeem.status_code, 200)
        self.assertEqual(redeem.data['token'], Token.objects.get(user=self.user).key)

        # un code = UN usage
        again = self.mcp.post('/api/auth/mcp-oauth/redeem/', {'code': code},
                              format='json', HTTP_X_MCP_SECRET=SECRET)
        self.assertEqual(again.status_code, 404)

    def test_secret_required(self):
        code = self._extract_code(self._approve().data['redirect'])
        r = self.mcp.post('/api/auth/mcp-oauth/redeem/', {'code': code},
                          format='json', HTTP_X_MCP_SECRET='wrong')
        self.assertEqual(r.status_code, 403)

    def test_expired_code_rejected(self):
        from django.utils import timezone
        from datetime import timedelta
        from core.models import McpOAuthCode
        code = self._extract_code(self._approve().data['redirect'])
        McpOAuthCode.objects.filter(code=code).update(
            expires_at=timezone.now() - timedelta(minutes=1))
        r = self.mcp.post('/api/auth/mcp-oauth/redeem/', {'code': code},
                          format='json', HTTP_X_MCP_SECRET=SECRET)
        self.assertEqual(r.status_code, 404)
