"""Per-user MCP token endpoint (Phase 4)."""
from django.contrib.auth.models import User
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient, APITestCase


class McpTokenTest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user('mcpuser', password='pw-mcp-12345')

    def test_requires_authentication(self):
        self.assertEqual(self.client.get(reverse('mcp-token')).status_code,
                         status.HTTP_401_UNAUTHORIZED)

    def test_get_creates_and_returns_token(self):
        self.client.force_authenticate(self.user)
        r = self.client.get(reverse('mcp-token'))
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertEqual(len(r.data['token']), 40)  # DRF token key length
        # idempotent: same token on a second call
        self.assertEqual(self.client.get(reverse('mcp-token')).data['token'], r.data['token'])

    def test_post_rotates_token(self):
        self.client.force_authenticate(self.user)
        first = self.client.get(reverse('mcp-token')).data['token']
        rotated = self.client.post(reverse('mcp-token')).data['token']
        self.assertNotEqual(first, rotated)

    def test_token_authenticates_the_api(self):
        """The token the MCP server will use must authenticate protected calls."""
        self.client.force_authenticate(self.user)
        token = self.client.get(reverse('mcp-token')).data['token']

        c = APIClient()
        c.credentials(HTTP_AUTHORIZATION=f'Token {token}')
        self.assertEqual(c.get(reverse('task-list')).status_code, status.HTTP_200_OK)
        # and it is scoped to the owner
        self.assertEqual(c.get(reverse('me')).data['username'], 'mcpuser')
