"""
Pytest configuration for Planner AI backend.
"""
import pytest
from django.contrib.auth.models import User


@pytest.fixture
def user(db):
    """Create a test user."""
    return User.objects.create_user(
        username='testuser',
        email='test@example.com',
        password='testpass123'
    )


@pytest.fixture
def authenticated_client(client, user):
    """Return an authenticated API client."""
    from rest_framework.test import APIClient
    from rest_framework_simplejwt.tokens import RefreshToken

    api_client = APIClient()
    refresh = RefreshToken.for_user(user)
    api_client.credentials(HTTP_AUTHORIZATION=f'Bearer {refresh.access_token}')
    return api_client
