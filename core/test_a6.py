"""
Regression tests for group A6 (core/views.py) security/behaviour fixes.

Covers:
  S2  - PublicPlanningByUsernameView gated behind UserProfile.public_planning_enabled
  S6  - GoogleAuthView rejects unverified email + handles duplicate emails
  S8  - CheckEmailView throttled via a scoped throttle
  B24 - LoginView enforces is_active via django authenticate()
"""
from datetime import time
from unittest import mock

import pytest
from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import override_settings
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient

from core.models import RecurringBlock


# --------------------------------------------------------------------------- #
# S2 - PublicPlanningByUsernameView opt-in gate
# --------------------------------------------------------------------------- #

@pytest.mark.django_db
def test_planning_not_opted_in_returns_no_blocks(user):
    """A user who did NOT opt in must not expose any planning (404, no data)."""
    RecurringBlock.objects.create(
        user=user, title='Cours secret', block_type='course',
        day_of_week=0, start_time=time(9, 0), end_time=time(11, 0),
        location='Salle B12',
    )
    # default public_planning_enabled is False
    assert user.profile.public_planning_enabled is False

    client = APIClient()
    url = reverse('public-planning', kwargs={'username': user.username})
    resp = client.get(url)

    assert resp.status_code == status.HTTP_404_NOT_FOUND
    # No schedule data leaks whatsoever.
    assert 'recurring_blocks' not in resp.data


@pytest.mark.django_db
def test_planning_not_opted_in_indistinguishable_from_missing(user):
    """Opted-out user and non-existent user return the same 404 (no enumeration)."""
    client = APIClient()
    resp_optout = client.get(reverse('public-planning', kwargs={'username': user.username}))
    resp_missing = client.get(reverse('public-planning', kwargs={'username': 'ghost_nobody'}))

    assert resp_optout.status_code == status.HTTP_404_NOT_FOUND
    assert resp_missing.status_code == status.HTTP_404_NOT_FOUND
    assert resp_optout.data == resp_missing.data


@pytest.mark.django_db
def test_planning_opted_in_exposes_blocks_without_location(user):
    """When opted in, blocks are returned but exact location/room is stripped."""
    profile = user.profile
    profile.public_planning_enabled = True
    profile.save()

    RecurringBlock.objects.create(
        user=user, title='Cours public', block_type='course',
        day_of_week=0, start_time=time(9, 0), end_time=time(11, 0),
        location='Salle B12',
    )

    client = APIClient()
    url = reverse('public-planning', kwargs={'username': user.username})
    resp = client.get(url)

    assert resp.status_code == status.HTTP_200_OK
    assert len(resp.data['recurring_blocks']) == 1
    block = resp.data['recurring_blocks'][0]
    assert block['title'] == 'Cours public'
    # Location must NOT be present in the public payload.
    assert 'location' not in block


# --------------------------------------------------------------------------- #
# B24 - LoginView enforces is_active
# --------------------------------------------------------------------------- #

@pytest.mark.django_db
def test_deactivated_user_cannot_login():
    """A deactivated account with valid credentials must not receive tokens."""
    user = User.objects.create_user(
        username='deactivated', email='dead@example.com', password='goodpass123'
    )
    user.is_active = False
    user.save()

    client = APIClient()
    url = reverse('login')
    resp = client.post(url, {'username': 'deactivated', 'password': 'goodpass123'})

    assert resp.status_code == status.HTTP_401_UNAUTHORIZED
    assert 'tokens' not in resp.data


@pytest.mark.django_db
def test_active_user_can_login():
    """An active account with valid credentials still logs in (no regression)."""
    User.objects.create_user(
        username='alive', email='alive@example.com', password='goodpass123'
    )
    client = APIClient()
    resp = client.post(reverse('login'), {'username': 'alive', 'password': 'goodpass123'})

    assert resp.status_code == status.HTTP_200_OK
    assert 'tokens' in resp.data
    assert resp.data['tokens']['access']


@pytest.mark.django_db
def test_wrong_password_rejected():
    User.objects.create_user(username='alive2', password='goodpass123')
    client = APIClient()
    resp = client.post(reverse('login'), {'username': 'alive2', 'password': 'wrong'})
    assert resp.status_code == status.HTTP_401_UNAUTHORIZED


# --------------------------------------------------------------------------- #
# S6 - GoogleAuthView rejects unverified email
# --------------------------------------------------------------------------- #

def _fake_google_response(payload, status_code=200):
    fake = mock.Mock()
    fake.status_code = status_code
    fake.json.return_value = payload
    return fake


@pytest.mark.django_db
@override_settings(GOOGLE_CLIENT_ID='test-client-id')
def test_google_auth_rejects_unverified_email():
    """A Google token with email_verified=false must be rejected (no account link)."""
    payload = {
        'aud': 'test-client-id',
        'email': 'victim@example.com',
        'email_verified': 'false',
        'given_name': 'Vic',
        'family_name': 'Tim',
    }
    client = APIClient()
    url = reverse('google-auth')
    with mock.patch('requests.get', return_value=_fake_google_response(payload)):
        resp = client.post(url, {'credential': 'fake-token'})

    assert resp.status_code == status.HTTP_401_UNAUTHORIZED
    assert 'tokens' not in resp.data
    # No user was created/linked from an unverified token.
    assert not User.objects.filter(email='victim@example.com').exists()


@pytest.mark.django_db
@override_settings(GOOGLE_CLIENT_ID='test-client-id')
def test_google_auth_accepts_verified_email():
    """A verified Google token creates/logs in the user (no regression)."""
    payload = {
        'aud': 'test-client-id',
        'email': 'newuser@example.com',
        'email_verified': 'true',
        'given_name': 'New',
        'family_name': 'User',
    }
    client = APIClient()
    url = reverse('google-auth')
    with mock.patch('requests.get', return_value=_fake_google_response(payload)):
        resp = client.post(url, {'credential': 'fake-token'})

    assert resp.status_code == status.HTTP_200_OK
    assert 'tokens' in resp.data
    assert User.objects.filter(email='newuser@example.com').exists()


@pytest.mark.django_db
@override_settings(GOOGLE_CLIENT_ID='test-client-id')
def test_google_auth_duplicate_email_no_500():
    """Two accounts sharing a (Google-verified) email must not crash, and must
    not hard-block: sign into one of them deterministically (a verified email
    has a single owner, so the accounts are the same person's duplicates)."""
    User.objects.create_user(username='dup1', email='dup@example.com', password='x1234567')
    User.objects.create_user(username='dup2', email='dup@example.com', password='x1234567')

    payload = {
        'aud': 'test-client-id',
        'email': 'dup@example.com',
        'email_verified': 'true',
    }
    client = APIClient()
    url = reverse('google-auth')
    with mock.patch('requests.get', return_value=_fake_google_response(payload)):
        resp = client.post(url, {'credential': 'fake-token'})

    assert resp.status_code == status.HTTP_200_OK
    assert resp.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR
    assert resp.data['user']['username'] in {'dup1', 'dup2'}


# --------------------------------------------------------------------------- #
# S8 - CheckEmailView scoped throttle
# --------------------------------------------------------------------------- #

@pytest.mark.django_db
def test_check_email_is_throttled():
    """CheckEmailView must eventually return 429 under rapid repeated calls."""
    cache.clear()
    client = APIClient()
    url = reverse('check-email')

    statuses = []
    # Fallback rate is 10/min -> the 11th request within the window must 429.
    for _ in range(12):
        resp = client.post(url, {'email': 'probe@example.com'})
        statuses.append(resp.status_code)

    assert status.HTTP_429_TOO_MANY_REQUESTS in statuses
    # The first request should have succeeded before the limit was hit.
    assert statuses[0] == status.HTTP_200_OK
    cache.clear()
