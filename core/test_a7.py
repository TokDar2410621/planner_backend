"""
Regression tests for group A7 fixes.

- S4: validate_upload_file (size / extension / magic bytes allowlist)
- S7: RecurringBlockCompletionSerializer scopes recurring_block to the
  requesting user (IDOR)
- UserProfile.public_planning_enabled defaults to False (opt-in for S2)
- Goal.progress is clamped to 0..100 on save
"""
import datetime
import io

import pytest
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.exceptions import ValidationError as DRFValidationError

from core.models import RecurringBlock, Goal, UserProfile
from core.serializers import RecurringBlockCompletionSerializer
from core.validators import validate_upload_file, MAX_UPLOAD_SIZE


PNG_HEADER = b'\x89PNG\r\n\x1a\n' + b'\x00' * 32


# --------------------------------------------------------------------------
# S4 - validate_upload_file
# --------------------------------------------------------------------------

def test_validate_upload_accepts_small_valid_png():
    f = SimpleUploadedFile('image.png', PNG_HEADER, content_type='image/png')
    assert validate_upload_file(f) == 'png'


def test_validate_upload_rejects_oversized_file():
    big = SimpleUploadedFile(
        'image.png', PNG_HEADER + b'\x00' * (MAX_UPLOAD_SIZE + 1),
        content_type='image/png',
    )
    with pytest.raises(DRFValidationError):
        validate_upload_file(big)


def test_validate_upload_rejects_exe_extension():
    f = SimpleUploadedFile('evil.exe', b'MZ\x90\x00' + b'\x00' * 20,
                           content_type='application/octet-stream')
    with pytest.raises(DRFValidationError):
        validate_upload_file(f)


def test_validate_upload_rejects_svg_extension():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
    f = SimpleUploadedFile('evil.svg', svg, content_type='image/svg+xml')
    with pytest.raises(DRFValidationError):
        validate_upload_file(f)


def test_validate_upload_rejects_png_extension_with_wrong_magic():
    # Allowed extension but content is actually HTML/script -> magic mismatch.
    payload = b'<html><script>alert(1)</script></html>'
    f = SimpleUploadedFile('fake.png', payload, content_type='image/png')
    with pytest.raises(DRFValidationError):
        validate_upload_file(f)


def test_validate_upload_does_not_consume_stream():
    f = SimpleUploadedFile('image.png', PNG_HEADER, content_type='image/png')
    validate_upload_file(f)
    # Cursor must be restored so downstream processing can read the file.
    assert f.read(8) == b'\x89PNG\r\n\x1a\n'


# --------------------------------------------------------------------------
# S7 - completion serializer scoped to request user
# --------------------------------------------------------------------------

class _FakeRequest:
    def __init__(self, user):
        self.user = user


@pytest.fixture
def user_a(db):
    return User.objects.create_user(username='alice', password='pw12345678')


@pytest.fixture
def user_b(db):
    return User.objects.create_user(username='bob', password='pw12345678')


def _make_block(user):
    return RecurringBlock.objects.create(
        user=user,
        title='Cours',
        block_type='course',
        day_of_week=0,
        start_time=datetime.time(9, 0),
        end_time=datetime.time(11, 0),
    )


def test_completion_serializer_rejects_other_users_block(user_a, user_b):
    block_a = _make_block(user_a)
    serializer = RecurringBlockCompletionSerializer(
        data={'recurring_block': block_a.id, 'date': '2026-07-20'},
        context={'request': _FakeRequest(user_b)},
    )
    assert not serializer.is_valid()
    assert 'recurring_block' in serializer.errors


def test_completion_serializer_accepts_own_block(user_a):
    block_a = _make_block(user_a)
    serializer = RecurringBlockCompletionSerializer(
        data={'recurring_block': block_a.id, 'date': '2026-07-20'},
        context={'request': _FakeRequest(user_a)},
    )
    assert serializer.is_valid(), serializer.errors


def test_completion_viewset_rejects_other_users_block(user_a, user_b):
    from rest_framework.test import APIClient
    from rest_framework_simplejwt.tokens import RefreshToken

    block_a = _make_block(user_a)
    client = APIClient()
    refresh = RefreshToken.for_user(user_b)
    client.credentials(HTTP_AUTHORIZATION=f'Bearer {refresh.access_token}')

    resp = client.post(
        '/api/recurring-completions/',
        {'recurring_block': block_a.id, 'date': '2026-07-20'},
        format='json',
    )
    assert resp.status_code == 400
    from core.models import RecurringBlockCompletion
    assert not RecurringBlockCompletion.objects.filter(
        recurring_block=block_a
    ).exists()


# --------------------------------------------------------------------------
# UserProfile.public_planning_enabled default
# --------------------------------------------------------------------------

def test_public_planning_enabled_defaults_false(db):
    user = User.objects.create_user(username='carol', password='pw12345678')
    profile = UserProfile.objects.get(user=user)
    assert profile.public_planning_enabled is False


# --------------------------------------------------------------------------
# Goal.progress clamp
# --------------------------------------------------------------------------

def test_goal_progress_clamped_high(db):
    user = User.objects.create_user(username='dave', password='pw12345678')
    goal = Goal.objects.create(user=user, title='Big', progress=250)
    goal.refresh_from_db()
    assert goal.progress == 100
