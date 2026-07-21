"""
Contract parity between the two upload paths.

Codex's prod test found /chat/ accepted document_type="schedule" but /documents/
rejected it with 400 (the model's choices are course_schedule/work_schedule/
other). The serializer now normalizes common aliases before validation so both
upload paths accept the same values.
"""
import tempfile
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from core.serializers import UploadedDocumentSerializer

# 1x1 transparent PNG (valid file bytes; FileField doesn't inspect content).
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000a49444154789c6300010000050001"
    "0d0a2db40000000049454e44ae426082"
)


def _png():
    return SimpleUploadedFile("horaire.png", _PNG, content_type="image/png")


class DocumentTypeAliasSerializerTest(APITestCase):
    def test_schedule_alias_normalizes_to_course_schedule(self):
        s = UploadedDocumentSerializer(
            data={"file": _png(), "document_type": "schedule"}
        )
        self.assertTrue(s.is_valid(), s.errors)
        self.assertEqual(s.validated_data["document_type"], "course_schedule")

    def test_work_alias_normalizes(self):
        s = UploadedDocumentSerializer(data={"file": _png(), "document_type": "work"})
        self.assertTrue(s.is_valid(), s.errors)
        self.assertEqual(s.validated_data["document_type"], "work_schedule")

    def test_canonical_value_still_accepted(self):
        s = UploadedDocumentSerializer(
            data={"file": _png(), "document_type": "course_schedule"}
        )
        self.assertTrue(s.is_valid(), s.errors)
        self.assertEqual(s.validated_data["document_type"], "course_schedule")

    def test_unknown_type_still_rejected(self):
        s = UploadedDocumentSerializer(
            data={"file": _png(), "document_type": "definitely-not-a-type"}
        )
        self.assertFalse(s.is_valid())
        self.assertIn("document_type", s.errors)


class DocumentTypeAliasAPITest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user("docalias", password="pw-doc-123456")
        self.client.force_authenticate(self.user)

    @override_settings(MEDIA_ROOT=tempfile.mkdtemp())
    @patch("core.views.DocumentProcessor")
    def test_documents_upload_accepts_schedule_alias(self, mock_proc):
        r = self.client.post(
            reverse("document-list"),
            {"file": _png(), "document_type": "schedule"},
            format="multipart",
        )
        self.assertEqual(r.status_code, status.HTTP_201_CREATED, r.data)
        self.assertEqual(r.data["document_type"], "course_schedule")
        # Processing is still kicked off (async) exactly as before.
        mock_proc.return_value.process_document_async.assert_called_once()
