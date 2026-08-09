"""Documents are stored on Cloudinary as raw with a neutral extension because a
.pdf public_id is blocked from delivery (HTTP 401). Consequences under test:

- the stored name never carries a .pdf (or the real) extension;
- the processor picks the extraction path from the file's magic bytes, not the
  suffix (the suffix is now meaningless);
- the human-facing original filename is preserved in `file_name`.
"""
import tempfile
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from rest_framework.test import APITestCase

from core.models import UploadedDocument, document_upload_to
from core.validators import sniff_kind
from services.document_processor import DocumentProcessor

# Smallest byte prefixes carrying the right magic numbers.
_PDF = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF"
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
_JPG = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 16
_WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 8
_GIF = b"GIF89a" + b"\x00" * 16


class SniffKindTests(TestCase):
    def test_pdf_magic(self):
        self.assertEqual(sniff_kind(_PDF[:16]), "pdf")

    def test_png_magic(self):
        self.assertEqual(sniff_kind(_PNG[:16]), "image")

    def test_jpeg_magic(self):
        self.assertEqual(sniff_kind(_JPG[:16]), "image")

    def test_webp_magic(self):
        self.assertEqual(sniff_kind(_WEBP[:16]), "image")

    def test_gif_magic(self):
        self.assertEqual(sniff_kind(_GIF[:16]), "image")

    def test_unknown_defaults_to_pdf_pipeline(self):
        # unknown content still routes to the pdf-smart path, which itself falls
        # back to vision, so nothing is silently dropped.
        self.assertEqual(sniff_kind(b"garbage bytes!!!"), "pdf")


class DocumentUploadPathTests(TestCase):
    def test_upload_path_strips_pdf_extension(self):
        path = document_upload_to(None, "Mon Horaire H2026.pdf")
        self.assertTrue(path.startswith("documents/"))
        self.assertFalse(path.lower().endswith(".pdf"))
        self.assertNotIn(".", path.split("/")[-1])  # neutral, extensionless

    def test_upload_paths_are_unique(self):
        a = document_upload_to(None, "x.pdf")
        b = document_upload_to(None, "x.pdf")
        self.assertNotEqual(a, b)


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class ProcessorRoutingByContentTests(TestCase):
    """The extraction path must follow the CONTENT, never the (neutral) suffix."""

    def setUp(self):
        self.user = User.objects.create_user("rawdoc", password="pw-123456")

    def _make_doc(self, raw_bytes):
        doc = UploadedDocument.objects.create(
            user=self.user,
            file=SimpleUploadedFile("whatever.pdf", raw_bytes),
            document_type="course_schedule",
            file_name="whatever.pdf",
        )
        # The stored name is neutral regardless of the uploaded filename.
        self.assertNotIn(".", doc.file.name.split("/")[-1])
        return doc

    @patch.object(DocumentProcessor, "_process_image", return_value=('{"courses": []}', "vision"))
    @patch.object(DocumentProcessor, "_process_pdf_smart", return_value=('{"courses": []}', "text"))
    def test_pdf_content_routes_to_pdf_smart(self, m_pdf, m_img):
        doc = self._make_doc(_PDF)
        DocumentProcessor().process_document(doc)
        m_pdf.assert_called_once()
        m_img.assert_not_called()

    @patch.object(DocumentProcessor, "_process_image", return_value=('{"courses": []}', "vision"))
    @patch.object(DocumentProcessor, "_process_pdf_smart", return_value=('{"courses": []}', "text"))
    def test_image_content_routes_to_image(self, m_pdf, m_img):
        doc = self._make_doc(_PNG)
        DocumentProcessor().process_document(doc)
        m_img.assert_called_once()
        m_pdf.assert_not_called()

    @patch.object(DocumentProcessor, "_process_image", return_value=('{"courses": []}', "vision"))
    @patch.object(DocumentProcessor, "_process_pdf_smart", return_value=('{"courses": []}', "text"))
    def test_original_file_name_is_preserved(self, m_pdf, m_img):
        doc = self._make_doc(_PDF)
        DocumentProcessor().process_document(doc)
        doc.refresh_from_db()
        # Not clobbered by the opaque stored name.
        self.assertEqual(doc.file_name, "whatever.pdf")


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class ChatUploadPreservesNameTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user("chatdoc", password="pw-123456")
        self.client.force_authenticate(self.user)

    @patch("core.views.DocumentProcessor")
    def test_chat_upload_keeps_original_filename(self, mock_proc):
        r = self.client.post(
            reverse("chat"),
            {"message": "", "attachment": SimpleUploadedFile("horaire_cours.pdf", _PDF)},
            format="multipart",
        )
        self.assertEqual(r.status_code, 200, getattr(r, "data", r))
        doc = UploadedDocument.objects.filter(user=self.user).latest("id")
        self.assertEqual(doc.file_name, "horaire_cours.pdf")
        self.assertNotIn(".", doc.file.name.split("/")[-1])
