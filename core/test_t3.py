"""
T3 / D3 -- durable document-processing backstop.

Covers the `process_pending_documents` management command:
- it finds pending (processed=False) documents and reprocesses them;
- a single failing document does not abort the whole batch;
- the --min-age flag excludes freshly uploaded documents.

DocumentProcessor.process_document is always mocked so no network / LLM call
is made and no file needs to exist on disk.
"""
from io import StringIO
from unittest.mock import patch

import pytest
from django.core.management import call_command

from core.models import UploadedDocument
from services.document_processor import DocumentProcessor


def _make_pending(user, name):
    """Create a pending (unprocessed) document with no real file."""
    return UploadedDocument.objects.create(
        user=user,
        file_name=name,
        document_type="course_schedule",
        extracted_data={},
        processed=False,
    )


def _mark_processed(document):
    """Simulate a successful DocumentProcessor.process_document call."""
    document.processed = True
    document.processing_error = None
    document.save(update_fields=["processed", "processing_error"])
    return {"courses": []}


@pytest.mark.django_db
def test_command_marks_pending_document_processed(user):
    doc = _make_pending(user, "horaire.pdf")

    with patch.object(
        DocumentProcessor, "process_document", side_effect=_mark_processed
    ) as mocked:
        call_command("process_pending_documents", stdout=StringIO())

    mocked.assert_called_once()
    doc.refresh_from_db()
    assert doc.processed is True
    assert doc.processing_error is None


@pytest.mark.django_db
def test_failing_document_does_not_abort_batch(user):
    bad = _make_pending(user, "corrupt.pdf")
    good = _make_pending(user, "valid.pdf")

    def side_effect(document):
        if document.id == bad.id:
            raise RuntimeError("boom: bad document")
        return _mark_processed(document)

    with patch.object(
        DocumentProcessor, "process_document", side_effect=side_effect
    ) as mocked:
        call_command("process_pending_documents", stdout=StringIO())

    # Both documents were attempted -> the bad one did not abort the run.
    assert mocked.call_count == 2

    good.refresh_from_db()
    bad.refresh_from_db()
    assert good.processed is True
    # The failing document stays pending (no silent success).
    assert bad.processed is False


@pytest.mark.django_db
def test_min_age_excludes_fresh_documents(user):
    # Freshly uploaded (uploaded_at == now); a large --min-age must skip it.
    _make_pending(user, "fresh.pdf")

    with patch.object(
        DocumentProcessor, "process_document", side_effect=_mark_processed
    ) as mocked:
        call_command(
            "process_pending_documents", "--min-age", "3600", stdout=StringIO()
        )

    mocked.assert_not_called()


@pytest.mark.django_db
def test_already_processed_documents_are_ignored(user):
    UploadedDocument.objects.create(
        user=user,
        file_name="done.pdf",
        document_type="course_schedule",
        extracted_data={"courses": []},
        processed=True,
    )

    with patch.object(
        DocumentProcessor, "process_document", side_effect=_mark_processed
    ) as mocked:
        call_command("process_pending_documents", stdout=StringIO())

    mocked.assert_not_called()
