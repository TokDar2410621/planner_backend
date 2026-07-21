"""
Durable backstop for document processing (T3 / D3).

The inline upload path processes documents in a background daemon thread (see
``DocumentProcessor.process_document_async``). That thread does NOT survive a
redeploy or a worker crash, so any ``UploadedDocument`` whose greenlet was
killed mid-flight stays stuck at ``processed=False`` forever.

This command is the durability backstop. It is meant to run on a schedule
(Railway cron, see ``railway.toml``) and reprocesses every pending document
idempotently. It is safe to run concurrently with the inline path because
``DocumentProcessor`` deduplicates recurring-block creation per document, so a
document processed twice does not create duplicate blocks.

Usage:
    python manage.py process_pending_documents
    python manage.py process_pending_documents --min-age 60 --limit 50
"""
import logging

from django.core.management.base import BaseCommand
from django.db import connections
from django.utils import timezone

from core.models import UploadedDocument

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = (
        "Reprocess UploadedDocument rows still marked processed=False. "
        "Durability backstop that survives redeploys (the inline daemon "
        "thread does not)."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--min-age',
            type=int,
            default=0,
            help=(
                'Only process documents uploaded at least N seconds ago, to '
                'avoid racing the inline processing path (default: 0).'
            ),
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=0,
            help='Maximum number of documents to process this run (0 = no limit).',
        )

    def handle(self, *args, **options):
        min_age = options['min_age']
        limit = options['limit']

        queryset = UploadedDocument.objects.filter(processed=False)

        if min_age and min_age > 0:
            cutoff = timezone.now() - timezone.timedelta(seconds=min_age)
            queryset = queryset.filter(uploaded_at__lte=cutoff)

        # Deterministic order (oldest first) so retries make forward progress.
        queryset = queryset.order_by('uploaded_at')

        # Materialize the id list up front so the connection can be closed and
        # each document reprocessed with a fresh connection/transaction.
        doc_ids = list(queryset.values_list('id', flat=True))
        if limit and limit > 0:
            doc_ids = doc_ids[:limit]

        total = len(doc_ids)
        logger.info(
            "process_pending_documents: found %d pending document(s) "
            "(min_age=%ss, limit=%s)",
            total, min_age, limit or 'none',
        )

        if total == 0:
            self.stdout.write("No pending documents to process.")
            return

        # Import here so the command still loads even if optional heavy deps
        # (PyMuPDF, google-genai, etc.) are unavailable at import time.
        from services.document_processor import DocumentProcessor

        processor = DocumentProcessor()

        succeeded = 0
        failed = 0
        skipped = 0

        for doc_id in doc_ids:
            # Per-document isolation: one bad document must never abort the run.
            try:
                document = UploadedDocument.objects.get(id=doc_id)
            except UploadedDocument.DoesNotExist:
                # Deleted between listing and processing; nothing to do.
                skipped += 1
                logger.info("Document %s no longer exists, skipping.", doc_id)
                continue

            # Re-check under fresh read: the inline path may have finished it
            # between our snapshot and now. Reprocessing is idempotent, but
            # skipping saves an unnecessary LLM call.
            if document.processed:
                skipped += 1
                logger.info("Document %s already processed, skipping.", doc_id)
                continue

            try:
                processor.process_document(document)
                succeeded += 1
                logger.info("Reprocessed document %s successfully.", doc_id)
            except Exception as exc:
                # process_document already logged the cause and persisted a
                # generic processing_error on the row. Swallow here so the
                # batch keeps going.
                failed += 1
                logger.error(
                    "Failed to reprocess document %s: %s", doc_id, exc,
                    exc_info=True,
                )
            finally:
                # Close DB connections between documents so a long batch never
                # holds a stale/broken connection open (mirrors the greenlet
                # path and keeps connections healthy across the run).
                connections.close_all()

        summary = (
            f"process_pending_documents done: {succeeded} processed, "
            f"{failed} failed, {skipped} skipped (of {total} candidates)."
        )
        logger.info(summary)
        self.stdout.write(summary)
