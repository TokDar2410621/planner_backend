"""After a schedule import, a vague follow-up ("gère ça", "c'est bon ?") used to
draw a false "je ne peux pas traiter de document" refusal because the agent had
no signal the import had succeeded. PlannerAgent._recent_import_context now reads
the real state and hands the model the list of already-created blocks plus an
explicit ban on denying the import.
"""
from datetime import time, timedelta

from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone

from core.models import RecurringBlock, UploadedDocument
from services.agent.agent import PlannerAgent


def _doc(user, processed=True, name="horaire.pdf"):
    return UploadedDocument.objects.create(
        user=user, file="documents/x", document_type="course_schedule",
        processed=processed, file_name=name,
    )


def _block(user, doc, title, dow, start, end, status=RecurringBlock.STATUS_ACTIVE):
    return RecurringBlock.objects.create(
        user=user, title=title, block_type="course", day_of_week=dow,
        start_time=start, end_time=end, source_document=doc, status=status,
    )


class RecentImportContextTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("importaware", password="pw-123456")
        self.agent = PlannerAgent()

    def test_none_when_no_document(self):
        self.assertIsNone(self.agent._recent_import_context(self.user))

    def test_none_when_document_not_processed(self):
        _doc(self.user, processed=False)
        self.assertIsNone(self.agent._recent_import_context(self.user))

    def test_lists_created_blocks_and_bans_denial(self):
        doc = _doc(self.user, name="mon_horaire.pdf")
        _block(self.user, doc, "Programmation", 0, time(9), time(11))
        _block(self.user, doc, "Anglais", 3, time(15), time(17))

        ctx = self.agent._recent_import_context(self.user)
        self.assertIsNotNone(ctx)
        self.assertIn("mon_horaire.pdf", ctx)
        self.assertIn("Programmation", ctx)
        self.assertIn("lundi 09:00-11:00", ctx)
        self.assertIn("Anglais", ctx)
        self.assertIn("jeudi 15:00-17:00", ctx)
        # The behavioural ban that fixes the S4/S5 false refusal.
        self.assertIn("JAMAIS", ctx)

    def test_ignores_document_older_than_window(self):
        doc = _doc(self.user)
        _block(self.user, doc, "Programmation", 0, time(9), time(11))
        # Age the upload past the 20-min window.
        UploadedDocument.objects.filter(pk=doc.pk).update(
            uploaded_at=timezone.now() - timedelta(minutes=45)
        )
        self.assertIsNone(self.agent._recent_import_context(self.user))

    def test_processed_but_empty_extraction_is_honest(self):
        doc = _doc(self.user)
        doc.processing_error = "extraction vide"
        doc.save(update_fields=["processing_error"])
        ctx = self.agent._recent_import_context(self.user)
        # Either None (nothing to say) or an honest "no data" note — never a
        # confident claim of success.
        if ctx is not None:
            self.assertNotIn("ont été ajoutées", ctx)

    def test_pending_blocks_are_flagged_for_confirmation(self):
        doc = _doc(self.user)
        _block(self.user, doc, "Programmation", 0, time(9), time(11))
        _block(self.user, doc, "Flou", 1, time(9), time(10),
               status=RecurringBlock.STATUS_PENDING)
        ctx = self.agent._recent_import_context(self.user)
        self.assertIn("confiance faible", ctx)
        # The pending one is not counted among the added blocks.
        self.assertIn("1 entrée(s) ont été ajoutées", ctx)
