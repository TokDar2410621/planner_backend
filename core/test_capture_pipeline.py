"""
Capture pipeline deltas: per-block confidence, pending status + hidden from the
planning, and the confirm/reject endpoints.
"""
import tempfile

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from core.models import RecurringBlock
from services.document_processor import (
    DocumentProcessor,
    compute_block_confidence,
    PENDING_CONFIDENCE_THRESHOLD,
)


class ConfidenceScoreTest(TestCase):
    def test_complete_block_is_max_confidence(self):
        self.assertEqual(
            compute_block_confidence(
                start_defaulted=False, end_defaulted=False, title_generic=False
            ),
            1.0,
        )

    def test_defaulted_times_and_generic_title_is_low(self):
        conf = compute_block_confidence(
            start_defaulted=True, end_defaulted=True, title_generic=True
        )
        self.assertLess(conf, PENDING_CONFIDENCE_THRESHOLD)

    def test_missing_end_only_stays_active(self):
        conf = compute_block_confidence(
            start_defaulted=False, end_defaulted=True, title_generic=False
        )
        self.assertGreaterEqual(conf, PENDING_CONFIDENCE_THRESHOLD)

    def test_llm_confidence_is_blended(self):
        # Full completeness (1.0) blended with an LLM self-rating of 0.5 -> 0.75.
        conf = compute_block_confidence(
            start_defaulted=False, end_defaulted=False, title_generic=False,
            llm_confidence=0.5,
        )
        self.assertAlmostEqual(conf, 0.75, places=3)

    def test_invalid_llm_confidence_ignored(self):
        conf = compute_block_confidence(
            start_defaulted=False, end_defaulted=False, title_generic=False,
            llm_confidence="not-a-number",
        )
        self.assertEqual(conf, 1.0)


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class CreateBlocksConfidenceTest(TestCase):
    def setUp(self):
        from core.models import UploadedDocument
        self.user = User.objects.create_user('capuser', password='pw-cap-123456')
        self.doc = UploadedDocument.objects.create(
            user=self.user,
            file=SimpleUploadedFile('h.png', b'x', content_type='image/png'),
            document_type='course_schedule',
        )

    def test_complete_course_active_incomplete_pending(self):
        data = {
            'detected_type': 'course_schedule',
            'courses': [
                {  # complete -> active
                    'name': 'Mathématiques', 'day': 'lundi',
                    'start_time': '09:00', 'end_time': '11:00', 'location': 'A-101',
                },
                {  # no times + no name -> pending
                    'name': '', 'day': 'mardi',
                    'start_time': '', 'end_time': '',
                },
            ],
            'shifts': [],
            'events': [],
        }
        blocks = DocumentProcessor()._create_recurring_blocks(self.doc, data)
        self.assertEqual(len(blocks), 2)

        # Default manager hides the pending one.
        visible = RecurringBlock.objects.filter(user=self.user)
        self.assertEqual(visible.count(), 1)
        self.assertEqual(visible.first().title, 'Mathématiques')
        self.assertEqual(visible.first().status, RecurringBlock.STATUS_ACTIVE)

        # all_objects sees both; the incomplete one is pending with low confidence.
        allb = RecurringBlock.all_objects.filter(user=self.user)
        self.assertEqual(allb.count(), 2)
        pending = allb.get(status=RecurringBlock.STATUS_PENDING)
        self.assertLess(pending.confidence, PENDING_CONFIDENCE_THRESHOLD)


class PendingEndpointsTest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user('penduser', password='pw-pend-12345')
        self.client.force_authenticate(self.user)
        self.active = RecurringBlock.objects.create(
            user=self.user, title='Cours actif', block_type='course',
            day_of_week=0, start_time='09:00', end_time='10:00',
            status=RecurringBlock.STATUS_ACTIVE, confidence=1.0,
        )
        self.pending = RecurringBlock.objects.create(
            user=self.user, title='Cours douteux', block_type='course',
            day_of_week=1, start_time='09:00', end_time='10:00',
            status=RecurringBlock.STATUS_PENDING, confidence=0.3,
        )

    @staticmethod
    def _titles(data):
        rows = data['results'] if isinstance(data, dict) and 'results' in data else data
        return [b['title'] for b in rows]

    def test_list_excludes_pending(self):
        r = self.client.get(reverse('recurring-block-list'))
        titles = self._titles(r.data)
        self.assertIn('Cours actif', titles)
        self.assertNotIn('Cours douteux', titles)

    def test_pending_endpoint_lists_pending(self):
        r = self.client.get(reverse('recurring-block-pending'))
        self.assertEqual(len(r.data), 1)
        self.assertEqual(r.data[0]['title'], 'Cours douteux')

    def test_confirm_makes_visible(self):
        r = self.client.post(reverse('recurring-block-confirm', kwargs={'pk': self.pending.pk}))
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.pending.refresh_from_db()
        self.assertEqual(self.pending.status, RecurringBlock.STATUS_ACTIVE)
        # Now appears in the normal list.
        titles = self._titles(self.client.get(reverse('recurring-block-list')).data)
        self.assertIn('Cours douteux', titles)

    def test_reject_deletes(self):
        r = self.client.post(reverse('recurring-block-reject', kwargs={'pk': self.pending.pk}))
        self.assertEqual(r.data['deleted'], 1)
        self.assertFalse(RecurringBlock.all_objects.filter(pk=self.pending.pk).exists())

    def test_confirm_all(self):
        r = self.client.post(reverse('recurring-block-confirm-all'))
        self.assertEqual(r.data['confirmed'], 1)
        self.assertEqual(
            RecurringBlock.all_objects.filter(
                user=self.user, status=RecurringBlock.STATUS_PENDING
            ).count(),
            0,
        )

    def test_cannot_confirm_other_users_block(self):
        other = User.objects.create_user('other-pend', password='pw-oth-123456')
        theirs = RecurringBlock.objects.create(
            user=other, title='X', block_type='course', day_of_week=2,
            start_time='09:00', end_time='10:00', status=RecurringBlock.STATUS_PENDING,
        )
        r = self.client.post(reverse('recurring-block-confirm', kwargs={'pk': theirs.pk}))
        self.assertEqual(r.status_code, status.HTTP_404_NOT_FOUND)


class PendingHiddenFromFeedsTest(TestCase):
    def test_pending_excluded_from_ical(self):
        from services.ical import build_calendar
        user = User.objects.create_user('icalpend', password='pw-icp-123456')
        RecurringBlock.objects.create(
            user=user, title='Bloc actif', block_type='course', day_of_week=0,
            start_time='09:00', end_time='10:00', status=RecurringBlock.STATUS_ACTIVE,
        )
        RecurringBlock.objects.create(
            user=user, title='Bloc pending', block_type='course', day_of_week=1,
            start_time='09:00', end_time='10:00', status=RecurringBlock.STATUS_PENDING,
        )
        ics = build_calendar(user)
        self.assertIn('Bloc actif', ics)
        self.assertNotIn('Bloc pending', ics)
