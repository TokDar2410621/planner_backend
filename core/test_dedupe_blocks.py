"""
Dedupe des blocs récurrents en double + garde-fou anti-chevauchement à
l'extraction (reproduit le cas prod de Darius: 'Travail' Ven/Sam créé 2 fois).
"""
import tempfile
from datetime import time

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import TestCase, override_settings
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from core.models import RecurringBlock
from services.blocks_maintenance import dedupe_recurring_blocks


def _block(user, day, s, e, btype='work', title='Travail', night=False):
    return RecurringBlock.objects.create(
        user=user, title=title, block_type=btype, day_of_week=day,
        start_time=s, end_time=e, is_night_shift=night,
    )


class DedupeHelperTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('deduper', password='pw-ded-123456')

    def test_exact_duplicate_night_shifts_removed(self):
        # Cas prod: 'Travail' Ven 19:00-07:00 créé deux fois.
        _block(self.user, 4, time(19, 0), time(7, 0), night=True, title='Travail')
        _block(self.user, 4, time(19, 0), time(7, 0), night=True, title='Travail')
        report = dedupe_recurring_blocks(self.user)
        self.assertEqual(report['removed'], 1)
        self.assertEqual(RecurringBlock.all_objects.filter(user=self.user).count(), 1)

    def test_same_time_type_different_title_is_duplicate(self):
        # 'Travail samedi' + 'Travail' même créneau -> 1 seul gardé (le plus ancien).
        keep = _block(self.user, 5, time(19, 0), time(7, 0), night=True, title='Travail samedi')
        _block(self.user, 5, time(19, 0), time(7, 0), night=True, title='Travail')
        report = dedupe_recurring_blocks(self.user)
        self.assertEqual(report['removed'], 1)
        remaining = RecurringBlock.all_objects.filter(user=self.user)
        self.assertEqual(remaining.count(), 1)
        self.assertEqual(remaining.first().id, keep.id)

    def test_distinct_blocks_untouched(self):
        _block(self.user, 0, time(9, 0), time(10, 0), btype='course', title='Maths')
        _block(self.user, 0, time(10, 0), time(11, 0), btype='course', title='Info')
        report = dedupe_recurring_blocks(self.user)
        self.assertEqual(report['removed'], 0)
        self.assertEqual(RecurringBlock.all_objects.filter(user=self.user).count(), 2)


class DedupeEndpointAndCommandTest(APITestCase):
    def test_endpoint_dedupes_current_user(self):
        user = User.objects.create_user('dedapi', password='pw-dda-123456')
        self.client.force_authenticate(user)
        _block(user, 4, time(19, 0), time(7, 0), night=True)
        _block(user, 4, time(19, 0), time(7, 0), night=True)
        r = self.client.post(reverse('recurring-block-deduplicate'))
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertEqual(r.data['removed'], 1)

    def test_command_targets_one_user(self):
        u1 = User.objects.create_user('cmd1', password='pw-cd1-123456')
        u2 = User.objects.create_user('cmd2', password='pw-cd2-123456')
        _block(u1, 4, time(19, 0), time(7, 0), night=True)
        _block(u1, 4, time(19, 0), time(7, 0), night=True)
        _block(u2, 4, time(19, 0), time(7, 0), night=True)
        _block(u2, 4, time(19, 0), time(7, 0), night=True)
        call_command('dedupe_blocks', '--user', str(u1.id))
        self.assertEqual(RecurringBlock.all_objects.filter(user=u1).count(), 1)
        self.assertEqual(RecurringBlock.all_objects.filter(user=u2).count(), 2)  # intact


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class ExtractionOverlapGuardTest(TestCase):
    def setUp(self):
        from core.models import UploadedDocument
        self.user = User.objects.create_user('extguard', password='pw-ext-123456')
        self.doc = UploadedDocument.objects.create(
            user=self.user,
            file=SimpleUploadedFile('h.png', b'x', content_type='image/png'),
            document_type='course_schedule',
        )

    def test_extraction_skips_overlapping_course(self):
        from services.document_processor import DocumentProcessor
        # Deux fois le même cours + un cours qui chevauche -> un seul gardé par créneau.
        data = {
            'detected_type': 'course_schedule',
            'courses': [
                {'name': 'Maths', 'day': 'lundi', 'start_time': '09:00', 'end_time': '11:00'},
                {'name': 'Maths (dup)', 'day': 'lundi', 'start_time': '09:00', 'end_time': '11:00'},
                {'name': 'Chevauche', 'day': 'lundi', 'start_time': '10:00', 'end_time': '12:00'},
            ],
            'shifts': [], 'events': [],
        }
        DocumentProcessor()._create_recurring_blocks(self.doc, data)
        # Seul le premier créneau lundi 09:00-11:00 existe (les 2 suivants chevauchent).
        blocks = RecurringBlock.all_objects.filter(user=self.user, day_of_week=0)
        self.assertEqual(blocks.count(), 1)
        self.assertEqual(blocks.first().title, 'Maths')
