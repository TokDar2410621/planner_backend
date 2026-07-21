"""iCal feed: generator correctness + public token endpoint."""
from datetime import time, timedelta

from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase

from core.models import CalendarFeed, RecurringBlock, Task
from services.ical import build_calendar


class ICalGeneratorTest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user('icaluser', password='pw-ical-12345')

    def test_recurring_block_emits_weekly_vevent(self):
        RecurringBlock.objects.create(
            user=self.user, title='Mathématiques', block_type='course',
            day_of_week=0, start_time=time(9, 0), end_time=time(11, 0),
            location='A-101',
        )
        ics = build_calendar(self.user)
        self.assertIn('BEGIN:VCALENDAR', ics)
        self.assertIn('BEGIN:VEVENT', ics)
        self.assertIn('RRULE:FREQ=WEEKLY;BYDAY=MO', ics)
        self.assertIn('SUMMARY:Mathématiques', ics)
        self.assertIn('LOCATION:A-101', ics)
        # Floating local time: no Z suffix, no TZID on the recurring DTSTART.
        self.assertRegex(ics, r'DTSTART:\d{8}T090000\b')
        self.assertNotIn('DTSTART;TZID', ics)
        # CRLF line endings (RFC 5545).
        self.assertIn('\r\n', ics)

    def test_special_chars_are_escaped(self):
        RecurringBlock.objects.create(
            user=self.user, title='Cours; test, virgule', block_type='course',
            day_of_week=1, start_time=time(14, 0), end_time=time(16, 0),
        )
        ics = build_calendar(self.user)
        self.assertIn(r'SUMMARY:Cours\; test\, virgule', ics)

    def test_overnight_block_rolls_dtend_to_next_day(self):
        RecurringBlock.objects.create(
            user=self.user, title='Nuit', block_type='work',
            day_of_week=2, start_time=time(22, 0), end_time=time(6, 0),
            is_night_shift=True,
        )
        ics = build_calendar(self.user)
        # Find the DTSTART and DTEND dates; DTEND must be exactly one day later.
        import re
        start = re.search(r'DTSTART:(\d{8})T220000', ics).group(1)
        end = re.search(r'DTEND:(\d{8})T060000', ics).group(1)
        from datetime import datetime
        d0 = datetime.strptime(start, '%Y%m%d').date()
        d1 = datetime.strptime(end, '%Y%m%d').date()
        self.assertEqual((d1 - d0).days, 1)

    def test_task_deadline_included_in_utc_when_requested(self):
        deadline = timezone.now() + timedelta(days=3)
        Task.objects.create(
            user=self.user, title='Rendre TP', deadline=deadline,
        )
        without = build_calendar(self.user, include_tasks=False)
        self.assertNotIn('task-deadline-', without)
        with_tasks = build_calendar(self.user, include_tasks=True)
        self.assertIn('task-deadline-', with_tasks)
        # Deadlines are precise instants -> UTC Z.
        self.assertRegex(with_tasks, r'DTSTART:\d{8}T\d{6}Z')


class CalendarFeedEndpointTest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user('feeduser', password='pw-feed-12345')

    def test_feed_get_creates_and_returns_urls(self):
        self.client.force_authenticate(self.user)
        r = self.client.get(reverse('calendar-feed'))
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertIn('.ics', r.data['url'])
        self.assertTrue(r.data['webcal_url'].startswith('webcal://'))
        self.assertEqual(CalendarFeed.objects.filter(user=self.user).count(), 1)

    def test_regenerate_rotates_token(self):
        self.client.force_authenticate(self.user)
        first = self.client.get(reverse('calendar-feed')).data['token']
        second = self.client.post(reverse('calendar-feed')).data['token']
        self.assertNotEqual(first, second)
        # Still only one feed row (rotated in place).
        self.assertEqual(CalendarFeed.objects.filter(user=self.user).count(), 1)

    def test_public_ics_endpoint_serves_calendar(self):
        RecurringBlock.objects.create(
            user=self.user, title='Anglais', block_type='course',
            day_of_week=3, start_time=time(10, 0), end_time=time(12, 0),
        )
        feed = CalendarFeed.objects.create(user=self.user)
        url = reverse('calendar-ics', kwargs={'token': feed.token})
        r = self.client.get(url)  # no auth
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertIn('text/calendar', r['Content-Type'])
        self.assertIn('BEGIN:VCALENDAR', r.content.decode('utf-8'))
        feed.refresh_from_db()
        self.assertEqual(feed.access_count, 1)

    def test_inactive_feed_returns_410(self):
        feed = CalendarFeed.objects.create(user=self.user, is_active=False)
        url = reverse('calendar-ics', kwargs={'token': feed.token})
        r = self.client.get(url)
        self.assertEqual(r.status_code, status.HTTP_410_GONE)

    def test_unknown_token_returns_404(self):
        import uuid
        url = reverse('calendar-ics', kwargs={'token': uuid.uuid4()})
        r = self.client.get(url)
        self.assertEqual(r.status_code, status.HTTP_404_NOT_FOUND)
