"""Tests for location-based reminders Phase 0/1.

- geocoding service (Nominatim, mocked network)
- auto-geocode on place create via the API
- Task.place FK + nested place_detail + IDOR guard
- send_reminders departure alerts for scheduled tasks with a place
"""
from datetime import datetime, time
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from rest_framework.test import APIClient

from core.models import (
    UserProfile, UserPlace, Task, ScheduledBlock, PushSubscription,
)
from services.geocoding import geocode_address


class _FakeResp:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class GeocodingServiceTests(TestCase):
    @patch("services.geocoding.urllib.request.urlopen")
    def test_geocode_ok(self, mock_open):
        mock_open.return_value = _FakeResp('[{"lat":"48.4212","lon":"-71.0534"}]')
        self.assertEqual(geocode_address("UQAC, Chicoutimi"), (48.4212, -71.0534))

    @patch("services.geocoding.urllib.request.urlopen")
    def test_geocode_no_result(self, mock_open):
        mock_open.return_value = _FakeResp("[]")
        self.assertIsNone(geocode_address("nowhere-xyzzy"))

    @patch("services.geocoding.urllib.request.urlopen", side_effect=OSError("boom"))
    def test_geocode_network_error_is_none(self, _mock):
        self.assertIsNone(geocode_address("UQAC"))

    def test_empty_address_no_call(self):
        # No network mock needed: must short-circuit before any request.
        self.assertIsNone(geocode_address("   "))


class PlaceGeocodeApiTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="geo", password="x")
        self.client = APIClient()
        self.client.force_authenticate(self.user)

    @patch("services.geocoding.geocode_address", return_value=(48.4259, -71.0625))
    def test_create_place_autogeocodes(self, _mock):
        r = self.client.post(
            reverse("place-list"),
            {"name": "UQAC", "kind": "school", "address": "555 Bd de l'Université, Chicoutimi", "travel_minutes": 12},
            format="json",
        )
        self.assertIn(r.status_code, (200, 201), r.content)
        self.assertEqual(r.data["latitude"], 48.4259)
        self.assertEqual(r.data["longitude"], -71.0625)
        place = UserPlace.objects.get(id=r.data["id"])
        self.assertTrue(place.has_coordinates)

    @patch("services.geocoding.geocode_address", return_value=None)
    def test_create_place_without_geocode_still_saves(self, _mock):
        r = self.client.post(
            reverse("place-list"),
            {"name": "Flou", "address": "adresse introuvable", "travel_minutes": 5},
            format="json",
        )
        self.assertIn(r.status_code, (200, 201))
        self.assertIsNone(r.data["latitude"])

    def test_latitude_is_read_only(self):
        # Client-supplied coordinates must be ignored (server geocodes).
        with patch("services.geocoding.geocode_address", return_value=None):
            r = self.client.post(
                reverse("place-list"),
                {"name": "Hack", "latitude": 0.0, "longitude": 0.0, "travel_minutes": 1},
                format="json",
            )
        self.assertIn(r.status_code, (200, 201))
        self.assertIsNone(r.data["latitude"])


class TaskPlaceTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="tsk", password="x")
        self.client = APIClient()
        self.client.force_authenticate(self.user)
        self.place = UserPlace.objects.create(
            user=self.user, name="UQAC", address="Chicoutimi",
            travel_minutes=15, latitude=48.42, longitude=-71.06,
        )

    def test_task_carries_place_and_exposes_detail(self):
        r = self.client.post(
            reverse("task-list"),
            {"title": "Réunion M. YEPRI", "place": self.place.id},
            format="json",
        )
        self.assertIn(r.status_code, (200, 201), r.content)
        self.assertEqual(r.data["place"], self.place.id)
        self.assertEqual(r.data["place_detail"]["latitude"], 48.42)
        self.assertEqual(r.data["place_detail"]["name"], "UQAC")

    def test_cannot_attach_another_users_place(self):
        other = User.objects.create_user(username="other", password="x")
        other_place = UserPlace.objects.create(user=other, name="Ailleurs", travel_minutes=5)
        r = self.client.post(
            reverse("task-list"),
            {"title": "Vol de lieu", "place": other_place.id},
            format="json",
        )
        self.assertEqual(r.status_code, 400)


class SendRemindersTaskTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="rem", password="x")
        UserProfile.objects.update_or_create(
            user=self.user,
            defaults={"prep_time_minutes": 15, "safety_margin_minutes": 10},
        )
        self.place = UserPlace.objects.create(
            user=self.user, name="UQAC", travel_minutes=30,
            latitude=48.42, longitude=-71.06,
        )
        self.task = Task.objects.create(user=self.user, title="Réunion M. YEPRI", place=self.place)
        self.today = timezone.localtime().date()
        ScheduledBlock.objects.create(
            user=self.user, task=self.task, date=self.today,
            start_time=time(13, 0), end_time=time(14, 0),
        )
        PushSubscription.objects.create(
            user=self.user, endpoint="https://push.example/x", p256dh="k", auth="a",
        )

    def _run_at(self, hh, mm, lead=15):
        fixed = timezone.localtime().replace(
            year=self.today.year, month=self.today.month, day=self.today.day,
            hour=hh, minute=mm, second=0, microsecond=0,
        )
        with patch("core.management.commands.send_reminders.timezone.localtime", return_value=fixed), \
             patch("core.management.commands.send_reminders.push_configured", return_value=True), \
             patch("core.management.commands.send_reminders.send_to_user", return_value=1) as mock_send:
            call_command("send_reminders", lead=lead)
        return mock_send

    def test_leave_now_fires_for_located_task(self):
        # start 13:00, travel 30, margin 10 -> departure 12:20. now 12:15, lead 15
        # -> window [12:15, 12:30] contains 12:20.
        mock_send = self._run_at(12, 15)
        calls = [(c.args[1], c.args[2]) for c in mock_send.call_args_list]
        leave = [body for (title, body) in calls if title == "Pars maintenant"]
        self.assertTrue(leave, f"no 'Pars maintenant' sent; calls={calls}")
        self.assertTrue(any("Réunion M. YEPRI" in b for b in leave))

    def test_prepare_fires_earlier(self):
        # unavailability_start = 12:20 - 15 = 12:05. now 12:00, lead 15 -> [12:00,12:15].
        mock_send = self._run_at(12, 0)
        titles = [c.args[1] for c in mock_send.call_args_list]
        self.assertIn("Prépare-toi", titles)

    def test_no_alert_when_no_travel(self):
        self.place.travel_minutes = 0
        self.place.save()
        mock_send = self._run_at(12, 15)
        self.assertEqual(mock_send.call_count, 0)
