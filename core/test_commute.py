"""
Commute engine (Phase 1 du moteur de planification).

Scénario de référence de la spec: travail à 18h, trajet 32 min, marge 10 min,
préparation 15 min -> départ limite 17h18, indisponibilité dès 17h03. Aucune
activité flexible ne doit être planifiée après 17h03.
"""
from datetime import date, time, timedelta

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from core.models import RecurringBlock, Task, UserPlace
from services.ai_scheduler import AIScheduler
from services.commute import (
    block_commute_minutes,
    commute_window,
    latest_departure,
    unavailability_start,
)


def _min(h, m=0):
    return h * 60 + m


class CommuteFormulaTest(TestCase):
    """Pure-function checks against the spec's worked example."""

    def test_spec_example_departure_17h18(self):
        # 18:00 - 32 min trajet - 10 min marge = 17:18
        self.assertEqual(latest_departure(_min(18), 32, 10), _min(17, 18))

    def test_spec_example_unavailability_17h03(self):
        # 17:18 - 15 min préparation = 17:03
        self.assertEqual(unavailability_start(_min(17, 18), 15), _min(17, 3))

    def test_commute_window_bundles_both(self):
        w = commute_window(_min(18), 32, 10, 15)
        self.assertEqual(w.departure, _min(17, 18))
        self.assertEqual(w.unavailability_start, _min(17, 3))

    def test_traffic_increase_moves_departure(self):
        # Trajet 32 -> 47 min: départ recule de 15 min (17:03), prépa dès 16:48.
        w = commute_window(_min(18), 47, 10, 15)
        self.assertEqual(w.departure, _min(17, 3))
        self.assertEqual(w.unavailability_start, _min(16, 48))


class BlockCommuteMinutesTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('commuser', password='pw-com-123456')
        self.profile = self.user.profile
        self.profile.prep_time_minutes = 15
        self.profile.safety_margin_minutes = 10
        self.profile.transport_time_minutes = 20  # legacy flat buffer
        self.profile.save()

    def test_block_with_place_uses_spec_formula(self):
        place = UserPlace.objects.create(
            user=self.user, name='Travail', kind='work', travel_minutes=32
        )
        block = RecurringBlock.objects.create(
            user=self.user, title='Job', block_type='work', day_of_week=0,
            start_time=time(18, 0), end_time=time(22, 0), place=place,
        )
        before, after = block_commute_minutes(block, self.profile)
        self.assertEqual(before, 15 + 32 + 10)  # prépa + trajet + marge = 57
        self.assertEqual(after, 32)             # retour

    def test_block_without_place_keeps_flat_transport(self):
        block = RecurringBlock.objects.create(
            user=self.user, title='Cours', block_type='course', day_of_week=1,
            start_time=time(9, 0), end_time=time(11, 0),
        )
        self.assertEqual(block_commute_minutes(block, self.profile), (20, 20))


class SchedulerCommuteIntegrationTest(TestCase):
    """The scheduler must not place anything after début_indisponibilité."""

    def _next_monday(self):
        today = date.today()
        return today + timedelta(days=(7 - today.weekday()) % 7 or 7)

    def setUp(self):
        self.user = User.objects.create_user('schedcom', password='pw-sch-123456')
        p = self.user.profile
        p.prep_time_minutes = 15
        p.safety_margin_minutes = 10
        p.transport_time_minutes = 0
        p.save()
        self.place = UserPlace.objects.create(
            user=self.user, name='Travail', kind='work', travel_minutes=32
        )
        RecurringBlock.objects.create(
            user=self.user, title='Job', block_type='work', day_of_week=0,
            start_time=time(18, 0), end_time=time(22, 0), place=self.place,
        )

    def test_free_slots_end_at_unavailability_start(self):
        monday = self._next_monday()
        slots = AIScheduler()._get_available_slots(self.user, monday, 1)
        self.assertTrue(slots)
        latest_end = max(s.end_time for s in slots)
        # 18:00 - 32 - 10 - 15 = 17:03 -> aucun créneau ne dépasse 17:03.
        self.assertEqual(latest_end, time(17, 3))

    def test_traffic_update_shrinks_the_day(self):
        self.place.travel_minutes = 47
        self.place.save()
        monday = self._next_monday()
        slots = AIScheduler()._get_available_slots(self.user, monday, 1)
        latest_end = max(s.end_time for s in slots)
        # 18:00 - 47 - 10 - 15 = 16:48.
        self.assertEqual(latest_end, time(16, 48))


class UnplacedReportTest(TestCase):
    """Spec §10: un plan impossible est signalé, jamais inventé."""

    def test_oversized_task_lands_in_unplaced(self):
        user = User.objects.create_user('unplaced', password='pw-unp-123456')
        # Bloque toute la semaine: blocs 08:00-22:00 chaque jour.
        for d in range(7):
            RecurringBlock.objects.create(
                user=user, title=f'Occupé {d}', block_type='other', day_of_week=d,
                start_time=time(8, 0), end_time=time(22, 0),
            )
        task = Task.objects.create(
            user=user, title='Marathon de révision', estimated_duration_minutes=120,
        )
        scheduler = AIScheduler()
        created = scheduler.generate_schedule(user, tasks=[task])
        self.assertEqual(created, [])
        self.assertEqual(len(scheduler.last_unplaced), 1)
        entry = scheduler.last_unplaced[0]
        self.assertEqual(entry['task_id'], task.id)
        self.assertEqual(entry['needed_minutes'], 120)
        self.assertIn('reason', entry)

    def test_deep_work_cap_gets_accurate_reason(self):
        """Découvert en prod: une tâche deep-work refusée par le PLAFOND
        quotidien doit dire 'plafond', pas 'créneau trop petit' (spec §10)."""
        user = User.objects.create_user('capreason', password='pw-capr-12345')
        # max_deep_work_hours_per_day = 4 (défaut) -> 240 min/jour.
        task = Task.objects.create(
            user=user, title='Marathon 10h', task_type='deep_work',
            estimated_duration_minutes=600, priority=9,
        )
        scheduler = AIScheduler()
        created = scheduler.generate_schedule(user, tasks=[task])
        self.assertEqual(created, [])
        entry = scheduler.last_unplaced[0]
        self.assertIn('plafond', entry['reason'])
        self.assertIn('240', entry['reason'])


class PlacesAPITest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user('placeapi', password='pw-plc-123456')
        self.client.force_authenticate(self.user)

    def test_create_and_list_place(self):
        r = self.client.post(reverse('place-list'), {
            'name': 'Travail', 'kind': 'work', 'travel_minutes': 32,
        }, format='json')
        self.assertEqual(r.status_code, status.HTTP_201_CREATED)
        lst = self.client.get(reverse('place-list'))
        rows = lst.data['results'] if isinstance(lst.data, dict) and 'results' in lst.data else lst.data
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['travel_minutes'], 32)

    def test_places_are_user_scoped(self):
        other = User.objects.create_user('placeother', password='pw-oth-123456')
        UserPlace.objects.create(user=other, name='Ailleurs', travel_minutes=5)
        lst = self.client.get(reverse('place-list'))
        rows = lst.data['results'] if isinstance(lst.data, dict) and 'results' in lst.data else lst.data
        self.assertEqual(len(rows), 0)

    def test_block_place_cannot_reference_other_users_place(self):
        other = User.objects.create_user('placeidor', password='pw-idr-123456')
        theirs = UserPlace.objects.create(user=other, name='Leur lieu', travel_minutes=10)
        r = self.client.post(reverse('recurring-block-list'), {
            'title': 'Cours', 'block_type': 'course', 'day_of_week': 0,
            'start_time': '09:00', 'end_time': '10:00', 'place': theirs.id,
        }, format='json')
        self.assertEqual(r.status_code, status.HTTP_400_BAD_REQUEST)

    def test_profile_exposes_prep_and_margin(self):
        r = self.client.get(reverse('profile'))
        self.assertIn('prep_time_minutes', r.data)
        self.assertIn('safety_margin_minutes', r.data)
