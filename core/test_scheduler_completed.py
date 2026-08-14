"""
Une tâche terminée ne bloque plus son créneau, et le passé n'est pas libre.

Vécu (2026-08-13, conversation de production): « une tâche d'une minute qui
commence dans 6 minutes » refusée parce que le créneau chevauchait une tâche
App Store COCHÉE LA VEILLE, dont le placement comptait toujours comme un mur;
puis l'agent a proposé « 07:00 à 08:00 » alors qu'il était 19:48. Ces tests
fixent les deux règles: cocher libère la place, et aujourd'hui commence
maintenant.
"""
from datetime import date, time, timedelta
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone

from core.models import ScheduledBlock, Task
from services.scheduling.placement import fixed_busy_intervals

DEMAIN = date.today() + timedelta(days=1)


def _place(user, task, start, end, done=False, jour=DEMAIN):
    return ScheduledBlock.objects.create(
        user=user, task=task, date=jour,
        start_time=start, end_time=end,
        actually_completed=done,
    )


class CompletedTaskFreesSlotTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='u', password='x')

    def _task(self, title='Tache', completed=False):
        return Task.objects.create(user=self.user, title=title, completed=completed)

    def test_un_placement_coche_ne_fait_plus_mur(self):
        _place(self.user, self._task(), time(18, 40), time(20, 40), done=True)
        murs = fixed_busy_intervals(self.user, DEMAIN)
        self.assertEqual(murs, [], 'le creneau d une tache faite doit etre libre')

    def test_une_tache_completee_libere_aussi_ses_placements_non_coches(self):
        # Ceinture: les deux drapeaux sont normalement synchronises par
        # mark_completed, mais un placement futur d une tache deja faite doit
        # aussi liberer sa place.
        _place(self.user, self._task(completed=True), time(9, 0), time(10, 0))
        self.assertEqual(fixed_busy_intervals(self.user, DEMAIN), [])

    def test_un_placement_actif_reste_un_mur(self):
        _place(self.user, self._task(), time(18, 40), time(20, 40))
        murs = fixed_busy_intervals(self.user, DEMAIN)
        self.assertEqual(len(murs), 1)
        self.assertEqual(murs[0], (18 * 60 + 40, 20 * 60 + 40))

    def test_le_conflit_nomme_ignore_les_taches_faites(self):
        from services.agent.tools.schedule import _first_conflicting_item
        _place(self.user, self._task('Faite'), time(18, 0), time(20, 0), done=True)
        _place(self.user, self._task('Active'), time(19, 0), time(21, 0))
        hit = _first_conflicting_item(self.user, DEMAIN, 18 * 60, 22 * 60)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], 'Active', 'jamais nommer une tache terminee comme conflit')


class FreeSlotsClampedToNowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='u2', password='x')

    def _slots_at(self, hour, minute):
        """find_free_slots pour AUJOURD'HUI avec l'heure murale simulée."""
        from services.agent.tools import schedule as mod
        aujourdhui = timezone.localtime()
        simule = aujourdhui.replace(hour=hour, minute=minute, second=0, microsecond=0)
        with patch.object(mod.timezone, 'localtime', return_value=simule):
            tool = mod.FindFreeSlotsTool()
            res = tool.execute(self.user, date=simule.date().isoformat(), min_duration_minutes=30)
        self.assertTrue(res.success, res.message)
        return res.data['free_slots']

    def test_aujourdhui_aucun_creneau_ne_commence_dans_le_passe(self):
        slots = self._slots_at(19, 48)
        for s in slots:
            self.assertGreaterEqual(
                s['start_time'], '19:50',
                f"creneau propose dans le passe: {s['start_time']} alors qu'il est 19:48",
            )

    def test_le_matin_la_journee_reste_entiere(self):
        slots = self._slots_at(7, 0)
        self.assertTrue(slots, 'a 7h, la journee doit offrir des creneaux')
        self.assertEqual(slots[0]['start_time'], '07:00')

    def test_demain_n_est_pas_rogne(self):
        from services.agent.tools.schedule import FindFreeSlotsTool
        res = FindFreeSlotsTool().execute(
            self.user, date=DEMAIN.isoformat(), min_duration_minutes=30
        )
        self.assertTrue(res.success)
        self.assertTrue(res.data['free_slots'])
        self.assertEqual(res.data['free_slots'][0]['start_time'], '07:00')
