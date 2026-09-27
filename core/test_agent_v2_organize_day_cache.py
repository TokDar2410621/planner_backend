"""Cache du plan OR-Tools pour organize_day: la proposition d'un jour est
resservie sans re-resoudre quand les entrees du solveur sont inchangees, et
l'apply reutilise l'arrangement valide.

Meme contrat de fraicheur que le plan semaine (test_agent_v2_plan_cache.py):
l'empreinte des ENTREES (pas de la solution) decide. Si une entree change,
on re-resout (comportement historique).
"""
from datetime import date, time
from unittest import mock

from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import TestCase

from core.models import (
    RecurringBlock,
    RecurringBlockException,
    ScheduledBlock,
    Task,
    UserProfile,
)
from services.agent.tools import TOOL_MAP
from services.agent_v2.outils import (
    _arrangement_jour_cache,
    _cle_plan_jour,
    _plan_propose_jour,
)
from services.scheduling.solve_day import empreinte_entrees_jour

LUNDI = "2026-10-05"  # un lundi


def _arrangement_fixe(block_id=7, start_min=480):
    debut = f"{start_min // 60:02d}:{start_min % 60:02d}"
    fin = f"{(start_min + 60) // 60:02d}:{(start_min + 60) % 60:02d}"
    return [{
        "block_id": block_id, "title": "Sport", "block_type": "sport",
        "start_min": start_min, "end_min": start_min + 60,
        "start_time": debut, "end_time": fin,
        "preferred": True, "shrunk": False, "skipped": False,
        "overnight_kept": False, "reporte_au_lendemain": None,
    }]


class PlanJourCacheTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(username="jourcache")
        self.outil = TOOL_MAP["organize_day"]
        self.bloc = RecurringBlock.objects.create(
            user=self.user, title="Sport", block_type="sport", day_of_week=0,
            start_time=time(6, 0), end_time=time(7, 0), flexibility="flexible")
        self.jour = date.fromisoformat(LUNDI)

    def _propose(self, **kwargs):
        params = {"date": LUNDI}
        params.update(kwargs)
        return _plan_propose_jour(self.outil, self.user, params)

    def test_propose_remplit_le_cache(self):
        proposition, entrees, arrangement = self._propose()
        self.assertTrue(proposition.success)
        self.assertEqual(len(entrees), 16)
        self.assertTrue(arrangement)
        cached = cache.get(_cle_plan_jour(self.user.pk, LUNDI))
        self.assertIsNotNone(cached)
        self.assertEqual(cached["entrees"], entrees)
        self.assertEqual(cached["data"], proposition.data)

    def test_second_propose_identique_ne_resout_pas(self):
        premiere, _, _ = self._propose()
        with mock.patch(
            "services.agent.tools.schedule.solve_placement",
            side_effect=AssertionError("re-solve inattendu"),
        ):
            seconde, _, arrangement = self._propose()
        self.assertTrue(seconde.success)
        self.assertEqual(seconde.data, premiere.data)
        self.assertTrue(arrangement)

    def test_entrees_modifiees_invalident_le_cache(self):
        import services.agent.tools.schedule as sched_module
        _, entrees_avant, _ = self._propose()
        self.bloc.start_time = time(8, 0)
        self.bloc.end_time = time(9, 0)
        self.bloc.save(update_fields=["start_time", "end_time"])
        with mock.patch.object(
            sched_module, "solve_placement", wraps=sched_module.solve_placement
        ) as spy:
            proposition, entrees_apres, _ = self._propose()
        self.assertTrue(proposition.success)
        # Re-solve effectif: le solveur a ete rappele (1 jour).
        self.assertEqual(spy.call_count, 1)
        self.assertNotEqual(entrees_avant, entrees_apres)

    def test_chaque_entree_change_l_empreinte_jour(self):
        """Garde-fou: chaque type d'entree, isole, change l'empreinte du jour."""
        avant = empreinte_entrees_jour(self.user, self.jour)

        # 1. heure d'un bloc
        self.bloc.start_time = time(9, 0)
        self.bloc.end_time = time(10, 0)
        self.bloc.save(update_fields=["start_time", "end_time"])
        self.assertNotEqual(avant, empreinte_entrees_jour(self.user, self.jour), "heure bloc")
        self.bloc.start_time = time(6, 0)
        self.bloc.end_time = time(7, 0)
        self.bloc.save(update_fields=["start_time", "end_time"])

        # 2. flexibilite
        self.bloc.flexibility = "fixed"
        self.bloc.save(update_fields=["flexibility"])
        self.assertNotEqual(avant, empreinte_entrees_jour(self.user, self.jour), "flexibilite")
        self.bloc.flexibility = "flexible"
        self.bloc.save(update_fields=["flexibility"])

        # 3. actif -> inactif
        self.bloc.active = False
        self.bloc.save(update_fields=["active"])
        self.assertNotEqual(avant, empreinte_entrees_jour(self.user, self.jour), "actif")
        self.bloc.active = True
        self.bloc.save(update_fields=["active"])

        # 4. occurrence sautee le jour meme
        RecurringBlockException.objects.create(
            user=self.user, recurring_block=self.bloc, date=self.jour)
        self.assertNotEqual(avant, empreinte_entrees_jour(self.user, self.jour), "exception")

        # 5. tache ponctuelle le jour meme
        tache = Task.objects.create(user=self.user, title="Dentiste")
        ScheduledBlock.objects.create(
            user=self.user, task=tache, date=self.jour,
            start_time=time(14, 0), end_time=time(15, 0))
        self.assertNotEqual(avant, empreinte_entrees_jour(self.user, self.jour), "ponctuel")

        # 6. tache completee (ne bloque plus)
        tache.completed = True
        tache.save(update_fields=["completed"])
        self.assertNotEqual(avant, empreinte_entrees_jour(self.user, self.jour), "tache completee")

        # 7. profil: temps de transport (murs de trajet)
        profil = UserProfile.objects.get(user=self.user)
        profil.transport_time_minutes = 20
        profil.save(update_fields=["transport_time_minutes"])
        self.assertNotEqual(avant, empreinte_entrees_jour(self.user, self.jour), "profil transport")

    def test_apply_applique_l_arrangement_fourni(self):
        arrangement = _arrangement_fixe(block_id=self.bloc.id, start_min=540)
        resultat = self.outil.execute(
            self.user, date=LUNDI, apply=True, _arrangement=arrangement)
        self.assertTrue(resultat.success)
        self.assertTrue(resultat.data["applied"])
        self.bloc.refresh_from_db()
        self.assertEqual((self.bloc.start_time.hour, self.bloc.start_time.minute), (9, 0))
        self.assertEqual((self.bloc.end_time.hour, self.bloc.end_time.minute), (10, 0))
        self.assertEqual(resultat.data["moved"],
                         [{"title": "Sport", "start_time": "09:00", "end_time": "10:00"}])

    def test_apply_via_cache_ne_resout_pas(self):
        """Niveau 3 de bout en bout: le cache alimente execute, sans re-solve."""
        self._propose()
        arrangement, _ = _arrangement_jour_cache(self.user, self.jour)
        self.assertIsNotNone(arrangement)
        with mock.patch(
            "services.agent.tools.schedule.solve_placement",
            side_effect=AssertionError("re-solve inattendu"),
        ):
            resultat = self.outil.execute(
                self.user, date=LUNDI, apply=True, _arrangement=arrangement)
        self.assertTrue(resultat.success)
        self.assertTrue(resultat.data["applied"])

    def test_apply_sans_arrangement_resout_comme_avant(self):
        """v1 / chemin historique: sans kwarg prive, on re-solve."""
        import services.agent.tools.schedule as sched_module
        with mock.patch.object(
            sched_module, "solve_placement", wraps=sched_module.solve_placement
        ) as spy:
            resultat = self.outil.execute(self.user, date=LUNDI, apply=True)
        self.assertTrue(resultat.success)
        self.assertEqual(spy.call_count, 1)

    def test_propose_date_invalide_sans_cache(self):
        proposition, entrees, arrangement = _plan_propose_jour(
            self.outil, self.user, {"date": "pas-une-date"})
        self.assertFalse(proposition.success)
        self.assertIsNone(entrees)
        self.assertIsNone(arrangement)
        self.assertIsNone(cache.get(_cle_plan_jour(self.user.pk, "pas-une-date")))
