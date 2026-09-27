"""Cache du plan OR-Tools (niveaux 2 et 3): la proposition d'optimize_week est
resservie sans re-resoudre quand les entrees du solveur sont inchangees, et
l'apply reutilise l'arrangement valide.

Contrat de fraicheur: l'empreinte des ENTREES (pas de la solution) decide.
Si une entree change, on re-resout (comportement historique).
"""
from datetime import time, timedelta
from unittest import mock

from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import TestCase
from django.utils import timezone

from core.models import (
    RecurringBlock,
    RecurringBlockException,
    ScheduledBlock,
    Task,
    UserProfile,
)
from services.agent.tools import TOOL_MAP
from services.agent_v2.outils import (
    _cle_plan_semaine,
    _plan_propose,
    _pour_empreinte,
)
from services.scheduling.solve_day import empreinte_entrees_semaine

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


class PlanCacheTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(username="plancache")
        self.outil = TOOL_MAP["optimize_week"]
        self.bloc = RecurringBlock.objects.create(
            user=self.user, title="Sport", block_type="sport", day_of_week=0,
            start_time=time(6, 0), end_time=time(7, 0), flexibility="flexible")

    def _propose(self, **kwargs):
        params = {"start_date": LUNDI}
        params.update(kwargs)
        return _plan_propose(self.outil, self.user, params)

    def test_propose_remplit_le_cache(self):
        proposition, empreinte, arrangements = self._propose()
        self.assertTrue(proposition.success)
        self.assertEqual(len(empreinte), 12)
        self.assertEqual(len(arrangements), 7)
        cached = cache.get(_cle_plan_semaine(self.user.pk, LUNDI))
        self.assertIsNotNone(cached)
        self.assertEqual(cached["empreinte"], empreinte)

    def test_second_propose_identique_ne_resout_pas(self):
        self._propose()
        with mock.patch(
            "services.agent.tools.schedule.solve_placement",
            side_effect=AssertionError("re-solve inattendu"),
        ):
            proposition, empreinte, arrangements = self._propose()
        self.assertTrue(proposition.success)
        self.assertEqual(len(arrangements), 7)

    def test_entrees_modifiees_invalident_le_cache(self):
        import services.agent.tools.schedule as sched_module
        _, empreinte_avant, _ = self._propose()
        self.bloc.start_time = time(8, 0)
        self.bloc.end_time = time(9, 0)
        self.bloc.save(update_fields=["start_time", "end_time"])
        with mock.patch.object(
            sched_module, "solve_placement", wraps=sched_module.solve_placement
        ) as spy:
            proposition, empreinte_apres, _ = self._propose()
        self.assertTrue(proposition.success)
        # Re-solve effectif: le solveur a ete rappele (7 jours).
        self.assertEqual(spy.call_count, 7)
        self.assertNotEqual(empreinte_avant, empreinte_apres)

    def test_empreinte_couvre_toutes_les_entrees(self):
        from datetime import date
        start = date.fromisoformat(LUNDI)
        avant = empreinte_entrees_semaine(self.user, start)

        cas = []
        # 1. heure d'un bloc
        self.bloc.start_time = time(9, 0)
        self.bloc.end_time = time(10, 0)
        self.bloc.save(update_fields=["start_time", "end_time"])
        cas.append("heure bloc")
        # 2. flexibilite
        self.bloc.flexibility = "fixed"
        self.bloc.save(update_fields=["flexibility"])
        cas.append("flexibilite")
        # 3. actif -> inactif
        self.bloc.active = False
        self.bloc.save(update_fields=["active"])
        cas.append("actif")
        self.bloc.active = True
        self.bloc.save(update_fields=["active"])
        # 4. occurrence sautee
        RecurringBlockException.objects.create(
            user=self.user, recurring_block=self.bloc, date=start)
        cas.append("exception")
        # 5. tache ponctuelle
        tache = Task.objects.create(user=self.user, title="Dentiste")
        ScheduledBlock.objects.create(
            user=self.user, task=tache, date=start,
            start_time=time(14, 0), end_time=time(15, 0))
        cas.append("ponctuel")
        # 6. tache completee (ne bloque plus)
        tache.completed = True
        tache.save(update_fields=["completed"])
        cas.append("tache completee")
        # 7. profil: temps de transport (murs de trajet)
        profil = UserProfile.objects.get(user=self.user)
        profil.transport_time_minutes = 20
        profil.save(update_fields=["transport_time_minutes"])
        cas.append("profil transport")

        apres = empreinte_entrees_semaine(self.user, start)
        self.assertNotEqual(avant, apres, f"aucun des cas {cas} ne change l'empreinte")

    def test_chaque_entree_change_l_empreinte_seule(self):
        """Garde-fou fin: chaque type d'entree, isole, change l'empreinte."""
        from datetime import date
        start = date.fromisoformat(LUNDI)
        base = empreinte_entrees_semaine(self.user, start)

        self.bloc.start_time = time(9, 0)
        self.bloc.end_time = time(10, 0)
        self.bloc.save(update_fields=["start_time", "end_time"])
        self.assertNotEqual(base, empreinte_entrees_semaine(self.user, start), "heure")
        self.bloc.start_time = time(6, 0)
        self.bloc.end_time = time(7, 0)
        self.bloc.save(update_fields=["start_time", "end_time"])

        RecurringBlockException.objects.create(
            user=self.user, recurring_block=self.bloc, date=start)
        self.assertNotEqual(base, empreinte_entrees_semaine(self.user, start), "exception")

    def test_apply_reutilise_arrangements_sans_resoudre(self):
        # L'arrangement place le bloc a 08:00 alors qu'il est a 06:00:
        # l'apply doit deplacer sans appeler le solveur.
        arrangements = [_arrangement_fixe(self.bloc.id, start_min=480) for _ in range(7)]
        with mock.patch(
            "services.agent.tools.schedule.solve_placement",
            side_effect=AssertionError("re-solve inattendu a l'apply"),
        ):
            resultat = self.outil.execute(
                self.user, start_date=LUNDI, apply=True, _arrangements=arrangements)
        self.assertTrue(resultat.success)
        self.assertTrue(resultat.data["applied"])
        self.assertEqual(resultat.data["moved_count"], 1)
        self.bloc.refresh_from_db()
        self.assertEqual((self.bloc.start_time, self.bloc.end_time), (time(8, 0), time(9, 0)))

    def test_apply_sans_arrangements_resout_comme_avant(self):
        """v1 et tout appelant historique: sans _arrangements, on resout."""
        with mock.patch(
            "services.agent.tools.schedule.solve_placement",
            side_effect=[_arrangement_fixe(self.bloc.id) for _ in range(7)],
        ) as spy:
            resultat = self.outil.execute(self.user, start_date=LUNDI, apply=True)
        self.assertTrue(resultat.success)
        self.assertEqual(spy.call_count, 7)
        self.assertEqual(resultat.data["moved_count"], 1)

    def test_pour_empreinte_exclut_les_kwargs_prives(self):
        self.assertEqual(
            _pour_empreinte("optimize_week", {"apply": True, "_arrangements": [1, 2]}),
            {"apply": True},
        )
        # Les outils sans kwargs prives sont inchanges.
        self.assertEqual(
            _pour_empreinte("create_block", {"title": "X"}), {"title": "X"})

    def test_propose_date_invalide_sans_cache(self):
        proposition, empreinte, arrangements = self._propose(start_date="pas-une-date")
        self.assertFalse(proposition.success)
        self.assertIsNone(empreinte)
        self.assertIsNone(arrangements)
        self.assertIsNone(cache.get(_cle_plan_semaine(self.user.pk, "pas-une-date")))
