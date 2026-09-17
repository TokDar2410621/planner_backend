"""
Architecture « 1 tache = 1 livrable observable » (2026-09-17), phase 1.

Quatre champs nouveaux sur Task: deliverable (le livrable), done_when (le
critere de fin), goal (l'objectif servi), depends_on (la tache d'avant).
Regles: goal et depends_on fermes au proprietaire (IDOR), pas
d'auto-dependance ni de cycle, SET_NULL libere (jamais d'orpheline
bloquee). Un objectif qui a des taches liees montre un progres CALCULE
(completees/total); sans taches, le curseur manuel reste.
"""
from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework.test import APIClient

from core.models import Goal, Task


class QuatreChampsTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(username="livrable", password="x")
        self.autre = User.objects.create_user(username="intrus", password="x")
        self.client_api = APIClient()
        self.client_api.force_authenticate(self.user)
        self.objectif = Goal.objects.create(user=self.user, title="Repartir Arivex")

    def _creer(self, **extra):
        base = {"title": "Décrire le client cible", "task_type": "deep_work"}
        base.update(extra)
        return self.client_api.post("/api/tasks/", base, format="json")

    def test_une_tache_quatre_champs_se_cree_et_se_relit(self):
        r = self._creer(
            deliverable="Une fiche de 5 lignes maximum.",
            done_when="Type d'entreprise, taille, décideur et contexte y figurent.",
            goal=self.objectif.id)
        self.assertEqual(r.status_code, 201, r.content)
        self.assertEqual(r.data["deliverable"], "Une fiche de 5 lignes maximum.")
        self.assertEqual(r.data["goal"], self.objectif.id)
        self.assertFalse(r.data["is_blocked"])
        relu = self.client_api.get(f"/api/tasks/{r.data['id']}/")
        self.assertEqual(relu.data["done_when"],
                         "Type d'entreprise, taille, décideur et contexte y figurent.")

    def test_l_objectif_d_un_autre_compte_est_refuse(self):
        vol = Goal.objects.create(user=self.autre, title="Pas à toi")
        r = self._creer(goal=vol.id)
        self.assertEqual(r.status_code, 400)

    def test_la_dependance_d_un_autre_compte_est_refusee(self):
        etrangere = Task.objects.create(user=self.autre, title="Ailleurs")
        r = self._creer(depends_on=etrangere.id)
        self.assertEqual(r.status_code, 400)

    def test_ni_auto_dependance_ni_cycle(self):
        a = Task.objects.create(user=self.user, title="A")
        b = Task.objects.create(user=self.user, title="B", depends_on=a)
        soi = self.client_api.patch(f"/api/tasks/{a.id}/", {"depends_on": a.id},
                                    format="json")
        self.assertEqual(soi.status_code, 400)
        cycle = self.client_api.patch(f"/api/tasks/{a.id}/", {"depends_on": b.id},
                                      format="json")
        self.assertEqual(cycle.status_code, 400)
        # La chaine saine passe: C peut dependre de B.
        c = self._creer(title="C", depends_on=b.id)
        self.assertEqual(c.status_code, 201)

    def test_is_blocked_suit_la_dependance(self):
        socle = Task.objects.create(user=self.user, title="Lister les dépôts")
        r = self._creer(title="Supprimer arivex-next", depends_on=socle.id)
        self.assertTrue(r.data["is_blocked"])
        self.client_api.post(f"/api/tasks/{socle.id}/complete/", {}, format="json")
        relu = self.client_api.get(f"/api/tasks/{r.data['id']}/")
        self.assertFalse(relu.data["is_blocked"])

    def test_supprimer_la_dependance_libere_la_tache(self):
        socle = Task.objects.create(user=self.user, title="Socle")
        suiveuse = Task.objects.create(user=self.user, title="Suiveuse", depends_on=socle)
        socle.delete()
        suiveuse.refresh_from_db()
        self.assertIsNone(suiveuse.depends_on)
        self.assertFalse(suiveuse.is_blocked)


class ProgresObservableTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(username="progres", password="x")
        self.client_api = APIClient()
        self.client_api.force_authenticate(self.user)
        self.objectif = Goal.objects.create(user=self.user, title="Arivex", progress=80)

    def test_sans_taches_le_curseur_manuel_reste(self):
        self.assertEqual(self.objectif.progres_effectif(), 80)
        r = self.client_api.get(f"/api/goals/{self.objectif.id}/")
        self.assertEqual(r.data["progress"], 80)
        self.assertEqual(r.data["tasks_total"], 0)

    def test_avec_taches_le_progres_devient_observable(self):
        Task.objects.create(user=self.user, title="Faite", goal=self.objectif,
                            completed=True)
        Task.objects.create(user=self.user, title="À faire", goal=self.objectif)
        # Le 80 manuel ne compte plus: 1 livrable atteint sur 2.
        self.assertEqual(self.objectif.progres_effectif(), 50)
        r = self.client_api.get(f"/api/goals/{self.objectif.id}/")
        self.assertEqual(r.data["progress"], 50)
        self.assertEqual(r.data["tasks_total"], 2)
        self.assertEqual(r.data["tasks_done"], 1)

    def test_completer_la_tache_fait_monter_l_objectif(self):
        t = Task.objects.create(user=self.user, title="Unique", goal=self.objectif)
        r = self.client_api.get(f"/api/goals/{self.objectif.id}/")
        self.assertEqual(r.data["progress"], 0)
        self.client_api.post(f"/api/tasks/{t.id}/complete/", {}, format="json")
        r = self.client_api.get(f"/api/goals/{self.objectif.id}/")
        self.assertEqual(r.data["progress"], 100)
