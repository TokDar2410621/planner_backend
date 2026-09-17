"""
La LISTE /recurring-blocks/ ne montre que les blocs vivants.

Un bloc archive par l'agent (soft delete: active=False, chemin delete_block
et clear_all_blocks) n'est plus un bloc du planning: la liste ne le montre
plus. Vu en prod le 2026-09-16: une serie supprimee par le chat restait
dans la liste.

Le DETAIL reste adressable par id: le contrat PATCH active en depend
(desactiver = metadonnee libre, reactiver = valide contre l'horaire,
test_qa_supervision_fixes.RecurringOverlapUpdateTests). Les flux pending
passent par all_objects et ne changent pas.
"""
from datetime import time as dt_time

from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework.test import APIClient

from core.models import RecurringBlock


class BlocsActifsSeulementTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(username="actifs", password="x")
        self.client_api = APIClient()
        self.client_api.force_authenticate(self.user)
        self.vivant = RecurringBlock.objects.create(
            user=self.user, title="Gym", block_type="sport", day_of_week=3,
            start_time=dt_time(18, 0), end_time=dt_time(19, 0))
        self.archive = RecurringBlock.objects.create(
            user=self.user, title="Ancien quart", block_type="work", day_of_week=4,
            start_time=dt_time(9, 0), end_time=dt_time(17, 0), active=False)

    def _ids(self, data):
        items = data.get("results", data) if isinstance(data, dict) else data
        return {b["id"] for b in items}

    def test_la_liste_ne_montre_que_les_blocs_vivants(self):
        reponse = self.client_api.get("/api/recurring-blocks/")
        self.assertEqual(reponse.status_code, 200)
        self.assertEqual(self._ids(reponse.data), {self.vivant.id})

    def test_le_detail_d_un_bloc_archive_reste_adressable(self):
        # Le contrat PATCH active (QA supervision) exige d'atteindre un bloc
        # inactif par id: desactivation et re-titrage restent des editions
        # de metadonnees, seule la reactivation repasse par l'horaire.
        reponse = self.client_api.get(f"/api/recurring-blocks/{self.archive.id}/")
        self.assertEqual(reponse.status_code, 200)
        patch = self.client_api.patch(
            f"/api/recurring-blocks/{self.archive.id}/", {"title": "Vieux quart"},
            format="json")
        self.assertEqual(patch.status_code, 200)
        # Mais il ne rejoint toujours pas la liste.
        liste = self.client_api.get("/api/recurring-blocks/")
        self.assertEqual(self._ids(liste.data), {self.vivant.id})

    def test_un_bloc_vivant_garde_tous_ses_chemins(self):
        reponse = self.client_api.get(f"/api/recurring-blocks/{self.vivant.id}/")
        self.assertEqual(reponse.status_code, 200)
        patch = self.client_api.patch(
            f"/api/recurring-blocks/{self.vivant.id}/", {"title": "Gym du jeudi"},
            format="json")
        self.assertEqual(patch.status_code, 200)

    def test_le_flux_pending_ne_change_pas(self):
        en_attente = RecurringBlock.all_objects.create(
            user=self.user, title="Extrait flou", block_type="course", day_of_week=1,
            start_time=dt_time(10, 0), end_time=dt_time(11, 0),
            status=RecurringBlock.STATUS_PENDING)
        liste = self.client_api.get("/api/recurring-blocks/pending/")
        self.assertEqual(liste.status_code, 200)
        self.assertEqual(self._ids(liste.data), {en_attente.id})
        confirme = self.client_api.post(f"/api/recurring-blocks/{en_attente.id}/confirm/")
        self.assertEqual(confirme.status_code, 200)
        en_attente.refresh_from_db()
        self.assertEqual(en_attente.status, RecurringBlock.STATUS_ACTIVE)
