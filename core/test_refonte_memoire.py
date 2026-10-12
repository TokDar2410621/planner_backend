"""Conservation de préférences compatibles dans la vraie base de test."""
from django.contrib.auth.models import User
from django.test import TestCase

from core.models import PreferenceUtilisateur
from services.agent_v2.memoire import memoriser, oublier, section_memoire


class MemoireCompatibleTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="memoire-refonte")

    def actives(self):
        return list(PreferenceUtilisateur.objects.filter(user=self.user, actif=True).values_list("enonce", flat=True))

    def test_sport_lundi_et_mardi_coexistent(self):
        memoriser(self.user, "Je fais du sport le lundi")
        memoriser(self.user, "Je fais du sport le mardi")
        self.assertEqual(len(self.actives()), 2)
        self.assertIn("lundi", section_memoire(self.user))
        self.assertIn("mardi", section_memoire(self.user))

    def test_reunions_avant9_et_apres17_coexistent(self):
        memoriser(self.user, "Pas de réunion avant 9h")
        memoriser(self.user, "Pas de réunion après 17h")
        self.assertEqual(len(self.actives()), 2)

    def test_doublon_exact_normalise_garde_la_ligne_existante(self):
        memoriser(self.user, "Je fais du sport le lundi")
        origine = PreferenceUtilisateur.objects.get(user=self.user)
        memoriser(self.user, "  JE fais du sport  le lundi. ")
        self.assertEqual(PreferenceUtilisateur.objects.filter(user=self.user).count(), 1)
        origine.refresh_from_db()
        self.assertTrue(origine.actif)

    def test_oubli_partiel_ne_supprime_pas_une_preference(self):
        memoriser(self.user, "Pas de réunion avant 9h")
        reponse = oublier(self.user, "réunion")
        self.assertEqual(len(self.actives()), 1)
        self.assertIn("précis", reponse.lower())

    def test_oubli_exact_desactive_seulement_la_cible(self):
        memoriser(self.user, "Je fais du sport le lundi")
        memoriser(self.user, "Je fais du sport le mardi")
        oublier(self.user, "Je fais du sport le lundi")
        self.assertEqual(self.actives(), ["Je fais du sport le mardi"])

    def test_remplacement_explicite_par_identifiant(self):
        memoriser(self.user, "Je fais du sport le lundi")
        memoriser(self.user, "Je fais du sport le mardi")
        cible = PreferenceUtilisateur.objects.get(user=self.user, enonce="Je fais du sport le lundi")
        memoriser(self.user, "Je fais du sport le mercredi", remplace_id=cible.pk)
        self.assertCountEqual(self.actives(), ["Je fais du sport le mardi", "Je fais du sport le mercredi"])

    def test_remplacement_autre_utilisateur_refuse(self):
        autre = User.objects.create_user(username="memoire-autre")
        cible = PreferenceUtilisateur.objects.create(user=autre, enonce="Sport lundi")
        reponse = memoriser(self.user, "Sport mardi", remplace_id=cible.pk)
        cible.refresh_from_db()
        self.assertTrue(cible.actif)
        self.assertEqual(self.actives(), [])
        self.assertIn("introuvable", reponse)

    def test_stockage_plein_ne_desactive_pas_les_anciennes(self):
        for numero in range(30):
            PreferenceUtilisateur.objects.create(user=self.user, enonce=f"Préférence {numero}")
        reponse = memoriser(self.user, "Sport mardi")
        self.assertEqual(len(self.actives()), 30)
        self.assertNotIn("Sport mardi", self.actives())
        self.assertIn("limite", reponse.lower())
