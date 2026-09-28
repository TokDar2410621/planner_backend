"""Memoire des preferences (2026-09-28): capture, gestion, injection.

Couvre les trois voies (explicite deterministe, gestion, inferee via chip),
les conflits, l'injection dans les briefs AGIR/DIRE et l'isolement par
utilisateur.
"""
from django.contrib.auth.models import User
from django.test import TestCase

from core.models import PreferenceUtilisateur
from services.agent_v2 import memoire
from services.agent_v2.agent import PlannerAgentV2
from services.agent_v2.prompts import prompt_agir
from services.agent_v2.registre import Registre


class MemoireBase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="memoire", password="x")
        self.autre = User.objects.create_user(username="autre", password="x")


class TestInterpretation(MemoireBase):
    def test_capture_simple(self):
        cmd = memoire.interpreter(
            "Souviens-toi que je ne veux jamais de réunion avant 9h")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.genre, "memoriser")
        self.assertEqual(cmd.enonce, "je ne veux jamais de réunion avant 9h")
        self.assertTrue(cmd.pure)

    def test_capture_variantes(self):
        for texte in ("rappelle-toi que le gym c'est mardi et jeudi",
                      "retiens que je déteste les réunions le lundi",
                      "note que mon dentiste est le Dr Tremblay",
                      "à partir de maintenant, pas de réunion après 18h",
                      "désormais je préfère le matin"):
            cmd = memoire.interpreter(texte)
            self.assertIsNotNone(cmd, texte)
            self.assertEqual(cmd.genre, "memoriser")

    def test_capture_mixte_garde_le_reste(self):
        cmd = memoire.interpreter(
            "Souviens-toi que je préfère le matin. Planifie ma réunion avec Marc")
        self.assertEqual(cmd.genre, "memoriser")
        self.assertEqual(cmd.enonce, "je préfère le matin")
        self.assertEqual(cmd.reste, "Planifie ma réunion avec Marc")
        self.assertFalse(cmd.pure)

    def test_oubli(self):
        cmd = memoire.interpreter("Oublie mes préférences du matin")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.genre, "oublier")
        self.assertIn("matin", cmd.enonce)

    def test_liste(self):
        for texte in ("Que sais-tu de moi ?",
                      "Qu'est-ce que tu sais sur moi ?",
                      "mes préférences"):
            cmd = memoire.interpreter(texte)
            self.assertIsNotNone(cmd, texte)
            self.assertEqual(cmd.genre, "lister")

    def test_pas_de_commande(self):
        for texte in ("Planifie ma réunion demain à 10h",
                      "C'est quoi mon horaire demain ?",
                      "Bonjour"):
            self.assertIsNone(memoire.interpreter(texte), texte)


class TestExecution(MemoireBase):
    def test_memoriser_puis_lister(self):
        phrase = memoire.memoriser(self.user, "pas de réunion avant 9h")
        self.assertIn("memorisee", phrase)
        self.assertEqual(
            PreferenceUtilisateur.objects.filter(user=self.user, actif=True).count(), 1)
        liste = memoire.lister(self.user)
        self.assertIn("pas de réunion avant 9h", liste)

    def test_lister_vide(self):
        self.assertIn("rien", memoire.lister(self.user).lower())

    def test_conflit_remplace_sans_hard_delete(self):
        memoire.memoriser(self.user, "je ne veux jamais de réunion avant 9h")
        phrase = memoire.memoriser(self.user, "pas de réunion avant 9h")
        self.assertIn("Remplace", phrase)
        actives = PreferenceUtilisateur.objects.filter(user=self.user, actif=True)
        self.assertEqual(actives.count(), 1)
        self.assertEqual(actives[0].enonce, "pas de réunion avant 9h")
        # L'ancienne reste en base, desactivee: historique auditable.
        self.assertEqual(
            PreferenceUtilisateur.objects.filter(user=self.user, actif=False).count(), 1)

    def test_sans_conflit_les_deux_restent(self):
        memoire.memoriser(self.user, "pas de réunion avant 9h")
        memoire.memoriser(self.user, "gym le mardi et le jeudi")
        self.assertEqual(
            PreferenceUtilisateur.objects.filter(user=self.user, actif=True).count(), 2)

    def test_oublier_flou(self):
        memoire.memoriser(self.user, "pas de réunion avant 9h")
        phrase = memoire.oublier(self.user, "les réunions du matin")
        self.assertIn("Oublie", phrase)
        self.assertEqual(
            PreferenceUtilisateur.objects.filter(user=self.user, actif=True).count(), 0)

    def test_oublier_inconnu(self):
        phrase = memoire.oublier(self.user, "mon avion pour Tokyo")
        self.assertIn("rien memorise", phrase)

    def test_categorie_auto(self):
        p = PreferenceUtilisateur.objects.create(
            user=self.user, enonce="pas de réunion avant 9h",
            categorie=memoire._categorie("pas de réunion avant 9h"))
        self.assertEqual(p.categorie, "horaire")


class TestInjection(MemoireBase):
    def test_section_vide(self):
        self.assertEqual(memoire.section_memoire(self.user), "")

    def test_section_contenu(self):
        memoire.memoriser(self.user, "pas de réunion avant 9h")
        section = memoire.section_memoire(self.user)
        self.assertIn("MEMOIRE", section)
        self.assertIn("pas de réunion avant 9h", section)

    def test_section_plafonnee(self):
        for i in range(20):
            PreferenceUtilisateur.objects.create(
                user=self.user, enonce=f"préférence numéro {i} sur les horaires du matin")
        lignes = [l for l in memoire.section_memoire(self.user).splitlines()
                   if l.strip().startswith("-")]
        self.assertEqual(len(lignes), memoire.LIMITE_INJECTION)

    def test_brief_dire_injecte(self):
        memoire.memoriser(self.user, "pas de réunion avant 9h")
        brief = PlannerAgentV2._brief_dire(
            "Planifie ma réunion", Registre(), {}, "",
            memoire=memoire.section_memoire(self.user))
        self.assertIn("pas de réunion avant 9h", brief)

    def test_brief_dire_sans_memoire_inchange(self):
        brief = PlannerAgentV2._brief_dire("Bonjour", Registre(), {}, "")
        self.assertNotIn("MEMOIRE", brief)

    def test_prompt_agir_injecte(self):
        memoire.memoriser(self.user, "pas de réunion avant 9h")
        prompt = prompt_agir(self.user)
        self.assertIn("pas de réunion avant 9h", prompt)

    def test_isolation_utilisateurs(self):
        memoire.memoriser(self.user, "pas de réunion avant 9h")
        self.assertEqual(memoire.section_memoire(self.autre), "")
        self.assertNotIn("9h", prompt_agir(self.autre))


class TestInference(MemoireBase):
    def test_chip_proposee(self):
        chip = memoire.chip_inference(
            self.user, "Je ne veux jamais de réunion avant 9h")
        self.assertIsNotNone(chip)
        self.assertEqual(chip["label"], "Mémoriser ?")
        self.assertTrue(chip["value"].startswith("souviens-toi que"))

    def test_chip_tap_rejoue_voie_explicite(self):
        # Le tap sur la chip produit un message que la voie explicite comprend.
        chip = memoire.chip_inference(
            self.user, "Je ne veux jamais de réunion avant 9h")
        cmd = memoire.interpreter(chip["value"])
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.genre, "memoriser")

    def test_pas_de_chip_sans_marqueur(self):
        self.assertIsNone(
            memoire.chip_inference(self.user, "Planifie ma réunion demain"))

    def test_pas_de_chip_question(self):
        self.assertIsNone(
            memoire.chip_inference(self.user, "Je ne viens jamais le lundi ?"))

    def test_pas_de_chip_si_deja_connue(self):
        memoire.memoriser(self.user, "je ne veux jamais de réunion avant 9h")
        self.assertIsNone(
            memoire.chip_inference(self.user, "Je ne veux jamais de réunion avant 9h"))

    def test_pas_de_chip_si_commande(self):
        self.assertIsNone(
            memoire.chip_inference(self.user, "Souviens-toi que j'aime le matin"))
