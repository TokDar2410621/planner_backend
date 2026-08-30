from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase

from services.agent.tools.base import ToolResult
from services.agent_v2.mesure import fuite_lexicale
from services.agent_v2.redaction import ActionCitee, ReponseDire


class FuiteLexicaleTests(SimpleTestCase):
    def test_le_futur_observe_par_la_sonde_est_detecte(self):
        """Phrase reellement produite par DeepSeek sur registre vide."""
        self.assertTrue(fuite_lexicale("Je vais organiser ton planning."))

    def test_le_passe_accompli_est_detecte(self):
        self.assertTrue(fuite_lexicale("J'ai supprim\u00e9 tes blocs du lundi."))

    def test_les_accents_ne_cachent_rien(self):
        self.assertTrue(fuite_lexicale("J'ai d\u00e9plac\u00e9 ton cours."))

    def test_une_proposition_ne_declenche_rien(self):
        self.assertEqual(fuite_lexicale("Veux-tu que je supprime ce bloc ?"), [])
        self.assertEqual(fuite_lexicale("Je peux organiser ta semaine si tu veux."), [])


def _agir_qui_pense_et_cree(self_agent, user, message, registre):
    registre.ajouter(
        "create_block",
        {"title": "Maths"},
        ToolResult(success=True, message="Bloc 'Maths' cree"),
    )
    registre.ajouter_ecart("a1", "verification mesure")
    return "I need to inspect the schedule before answering."


class FluxMesureTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="mesure", password="x")
        from services.agent_v2 import PlannerAgentV2

        self.Agent = PlannerAgentV2

    def test_le_flux_emet_thinking_et_done_reste_autoritaire(self):
        brut = ReponseDire(
            ouverture="Je vais organiser ton planning.",
            actions=[
                ActionCitee(ref="a1", phrase="Maths est cale."),
                ActionCitee(ref="a42", phrase="J'ai aussi tout reorganise."),
            ],
            suite="J'ai supprime les doublons.",
        )

        with patch.object(self.Agent, "_agir", _agir_qui_pense_et_cree), patch.object(
            self.Agent, "_dire", return_value=brut
        ):
            events = list(self.Agent().process_message_stream(self.user, "organise"))

        self.assertEqual(
            [event["type"] for event in events],
            ["status", "thinking", "status", "delta", "done"],
        )
        self.assertIn("inspect the schedule", events[1]["text"])
        self.assertNotEqual(events[3]["text"], events[-1]["response"])
        # CONTRAT RETOURNE le 2026-08-30, decision de bascule a l'appui: la
        # prose qui AFFIRME une action ne part plus. « Je vais organiser » et
        # « J'ai supprime les doublons » sont des fuites: la guillotine de
        # prose les supprime (voir test_agent_v2_prose). Ce test verrouillait
        # l'ancien monde ou elles etaient livrees telles quelles.
        self.assertNotIn("Je vais organiser ton planning.", events[-1]["response"])
        self.assertNotIn("J'ai supprime les doublons.", events[-1]["response"])
        # L'action CITEE avec une vraie reference survit, elle: c'est tout
        # l'interet du canal structure.
        self.assertIn("Maths est cale.", events[-1]["response"])
        self.assertNotIn("tout reorganise", events[-1]["response"])
        self.assertEqual(events[-1]["raisonnement"], events[1]["text"])

    def test_un_tour_journalise_une_ligne_agregee_sans_contenu(self):
        brut = ReponseDire(
            ouverture="Je vais organiser ton planning.",
            actions=[
                ActionCitee(ref="a1", phrase="Maths est cale."),
                ActionCitee(ref="a404", phrase="Action inventee."),
            ],
            suite="",
        )

        with patch.object(self.Agent, "_agir", _agir_qui_pense_et_cree), patch.object(
            self.Agent, "_dire", return_value=brut
        ), self.assertLogs("services.agent_v2.agent", level="INFO") as logs:
            list(
                self.Agent().process_message_stream(
                    self.user, "secret utilisateur a ne pas journaliser"
                )
            )

        lignes = [
            ligne for ligne in logs.output if "agent_v2 tour" in ligne
        ]
        self.assertEqual(len(lignes), 1)
        self.assertEqual(logs.output, lignes)
        self.assertIn("actions=1", lignes[0])
        self.assertIn("rejetees=1", lignes[0])
        self.assertIn("fuites=1", lignes[0])
        # La guillotine de prose a coupe l'ouverture menteuse: la ligne le dit.
        self.assertIn("supprimees=", lignes[0])
        self.assertIn("ecarts=1", lignes[0])
        self.assertNotIn("secret utilisateur", lignes[0])


class LesTroisPhrasesDu18AoutTests(SimpleTestCase):
    """Le detecteur existe pour mesurer CE tour-la. S'il en rate une, il
    mesurerait zero en production tout en restant vert ici.

    Logs de prod du 2026-08-18, 21h59 a 22h02: trois tours d'affilee, zero
    appel d'outil, et l'utilisateur a attendu un resultat qui n'est jamais
    venu."""

    def test_les_trois_phrases_sont_toutes_detectees(self):
        for phrase in (
            "Je vais supprimer les blocs existants puis ajouter tes cours.",
            "Je suis en train de mettre a jour ton horaire.",
            "Je vais organiser ton planning.",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(fuite_lexicale(phrase), f"non detectee: {phrase}")

    def test_l_apostrophe_typographique_ne_cache_rien(self):
        """Un modele qui rend U+2019 au lieu de l'apostrophe droite passerait
        au travers d'une regle ecrite avec le seul caractere ASCII."""
        self.assertTrue(fuite_lexicale("J’ai créé le bloc."))

    def test_une_tournure_anodine_avec_les_memes_mots_ne_declenche_rien(self):
        """Contre-epreuve: « je vais » et « j'ai » sans verbe d'action sont du
        francais courant. Un detecteur qui les attrape noierait l'indicateur."""
        for phrase in ("Je vais bien, merci.", "J'ai une question pour toi."):
            with self.subTest(phrase=phrase):
                self.assertEqual(fuite_lexicale(phrase), [])
