"""
Mode « une seule tete » (voix_agir, 2026-09-17): celui qui reflechit parle.

Le texte final d'AGIR devient la reponse et DIRE ne tourne pas. Le contrat
du narrateur unique reste ENTIER: la prose d'AGIR passe par la meme
epuration (aucune action affirmee, les faits du registre parlent) et par le
meme composer (questions promues, question du code prioritaire). Bascule
par profil, defaut eteint: flag off = comportement DIRE inchange.
"""
from unittest.mock import MagicMock, patch

from core.models import ConversationMessage
from core.test_agent_v2_narrateur import (NarrateurBase, QUESTION_PORTEE,
                                          demande, ok, refus)
from services.agent_v2.agent import PlannerAgentV2
from services.agent_v2.redaction import ReponseDire


class VoixAgirBase(NarrateurBase):

    def setUp(self):
        super().setUp()
        profil = self.user.profile
        profil.voix_agir = True
        profil.save(update_fields=["voix_agir"])

    def tour_voix(self, brouillon="", actions=(), message="bonjour"):
        """Un tour ou AGIR laisse `brouillon` en texte final; DIRE est un
        mouchard qui ne doit jamais etre appele."""
        dire = MagicMock(return_value=ReponseDire(ouverture="DIRE A PARLE"))

        def _agir(self_agent, user, msg, registre):
            for outil, params, res in actions:
                self_agent.signaler_outil(registre.ajouter(outil, params, res))
            self_agent._brouillon_agir = brouillon
            return ""

        with patch.object(PlannerAgentV2, "_agir", _agir), \
             patch.object(PlannerAgentV2, "_dire", dire):
            evts = list(PlannerAgentV2().process_message_stream(self.user, message))
        return evts, evts[-1], dire


class VoixAgirTests(VoixAgirBase):

    def test_agir_parle_et_dire_ne_tourne_pas(self):
        _, done, dire = self.tour_voix(
            brouillon="Bonne idée. Tu préfères le matin ou le soir ?")
        dire.assert_not_called()
        self.assertIn("Bonne idée.", done["response"])
        # Sa question en prose devient LA question du tour.
        self.assertTrue(done["response"].endswith("Tu préfères le matin ou le soir ?"))
        self.assertTrue(done["question_posee"])

    def test_une_affirmation_d_action_est_epuree_les_faits_parlent(self):
        _, done, dire = self.tour_voix(
            brouillon="J'ai ajouté ton gym jeudi. Bonne séance !",
            actions=[ok("create_block", created=[{"title": "Gym"}])])
        dire.assert_not_called()
        # Le registre parle (FAITS du faux rendu); la phrase d'action d'AGIR
        # est retiree, le reste de sa voix survit.
        self.assertIn("FAITS", done["response"])
        self.assertNotIn("J'ai ajout", done["response"])
        self.assertIn("Bonne séance !", done["response"])

    def test_sous_une_garde_la_question_du_code_gagne(self):
        d = demande("portee_jour", "p1")
        _, done, dire = self.tour_voix(
            brouillon="Je peux le faire. Tu veux supprimer quoi exactement ?",
            actions=[refus("delete_block", demande=d)])
        dire.assert_not_called()
        self.assertEqual(done["question"], QUESTION_PORTEE)
        self.assertEqual(done["question_motif"], "portee_jour")
        # Motif destructif: le code parle seul, la prose d'AGIR se tait.
        self.assertNotIn("Je peux le faire", done["response"])
        self.assertNotIn("quoi exactement", done["response"])
        # La demande rendue est persistee comme d'habitude.
        self.assertEqual([x["cle"] for x in self.metadonnees()["demandes"]], ["p1"])

    def test_le_brouillon_vide_laisse_les_faits_parler(self):
        _, done, dire = self.tour_voix(
            brouillon="", actions=[ok("create_block", created=[{"title": "Gym"}])])
        dire.assert_not_called()
        self.assertIn("FAITS", done["response"])

    def test_flag_eteint_regression_zero(self):
        profil = self.user.profile
        profil.voix_agir = False
        profil.save(update_fields=["voix_agir"])
        _, done = self.tour()  # harnais NarrateurBase: DIRE simule repond Ok.
        self.assertEqual(done["response"], "Ok.")

    def test_le_message_assistant_est_persiste_normalement(self):
        self.tour_voix(brouillon="Avec plaisir.")
        dernier = ConversationMessage.objects.filter(
            user=self.user, role="assistant").latest("pk")
        self.assertEqual(dernier.content, "Avec plaisir.")
        self.assertEqual(dernier.metadata.get("agent"), "v2")
