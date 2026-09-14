"""Suggestions differees: jamais destructives, jamais par-dessus une question.

Cote serveur de l'extra « chips differees »: la vue /chat/quick-replies/ ne
consulte pas le modele quand le tour vient de poser une question, et filtre
toute suggestion qui supprime, efface, vide ou annule quelque chose.
Aussi: la vue /chat/ non streamee relaie question_posee, question et
question_motif.
"""
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from core.models import ConversationMessage
from services.agent_v2 import PlannerAgentV2
from services.agent_v2.suggestions import (filtrer_suggestions,
                                           tour_a_pose_une_question)


class _AvecClient(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("differe", password="pw-123456")
        self.user.profile.ai_consent_at = timezone.now()
        self.user.profile.save(update_fields=["ai_consent_at"])
        self.client_api = APIClient()
        self.client_api.force_authenticate(self.user)

    def _message(self, role, contenu="x", metadata=None, user=None):
        return ConversationMessage.objects.create(
            user=user or self.user, role=role, content=contenu,
            metadata=metadata or {})

    def _suggestions(self):
        return self.client_api.post(
            "/api/chat/quick-replies/",
            {"message": "a", "response": "b"}, format="json")


class VueSuggestionsTests(_AvecClient):
    def test_pas_de_suggestion_quand_le_tour_a_pose_une_question(self):
        self._message("user", "ajoute mon quart jeudi")
        self._message("assistant", "À quelle heure commence ton quart ?",
                      {"question_posee": True})
        with patch.object(PlannerAgentV2, "quick_replies_for") as generer:
            reponse = self._suggestions()
        self.assertEqual(reponse.status_code, 200)
        self.assertEqual(reponse.json(), {"quick_replies": []})
        generer.assert_not_called()

    def test_chips_de_choix_ou_formulaire_comptent_comme_question(self):
        for meta in ({"quick_replies": [{"label": "19 h", "value": "19 h"}]},
                     {"interactive_inputs": [{"type": "text", "name": "heure"}]}):
            with self.subTest(meta=meta):
                self._message("user", "salut")
                self._message("assistant", "Choisis.", meta)
                with patch.object(PlannerAgentV2, "quick_replies_for") as generer:
                    reponse = self._suggestions()
                self.assertEqual(reponse.json(), {"quick_replies": []})
                generer.assert_not_called()

    def test_sans_question_les_suggestions_sont_filtrees(self):
        self._message("user", "montre ma semaine")
        self._message("assistant", "Voici ta semaine.",
                      {"question_posee": False, "quick_replies": []})
        brutes = [
            {"label": "🗑️ Vide ma journée", "value": "Vide ma journée de jeudi"},
            {"label": "📅 Ma semaine", "value": "Montre ma semaine"},
            {"label": "Ajoute du sport", "value": "Ajoute du sport mardi"},
        ]
        with patch.object(PlannerAgentV2, "quick_replies_for",
                          return_value=brutes) as generer:
            reponse = self._suggestions()
        generer.assert_called_once()
        self.assertEqual(reponse.json(), {"quick_replies": [
            {"label": "📅 Ma semaine", "value": "Montre ma semaine"},
            {"label": "Ajoute du sport", "value": "Ajoute du sport mardi"},
        ]})

    def test_une_question_deja_repondue_ne_bloque_plus(self):
        self._message("assistant", "À quelle heure ?", {"question_posee": True})
        self._message("user", "19 h")
        with patch.object(PlannerAgentV2, "quick_replies_for",
                          return_value=[]) as generer:
            reponse = self._suggestions()
        self.assertEqual(reponse.status_code, 200)
        generer.assert_called_once()

    def test_une_panne_du_modele_rend_une_liste_vide(self):
        with patch.object(PlannerAgentV2, "quick_replies_for",
                          side_effect=RuntimeError("boom")):
            reponse = self._suggestions()
        self.assertEqual(reponse.status_code, 200)
        self.assertEqual(reponse.json(), {"quick_replies": []})


class FiltreSuggestionsTests(TestCase):
    def test_suggestions_destructives_filtrees(self):
        self.assertEqual(
            filtrer_suggestions([
                {"label": "🗑️ Supprime tout", "value": "Supprime tout mon jeudi"},
                {"label": "📅 Ma semaine", "value": "Montre ma semaine"},
                {"label": "Enlève le cours", "value": "Enlève le cours de chimie"},
            ]),
            [{"label": "📅 Ma semaine", "value": "Montre ma semaine"}],
        )

    def test_suggestion_malformee_ignoree(self):
        self.assertEqual(filtrer_suggestions([{"label": "x"}]), [])
        self.assertEqual(filtrer_suggestions([
            "Montre ma semaine", None, {"label": "", "value": "x"},
            {"label": "x", "value": 3}, {"label": "  ", "value": "y"},
        ]), [])
        self.assertEqual(filtrer_suggestions(None), [])

    def test_la_valeur_seule_suffit_a_retirer(self):
        # Le libelle est anodin, mais c'est la valeur qui part au tap.
        self.assertEqual(filtrer_suggestions([
            {"label": "Ta journée", "value": "Efface ma journée de demain"},
        ]), [])

    def test_lexique_sans_accents_ni_casse(self):
        destructives = [
            "EFFACE mon lundi", "Réinitialise mon planning", "Vider la semaine",
            "Annule le rendez-vous", "Retire le gym", "Archive mes tâches",
            "Reset", "On recommence tout ?", "Enlève tout", "Supprimer le quart",
        ]
        for texte in destructives:
            with self.subTest(texte=texte):
                self.assertEqual(
                    filtrer_suggestions([{"label": texte, "value": texte}]), [])

    def test_mots_proches_conserves(self):
        gardees = [
            {"label": "Ma vidéo du soir", "value": "Ajoute ma vidéo du soir"},
            {"label": "Retour au calme", "value": "Place un retour au calme"},
            {"label": "Mes cours de jeudi", "value": "Montre mes cours de jeudi"},
        ]
        self.assertEqual(filtrer_suggestions(gardees), gardees)

    def test_au_plus_trois(self):
        items = [{"label": f"Idée {i}", "value": f"Montre l'idée {i}"}
                 for i in range(5)]
        self.assertEqual(filtrer_suggestions(items), items[:3])


class TourAPoseUneQuestionTests(_AvecClient):
    def test_sans_message(self):
        self.assertFalse(tour_a_pose_une_question(self.user))

    def test_dernier_message_par_pk(self):
        self._message("assistant", "Quelle heure ?", {"question_posee": True})
        self.assertTrue(tour_a_pose_une_question(self.user))
        self._message("assistant", "Ok.", {"question_posee": False,
                                           "quick_replies": [],
                                           "interactive_inputs": []})
        self.assertFalse(tour_a_pose_une_question(self.user))

    def test_message_v1_sans_metadonnee(self):
        self._message("assistant", "Réponse v1.")
        self.assertFalse(tour_a_pose_une_question(self.user))

    def test_la_conversation_d_un_autre_ne_compte_pas(self):
        autre = User.objects.create_user("autre", password="pw-123456")
        self._message("assistant", "Quelle heure ?", {"question_posee": True},
                      user=autre)
        self._message("assistant", "Ok.")
        self.assertFalse(tour_a_pose_une_question(self.user))
        self.assertTrue(tour_a_pose_une_question(autre))


class ChatViewQuestionTests(_AvecClient):
    def test_chat_view_relaie_question_posee(self):
        resultat = {"response": "Quelle heure ?", "question_posee": True,
                    "question": "Quelle heure ?", "question_motif": "dire"}
        with patch.object(PlannerAgentV2, "process_message",
                          return_value=resultat):
            reponse = self.client_api.post("/api/chat/", {"message": "ajoute mon quart"})
        self.assertEqual(reponse.status_code, 200)
        corps = reponse.json()
        self.assertEqual(corps["question_posee"], True)
        self.assertEqual(corps["question"], "Quelle heure ?")
        self.assertEqual(corps["question_motif"], "dire")

    def test_question_vide_du_formulaire_relayee(self):
        resultat = {"response": "", "question_posee": True, "question": "",
                    "question_motif": "formulaire",
                    "interactive_inputs": [{"type": "text", "name": "heure"}]}
        with patch.object(PlannerAgentV2, "process_message",
                          return_value=resultat):
            corps = self.client_api.post("/api/chat/", {"message": "x"}).json()
        self.assertEqual(corps["question"], "")
        self.assertEqual(corps["question_motif"], "formulaire")
        self.assertTrue(corps["question_posee"])

    def test_resultat_sans_question_n_ajoute_rien(self):
        with patch.object(PlannerAgentV2, "process_message",
                          return_value={"response": "ok"}):
            corps = self.client_api.post("/api/chat/", {"message": "salut"}).json()
        self.assertEqual(corps, {"response": "ok"})
