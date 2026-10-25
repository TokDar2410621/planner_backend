"""Question du modele, reponse structuree et isolation des conversations."""
import asyncio
from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase, TransactionTestCase
from django.utils import timezone

from core.models import Conversation, ConversationMessage
from services.agent.tools import TOOL_MAP
from services.agent_v2.reponse_question import consommer_reponse, ReponseQuestionRefusee
from core.test_agent_v2_narrateur import NarrateurBase
from services.agent_v2 import redaction, rendu
from services.agent_v2.outils import outils_pour
from services.agent_v2.registre import Registre


class AskQuestionCustomTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("question-user")
        self.conversation = Conversation.objects.create(user=self.user)

    def test_choice_ne_tronque_pas_les_options(self):
        result = TOOL_MAP["ask_question_custom"].execute(
            self.user, kind="choice", question="Laquelle ?",
            options=[{"label": str(n), "value": str(n)} for n in range(5)])
        self.assertFalse(result.success)
        self.assertNotIn("demande", result.data)

    def test_confirmation_ne_peut_pas_etre_inventee(self):
        result = TOOL_MAP["ask_question_custom"].execute(
            self.user, kind="confirmation", question="Je supprime tout ?")
        self.assertFalse(result.success)
        self.assertNotIn("demande", result.data)

    def test_formulaire_autre_et_reponse_unique(self):
        result = TOOL_MAP["ask_question_custom"].execute(
            self.user, kind="form", question="Lequel ?", fields=[{
                "id": "choix", "type": "radio", "label": "Choix", "question": "Lequel ?",
                "options": [{"label": "A", "value": "a"}, {"label": "B", "value": "b"}],
                "allow_other": True,
            }])
        self.assertTrue(result.success, result.message)
        self.assertTrue(result.data["interactive_inputs"][0]["allowOther"])
        question = result.data["question_custom"]
        assistant = ConversationMessage.objects.create(
            user=self.user, conversation=self.conversation, role="assistant",
            content="Lequel ?", metadata={"question_custom": question})
        courant = ConversationMessage.objects.create(
            user=self.user, conversation=self.conversation, role="user", content="Autre")
        payload = {"id": question["id"], "values": {"choix": "Autre option"}}
        self.assertEqual(consommer_reponse(self.user, self.conversation, courant, payload), payload["values"])
        assistant.refresh_from_db()
        self.assertEqual(assistant.metadata["question_answered_by"], courant.pk)
        with self.assertRaises(ReponseQuestionRefusee):
            consommer_reponse(self.user, self.conversation, courant, payload)

    def test_reponse_autre_conversation_refusee(self):
        question = {"id": "q1", "kind": "text", "fields": [{"id": "reponse", "type": "text"}]}
        autre = Conversation.objects.create(user=self.user)
        ConversationMessage.objects.create(user=self.user, conversation=autre, role="assistant",
                                           content="Pourquoi ?", metadata={"question_custom": question})
        courant = ConversationMessage.objects.create(user=self.user, conversation=self.conversation,
                                                     role="user", content="Parce que")
        with self.assertRaises(ReponseQuestionRefusee):
            consommer_reponse(self.user, self.conversation, courant,
                              {"id": "q1", "values": {"reponse": "Parce que"}})

    def test_reponse_expiree_refusee(self):
        question = {"id": "q2", "kind": "text", "fields": [{"id": "reponse", "type": "text"}]}
        assistant = ConversationMessage.objects.create(
            user=self.user, conversation=self.conversation, role="assistant",
            content="Pourquoi ?", metadata={"question_custom": question})
        ConversationMessage.objects.filter(pk=assistant.pk).update(
            created_at=timezone.now() - timedelta(minutes=31))
        courant = ConversationMessage.objects.create(
            user=self.user, conversation=self.conversation, role="user", content="Parce que")
        with self.assertRaises(ReponseQuestionRefusee):
            consommer_reponse(self.user, self.conversation, courant,
                              {"id": "q2", "values": {"reponse": "Parce que"}})

    def test_valeur_invalide_permet_de_representer_le_formulaire(self):
        question = {"id": "q3", "kind": "text", "fields": [{"id": "reponse", "type": "text"}]}
        ConversationMessage.objects.create(user=self.user, conversation=self.conversation,
                                           role="assistant", content="Pourquoi ?",
                                           metadata={"question_custom": question})
        courant = ConversationMessage.objects.create(user=self.user, conversation=self.conversation,
                                                     role="user", content="Vide")
        with self.assertRaises(ReponseQuestionRefusee) as erreur:
            consommer_reponse(self.user, self.conversation, courant,
                              {"id": "q3", "values": {"reponse": ""}})
        self.assertEqual(erreur.exception.question, question)


class AskQuestionCustomRenduTests(NarrateurBase):
    def test_choix_conserve_ses_boutons_et_son_identifiant(self):
        result = TOOL_MAP["ask_question_custom"].execute(
            self.user, kind="choice", question="Tu préfères quand ?",
            options=[{"label": "Matin", "value": "Le matin."},
                     {"label": "Soir", "value": "Le soir."}])
        self.assertTrue(result.success, result.message)
        with patch.object(redaction, "_charger_rendu", return_value=rendu):
            _, done = self.tour(actions=[("ask_question_custom", {}, result)],
                                message="Je veux choisir une heure")
        self.assertEqual(len(done["quick_replies"]), 2)
        self.assertEqual(done["question_custom"]["id"], result.data["question_custom"]["id"])
        self.assertEqual(self.metadonnees()["question_custom"], done["question_custom"])

    def test_question_et_champs_survivent_au_done_et_a_l_historique(self):
        result = TOOL_MAP["ask_question_custom"].execute(
            self.user, kind="text", question="Quel horaire te convient ?")
        self.assertTrue(result.success, result.message)
        _, done = self.tour(actions=[("ask_question_custom", {}, result)], message="Je veux changer l'heure")
        self.assertTrue(done["question_posee"])
        self.assertEqual(done["interactive_inputs"][0]["type"], "text")
        self.assertEqual(done["question_custom"]["id"], result.data["question_custom"]["id"])
        self.assertEqual(self.metadonnees()["question_custom"], done["question_custom"])


class AskQuestionCustomRuntimeTests(TransactionTestCase):
    def test_un_seul_appel_de_question_par_tour(self):
        user = User.objects.create_user("runtime-question")
        registre = Registre()
        tools = {tool.name: tool for tool in outils_pour(
            user, registre, "Quelle heure ?", tache="test:question", message_brut="Quelle heure ?")}

        def appeler(question):
            return asyncio.run(tools["ask_question_custom"].function_schema.function(
                kind="choice", question=question,
                options=[{"label": "Matin", "value": "Le matin."},
                         {"label": "Soir", "value": "Le soir."}]))

        premier = appeler("Tu préfères quand ?")
        second = appeler("Tu es libre quand ?")
        self.assertIn('"success": true', premier)
        self.assertIn("deja pose une question", second)
