"""
Les boutons forces par le code, branches dans le flux (deplace depuis
test_agent_v2_boutons.py, classe BranchementDansLeFluxTests).

Depuis le 2026-09-14, la question forcee ne se colle plus au texte de DIRE:
elle devient la section QUESTION du tour, ses boutons partent dans
done.quick_replies et dans les metadonnees du message persiste. Les libelles
des creneaux ne sont donc plus recopies dans la prose; c'est _historique qui
les rend au modele au tour suivant.

Ces tests passent par le vrai chargeur (_charger_question_forcee): ils
tiennent sur la branche comme apres l'integration du lot b5.
"""
from datetime import date, time as dtime
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase

from core.models import ConversationMessage, RecurringBlock, UploadedDocument
from services.agent.tools.base import ToolResult
from services.agent_v2.redaction import ReponseDire

# Un jour lointain: jamais « aujourd'hui », donc aucun rognage des creneaux
# deja passes ne vient dependre de l'heure a laquelle on lance les tests.
JOUR = date(2030, 6, 10)

CHIPS_FIN_PHYSIQUE = [
    {"label": "🏁 Je te donne la date de fin",
     "value": "Je vais te donner la date de fin pour Physique."},
    {"label": "♾️ Pas de fin prévue",
     "value": "Physique n'a pas de date de fin, garde-le tel quel."},
]


class BranchementDansLeFluxTests(TestCase):
    """Le drapeau « traite ce tour » se lit autour de _contexte_document, et
    la question doit atteindre done ET les metadonnees du message persiste."""

    def setUp(self):
        self.user = User.objects.create_user(username="flux", password="x")
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def _document(self, processed):
        doc = UploadedDocument.objects.create(
            user=self.user, file_name="horaire.pdf",
            document_type="course_schedule", processed=processed,
            extracted_data={"courses": [{"name": "Physique"}]})
        RecurringBlock.objects.create(
            user=self.user, title="Physique", block_type="course", day_of_week=2,
            start_time=dtime(9, 0), end_time=dtime(11, 0), source_document=doc)
        return doc

    def _done(self, doc, agir=None, message="voici mon horaire"):
        def _muet(self_agent, user, message, registre):
            return ""
        with patch.object(self.Agent, "_agir", agir or _muet), \
             patch.object(self.Agent, "_dire", return_value=ReponseDire(ouverture="Reçu.")):
            return self.Agent().process_message(self.user, message, doc)

    def _persiste(self):
        return ConversationMessage.objects.filter(
            user=self.user, role="assistant").latest("pk")

    def test_le_document_qui_finit_son_analyse_pendant_le_tour_declenche_la_question(self):
        doc = self._document(processed=False)

        def _analyse_terminee(_secondes):
            UploadedDocument.objects.filter(pk=doc.pk).update(processed=True)

        with self.settings(ATTACHMENT_WAIT_SECONDS=1), \
             patch("services.agent_v2.agent.time.sleep", _analyse_terminee):
            done = self._done(doc)
        self.assertEqual(done["quick_replies"], CHIPS_FIN_PHYSIQUE)
        self.assertEqual(done["question_motif"], "fin_recurrence")
        self.assertTrue(done["question_posee"])
        self.assertIn("jusqu", done["question"].lower())
        self.assertTrue(done["response"].endswith(done["question"]), done["response"])
        persiste = self._persiste()
        self.assertEqual(persiste.content, done["response"])
        self.assertEqual(persiste.metadata["quick_replies"], CHIPS_FIN_PHYSIQUE)
        self.assertEqual(persiste.metadata["question_motif"], "fin_recurrence")

    def test_un_document_deja_traite_a_l_arrivee_ne_declenche_rien(self):
        """Equivalence avec v1: le meme document, envoye a un tour ulterieur
        (« c'est bon ? »), ne repose pas la question a chaque fois."""
        done = self._done(self._document(processed=True))
        self.assertEqual(done["quick_replies"], [])
        self.assertNotIn("jusqu", done["response"].lower())
        self.assertNotEqual(done["question_motif"], "fin_recurrence")

    def test_la_question_forcee_recoit_le_message_brut_et_non_le_message_enrichi(self):
        """La troisieme jambe de v1 cherche un verbe et deux heures dans le
        message, et le contexte d'un import en est plein. Le document est deja
        traite: son contexte enrichit le message vu par AGIR sans declencher
        la question de fin, ce qui isole l'argument observe."""
        from services.agent_v2 import agent as module_agent

        doc = self._document(processed=True)
        vus = {}
        vraie = module_agent._charger_question_forcee()
        appels = []

        def espion(*args, **kwargs):
            appels.append(args)
            return vraie(*args, **kwargs)

        def _observe(self_agent, user, message, registre):
            vus["agir"] = message
            return ""

        with patch.object(module_agent, "_charger_question_forcee", return_value=espion):
            done = self._done(doc, _observe)
        self.assertTrue(vus["agir"].startswith("voici mon horaire\n\n"), vus["agir"])
        self.assertEqual(len(appels), 1)
        self.assertEqual(appels[0][1], "voici mon horaire")
        self.assertEqual(done["quick_replies"], [])

    def test_les_creneaux_vont_dans_les_boutons_et_non_dans_la_prose(self):
        """Les libelles ne sont plus recopies dans le texte: ils vivent dans
        done.quick_replies et dans les metadonnees, et le texte persiste est
        exactement celui de done."""
        RecurringBlock.objects.create(
            user=self.user, title="Cours", block_type="course",
            day_of_week=JOUR.weekday(), flexibility="fixed",
            start_time=dtime(7, 0), end_time=dtime(15, 0))

        def _conflit(self_agent, user, message, registre):
            registre.ajouter("schedule_task_at",
                             {"title": "Révision", "date": JOUR.isoformat(),
                              "start_time": "10:00", "end_time": "11:00"},
                             ToolResult(success=False, message="Conflit avec Cours",
                                        data={"conflict": {"title": "Cours"}}))
            return ""

        done = self._done(None, _conflit, message="planifie ma révision")
        chips = done["quick_replies"]
        self.assertTrue(1 <= len(chips) <= 4, done)
        self.assertIn("15", chips[0]["label"])
        self.assertEqual(done["question_motif"], "creneaux")
        self.assertTrue(done["question"])
        self.assertTrue(done["response"].endswith(done["question"]), done["response"])
        for chip in chips:
            self.assertNotIn(chip["label"], done["response"])
        persiste = self._persiste()
        self.assertEqual(persiste.content, done["response"])
        self.assertEqual(persiste.metadata["quick_replies"], chips)
