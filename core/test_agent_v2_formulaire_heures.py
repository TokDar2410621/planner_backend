"""Les heures d'un formulaire restent des heures d'horloge pour les gardes."""

import time
import json
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase, TransactionTestCase
from django.utils.datastructures import MultiValueDict

from core.chat_turns import _fingerprint
from core.models import ScheduledBlock
from core.test_agent_v2_gardes import HarnaisGardes
from services.agent_v2 import demandes
from services.agent_v2.conversation import AgentConversation
from services.agent_v2.trace_tour import TraceTour


class HeuresFormulaireTests(SimpleTestCase):
    def test_plage_horaire_explicitement_donnee_est_conservee(self):
        texte = "Voici mes réponses :\nHeures de l'examen: 15:00 - 17:00"

        self.assertEqual(demandes.heures_dites(texte), ["15:00", "17:00"])

    def test_duree_en_heures_reste_une_duree(self):
        self.assertEqual(demandes.heures_dites("Heures d'étude: 4 h"), [])


class PlanificationFormulaireTests(HarnaisGardes, TransactionTestCase):
    def test_la_plage_de_l_examen_est_planifiee_sans_redemander_l_heure(self):
        texte = "Voici mes réponses :\nHeures de l'examen: 15:00 - 17:00"
        self.message_courant(texte)
        registre, outils = self.outils(texte)

        self.appeler(outils, "schedule_task_at", title="Examen", date="2026-09-15",
                     start_time="15:00", end_time="17:00")

        action = registre.actions[-1]
        self.assertTrue(action.succes, action.message)
        self.assertTrue(ScheduledBlock.objects.filter(
            user=self.user, task__title="Examen", start_time="15:00",
            end_time="17:00").exists())

    def test_une_vraie_collision_est_signalee_sans_inventer_une_ambiguite(self):
        texte = "Voici mes réponses :\nHeures de l'examen: 15:00 - 17:00"
        self.bloc("Développement d'applications", 0, "14:00", "16:00")
        self.message_courant(texte)
        registre, outils = self.outils(texte)

        self.appeler(outils, "schedule_task_at", title="Examen", date="2026-09-14",
                     start_time="15:00", end_time="17:00")

        action = registre.actions[-1]
        self.assertFalse(action.succes)
        self.assertNotIn("heure_dite", action.donnees)
        self.assertEqual(action.donnees["demande"]["motif"], "heure_refusee")
        self.assertFalse(ScheduledBlock.objects.filter(user=self.user, task__title="Examen").exists())


class EntreeConversationFormulaireTests(SimpleTestCase):
    def test_la_reponse_structuree_traverse_le_flux_public(self):
        reponse = {"id": "q1", "values": {"horaire": {"start": "15:00", "end": "17:00"}}}
        trace = TraceTour(time.monotonic() + 30)
        with patch("services.agent_v2.budget_tour.ouvrir_trace", return_value=trace), \
             patch("services.agent_v2.budget_tour.fermer_trace"), \
             patch.object(AgentConversation, "_traiter_tour",
                          return_value=iter([{"type": "done", "response": "Reçu."}])) as traiter:
            evenements = list(AgentConversation().process_message_stream(
                object(), "Voici mes réponses : Heures: 15:00 - 17:00",
                question_answer=reponse))

        self.assertEqual(evenements[-1]["response"], "Reçu.")
        self.assertEqual(traiter.call_args.kwargs["question_answer"], reponse)


class IdentiteTourFormulaireTests(SimpleTestCase):
    def test_une_autre_reponse_ne_rejoue_pas_le_tour_precedent(self):
        def demande(debut):
            return SimpleNamespace(data={
                "message": "Voici mes réponses :\nHeures de l'examen: 15:00 - 17:00",
                "question_answer": json.dumps({"id": "q1", "values": {
                    "horaire": {"start": debut, "end": "17:00"}}}),
            }, FILES=MultiValueDict())

        self.assertNotEqual(_fingerprint(demande("15:00")),
                            _fingerprint(demande("16:00")))
