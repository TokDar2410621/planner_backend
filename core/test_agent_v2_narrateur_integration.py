"""
Les flux du narrateur contre les VRAIS modules des lots b3, b4 et b5.

Ignores sur la branche b6 seule (les modules n'existent pas encore), ces tests
doivent tourner avec ZERO skip a l'etape 4 de l'integration: le decorateur
verifie la presence de rendu.py, de present_choices, de mesure.fuite_question
et de outils.appliquer_choix_en_attente.

TransactionTestCase: AGIR tourne dans un thread du pool et les outils passent
par sync_to_async; ces threads ne voient pas la transaction d'un TestCase.
"""
import asyncio
import importlib.util
from datetime import time as dtime, timedelta
from unittest import skipUnless
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TransactionTestCase
from django.utils import timezone

from core.models import ConversationMessage, RecurringBlock, RecurringBlockException
from services.agent_v2.redaction import ReponseDire


def _modules_prets() -> bool:
    try:
        if importlib.util.find_spec("services.agent_v2.rendu") is None:
            return False
        from services.agent.tools import TOOL_MAP
        from services.agent_v2 import mesure, outils
    except Exception:  # noqa: BLE001
        return False
    return ("present_choices" in TOOL_MAP
            and hasattr(mesure, "fuite_question")
            and hasattr(outils, "appliquer_choix_en_attente"))


LIBELLES_PORTEE = ["Seulement ce jeudi", "Tous les jeudis", "Non, garde tout"]


def agir_qui_appelle(*appels):
    """Un AGIR simule qui appelle de VRAIS outils par l'adaptateur de v2."""
    def _agir(self_agent, user, message, registre):
        from services.agent_v2.outils import outils_pour
        outils = {t.name: t for t in outils_pour(
            user, registre, message_du_tour=message, tache=self_agent._tache,
            signaler=self_agent.signaler_outil, message_brut=self_agent._message_brut)}
        for nom, kwargs in appels:
            asyncio.run(outils[nom].function_schema.function(**kwargs))
        return ""
    return _agir


def agir_muet(self_agent, user, message, registre):
    return ""


@skipUnless(_modules_prets(), "integration")
class NarrateurIntegrationTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="integ", password="x")
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def _quart(self):
        return RecurringBlock.objects.create(
            user=self.user, title="Quart au dépanneur", block_type="work",
            day_of_week=3, start_time=dtime(19, 0), end_time=dtime(2, 0),
            flexibility="fixed", is_night_shift=True)

    def _tour(self, message, agir=agir_muet, dire=None):
        with patch.object(self.Agent, "_agir", agir), \
             patch.object(self.Agent, "_dire",
                          return_value=dire or ReponseDire(ouverture="D'accord.")):
            return self.Agent().process_message(self.user, message)

    def _meta(self):
        return ConversationMessage.objects.filter(
            user=self.user, role="assistant").latest("pk").metadata

    def test_question_menteuse_supprimee(self):
        done = self._tour("bonjour", dire=ReponseDire(
            ouverture="Salut.", question="J'ai supprimé ton cours, autre chose ?",
            options=["Oui", "Non"]))
        self.assertEqual(done["question"], "")
        self.assertEqual(done["quick_replies"], [])
        self.assertFalse(done["question_posee"])

    def test_chips_reelles_portee_jour(self):
        q = self._quart()
        done = self._tour("efface le quart de jeudi",
                          agir_qui_appelle(("delete_block", {"block_id": q.pk})))
        self.assertEqual(done["question_motif"], "portee_jour")
        self.assertEqual([c["label"] for c in done["quick_replies"]], LIBELLES_PORTEE)
        self.assertTrue(done["question"].endswith("?"))
        self.assertTrue(RecurringBlock.all_objects.get(pk=q.pk).active)
        meta = self._meta()
        self.assertEqual(len(meta["demandes"]), 1)
        self.assertEqual(meta["demandes"][0]["motif"], "portee_jour")
        self.assertEqual({c["option"] for c in meta["demandes"][0]["chips"]},
                         {"occurrence", "serie", "annuler"})

    def _premier_tour_portee(self):
        q = self._quart()
        done = self._tour("efface le quart de jeudi",
                          agir_qui_appelle(("delete_block", {"block_id": q.pk})))
        self.assertEqual(done["question_motif"], "portee_jour")
        return q, done

    def test_deux_tours_serie_de_bout_en_bout(self):
        q, _ = self._premier_tour_portee()
        with self.assertLogs("services.agent_v2.agent", level="INFO") as logs:
            done = self._tour("Tous les jeudis (supprimer la série).")
        self.assertFalse(RecurringBlock.all_objects.get(pk=q.pk).active)
        self.assertIn("Quart au dépanneur", done["response"])
        self.assertTrue(any("choix_code=1" in ligne for ligne in logs.output), logs.output)
        self.assertEqual(self._meta()["actions"][0]["par_le_code"], True)

    def test_deux_tours_occurrence_de_bout_en_bout(self):
        q, premier = self._premier_tour_portee()
        valeur = next(c["value"] for c in premier["quick_replies"]
                      if c["label"] == "Seulement ce jeudi")
        with self.assertLogs("services.agent_v2.agent", level="INFO") as logs:
            done = self._tour(valeur)
        self.assertTrue(RecurringBlockException.objects.filter(recurring_block=q).exists())
        self.assertTrue(RecurringBlock.all_objects.get(pk=q.pk).active)
        self.assertIn("Quart au dépanneur", done["response"])
        self.assertTrue(any("choix_code=1" in ligne for ligne in logs.output), logs.output)

    def test_un_oui_seul_ne_tranche_pas_la_portee(self):
        q, _ = self._premier_tour_portee()
        with self.assertLogs("services.agent_v2.agent", level="INFO") as logs:
            self._tour("oui")
        self.assertTrue(RecurringBlock.all_objects.get(pk=q.pk).active)
        self.assertTrue(any("choix_code=0" in ligne for ligne in logs.output), logs.output)

    def test_retenue_non_posee_une_ligne(self):
        q = self._quart()
        aujourdhui = timezone.localdate()
        samedi = aujourdhui + timedelta(days=(5 - aujourdhui.weekday()) % 7 or 7)
        agir = agir_qui_appelle(
            ("delete_block", {"block_id": q.pk}),
            ("create_block", {"title": "Études", "block_type": "revision",
                              "days": [0, 1, 2, 3, 4],
                              "start_time": "08:00", "end_time": "09:00"}),
            ("schedule_task_at", {"title": "Lecture", "date": samedi.isoformat(),
                                  "start_time": "10:00", "end_time": "11:00"}),
        )
        done = self._tour("efface le quart de jeudi, ajoute mes études et ma lecture", agir)
        self.assertEqual(done["question_motif"], "portee_jour")
        self.assertEqual(done["response"].count("?"), 1, done["response"])
        self.assertIn("pas encore", done["response"])
        self.assertIn("Lecture", done["response"])
        self.assertEqual([d["motif"] for d in self._meta()["demandes"]], ["portee_jour"])

    def test_choix_du_modele_reel(self):
        RecurringBlock.objects.create(
            user=self.user, title="Calcul différentiel", block_type="course",
            day_of_week=0, start_time=dtime(10, 0), end_time=dtime(11, 50))
        RecurringBlock.objects.create(
            user=self.user, title="Physique mécanique", block_type="course",
            day_of_week=2, start_time=dtime(13, 0), end_time=dtime(15, 0))
        agir = agir_qui_appelle(("present_choices", {
            "question": "Lequel de tes cours ?",
            "options": [{"label": "Calcul différentiel", "value": "Le cours de Calcul différentiel"},
                        {"label": "Physique mécanique", "value": "Le cours de Physique mécanique"}],
            "source": "blocs"}))
        done = self._tour("déplace mon cours", agir)
        self.assertEqual(done["question_motif"], "choix_modele")
        self.assertEqual([c["label"] for c in done["quick_replies"]],
                         ["Calcul différentiel", "Physique mécanique"])
        self.assertEqual(done["question"], "Lequel de tes cours ?")

    def test_aucun_marqueur_brut_sur_une_creation(self):
        agir = agir_qui_appelle(("create_block", {
            "title": "Statistiques", "block_type": "course", "days": [4],
            "start_time": "10:00", "end_time": "12:00"}))
        done = self._tour("ajoute Statistiques le vendredi de 10 h à midi", agir)
        self.assertIn("Statistiques", done["response"])
        self.assertEqual(self._meta()["raw_markers"], [])
