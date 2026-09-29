"""
Round 4, la voix: brouillon d'AGIR filtre (V1), tiret long (B2), question
ecrite hors de son champ (B3), mecanique et seconde question (B4), demande
reemise par le code apres une reponse floue (B1).

Chaque classe a ete ecrite AVANT son correctif et vue en echec.
"""
import asyncio
from datetime import time as dtime
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TransactionTestCase

from core.models import ConversationMessage, RecurringBlock
from core.test_agent_v2_narrateur import (QUESTION_PORTEE, NarrateurBase, demande,
                                          formulaire, refus)
from services.agent_v2.agent import PlannerAgentV2
from services.agent_v2.mesure import epurer_reponse, fuite_lexicale
from services.agent_v2.redaction import ReponseDire
from services.agent_v2.registre import Registre
from services.agent.tools.base import ToolResult

TIRET_LONG = chr(0x2014)


# ── V1: le brouillon d'AGIR n'entre au brief que par ses questions et offres ──




# BrouillonFiltreTests SUPPRIME (boucle unique, 2026-09-29): testait
# _brief_dire qui filtrait le brouillon d'AGIR. Le brouillon n'existe
# plus; verifier_prose coupe les affirmations sans recu (voir
# test_agent_v2_final_r5.py V1BrouillonTests).

class ParticipeNuEnProseTests(SimpleTestCase):
    def test_participe_nu_coupe_en_ouverture(self):
        for phrase in ("Gym retiré pour jeudi.", "Voilà ton gym ajouté jeudi à 9 h.",
                       "Ton gym, retiré pour jeudi."):
            with self.subTest(phrase=phrase):
                propre, n = epurer_reponse(ReponseDire(ouverture=phrase))
                self.assertEqual(propre.ouverture, "")
                self.assertEqual(n, 1)

    def test_les_phrases_legitimes_survivent(self):
        for phrase in ("Il te reste une place libre jeudi soir.",
                       "Place ta révision avant ton quart.",
                       "Dis-moi l'heure et je le place."):
            with self.subTest(phrase=phrase):
                self.assertEqual(fuite_lexicale(phrase), [])

    def test_retenue_et_participe_nu_ne_disent_rien_de_faux(self):
        """Regression demandee par la revue de verite: une portee retenue et
        une ouverture au participe nu ne produisent aucune affirmation."""
        registre = Registre()
        registre.ajouter("delete_block", {"block_id": 5}, ToolResult(
            success=False, message="retenue",
            data={"needs_confirmation": True, "demande": demande("portee_jour", "p1")}))
        propre, _ = epurer_reponse(ReponseDire(ouverture="Gym retiré pour jeudi."))
        self.assertEqual(propre.ouverture, "")

    def test_futur_simple_coupe(self):
        self.assertTrue(fuite_lexicale(
            "Je m'occuperai de te trouver les créneaux."))
        self.assertEqual(fuite_lexicale("Dis-moi l'heure et je le placerai."), [])


# ── B2: aucun tiret long ne part, et le compteur le voit ──────────────────


class TiretLongTests(NarrateurBase):
    def test_la_prose_de_dire_est_assainie(self):
        _, done = self.tour(dire=ReponseDire(
            ouverture=f"Je te propose lundi, mercredi et vendredi {TIRET_LONG} une heure chaque fois.",
            question=f"Tu préfères le matin {TIRET_LONG} ou le soir ?",
            options=[f"Matin {TIRET_LONG} 7 h", "Soir"]))
        self.assertNotIn(TIRET_LONG, done["response"])
        self.assertIn("vendredi : une heure", done["response"])
        for chip in done["quick_replies"]:
            self.assertNotIn(TIRET_LONG, chip["label"])
            self.assertNotIn(TIRET_LONG, chip["value"])
        self.assertNotIn(TIRET_LONG, self.metadonnees()["question"])

    def test_le_marqueur_tiret_long_existe(self):
        from services.agent_v2.rendu import marqueurs_bruts
        self.assertIn("tiret_long", marqueurs_bruts(f"une heure {TIRET_LONG} ajuste"))
        self.assertEqual(marqueurs_bruts("de 14 h à 16 h, 14–16"), [])


# ── B3: une question dans la prose compte comme une question ──────────────


class QuestionHorsChampTests(NarrateurBase):
    def test_question_dans_l_ouverture_est_comptee(self):
        _, done = self.tour(dire=ReponseDire(
            ouverture="Pour confirmer, est-ce que ça concerne seulement ce jeudi ou tous les jeudis?"))
        self.assertTrue(done["question_posee"])
        self.assertTrue(done["question"].endswith("?"))
        self.assertTrue(self.metadonnees()["question_posee"])
        from services.agent_v2.suggestions import tour_a_pose_une_question
        self.assertTrue(tour_a_pose_une_question(self.user))

    def test_une_seule_question_par_reponse(self):
        _, done = self.tour(dire=ReponseDire(
            ouverture="Quel jour préfères-tu pour ta révision ?",
            question="Quel jour veux-tu placer ta révision de chimie ?"))
        self.assertEqual(done["response"].count("?"), 1)
        self.assertEqual(done["question"], "Quel jour veux-tu placer ta révision de chimie ?")

    def test_les_suggestions_differees_se_taisent_sur_un_texte_qui_demande(self):
        from services.agent_v2.suggestions import tour_a_pose_une_question
        ConversationMessage.objects.create(user=self.user, role="user", content="oui")
        ConversationMessage.objects.create(
            user=self.user, role="assistant",
            content="Seulement ce jeudi 17 sept. ou tous les jeudis?",
            metadata={"agent": "v2", "question_posee": False, "quick_replies": []})
        self.assertTrue(tour_a_pose_une_question(self.user))


# ── B4: ni mecanique, ni seconde question quand le code demande deja ──────


class MecaniqueEtSecondeQuestionTests(NarrateurBase):
    def test_la_question_du_code_ecarte_la_question_en_prose(self):
        _, done = self.tour(
            actions=[refus("delete_block", demande=demande("portee_jour", "p1"))],
            dire=ReponseDire(ouverture=(
                "Parfait, tous les jeudis donc. Réponds « Tous les jeudis » pour chacun des trois. "
                "Tu veux aussi retirer ton sommeil le jeudi ?")))
        self.assertNotIn("Réponds", done["response"])
        self.assertNotIn("sommeil", done["response"])
        self.assertEqual(done["response"].count("?"), 1)
        self.assertTrue(done["response"].endswith(QUESTION_PORTEE))

    def test_le_formulaire_ne_se_decrit_pas(self):
        _, done = self.tour(actions=[formulaire()], dire=ReponseDire(
            ouverture="Remplis ce qui te convient : le tout est pré-rempli à 2 h.",
            suite="Dis-moi aussi vers quel jour tu veux le déplacer."))
        self.assertNotIn("Remplis", done["response"])
        self.assertNotIn("pré-rempli", done["response"])
        self.assertNotIn("Dis-moi aussi", done["response"])

    def test_la_mecanique_tombe_meme_sans_question_du_code(self):
        _, done = self.tour(dire=ReponseDire(
            ouverture="Coche les jours voulus.",
            suite="Touche un des boutons ci-dessous."))
        self.assertNotIn("Coche", done["response"])
        self.assertNotIn("boutons", done["response"])


# ── B1: la reponse floue fait reemettre la demande par le code ────────────



class DemandeReemiseTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="reemise", password="x")
        self.quart = RecurringBlock.objects.create(
            user=self.user, title="Quart au dépanneur", block_type="work",
            day_of_week=3, start_time=dtime(19, 0), end_time=dtime(2, 0),
            flexibility="fixed", is_night_shift=True)

    def _tour(self, message, boucle=None, dire=None):
        """Boucle unique: `boucle` est l'effet sur le registre (signature
        (self_agent, user, message, registre)), `dire` le ReponseDire rendu."""
        def _boucle(self_agent, user, msg, registre):
            if boucle is not None:
                boucle(self_agent, user, msg, registre)
            return dire or ReponseDire(ouverture="D'accord.")
        with patch.object(PlannerAgentV2, "_boucle", _boucle):
            return PlannerAgentV2().process_message(self.user, message)

    def _meta(self):
        return ConversationMessage.objects.filter(
            user=self.user, role="assistant").latest("pk").metadata

    @property
    def _supprimer(self):
        quart = self.quart

        def _boucle_effet(self_agent, user, message, registre):
            from services.agent_v2.outils import outils_pour
            outils = {t.name: t for t in outils_pour(
                user, registre, message_du_tour=message, tache=self_agent._tache,
                signaler=self_agent.signaler_outil, message_brut=self_agent._message_brut)}
            asyncio.run(outils["delete_block"].function_schema.function(block_id=quart.pk))
        return _boucle_effet

    def test_oui_flou_puis_tous_les_jeudis(self):
        premier = self._tour("efface tout jeudi", boucle=self._supprimer)
        self.assertEqual(premier["question_motif"], "portee_jour")
        cle = self._meta()["demandes"][0]["cle"]

        flou = self._tour("Oui, supprime ces trois blocs.", dire=ReponseDire(
            ouverture="Pour confirmer, est-ce que ça concerne seulement ce jeudi 17 sept. "
                      "ou tous les jeudis?"))
        self.assertEqual(flou["question_motif"], "portee_jour")
        self.assertTrue(flou["question_posee"])
        self.assertEqual([c["label"] for c in flou["quick_replies"]],
                         ["Seulement ce jeudi", "Tous les jeudis", "Non, garde tout"])
        self.assertEqual(flou["response"].count("?"), 1)
        self.assertNotIn("Pour confirmer", flou["response"])
        meta = self._meta()
        self.assertEqual([d["cle"] for d in meta["demandes"]], [cle])
        self.assertTrue(RecurringBlock.all_objects.get(pk=self.quart.pk).active)

        self._tour("Tous les jeudis")
        self.assertFalse(RecurringBlock.all_objects.get(pk=self.quart.pk).active)

    def test_une_reponse_claire_n_est_pas_reemise(self):
        self._tour("efface tout jeudi", boucle=self._supprimer)
        done = self._tour("Non, ne change rien.")
        self.assertEqual(self._meta()["demandes"], [])
        self.assertNotEqual(done["question_motif"], "portee_jour")
        self.assertTrue(RecurringBlock.all_objects.get(pk=self.quart.pk).active)
