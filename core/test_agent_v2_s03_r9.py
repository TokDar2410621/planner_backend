"""
K6 (banc r8, s03-2 PARTIEL, 71 s): « mets mon cours de maths », l'agent
demande quand, l'utilisateur repond « Mardi et jeudi de 16 h a 17 h 50 » et
rien n'est cree.

Cause relevee dans runs/s03-2.json: AGIR applique la regle COURS EXISTANT du
prompt (present_choices « Calcul differentiel » + « Un autre cours »). Le code
rejette l'option inventee, son refus dit « Pose plutot une question courte »,
et le tour se termine sur une question au lieu du create_block que r5 faisait
en 9,6 s. La regle contredisait le code et ne distinguait pas un AJOUT complet
d'une designation ambigue.
"""
from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase

from core.models import RecurringBlock
from services.agent.tools import execute_tool


def _bloc(user, titre, jour, debut, fin):
    from datetime import time as dtime
    return RecurringBlock.objects.create(
        user=user, title=titre, day_of_week=jour, block_type="course",
        start_time=dtime(*debut), end_time=dtime(*fin))


class RegleCoursExistantTests(SimpleTestCase):

    def test_plus_d_option_inventee_commandee(self):
        from services.agent_v2.prompts import REGLES_AGIR
        self.assertNotIn("ajoute l'option « Un autre cours »", REGLES_AGIR)

    def test_un_ajout_complet_se_cree(self):
        from services.agent_v2.prompts import REGLES_AGIR
        self.assertIn("REPONSE A TA QUESTION", REGLES_AGIR)
        # Doctrine 2026-09-29: la mecanique d'ajout vit dans la description
        # V2 de create_block, pas dans le prompt.
        from services.agent.tools import TOOL_MAP
        from services.agent_v2.outils import description_v2
        desc = description_v2(TOOL_MAP["create_block"])
        for requis in ("y compris en reponse a ta propre question",
                       "c'est un nouveau cours", "create_block dans ce tour",
                       "Ne demande ni lequel"):
            with self.subTest(requis=requis):
                self.assertIn(requis, desc)

    def test_la_designation_ambigue_reste_un_choix_entre_vrais_cours(self):
        from services.agent_v2.prompts import REGLES_AGIR
        self.assertIn('present_choices (source "blocs") avec ces seuls cours', REGLES_AGIR)
        self.assertIn("Jamais d'option inventee", REGLES_AGIR)


class RefusDUnSeulCoursTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="s03", password="x")
        _bloc(self.user, "Calcul différentiel", 0, (9, 0), (11, 50))
        _bloc(self.user, "Calcul différentiel", 2, (9, 0), (11, 50))
        _bloc(self.user, "Chimie générale", 0, (13, 0), (15, 50))

    def _choix(self, options):
        return execute_tool("present_choices", self.user, {
            "question": "Ton cours de maths, c'est lequel ?",
            "options": options, "source": "blocs"})

    def _verifie_consigne(self, r):
        self.assertFalse(r.success)
        self.assertIn("un seul élément réel correspond (« Calcul différentiel »)", r.message)
        self.assertIn("crée-le (create_block) sans demander lequel", r.message)
        self.assertNotIn("Pose plutôt une question courte", r.message)

    def test_tour_1_du_banc(self):
        r = self._choix([{"label": "Calcul différentiel", "value": "Calcul différentiel"},
                         {"label": "Un autre cours", "value": "Un autre cours"}])
        self._verifie_consigne(r)
        self.assertIn("Un autre cours", r.message)

    def test_tour_2_du_banc_valeurs_qui_racontent(self):
        # Arguments exacts de runs/s03-2.json: la valeur du vrai cours est
        # ecartee comme affirmation d'action, il reste un seul cours reel.
        tiret = chr(0x2014)
        r = self._choix([
            {"label": "Calcul différentiel",
             "value": f"C'est Calcul différentiel {tiret} déplace-le à mardi et jeudi 16 h à 17 h 50"},
            {"label": "Un autre cours",
             "value": f"Un autre cours {tiret} crée-le mardi et jeudi 16 h à 17 h 50"}])
        self._verifie_consigne(r)

    def test_aucun_element_reel_garde_le_refus_d_origine(self):
        r = self._choix([{"label": "Algèbre", "value": "Algèbre"},
                         {"label": "Un autre cours", "value": "Un autre cours"}])
        self.assertFalse(r.success)
        self.assertIn("il faut au moins 2 options réelles", r.message)
        self.assertNotIn("un seul élément réel", r.message)

    def test_deux_vrais_cours_passent_toujours(self):
        r = self._choix([{"label": "Calcul différentiel", "value": "Calcul différentiel"},
                         {"label": "Chimie générale", "value": "Chimie générale"}])
        self.assertTrue(r.success, r.message)

    def test_aucun_tiret_long_dans_le_refus(self):
        r = self._choix([{"label": "Calcul différentiel", "value": "Calcul différentiel"},
                         {"label": "Un autre cours", "value": "Un autre cours"}])
        self.assertNotIn(chr(0x2014), r.message)
