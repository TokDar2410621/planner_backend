"""La voie rapide sociale (agent.py): les simples interactions sociales
(salutation, remerciement) ne meritent pas la boucle lourde.

La decision est SEMANTIQUE (juge Jev, question typee q_interaction_sociale),
jamais une liste de mots ni une regex. Les garde-fous sont STRUCTURELS:
pas de piece jointe, pas de tap, pas de demande en attente, message court.
Seuil de confiance 0.9: rater une optimisation vaut mieux que rater une
vraie demande.
"""
from unittest import mock

from django.contrib.auth.models import User
from django.test import TestCase

from core.test_agent_v2_jugement import juger_script
from services.agent_v2.agent import PlannerAgentV2


def _script_sociale(valeur, confiance):
    """Un juge scripte qui rend (valeur, confiance) pour q_interaction_sociale."""
    return juger_script({"yo": {"sociale": (valeur, confiance)}})


class VoieRapideSocialeTests(TestCase):

    def setUp(self):
        self.user = User.objects.create(username="voie_rapide")
        self.agent = PlannerAgentV2()
        self.agent._tap = None

    def _voie(self, message, attachment=None, script=None):
        with mock.patch("services.agent_v2.jugement.juger",
                        script or _script_sociale("oui", 0.95)):
            with mock.patch("services.agent_v2.demandes.demandes_en_attente",
                            return_value=[]):
                return self.agent._voie_rapide_sociale(
                    self.user, message, attachment)

    def test_salutation_prend_la_voie_rapide(self):
        self.assertTrue(self._voie("yo"))

    def test_vraie_demande_ne_prend_pas_la_voie_rapide(self):
        script = juger_script({"supprime mon gym": {"sociale": ("non", 0.95)}})
        self.assertFalse(self._voie("supprime mon gym", script=script))

    def test_confiance_basse_ne_prend_pas_la_voie_rapide(self):
        script = _script_sociale("oui", 0.85)
        self.assertFalse(self._voie("yo", script=script))

    def test_juge_incertain_ne_prend_pas_la_voie_rapide(self):
        script = juger_script({})
        self.assertFalse(self._voie("yo", script=script))

    def test_message_long_ne_prend_pas_la_voie_rapide(self):
        long = "yo " * 10
        self.assertFalse(self._voie(long.strip()))

    def test_piece_jointe_ne_prend_pas_la_voie_rapide(self):
        faux_doc = mock.Mock()
        self.assertFalse(self._voie("yo", attachment=faux_doc))

    def test_demande_en_attente_ne_prend_pas_la_voie_rapide(self):
        with mock.patch("services.agent_v2.jugement.juger",
                        _script_sociale("oui", 0.95)):
            with mock.patch("services.agent_v2.demandes.demandes_en_attente",
                            return_value=[{"cle": "x"}]):
                self.assertFalse(
                    self.agent._voie_rapide_sociale(self.user, "yo", None))

    def test_tap_ne_prend_pas_la_voie_rapide(self):
        self.agent._tap = {"demande": "x", "option": "y"}
        self.assertFalse(self._voie("yo"))
