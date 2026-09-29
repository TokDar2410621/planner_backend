"""Banc francais (quebecois) de la couche de jugement.

Les phrases sont reelles, dont « Mon emploi du jour », le rate du
2026-09-29 qui a motive ce lot (une formulation voisine appelait
get_today_schedule, celle-ci n'a pas ete comprise). Le juge est scripte:
ces tests verifient que le code achemine la phrase BRUTE au juge, pose des
questions bien formees en francais, et applique correctement les decisions
typees. Ils ne prouvent pas que Jev comprend le quebecois: ca, c'est le
banc de validation manuel, a rejouer contre le vrai fournisseur.
"""
from unittest.mock import patch

from django.test import SimpleTestCase

from services.agent_v2 import demandes as dem
from services.agent_v2 import jugement


def _script(reponses: dict):
    def faux_juger(etat, questions):
        return {qid: {"valeur": v, "confiance": c, "probabilites": None,
                      "statut": ("decision" if c >= 0.8 else "incertain")}
                for qid, (v, c) in reponses.items()}
    return faux_juger


PORTEE = {'motif': 'portee_jour', 'cle': 'p', 'outil': 'delete_block',
          'question': 'Supprimer le cours de chimie : seulement jeudi, ou tous les jeudis ?',
          'cible': {'titre': 'Cours de chimie', 'jour': 3},
          'options': [{'id': 'occurrence'}, {'id': 'serie'}, {'id': 'annuler'}]}
DESTR = {'motif': 'destructif', 'cle': 'd', 'outil': 'delete_block',
         'question': 'Supprimer le cours de chimie ?',
         'cible': {'titre': 'Cours de chimie', 'jour': 3},
         'options': [{'id': 'confirmer'}, {'id': 'annuler'}]}


class BancTransmissionTests(SimpleTestCase):
    """Le message brut arrive tel quel au juge, sans lecture lexicale."""

    def test_message_brut_transmis_verbatim(self):
        capte = {}

        def espion(etat, questions):
            capte["etat"] = etat
            capte["questions"] = questions
            return {qid: {"valeur": None, "confiance": 0.0,
                          "probabilites": None, "statut": "indisponible"}
                    for qid in questions}

        with patch("services.agent_v2.jugement.juger", espion):
            dem.suppression_demandee("Mon emploi du jour")
        self.assertEqual(capte["etat"]["message"], "Mon emploi du jour")

    def test_questions_bien_formees(self):
        capte = {}

        def espion(etat, questions):
            capte["etat"] = etat
            capte["questions"] = questions
            return {}

        with patch("services.agent_v2.jugement.juger", espion):
            dem.option_choisie("seulement jeudi", PORTEE)
        questions = capte["questions"]
        self.assertIn("intention", questions)
        # D1: pas de question de portee en texte libre, la puce seule tranche.
        self.assertNotIn("portee", questions)
        for qid, q in questions.items():
            # Le contrat Jev valide chaque question cote client.
            jugement._question_jev(q)
            self.assertTrue(q["instructions"],
                            f"question {qid} sans instructions")
        intention = questions["intention"]
        self.assertEqual(intention["type"], "choice")
        # La question porte le contexte: le juge ne devine pas dans le vide.
        self.assertIn("cours de chimie",
                      intention["instructions"].lower().replace("é", "e"))


class BancPhrasesTests(SimpleTestCase):
    """Les phrases du banc, juge scripte: le mapping est correct."""

    def test_mon_emploi_du_jour_nest_pas_une_suppression(self):
        # Le rate du 2026-09-29: « Mon emploi du jour » n'est pas une
        # demande de suppression. Le juge tranche, pas un lexique.
        with patch("services.agent_v2.jugement.juger",
                   _script({"suppression": (False, 0.9)})):
            self.assertFalse(dem.suppression_demandee("Mon emploi du jour"))

    def test_supprime_mon_cours_de_chimie(self):
        with patch("services.agent_v2.jugement.juger",
                   _script({"suppression": (True, 0.95)})):
            self.assertTrue(
                dem.suppression_demandee("supprime mon cours de chimie"))

    def test_a_la_place_du_cours_dentreprise_doute_prudent(self):
        # Phrase ambigue du banc: le juge doute -> prudent, jamais de feu
        # vert silencieux.
        with patch("services.agent_v2.jugement.juger",
                   _script({"suppression": (True, 0.4)})):
            self.assertTrue(
                dem.suppression_demandee("À la place du cours d'entreprise"))

    def test_garde_ferme_la_demande(self):
        with patch("services.agent_v2.jugement.juger",
                   _script({"intention": ("refuse", 0.92)})):
            self.assertEqual(dem.option_choisie("garde", DESTR), "annuler")

    def test_seulement_jeudi_repose_la_question_de_portee(self):
        # D1: la portee ne se lit que sur la puce exacte. Meme quand le juge
        # comprend « seulement jeudi », le code repose la question au lieu
        # de trancher seul.
        with patch("services.agent_v2.jugement.juger",
                   _script({"intention": ("precise", 0.93)})):
            self.assertIsNone(dem.option_choisie("seulement jeudi", PORTEE))
            self.assertEqual(dem.classification_reponse("seulement jeudi",
                                                       PORTEE),
                             "reponse")

    def test_change_davis_annule(self):
        with patch("services.agent_v2.jugement.juger",
                   _script({"intention": ("refuse", 0.88)})):
            self.assertEqual(dem.option_choisie("change d'avis", PORTEE),
                             "annuler")

    def test_puce_exacte_sans_juge(self):
        portee_puces = dict(PORTEE, chips=[
            {'label': 'Seulement jeudi', 'value': 'Seulement jeudi',
             'option': 'occurrence'},
            {'label': 'Tous les jeudis', 'value': 'Tous les jeudis',
             'option': 'serie'},
            {'label': 'Annuler', 'value': 'Annuler', 'option': 'annuler'}])
        with patch("services.agent_v2.jugement.juger") as juger:
            self.assertEqual(
                dem.option_choisie("Tous les jeudis", portee_puces), "serie")
            juger.assert_not_called()
