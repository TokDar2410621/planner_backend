"""
Round 9, fixeur f9-rendu. Ecrits AVANT les correctifs et vus en echec.

K2  une annulation d'evenement ne fusionne jamais deux creneaux separes du
    meme titre (« Lecture » 9-10 h et 14-15 h ne devient pas « de 9 h a
    15 h »). Seule la queue de minuit (meme tache, lendemain a 00:00) se
    rattache au morceau du soir.
K5  « puce » ne tombe que dans un contexte d'interface: « Ton marche aux
    puces est samedi a 9 h. » survit.
"""
from django.test import SimpleTestCase

from core.test_agent_v2_final_r8 import INTERFACE
from core.test_agent_v2_rendu import faits, registre
from services.agent_v2.redaction import ReponseDire, composer
from services.agent_v2.registre import Registre


def _annulation(*rangees):
    return registre(("cancel_scheduled_block", {"date": rangees[0][1], "title": rangees[0][0]}, True,
                     {"cancelled": [{"title": t, "date": d, "start_time": s, "end_time": e}
                                    for t, d, s, e in rangees]}))


class K2AnnulationCreneauxSeparesTests(SimpleTestCase):

    def test_deux_creneaux_separes_ne_fusionnent_pas(self):
        sortie = faits(_annulation(("Lecture", "2026-09-15", "14:00", "15:00"),
                                   ("Lecture", "2026-09-15", "09:00", "10:00")))
        self.assertNotIn("9 h à 15 h", sortie)
        self.assertIn("Annulé : Lecture, demain de 9 h à 10 h et de 14 h à 15 h.", sortie)

    def test_la_queue_de_minuit_se_rattache(self):
        sortie = faits(_annulation(("Étude", "2026-09-15", "22:00", "23:59"),
                                   ("Étude", "2026-09-16", "00:00", "01:00")))
        self.assertIn("Annulé : Étude, demain de 22 h à 1 h.", sortie)

    def test_creneau_separe_et_queue_de_minuit(self):
        sortie = faits(_annulation(("Étude", "2026-09-15", "08:00", "09:00"),
                                   ("Étude", "2026-09-15", "22:00", "23:59"),
                                   ("Étude", "2026-09-16", "00:00", "01:00")))
        self.assertIn("Annulé : Étude, demain de 8 h à 9 h et de 22 h à 1 h.", sortie)
        self.assertNotIn("8 h à 1 h", sortie)

    def test_un_morceau_de_minuit_sans_soiree_reste_a_part(self):
        sortie = faits(_annulation(("Étude", "2026-09-15", "18:00", "20:00"),
                                   ("Étude", "2026-09-16", "00:00", "01:00")))
        self.assertNotIn("18 h à 1 h", sortie)
        self.assertIn("Annulé : Étude, demain de 18 h à 20 h.", sortie)
        self.assertIn("1 h", sortie)

    def test_un_seul_creneau(self):
        sortie = faits(_annulation(("Dentiste", "2026-09-15", "09:00", "10:00")))
        self.assertIn("Annulé : Dentiste, demain de 9 h à 10 h.", sortie)


class K5PuceHorsInterfaceTests(SimpleTestCase):

    def _prose(self, phrase):
        return composer(ReponseDire(ouverture=phrase), Registre(), "", None).prose

    def test_le_marche_aux_puces_survit(self):
        for phrase in ("Ton marché aux puces est samedi à 9 h.",
                       "Le marché aux puces de ton quartier ouvre à 8 h.",
                       "Les puces électroniques du collier se rechargent le soir."):
            with self.subTest(phrase=phrase):
                self.assertEqual(self._prose(phrase), phrase)

    def test_la_puce_d_interface_tombe_toujours(self):
        for phrase in INTERFACE + ("Touche une des puces.", "Choisis une puce.",
                                   "Sélectionne la puce Chaque semaine.",
                                   "Les puces te proposent deux choix."):
            with self.subTest(phrase=phrase):
                self.assertEqual(self._prose(phrase), "")
