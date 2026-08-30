"""
La section RESTE: demande contre place, une soustraction rendue par du code.

Piece de la spec (« comparateur de quantite et section RESTE ») signalee
absente des la verification du plan du 2026-08-24, reclamee par les faits le
2026-08-30: sur « ajoute 6 h de revision », l'agent placait 2 h, le bloc
factuel annoncait un succes, et rien ne nommait le manque. Le modele ne peut
pas etre charge de ce calcul.
"""
from django.test import SimpleTestCase

from services.agent.tools.base import ToolResult
from services.agent_v2.redaction import bloc_reste
from services.agent_v2.registre import Registre


def _registre_cree(*entrees, outil="create_block"):
    r = Registre()
    if outil == "schedule_task_at":
        for e in entrees:
            r.ajouter(outil, {}, ToolResult(success=True, message="ok",
                                            data={"scheduled_block": e}))
    else:
        r.ajouter(outil, {}, ToolResult(success=True, message="ok",
                                        data={"created": list(entrees)}))
    return r


UN_BLOC_2H = {"id": 1, "start_time": "14:00", "end_time": "16:00"}


class ResteTests(SimpleTestCase):
    def test_le_manque_d_heures_est_nomme(self):
        """Le cas du banc: 6 h demandees, 2 h placees."""
        ligne = bloc_reste("Ajoute 6 h de révision cette semaine",
                           _registre_cree(UN_BLOC_2H))
        self.assertEqual(ligne, "- Demandé : 6 h. Placé : 2 h. Il manque 4 h.")

    def test_le_manque_d_elements_est_nomme(self):
        ligne = bloc_reste("ajoute 3 blocs de sport cette semaine",
                           _registre_cree(UN_BLOC_2H))
        self.assertEqual(ligne, "- Demandé : 3. Créé : 1. Il en manque 2.")

    def test_la_ligne_satisfait_le_detecteur_du_banc(self):
        """C'est sa raison d'etre: _dit_le_manque cherche « manque »."""
        ligne = bloc_reste("Ajoute 6 h de révision", _registre_cree(UN_BLOC_2H))
        self.assertIn("manque", ligne.lower())

    def test_quand_tout_rentre_la_ligne_se_tait(self):
        self.assertEqual(
            bloc_reste("Ajoute 2 h de révision", _registre_cree(UN_BLOC_2H)), "")

    def test_sans_quantite_demandee_rien(self):
        self.assertEqual(
            bloc_reste("Ajoute un peu de révision", _registre_cree(UN_BLOC_2H)), "")

    def test_sans_creation_rien(self):
        """Sur une suppression ou un refus, comparer n'a aucun sens: le
        succes n'est pas annonce, le manque n'a pas a l'etre."""
        self.assertEqual(bloc_reste("supprime 3 blocs", Registre()), "")

    def test_l_overnight_compte_juste(self):
        nuit = {"id": 2, "start_time": "23:00", "end_time": "05:00",
                "is_night_shift": True}
        ligne = bloc_reste("ajoute 8 h de sommeil", _registre_cree(nuit))
        self.assertEqual(ligne, "- Demandé : 8 h. Placé : 6 h. Il manque 2 h.")

    def test_le_rejeu_idempotent_ne_compte_pas_double(self):
        """Un rejeu inscrit la MEME action une seconde fois au registre: une
        somme naive dirait 4 h placees sur 6 et se tairait a tort... ou
        pretendrait un manque faux dans l'autre sens."""
        r = _registre_cree(UN_BLOC_2H)
        r.ajouter("create_block", {}, ToolResult(success=True, message="ok",
                                                 data={"created": [UN_BLOC_2H]}))
        ligne = bloc_reste("Ajoute 6 h de révision", r)
        self.assertIn("Placé : 2 h", ligne)

    def test_les_evenements_dates_comptent_aussi(self):
        sb = {"id": 9, "start_time": "10:00", "end_time": "11:00"}
        ligne = bloc_reste("ajoute 3 séances de course",
                           _registre_cree(sb, outil="schedule_task_at"))
        self.assertEqual(ligne, "- Demandé : 3. Créé : 1. Il en manque 2.")

    def test_les_nombres_en_mots_sont_lus(self):
        ligne = bloc_reste("ajoute trois séances de sport",
                           _registre_cree(UN_BLOC_2H))
        self.assertIn("Il en manque 2", ligne)
