"""
Round 9, integration. Ecrit AVANT le correctif et vu en echec.

K1  la ligne du code pour une cible changee sort telle quelle dans les faits
    (« Quart au dépanneur a changé depuis ma question, je n'ai rien supprimé.
    Redis-le si tu veux toujours. »), sans la ligne generique « Je laisse
    tomber ... » en plus. Les autres abandons gardent leur ligne D2.
"""
from django.test import SimpleTestCase, TransactionTestCase

from core.models import RecurringBlock
from core.test_agent_v2_gardes import HarnaisGardes, puces
from core.test_agent_v2_narrateur import demande
from services.agent.tools.base import ToolResult
from services.agent_v2 import outils as outils_v2
from services.agent_v2 import rendu
from services.agent_v2.registre import Registre

LIGNE = ("Quart au dépanneur a changé depuis ma question, je n'ai rien supprimé. "
         "Redis-le si tu veux toujours.")
SERIE = 'Tous les jeudis (supprimer la série).'


def changee(dem, ligne=LIGNE):
    return ToolResult(success=False, message="cible changee",
                      data={"demande": dem, "abandonnee_par_le_code": True,
                            "cible_changee": True, "decision_code": "abandonnee",
                            "ligne_cible_changee": ligne})


def abandonnee(dem):
    return ToolResult(success=False, message="abandonnee",
                      data={"demande": dem, "abandonnee_par_le_code": True,
                            "decision_code": "abandonnee"})


class K1LigneCibleChangeeRenduTests(SimpleTestCase):

    def test_la_ligne_du_code_sort_seule(self):
        registre = Registre()
        dem = demande("destructif", "d1")
        registre.ajouter("delete_block", {"block_id": 5}, changee(dem))
        faits = rendu.rendre_faits(registre)
        self.assertEqual(faits, LIGNE)
        self.assertNotIn("Je laisse tomber", faits)
        self.assertNotIn(chr(0x2014), faits)

    def test_un_autre_abandon_garde_sa_ligne(self):
        registre = Registre()
        registre.ajouter("delete_block", {"block_id": 5}, changee(demande("destructif", "d1")))
        registre.ajouter("delete_task", {"task_id": 7}, abandonnee(
            demande("destructif", "d2", outil="delete_task", cible={"titre": "Rapport"})))
        faits = rendu.rendre_faits(registre)
        self.assertIn(LIGNE, faits)
        self.assertIn("Je laisse tomber la suppression de la tâche Rapport. "
                      "Redis-le si tu veux toujours.", faits)
        self.assertNotIn("suppression de Quart", faits)

    def test_deux_fois_la_meme_ligne_ne_se_repete_pas(self):
        registre = Registre()
        registre.ajouter("delete_block", {"block_id": 5}, changee(demande("destructif", "d1")))
        registre.ajouter("delete_block", {"block_id": 5}, changee(demande("destructif", "d1")))
        self.assertEqual(rendu.rendre_faits(registre).count("a changé depuis ma question"), 1)


class K1LigneCibleChangeeBoutEnBoutTests(HarnaisGardes, TransactionTestCase):

    def test_la_puce_sur_un_bloc_deplace_dit_la_ligne(self):
        self.attendre([puces(self.demande_portee(tache='i9:1'))], SERIE)
        RecurringBlock.objects.filter(id=self.q.id).update(start_time='20:00')
        registre = Registre()
        outils_v2.appliquer_choix_en_attente(self.user, registre, SERIE, 'i9:2')
        faits = rendu.rendre_faits(registre)
        self.assertIn("a changé depuis ma question, je n'ai rien supprimé. "
                      "Redis-le si tu veux toujours.", faits)
        self.assertNotIn("Je laisse tomber", faits)
        self.assertActif(self.q)
