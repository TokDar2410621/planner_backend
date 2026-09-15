"""
Deux defauts vus au banc reel du 2026-09-15, choisis par Darius (points 2 et 4).

2. Un creneau deja commence propose en bouton: a 16 h 31, la lecture du jour
   affichait « Libre : 15 h 50 à 18 h » et ce creneau est devenu un bouton. La
   lecture du jour rogne maintenant le passe, comme find_free_slots.
4. Un nouveau cours cree sous le titre d'un cours existant: « mon labo de
   chimie », donne dans le formulaire du code, est devenu une seconde serie
   « Chimie générale (labo) ». Au tour de la reponse, ce titre est renvoye une
   fois au modele avec le nom de l'utilisateur.

Horloge du harnais: lundi 14 septembre 2026, 8 h, America/Toronto.
"""
from django.test import TransactionTestCase

from core.models import ConversationMessage, RecurringBlock
from core.test_agent_v2_gardes import HarnaisGardes
from services.agent.tools import execute_tool

REPONSE = "Voici mes réponses :\nJours de « mon labo de chimie »: Vendredi\nPlage horaire: 15:00 - 16:50"
VENDREDI = dict(block_type="course", days=["vendredi"], start_time="15:00", end_time="16:50")


class JourneeLibreDepuisMaintenantTests(HarnaisGardes, TransactionTestCase):

    def test_aujourd_hui_le_libre_commence_maintenant(self):
        r = execute_tool("get_today_schedule", self.user, {"date": "2026-09-14"})
        self.assertTrue(r.success)
        self.assertEqual(r.data["free_slots"][0]["start_time"], "08:00")

    def test_un_autre_jour_garde_toute_la_journee(self):
        r = execute_tool("get_today_schedule", self.user, {"date": "2026-09-15"})
        self.assertTrue(r.success)
        self.assertEqual(r.data["free_slots"][0]["start_time"], "07:00")


class TitreDuFormulaireTests(HarnaisGardes, TransactionTestCase):

    def setUp(self):
        super().setUp()
        self.bloc("Chimie générale (labo)", 3, "08:00", "09:50")

    def formulaire_pose(self, nom="mon labo de chimie"):
        ConversationMessage.objects.create(user=self.user, role="assistant", content="Quels jours ?",
                                           metadata={"formulaire_nom": nom})

    def tour(self, brut=REPONSE):
        courant = self.message_courant(brut)
        return self.outils(brut, tache=f"{self.user.pk}:{courant.pk}")

    def vendredis(self):
        return list(RecurringBlock.objects.filter(user=self.user, day_of_week=4)
                    .values_list("title", flat=True))

    def test_le_titre_d_un_cours_existant_est_renvoye_une_fois(self):
        self.formulaire_pose()
        registre, tools = self.tour()
        sortie = self.appeler(tools, "create_block", title="Chimie générale (labo)", **VENDREDI)
        self.assertIn("« mon labo de chimie »", sortie)
        self.assertNotIn("\u2014", sortie)
        self.assertEqual(self.vendredis(), [])
        self.appeler(tools, "create_block", title="Chimie générale (labo)", **VENDREDI)
        self.assertEqual(self.vendredis(), ["Chimie générale (labo)"])

    def test_le_titre_nomme_par_l_utilisateur_passe(self):
        self.bloc("Chimie générale", 0, "13:00", "15:50")
        self.formulaire_pose("mon cours de chimie générale")
        registre, tools = self.tour()
        self.appeler(tools, "create_block", title="Chimie générale", **VENDREDI)
        self.assertEqual(self.vendredis(), ["Chimie générale"])

    def test_le_nom_de_l_utilisateur_passe(self):
        self.formulaire_pose()
        registre, tools = self.tour()
        self.appeler(tools, "create_block", title="Labo de chimie", **VENDREDI)
        self.assertEqual(self.vendredis(), ["Labo de chimie"])

    def test_sans_formulaire_du_code_une_seance_de_plus_passe(self):
        ConversationMessage.objects.create(user=self.user, role="assistant", content="Autre chose",
                                           metadata={})
        registre, tools = self.tour("ajoute une seance de Chimie générale (labo) vendredi de 15 h a 16 h 50")
        self.appeler(tools, "create_block", title="Chimie générale (labo)", **VENDREDI)
        self.assertEqual(self.vendredis(), ["Chimie générale (labo)"])

    def test_un_tour_plus_ancien_n_est_pas_touche(self):
        ancien = self.message_courant("ancien message")
        self.formulaire_pose()
        self.message_courant(REPONSE)
        registre, tools = self.outils("ancien message", tache=f"{self.user.pk}:{ancien.pk}")
        self.appeler(tools, "create_block", title="Chimie générale (labo)", **VENDREDI)
        self.assertEqual(self.vendredis(), ["Chimie générale (labo)"])


class LibreDuJourRenduTests(TransactionTestCase):
    """La journee rognee ne doit pas nier le temps libre deja passe."""

    def rendre(self, date_iso):
        from core.test_agent_v2_gardes import AUJOURDHUI
        from services.agent.tools.base import ToolResult
        from services.agent_v2 import rendu
        from services.agent_v2.registre import Registre

        registre = Registre()
        registre.ajouter("get_today_schedule", {"date": date_iso},
                         ToolResult(success=True, message="ok", data={
                             "date": date_iso, "day_name": "lundi",
                             "blocks": [], "free_slots": []}))
        return rendu.rendre_lecture(registre, AUJOURDHUI)

    def test_aujourd_hui_dit_qu_il_ne_reste_plus_de_temps(self):
        self.assertIn("Plus de temps libre aujourd'hui.", self.rendre("2026-09-14"))

    def test_un_autre_jour_garde_la_phrase_d_origine(self):
        self.assertIn("Aucun moment libre dans la journée.", self.rendre("2026-09-15"))
