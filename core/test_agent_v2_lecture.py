"""
Ce que l'agent a VU se rend par du CODE, comme ce qu'il a fait.

Defaut remonte par Darius le 2026-08-30. A « qu'est-ce que j'ai cette
semaine ? », l'agent repondait:

    Tu as deux grosses journees (lundi avec cours et quart au depanneur,
    puis jeudi a la piscine) et trois jours bien degages.

Des noms noyes dans une prose, aucune heure, rien a lire. La cause tenait a
la PORTEE de la garantie structurelle: `bloc_factuel` ne rend que les
MUTATIONS. Sur un tour de lecture il est vide, donc tout ce que l'utilisateur
lit vient du modele, et un modele resume.

Or les outils rendent deja la matiere exacte. `get_week_schedule` donne
`days[].blocks` sous la forme « Cours de geologie (09:00-12:00) ». On la met
en page; on n'en fabrique rien et on ne demande rien au modele.
"""
from django.test import SimpleTestCase

from services.agent.tools.base import ToolResult
from services.agent_v2.redaction import bloc_lecture
from services.agent_v2.registre import Registre


SEMAINE = {
    "week_start": "2026-08-24",
    "total_hours": 10.0,
    "days": [
        {"date": "2026-08-24", "day_name": "Lundi", "block_count": 2,
         "blocks": ["Cours de geologie (09:00-12:00)", "Quart au depanneur (17:00-22:00)"]},
        {"date": "2026-08-25", "day_name": "Mardi", "block_count": 1,
         "blocks": ["Cours de chimie (14:00-16:00)"]},
        {"date": "2026-08-26", "day_name": "Mercredi", "block_count": 0, "blocks": []},
    ],
}


def _registre(*actions):
    r = Registre()
    for outil, donnees in actions:
        r.ajouter(outil, {}, ToolResult(success=True, data=donnees, message="ok"))
    return r


class ListeDeLectureTests(SimpleTestCase):
    def test_la_semaine_se_lit_jour_par_jour_avec_les_noms(self):
        """Le defaut d'origine: des noms en prose, sans heures."""
        texte = bloc_lecture(_registre(("get_week_schedule", SEMAINE)))
        self.assertIn("Lundi : Cours de geologie (09:00-12:00)", texte)
        self.assertIn("Quart au depanneur (17:00-22:00)", texte)
        self.assertIn("Mardi : Cours de chimie (14:00-16:00)", texte)

    def test_un_jour_vide_ne_prend_pas_de_ligne(self):
        """Trois jours libres n'ont pas besoin de trois lignes vides."""
        self.assertNotIn("Mercredi", bloc_lecture(_registre(("get_week_schedule", SEMAINE))))

    def test_list_blocks_groupe_par_jour(self):
        """A plat, il faudrait trier la liste de tete."""
        donnees = {"count": 3, "blocks": [
            {"title": "Cours de geologie", "day_name": "Lundi",
             "start_time": "09:00", "end_time": "12:00"},
            {"title": "Quart au depanneur", "day_name": "Lundi",
             "start_time": "17:00", "end_time": "22:00"},
            {"title": "Piscine", "day_name": "Jeudi",
             "start_time": "07:00", "end_time": "08:00"},
        ]}
        texte = bloc_lecture(_registre(("list_blocks", donnees)))
        self.assertEqual(len([l for l in texte.split("\n") if l.strip()]), 2)
        self.assertIn("Lundi : Cours de geologie (09:00-12:00), Quart au depanneur (17:00-22:00)", texte)
        self.assertIn("Jeudi : Piscine (07:00-08:00)", texte)

    def test_la_journee_se_lit_aussi(self):
        donnees = {"day_name": "Lundi", "blocks": [
            {"title": "Cours de chimie", "start_time": "14:00", "end_time": "16:00"}]}
        self.assertIn("Lundi : Cours de chimie (14:00-16:00)",
                      bloc_lecture(_registre(("get_today_schedule", donnees))))

    def test_un_tour_QUI_MODIFIE_ne_liste_pas(self):
        """Si quelque chose a change, c'est le changement qui compte et le
        compte rendu le dit deja. Empiler les deux noierait l'important."""
        r = _registre(("get_week_schedule", SEMAINE))
        r.ajouter("create_block", {"title": "Revision"},
                  ToolResult(success=True, data={"created": [1]}, message="cree"))
        self.assertEqual(bloc_lecture(r), "")

    def test_sans_lecture_d_horaire_rien_ne_s_affiche(self):
        """Contre-epreuve: on n'invente pas une liste sur un tour de politesse."""
        self.assertEqual(bloc_lecture(Registre()), "")
        self.assertEqual(bloc_lecture(_registre(("get_preferences", {"a": 1}))), "")

    def test_une_lecture_qui_ne_rend_rien_ne_casse_pas(self):
        """Une semaine entierement libre ne doit pas produire de ligne vide."""
        vide = {"days": [{"day_name": "Lundi", "blocks": []}]}
        self.assertEqual(bloc_lecture(_registre(("get_week_schedule", vide))), "")

    def test_la_derniere_lecture_fait_foi(self):
        """Si l'agent a relu apres coup, c'est la vue la plus recente qui compte."""
        ancienne = {"days": [{"day_name": "Lundi", "blocks": ["Ancien (08:00-09:00)"]}]}
        recente = {"days": [{"day_name": "Lundi", "blocks": ["Recent (10:00-11:00)"]}]}
        texte = bloc_lecture(_registre(("get_week_schedule", ancienne),
                                       ("get_week_schedule", recente)))
        self.assertIn("Recent", texte)
        self.assertNotIn("Ancien", texte)
