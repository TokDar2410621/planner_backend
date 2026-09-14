"""
Ce que l'agent a VU se rend par du CODE, comme ce qu'il a fait.

Defaut remonte par Darius le 2026-08-30. A « qu'est-ce que j'ai cette
semaine ? », l'agent repondait:

    Tu as deux grosses journees (lundi avec cours et quart au depanneur,
    puis jeudi a la piscine) et trois jours bien degages.

Des noms noyes dans une prose, aucune heure, rien a lire. La mise en page
elle-meme vit desormais dans rendu.py (lot b4, 2026-09-14), qui a ses propres
tests de format. Ici on garde ce qui doit rester vrai quel que soit le rendu
branche: les noms vus apparaissent, un tour qui modifie ne liste pas, la
derniere lecture fait foi. Les assertions sur l'ancien format « Lundi : X
(09:00-12:00) » ont ete retirees: ce format est justement celui qu'on quitte.
"""
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase

from services.agent.tools.base import ToolResult
from services.agent_v2 import redaction
from services.agent_v2.redaction import bloc_lecture
from services.agent_v2.registre import Registre


def _detail(titre, debut, fin, genre="course"):
    return {"title": titre, "start_time": debut, "end_time": fin,
            "block_type": genre, "is_flexible": False}


# Les deux formes que rend get_week_schedule: la chaine historique « blocks »
# et le detail structure ajoute pour le rendu (contrat, section 3).
SEMAINE = {
    "week_start": "2026-08-24",
    "total_hours": 10.0,
    "days": [
        {"date": "2026-08-24", "day_name": "Lundi", "block_count": 2,
         "blocks": ["Cours de geologie (09:00-12:00)", "Quart au depanneur (17:00-22:00)"],
         "detail": [_detail("Cours de geologie", "09:00", "12:00"),
                    _detail("Quart au depanneur", "17:00", "22:00", "work")]},
        {"date": "2026-08-25", "day_name": "Mardi", "block_count": 1,
         "blocks": ["Cours de chimie (14:00-16:00)"],
         "detail": [_detail("Cours de chimie", "14:00", "16:00")]},
        {"date": "2026-08-26", "day_name": "Mercredi", "block_count": 0,
         "blocks": [], "detail": []},
    ],
}


def _registre(*actions):
    r = Registre()
    for outil, donnees in actions:
        r.ajouter(outil, {}, ToolResult(success=True, data=donnees, message="ok"))
    return r


def _semaine(titre):
    return {"days": [{"date": "2026-08-24", "day_name": "Lundi", "block_count": 1,
                      "blocks": [f"{titre} (08:00-09:00)"],
                      "detail": [_detail(titre, "08:00", "09:00")]}]}


class ListeDeLectureTests(SimpleTestCase):
    def test_la_semaine_nomme_chaque_bloc_vu(self):
        """Le defaut d'origine: des noms en prose, sans heures."""
        texte = bloc_lecture(_registre(("get_week_schedule", SEMAINE)))
        for titre in ("Cours de geologie", "Quart au depanneur", "Cours de chimie"):
            self.assertIn(titre, texte)

    def test_list_blocks_nomme_chaque_bloc(self):
        donnees = {"count": 3, "blocks": [
            {"title": "Cours de geologie", "day_name": "Lundi", "day_of_week": 0,
             "start_time": "09:00", "end_time": "12:00"},
            {"title": "Quart au depanneur", "day_name": "Lundi", "day_of_week": 0,
             "start_time": "17:00", "end_time": "22:00"},
            {"title": "Piscine", "day_name": "Jeudi", "day_of_week": 3,
             "start_time": "07:00", "end_time": "08:00"},
        ]}
        texte = bloc_lecture(_registre(("list_blocks", donnees)))
        for titre in ("Cours de geologie", "Quart au depanneur", "Piscine"):
            self.assertIn(titre, texte)

    def test_la_journee_se_lit_aussi(self):
        donnees = {"date": "2026-08-24", "day_name": "Lundi", "blocks": [
            {"title": "Cours de chimie", "start_time": "14:00", "end_time": "16:00",
             "block_type": "course"}]}
        self.assertIn("Cours de chimie",
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

    def test_la_derniere_lecture_fait_foi(self):
        """Si l'agent a relu apres coup, c'est la vue la plus recente qui compte."""
        texte = bloc_lecture(_registre(("get_week_schedule", _semaine("Ancien")),
                                       ("get_week_schedule", _semaine("Recent"))))
        self.assertIn("Recent", texte)
        self.assertNotIn("Ancien", texte)

    def test_une_lecture_ratee_ne_montre_rien(self):
        r = Registre()
        r.ajouter("get_week_schedule", {}, ToolResult(
            success=False, message="Format de date invalide."))
        self.assertEqual(bloc_lecture(r), "")


class CoutureLectureTests(SimpleTestCase):
    def test_bloc_lecture_delegue_a_rendre_lecture(self):
        vus = []

        def rendre_lecture(registre, aujourdhui=None):
            vus.append(registre)
            return "LECTURE"

        faux = SimpleNamespace(rendre_lecture=rendre_lecture)
        r = _registre(("get_week_schedule", SEMAINE))
        with patch.object(redaction, "_charger_rendu", return_value=faux):
            self.assertEqual(bloc_lecture(r), "LECTURE")
        self.assertEqual(vus, [r])
