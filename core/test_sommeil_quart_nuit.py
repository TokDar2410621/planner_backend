"""Sommeil a cote d'un quart de nuit (lot 1b, 2026-09-14).

Vecu en enquete: quart fixe du jeudi 19:00-02:00 et sommeil souple 23:00-07:00
tous les jours. place_day posait le sommeil du jeudi a 11:00-19:00, donc
find_free_slots voyait un jeudi plein et schedule_task_at refusait 10:30 comme
« sommeil protege (11:00-24:00) ». Le sommeil est maintenant reporte au
lendemain (vendredi 02:00-10:00) quand il tient avant midi.

Conventions: day_of_week 0 = lundi; heures murales America/Toronto; un bloc
overnight a end_time < start_time volontairement.
"""
from datetime import date, datetime, time
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from django.contrib.auth.models import User
from django.db import connection
from django.test import SimpleTestCase, TestCase
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from core.models import RecurringBlock, RecurringBlockException, UserPlace
from services.agent.tools import execute_tool
from services.scheduling import placement
from services.scheduling.placement import (
    SEUIL_REPORT_LENDEMAIN,
    _report_sommeil,
    intervalles_sommeil_reporte,
    occupied_intervals,
    open_intervals,
    place_day,
)
from services.scheduling.solve_day import solve_day, solve_placement

LUNDI = date(2026, 9, 14)
JEUDI = date(2026, 9, 17)
VENDREDI = date(2026, 9, 18)
DIMANCHE = date(2026, 9, 20)
LUNDI_SUIVANT = date(2026, 9, 21)

MAINTENANT = datetime(2026, 9, 14, 8, 0, tzinfo=ZoneInfo("America/Toronto"))
_VRAI_LOCALTIME = timezone.localtime


def _faux_localtime(value=None, timezone=None):
    if value is None:
        return MAINTENANT
    return _VRAI_LOCALTIME(value, timezone)


def _par_titre(placements, titre):
    return next(p for p in placements if p["title"] == titre)


class ReportSommeilPurTests(SimpleTestCase):
    """_report_sommeil: fonction pure, sans ORM."""

    def test_quart_19h_2h_reporte_2h_10h(self):
        self.assertEqual(
            _report_sommeil([(19 * 60, 2 * 60)], 23 * 60, 7 * 60, 480),
            {"start_min": 120, "end_min": 600},
        )

    def test_quart_19h_7h_pas_de_report(self):
        self.assertIsNone(_report_sommeil([(19 * 60, 7 * 60)], 23 * 60, 7 * 60, 480))

    def test_debut_libre_pas_de_report(self):
        # Mur qui commence apres le debut du sommeil: le debut n'est pas mure.
        self.assertIsNone(_report_sommeil([(23 * 60 + 30, 2 * 60)], 23 * 60, 7 * 60, 480))

    def test_sans_mur_pas_de_report(self):
        self.assertIsNone(_report_sommeil([], 23 * 60, 7 * 60, 480))

    def test_seuil_inclus(self):
        self.assertEqual(SEUIL_REPORT_LENDEMAIN, 720)
        self.assertEqual(
            _report_sommeil([(19 * 60, 4 * 60)], 23 * 60, 7 * 60, 480),
            {"start_min": 240, "end_min": 720},
        )
        self.assertIsNone(_report_sommeil([(19 * 60, 4 * 60 + 1)], 23 * 60, 7 * 60, 480))

    def test_plusieurs_murs_prend_la_fin_la_plus_tardive(self):
        self.assertEqual(
            _report_sommeil([(18 * 60, 1 * 60), (22 * 60, 3 * 60)], 23 * 60, 7 * 60, 480),
            {"start_min": 180, "end_min": 660},
        )

    def test_quart_finissant_a_minuit(self):
        self.assertEqual(
            _report_sommeil([(19 * 60, 0)], 23 * 60, 7 * 60, 480),
            {"start_min": 0, "end_min": 480},
        )

    def test_duree_absente_deduite_des_heures(self):
        self.assertEqual(
            _report_sommeil([(19 * 60, 2 * 60)], 23 * 60, 7 * 60, 0),
            {"start_min": 120, "end_min": 600},
        )


class SommeilQuartDeNuitTests(TestCase):
    def setUp(self):
        patcher = patch("django.utils.timezone.localtime", side_effect=_faux_localtime)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.user = User.objects.create_user("quart-nuit", password="pw")
        self.assertIsNotNone(self.user.profile)
        self.quart = RecurringBlock.objects.create(
            user=self.user, title="Quart au dépanneur", block_type="work",
            flexibility="fixed", day_of_week=3,
            start_time=time(19, 0), end_time=time(2, 0), is_night_shift=True,
        )
        self.sommeils = {
            d: RecurringBlock.objects.create(
                user=self.user, title="Sommeil", block_type="sleep",
                flexibility="flexible", day_of_week=d,
                start_time=time(23, 0), end_time=time(7, 0),
            )
            for d in range(7)
        }

    # Acceptation du contrat

    def test_jeudi_le_sommeil_est_reporte_au_lendemain(self):
        sommeil = _par_titre(place_day(self.user, JEUDI), "Sommeil")
        self.assertTrue(sommeil["skipped"])
        self.assertIsNone(sommeil["start_min"])
        self.assertIsNone(sommeil["end_min"])
        self.assertIsNone(sommeil["start_time"])
        self.assertFalse(sommeil["overnight_kept"])
        self.assertEqual(
            sommeil["reporte_au_lendemain"],
            {"start_time": "02:00", "end_time": "10:00"},
        )

    def test_jeudi_la_journee_est_libre(self):
        self.assertEqual(open_intervals(self.user, JEUDI, 420, 1380), [(420, 1140)])

    def test_find_free_slots_jeudi(self):
        res = execute_tool(
            "find_free_slots", self.user, {"date": "2026-09-17", "min_duration_minutes": 120}
        )
        self.assertTrue(res.success, res.message)
        self.assertIn(
            ("07:00", "19:00"),
            [(s["start_time"], s["end_time"]) for s in res.data["free_slots"]],
        )

    def test_schedule_task_at_jeudi_matin_passe(self):
        res = execute_tool("schedule_task_at", self.user, {
            "title": "Dentiste", "date": "2026-09-17",
            "start_time": "10:30", "end_time": "11:30",
        })
        self.assertTrue(res.success, res.message)

    def test_vendredi_le_sommeil_reporte_occupe_le_matin(self):
        self.assertEqual(intervalles_sommeil_reporte(self.user, VENDREDI), [(120, 600)])
        res = execute_tool(
            "find_free_slots", self.user, {"date": "2026-09-18", "min_duration_minutes": 30}
        )
        self.assertTrue(res.success, res.message)
        self.assertEqual(res.data["free_slots"][0]["start_time"], "10:00")

    def test_intervalles_non_recursifs(self):
        user = User.objects.get(pk=self.user.pk)  # profil non precharge
        faux_place_day = MagicMock()
        faux_occupied = MagicMock()
        with patch("services.scheduling.placement.place_day", faux_place_day), \
                patch("services.scheduling.placement.occupied_intervals", faux_occupied), \
                patch("services.scheduling.placement._placer_jour") as faux_placer, \
                patch("services.scheduling.solve_day.solve_placement") as faux_solve:
            with CaptureQueriesContext(connection) as requetes:
                resultat = placement.intervalles_sommeil_reporte(user, VENDREDI)
        self.assertEqual(resultat, [(120, 600)])
        faux_place_day.assert_not_called()
        faux_occupied.assert_not_called()
        faux_placer.assert_not_called()
        faux_solve.assert_not_called()
        self.assertLessEqual(len(requetes.captured_queries), 6)

    def test_vendredi_un_souple_evite_le_sommeil_reporte(self):
        RecurringBlock.objects.create(
            user=self.user, title="Sport", block_type="sport", flexibility="flexible",
            day_of_week=4, start_time=time(8, 0), end_time=time(9, 0),
        )
        sport = _par_titre(place_day(self.user, VENDREDI), "Sport")
        self.assertFalse(sport["skipped"])
        self.assertGreaterEqual(sport["start_min"], 600)

    def test_quart_jusqu_a_7h_garde_la_relocalisation(self):
        RecurringBlock.objects.create(
            user=self.user, title="Quart long", block_type="work", flexibility="fixed",
            day_of_week=0, start_time=time(19, 0), end_time=time(7, 0), is_night_shift=True,
        )
        sommeil = _par_titre(place_day(self.user, LUNDI), "Sommeil")
        self.assertFalse(sommeil["skipped"])
        self.assertIsNone(sommeil["reporte_au_lendemain"])
        self.assertLessEqual(sommeil["end_min"], 1140)
        self.assertEqual(sommeil["end_min"] - sommeil["start_min"], 480)
        # Pas de report: le mardi matin ne recoit rien.
        self.assertEqual(intervalles_sommeil_reporte(self.user, date(2026, 9, 15)), [])

    def test_sans_quart_rien_ne_change(self):
        sommeil = _par_titre(place_day(self.user, LUNDI), "Sommeil")
        self.assertTrue(sommeil["overnight_kept"])
        self.assertEqual(sommeil["start_min"], 1380)
        self.assertEqual(sommeil["end_min"], 420)
        self.assertIsNone(sommeil["reporte_au_lendemain"])

    def test_organize_day_jeudi_ne_met_pas_le_sommeil_le_jour(self):
        res = execute_tool("organize_day", self.user, {"date": "2026-09-17"})
        self.assertTrue(res.success, res.message)
        for entree in res.data["placed"]:
            if entree["title"] == "Sommeil":
                self.assertFalse("07:00" <= entree["start_time"] < "19:00", entree)

    # Cas limites

    def test_solve_placement_suit_la_meme_regle(self):
        jeudi = _par_titre(solve_placement(self.user, JEUDI), "Sommeil")
        self.assertTrue(jeudi["skipped"])
        self.assertEqual(
            jeudi["reporte_au_lendemain"], {"start_time": "02:00", "end_time": "10:00"}
        )
        lundi = _par_titre(solve_placement(self.user, LUNDI), "Sommeil")
        self.assertIsNone(lundi["reporte_au_lendemain"])
        self.assertTrue(lundi["overnight_kept"])

    def test_solve_placement_vendredi_souple_apres_le_sommeil_reporte(self):
        RecurringBlock.objects.create(
            user=self.user, title="Sport", block_type="sport", flexibility="flexible",
            day_of_week=4, start_time=time(8, 0), end_time=time(9, 0),
        )
        sport = _par_titre(solve_placement(self.user, VENDREDI), "Sport")
        self.assertFalse(sport["skipped"])
        self.assertGreaterEqual(sport["start_min"], 600)

    def test_faisabilite_vendredi_respecte_le_sommeil_reporte(self):
        res = solve_day(
            self.user, VENDREDI,
            extra=[{"title": "Révision", "duration_minutes": 120, "preferred_start": 7 * 60}],
            day_start=7 * 60, day_end=23 * 60,
        )
        self.assertTrue(res["feasible"])
        revision = next(p for p in res["placements"] if p["kind"] == "extra")
        self.assertGreaterEqual(revision["start_min"], 600)

    def test_faisabilite_jeudi_journee_libre(self):
        res = solve_day(
            self.user, JEUDI,
            extra=[{"title": "Étude", "duration_minutes": 11 * 60}],
            day_start=7 * 60, day_end=23 * 60,
        )
        self.assertTrue(res["feasible"])

    def test_occupied_intervals_vendredi_inclut_le_report(self):
        occupe = occupied_intervals(self.user, VENDREDI)
        self.assertEqual(occupe[0], (0, 600))

    def test_chaque_placement_porte_la_cle(self):
        for jour in (LUNDI, JEUDI, VENDREDI):
            for p in place_day(self.user, jour) + solve_placement(self.user, jour):
                self.assertIn("reporte_au_lendemain", p)

    def test_occurrence_du_quart_sautee_pas_de_report(self):
        RecurringBlockException.objects.create(
            user=self.user, recurring_block=self.quart, date=JEUDI
        )
        sommeil = _par_titre(place_day(self.user, JEUDI), "Sommeil")
        self.assertTrue(sommeil["overnight_kept"])
        self.assertIsNone(sommeil["reporte_au_lendemain"])
        self.assertEqual(intervalles_sommeil_reporte(self.user, VENDREDI), [])

    def test_occurrence_du_sommeil_sautee_pas_de_report(self):
        RecurringBlockException.objects.create(
            user=self.user, recurring_block=self.sommeils[3], date=JEUDI
        )
        self.assertEqual(intervalles_sommeil_reporte(self.user, VENDREDI), [])

    def test_quart_hors_bornes_pas_de_report_le_lendemain(self):
        self.quart.end_date = date(2026, 9, 16)
        self.quart.save(update_fields=["end_date"])
        self.assertEqual(intervalles_sommeil_reporte(self.user, VENDREDI), [])
        self.quart.end_date = None
        self.quart.start_date = date(2026, 9, 24)
        self.quart.save(update_fields=["end_date", "start_date"])
        self.assertEqual(intervalles_sommeil_reporte(self.user, VENDREDI), [])
        self.assertEqual(intervalles_sommeil_reporte(self.user, date(2026, 9, 25)), [(120, 600)])

    def test_trajet_retour_decale_le_report(self):
        lieu = UserPlace.objects.create(
            user=self.user, name="Dépanneur", kind="work", travel_minutes=30
        )
        self.quart.place = lieu
        self.quart.save(update_fields=["place"])
        sommeil = _par_titre(place_day(self.user, JEUDI), "Sommeil")
        self.assertEqual(
            sommeil["reporte_au_lendemain"], {"start_time": "02:30", "end_time": "10:30"}
        )
        self.assertEqual(intervalles_sommeil_reporte(self.user, VENDREDI), [(150, 630)])

    def test_dimanche_soir_deborde_sur_lundi(self):
        # day_of_week 6 = dimanche: le report tombe le lundi (rebouclage).
        RecurringBlock.objects.create(
            user=self.user, title="Quart du dimanche", block_type="work", flexibility="fixed",
            day_of_week=6, start_time=time(19, 0), end_time=time(2, 0), is_night_shift=True,
        )
        sommeil = _par_titre(place_day(self.user, DIMANCHE), "Sommeil")
        self.assertEqual(
            sommeil["reporte_au_lendemain"], {"start_time": "02:00", "end_time": "10:00"}
        )
        self.assertEqual(intervalles_sommeil_reporte(self.user, LUNDI_SUIVANT), [(120, 600)])

    def test_souple_overnight_autre_que_sommeil_non_reporte(self):
        RecurringBlock.objects.create(
            user=self.user, title="Veille", block_type="other", flexibility="flexible",
            day_of_week=3, start_time=time(22, 0), end_time=time(1, 0),
        )
        veille = _par_titre(place_day(self.user, JEUDI), "Veille")
        self.assertIsNone(veille["reporte_au_lendemain"])
        self.assertEqual(intervalles_sommeil_reporte(self.user, VENDREDI), [(120, 600)])

    def test_journee_du_jeudi_ne_montre_pas_le_sommeil(self):
        res = execute_tool("get_today_schedule", self.user, {"date": "2026-09-17"})
        self.assertTrue(res.success, res.message)
        self.assertNotIn("Sommeil", [b["title"] for b in res.data["blocks"]])
