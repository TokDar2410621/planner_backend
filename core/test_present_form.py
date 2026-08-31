"""present_form: contrat de donnees des champs rendus par InteractiveInputs.

Les six types historiques doivent sortir a l'identique (cle par cle, dans le
meme ordre) quand aucune nouveaute n'est utilisee; l'injection des raccourcis
jours de semaine et le passage des values d'options en chaines sont les deux
exceptions voulues. Tout ce qui est mal forme est jete avant d'atteindre le
frontend (qui n'a pas d'ErrorBoundary). execute() est appele en direct, sans
LLM ni base.
"""
import datetime
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase

from services.agent.tools.interactive import PresentFormTool

USER = SimpleNamespace(id=1)

JOURS_FR = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
JOURS_0_6 = [{"value": str(i), "label": nom} for i, nom in enumerate(JOURS_FR)]


def _champ(**extra):
    base = {"id": "champ", "label": "Champ", "question": "Question ?"}
    base.update(extra)
    return base


def _run(*inputs):
    return PresentFormTool().execute(USER, inputs=list(inputs))


def _seul(*inputs):
    result = _run(*inputs)
    assert result.success, result.message
    return result.data["interactive_inputs"][0]


class TypesHistoriquesTests(SimpleTestCase):
    def test_six_types_inchanges_cle_par_cle(self):
        activites = [{"value": "gym", "label": "Gym"}, {"value": "piscine", "label": "Piscine"}]
        result = _run(
            _champ(id="sommeil", type="time_range", default={"start": "23:00", "end": "07:00"}),
            _champ(id="reveil", type="time", default="07:00"),
            _champ(id="heures", type="number", default=8, min=4, max=12),
            _champ(id="sport", type="checkbox", options=activites),
            _champ(id="ville", type="select", options=activites),
            _champ(id="occupation", type="radio", options=activites, default="gym",
                   allow_other=False, other_placeholder="Autre chose"),
        )
        self.assertTrue(result.success)
        attendu = [
            {"id": "sommeil", "type": "time_range", "label": "Champ", "question": "Question ?",
             "default": {"start": "23:00", "end": "07:00"}},
            {"id": "reveil", "type": "time", "label": "Champ", "question": "Question ?",
             "default": "07:00"},
            {"id": "heures", "type": "number", "label": "Champ", "question": "Question ?",
             "default": 8, "min": 4, "max": 12},
            {"id": "sport", "type": "checkbox", "label": "Champ", "question": "Question ?",
             "options": activites},
            {"id": "ville", "type": "select", "label": "Champ", "question": "Question ?",
             "options": activites},
            {"id": "occupation", "type": "radio", "label": "Champ", "question": "Question ?",
             "options": activites, "default": "gym", "allowOther": False,
             "otherPlaceholder": "Autre chose"},
        ]
        champs = result.data["interactive_inputs"]
        self.assertEqual(champs, attendu)
        self.assertEqual([list(c) for c in champs], [list(a) for a in attendu])
        self.assertEqual(
            result.message,
            "Formulaire avec 6 champ(s) présenté à l'utilisateur. Attends sa réponse.",
        )

    def test_default_checkbox_relaye(self):
        champ = _seul(_champ(type="checkbox", options=JOURS_0_6, default=["0", "1", "2", "3", "4"]))
        self.assertEqual(champ["default"], ["0", "1", "2", "3", "4"])

    def test_presets_time_range_relayes(self):
        presets = [
            {"label": "22h-6h", "start": "22:00", "end": "06:00"},
            {"label": "23h-7h", "start": "23:00", "end": "07:00"},
        ]
        champ = _seul(_champ(type="time_range", default={"start": "23:00", "end": "07:00"},
                             presets=presets))
        self.assertEqual(champ["presets"], presets)
        self.assertEqual(list(champ), ["id", "type", "label", "question", "default", "presets"])

    def test_time_range_sans_presets_n_en_recoit_pas(self):
        champ = _seul(_champ(type="time_range"))
        self.assertNotIn("presets", champ)

    def test_number_ignore_presets(self):
        champ = _seul(_champ(type="number", presets=[1, 2]))
        self.assertNotIn("presets", champ)


class AutoPresetsJoursTests(SimpleTestCase):
    def test_labels_francais_avec_values_libres(self):
        options = [{"value": nom.lower()[:3], "label": nom} for nom in JOURS_FR]
        champ = _seul(_champ(type="checkbox", options=options))
        self.assertEqual(champ["presets"], [
            {"label": "Lun-ven", "values": ["lun", "mar", "mer", "jeu", "ven"]},
            {"label": "Tous", "values": ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"]},
        ])

    def test_labels_sans_casse_ni_accents_ni_espaces(self):
        labels = ["LUNDI", "mardi ", " Mercredi", "JeUdI", "vendredí", "Samedi", "DIMANCHE"]
        options = [{"value": f"j{i}", "label": lab} for i, lab in enumerate(labels)]
        champ = _seul(_champ(type="checkbox", options=options))
        self.assertEqual(champ["presets"][0], {"label": "Lun-ven", "values": ["j0", "j1", "j2", "j3", "j4"]})
        self.assertEqual(champ["presets"][1]["values"], [f"j{i}" for i in range(7)])

    def test_values_0_a_6_avec_labels_courts(self):
        options = [{"value": str(i), "label": lettre} for i, lettre in enumerate("LMMJVSD")]
        champ = _seul(_champ(type="checkbox", options=options))
        self.assertEqual(champ["presets"], [
            {"label": "Lun-ven", "values": ["0", "1", "2", "3", "4"]},
            {"label": "Tous", "values": ["0", "1", "2", "3", "4", "5", "6"]},
        ])

    def test_values_entieres_castees_en_chaines(self):
        options = [{"value": i, "label": nom} for i, nom in enumerate(JOURS_FR)]
        champ = _seul(_champ(type="checkbox", options=options))
        self.assertEqual(champ["presets"][0]["values"], ["0", "1", "2", "3", "4"])

    def test_ordre_des_options_respecte(self):
        options = [JOURS_0_6[6]] + JOURS_0_6[:6]
        champ = _seul(_champ(type="checkbox", options=options))
        self.assertEqual(champ["presets"][0]["values"], ["0", "1", "2", "3", "4"])
        self.assertEqual(champ["presets"][1]["values"], ["6", "0", "1", "2", "3", "4", "5"])

    def test_lun_ven_seulement_ne_duplique_pas_tous(self):
        champ = _seul(_champ(type="checkbox", options=JOURS_0_6[:5]))
        self.assertEqual(champ["presets"], [{"label": "Tous", "values": ["0", "1", "2", "3", "4"]}])

    def test_pas_d_injection_si_presets_fournis(self):
        presets = [{"label": "Semaine", "values": ["0", "1", "2", "3", "4"]}]
        champ = _seul(_champ(type="checkbox", options=JOURS_0_6, presets=presets))
        self.assertEqual(champ["presets"], presets)

    def test_pas_d_injection_si_pas_des_jours(self):
        options = [{"value": v, "label": v.title()} for v in ["gym", "piscine", "velo", "course", "yoga"]]
        champ = _seul(_champ(type="checkbox", options=options))
        self.assertNotIn("presets", champ)

    def test_pas_d_injection_si_un_intrus(self):
        options = JOURS_0_6[:6] + [{"value": "jamais", "label": "Jamais"}]
        champ = _seul(_champ(type="checkbox", options=options))
        self.assertNotIn("presets", champ)

    def test_pas_d_injection_sous_cinq_jours(self):
        champ = _seul(_champ(type="checkbox", options=JOURS_0_6[5:]))
        self.assertNotIn("presets", champ)

    def test_pas_d_injection_pour_select_ni_radio(self):
        result = _run(_champ(id="a", type="select", options=JOURS_0_6),
                      _champ(id="b", type="radio", options=JOURS_0_6))
        self.assertTrue(result.success)
        for champ in result.data["interactive_inputs"]:
            self.assertNotIn("presets", champ)


class DurationTests(SimpleTestCase):
    def test_presets_par_defaut_sans_options(self):
        champ = _seul(_champ(type="duration"))
        self.assertEqual(champ, {"id": "champ", "type": "duration", "label": "Champ",
                                 "question": "Question ?", "presets": [30, 60, 90, 120]})

    def test_presets_par_defaut_sont_une_copie(self):
        a = _seul(_champ(type="duration"))
        a["presets"].append(999)
        self.assertEqual(_seul(_champ(type="duration"))["presets"], [30, 60, 90, 120])

    def test_default_presets_et_bornes_relayes(self):
        champ = _seul(_champ(type="duration", default=90, presets=[45, 90, 180], min=15, max=240))
        self.assertEqual(champ["default"], 90)
        self.assertEqual(champ["presets"], [45, 90, 180])
        self.assertEqual((champ["min"], champ["max"]), (15, 240))
        self.assertEqual(list(champ), ["id", "type", "label", "question", "default", "presets", "min", "max"])

    def test_presets_vides_ou_invalides_retombent_sur_le_defaut(self):
        self.assertEqual(_seul(_champ(type="duration", presets=[]))["presets"], [30, 60, 90, 120])
        self.assertEqual(_seul(_champ(type="duration", presets="60"))["presets"], [30, 60, 90, 120])


class DateTests(SimpleTestCase):
    def _presets(self, aujourdhui):
        with patch("services.agent.tools.interactive.timezone.localdate", return_value=aujourdhui):
            return _seul(_champ(type="date"))["presets"]

    def test_presets_calcules_un_dimanche(self):
        self.assertEqual(self._presets(datetime.date(2026, 8, 30)), [
            {"label": "Aujourd'hui", "value": "2026-08-30"},
            {"label": "Demain", "value": "2026-08-31"},
            {"label": "Samedi 5 sept.", "value": "2026-09-05"},
            {"label": "Dimanche 6 sept.", "value": "2026-09-06"},
        ])

    def test_prochain_samedi_strictement_a_venir(self):
        presets = self._presets(datetime.date(2026, 9, 5))
        self.assertEqual(presets[0]["value"], "2026-09-05")
        self.assertEqual(presets[1], {"label": "Demain", "value": "2026-09-06"})
        self.assertEqual(presets[2], {"label": "Samedi 12 sept.", "value": "2026-09-12"})
        self.assertEqual(len(presets), 3)

    def test_demain_samedi_non_repete(self):
        presets = self._presets(datetime.date(2026, 9, 4))
        self.assertEqual([p["value"] for p in presets], ["2026-09-04", "2026-09-05", "2026-09-06"])
        self.assertEqual(presets[2]["label"], "Dimanche 6 sept.")

    def test_mois_courts_accentues(self):
        presets = self._presets(datetime.date(2026, 7, 29))
        self.assertEqual(presets[2]["label"], "Samedi 1 août")
        presets = self._presets(datetime.date(2027, 2, 1))
        self.assertEqual(presets[2]["label"], "Samedi 6 févr.")
        presets = self._presets(datetime.date(2026, 12, 28))
        self.assertEqual(presets[3]["label"], "Dimanche 3 janv.")

    def test_default_et_presets_fournis_relayes(self):
        presets = [{"label": "Lundi prochain", "value": "2026-09-07"}]
        champ = _seul(_champ(type="date", default="2026-09-07", presets=presets))
        self.assertEqual(champ["default"], "2026-09-07")
        self.assertEqual(champ["presets"], presets)
        self.assertEqual(list(champ), ["id", "type", "label", "question", "default", "presets"])

    def test_sans_options_ni_bornes(self):
        champ = _seul(_champ(type="date", min=1, max=2))
        self.assertNotIn("options", champ)
        self.assertNotIn("min", champ)


class ValidationTests(SimpleTestCase):
    def test_type_inconnu_refuse(self):
        result = _run(_champ(id="x", type="text"))
        self.assertFalse(result.success)
        self.assertIn("'text'", result.message)
        self.assertIn("'x'", result.message)
        for t in ("time_range", "time", "number", "checkbox", "select", "radio", "duration", "date"):
            self.assertIn(t, result.message)

    def test_type_absent_refuse(self):
        result = _run({"id": "x", "label": "L", "question": "Q"})
        self.assertFalse(result.success)

    def test_options_manquantes_refusees(self):
        for t in ("checkbox", "select", "radio"):
            with self.subTest(type=t):
                result = _run(_champ(id="x", type=t))
                self.assertFalse(result.success)
                self.assertEqual(result.message, f"Le champ 'x' de type {t} nécessite des options.")
                self.assertFalse(_run(_champ(id="x", type=t, options=[])).success)

    def test_aucun_champ_refuse(self):
        result = PresentFormTool().execute(USER, inputs=[])
        self.assertFalse(result.success)
        self.assertEqual(result.message, "Aucun champ spécifié pour le formulaire.")

    def test_schema_documente_les_nouveaux_types(self):
        item = PresentFormTool.parameters["properties"]["inputs"]["items"]
        self.assertEqual(item["properties"]["type"]["enum"],
                         ["time_range", "time", "number", "checkbox", "select", "radio", "duration", "date"])
        self.assertIn("presets", item["properties"])
        self.assertIn("duration", item["properties"]["min"]["description"])


class PresetsInvalidesTests(SimpleTestCase):
    """Un preset mal forme est jete, jamais relaye; s'il ne reste rien, retour
    aux defauts (ou omission pour time_range)."""

    ACTIVITES = [{"value": v, "label": v.title()} for v in ["gym", "piscine", "velo"]]

    def test_checkbox_value_au_lieu_de_values_retombe_sur_les_jours(self):
        presets = [{"label": "Semaine", "value": ["0", "1", "2", "3", "4"]}]
        champ = _seul(_champ(type="checkbox", options=JOURS_0_6, presets=presets))
        self.assertEqual(champ["presets"], [
            {"label": "Lun-ven", "values": ["0", "1", "2", "3", "4"]},
            {"label": "Tous", "values": ["0", "1", "2", "3", "4", "5", "6"]},
        ])

    def test_checkbox_liste_de_chaines_jetee(self):
        champ = _seul(_champ(type="checkbox", options=self.ACTIVITES, presets=["gym", "velo"]))
        self.assertNotIn("presets", champ)

    def test_checkbox_values_filtrees_aux_options(self):
        presets = [
            {"label": "Cardio", "values": ["gym", "inconnu", "velo", "gym"]},
            {"label": "Rien", "values": ["inconnu"]},
            {"label": "", "values": ["gym"]},
            {"label": "Sans values"},
            {"label": "Chaine", "values": "gym"},
        ]
        champ = _seul(_champ(type="checkbox", options=self.ACTIVITES, presets=presets))
        self.assertEqual(champ["presets"], [{"label": "Cardio", "values": ["gym", "velo"]}])

    def test_time_range_invalides_omis(self):
        presets = [
            {"label": "Nuit", "start": "22h", "end": "06:00"},
            {"label": "Matin", "value": "07:00"},
            "23:00-07:00",
        ]
        self.assertNotIn("presets", _seul(_champ(type="time_range", presets=presets)))

    def test_time_range_garde_les_valides_seulement(self):
        presets = [
            {"label": "Nuit", "start": "22:00", "end": "06:00"},
            {"label": "Faux", "start": "25:00", "end": "06:00"},
        ]
        champ = _seul(_champ(type="time_range", presets=presets))
        self.assertEqual(champ["presets"], [{"label": "Nuit", "start": "22:00", "end": "06:00"}])

    def test_date_invalides_retombent_sur_le_calcul(self):
        presets = [{"label": "Lundi", "value": "lundi prochain"}, {"label": "Mardi", "value": "2026-13-01"}]
        with patch("services.agent.tools.interactive.timezone.localdate",
                   return_value=datetime.date(2026, 8, 30)):
            champ = _seul(_champ(type="date", presets=presets))
        self.assertEqual(champ["presets"][0], {"label": "Aujourd'hui", "value": "2026-08-30"})
        self.assertEqual(len(champ["presets"]), 4)

    def test_date_garde_les_valides_seulement(self):
        presets = [{"label": "Lundi", "value": "2026-09-07"}, {"label": "Faux", "value": "bientot"}]
        champ = _seul(_champ(type="date", presets=presets))
        self.assertEqual(champ["presets"], [{"label": "Lundi", "value": "2026-09-07"}])

    def test_duration_invalides_retombent_sur_le_defaut(self):
        champ = _seul(_champ(type="duration", presets=["60", 1.5, 0, -30, True, None]))
        self.assertEqual(champ["presets"], [30, 60, 90, 120])


class ValuesEnChainesTests(SimpleTestCase):
    """Le frontend compare des chaines: tout ce qui designe une option passe
    par str(), sinon rien n'est pre-coche."""

    def test_options_et_default_entiers_castes(self):
        options = [{"value": i, "label": nom} for i, nom in enumerate(JOURS_FR)]
        champ = _seul(_champ(type="checkbox", options=options, default=[0, 1, 2, 3, 4]))
        self.assertEqual([o["value"] for o in champ["options"]], [str(i) for i in range(7)])
        self.assertEqual(champ["default"], ["0", "1", "2", "3", "4"])
        self.assertEqual(champ["presets"][0]["values"], ["0", "1", "2", "3", "4"])

    def test_presets_values_entieres_castees(self):
        presets = [{"label": "Semaine", "values": [0, 1, 2, 3, 4]}]
        champ = _seul(_champ(type="checkbox", options=JOURS_0_6, presets=presets))
        self.assertEqual(champ["presets"], [{"label": "Semaine", "values": ["0", "1", "2", "3", "4"]}])

    def test_default_checkbox_non_liste_ignore(self):
        self.assertNotIn("default", _seul(_champ(type="checkbox", options=JOURS_0_6, default="0")))
        self.assertNotIn("default", _seul(_champ(type="checkbox", options=JOURS_0_6, default=0)))

    def test_default_checkbox_filtre_aux_values_presentes(self):
        champ = _seul(_champ(type="checkbox", options=JOURS_0_6, default=["0", "9", "4", "0"]))
        self.assertEqual(champ["default"], ["0", "4"])
        self.assertNotIn("default", _seul(_champ(type="checkbox", options=JOURS_0_6, default=["9"])))

    def test_default_select_et_radio_castes_et_filtres(self):
        options = [{"value": 1, "label": "Un"}, {"value": 2, "label": "Deux"}]
        for t in ("select", "radio"):
            with self.subTest(type=t):
                champ = _seul(_champ(type=t, options=options, default=2))
                self.assertEqual(champ["options"], [{"value": "1", "label": "Un"}, {"value": "2", "label": "Deux"}])
                self.assertEqual(champ["default"], "2")
                self.assertNotIn("default", _seul(_champ(type=t, options=options, default="trois")))

    def test_option_sans_value_ecartee(self):
        options = [{"label": "Sans value"}, {"value": "ok", "label": "Ok"}, "gym"]
        champ = _seul(_champ(type="radio", options=options))
        self.assertEqual(champ["options"], [{"value": "ok", "label": "Ok"}])
        self.assertFalse(_run(_champ(id="x", type="radio", options=[{"label": "Rien"}])).success)


class DurationEntiersTests(SimpleTestCase):
    """Tout en minutes entieres: une valeur decimale (heures) ne devient pas
    une pastille."""

    def test_default_decimal_ecarte(self):
        for mauvais in (1.5, "90", 0, -15, True):
            with self.subTest(default=mauvais):
                self.assertNotIn("default", _seul(_champ(type="duration", default=mauvais)))

    def test_default_entier_flottant_accepte(self):
        champ = _seul(_champ(type="duration", default=90.0))
        self.assertEqual(champ["default"], 90)
        self.assertIsInstance(champ["default"], int)

    def test_presets_tries_dedupliques(self):
        # 60.0 vaut 60; une liste avec un decimal ou une chaine est jugee non
        # fiable en bloc (voir EntreesHostilesResiduellesTests).
        champ = _seul(_champ(type="duration", presets=[90, 30, 60.0, 60]))
        self.assertEqual(champ["presets"], [30, 60, 90])
        self.assertTrue(all(isinstance(m, int) for m in champ["presets"]))

    def test_bornes_castees_en_int(self):
        champ = _seul(_champ(type="duration", min=15.0, max=240.7))
        self.assertEqual((champ["min"], champ["max"]), (15, 240))
        self.assertNotIn("min", _seul(_champ(type="duration", min="15")))

    def test_number_garde_ses_bornes_decimales(self):
        champ = _seul(_champ(type="number", default=7.5, min=0.5, max=12))
        self.assertEqual((champ["default"], champ["min"], champ["max"]), (7.5, 0.5, 12))
        self.assertNotIn("default", _seul(_champ(type="number", default="huit")))

    def test_schema_precise_entiers_en_minutes(self):
        props = PresentFormTool.parameters["properties"]["inputs"]["items"]["properties"]
        for cle in ("type", "default", "presets", "min", "max"):
            with self.subTest(cle=cle):
                self.assertIn("entier", props[cle]["description"])
                self.assertIn("minutes", props[cle]["description"])


class ChampIncompletTests(SimpleTestCase):
    """Un champ sans id/label/question est refuse proprement, pas un KeyError."""

    def test_sans_label_refuse(self):
        result = _run({"id": "x", "type": "time", "question": "Q"})
        self.assertFalse(result.success)
        self.assertEqual(result.message, "Le champ 'x' doit avoir id, label et question.")

    def test_sans_id_refuse(self):
        result = _run({"type": "time", "label": "L", "question": "Q"})
        self.assertFalse(result.success)
        self.assertEqual(result.message, "Le champ '?' doit avoir id, label et question.")

    def test_question_vide_refusee(self):
        result = _run(_champ(id="y", type="date", question=""))
        self.assertFalse(result.success)
        self.assertEqual(result.message, "Le champ 'y' doit avoir id, label et question.")

    def test_champ_non_objet_refuse(self):
        result = _run("time")
        self.assertFalse(result.success)
        self.assertIn("'?'", result.message)


class FormatsHeuresEtDatesTests(SimpleTestCase):
    """Heures HH:MM zero-paddees, dates ISO valides; sinon omis."""

    def test_time_range_default_zero_padde(self):
        champ = _seul(_champ(type="time_range", default={"start": "9:00", "end": "17:00"}))
        self.assertEqual(champ["default"], {"start": "09:00", "end": "17:00"})

    def test_time_range_default_invalide_omis(self):
        mauvais = (
            {"start": "9h", "end": "17:00"},
            {"start": "09:00"},
            "09:00-17:00",
            {"start": "24:00", "end": "07:00"},
            {"start": "09:00", "end": "17:5"},
        )
        for d in mauvais:
            with self.subTest(default=d):
                self.assertNotIn("default", _seul(_champ(type="time_range", default=d)))

    def test_time_range_presets_zero_paddes(self):
        presets = [{"label": "Matin", "start": "9:00", "end": "12:00"}]
        champ = _seul(_champ(type="time_range", presets=presets))
        self.assertEqual(champ["presets"], [{"label": "Matin", "start": "09:00", "end": "12:00"}])

    def test_time_default_zero_padde_ou_omis(self):
        self.assertEqual(_seul(_champ(type="time", default="9:00"))["default"], "09:00")
        self.assertEqual(_seul(_champ(type="time", default=" 18:30 "))["default"], "18:30")
        for mauvais in ("neuf heures", "9h", 900, "09:60"):
            with self.subTest(default=mauvais):
                self.assertNotIn("default", _seul(_champ(type="time", default=mauvais)))

    def test_date_default_iso_ou_omis(self):
        self.assertEqual(_seul(_champ(type="date", default="2026-09-07"))["default"], "2026-09-07")
        for mauvais in ("2026-9-7", "07/09/2026", "demain", 20260907, "2026-02-30"):
            with self.subTest(default=mauvais):
                self.assertNotIn("default", _seul(_champ(type="date", default=mauvais)))


class AbreviationsJoursTests(SimpleTestCase):
    """Les jours abreges a trois lettres sont reconnus, avec ou sans point."""

    def test_abreviations_avec_et_sans_point(self):
        labels = ["Lun", "Mar.", "mer", "JEU.", "Ven", "sam.", "Dim"]
        options = [{"value": f"j{i}", "label": lab} for i, lab in enumerate(labels)]
        champ = _seul(_champ(type="checkbox", options=options))
        self.assertEqual(champ["presets"], [
            {"label": "Lun-ven", "values": ["j0", "j1", "j2", "j3", "j4"]},
            {"label": "Tous", "values": [f"j{i}" for i in range(7)]},
        ])

    def test_abreviations_semaine_seule(self):
        options = [{"value": lab.lower(), "label": lab} for lab in ["Lun", "Mar", "Mer", "Jeu", "Ven"]]
        champ = _seul(_champ(type="checkbox", options=options))
        self.assertEqual(champ["presets"], [{"label": "Tous", "values": ["lun", "mar", "mer", "jeu", "ven"]}])

    def test_abreviation_inconnue_bloque_l_injection(self):
        labels = ["Lun", "Mar", "Mer", "Jeu", "Vend"]
        options = [{"value": f"j{i}", "label": lab} for i, lab in enumerate(labels)]
        self.assertNotIn("presets", _seul(_champ(type="checkbox", options=options)))


class EntreesHostilesResiduellesTests(SimpleTestCase):
    """Les six restes de la verification adverse du 2026-08-30."""

    def test_un_nombre_non_fini_est_omis(self):
        # json.loads accepte NaN; json.dumps le reemettrait et la trame SSE
        # deviendrait illisible pour le client.
        champ = _seul(_champ(type="number", default=float("nan"),
                             min=float("-inf"), max=float("inf")))
        self.assertNotIn("default", champ)
        self.assertNotIn("min", champ)
        self.assertNotIn("max", champ)

    def test_une_option_a_value_vide_est_ecartee(self):
        champ = _seul(_champ(type="select", options=[
            {"value": "", "label": "Vide"}, {"value": "a", "label": "A"}]))
        self.assertEqual([o["value"] for o in champ["options"]], ["a"])

    def test_un_id_ou_label_non_chaine_est_refuse(self):
        for cle, valeur in (("id", 12), ("label", 5), ("question", "  ")):
            with self.subTest(cle=cle):
                result = _run(_champ(type="time", **{cle: valeur}))
                self.assertFalse(result.success)
                self.assertIn("doit avoir id, label et question", result.message)

    def test_des_presets_duration_partiellement_invalides_repartent_des_defauts(self):
        champ = _seul(_champ(type="duration", presets=[0.5, 1, 1.5]))
        self.assertEqual(champ["presets"], [30, 60, 90, 120])

    def test_des_presets_duration_tous_valides_sont_gardes(self):
        champ = _seul(_champ(type="duration", presets=[45, 15, 45]))
        self.assertEqual(champ["presets"], [15, 45])
