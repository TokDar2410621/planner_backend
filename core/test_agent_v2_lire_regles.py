"""Règles unitaires de lecture typée : formulaire de cours et créneaux."""
import re
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase, override_settings
from django.utils import timezone

from core.test_agent_v2_lire_ombre import _element, _heure, _ref
from services.agent.tools.base import ToolResult
from services.agent_v2 import boutons, lecture, regles
from services.agent_v2 import lecture_schema as schema
from services.agent_v2.registre import Registre

JOURNAL = "services.agent_v2.lecture"
JOURS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
QUESTION_MATHS = "Quels jours et à quelles heures as-tu « mon cours de maths » ?"
MOTS_D_INTERFACE = re.compile(r"\b(bouton|formulaire|champ|coche|bloc)", re.IGNORECASE)
OCCUPE = [(18 * 60, 22 * 60)]
LIBRE = [(0, 24 * 60)]


def _ldt(*elements, refs=None, aujourdhui=None, origine="tape"):
    lu = schema.LectureTour.model_validate({"elements": list(elements), "reponses": []})
    return regles.LectureDuTour(lecture=lu, refs=refs or {"s": {}, "t": {}, "q": {}},
                                aujourdhui=aujourdhui or timezone.localdate(), origine=origine)


def _groupe(titre):
    return {"titre": titre, "jours": [0], "debut": "13:00", "fin": "15:50", "type": "course"}


def _registre(outil, succes=True, donnees=None):
    registre = Registre()
    registre.ajouter(outil, {}, ToolResult(success=succes, data=donnees or {}, message="ok"))
    return registre


REFS = {"s": {"s1": _groupe("Calcul différentiel"), "s2": _groupe("Calcul différentiel"),
              "s3": _groupe("Anglais")},
        "t": {"t1": "Remettre le labo"}, "q": {}}
COURS_MATHS = _element(mention="mon cours de maths", genre="course", candidats=["s1", "s2", "s3", "t1"])


def _revision(reference, *heures):
    return _element(mention="revision", genre="revision", dates=[reference],
                    heures=list(heures) or [_heure("14h", "14:00"), _heure("16h", "16:00", role="fin")])


DEMIN = _ref("demin", "jour_relatif", jour_relatif="demain")


# ------------------------------------------------------------------ reglages

class ReglagesTests(SimpleTestCase):
    def test_les_deux_regles_par_defaut_et_la_liste_vide_garde_l_ombre(self):
        with patch.object(lecture, "settings", SimpleNamespace()):
            self.assertEqual(lecture.regles_actives(), frozenset({"formulaire_cours", "creneaux"}))
        for valeur, attendu in (("", frozenset()), (" creneaux , ", frozenset({"creneaux"})),
                                ("formulaire_cours,creneaux", frozenset({"formulaire_cours", "creneaux"}))):
            with override_settings(LIRE_REGLES=valeur):
                self.assertEqual(lecture.regles_actives(), attendu, valeur)
        self.assertEqual((regles.FORMULAIRE_COURS, regles.CRENEAUX), ("formulaire_cours", "creneaux"))

    def test_la_ligne_du_tour_nomme_la_regle(self):
        suivi = lecture.sautee()
        with self.assertLogs(JOURNAL, "INFO") as journal:
            meta = lecture.clore(suivi, lecture.recueillir(suivi), regles.CRENEAUX)
        self.assertTrue(journal.records[-1].getMessage().endswith(" regle=creneaux"))
        self.assertEqual(meta["lecture_statut"], "sautee")
        with self.assertLogs(JOURNAL, "INFO") as journal:
            lecture.finir(lecture.sautee())
        self.assertTrue(journal.records[-1].getMessage().endswith(" regle=-"))

    def test_seule_une_lecture_ok_ou_partielle_sert(self):
        prep = lecture.Preparation(message="m", aujourdhui=date(2026, 9, 14), contexte="", refs=REFS,
                                   modeles=(), origine="formulaire")
        lu = schema.LectureTour.model_validate({"elements": [COURS_MATHS], "reponses": []})
        for statut in ("ok", "partielle", "absente", "hors_budget", "erreur", "desactivee", "sautee"):
            avec = lu if statut in ("ok", "partielle") else None
            ldt = regles.lecture_du_tour(lecture.Suivi(preparation=prep), lecture.Resultat(statut, lecture=avec))
            self.assertEqual(ldt is not None, statut in ("ok", "partielle"), statut)
        ldt = regles.lecture_du_tour(lecture.Suivi(preparation=prep), lecture.Resultat("ok", lecture=lu))
        self.assertEqual((ldt.refs, ldt.aujourdhui, ldt.origine), (REFS, date(2026, 9, 14), "formulaire"))
        self.assertIsNone(regles.lecture_du_tour(lecture.Suivi(), lecture.Resultat("ok", lecture=lu)))


# ------------------------------------------------- regle 1: formulaire du code

class FormulaireCoursTests(SimpleTestCase):
    def _appliquer(self, *elements, registre=None, refs=REFS, origine="tape", attachment=None, reemises=()):
        registre = Registre() if registre is None else registre
        avant = len(registre.actions)
        prose = regles.appliquer_formulaire_cours(_ldt(*elements, refs=refs, origine=origine), registre,
                                                  attachment=attachment, reemises=reemises)
        return prose, registre.actions[avant:]

    def test_un_cours_sans_jours_ni_heures_recoit_le_formulaire_du_code(self):
        prose, ajoutees = self._appliquer(COURS_MATHS)
        self.assertEqual(prose, f"Déjà à ton horaire : Calcul différentiel, Anglais\n{QUESTION_MATHS}")
        self.assertIsNone(MOTS_D_INTERFACE.search(prose))
        self.assertEqual(len(ajoutees), 1)
        action = ajoutees[0]
        self.assertEqual((action.outil, action.succes, action.donnees.get("par_le_code")),
                         ("present_form", True, True))
        jours, heures = action.donnees["interactive_inputs"]
        self.assertEqual(jours["type"], "checkbox")
        self.assertEqual([(o["value"], o["label"]) for o in jours["options"]],
                         [(str(i), nom) for i, nom in enumerate(JOURS)])
        self.assertNotIn("default", jours)
        self.assertEqual(heures["type"], "time_range")
        self.assertNotIn("default", heures)
        self.assertNotIn("presets", heures)
        self.assertIn("« mon cours de maths »", jours["label"])
        self.assertIn("« mon cours de maths »", heures["question"])
        for champ in (jours, heures):
            self.assertIsNone(MOTS_D_INTERFACE.search(f"{champ['label']} {champ['question']}"))
        # La reponse exacte du frontend garde les deux heures pour les lecteurs
        # geles: « Heures de ... » passait pour une duree et la garde imposait
        # 17:50 au lieu de 16:00 (banc reel du 2026-09-15).
        from services.agent_v2 import demandes as dem
        reponse = f"Voici mes réponses :\n{jours['label']}: Mardi, Jeudi\n{heures['label']}: 16:00 - 17:50"
        self.assertEqual(dem.heures_dites(reponse), ["16:00", "17:50"])

    def test_sans_candidat_de_la_semaine_type_une_seule_ligne(self):
        prose, _ = self._appliquer(_element(mention="mon cours de maths", genre="course", candidats=["t1"]))
        self.assertEqual(prose, QUESTION_MATHS)

    def test_au_plus_six_titres_distincts(self):
        refs = {"s": {f"s{i}": _groupe(f"Cours {i}") for i in range(1, 9)}, "t": {}, "q": {}}
        element = _element(mention="mon cours de maths", genre="course",
                           candidats=[f"s{i}" for i in range(1, 9)])
        prose, _ = self._appliquer(element, refs=refs)
        self.assertEqual(prose.splitlines()[0],
                         "Déjà à ton horaire : Cours 1, Cours 2, Cours 3, Cours 4, Cours 5, Cours 6")

    def test_un_quart_de_travail_aussi(self):
        prose, ajoutees = self._appliquer(_element(mention="mon quart au dépanneur", genre="work"))
        self.assertEqual(prose, "Quels jours et à quelles heures as-tu « mon quart au dépanneur » ?")
        self.assertEqual(len(ajoutees), 1)

    def test_ne_se_declenche_pas(self):
        cas = {
            "heures dites": dict(elements=[{**COURS_MATHS, "heures": [_heure("13h", "13:00")]}]),
            "jours dits": dict(elements=[{**COURS_MATHS, "jours_semaine": [{"extrait": "lundi", "jour": "lundi"}]}]),
            "date dite": dict(elements=[{**COURS_MATHS,
                                         "dates": [_ref("lundi", "jour_semaine", jour_semaine="lundi")]}]),
            "deplacement": dict(elements=[{**COURS_MATHS, "operation": "deplacer"}]),
            "suppression": dict(elements=[{**COURS_MATHS, "operation": "supprimer"}]),
            "saut d'une fois": dict(elements=[{**COURS_MATHS, "operation": "sauter_une_fois"}]),
            "modification": dict(elements=[{**COURS_MATHS, "operation": "modifier"}]),
            "question de lecture": dict(elements=[{**COURS_MATHS, "polarite": "question", "operation": "consulter"}]),
            "refus": dict(elements=[{**COURS_MATHS, "polarite": "refus"}]),
            "seance de revision": dict(elements=[{**COURS_MATHS, "genre": "revision"}]),
            "deux demandes": dict(elements=[COURS_MATHS, {**COURS_MATHS, "mention": "mon labo"}]),
            "nom non ancre": dict(elements=[{**COURS_MATHS, "mention": ""}]),
            "reponse de formulaire": dict(origine="formulaire"),
            "choix deja pose": dict(registre=_registre("present_choices", True, {"demande": None})),
            "formulaire du modele": dict(registre=_registre("present_form", True, {"interactive_inputs": []})),
            "creation reussie": dict(registre=_registre("create_block", True, {"created": []})),
            "demande en attente": dict(registre=_registre("delete_block", False,
                                                          {"demande": {"cle": "k", "motif": "destructif"}})),
            "demande reemise": dict(reemises=[{"cle": "k", "motif": "destructif"}]),
            "piece jointe": dict(attachment=object()),
        }
        for nom, options in cas.items():
            with self.subTest(cas=nom):
                prose, ajoutees = self._appliquer(*options.pop("elements", [COURS_MATHS]), **options)
                self.assertEqual((prose, ajoutees), ("", []))

    def test_sans_lecture_rien(self):
        registre = Registre()
        self.assertEqual(regles.appliquer_formulaire_cours(None, registre, attachment=None, reemises=()), "")
        self.assertEqual(registre.actions, [])

    def test_un_message_en_plusieurs_parties_ou_une_consultation_garde_main(self):
        horaire = {**COURS_MATHS, "mention": "mon horaire", "polarite": "question",
                   "operation": "consulter"}
        prose, ajoutees = self._appliquer(COURS_MATHS, horaire)
        self.assertEqual((prose, ajoutees), ("", []))
        for lecture_affichee in ("get_today_schedule", "get_week_schedule", "find_free_slots"):
            prose, ajoutees = self._appliquer(COURS_MATHS, registre=_registre(lecture_affichee))
            self.assertEqual((prose, ajoutees), ("", []), lecture_affichee)
        prose, ajoutees = self._appliquer(COURS_MATHS, registre=_registre("list_blocks"))
        self.assertTrue(prose)


# ---------------------------------------------------- regle 2: creneaux types

class CreneauxTypesTests(TestCase):
    """Chaque cas type se compare a ce que main produit pour le message bien ecrit."""

    def setUp(self):
        self.user = User.objects.create_user(username="creneaux", password="x")

    def _types(self, element, libre=OCCUPE, aujourdhui=None, registre=None, attachment=None, ldt=True):
        lu = _ldt(element, aujourdhui=aujourdhui) if ldt else None
        with patch("services.scheduling.placement.open_intervals", return_value=libre):
            return boutons.creneaux_types(self.user, "planifie", attachment, registre or Registre(), lu)

    def _main(self, message, libre=OCCUPE):
        with patch("services.scheduling.placement.open_intervals", return_value=libre):
            return boutons.question_forcee(self.user, message, None, Registre(), False)

    def test_demin_donne_les_puces_de_main_pour_demain_bien_ecrit(self):
        attendu = self._main("planifie « revision » demain de 14h a 16h")
        self.assertIsNotNone(attendu)
        self.assertIsNone(self._main("planifie « revision » demin de 14h a 16h"))
        self.assertEqual(self._types(_revision(DEMIN)), attendu)

    def test_aujourdui_donne_les_puces_de_main_pour_aujourd_hui_bien_ecrit(self):
        attendu = self._main("planifie « revision » aujourd'hui de 14h a 16h")
        self.assertIsNone(self._main("planifie « revision » aujourdui de 14h a 16h"))
        aujourdui = _ref("aujourdui", "jour_relatif", jour_relatif="aujourdhui")
        self.assertEqual(self._types(_revision(aujourdui)), attendu)

    def test_un_jour_deja_passe_cette_semaine_donne_une_seule_date(self):
        aujourdhui = timezone.localdate()
        base = aujourdhui if aujourdhui.weekday() > 0 else aujourdhui + timedelta(days=1)
        passe = schema.JOURS_NOMS[base.weekday() - 1]
        cible = base + timedelta(days=6)
        attendu = self._main(f"planifie « revision » le {cible.isoformat()} de 14h a 16h")
        self.assertIsNotNone(attendu)
        obtenu = self._types(_revision(_ref(passe, "jour_semaine", jour_semaine=passe)), aujourdhui=base)
        self.assertEqual(obtenu, attendu)

    def test_un_jour_encore_a_venir_sans_semaine_fixee_ne_force_rien(self):
        aujourdhui = timezone.localdate()
        base = aujourdhui if aujourdhui.weekday() < 6 else aujourdhui + timedelta(days=1)
        a_venir = schema.JOURS_NOMS[base.weekday() + 1]
        self.assertIsNone(self._types(_revision(_ref(a_venir, "jour_semaine", jour_semaine=a_venir)),
                                      aujourdhui=base))

    def test_une_heure_nue_approximative_ou_incomplete_ne_force_rien(self):
        for nom, heures in (
                ("heures nues", (_heure("2h", "02:00", "14:00"), _heure("4h", "04:00", "16:00", role="fin"))),
                ("approximative", (_heure("vers 14h", "14:00", genre="approx"), _heure("16h", "16:00", role="fin"))),
                ("sans fin", (_heure("14h", "14:00"),)),
                ("fin avant debut", (_heure("16h", "16:00"), _heure("14h", "14:00", role="fin")))):
            with self.subTest(cas=nom):
                self.assertIsNone(self._types(_revision(DEMIN, *heures)))

    def test_une_fenetre_libre_ne_force_rien(self):
        self.assertIsNone(self._main("planifie « revision » demain de 14h a 16h", libre=LIBRE))
        self.assertIsNone(self._types(_revision(DEMIN), libre=LIBRE))

    def test_une_fenetre_hors_de_la_journee_ne_force_rien(self):
        tard = _revision(DEMIN, _heure("23h", "23:00"), _heure("23h30", "23:30", role="fin"))
        self.assertIsNone(self._types(tard))

    def test_les_lecteurs_geles_gardent_le_plancher(self):
        lu = _ldt(_revision(DEMIN))

        def types(message):
            with patch("services.scheduling.placement.open_intervals", return_value=OCCUPE):
                return boutons.creneaux_types(self.user, message, None, Registre(), lu)

        # Main lit « aujourd'hui », la lecture « demain »: la jambe typee se tait.
        self.assertIsNone(types("planifie « revision » aujourd'hui de 14h a 16h"))
        # Main lit 15 h, la lecture 14 h: idem.
        self.assertIsNone(types("planifie « revision » demin de 15h a 16h"))
        # Main ne lit aucune date (faute) et les memes heures: la lecture decide.
        self.assertIsNotNone(types("planifie « revision » demin de 14h a 16h"))

    def test_ce_que_la_regle_laisse_a_main(self):
        demin = _revision(DEMIN)
        vendredi = _ref("vendredi", "jour_semaine", jour_semaine="vendredi", semaine="semaine_prochaine")
        cas = {
            "placement tente": dict(registre=_registre("schedule_task_at", False, {"conflict": {"x": 1}})),
            "creneaux consultes": dict(registre=_registre("find_free_slots", True, {"free_slots": []})),
            "placement reussi": dict(registre=_registre("schedule_task_at", True, {})),
            "creation reussie": dict(registre=_registre("create_block", True, {"created": []})),
            "choix deja pose": dict(registre=_registre("present_choices", True, {})),
            "piece jointe": dict(attachment=object()),
            "sans lecture": dict(ldt=False),
            "deplacement": dict(element={**demin, "operation": "deplacer"}),
            "deux demandes": dict(element=None),
            "deux dates": dict(element={**demin, "dates": [DEMIN, vendredi]}),
        }
        for nom, options in cas.items():
            with self.subTest(cas=nom):
                element = options.pop("element", demin)
                if element is None:
                    with patch("services.scheduling.placement.open_intervals", return_value=OCCUPE):
                        resultat = boutons.creneaux_types(self.user, "planifie", None, Registre(),
                                                          _ldt(demin, {**demin, "mention": "gym"}))
                else:
                    resultat = self._types(element, **options)
                self.assertIsNone(resultat)
