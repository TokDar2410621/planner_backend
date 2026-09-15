"""
LIRE decide: les deux premieres regles (formulaire du code, creneaux types).

Le fondateur a valide deux regles ou la lecture typee DECIDE, sous ses
principes: le modele lit la langue, le code decide sur des champs types; aucune
liste de phrases ni regex neuve sur le texte de l'utilisateur; les lecteurs
regex geles restent le repli et le plancher; une regle ajoute une question ou
un formulaire, jamais une ecriture; rien n'est pire que main (I4).

Regle 1 (formulaire_cours), le bug « mets mon cours de maths »: AGIR voyait
« Calcul differentiel », concluait qu'il n'y avait rien a faire, et DIRE
ecrivait « Je ne vois pas de cours de maths dans ton horaire » avec des jours
inventes en puces. Regle 2 (creneaux), le bug « aujourdui »: la jambe regex de
v1 ne lit que aujourd'hui, demain ou une date ISO.

Sans lecture utilisable, LIRE coupe ou regle coupee: le tour est celui de main.
"""
import json
import re
from contextlib import ExitStack
from datetime import date, time as heure, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase, TransactionTestCase, override_settings
from django.utils import timezone
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from core.models import ConversationMessage, RecurringBlock
from core.test_agent_v2_lire_ombre import _agir_scripte, _element, _heure, _normaliser, _ref
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
        for champ in (jours, heures):
            self.assertIn("« mon cours de maths »", champ["label"])
            self.assertIsNone(MOTS_D_INTERFACE.search(f"{champ['label']} {champ['question']}"))

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


# ------------------------------------------------------------- tours complets

DIRE_ABSENCE = {"ouverture": "Je ne vois pas de cours de maths dans ton horaire.", "suite": "",
                "question": "Quel jour veux-tu ?", "options": ["Lundi", "Mardi"], "refs": []}
DIRE_NEUTRE = {"ouverture": "D'accord.", "suite": "", "question": "", "options": [], "refs": []}
COURS_MATHS_TOUR = _element(mention="mon cours de maths", genre="course", candidats=["s1"])
REPONSE_MATHS = f"Déjà à ton horaire : Calcul différentiel\n{QUESTION_MATHS}"
REVISION_DEMIN = _revision(DEMIN)


class ToursTests(TransactionTestCase):
    """AGIR et DIRE simules, outils et gardes reels, lecture injectee au point
    ou le tour la soumet (lecture.demarrer)."""

    def _utilisateur(self, nom):
        user = User.objects.create_user(username=nom, password="x")
        for jour in (0, 2):
            RecurringBlock.objects.create(user=user, title="Calcul différentiel", block_type="course",
                                          day_of_week=jour, start_time=heure(13, 0), end_time=heure(15, 50))
        return user

    def _tour(self, user, message, elements=None, statut="ok", lire="1", regles_actives=None,
              appel=None, dire=DIRE_ABSENCE, libre=None, rapide=False):
        from services.agent_v2 import agent as module_agent
        from services.agent_v2.agent import PlannerAgentV2

        appels_dire: list = []

        def rediger(messages, info: AgentInfo):
            appels_dire.append(1)
            return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=dire)])

        def demarrer(u, m):
            prep = lecture.preparer(u, m)
            lu = None
            if elements is not None and statut in (lecture.OK, lecture.PARTIELLE):
                brute = schema.LectureTour.model_validate({"elements": elements, "reponses": []})
                lu, _rejets = schema.ancrer(brute, m, prep.aujourdhui, prep.refs)
            return lecture.Suivi(preparation=prep, fixe=lecture.Resultat(
                statut, lecture=lu, fournisseur="deepseek-flash" if lu is not None else ""))

        reglages = {"LIRE_OMBRE": lire}
        if regles_actives is not None:
            reglages["LIRE_REGLES"] = regles_actives
        with ExitStack() as pile:
            pile.enter_context(override_settings(**reglages))
            pile.enter_context(patch.object(module_agent, "modele_agir", side_effect=lambda: _agir_scripte(appel)))
            pile.enter_context(patch.object(module_agent, "modele_dire", side_effect=lambda: FunctionModel(rediger)))
            if lire == "1":
                pile.enter_context(patch.object(lecture, "demarrer", side_effect=demarrer))
            if libre is not None:
                pile.enter_context(patch("services.scheduling.placement.open_intervals", return_value=libre))
            if rapide:
                pile.enter_context(patch.object(PlannerAgentV2, "_tour_decide",
                                                staticmethod(lambda registre, message: True)))
            conclure = pile.enter_context(patch.object(lecture, "conclure", wraps=lecture.conclure))
            journal = pile.enter_context(self.assertLogs(JOURNAL, "INFO"))
            evenements = list(PlannerAgentV2().process_message_stream(user, message))
        assistant = ConversationMessage.objects.filter(user=user, role="assistant").latest("pk")
        lignes = [r.getMessage() for r in journal.records
                  if r.getMessage().startswith("agent_v2 lire statut=")]
        return SimpleNamespace(done=evenements[-1], assistant=assistant, dire=len(appels_dire),
                               conclure=conclure.call_count, lignes=lignes)

    # -- regle 1

    def test_mets_mon_cours_de_maths_formulaire_du_code_et_dire_muet(self):
        t = self._tour(self._utilisateur("maths"), "mets mon cours de maths", [COURS_MATHS_TOUR])
        self.assertEqual(t.dire, 0)
        self.assertEqual(t.done["response"], REPONSE_MATHS)
        self.assertEqual(t.assistant.content, REPONSE_MATHS)
        self.assertEqual(t.done["quick_replies"], [])
        self.assertEqual(t.done["question_motif"], "formulaire")
        self.assertEqual([c["type"] for c in t.done["interactive_inputs"]], ["checkbox", "time_range"])
        self.assertEqual(t.assistant.metadata["interactive_inputs"], t.done["interactive_inputs"])
        self.assertTrue(any(a["outil"] == "present_form" and a["par_le_code"]
                            for a in t.assistant.metadata["actions"]))
        self.assertEqual(t.conclure, 1, "la lecture ne s'attend qu'une fois par tour")
        self.assertEqual(len(t.lignes), 1)
        self.assertTrue(t.lignes[0].endswith(" regle=formulaire_cours"), t.lignes[0])

    def test_regle_coupee_le_tour_de_main_reste(self):
        t = self._tour(self._utilisateur("coupee"), "mets mon cours de maths", [COURS_MATHS_TOUR],
                       regles_actives="creneaux")
        self.assertEqual(t.dire, 1)
        self.assertIn("Je ne vois pas de cours de maths dans ton horaire.", t.done["response"])
        self.assertTrue(t.lignes[0].endswith(" regle=-"))

    def test_sans_lecture_utilisable_le_tour_est_celui_de_main(self):
        message = "mets mon cours de maths"
        reference = self._tour(self._utilisateur("main"), message, lire="0")

        def sans_lire(assistant):
            return _normaliser({k: v for k, v in assistant.metadata.items() if k not in lecture.CLES_METADONNEES})

        for nom, options in (("regles-coupees", dict(elements=[COURS_MATHS_TOUR], regles_actives="")),
                             ("lecture-tardive", dict(elements=[COURS_MATHS_TOUR], statut=lecture.HORS_BUDGET)),
                             ("lecture-absente", dict(statut=lecture.ABSENTE))):
            with self.subTest(cas=nom):
                t = self._tour(self._utilisateur(nom), message, **options)
                self.assertEqual(json.dumps(_normaliser(t.done), sort_keys=True, ensure_ascii=False),
                                 json.dumps(_normaliser(reference.done), sort_keys=True, ensure_ascii=False))
                self.assertEqual(t.assistant.content, reference.assistant.content)
                self.assertEqual(sans_lire(t.assistant), sans_lire(reference.assistant))
                self.assertEqual(t.conclure, 1)

    def test_une_creation_reussie_ce_tour_garde_dire(self):
        appel = ("create_block", {"title": "Maths", "block_type": "course", "days": ["mardi"],
                                  "start_time": "09:00", "end_time": "10:00"})
        t = self._tour(self._utilisateur("creation"), "mets mon cours de maths", [COURS_MATHS_TOUR],
                       appel=appel, dire=DIRE_NEUTRE)
        self.assertEqual(t.dire, 1)
        self.assertFalse(t.done.get("interactive_inputs"))
        self.assertTrue(t.lignes[0].endswith(" regle=-"))

    def test_une_reponse_de_formulaire_garde_dire(self):
        message = "Voici mes réponses :\nCours: maths"
        t = self._tour(self._utilisateur("reponse"), message,
                       [_element(mention="maths", genre="course", candidats=["s1"])], dire=DIRE_NEUTRE)
        self.assertEqual(t.dire, 1)
        self.assertFalse(t.done.get("interactive_inputs"))
        self.assertTrue(t.lignes[0].endswith(" regle=-"))

    def test_le_chemin_rapide_n_evalue_aucune_regle(self):
        t = self._tour(self._utilisateur("rapide"), "oui", [COURS_MATHS_TOUR], rapide=True)
        self.assertEqual(t.dire, 0)
        self.assertFalse(t.done.get("interactive_inputs"))
        self.assertEqual(t.assistant.metadata["lecture_statut"], "sautee")
        self.assertTrue(t.lignes[0].endswith(" regle=-"))

    # -- regle 2

    def test_demin_force_les_memes_puces_que_main_pour_demain_bien_ecrit(self):
        main = self._tour(self._utilisateur("main-demain"), "planifie « revision » demain de 14h a 16h",
                          lire="0", dire=DIRE_NEUTRE, libre=OCCUPE)
        self.assertTrue(main.done["quick_replies"])
        aveugle = self._tour(self._utilisateur("main-demin"), "planifie revision demin de 14h a 16h",
                             lire="0", dire=DIRE_NEUTRE, libre=OCCUPE)
        self.assertEqual(aveugle.done["quick_replies"], [])
        t = self._tour(self._utilisateur("type-demin"), "planifie revision demin de 14h a 16h",
                       [REVISION_DEMIN], dire=DIRE_NEUTRE, libre=OCCUPE)
        for cle in ("quick_replies", "question", "question_motif", "response"):
            self.assertEqual(t.done[cle], main.done[cle], cle)
        self.assertEqual(t.dire, 1)
        self.assertEqual(t.conclure, 1)
        self.assertTrue(t.lignes[0].endswith(" regle=creneaux"), t.lignes[0])

    def test_creneaux_coupe_ou_lecture_tardive_comme_main(self):
        for nom, options in (("creneaux-coupe", dict(elements=[REVISION_DEMIN], regles_actives="formulaire_cours")),
                             ("demin-tardive", dict(elements=[REVISION_DEMIN], statut=lecture.HORS_BUDGET))):
            with self.subTest(cas=nom):
                t = self._tour(self._utilisateur(nom), "planifie revision demin de 14h a 16h",
                               dire=DIRE_NEUTRE, libre=OCCUPE, **options)
                self.assertEqual(t.done["quick_replies"], [])
                self.assertTrue(t.lignes[0].endswith(" regle=-"))
