"""
Lot 2 « demander quand il le faut »: present_choices, les prompts et les
questions forcees par le code.

L'enquete du 2026-09-14 a mesure 0 formulaire et 0 bouton de choix sur 37
tours v2 en production: le modele ne pouvait pas proposer de reponses en un
tap, et ses prompts lui disaient que la question etait presque toujours
inutile. Ces tests verrouillent l'outil de choix (ancre dans le planning
reel, jamais un canal d'affirmation d'action), la table de decision des
prompts, la semaine type injectee a AGIR et le silence d'une question forcee
ignoree deux fois.
"""
from datetime import date, datetime, time as dtime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase

from core.models import (ConversationMessage, RecurringBlock, ScheduledBlock, Task,
                         UploadedDocument)
from services.agent.tools import (ALL_TOOLS, TOOL_MAP, V2_SEULEMENT, execute_tool,
                                  get_tools_for_claude)
from services.agent.tools.base import ToolResult
from services.agent_v2.boutons import question_forcee
from services.agent_v2.registre import Registre

TORONTO = ZoneInfo("America/Toronto")
LUNDI_8H = datetime(2026, 9, 14, 8, 0, tzinfo=TORONTO)
JEUDI = "2026-09-17"


def _figer(instant=LUNDI_8H):
    return patch("services.agent.tools.interactive._maintenant", return_value=instant)


def _bloc(user, titre, jour, debut, fin, **extra):
    extra.setdefault("block_type", "course")
    return RecurringBlock.objects.create(
        user=user, title=titre, day_of_week=jour,
        start_time=dtime(*debut), end_time=dtime(*fin), **extra)


class ExpositionTests(SimpleTestCase):
    def test_present_choices_expose(self):
        self.assertIn("present_choices", TOOL_MAP)
        self.assertIn("present_choices", [t.name for t in ALL_TOOLS])
        self.assertNotIn("present_choices", [t["name"] for t in get_tools_for_claude()])

    def test_v1_garde_tous_les_autres_outils(self):
        noms_v1 = [t["name"] for t in get_tools_for_claude()]
        attendus = [t.name for t in ALL_TOOLS if t.name not in V2_SEULEMENT]
        self.assertEqual(noms_v1, attendus)
        self.assertIn("present_form", noms_v1)

    def test_present_choices_suit_present_form(self):
        noms = [t.name for t in ALL_TOOLS]
        self.assertEqual(noms.index("present_choices"), noms.index("present_form") + 1)

    def test_schema_sans_source_autre(self):
        schema = TOOL_MAP["present_choices"].parameters
        self.assertEqual(schema["required"], ["question", "options", "source"])
        self.assertEqual(schema["properties"]["source"]["enum"],
                         ["creneaux", "blocs", "taches", "jours"])
        self.assertIn("date", schema["properties"])


class PresentChoicesTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="choix", password="x")

    def _choix(self, **args):
        return execute_tool("present_choices", self.user, args)

    def test_present_choices_ancre_sur_les_blocs(self):
        _bloc(self.user, "Calcul différentiel", 0, (10, 0), (11, 50))
        _bloc(self.user, "Physique mécanique", 2, (13, 0), (15, 0))
        r = self._choix(
            question="Lequel de tes cours ?",
            options=[{"label": "Calcul différentiel", "value": "Le cours de Calcul différentiel"},
                     {"label": "Physique mécanique", "value": "Le cours de Physique mécanique"}],
            source="blocs")
        self.assertTrue(r.success, r.message)
        demande = r.data["demande"]
        self.assertEqual(demande["motif"], "choix_modele")
        self.assertEqual(demande["type"], "choix")
        self.assertEqual([o["id"] for o in demande["options"]], ["o1", "o2"])
        self.assertEqual(demande["options"][0]["libelle"], "Calcul différentiel")
        self.assertEqual(demande["options"][0]["valeur"], "Le cours de Calcul différentiel")
        self.assertEqual(demande["options"][0]["cible"], {"titre": "Calcul différentiel"})
        self.assertIsNone(demande["options"][0]["effet"])
        self.assertEqual(demande["question"], "Lequel de tes cours ?")
        self.assertEqual(demande["source"], "blocs")
        self.assertEqual(demande["outil"], "present_choices")
        self.assertTrue(demande["cle"].startswith("choix:"))
        self.assertEqual(len(demande["cle"]), len("choix:") + 12)
        datetime.fromisoformat(demande["emise_le"])

    def test_la_cle_depend_de_la_question_seulement(self):
        _bloc(self.user, "Chimie générale", 0, (8, 0), (9, 0))
        _bloc(self.user, "Chimie organique", 1, (8, 0), (9, 0))
        options = [{"label": "Chimie générale", "value": "Chimie générale"},
                   {"label": "Chimie organique", "value": "Chimie organique"}]
        a = self._choix(question="Lequel de tes cours de chimie ?", options=options, source="blocs")
        b = self._choix(question="Lequel de tes cours de chimie ?",
                        options=list(reversed(options)), source="blocs")
        c = self._choix(question="Quel cours de chimie ?", options=options, source="blocs")
        self.assertEqual(a.data["demande"]["cle"], b.data["demande"]["cle"])
        self.assertNotEqual(a.data["demande"]["cle"], c.data["demande"]["cle"])

    def test_present_choices_rejette_l_invente(self):
        _bloc(self.user, "Calcul différentiel", 0, (10, 0), (11, 50))
        r = self._choix(
            question="Lequel de tes cours ?",
            options=[{"label": "Chimie organique", "value": "Chimie organique"},
                     {"label": "Calcul différentiel", "value": "Calcul différentiel"}],
            source="blocs")
        self.assertFalse(r.success)
        self.assertNotIn("demande", r.data)

    def test_les_accents_et_la_casse_ne_comptent_pas(self):
        _bloc(self.user, "Calcul différentiel", 0, (10, 0), (11, 50))
        ScheduledBlock.objects.create(
            user=self.user, task=Task.objects.create(user=self.user, title="Dentiste"),
            date=date(2026, 9, 17), start_time=dtime(10, 30), end_time=dtime(11, 30))
        r = self._choix(
            question="Lequel ?",
            options=[{"label": "calcul differentiel", "value": "Le calcul"},
                     {"label": "RDV Dentiste", "value": "Le dentiste"}],
            source="blocs")
        self.assertTrue(r.success, r.message)
        self.assertEqual([o["cible"]["titre"] for o in r.data["demande"]["options"]],
                         ["Calcul différentiel", "Dentiste"])

    def test_un_bloc_inactif_ou_un_mot_partiel_ne_suffit_pas(self):
        _bloc(self.user, "Gym", 0, (18, 0), (19, 0), block_type="sport", active=False)
        _bloc(self.user, "Art", 1, (18, 0), (19, 0), block_type="other")
        r = self._choix(
            question="Lequel ?",
            options=[{"label": "Gym", "value": "Gym"},
                     {"label": "Artisanat", "value": "Artisanat"}],
            source="blocs")
        self.assertFalse(r.success)

    def test_ancre_sur_les_taches(self):
        Task.objects.create(user=self.user, title="Rapport de labo")
        Task.objects.create(user=self.user, title="Lecture chapitre 3")
        r = self._choix(
            question="Laquelle de tes tâches ?",
            options=[{"label": "Rapport de labo", "value": "Le rapport de labo"},
                     {"label": "Lecture chapitre 3", "value": "La lecture"},
                     {"label": "Dissertation", "value": "La dissertation"}],
            source="taches")
        self.assertTrue(r.success, r.message)
        self.assertEqual(len(r.data["demande"]["options"]), 2)

    def test_present_choices_creneaux_libres_seulement(self):
        _bloc(self.user, "Calcul", 3, (10, 0), (11, 50))
        options = [{"label": "10 h à 11 h", "value": "Va pour 10 h à 11 h jeudi."},
                   {"label": "13 h à 14 h", "value": "Va pour 13 h à 14 h jeudi."},
                   {"label": "15 h à 16 h", "value": "Va pour 15 h à 16 h jeudi."}]
        with _figer():
            r = self._choix(question="Quel créneau te va jeudi ?", options=options,
                            source="creneaux", date=JEUDI)
            sans_date = self._choix(question="Quel créneau te va jeudi ?", options=options,
                                    source="creneaux")
        self.assertTrue(r.success, r.message)
        gardees = r.data["demande"]["options"]
        self.assertEqual([o["libelle"] for o in gardees], ["13 h à 14 h", "15 h à 16 h"])
        self.assertEqual(gardees[0]["cible"], {"date": JEUDI, "debut": "13:00", "fin": "14:00"})
        self.assertEqual(r.data["demande"]["cible"], {"date": JEUDI})
        self.assertFalse(sans_date.success)

    def test_creneau_sans_fin_vaut_trente_minutes(self):
        _bloc(self.user, "Calcul", 3, (10, 0), (11, 50))
        with _figer():
            r = self._choix(
                question="Quelle heure ?",
                options=[{"label": "9 h 45", "value": "9 h 45"},
                         {"label": "11 h 50", "value": "11 h 50"},
                         {"label": "12 h", "value": "12 h"}],
                source="creneaux", date=JEUDI)
        self.assertTrue(r.success, r.message)
        self.assertEqual([(o["cible"]["debut"], o["cible"]["fin"])
                          for o in r.data["demande"]["options"]],
                         [("11:50", "12:20"), ("12:00", "12:30")])

    def test_le_quart_de_nuit_de_la_veille_occupe_le_matin(self):
        """Mercredi 19:00-02:00: jeudi de minuit a 2 h est pris (fin < debut voulue)."""
        _bloc(self.user, "Quart au dépanneur", 2, (19, 0), (2, 0),
              block_type="work", is_night_shift=True)
        with _figer():
            r = self._choix(
                question="Quel créneau jeudi ?",
                options=[{"label": "1 h à 2 h", "value": "1 h à 2 h"},
                         {"label": "3 h à 4 h", "value": "3 h à 4 h"},
                         {"label": "23 h à minuit", "value": "23 h à minuit"}],
                source="creneaux", date=JEUDI)
        self.assertTrue(r.success, r.message)
        self.assertEqual([o["libelle"] for o in r.data["demande"]["options"]],
                         ["3 h à 4 h", "23 h à minuit"])
        self.assertEqual(r.data["demande"]["options"][1]["cible"]["fin"], "00:00")

    def test_un_creneau_deja_passe_est_ecarte(self):
        options = [{"label": "7 h à 8 h", "value": "7 h"},
                   {"label": "9 h à 10 h", "value": "9 h"},
                   {"label": "10 h à 11 h", "value": "10 h"}]
        with _figer():
            aujourdhui = self._choix(question="Quel créneau ?", options=options,
                                     source="creneaux", date="2026-09-14")
            hier = self._choix(question="Quel créneau ?", options=options,
                               source="creneaux", date="2026-09-13")
        self.assertTrue(aujourdhui.success, aujourdhui.message)
        self.assertEqual([o["libelle"] for o in aujourdhui.data["demande"]["options"]],
                         ["9 h à 10 h", "10 h à 11 h"])
        self.assertFalse(hier.success)

    def test_ancre_sur_les_jours(self):
        r = self._choix(
            question="Pour quelle journée ?",
            options=[{"label": "Jeudi", "value": "Jeudi"},
                     {"label": "Demain", "value": "Demain"},
                     {"label": "Le 24 sept.", "value": "Le 24 septembre"},
                     {"label": "Quand tu veux", "value": "Quand tu veux"}],
            source="jours")
        self.assertTrue(r.success, r.message)
        options = r.data["demande"]["options"]
        self.assertEqual([o["libelle"] for o in options], ["Jeudi", "Demain", "Le 24 sept."])
        self.assertEqual(options[0]["cible"], {"jour": 3})

    def test_present_choices_sans_source_autre(self):
        r = self._choix(question="Tu préfères quoi ?",
                        options=[{"label": "A", "value": "A"}, {"label": "B", "value": "B"}],
                        source="autre")
        self.assertFalse(r.success)

    def test_present_choices_rejette_une_affirmation(self):
        _bloc(self.user, "Calcul différentiel", 0, (10, 0), (11, 50))
        _bloc(self.user, "Physique mécanique", 2, (13, 0), (15, 0))
        valides = [{"label": "Calcul différentiel", "value": "Calcul différentiel"},
                   {"label": "Physique mécanique", "value": "Physique mécanique"}]
        r = self._choix(
            question="Lequel de tes cours ?",
            options=[{"label": "J'ai déplacé ton cours", "value": "Calcul différentiel"}] + valides,
            source="blocs")
        self.assertTrue(r.success, r.message)
        self.assertEqual([o["libelle"] for o in r.data["demande"]["options"]],
                         ["Calcul différentiel", "Physique mécanique"])

        valeur_menteuse = self._choix(
            question="Lequel de tes cours ?",
            options=[{"label": "Calcul différentiel", "value": "Calcul différentiel est supprimé"},
                     valides[1]],
            source="blocs")
        self.assertFalse(valeur_menteuse.success)

        for question in ("J'ai supprimé tes blocs, lequel remettre ?",
                         "Ton cours déplacé à 14 h te convient ?"):
            with self.subTest(question=question):
                self.assertFalse(self._choix(question=question, options=valides,
                                             source="blocs").success)

    def test_present_choices_bornes(self):
        titres = ["Anglais", "Biologie", "Chimie", "Histoire", "Physique"]
        for jour, titre in enumerate(titres):
            _bloc(self.user, titre, jour, (8, 0), (9, 0))
        options = [{"label": t, "value": f"Le cours de {t}"} for t in titres]
        une = self._choix(question="Lequel ?", options=options[:1], source="blocs")
        cinq = self._choix(question="Lequel ?", options=options, source="blocs")
        self.assertFalse(une.success)
        self.assertTrue(cinq.success, cinq.message)
        self.assertEqual([o["libelle"] for o in cinq.data["demande"]["options"]], titres[:4])

    def test_question_mal_formee_refusee(self):
        _bloc(self.user, "Anglais", 0, (8, 0), (9, 0))
        _bloc(self.user, "Biologie", 1, (8, 0), (9, 0))
        options = [{"label": "Anglais", "value": "Anglais"},
                   {"label": "Biologie", "value": "Biologie"}]
        for question in ("Lequel de tes cours", "", None, "x" * 140 + " ?"):
            with self.subTest(question=question):
                self.assertFalse(self._choix(question=question, options=options,
                                             source="blocs").success)

    def test_options_mal_formees_ou_en_double_ecartees(self):
        _bloc(self.user, "Anglais", 0, (8, 0), (9, 0))
        _bloc(self.user, "Biologie", 1, (8, 0), (9, 0))
        r = self._choix(
            question="Lequel ?",
            options=["Anglais", {"label": "Anglais"}, {"label": "Anglais", "value": "A"},
                     {"label": "anglais", "value": "B"}, {"label": "B" * 41, "value": "B"},
                     {"label": "Biologie", "value": "Biologie"}],
            source="blocs")
        self.assertTrue(r.success, r.message)
        self.assertEqual([(o["libelle"], o["valeur"]) for o in r.data["demande"]["options"]],
                         [("Anglais", "A"), ("Biologie", "Biologie")])

    def test_le_choix_n_est_pas_une_mutation(self):
        from services.agent_v2.registre import OUTILS_DE_MUTATION
        self.assertNotIn("present_choices", OUTILS_DE_MUTATION)


class PromptsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="prompts", password="x")

    def test_regles_agir_ne_poussent_plus_a_deviner(self):
        from services.agent_v2.prompts import REGLES_AGIR
        for interdit in ("Neuf fois sur dix", "AGIS avant de demander",
                         "PROPOSE PLUTOT QUE DEMANDER", "create_block (recurrent) tout de suite",
                         "(tu choisis l'heure toi-meme)", "la question etait inutile"):
            with self.subTest(interdit=interdit):
                self.assertNotIn(interdit, REGLES_AGIR)
        for requis in ("present_choices", "QUESTION A CHOIX", "rendez-vous", "SUITE AU CHOIX",
                       "Une heure donnee par l'utilisateur ne se change jamais sans lui demander",
                       "Quand un outil te repond qu'une question est posee par le code, n'agis "
                       "pas sur ce point et ne repose pas la question",
                       "ne refais rien de ce qui y est marque FAIT, ne touche pas a ce qui est REFUSE",
                       "une demande pour cette semaine ne cree pas d'habitude sans fin",
                       "optimize_week apply=true seulement apres confirmation"):
            with self.subTest(requis=requis):
                self.assertIn(requis, REGLES_AGIR)

    def test_table_de_decision_nomme_les_cas_a_demander(self):
        from services.agent_v2.prompts import REGLES_AGIR
        for mot in ("DEVINE", "DEMANDE", "cours", "quart", "reunion", "lecon",
                    "plus", "davantage", "mieux", "planning vide", "present_form"):
            with self.subTest(mot=mot):
                self.assertIn(mot, REGLES_AGIR)

    def test_premier_contact(self):
        from services.agent_v2.prompts import prompt_agir
        self.assertFalse(self.user.profile.onboarding_completed)
        self.assertIn("PREMIER CONTACT", prompt_agir(self.user))
        _bloc(self.user, "Gym", 0, (18, 0), (19, 0), block_type="sport")
        self.assertNotIn("PREMIER CONTACT", prompt_agir(self.user))

    def test_premier_contact_absent_apres_l_onboarding(self):
        from services.agent_v2.prompts import prompt_agir
        self.user.profile.onboarding_completed = True
        self.user.profile.save()
        self.assertNotIn("PREMIER CONTACT", prompt_agir(self.user))

    def test_resume_semaine(self):
        from services.agent_v2.prompts import prompt_agir, resume_semaine
        for jour in (0, 2, 4):
            _bloc(self.user, "Gym", jour, (18, 0), (19, 0),
                  block_type="sport", flexibility="flexible")
        _bloc(self.user, "Quart au dépanneur", 3, (19, 0), (2, 0),
              block_type="work", flexibility="fixed", is_night_shift=True)
        resume = resume_semaine(self.user)
        self.assertIn("- Gym: lun, mer, ven 18:00-19:00 (souple)", resume)
        self.assertIn("- Quart au dépanneur: jeu 19:00-02:00", resume)
        self.assertNotIn("- Quart au dépanneur: jeu 19:00-02:00 (souple)", resume)
        self.assertEqual(resume.splitlines()[0], "- Gym: lun, mer, ven 18:00-19:00 (souple)")
        prompt = prompt_agir(self.user)
        self.assertIn("SEMAINE TYPE", prompt)
        self.assertIn("- Gym: lun, mer, ven 18:00-19:00 (souple)", prompt)

    def test_resume_semaine_separe_les_heures_differentes_et_ignore_les_blocs_finis(self):
        from services.agent_v2.prompts import resume_semaine
        _bloc(self.user, "Chimie", 1, (8, 0), (9, 50))
        _bloc(self.user, "Chimie", 3, (13, 0), (14, 50))
        _bloc(self.user, "Ancien cours", 0, (8, 0), (9, 0), end_date=date(2020, 1, 1))
        _bloc(self.user, "Supprimé", 0, (8, 0), (9, 0), active=False)
        lignes = resume_semaine(self.user).splitlines()
        self.assertEqual(lignes, ["- Chimie: mar 08:00-09:50", "- Chimie: jeu 13:00-14:50"])

    def test_resume_semaine_plafonne(self):
        from services.agent_v2.prompts import MAX_LIGNES_SEMAINE, resume_semaine
        for rang in range(MAX_LIGNES_SEMAINE + 3):
            _bloc(self.user, f"Cours {rang:02d}", rang % 7, (6 + rang // 7, 0), (6 + rang // 7, 30))
        lignes = resume_semaine(self.user).splitlines()
        self.assertEqual(len(lignes), MAX_LIGNES_SEMAINE + 1)
        self.assertEqual(lignes[-1], "- ... et 3 autres")

    def test_semaine_vide(self):
        from services.agent_v2.prompts import prompt_agir, resume_semaine
        self.assertEqual(resume_semaine(self.user), "")
        self.assertIn("(aucun bloc recurrent)", prompt_agir(self.user))

    def test_prompt_dire(self):
        from services.agent_v2.prompts import PROMPT_DIRE
        for requis in ("12 mots", "question", "options", "import_recent", "refs",
                       "ouverture", "suite", "QUESTION DEJA POSEE PAR LE CODE",
                       "BROUILLON D'AGIR", "CONTEXTE (ne pas citer)", "9 h 30"):
            with self.subTest(requis=requis):
                self.assertIn(requis, PROMPT_DIRE)
        self.assertNotIn("Chaque action mentionnee DOIT porter la reference", PROMPT_DIRE)

    def test_aucun_tiret_cadratin(self):
        from services.agent_v2 import prompts
        for nom in ("REGLES_AGIR", "PROMPT_DIRE", "PREMIER_CONTACT"):
            with self.subTest(nom=nom):
                self.assertNotIn(chr(0x2014), getattr(prompts, nom))


JOUR_LOIN = date(2030, 6, 10)


def _registre_conflit():
    r = Registre()
    r.ajouter("schedule_task_at",
              {"title": "Révision", "date": JOUR_LOIN.isoformat(),
               "start_time": "10:00", "end_time": "11:00"},
              ToolResult(success=False, message="Conflit avec Cours",
                         data={"conflict": {"title": "Cours"}}))
    return r


class QuestionForceeTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="forcee", password="x")
        self.doc = UploadedDocument.objects.create(
            user=self.user, file_name="horaire.pdf",
            document_type="course_schedule", processed=True,
            extracted_data={"courses": [{"name": "Physique"}]})
        _bloc(self.user, "Physique", 2, (9, 0), (11, 0), source_document=self.doc)

    def _import(self):
        return question_forcee(self.user, "voici mon horaire", self.doc, Registre(), True)

    def _echange(self, reponse_utilisateur, motif="fin_recurrence", chips=None):
        chips = chips if chips is not None else [
            {"label": "🏁 Je te donne la date de fin",
             "value": "Je vais te donner la date de fin pour Physique."},
            {"label": "♾️ Pas de fin prévue",
             "value": "Physique n'a pas de date de fin, garde-le tel quel."}]
        ConversationMessage.objects.create(
            user=self.user, role="assistant", content="Jusqu'à quand ?",
            metadata={"question_motif": motif, "quick_replies": chips})
        ConversationMessage.objects.create(user=self.user, role="user", content=reponse_utilisateur)

    def test_fin_de_recurrence_sans_libelles_dans_la_question(self):
        q = self._import()
        self.assertEqual(q["motif"], "fin_recurrence")
        self.assertEqual(
            q["question"],
            "« Physique » n'a pas de date de fin pour l'instant : "
            "jusqu'à quand veux-tu le garder à l'horaire ?")
        self.assertEqual([c["label"] for c in q["chips"]],
                         ["🏁 Je te donne la date de fin", "♾️ Pas de fin prévue"])
        for chip in q["chips"]:
            self.assertNotIn(chip["label"], q["question"])
        self.assertEqual(set(q), {"question", "chips", "motif"})

    def test_question_forcee_ignoree_deux_fois_se_tait(self):
        ConversationMessage.objects.create(user=self.user, role="user", content="voici mon horaire")
        self._echange("bonjour")
        self._echange("et mon cours de jeudi ?")
        self.assertIsNone(self._import())

    def test_ignoree_une_seule_fois_reste_posee(self):
        ConversationMessage.objects.create(user=self.user, role="user", content="voici mon horaire")
        self._echange("Physique n'a pas de date de fin, garde-le tel quel.")
        self._echange("bonjour")
        q = self._import()
        self.assertIsNotNone(q)
        self.assertEqual(q["motif"], "fin_recurrence")

    def test_un_autre_motif_ne_compte_pas(self):
        self._echange("bonjour", motif="creneaux")
        self._echange("salut")
        self.assertEqual(self._import()["motif"], "fin_recurrence")

    def test_creneaux_aux_heures_humaines(self):
        _bloc(self.user, "Cours", JOUR_LOIN.weekday(), (7, 0), (15, 0), flexibility="fixed")
        q = question_forcee(self.user, "planifie ma révision", None, _registre_conflit(), False)
        self.assertEqual(q["motif"], "creneaux")
        self.assertTrue(q["question"].endswith("?"), q["question"])
        self.assertIn("pris", q["question"])
        self.assertTrue(1 <= len(q["chips"]) <= 3, q["chips"])
        self.assertEqual(q["chips"][0]["label"], "15 h à 16 h")
        for chip in q["chips"]:
            self.assertNotRegex(chip["label"], r"\d{2}:\d{2}")
            self.assertNotRegex(chip["value"], r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}")
            self.assertIn("Planifie « Révision »", chip["value"])
            self.assertNotIn(chip["label"], q["question"])

    def test_creneaux_ignores_deux_fois_se_taisent(self):
        _bloc(self.user, "Cours", JOUR_LOIN.weekday(), (7, 0), (15, 0), flexibility="fixed")
        self._echange("non merci", motif="creneaux", chips=[{"label": "15 h à 16 h", "value": "x"}])
        self._echange("plus tard", motif="creneaux", chips=[{"label": "15 h à 16 h", "value": "x"}])
        self.assertIsNone(
            question_forcee(self.user, "planifie ma révision", None, _registre_conflit(), False))

    def test_rien_a_forcer(self):
        self.assertIsNone(question_forcee(self.user, "bonjour", None, Registre(), False))
