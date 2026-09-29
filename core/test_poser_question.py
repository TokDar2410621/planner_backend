"""
poser_question: l'outil de question libre du modele (AskUserQuestion de v2).

Complement de present_choices (options ancrees dans le planning reel):
des options arbitraires pour les clarifications bornees qui ne portent
sur aucun element du planning. Ces tests verrouillent le contrat:
validation de la question et des options, rejet des affirmations d'action,
options SANS effet (un tap n'execute jamais d'outil), retour du choix vers
AGIR, une seule question par tour, dedup par cle, et exposition v2 seule.
"""
import asyncio

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase, TransactionTestCase

from services.agent.tools import ALL_TOOLS, TOOL_MAP, V2_SEULEMENT, execute_tool, get_tools_for_claude
from services.agent_v2 import agent as agent_v2
from services.agent_v2 import demandes as dem
from services.agent_v2 import outils as outils_v2
from services.agent_v2 import rendu
from services.agent_v2.outils import _resume_sans_effet
from services.agent_v2.prompts import REGLES_AGIR
from services.agent_v2.registre import Registre

QUESTION = "Tu veux un rappel la veille ?"
OPTIONS = [
    {"label": "Oui", "value": "Oui, rappelle-moi la veille."},
    {"label": "Non", "value": "Non, pas de rappel."},
]


def _demande(**extra):
    outil = TOOL_MAP["poser_question"]
    user = getattr(_demande, "user", None)
    if user is None:
        raise AssertionError("appeler _demande via PoserQuestionTests seulement")
    resultat = outil.execute(user, question=extra.pop("question", QUESTION),
                             options=extra.pop("options", OPTIONS), **extra)
    assert resultat.success, resultat.message
    return resultat.data["demande"]


class ExpositionTests(SimpleTestCase):
    def test_pose_question_expose_v2_seulement(self):
        self.assertIn("poser_question", TOOL_MAP)
        self.assertIn("poser_question", [t.name for t in ALL_TOOLS])
        self.assertIn("poser_question", V2_SEULEMENT)
        self.assertNotIn("poser_question", [t["name"] for t in get_tools_for_claude()])

    def test_schema_sans_ancrage(self):
        schema = TOOL_MAP["poser_question"].parameters
        self.assertEqual(schema["required"], ["question", "options"])
        self.assertNotIn("source", schema["properties"])
        self.assertIn("label", schema["properties"]["options"]["items"]["properties"])
        self.assertIn("value", schema["properties"]["options"]["items"]["properties"])

    def test_prompt_agir_documente_l_outil(self):
        # Doctrine 2026-09-29: l'usage des outils vit dans leurs descriptions,
        # pas dans le prompt.
        self.assertIn("present_choices", REGLES_AGIR)
        from services.agent.tools import TOOL_MAP
        from services.agent_v2.outils import description_v2
        self.assertIn("choix libre", description_v2(TOOL_MAP["poser_question"]))

    def test_priorites_connaissent_le_motif(self):
        self.assertIn("question_libre", rendu.PRIORITE)
        self.assertIn("question_libre", agent_v2.PRIORITE)
        self.assertLess(rendu.PRIORITE.index("question_libre"),
                        rendu.PRIORITE.index("chevauchement"))


class ValidationTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="libre", password="x")
        _demande.user = self.user

    def tearDown(self):
        _demande.user = None

    def _appeler(self, **kwargs):
        return execute_tool("poser_question", self.user, kwargs)

    def test_question_valide(self):
        demande = _demande()
        self.assertEqual(demande["motif"], "question_libre")
        self.assertEqual(demande["type"], "choix")
        self.assertEqual(demande["outil"], "poser_question")
        self.assertEqual(demande["question"], QUESTION)
        self.assertTrue(demande["cle"].startswith("question:"))
        self.assertIn("emise_le", demande)

    def test_options_sans_effet(self):
        demande = _demande()
        options = demande["options"]
        self.assertEqual([o["id"] for o in options], ["o1", "o2"])
        for o in options:
            self.assertIsNone(o["effet"])
        self.assertEqual(options[0]["libelle"], "Oui")
        self.assertEqual(options[0]["valeur"], "Oui, rappelle-moi la veille.")

    def test_question_sans_point_d_interrogation_refusee(self):
        r = self._appeler(question="Tu veux un rappel", options=OPTIONS)
        self.assertFalse(r.success)
        self.assertNotIn("demande", r.data)

    def test_question_trop_longue_refusee(self):
        r = self._appeler(question="x" * 140 + " ?", options=OPTIONS)
        self.assertFalse(r.success)

    def test_question_qui_affirme_une_action_refusee(self):
        r = self._appeler(question="J'ai déplacé ton cours, ça te va ?",
                          options=OPTIONS)
        self.assertFalse(r.success)
        self.assertNotIn("demande", r.data)

    def test_option_qui_affirme_une_action_ecartee(self):
        options = [OPTIONS[0],
                   {"label": "C'est fait", "value": "C'est fait, je l'ai déplacé."},
                   OPTIONS[1]]
        demande = _demande(options=options)
        self.assertEqual([o["libelle"] for o in demande["options"]],
                         ["Oui", "Non"])

    def test_moins_de_deux_options_refuse(self):
        r = self._appeler(question=QUESTION, options=[OPTIONS[0]])
        self.assertFalse(r.success)

    def test_cinq_options_tronquees_a_quatre(self):
        options = OPTIONS + [
            {"label": "Plus tard", "value": "Demande-moi plus tard."},
            {"label": "Jamais", "value": "Ne me le propose plus."},
            {"label": "Autre", "value": "Autre chose."},
        ]
        demande = _demande(options=options)
        self.assertEqual(len(demande["options"]), 4)

    def test_libelles_dupliques_dedoublonnes(self):
        demande = _demande(options=[OPTIONS[0], dict(OPTIONS[0]), OPTIONS[1]])
        self.assertEqual(len(demande["options"]), 2)

    def test_meme_question_meme_cle(self):
        self.assertEqual(_demande()["cle"], _demande()["cle"])

    def test_questions_differentes_cles_differentes(self):
        self.assertNotEqual(_demande()["cle"],
                            _demande(question="Tu préfères le matin ?")["cle"])


class RenduTests(SimpleTestCase):
    def _demande_brute(self, question=QUESTION, options=None):
        options = options if options is not None else OPTIONS
        return {
            "type": "choix", "motif": "question_libre", "cle": "question:abc",
            "outil": "poser_question", "question": question,
            "options": [{"id": f"o{i+1}", "effet": None, "libelle": o["label"],
                         "valeur": o["value"]} for i, o in enumerate(options)],
        }

    def test_rendu_question_et_chips(self):
        question, chips, cles = rendu.rendre_demandes([self._demande_brute()])
        self.assertEqual(question, QUESTION)
        self.assertEqual([(c["label"], c["option"]) for c in chips],
                         [("Oui", "o1"), ("Non", "o2")])
        self.assertEqual(cles, ["question:abc"])

    def test_deux_questions_libres_premiere_seule(self):
        demandes = [self._demande_brute(),
                    self._demande_brute(question="Tu préfères le matin ?")]
        demandes[1]["cle"] = "question:def"
        question, chips, cles = rendu.rendre_demandes(demandes)
        self.assertEqual(question, QUESTION)
        self.assertEqual(cles, ["question:abc"])

    def test_question_libre_passe_apres_garde_destructive(self):
        destructif = dict(self._demande_brute(), motif="destructif", cle="d:x")
        question, _, cles = rendu.rendre_demandes([self._demande_brute(), destructif])
        self.assertEqual(cles, ["d:x"])
        self.assertNotEqual(question, QUESTION)


class LectureTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="tap", password="x")
        _demande.user = self.user

    def tearDown(self):
        _demande.user = None

    def test_tap_retrouve_l_option(self):
        demande = _demande()
        tap = {"demande": demande["cle"], "option": "o1"}
        self.assertEqual(dem.option_choisie("Oui, rappelle-moi la veille.",
                                           demande, tap=tap), "o1")

    def test_tap_n_execute_aucun_outil(self):
        demande = _demande()
        for o in demande["options"]:
            self.assertIsNone(o["effet"])

    def test_resume_repart_vers_agir(self):
        demande = _demande()
        resume = _resume_sans_effet(demande, "o2")
        self.assertEqual(resume, "CHOISI PAR L'UTILISATEUR: Non, pas de rappel.")

    def test_tap_au_tour_suivant_sans_execution(self):
        demande = _demande()
        from core.models import ConversationMessage
        u1 = ConversationMessage.objects.create(user=self.user, role="user",
                                                content="mets un rappel")
        ConversationMessage.objects.create(
            user=self.user, role="assistant", content=QUESTION,
            metadata={"en_reponse_a": u1.pk, "demandes": [demande]})
        brut = "Oui, rappelle-moi la veille."
        ConversationMessage.objects.create(user=self.user, role="user", content=brut)
        registre = Registre()
        tap = {"demande": demande["cle"], "option": "o1"}
        sorties = outils_v2.appliquer_choix_en_attente(
            self.user, registre, brut, "u:2", tap=tap)
        self.assertTrue(any("CHOISI PAR L'UTILISATEUR" in (s.get("resume") or "")
                            for s in sorties))
        self.assertFalse(any(a.succes and (a.donnees or {}).get("par_le_code")
                             for a in registre.actions))


class UneSeuleQuestionTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="garde", password="x")

    def _outils(self, registre, brut="une question"):
        from core.models import ConversationMessage
        ConversationMessage.objects.create(user=self.user, role="user", content=brut)
        tools = {t.name: t for t in outils_v2.outils_pour(
            self.user, registre, brut, tache="u:1", message_brut=brut)}
        return tools

    def _appeler(self, tools, nom, **kwargs):
        return asyncio.run(tools[nom].function_schema.function(**kwargs))

    def test_deuxieme_question_du_tour_refusee(self):
        registre = Registre()
        tools = self._outils(registre)
        premier = self._appeler(tools, "poser_question", question=QUESTION,
                                options=OPTIONS)
        self.assertIn("Question posee", premier)
        second = self._appeler(tools, "poser_question",
                               question="Tu préfères le matin ?",
                               options=OPTIONS)
        self.assertIn("deja pose une question", second)

    def test_present_choices_puis_poser_question_refuse(self):
        from core.models import Task
        Task.objects.create(user=self.user, title="Rapport")
        Task.objects.create(user=self.user, title="Courses")
        registre = Registre()
        tools = self._outils(registre)
        premier = self._appeler(tools, "present_choices", question="Lequel ?",
                                options=[{"label": "Rapport", "value": "Rapport."},
                                         {"label": "Courses", "value": "Courses."}],
                                source="taches")
        self.assertIn('"success": true', premier)
        second = self._appeler(tools, "poser_question", question=QUESTION,
                               options=OPTIONS)
        self.assertIn("deja pose une question", second)


class DurcissementsTests(TransactionTestCase):
    """Les 3 points durcis apres la revue: batch sequentiel, refus des
    confirmations destructives, oui/non libre sur question binaire."""

    def setUp(self):
        self.user = User.objects.create_user(username="durci", password="x")
        from core.models import ConversationMessage
        ConversationMessage.objects.create(user=self.user, role="user",
                                           content="une question")
        self.registre = Registre()
        self.tools = {t.name: t for t in outils_v2.outils_pour(
            self.user, self.registre, "une question", tache="u:1",
            message_brut="une question")}

    def _demande_oui_non(self):
        outil = TOOL_MAP["poser_question"]
        resultat = outil.execute(self.user, question=QUESTION, options=OPTIONS)
        assert resultat.success, resultat.message
        return resultat.data["demande"]

    # -- 1. batch sequentiel

    def test_outils_question_sequentiels(self):
        self.assertTrue(self.tools["poser_question"].sequential)
        self.assertTrue(self.tools["present_choices"].sequential)

    def test_lecture_pure_reste_parallele(self):
        self.assertFalse(self.tools["list_blocks"].sequential)

    def test_mutation_reste_sequentielle(self):
        self.assertTrue(self.tools["create_block"].sequential)
        self.assertTrue(self.tools["delete_block"].sequential)

    # -- 2. confirmation destructive refusee par le code

    def _executer(self, **kwargs):
        return TOOL_MAP["poser_question"].execute(self.user, **kwargs)

    def test_question_suppression_refusee(self):
        resultat = self._executer(question="Je supprime ce bloc ?",
                                  options=OPTIONS)
        self.assertFalse(resultat.success)
        self.assertIn("suppression", resultat.message)

    def test_option_suppression_refusee(self):
        resultat = self._executer(
            question="Tu veux un rappel la veille ?",
            options=[{"label": "Oui, supprime-le",
                      "value": "Oui, supprime le bloc."},
                     {"label": "Non", "value": "Non, garde-le."}])
        self.assertFalse(resultat.success)
        self.assertIn("suppression", resultat.message)

    def test_question_annulation_refusee(self):
        resultat = self._executer(question="J'annule le rappel ?",
                                  options=OPTIONS)
        self.assertFalse(resultat.success)

    def test_question_saine_acceptee(self):
        resultat = self._executer(question=QUESTION, options=OPTIONS)
        self.assertTrue(resultat.success)

    # -- 3. oui/non libre sur question binaire

    def test_oui_libre_vaut_la_puce(self):
        demande = self._demande_oui_non()
        ids = {o["libelle"]: o["id"] for o in demande["options"]}
        self.assertEqual(dem.option_choisie("oui", demande), ids["Oui"])
        self.assertEqual(dem.option_choisie("Oui stp", demande), ids["Oui"])

    def test_non_libre_vaut_la_puce(self):
        demande = self._demande_oui_non()
        ids = {o["libelle"]: o["id"] for o in demande["options"]}
        self.assertEqual(dem.option_choisie("non", demande), ids["Non"])
        self.assertEqual(dem.option_choisie("non merci", demande), ids["Non"])

    def test_reponse_floue_ignoree(self):
        demande = self._demande_oui_non()
        self.assertIsNone(dem.option_choisie("peut-être", demande))
        self.assertIsNone(dem.option_choisie("je ne sais pas", demande))

    def test_question_non_binaire_ignoree(self):
        demande = self._demande_oui_non()
        for option in demande["options"]:
            option["libelle"] = {"o1": "Le matin", "o2": "Le soir"}.get(
                option["id"], option["libelle"])
        self.assertIsNone(dem.option_choisie("oui", demande))

    def test_trois_options_ignoree(self):
        demande = self._demande_oui_non()
        demande["options"].append({"id": "o3", "libelle": "Peut-être",
                                   "valeur": "Peut-etre.", "effet": None})
        self.assertIsNone(dem.option_choisie("oui", demande))

    def test_motif_garde_oui_libre_inchange(self):
        demande = self._demande_oui_non()
        demande["motif"] = "destructif"
        self.assertIsNone(dem.option_choisie("oui", demande))
