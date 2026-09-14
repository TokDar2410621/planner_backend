"""
Un seul narrateur, une seule question, un ordre stable (lots 1c, 2a, 3b, 3e, 3g).

Tests verts sur la branche seule: rendu.py (lot b4), question_forcee (b5) et
appliquer_choix_en_attente (b3) sont remplaces par des faux via les chargeurs
de agent.py et redaction.py. Les memes flux contre les vrais modules vivent
dans test_agent_v2_narrateur_integration.py.
"""
import asyncio
import queue
import re
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase
from django.utils import timezone
from pydantic_ai.messages import (ModelResponse, PartDeltaEvent, PartStartEvent,
                                  TextPart, TextPartDelta, ThinkingPart,
                                  ThinkingPartDelta)

from core.models import ConversationMessage, RecurringBlock, UploadedDocument
from services.agent.tools.base import ToolResult
from services.agent_v2 import agent as module_agent
from services.agent_v2 import redaction
from services.agent_v2.agent import PRIORITE, PlannerAgentV2
from services.agent_v2.redaction import ActionCitee, ReponseDire, composer
from services.agent_v2.registre import Registre

QUESTION_PORTEE = "Seulement ce jeudi ou tous les jeudis ?"
CHIPS_PORTEE = [
    {"label": "Seulement ce jeudi",
     "value": "Seulement ce jeudi jeu. 17 sept. (sauter l'occurrence).", "option": "occurrence"},
    {"label": "Tous les jeudis", "value": "Tous les jeudis (supprimer la série).", "option": "serie"},
    {"label": "Non, garde tout", "value": "Non, ne change rien.", "option": "annuler"},
]
BROUILLON = "Tu veux que je supprime toute la série, ou seulement ce jeudi ?"


def faux_rendu(journal: dict) -> SimpleNamespace:
    def rendre_faits(registre, aujourdhui=None, cles_posees=None):
        journal.setdefault("cles_posees", []).append(cles_posees)
        return "FAITS" if any(a.succes and a.est_mutation for a in registre.actions) else ""

    def rendre_lecture(registre, aujourdhui=None):
        return ""

    def rendre_demandes(demandes, aujourdhui=None):
        if not demandes:
            return "", [], []
        haut = min(demandes, key=lambda d: PRIORITE.index(d["motif"]))["motif"]
        retenues = [d for d in demandes if d["motif"] == haut]
        cles = list(dict.fromkeys(d["cle"] for d in retenues))
        if haut == "choix_modele":
            d = retenues[0]
            return d["question"], [
                {"label": o["libelle"], "value": o["valeur"], "option": o["id"]}
                for o in d["options"]], cles
        return QUESTION_PORTEE, [dict(c) for c in CHIPS_PORTEE], cles

    return SimpleNamespace(rendre_faits=rendre_faits, rendre_lecture=rendre_lecture,
                           rendre_demandes=rendre_demandes, marqueurs_bruts=lambda t: [])


def demande(motif, cle, outil="delete_block", **extra):
    base = {
        "type": "confirmation", "motif": motif, "cle": cle, "outil": outil,
        "parametres": {"block_id": 5},
        "cible": {"titre": "Quart au dépanneur", "jour": 3, "date": "2026-09-17"},
        "options": [{"id": "occurrence", "effet": None, "cible": {}},
                    {"id": "serie", "effet": None, "cible": {}},
                    {"id": "annuler", "effet": None, "cible": {}}],
        "emise_le": "2026-09-14T12:00:00+00:00",
    }
    base.update(extra)
    return base


def ok(outil, params=None, **data):
    return (outil, params or {}, ToolResult(success=True, message="ok", data=data))


def refus(outil, params=None, **data):
    return (outil, params or {}, ToolResult(
        success=False, message="Action retenue par le code: une question est posee a "
                               "l'utilisateur. N'agis pas sur ce point et ne repose pas "
                               "la question.",
        data={"needs_confirmation": True, **data}))


def formulaire():
    return ok("present_form", interactive_inputs=[
        {"id": "heure", "type": "time", "label": "Heure", "question": "À quelle heure ?"}])


class NarrateurBase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="narrateur", password="x")
        self.journal: dict = {}
        self.choix: list = []
        self.ordre: list = []
        self.forcee = None
        for cible, nom, valeur in (
            (redaction, "_charger_rendu", faux_rendu(self.journal)),
            (module_agent, "_charger_question_forcee", self._question_forcee),
            (module_agent, "_charger_appliquer_choix", self._appliquer),
        ):
            patcher = patch.object(cible, nom, return_value=valeur)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _question_forcee(self, *a, **k):
        return self.forcee

    def _appliquer(self, *a, **k):
        self.ordre.append("appliquer")
        self.journal["appliquer"] = (a, k)
        return list(self.choix)

    def tour(self, actions=(), dire=None, message="bonjour", agir=None, dire_effet=None):
        vus = {}

        def _agir(self_agent, user, msg, registre):
            self.ordre.append("agir")
            vus["message_agir"] = msg
            for outil, params, res in actions:
                self_agent.signaler_outil(registre.ajouter(outil, params, res))
            return ""

        options = ({"side_effect": dire_effet} if dire_effet is not None else
                   {"return_value": dire if dire is not None else ReponseDire(ouverture="Ok.")})
        with patch.object(PlannerAgentV2, "_agir", agir or _agir), \
             patch.object(PlannerAgentV2, "_dire", **options):
            evts = list(PlannerAgentV2().process_message_stream(self.user, message))
        self.vus = vus
        return evts, evts[-1]

    def metadonnees(self):
        return ConversationMessage.objects.filter(
            user=self.user, role="assistant").latest("pk").metadata

    def dernier_message_utilisateur(self):
        return ConversationMessage.objects.filter(user=self.user, role="user").latest("pk")

    def deltas(self, evts):
        return [e["text"] for e in evts if e["type"] == "delta"]


# ── 1c: le volet de raisonnement ─────────────────────────────────────────


class VoletDeRaisonnementTests(SimpleTestCase):
    def _pousser(self, evenements):
        agent = PlannerAgentV2()
        agent._file_pensees = queue.Queue()

        async def flux():
            for e in evenements:
                yield e

        asyncio.run(agent._sur_evenements(None, flux()))
        sortie = []
        while not agent._file_pensees.empty():
            sortie.append(agent._file_pensees.get_nowait())
        return sortie

    def test_seul_le_raisonnement_part_au_volet(self):
        sortie = self._pousser([
            PartStartEvent(index=0, part=ThinkingPart(content="The user")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=" wants")),
            PartStartEvent(index=1, part=TextPart(content="Voici")),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=" ta semaine")),
        ])
        self.assertEqual([t for _, t in sortie], ["The user", " wants"])
        self.assertEqual({g for g, _ in sortie}, {"thinking"})

    def test_un_delta_de_signature_sans_texte_n_emet_rien(self):
        sortie = self._pousser([
            PartStartEvent(index=0, part=ThinkingPart(content="")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(signature_delta="sig")),
        ])
        self.assertEqual(sortie, [])


# ── Composition (regles b, c, e, f) ──────────────────────────────────────


class ComposerTests(SimpleTestCase):
    def _registre(self):
        r = Registre()
        r.ajouter("create_block", {}, ToolResult(success=True, message="ok"))
        return r

    def test_rejetee_efface_tout_ce_que_dire_a_ecrit(self):
        brut = ReponseDire(ouverture="Top !", suite="Tout roule.", question="Tu gardes ça ?",
                           options=["Oui", "Non"], refs=["a9"])
        compo = composer(brut, self._registre(), "FAITS", None)
        self.assertEqual((compo.prose, compo.question, compo.chips, compo.rejetees),
                         ("", "", [], 1))

    def test_une_action_citee_valide_n_est_jamais_rendue(self):
        brut = ReponseDire(ouverture="Bon.", actions=[ActionCitee(ref="a1", phrase="J'ai tout fait.")])
        compo = composer(brut, self._registre(), "FAITS", None)
        self.assertEqual(compo.texte, "FAITS\n\nBon.")

    def test_options_seulement_avec_question_et_entre_deux_et_quatre(self):
        r = Registre()
        sans_question = composer(ReponseDire(options=["19 h", "22 h"]), r, "", None)
        self.assertEqual(sans_question.chips, [])
        cinq = composer(ReponseDire(question="Quelle heure ?",
                                    options=["7 h", "8 h", "9 h", "10 h", "11 h"]), r, "", None)
        self.assertEqual([c["label"] for c in cinq.chips], ["7 h", "8 h", "9 h", "10 h"])
        doublons = composer(ReponseDire(question="Quelle heure ?",
                                        options=["7 h", "7 h", " "]), r, "", None)
        self.assertEqual(doublons.chips, [])

    def test_sans_faits_les_annonces_vides_tombent(self):
        brut = ReponseDire(ouverture="Tu as 3 créneaux libres, que voici.",
                           suite="Je peux ajouter un bloc si tu veux.")
        compo = composer(brut, Registre(), "", None)
        self.assertNotIn("voici", compo.prose)
        self.assertEqual(compo.prose, "Je peux ajouter un bloc si tu veux.")
        self.assertTrue(compo.lecture_sans_liste)

    def test_avec_faits_l_annonce_reste(self):
        brut = ReponseDire(ouverture="Voici ta semaine.")
        compo = composer(brut, Registre(), "**Lundi**", None)
        self.assertEqual(compo.prose, "Voici ta semaine.")
        self.assertFalse(compo.lecture_sans_liste)

    def test_un_compte_en_mots_est_aussi_un_compte(self):
        compo = composer(ReponseDire(ouverture="Tu as deux blocs demain."), Registre(), "", None)
        self.assertEqual(compo.prose, "")

    def test_la_question_du_code_ecarte_celle_de_dire(self):
        code = {"question": "Seulement ce jeudi ?", "chips": [{"label": "a", "value": "b"}],
                "motif": "portee_jour", "demandes": [{"cle": "p1"}], "cles_posees": ["p1"]}
        compo = composer(ReponseDire(ouverture="Ok.", question="Autre chose ?",
                                     options=["Oui", "Non"]), Registre(), "", code)
        self.assertEqual((compo.question, compo.motif, compo.cles_posees),
                         ("Seulement ce jeudi ?", "portee_jour", ["p1"]))
        self.assertEqual(compo.chips, [{"label": "a", "value": "b"}])

    def test_sans_sortie_de_dire_la_question_du_code_reste(self):
        code = {"question": "Tu confirmes ?", "chips": [], "motif": "destructif"}
        compo = composer(None, Registre(), "FAITS", code)
        self.assertEqual(compo.texte, "FAITS\n\nTu confirmes ?")


# ── 1d-branchement: message brut, choix executes avant AGIR ──────────────


class AgentFactice:
    sortie = ""

    def __init__(self, *a, **kw):
        pass

    def run_sync(self, *a, **kw):
        return SimpleNamespace(output=AgentFactice.sortie, all_messages=lambda: [],
                               usage=lambda: None)


class MessageBrutTests(NarrateurBase):
    def _import_recent(self):
        doc = UploadedDocument.objects.create(
            user=self.user, file_name="recent.pdf", document_type="course_schedule",
            processed=True, extracted_data={"courses": [{"name": "Anglais"}]})
        RecurringBlock.objects.create(
            user=self.user, title="Anglais", block_type="course", day_of_week=3,
            start_time="14:00", end_time="16:00", source_document=doc)

    def _vrai_agir(self, outils_pour, sortie="", dire=None):
        AgentFactice.sortie = sortie
        with patch.object(module_agent, "Agent", AgentFactice), \
             patch.object(module_agent, "modele_agir", return_value=None), \
             patch.object(module_agent, "prompt_agir", return_value=""), \
             patch.object(module_agent, "outils_pour", outils_pour), \
             patch.object(PlannerAgentV2, "_historique", return_value=[]), \
             patch.object(PlannerAgentV2, "_dire", **(dire or {"return_value": ReponseDire(ouverture="Ok.")})):
            return PlannerAgentV2().process_message(self.user, "c'est bon ?")

    def test_le_guard_recoit_le_message_brut(self):
        self._import_recent()
        outils_pour = MagicMock(return_value=[])
        self._vrai_agir(outils_pour)
        kwargs = outils_pour.call_args.kwargs
        self.assertEqual(kwargs["message_brut"], "c'est bon ?")
        self.assertTrue(kwargs["message_du_tour"].startswith("c'est bon ?\n\n"))
        self.assertNotEqual(kwargs["message_du_tour"], kwargs["message_brut"])

    def test_le_texte_final_d_agir_devient_le_brouillon(self):
        vus = {}

        def _dire(user, message, registre, etat, faits, **kw):
            vus.update(kw)
            return ReponseDire(ouverture="Ok.")

        self._vrai_agir(MagicMock(return_value=[]), sortie=f"  {BROUILLON}  ",
                        dire={"side_effect": _dire})
        self.assertEqual(vus["brouillon"], BROUILLON)

    def test_choix_du_code_avant_agir(self):
        resume = "FAIT PAR LE CODE (a1): delete_block Quart"
        self.choix = [{"cle": "k", "motif": "portee_jour", "option": "serie",
                       "action_id": "a1", "resume": resume}]
        vus = {}

        def _dire(user, message, registre, etat, faits, **kw):
            vus["message_dire"] = message
            return ReponseDire(ouverture="Ok.")

        with self.assertLogs("services.agent_v2.agent", level="INFO") as logs:
            self.tour(message="Tous les jeudis (supprimer la série).", dire_effet=_dire)
        self.assertEqual(self.ordre, ["appliquer", "agir"])
        self.assertIn("SUITE AU CHOIX", self.vus["message_agir"])
        self.assertIn(resume, self.vus["message_agir"])
        self.assertEqual(vus["message_dire"], "Tous les jeudis (supprimer la série).")
        args, kwargs = self.journal["appliquer"]
        self.assertEqual(args[0], self.user)
        self.assertEqual(args[2], "Tous les jeudis (supprimer la série).")
        self.assertEqual(kwargs["tache"], f"{self.user.pk}:{self.dernier_message_utilisateur().pk}")
        self.assertTrue(any("choix_code=1" in ligne for ligne in logs.output), logs.output)

    def test_un_refus_de_l_utilisateur_n_est_pas_compte_comme_execute(self):
        self.choix = [{"cle": "k", "motif": "destructif", "option": "annuler",
                       "action_id": None, "resume": "REFUSE PAR L'UTILISATEUR: clear_all_blocks"}]
        with self.assertLogs("services.agent_v2.agent", level="INFO") as logs:
            self.tour(message="Non, ne change rien.")
        self.assertIn("REFUSE", self.vus["message_agir"])
        self.assertTrue(any("choix_code=0" in ligne for ligne in logs.output), logs.output)

    def test_l_effet_execute_par_le_code_est_trace_et_persiste(self):
        def appliquer(user, registre, message, tache="", signaler=None):
            action = registre.ajouter("delete_block", {"block_id": 5}, ToolResult(
                success=True, message="ok",
                data={"cle_demande": "k", "par_le_code": True}))
            signaler(action)
            return [{"cle": "k", "motif": "portee_jour", "option": "serie",
                     "action_id": action.id, "resume": "FAIT PAR LE CODE (a1)"}]

        with patch.object(module_agent, "_charger_appliquer_choix", return_value=appliquer):
            evts, done = self.tour(message="Tous les jeudis (supprimer la série).")
        types = [e["type"] for e in evts]
        outil = next(e for e in evts if e["type"] == "tool")
        self.assertEqual(outil["name"], "delete_block")
        self.assertLess(types.index("tool"), types.index("status"))
        self.assertEqual(self.metadonnees()["actions"],
                         [{"id": "a1", "outil": "delete_block", "succes": True, "par_le_code": True}])
        self.assertTrue(done["response"].startswith("FAITS"))

    def test_une_panne_du_choix_ne_fait_pas_tomber_le_tour(self):
        def casse(*a, **k):
            raise RuntimeError("base indisponible")

        with patch.object(module_agent, "_charger_appliquer_choix", return_value=casse):
            _, done = self.tour()
        self.assertEqual(done["response"], "Ok.")


# ── 2a: DIRE voit le brouillon, pose sa question ─────────────────────────


class BriefDeDireTests(NarrateurBase):
    def _brief(self, actions=(), agir=None, message="bonjour"):
        vus = {}

        def _dire(user, message, registre, etat, faits, **kw):
            vus["brief"] = PlannerAgentV2._brief_dire(message, registre, etat, faits, **kw)
            return ReponseDire(ouverture="Ok.")

        self.tour(actions=actions, agir=agir, message=message, dire_effet=_dire)
        return vus["brief"]

    def test_le_brouillon_d_agir_atteint_dire(self):
        def _agir(self_agent, user, message, registre):
            self_agent._brouillon_agir = BROUILLON
            return ""

        brief = self._brief(agir=_agir)
        self.assertIn("BROUILLON D'AGIR", brief)
        self.assertIn(BROUILLON, brief)

    def test_la_question_du_code_est_annoncee_a_dire(self):
        brief = self._brief(actions=[refus("delete_block", demande=demande("portee_jour", "p1"))])
        self.assertIn("QUESTION DEJA POSEE PAR LE CODE", brief)
        self.assertIn(QUESTION_PORTEE, brief)

    def test_les_deux_derniers_echanges_arrivent(self):
        ConversationMessage.objects.create(user=self.user, role="user", content="ajoute mon quart")
        ConversationMessage.objects.create(user=self.user, role="assistant",
                                           content="À quelle heure commence ton quart ?")
        brief = self._brief(message="19 h")
        self.assertIn("DEUX DERNIERS ECHANGES", brief)
        self.assertIn("Assistant: À quelle heure commence ton quart ?", brief)
        self.assertNotIn("Utilisateur: 19 h", brief)

    def test_import_recent_en_contexte(self):
        actions = [("import_recent", {"document": "recent.pdf"}, ToolResult(
            success=True, message="Horaire importé depuis « recent.pdf » : 1 entrée ajoutée",
            data={"fichier": "recent.pdf", "blocs": [{"titre": "Anglais"}]}))]
        brief = self._brief(actions=actions)
        self.assertIn("CONTEXTE (ne pas citer)", brief)
        _, done = self.tour(actions=actions)
        self.assertNotIn("import", done["response"].lower())


class QuestionDeDireTests(NarrateurBase):
    def test_question_de_dire_avec_options(self):
        _, done = self.tour(dire=ReponseDire(question="À quelle heure commence ton quart ?",
                                             options=["19 h", "22 h"]))
        self.assertTrue(done["response"].endswith("À quelle heure commence ton quart ?"))
        self.assertEqual(done["quick_replies"], [{"label": "19 h", "value": "19 h"},
                                                 {"label": "22 h", "value": "22 h"}])
        self.assertTrue(done["question_posee"])
        self.assertEqual(done["question_motif"], "dire")
        meta = self.metadonnees()
        self.assertTrue(meta["question_posee"])
        self.assertEqual(meta["en_reponse_a"], self.dernier_message_utilisateur().pk)
        self.assertEqual(meta["demandes"], [])

    def test_une_seule_option_pas_de_bouton(self):
        _, done = self.tour(dire=ReponseDire(question="À quelle heure ?", options=["19 h"]))
        self.assertEqual(done["quick_replies"], [])
        self.assertEqual(done["question"], "À quelle heure ?")

    def test_reference_inventee_coupe_tout(self):
        with self.assertLogs("services.agent_v2.agent", level="WARNING") as logs:
            _, done = self.tour(
                actions=[ok("create_block", created=[{"title": "Maths"}])],
                dire=ReponseDire(refs=["a9"], ouverture="Top !", suite="Tout roule.",
                                 question="Tu gardes ça ?", options=["Oui", "Non"]))
        self.assertEqual(done["response"], "FAITS")
        self.assertEqual(done["question"], "")
        self.assertEqual(done["quick_replies"], [])
        self.assertFalse(done["question_posee"])
        self.assertTrue(any("rejetees=1" in ligne for ligne in logs.output), logs.output)


# ── Une question par tour, selon PRIORITE ────────────────────────────────


class PrioriteTests(NarrateurBase):
    def test_la_garde_prime_sur_dire(self):
        d = demande("portee_jour", "p1")
        _, done = self.tour(actions=[refus("delete_block", demande=d)],
                            dire=ReponseDire(ouverture="D'accord.", question="Autre chose ?",
                                             options=["Oui", "Non"]))
        self.assertEqual(done["question"], QUESTION_PORTEE)
        self.assertEqual(done["question_motif"], "portee_jour")
        self.assertEqual(done["quick_replies"],
                         [{"label": c["label"], "value": c["value"]} for c in CHIPS_PORTEE])
        self.assertNotIn("Autre chose", done["response"])
        self.assertTrue(done["response"].endswith(QUESTION_PORTEE))
        meta = self.metadonnees()
        self.assertEqual(len(meta["demandes"]), 1)
        self.assertEqual(meta["demandes"][0]["cle"], "p1")
        self.assertEqual(meta["demandes"][0]["emise_le"], d["emise_le"])
        self.assertEqual(meta["demandes"][0]["chips"], CHIPS_PORTEE)
        self.assertEqual(meta["quick_replies"], done["quick_replies"])

    def test_seules_les_demandes_posees_sont_persistees(self):
        _, done = self.tour(actions=[
            refus("delete_block", demande=demande("portee_jour", "p1")),
            refus("clear_all_blocks", demande=demande("destructif", "d1", outil="clear_all_blocks")),
        ])
        self.assertEqual([d["cle"] for d in self.metadonnees()["demandes"]], ["p1"])
        self.assertEqual(self.journal["cles_posees"][-1], {"p1"})

    def test_sans_question_de_demande_les_retenues_gardent_leur_ligne(self):
        """Un formulaire gagne sur un chevauchement: bloc_factuel recoit un
        ensemble vide, donc chaque retenue aura sa ligne « pas encore »."""
        _, done = self.tour(actions=[
            formulaire(),
            ("create_block", {}, ToolResult(success=True, message="ok", data={
                "created": [], "demande": demande("chevauchement", "c1")})),
        ])
        self.assertEqual(done["question_motif"], "formulaire")
        self.assertEqual(self.journal["cles_posees"][-1], set())
        self.assertEqual(self.metadonnees()["demandes"], [])

    def test_choix_du_modele_relaye(self):
        d = {"type": "choix", "motif": "choix_modele", "cle": "choix:abc", "outil": "present_choices",
             "question": "Lequel de tes cours ?", "source": "blocs",
             "options": [{"id": "o1", "effet": None, "libelle": "Calcul différentiel",
                          "valeur": "Le cours de Calcul différentiel"},
                         {"id": "o2", "effet": None, "libelle": "Physique mécanique",
                          "valeur": "Le cours de Physique mécanique"}],
             "emise_le": "2026-09-14T12:00:00+00:00"}
        _, done = self.tour(actions=[ok("present_choices", demande=d)])
        self.assertEqual(done["question_motif"], "choix_modele")
        self.assertEqual(done["question"], "Lequel de tes cours ?")
        self.assertEqual(done["quick_replies"], [
            {"label": "Calcul différentiel", "value": "Le cours de Calcul différentiel"},
            {"label": "Physique mécanique", "value": "Le cours de Physique mécanique"}])

    def test_formulaire_et_garde(self):
        with self.subTest("garde + formulaire"):
            _, done = self.tour(actions=[
                formulaire(), refus("delete_block", demande=demande("portee_jour", "p1"))])
            self.assertNotIn("interactive_inputs", done)
            self.assertEqual(done["question_motif"], "portee_jour")
        with self.subTest("formulaire seul"):
            _, done = self.tour(actions=[formulaire()],
                                dire=ReponseDire(question="Autre chose ?", options=["a", "b"]))
            self.assertEqual(done["interactive_inputs"][0]["id"], "heure")
            self.assertTrue(done["question_posee"])
            self.assertEqual(done["question_motif"], "formulaire")
            self.assertEqual(done["quick_replies"], [])
            self.assertEqual(done["question"], "")
            self.assertEqual(self.metadonnees()["interactive_inputs"][0]["id"], "heure")

    def test_le_formulaire_passe_avant_le_choix_du_modele(self):
        d = {"type": "choix", "motif": "choix_modele", "cle": "choix:x", "question": "Lequel ?",
             "options": [{"id": "o1", "libelle": "A", "valeur": "a"},
                         {"id": "o2", "libelle": "B", "valeur": "b"}]}
        _, done = self.tour(actions=[ok("present_choices", demande=d), formulaire()])
        self.assertEqual(done["question_motif"], "formulaire")

    def test_la_question_forcee_passe_apres_les_demandes(self):
        self.forcee = {"question": "Jusqu'à quand veux-tu le garder ?",
                       "chips": [{"label": "Je te donne la date", "value": "Je te donne la date."},
                                 {"label": "Pas de fin", "value": "Pas de fin."}],
                       "motif": "fin_recurrence"}
        with self.subTest("seule"):
            _, done = self.tour(dire=ReponseDire(ouverture="Reçu.", question="Autre chose ?",
                                                 options=["a", "b"]))
            self.assertEqual(done["question_motif"], "fin_recurrence")
            self.assertEqual(len(done["quick_replies"]), 2)
            self.assertEqual(done["response"], "Reçu.\n\nJusqu'à quand veux-tu le garder ?")
        with self.subTest("apres un chevauchement"):
            _, done = self.tour(actions=[("create_block", {}, ToolResult(
                success=True, message="ok",
                data={"created": [], "demande": demande("chevauchement", "c1")}))])
            self.assertEqual(done["question_motif"], "chevauchement")

    def test_la_question_forcee_recoit_le_message_brut(self):
        appels = []

        def forcee(*a, **k):
            appels.append(a)
            return None

        doc = UploadedDocument.objects.create(
            user=self.user, file_name="recent.pdf", document_type="course_schedule",
            processed=True, extracted_data={"courses": [{"name": "Anglais"}]})
        RecurringBlock.objects.create(
            user=self.user, title="Anglais", block_type="course", day_of_week=3,
            start_time="14:00", end_time="16:00", source_document=doc)
        with patch.object(module_agent, "_charger_question_forcee", return_value=forcee):
            self.tour(message="c'est bon ?")
        self.assertTrue(self.vus["message_agir"].startswith("c'est bon ?\n\n"))
        self.assertEqual(appels[0][1], "c'est bon ?")


# ── 3b, 3e: un seul narrateur, un ordre stable ───────────────────────────


class NarrateurUniqueTests(NarrateurBase):
    def test_un_seul_narrateur(self):
        gym = ok("update_block", {"block_id": 1}, block={"title": "Gym"})
        _, done = self.tour(
            actions=[gym, gym, gym],
            dire=ReponseDire(actions=[ActionCitee(ref="a1", phrase="J'ai mis à jour le bloc Gym.")]))
        self.assertNotIn("J'ai mis à jour le bloc Gym", done["response"])
        self.assertEqual(done["response"].count("FAITS"), 1)

    def test_ordre_stable(self):
        evts, done = self.tour(
            actions=[ok("create_block", created=[{"title": "Maths"}])],
            dire=ReponseDire(ouverture="Bonne soirée.", question="Autre chose à ajouter ?",
                             options=["Oui", "Non merci"]))
        deltas = self.deltas(evts)
        self.assertEqual("".join(deltas), done["response"])
        self.assertEqual(deltas[0], "FAITS")
        self.assertEqual(deltas, ["FAITS", "\n\nBonne soirée.", "\n\nAutre chose à ajouter ?"])
        self.assertEqual(evts[-1]["type"], "done")
        self.assertEqual([e["type"] for e in evts].count("done"), 1)

    def test_sans_faits_la_prose_ouvre_sans_saut_de_ligne(self):
        evts, done = self.tour(dire=ReponseDire(ouverture="Salut !"))
        self.assertEqual(self.deltas(evts), ["Salut !"])
        self.assertEqual(done["response"], "Salut !")

    def test_voici_sans_liste_retire(self):
        with self.assertLogs("services.agent_v2.agent", level="INFO") as logs:
            _, done = self.tour(
                actions=[ok("find_free_slots", {"date": "2026-09-17"},
                            free_slots=[{"start_time": "07:00", "end_time": "10:00"}])],
                message="quand suis-je libre jeudi ?",
                dire=ReponseDire(ouverture="Tu as 3 créneaux libres, que voici."))
        self.assertNotIn("voici", done["response"])
        self.assertTrue(any("read_without_list=1" in ligne for ligne in logs.output), logs.output)
        self.assertTrue(self.metadonnees()["lecture_sans_liste"])

    def test_repli_accentue(self):
        with self.subTest("avec faits"):
            evts, done = self.tour(actions=[ok("create_block", created=[{"title": "Maths"}])],
                                   dire_effet=RuntimeError("502"))
            self.assertTrue(done["response"].endswith(
                "Voici ce qui a changé. Dis-moi si tu veux autre chose."))
            self.assertEqual("".join(self.deltas(evts)), done["response"])
        with self.subTest("sans faits"):
            evts, done = self.tour(dire_effet=RuntimeError("502"))
            self.assertEqual(done["response"],
                             "Je n'ai pas compris. Tu veux ajouter, déplacer ou voir quelque chose ?")
            self.assertEqual(done["question_motif"], "dire")
            self.assertTrue(done["question_posee"])
            self.assertEqual(done["quick_replies"], [])
            self.assertEqual("".join(self.deltas(evts)), done["response"])

    def test_la_panne_de_dire_garde_la_question_du_code(self):
        _, done = self.tour(actions=[refus("delete_block", demande=demande("portee_jour", "p1"))],
                            dire_effet=RuntimeError("502"))
        self.assertEqual(done["response"], QUESTION_PORTEE)
        self.assertEqual(done["question_motif"], "portee_jour")

    def test_un_formulaire_seul_a_quand_meme_une_phrase(self):
        _, done = self.tour(actions=[formulaire()], dire=ReponseDire())
        self.assertEqual(done["response"], "Il me manque quelques précisions.")
        self.assertIn("interactive_inputs", done)


# ── 3g: mesure ───────────────────────────────────────────────────────────


class MesureTests(NarrateurBase):
    LIGNE = re.compile(
        r"agent_v2 tour actions=\d+ rejetees=\d+ fuites=\d+ supprimees=\d+ ecarts=\d+.*"
        r" asked=[01] form=[01] choices=\d read_without_list=[01] raw_marker_count=\d+"
        r" motif=\S+ choix_code=\d+")

    def test_metadonnees_et_compteurs(self):
        with self.assertLogs("services.agent_v2.agent", level="INFO") as logs:
            _, done = self.tour(dire=ReponseDire(question="À quelle heure ?", options=["7 h", "8 h"]))
        meta = self.metadonnees()
        for cle in ("agent", "en_reponse_a", "quick_replies", "interactive_inputs", "question_posee",
                    "question", "question_motif", "demandes", "faits_rendus", "raw_markers",
                    "lecture_sans_liste", "actions"):
            self.assertIn(cle, meta)
        self.assertEqual(meta["agent"], "v2")
        lignes = [ligne for ligne in logs.output if "agent_v2 tour" in ligne]
        self.assertEqual(len(lignes), 1)
        self.assertRegex(lignes[0], self.LIGNE)
        self.assertIn("asked=1 form=0 choices=2", lignes[0])
        self.assertIn("motif=dire", lignes[0])
        for cle in ("question_posee", "question", "question_motif"):
            self.assertIn(cle, done)

    def test_un_tour_sans_question_compte_zero(self):
        with self.assertLogs("services.agent_v2.agent", level="INFO") as logs:
            _, done = self.tour(dire=ReponseDire(ouverture="Salut !"))
        ligne = next(l for l in logs.output if "agent_v2 tour" in l)
        self.assertIn("asked=0 form=0 choices=0", ligne)
        self.assertIn("motif=- choix_code=0", ligne)
        self.assertFalse(done["question_posee"])
        self.assertEqual(done["question_motif"], "")

    def test_les_marqueurs_bruts_sont_comptes(self):
        journal = {}
        rendu = faux_rendu(journal)
        rendu.marqueurs_bruts = lambda texte: ["date_iso", "refus"]
        with patch.object(redaction, "_charger_rendu", return_value=rendu), \
             self.assertLogs("services.agent_v2.agent", level="INFO") as logs:
            self.tour(dire=ReponseDire(ouverture="Salut !"))
        self.assertEqual(self.metadonnees()["raw_markers"], ["date_iso", "refus"])
        self.assertTrue(any("raw_marker_count=2" in l for l in logs.output), logs.output)


class HistoriqueTests(NarrateurBase):
    def test_historique_porte_les_choix(self):
        maintenant = timezone.now()
        u = ConversationMessage.objects.create(user=self.user, role="user",
                                               content="quand suis-je libre ?")
        a = ConversationMessage.objects.create(
            user=self.user, role="assistant", content="Libre : 7 h à 8 h 30, 15 h 50 à 17 h 20",
            metadata={"quick_replies": [
                {"label": "7 h à 8 h 30", "value": "Va pour 7 h à 8 h 30."},
                {"label": "15 h 50 à 17 h 20", "value": "Va pour 15 h 50 à 17 h 20."}]})
        ConversationMessage.objects.filter(pk=u.pk).update(created_at=maintenant - timedelta(minutes=2))
        ConversationMessage.objects.filter(pk=a.pk).update(created_at=maintenant - timedelta(minutes=1))
        historique = PlannerAgentV2()._historique(self.user)
        self.assertIsInstance(historique[-1], ModelResponse)
        self.assertIn("[Choix proposés : 7 h à 8 h 30 | 15 h 50 à 17 h 20]",
                      historique[-1].parts[0].content)

    def test_sans_choix_le_texte_reste_tel_quel(self):
        ConversationMessage.objects.create(user=self.user, role="assistant", content="Salut !")
        historique = PlannerAgentV2()._historique(self.user)
        self.assertEqual(historique[-1].parts[0].content, "Salut !")
