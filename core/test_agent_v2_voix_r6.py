"""
Round 6, la voix. Chaque classe a ete ecrite AVANT son correctif et vue en
echec.

D2  une demande abandonnee par le code se rend en UNE ligne (« Je laisse
    tomber la suppression de X. Redis-le si tu veux toujours. ») et n'est
    jamais reposee ni persistee.
D4  le brouillon d'AGIR n'entre au brief que si CHAQUE proposition, la
    derniere comprise, est une question, une offre ou une conditionnelle.
D5  PROSE_REPRISE seulement quand aucune mutation n'a reussi ce tour; sinon
    les faits, puis la question du code.
D6  un tour entierement decide par le code (reponse ou non-reponse a une
    demande, sans nouvelle requete) ne lance ni AGIR ni DIRE.
D7  l'absence ne tombe que si elle contredit un titre de la liste affichee;
    la mecanique vue au banc du round 5 tombe, les phrases ordinaires restent.

Contrat partage avec le correcteur des gardes: ToolResult.data porte
'demande', 'reposee_par_le_code', 'abandonnee_par_le_code', 'decision_code';
outils.tour_entierement_decide_par_le_code(registre, message) -> bool.
"""
import logging
import time
from unittest.mock import patch

from django.test import SimpleTestCase

from core.test_agent_v2_narrateur import (QUESTION_PORTEE, NarrateurBase, demande, ok)
from services.agent.tools.base import ToolResult
from services.agent_v2 import agent as module_agent
from services.agent_v2 import redaction, rendu
from services.agent_v2.agent import PROSE_REPRISE, PlannerAgentV2
from services.agent_v2.mesure import questions_et_offres
from services.agent_v2.redaction import ReponseDire, composer
from services.agent_v2.registre import Registre

LIGNE_ABANDON = "Je laisse tomber la suppression de Quart au dépanneur. Redis-le si tu veux toujours."


def reposee(dem):
    return ToolResult(success=False, message="retenue",
                      data={"demande": dem, "reposee_par_le_code": True,
                            "needs_confirmation": True, "decision_code": "reposee"})


def abandonnee(dem):
    return ToolResult(success=False, message="abandonnee",
                      data={"demande": dem, "abandonnee_par_le_code": True,
                            "decision_code": "abandonnee"})


def executee():
    return ToolResult(success=True, message="ok",
                      data={"par_le_code": True, "decision_code": "execute"})


# ── D2: la ligne d'abandon ─────────────────────────────────────────────────


class D2LigneAbandonRenduTests(SimpleTestCase):

    def _faits(self, dem, outil="delete_block", cles_posees=None):
        registre = Registre()
        registre.ajouter(outil, dict(dem.get("parametres") or {}), abandonnee(dem))
        return rendu.rendre_faits(registre, cles_posees=cles_posees)

    def test_portee_rend_une_seule_ligne(self):
        faits = self._faits(demande("portee_jour", "p1"))
        self.assertEqual(faits, LIGNE_ABANDON)
        self.assertNotIn("pas encore", faits)

    def test_la_ligne_part_meme_si_la_question_du_tour_couvre_la_cle(self):
        self.assertEqual(self._faits(demande("portee_jour", "p1"), cles_posees={"p1"}),
                         LIGNE_ABANDON)

    def test_la_ligne_part_meme_sans_outil_de_mutation(self):
        self.assertEqual(self._faits(demande("portee_jour", "p1"), outil="demande"),
                         LIGNE_ABANDON)

    def test_une_ligne_par_motif(self):
        cas = (
            (demande("destructif", "d1", outil="delete_block"), LIGNE_ABANDON),
            (demande("destructif", "d2", outil="clear_all_blocks", cible={}),
             "Je laisse tomber le vidage de ton planning. Redis-le si tu veux toujours."),
            (demande("destructif", "d3", outil="delete_task", cible={"titre": "Rapport"}),
             "Je laisse tomber la suppression de la tâche Rapport. Redis-le si tu veux toujours."),
            (demande("destructif", "d4", outil="cancel_scheduled_block",
                     cible={"titre": "Dentiste"}),
             "Je laisse tomber l'annulation de Dentiste. Redis-le si tu veux toujours."),
            (demande("destructif", "d5", outil="update_block", cible={"titre": "Gym"}),
             "Je laisse tomber l'arrêt de Gym. Redis-le si tu veux toujours."),
            (demande("optimisation", "o1", outil="optimize_week", cible={}),
             "Je laisse tomber le nouveau plan de ta semaine. Redis-le si tu veux toujours."),
            (demande("creation_en_masse", "m1", outil="create_block", cible={}),
             "Je laisse tomber le reste des ajouts. Redis-le si tu veux toujours."),
        )
        for dem, attendu in cas:
            with self.subTest(outil=dem["outil"], motif=dem["motif"]):
                self.assertEqual(self._faits(dem, outil=dem["outil"]), attendu)

    def test_sans_tiret_long(self):
        self.assertNotIn(chr(0x2014), self._faits(demande("portee_jour", "p1")))


class D2AbandonAuTourTests(NarrateurBase):

    def setUp(self):
        super().setUp()
        # Le vrai rendu: la ligne d'abandon est ecrite par rendu.py.
        patcheur = patch.object(redaction, "_charger_rendu", return_value=rendu)
        patcheur.start()
        self.addCleanup(patcheur.stop)

    def test_la_demande_abandonnee_n_est_ni_reposee_ni_persistee(self):
        _, done = self.tour(actions=[("delete_block", {"block_id": 5},
                                      abandonnee(demande("portee_jour", "p1")))],
                            message="ajoute gym demain à 18 h",
                            dire=ReponseDire(ouverture="Ok."))
        self.assertIn(LIGNE_ABANDON, done["response"])
        self.assertNotEqual(done["question_motif"], "portee_jour")
        self.assertEqual(self.metadonnees()["demandes"], [])
        self.assertEqual(done["response"].count("laisse tomber"), 1)


# ── D4: le brouillon d'AGIR, proposition par proposition ────────────────────


class D4BrouillonStructurelTests(SimpleTestCase):

    def test_une_derniere_proposition_declarative_emporte_la_phrase(self):
        for brouillon in ("Tu veux un rappel, ton gym est jeudi à 9 h ?",
                          "Veux-tu autre chose, ton cours reste à 14 h ?",
                          "Quel jour te va, ta révision attend vendredi ?"):
            with self.subTest(brouillon=brouillon):
                self.assertEqual(questions_et_offres(brouillon), "")

    def test_chaque_proposition_qui_demande_passe(self):
        for brouillon in ("Veux-tu que je le déplace, ou tu préfères vendredi ?",
                          "Quel jour te va le mieux ?",
                          "Dis-moi l'heure et je le place.",
                          "Tu veux enlever ton cours seulement ce jeudi ou tous les jeudis ?"):
            with self.subTest(brouillon=brouillon):
                self.assertEqual(questions_et_offres(brouillon), brouillon)


# ── D5 et D6: le tour de reprise et le chemin rapide ───────────────────────


class _TourDecideBase(NarrateurBase):
    """appliquer_choix_en_attente simule: il inscrit au registre ce que le
    correcteur des gardes y inscrit, et rend ses resumes."""

    def setUp(self):
        super().setUp()
        self.inscrire: list = []
        self.decide = False
        patcheur = patch.object(module_agent, "_charger_tour_decide",
                                return_value=lambda registre, message: self.decide,
                                create=True)
        patcheur.start()
        self.addCleanup(patcheur.stop)

    def _appliquer(self, *a, **k):
        self.ordre.append("appliquer")
        registre = a[1]
        for outil, params, res in self.inscrire:
            registre.ajouter(outil, params, res)
        return list(self.choix)


class D5ProseDeRepriseTests(_TourDecideBase):

    def test_reprise_sans_mutation_parle_seule(self):
        self.inscrire = [("delete_block", {"block_id": 5}, reposee(demande("portee_jour", "p1")))]
        _, done = self.tour(message="ouais",
                            dire=ReponseDire(ouverture="D'accord, je garde ton quart."))
        self.assertEqual(done["response"], f"{PROSE_REPRISE}\n\n{QUESTION_PORTEE}")
        self.assertEqual([d["cle"] for d in self.metadonnees()["demandes"]], ["p1"])

    def test_reprise_avec_mutation_montre_les_faits_puis_la_question(self):
        self.inscrire = [("delete_block", {"block_id": 9}, executee()),
                         ("delete_block", {"block_id": 5}, reposee(demande("portee_jour", "p1")))]
        _, done = self.tour(message="ouais",
                            dire=ReponseDire(ouverture="D'accord, je garde ton quart."))
        self.assertNotIn(PROSE_REPRISE, done["response"])
        self.assertNotIn("je garde", done["response"])
        self.assertEqual(done["response"], f"FAITS\n\n{QUESTION_PORTEE}")


class D6CheminRapideTests(_TourDecideBase):

    def setUp(self):
        super().setUp()
        self.decide = True

    def _interdits(self):
        def _agir(*a, **k):
            raise AssertionError("AGIR ne doit pas tourner")

        def _dire(*a, **k):
            raise AssertionError("DIRE ne doit pas tourner")
        return _agir, _dire

    def _tour_rapide(self, message):
        agir, dire = self._interdits()
        depart = time.perf_counter()
        with self.assertLogs("services.agent_v2.agent", level="INFO") as journal:
            _, done = self.tour(message=message, agir=agir, dire_effet=dire)
        self.duree = time.perf_counter() - depart
        self.assertTrue(any("chemin=code" in l for l in journal.output), journal.output)
        return done

    def test_execute_rend_les_faits_seuls(self):
        self.inscrire = [("delete_block", {"block_id": 5}, executee())]
        done = self._tour_rapide("Tous les jeudis (supprimer la série).")
        self.assertEqual(done["response"], "FAITS")
        self.assertFalse(done["question_posee"])
        self.assertLess(self.duree, 5.0)

    def test_reposee_rend_la_reprise_et_la_question(self):
        self.inscrire = [("delete_block", {"block_id": 5}, reposee(demande("portee_jour", "p1")))]
        done = self._tour_rapide("ouais")
        self.assertEqual(done["response"], f"{PROSE_REPRISE}\n\n{QUESTION_PORTEE}")
        self.assertEqual(done["question_motif"], "portee_jour")
        self.assertEqual([d["cle"] for d in self.metadonnees()["demandes"]], ["p1"])

    def test_abandonnee_rend_une_ligne(self):
        with patch.object(redaction, "_charger_rendu", return_value=rendu):
            self.inscrire = [("delete_block", {"block_id": 5},
                              abandonnee(demande("portee_jour", "p1")))]
            done = self._tour_rapide("bof")
        self.assertEqual(done["response"], LIGNE_ABANDON)
        self.assertEqual(self.metadonnees()["demandes"], [])

    def test_annulee_rend_une_ligne(self):
        self.choix = [{"cle": "p1", "motif": "portee_jour", "option": "annuler",
                       "action_id": None, "decision_code": "annulee",
                       "resume": "REFUSE PAR L'UTILISATEUR"}]
        done = self._tour_rapide("laisse faire")
        self.assertEqual(done["response"], module_agent.PROSE_ANNULEE)
        self.assertFalse(done["question_posee"])

    def test_sans_decision_du_code_agir_tourne(self):
        self.decide = False
        _, done = self.tour(message="ajoute gym demain")
        self.assertIn("agir", self.ordre)


class D6ChargeurTests(SimpleTestCase):

    def test_sans_la_fonction_des_gardes_le_tour_n_est_pas_decide(self):
        with patch("services.agent_v2.outils.tour_entierement_decide_par_le_code",
                   None, create=True):
            self.assertFalse(PlannerAgentV2._tour_decide(Registre(), "ouais"))

    def test_une_panne_de_la_fonction_ne_decide_rien(self):
        def _panne(registre, message):
            raise RuntimeError("x")
        with patch.object(module_agent, "_charger_tour_decide", return_value=_panne):
            self.assertFalse(PlannerAgentV2._tour_decide(Registre(), "ouais"))


# ── D7: absence contredite et mecanique ────────────────────────────────────


def _lecture():
    registre = Registre()
    registre.ajouter("list_blocks", {}, ToolResult(
        success=True, message="ok",
        data={"blocks": [{"title": "Calcul différentiel", "start_time": "09:00",
                          "end_time": "12:00", "day_of_week": 0}]}))
    return registre


FAITS_LUS = "Lundi\n- 9 h à 12 h · Calcul différentiel"


class D7AbsenceTests(SimpleTestCase):

    def _prose(self, phrase, faits=FAITS_LUS):
        return composer(ReponseDire(ouverture=phrase), _lecture(), faits, None).prose

    def test_l_absence_d_un_titre_affiche_tombe(self):
        for phrase in ("Tu n'as pas de Calcul différentiel lundi.",
                       "Il n'y a pas de calcul differentiel dans ton horaire.",
                       "Calcul différentiel n'apparaît pas lundi."):
            with self.subTest(phrase=phrase):
                self.assertEqual(self._prose(phrase), "")

    def test_une_absence_legitime_survit(self):
        for phrase in ("Tu n'as pas d'examen lundi.",
                       "Il n'y a pas de cours de maths dans ton horaire.",
                       "Tu n'as rien après midi."):
            with self.subTest(phrase=phrase):
                self.assertEqual(self._prose(phrase), phrase)


MECANIQUE_R5 = (
    "Précise tes heures et tes jours dans tes réponses.",
    "Choisis tes trois jours et l'heure.",
    "Ton mardi est déjà affiché.",
    "Je n'ai pas tenu compte de ton sommeil dans la liste, mais il commence à 23 h.",
    "Indique l'étendue pour que je puisse tout retirer d'un coup.",
    "Le Stage prend 10 h à 12 h, donc choisis parmi les moments libres proposés.",
)
ORDINAIRES = (
    "Ta journée est bien remplie.",
    "Si tu choisis le matin, tu seras plus frais.",
    "La liste de tes cours est longue cette session.",
    "Ton cours de maths commence à 9 h.",
    "Il te reste une place libre jeudi soir.",
    "Bonne répartition sur tes quatre journées.",
)


class D7MecaniqueTests(SimpleTestCase):

    def test_la_mecanique_du_banc_r5_tombe(self):
        for phrase in MECANIQUE_R5:
            with self.subTest(phrase=phrase):
                compo = composer(ReponseDire(ouverture=phrase), Registre(), "", None)
                self.assertEqual(compo.prose, "")

    def test_les_phrases_ordinaires_restent(self):
        for phrase in ORDINAIRES:
            with self.subTest(phrase=phrase):
                compo = composer(ReponseDire(ouverture=phrase), Registre(), "", None)
                self.assertEqual(compo.prose, phrase)
