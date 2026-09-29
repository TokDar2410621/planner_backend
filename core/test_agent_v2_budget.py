"""
Deadline globale du tour et budgets jetons (agent v2).

Verrouille les garde-fous ajoutes contre les deux risques documentes mais non
bornes: un tour sans deadline murale (397 s en prod le 2026-08-29) et une
facture LLM sans plafond.

Philosophie inchangee: le tour est tronque, jamais rate. Le registre survit,
les faits deja vrais sont rendus.

Aucun appel reseau ici: modeles simules, pool reel, delais courts.
"""
import queue
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.utils import timezone

from core.models import BudgetJetonsJournalier
from services.agent.tools.base import ToolResult
from services.agent_v2 import agent as module_agent
from services.agent_v2.agent import PlannerAgentV2
from services.agent_v2.redaction import ReponseDire
from services.agent_v2.registre import Registre


class _FauxResultat:
    output = ""

    def usage(self):
        raise RuntimeError("pas d'usage simule")


class _FauxAgent:
    """Capte les kwargs de run_sync pour verifier les UsageLimits."""

    def __init__(self, *args, **kwargs):
        self.run_kwargs = None

    def output_validator(self, func):
        return func

    def run_sync(self, *args, **kwargs):
        self.run_kwargs = kwargs
        return _FauxResultat()


def _agir_muet(self_agent, user, message, registre):
    return None


def _agir_qui_cree(self_agent, user, message, registre):
    registre.ajouter('create_block', {'title': 'Maths'},
                     ToolResult(success=True, message="Bloc 'Maths' cree",
                                data={'created': [{
                                    'title': 'Maths', 'day_of_week': 0,
                                    'day_name': 'Lundi', 'start_time': '09:00',
                                    'end_time': '12:00'}]}))
    return None


class DelaiDuTourTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='delai', password='x')

    def test_delai_par_defaut(self):
        self.assertEqual(module_agent._delai_tour(), 180.0)

    def test_delai_restant_plancher(self):
        """Meme en fin de tour, un appel borne a le temps de tenter."""
        agent = PlannerAgentV2()
        agent._depart_tour = time.perf_counter() - 10_000
        self.assertGreaterEqual(agent._delai_restant(), 1.0)

    def test_delai_restant_hors_tour(self):
        """Hors process_message_stream (tests directs), deadline entiere."""
        agent = PlannerAgentV2()
        self.assertAlmostEqual(agent._delai_restant(), 180.0, delta=1.0)

    def test_agir_en_fond_abandonne_sur_delai(self):
        """AGIR qui ne rend jamais la main: le tour continue, tronque."""
        def agir_lent(self_agent, user, message, registre):
            time.sleep(2.0)
            return "trop tard"

        agent = PlannerAgentV2()
        agent._file_pensees = queue.Queue()
        registre = Registre()
        with patch.object(PlannerAgentV2, '_agir', agir_lent), \
             override_settings(AGENT_V2_DELAI_TOUR=0.5):
            depart = time.monotonic()
            # _agir_en_fond est un generateur: (raisonnement, panne) arrive
            # comme valeur de retour, pas comme element itere.
            gen = agent._agir_en_fond(self.user, "bonjour", registre)
            try:
                while True:
                    next(gen)
            except StopIteration as fin:
                raisonnement, panne = fin.value
            duree = time.monotonic() - depart
        self.assertTrue(registre.delai_depasse)
        # TimeoutError natif, celui que leve _agir_en_fond. Jusqu'a Python
        # 3.10, concurrent.futures.TimeoutError est une classe DISTINCTE du
        # TimeoutError natif (les deux ont fusionne en 3.11): exiger la
        # version futures rendait le test rouge sur 3.10, pour un type qui
        # ne change rien au comportement (la panne est seulement journalisee).
        self.assertIsInstance(panne, TimeoutError)
        # Plancher de 1 s sur le delai restant: le test attend ~1 s, pas 2.
        self.assertLess(duree, 1.8)

    def test_dire_en_delai_rend_les_faits(self):
        """DIRE qui depasse la deadline: les faits restent, la prose le dit."""
        with patch.object(PlannerAgentV2, '_agir', _agir_qui_cree), \
             patch.object(PlannerAgentV2, '_dire',
                          side_effect=FuturesTimeoutError("trop lent")):
            res = PlannerAgentV2().process_message(self.user, "ajoute maths")
        self.assertIn('Maths', res['response'])
        self.assertIn('trop de temps', res['response'])

    def test_registre_connait_le_delai(self):
        self.assertFalse(Registre().delai_depasse)


class BudgetJetonsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='budget', password='x')

    def test_agir_recoit_le_budget_jetons(self):
        faux = _FauxAgent()
        with patch.object(module_agent, 'Agent', return_value=faux), \
             patch.object(module_agent, 'modele_agir', return_value=object()), \
             patch.object(module_agent, 'outils_pour', return_value=[]), \
             patch.object(module_agent, 'prompt_agir', return_value=""):
            PlannerAgentV2()._agir(self.user, "bonjour", Registre())
        limites = faux.run_kwargs['usage_limits']
        self.assertEqual(limites.request_limit, 10)
        self.assertEqual(limites.total_tokens_limit, 300000)

    def test_dire_recoit_le_budget_jetons(self):
        faux = _FauxAgent()
        with patch.object(module_agent, 'Agent', return_value=faux), \
             patch.object(module_agent, 'modele_dire', return_value=object()), \
             patch.object(PlannerAgentV2, '_brief_dire', return_value=""):
            PlannerAgentV2()._dire(self.user, "bonjour", Registre(), {}, "")
        limites = faux.run_kwargs['usage_limits']
        self.assertEqual(limites.total_tokens_limit, 100000)

    def test_budget_nul_desactive_la_garde(self):
        """Un budget a zero ne doit pas tronquer chaque tour."""
        self.assertEqual(module_agent._limites_jetons(0), {})
        self.assertEqual(module_agent._limites_jetons(-5), {})

    def test_jetons_phase_additionne(self):
        cout = {"entree": 100, "sortie": 50, "raisonnement": 25, "duree": 1.0}
        self.assertEqual(module_agent._jetons_phase(cout), 175)
        self.assertEqual(module_agent._jetons_phase({}), 0)
        self.assertEqual(module_agent._jetons_phase(None), 0)

    def test_compteur_journalier_incremental(self):
        module_agent._enregistrer_jetons(self.user, 1500)
        module_agent._enregistrer_jetons(self.user, 500)
        ligne = BudgetJetonsJournalier.objects.get(user=self.user)
        self.assertEqual(ligne.jetons, 2000)

    def test_compteur_nul_n_ecrit_rien(self):
        module_agent._enregistrer_jetons(self.user, 0)
        self.assertFalse(
            BudgetJetonsJournalier.objects.filter(user=self.user).exists())

    def test_budget_jour_bloque_les_phases_llm(self):
        """Compteur epuise: ni AGIR ni DIRE ne tournent, le tour le dit."""
        BudgetJetonsJournalier.objects.create(
            user=self.user, jour=timezone.localdate(), jetons=2000000)
        with patch.object(PlannerAgentV2, '_agir',
                          side_effect=AssertionError("AGIR ne doit pas tourner")), \
             patch.object(PlannerAgentV2, '_dire',
                          side_effect=AssertionError("DIRE ne doit pas tourner")):
            res = PlannerAgentV2().process_message(self.user, "bonjour")
        self.assertIn("limite d'IA", res['response'])

    def test_budget_jour_zero_desactive(self):
        """Plafond a zero = garde coupee, meme avec un compteur sature."""
        BudgetJetonsJournalier.objects.create(
            user=self.user, jour=timezone.localdate(), jetons=999999999)
        with override_settings(AGENT_V2_BUDGET_JETONS_JOUR=0):
            self.assertFalse(module_agent._budget_jour_epuise(self.user))

    def test_compteur_illisible_ne_bloque_pas(self):
        """Une garde qui casse les tours serait pire que pas de garde."""
        with patch.object(module_agent.BudgetJetonsJournalier.objects,
                          'filter', side_effect=RuntimeError("db en vrac")):
            self.assertFalse(module_agent._budget_jour_epuise(self.user))

    def test_le_rendu_dit_le_delai(self):
        """Le bloc factuel nomme la troncature, comme pour budget_epuise."""
        from services.agent_v2.redaction import bloc_factuel
        registre = Registre()
        registre.delai_depasse = True
        faits = bloc_factuel(registre)
        self.assertIn('délai', faits)
