"""Réservations réellement exécutées en base, fournisseurs locaux scriptés."""
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, TransactionTestCase, override_settings
from django.db import close_old_connections
from django.utils import timezone

from services.agent_v2 import agent
from services.agent_v2.budget_tour import (
    BudgetIndisponible, BudgetEpuise, AppelFournisseur, ouvrir_trace,
    fermer_trace, appel_fournisseur, modele_borne,
)
from services.agent_v2.trace_tour import trace_courante
from core.models import BudgetJetonsJournalier, ReservationBudgetJetons


class BudgetFermeTests(TestCase):
    def test_compteur_illisible_interdit_un_appel_payant(self):
        user = get_user_model().objects.create_user(username='budget-indisponible')
        with patch.object(agent.BudgetJetonsJournalier.objects, 'filter',
                          side_effect=RuntimeError('indisponible')):
            self.assertTrue(agent._budget_jour_epuise(user))

    def test_usage_inconnu_ne_vaut_pas_zero(self):
        from services.agent_v2.trace_tour import TraceTour
        trace = TraceTour(time.monotonic() + 10)
        trace.ajouter({'phase': 'decision', 'usage_connu': False,
                      'jetons_provisionnes': 5000})
        self.assertEqual(trace.total_jetons, 5000)

    def test_fermeture_clients_de_toute_chaine_avant_arret_asyncio(self):
        from pydantic_ai.models.function import FunctionModel
        from pydantic_ai.models.fallback import FallbackModel
        from services.agent_v2.modeles import fermer_modele
        fermes = []
        class ClientLocal:
            async def close(self):
                self.boucle_active = asyncio.get_running_loop().is_running()
                fermes.append(self.boucle_active)
        def modele_local():
            modele = FunctionModel(lambda messages, info: None)
            modele.client = ClientLocal()
            return modele
        chaine = modele_borne(FallbackModel(modele_local(), modele_local()))
        asyncio.run(fermer_modele(chaine))
        self.assertEqual(fermes, [True, True])


class ReservationBudgetTests(TransactionTestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username='reservation-budget')

    def ouvrir(self, **kwargs):
        trace = ouvrir_trace(self.user, **kwargs)
        token = trace_courante.set(trace)
        self.addCleanup(trace_courante.reset, token)
        self.addCleanup(fermer_trace, trace)
        return trace

    def compteur(self):
        return BudgetJetonsJournalier.objects.get(user=self.user)

    def test_reservation_initiale_legere_et_liberation_sans_appel(self):
        trace = self.ouvrir()
        self.assertEqual(self.compteur().jetons_reserves, 12000)
        self.assertEqual(self.compteur().jetons, 0)
        fermer_trace(trace)
        fermer_trace(trace)
        self.assertEqual(self.compteur().jetons_reserves, 0)
        self.assertEqual(self.compteur().jetons, 0)
        self.assertTrue(ReservationBudgetJetons.objects.get().cloturee)

    def test_usage_connu_rembourse_une_seule_fois(self):
        trace = self.ouvrir()
        with appel_fournisseur(phase='decision', fournisseur='local',
                              jetons_entree=2000, max_sortie=1000) as appel:
            self.assertEqual(self.compteur().jetons, 3000)
            appel.terminer({'entree': 100, 'sortie': 20}, True)
            appel.terminer({'entree': 100, 'sortie': 20}, True)
        fermer_trace(trace)
        self.assertEqual(self.compteur().jetons, 120)
        self.assertEqual(self.compteur().jetons_mesures, 120)
        self.assertEqual(self.compteur().jetons_reserves, 0)
        self.assertEqual(trace.total_jetons, 120)
        self.assertEqual(len(trace.usages), 1)

    def test_timeout_usage_inconnu_conserve_provision(self):
        trace = self.ouvrir()
        with self.assertRaises(TimeoutError):
            with appel_fournisseur(phase='extraction', fournisseur='local',
                                  jetons_entree=2000, max_sortie=1000):
                raise TimeoutError()
        fermer_trace(trace)
        self.assertEqual(self.compteur().jetons, 3000)
        self.assertEqual(trace.total_jetons, 3000)
        self.assertFalse(trace.usages[0]['usage_connu'])

    def test_retour_tardif_connu_rapproche_apres_fermeture(self):
        trace = self.ouvrir()
        appel = AppelFournisseur('extraction', 'local', 2000, 1000)
        fermer_trace(trace)
        self.assertEqual(self.compteur().jetons, 3000)
        appel.terminer({'total': 150}, True)
        self.assertEqual(self.compteur().jetons, 150)
        self.assertEqual(self.compteur().jetons_reserves, 0)
        with self.assertRaises(BudgetIndisponible):
            AppelFournisseur('planning', 'local', 10, 10)

    def test_rapprochement_impossible_ne_rembourse_pas(self):
        trace = self.ouvrir()
        with patch('services.agent_v2.budget_tour.ReservationTour.rapprocher',
                   side_effect=RuntimeError('indisponible')):
            with appel_fournisseur(phase='decision', fournisseur='local',
                                  jetons_entree=2000, max_sortie=1000) as appel:
                appel.terminer({'total': 120}, True)
        fermer_trace(trace)
        self.assertEqual(self.compteur().jetons, 3000)
        self.assertEqual(trace.usages[0]['jetons_factures'], 3000)

    def test_deadline_et_absence_admission_empechent_appel(self):
        with self.assertRaises(BudgetIndisponible):
            AppelFournisseur('decision', 'local', 1000, 1000)
        trace = self.ouvrir(delai=-1)
        with self.assertRaises(TimeoutError):
            AppelFournisseur('decision', 'local', 1000, 1000)
        self.assertEqual(self.compteur().jetons, 0)
        self.assertEqual(trace.usages, [])

    @override_settings(AGENT_V2_BUDGET_JETONS_JOUR=15000, AGENT_V2_GARDE_COUT_JOUR=15000)
    def test_extension_refusee_avant_appel(self):
        trace = self.ouvrir()
        with self.assertRaises(BudgetEpuise):
            AppelFournisseur('planning', 'local', 15000, 1000)
        self.assertEqual(self.compteur().jetons_reserves, 12000)
        self.assertEqual(self.compteur().jetons, 0)
        self.assertEqual(trace.usages, [])

    @override_settings(AGENT_V2_BUDGET_JETONS_JOUR=15000, AGENT_V2_GARDE_COUT_JOUR=15000)
    def test_solde_inferieur_reservation_initiale_est_utilisable(self):
        BudgetJetonsJournalier.objects.create(user=self.user, jour=timezone.localdate(), jetons=10000)
        trace = self.ouvrir()
        self.assertEqual(self.compteur().jetons_reserves, 5000)
        with appel_fournisseur(phase='decision', fournisseur='local',
                              jetons_entree=2000, max_sortie=1000) as appel:
            appel.terminer({'total': 120}, True)
        fermer_trace(trace)
        self.assertEqual(self.compteur().jetons, 10120)

    @override_settings(AGENT_V2_BUDGET_JETONS_JOUR=12000, AGENT_V2_GARDE_COUT_JOUR=12000)
    def test_admissions_concurrentes_ne_depassent_pas_le_plafond(self):
        BudgetJetonsJournalier.objects.create(user=self.user, jour=timezone.localdate())
        barriere = threading.Barrier(2)
        def admettre():
            close_old_connections()
            try:
                barriere.wait(timeout=3)
                return ouvrir_trace(self.user)
            except BudgetIndisponible:
                return None
            finally:
                close_old_connections()
        with ThreadPoolExecutor(max_workers=2) as pool:
            traces = list(pool.map(lambda _: admettre(), range(2)))
        try:
            self.assertEqual(sum(t is not None for t in traces), 1)
            self.assertEqual(self.compteur().jetons_reserves, 12000)
            self.assertEqual(ReservationBudgetJetons.objects.count(), 1)
        finally:
            for trace in traces:
                if trace:
                    fermer_trace(trace)

    def test_db_indisponible_ne_cree_aucune_trace_admise(self):
        with patch.object(BudgetJetonsJournalier.objects, 'get_or_create', side_effect=RuntimeError):
            with self.assertRaises(BudgetIndisponible):
                ouvrir_trace(self.user)

    def test_vrai_sdk_borne_sortie_et_compte_chaque_fallback(self):
        from pydantic_ai import Agent
        from pydantic_ai.models.function import FunctionModel
        from pydantic_ai.models.fallback import FallbackModel
        from pydantic_ai.messages import ModelResponse, TextPart
        from pydantic_ai.usage import RequestUsage
        from pydantic_ai.exceptions import ModelHTTPError
        appels = []
        def panne(messages, info):
            appels.append(info.model_settings['max_tokens'])
            raise ModelHTTPError(503, 'local')
        def succes(messages, info):
            appels.append(info.model_settings['max_tokens'])
            return ModelResponse(parts=[TextPart('Salut')], usage=RequestUsage(input_tokens=20, output_tokens=3))
        trace = self.ouvrir()
        modele = modele_borne(FallbackModel(FunctionModel(panne), FunctionModel(succes)), 'decision')
        resultat = Agent(modele).run_sync('bonjour', model_settings={'max_tokens': 999999})
        fermer_trace(trace)
        self.assertEqual(resultat.output, 'Salut')
        self.assertEqual(appels, [4096, 4096])
        self.assertEqual(len(trace.usages), 2)
        self.assertFalse(trace.usages[0]['usage_connu'])
        self.assertEqual(trace.usages[1]['total'], 23)
        self.assertEqual(self.compteur().jetons, trace.usages[0]['jetons_provisionnes'] + 23)

    def test_sdk_annule_conserve_usage_inconnu(self):
        from pydantic_ai import Agent
        from pydantic_ai.models.function import FunctionModel
        async def lent(messages, info):
            await asyncio.sleep(1)
        trace = self.ouvrir()
        async def executer():
            await asyncio.wait_for(Agent(modele_borne(FunctionModel(lent))).run('bonjour'), 0.08)
        with self.assertRaises(asyncio.TimeoutError):
            asyncio.run(executer())
        fermer_trace(trace)
        self.assertGreater(self.compteur().jetons, 0)
        self.assertFalse(trace.usages[0]['usage_connu'])

    def test_sdk_stream_reconcilie_usage_a_fin_flux(self):
        from pydantic_ai import Agent
        from pydantic_ai.models.function import FunctionModel
        async def fragments(messages, info):
            self.assertEqual(info.model_settings['max_tokens'], 1024)
            yield 'Salut'
        trace = self.ouvrir()
        modele = modele_borne(FunctionModel(stream_function=fragments), 'decision')
        async def executer():
            async with Agent(modele).run_stream('bonjour', model_settings={'max_tokens': 1024}) as resultat:
                self.assertEqual(''.join([texte async for texte in resultat.stream_text(delta=True)]), 'Salut')
        asyncio.run(executer())
        fermer_trace(trace)
        self.assertEqual(len(trace.usages), 1)
        self.assertTrue(trace.usages[0]['usage_connu'])
        self.assertGreater(self.compteur().jetons, 0)
        self.assertEqual(self.compteur().jetons, trace.usages[0]['total'])

    def test_google_deadline_serveur_dix_secondes_conserve_annulation_locale(self):
        from pydantic_ai import Agent
        from pydantic_ai.models.function import FunctionModel
        class GoogleLocal(FunctionModel):
            @property
            def system(self):
                return 'google-gla'
        reglages, annulations = [], []
        async def lent(messages, info):
            reglages.append(info.model_settings)
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                annulations.append(True)
                raise
        trace = self.ouvrir()
        depart = time.monotonic()
        with self.assertRaises(asyncio.TimeoutError):
            Agent(modele_borne(GoogleLocal(lent), 'jugement')).run_sync(
                'question', model_settings={'timeout': 0.1})
        self.assertLess(time.monotonic() - depart, 0.8)
        self.assertEqual(reglages[0]['timeout'], 10.0)
        self.assertEqual(annulations, [True])
        self.assertFalse(trace.usages[0]['usage_connu'])

    def test_google_stream_annule_et_conserve_usage_inconnu(self):
        from pydantic_ai import Agent
        from pydantic_ai.models.function import FunctionModel
        class GoogleLocal(FunctionModel):
            @property
            def system(self):
                return 'google-gla'
        annulations = []
        async def fragments(messages, info):
            self.assertEqual(info.model_settings['timeout'], 10.0)
            yield 'début'
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                annulations.append(True)
                raise
            yield 'fin'
        trace = self.ouvrir()
        async def executer():
            async with Agent(modele_borne(GoogleLocal(stream_function=fragments))).run_stream(
                    'question', model_settings={'timeout': 0.1}) as resultat:
                async for texte in resultat.stream_text(delta=True):
                    pass
        with self.assertRaises(asyncio.TimeoutError):
            asyncio.run(executer())
        self.assertEqual(annulations, [True])
        self.assertFalse(trace.usages[0]['usage_connu'])
        self.assertGreater(self.compteur().jetons, 0)

    def test_base_devient_indisponible_avant_fournisseur(self):
        from pydantic_ai import Agent
        from pydantic_ai.models.function import FunctionModel
        from unittest.mock import Mock
        fournisseur = Mock()
        fournisseur.__name__ = 'fournisseur_local'
        trace = self.ouvrir()
        with patch.object(ReservationBudgetJetons.objects, 'filter', side_effect=RuntimeError):
            with self.assertRaises(BudgetIndisponible):
                Agent(modele_borne(FunctionModel(fournisseur))).run_sync('bonjour')
        fournisseur.assert_not_called()
        self.assertEqual(trace.usages, [])
        self.assertEqual(self.compteur().jetons, 0)

    @override_settings(AGENT_V2_BUDGET_JETONS_JOUR=20000, AGENT_V2_GARDE_COUT_JOUR=20000)
    def test_usage_inconnu_reduit_admission_suivante(self):
        trace = self.ouvrir()
        with appel_fournisseur(phase='planning', fournisseur='local',
                              jetons_entree=15000, max_sortie=1000):
            pass
        fermer_trace(trace)
        suivant = ouvrir_trace(self.user)
        token = trace_courante.set(suivant)
        try:
            with self.assertRaises(BudgetEpuise):
                AppelFournisseur('decision', 'local', 4000, 1000)
            self.assertEqual(self.compteur().jetons, 16000)
            self.assertEqual(self.compteur().jetons_reserves, 4000)
        finally:
            trace_courante.reset(token)
            fermer_trace(suivant)

    @override_settings(AGENT_V2_BUDGET_JETONS_JOUR=10000, AGENT_V2_GARDE_COUT_JOUR=30000)
    def test_usage_inconnu_reussi_reste_hors_jauge(self):
        from services.agent_v2.budget_tour import statut_usage
        trace = self.ouvrir()
        with appel_fournisseur(phase='planning', fournisseur='local',
                              jetons_entree=4000, max_sortie=1000) as appel:
            appel.terminer({}, True)
        fermer_trace(trace)
        self.assertEqual(self.compteur().jetons, 5000)
        self.assertEqual(self.compteur().jetons_mesures, 0)
        self.assertEqual(statut_usage(self.user)['appels_non_mesures'], 1)
        self.assertIsNone(statut_usage(self.user)['bloque_par'])

    @override_settings(AGENT_V2_BUDGET_JETONS_JOUR=1000, AGENT_V2_GARDE_COUT_JOUR=30000)
    def test_quota_mesure_et_garde_sont_deux_blocages_distincts(self):
        trace = self.ouvrir()
        with appel_fournisseur(phase='planning', fournisseur='local',
                              jetons_entree=2000, max_sortie=1000) as appel:
            appel.terminer({'total': 1000}, True)
        fermer_trace(trace)
        self.assertEqual(self.compteur().jetons_mesures, 1000)
        with self.assertRaises(BudgetEpuise) as blocage:
            ouvrir_trace(self.user)
        self.assertEqual(blocage.exception.motif, 'quota_mesure')

    @override_settings(AGENT_V2_BUDGET_JETONS_JOUR=10000, AGENT_V2_GARDE_COUT_JOUR=12000)
    def test_garde_de_cout_bloque_sans_epuiser_quota_mesure(self):
        from services.agent_v2.budget_tour import statut_usage
        BudgetJetonsJournalier.objects.create(
            user=self.user, jour=timezone.localdate(), jetons=12000,
            jetons_mesures=0,
        )
        self.assertEqual(statut_usage(self.user)['bloque_par'], 'garde_cout')
        with self.assertRaises(BudgetEpuise) as blocage:
            ouvrir_trace(self.user)
        self.assertEqual(blocage.exception.motif, 'garde_cout')
