import asyncio
import time
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TransactionTestCase
from pydantic_ai.usage import RunUsage

from core.models import ConversationMessage
from services.agent_v2.agent import PlannerAgentV2, _cout, _jetons_phase
from services.agent_v2.conversation import DecisionTour
from services.agent_v2.registre import Registre


class ContexteEtUsageTests(TransactionTestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username='contexte-borne')

    def test_raisonnement_est_inclus_dans_sortie(self):
        usage = RunUsage(requests=1, input_tokens=100, output_tokens=20,
                         cache_read_tokens=60, details={'reasoning_tokens': 8})
        cout = _cout(SimpleNamespace(usage=lambda: usage), 1)
        self.assertEqual(_jetons_phase(cout), 120)
        self.assertEqual(cout['cache'], 60)
        self.assertTrue(cout['usage_connu'])

    def test_usage_absent_est_inconnu(self):
        cout = _cout(None, 1)
        self.assertFalse(cout['usage_connu'])
        self.assertIsNone(cout['entree'])

    def test_historique_total_et_message_bornes(self):
        for i in range(20):
            ConversationMessage.objects.create(user=self.user, role='user', content=f'{i}:'+'a'*10000)
        histoire = PlannerAgentV2()._historique(self.user)
        contenus = [p.content for m in histoire for p in m.parts]
        self.assertLessEqual(sum(map(len, contenus)), 16000)
        self.assertTrue(all(len(t) <= 4000 for t in contenus))
        self.assertTrue(contenus[-1].startswith('19:'))

    def test_prompt_n_extrait_pas_le_calendrier(self):
        from services.agent_v2.prompts import prompt_planification
        with patch('services.agent_v2.prompts.build_context', side_effect=AssertionError('calendrier')):
            texte = prompt_planification(self.user, DecisionTour(mode='planning', objectif='Voir la semaine'))
        self.assertLess(len(texte), 6500)
        self.assertNotIn('SEMAINE TYPE', texte)

    def test_deadline_epuisee_reste_a_zero(self):
        agent = PlannerAgentV2()
        agent._depart_tour = time.perf_counter() - 300
        self.assertEqual(agent._delai_restant(), 0)

    def test_boucle_async_est_annulee_a_la_deadline(self):
        annule = []
        async def run(*args, **kwargs):
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                annule.append(True)
                raise
        with self.assertRaises(asyncio.TimeoutError):
            PlannerAgentV2._executer_boucle(SimpleNamespace(run=run), 'message', delai=0.02)
        self.assertEqual(annule, [True])

    def test_boucle_garde_usage_partiel_sur_budget(self):
        from pydantic_ai.exceptions import UsageLimitExceeded
        def arreter(agent, message, **kwargs):
            kwargs['usage'].incr(RunUsage(requests=1, input_tokens=100, output_tokens=20))
            raise UsageLimitExceeded('budget')
        agent = PlannerAgentV2()
        with patch.object(agent, '_executer_boucle', side_effect=arreter), \
             patch('services.agent_v2.agent.modele_agir', return_value='test'):
            registre = Registre()
            agent._boucle(self.user, 'Montre la semaine', registre)
        self.assertTrue(registre.budget_epuise)
        self.assertEqual(agent._cout_boucle['entree'], 100)
        self.assertEqual(agent._cout_boucle['etapes'], 1)

    def test_lecture_seule_refuse_mutation_meme_via_chargeur(self):
        from services.agent_v2.outils import outils_pour_le_modele
        registre = Registre()
        registre.lecture_seule = True
        outils = outils_pour_le_modele(self.user, registre)
        self.assertNotIn('create_block', [o.name for o in outils])
        appeler = next(o for o in outils if o.name == 'appeler_outil')
        resultat = asyncio.run(appeler.function_schema.function(
            nom='create_goal', parametres={'title': 'Interdit'}))
        self.assertIn('consultation', resultat.lower())

    def test_ecriture_synchrone_expiree_est_annulee_en_base(self):
        from core.models import Task
        from services.agent.tools.base import ToolResult
        from services.agent_v2.outils import outils_pour, TOOL_MAP
        registre = Registre()
        registre.echeance = time.monotonic() + .04
        def lent(user, **kwargs):
            task = Task.objects.create(user=user, title='Ne doit pas survivre')
            time.sleep(.06)
            return ToolResult(success=True, data={'task': {'id': task.pk}}, message='Cree')
        outils = outils_pour(self.user, registre)
        outil = next(o for o in outils if o.name == 'create_task')
        with patch.object(TOOL_MAP['create_task'], 'execute', side_effect=lent) as execution:
            asyncio.run(outil.function_schema.function(title='Ne doit pas survivre'))
        execution.assert_called_once()
        self.assertFalse(Task.objects.filter(user=self.user).exists())
        self.assertFalse(any(a.succes for a in registre.actions))

    def test_sortie_outil_enorme_ne_sature_pas_le_modele(self):
        import json
        from services.agent_v2.outils import borner_resultat_modele
        rendu = borner_resultat_modele(json.dumps({'data': ['x'*10000]*20}))
        self.assertLess(len(rendu), 12000)
        self.assertTrue(json.loads(rendu)['resultat_tronque'])

    def test_secours_conserve_date_demandee(self):
        from datetime import date
        agent = PlannerAgentV2()
        agent._decision_tour = DecisionTour(mode='planning', objectif='Voir le 17 novembre',
            consultation='journee', date_consultation=date(2026, 11, 17))
        agent._verdict_consultation = {'statut': 'decision', 'valeur': 'journee', 'confiance': 1}
        with patch('services.agent_v2.outils.executer_lecture_par_le_code',
                   return_value=SimpleNamespace(succes=True)) as lecture:
            agent._lecture_de_secours(self.user, 'Montre le 17 novembre', Registre())
        self.assertEqual(lecture.call_args.kwargs['date'], '2026-11-17')

    def test_secours_sans_date_ne_substitue_pas_aujourdhui(self):
        agent = PlannerAgentV2()
        agent._decision_tour = DecisionTour(mode='planning', objectif='Voir le 17 novembre', consultation='journee')
        agent._verdict_consultation = {'statut': 'decision', 'valeur': 'journee', 'confiance': 1}
        with patch('services.agent_v2.outils.executer_lecture_par_le_code') as lecture:
            agent._lecture_de_secours(self.user, 'Montre le 17 novembre', Registre())
        lecture.assert_not_called()

    def test_lundi_zero_ne_contourne_pas_la_garde_composite(self):
        from services.agent_v2.outils import _appel_composite
        refus = _appel_composite('update_block', {'day_of_week': 0, 'end_date': '2026-12-01'})
        self.assertIsNotNone(refus)
        self.assertTrue(refus.data['appel_composite'])

