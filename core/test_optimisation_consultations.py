"""Une consultation simple lit les faits avec un seul appel sémantique."""
import time
from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TransactionTestCase
from django.utils import timezone
from pydantic import ValidationError

from core.models import BudgetJetonsJournalier, ConversationMessage, RecurringBlock
from core.test_refonte_conversation import modele_decision
from services.agent.tools.base import ToolResult
from services.agent_v2.conversation import AgentConversation, DecisionTour


class ConsultationDirecteTests(TransactionTestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username='lecture-directe')
        self.jour = timezone.localdate() + timedelta(days=1)
        self.bloc = RecurringBlock.objects.create(user=self.user, title='Cours témoin',
            block_type='course', day_of_week=self.jour.weekday(),
            start_time='09:00', end_time='10:00')

    def decision(self, **kwargs):
        return dict(mode='planning', objectif='Afficher le calendrier demandé',
            consultation='demain', date_consultation=self.jour.isoformat(),
            affichage_direct=True, **kwargs)

    def executer(self, payload=None, message='Ma journée de demain', attachment=None):
        calls = []
        with patch('services.agent_v2.conversation.modele_agir',
                   return_value=modele_decision(payload or self.decision(), calls)), \
             patch('services.agent_v2.agent.modele_agir', side_effect=AssertionError('second modèle')), \
             patch('services.agent_v2.agent._jugement.juger', side_effect=AssertionError('jugement en plus')):
            events = list(AgentConversation().process_message_stream(
                self.user, message, attachment, generate_quick_replies=True))
        done = next(e for e in events if e['type'] == 'done')
        self.assertEqual(done['response'], ''.join(e['text'] for e in events if e['type'] == 'delta'))
        meta = ConversationMessage.objects.get(pk=done['assistant_message_id']).metadata
        return done, meta, calls, events

    def test_jour_un_appel_et_recu_orm_sans_prompt_planning(self):
        with patch('services.agent_v2.prompts.build_context', side_effect=AssertionError('contexte lourd')):
            done, meta, calls, events = self.executer()
        self.assertEqual(len(calls), 1)
        self.assertIn('Cours témoin', done['response'])
        self.assertIn('9 h', done['response'])
        self.assertEqual(meta['lecture'], 'directe')
        self.assertEqual(meta['contrat_tour'], 'planning')
        self.assertEqual([(a['outil'], a['succes'], a['par_le_code']) for a in meta['actions']],
                         [('get_today_schedule', True, True)])
        self.assertEqual([u['phase'] for u in meta['usage']], ['decision'])
        self.assertEqual(done['quick_replies'], [])
        self.assertEqual(ConversationMessage.objects.filter(user=self.user).count(), 2)
        budget = BudgetJetonsJournalier.objects.get(user=self.user)
        self.assertEqual((budget.jetons, budget.jetons_reserves), (120, 0))

    def test_contrat_date_prime_sur_texte(self):
        payload = self.decision()
        payload.update(consultation='journee', date_consultation=(self.jour + timedelta(days=1)).isoformat())
        done, meta, _, _ = self.executer(payload)
        self.assertNotIn('Cours témoin', done['response'])
        self.assertIn('Rien de prévu', done['response'])
        self.assertEqual(meta['date_consultation'], payload['date_consultation'])

    def test_semaine_sans_deuxieme_modele(self):
        payload = self.decision()
        debut = self.jour - timedelta(days=self.jour.weekday())
        payload.update(consultation='semaine', date_consultation=debut.isoformat())
        done, meta, calls, _ = self.executer(payload)
        self.assertEqual(len(calls), 1)
        self.assertIn('Cours témoin', done['response'])
        self.assertIn('Ta semaine', done['response'])
        self.assertEqual(meta['actions'][0]['outil'], 'get_week_schedule')

    def test_liste_taches_vide_reste_reponse_valide(self):
        payload = self.decision()
        payload.update(consultation='taches', date_consultation=None)
        done, meta, _, _ = self.executer(payload)
        self.assertEqual(done['response'], 'Ta liste de tâches est vide.')
        self.assertEqual(meta['actions'][0]['outil'], 'list_tasks')

    def test_isolation_utilisateur(self):
        autre = get_user_model().objects.create_user(username='autre-lecture-directe')
        RecurringBlock.objects.create(user=autre, title='Secret autre compte', block_type='course',
            day_of_week=self.jour.weekday(), start_time='11:00', end_time='12:00')
        done, _, _, _ = self.executer()
        self.assertIn('Cours témoin', done['response'])
        self.assertNotIn('Secret', done['response'])

    def test_chaque_tour_relit_les_donnees(self):
        self.executer()
        self.bloc.title = 'Titre corrigé depuis le dernier tour'
        self.bloc.save(update_fields=['title'])
        done, _, _, _ = self.executer()
        self.assertIn(self.bloc.title, done['response'])
        self.assertNotIn('Cours témoin', done['response'])

    def test_echec_outil_ne_devient_pas_planning_vide(self):
        with patch('services.agent.tools.get_today_schedule.execute',
                   return_value=ToolResult(success=False, message='Erreur confidentielle')):
            done, meta, _, _ = self.executer()
        self.assertIn('réessayer', done['response'])
        self.assertNotIn('Rien de prévu', done['response'])
        self.assertNotIn('confidentielle', done['response'])
        self.assertFalse(meta['actions'][0]['succes'])

    def test_date_retournee_doit_correspondre_a_la_date_demandee(self):
        with patch('services.agent.tools.get_today_schedule.execute',
                   return_value=ToolResult(success=True, data={
                       'date': (self.jour + timedelta(days=1)).isoformat(), 'blocks': [], 'tasks': []})):
            done, meta, _, _ = self.executer()
        self.assertIn('réessayer', done['response'])
        self.assertEqual(meta['faits_rendus'], '')

    def test_panne_du_rendu_n_est_pas_un_fait_confirme(self):
        with patch('services.agent_v2.redaction.bloc_factuel', side_effect=RuntimeError('rendu')):
            done, meta, _, _ = self.executer()
        self.assertIn('réessayer', done['response'])
        self.assertTrue(meta['actions'][0]['succes'])
        self.assertEqual(meta['faits_rendus'], '')

    def test_exception_lecture_ne_fuit_pas_et_ne_relance_pas_modele(self):
        with patch('services.agent_v2.outils.executer_lecture_par_le_code',
                   side_effect=RuntimeError('SECRET_RESEAU')):
            done, _, calls, _ = self.executer()
        self.assertEqual(len(calls), 1)
        self.assertNotIn('SECRET_RESEAU', done['response'])
        self.assertIn('réessayer', done['response'])

    def test_echeance_expiree_avant_lecture(self):
        original = AgentConversation._decider
        def expirer(instance, *args, **kwargs):
            decision = original(instance, *args, **kwargs)
            instance._trace.echeance = time.monotonic() - 1
            return decision
        with patch.object(AgentConversation, '_decider', expirer), \
             patch('services.agent.tools.get_today_schedule.execute') as read:
            done, _, _, _ = self.executer()
        read.assert_not_called()
        self.assertIn('réessayer', done['response'])

    def test_schema_refuse_raccourci_incoherent(self):
        cases = [dict(autoriser_modifications=True), dict(memoire_action='lister'),
                 dict(consultation='autre'), dict(date_consultation=None),
                 dict(mode='conversation', reponse='Salut'),
                 dict(consultation='taches')]
        for updates in cases:
            with self.subTest(updates=updates), self.assertRaises(ValidationError):
                DecisionTour(**(self.decision() | updates))

    def test_schema_defaut_conserve_boucle_complexe(self):
        payload = self.decision()
        payload.pop('affichage_direct')
        self.assertFalse(DecisionTour(**payload).affichage_direct)

    def test_demande_complexe_utilise_boucle_existante(self):
        payload = self.decision() | {'affichage_direct': False}
        with patch('services.agent_v2.conversation.modele_agir', return_value=modele_decision(payload, [])), \
             patch('services.agent_v2.agent.PlannerAgentV2.process_message_stream',
                   return_value=iter([{'type': 'done', 'response': 'Analyse complète'}])) as full:
            result = AgentConversation().process_message(self.user, 'Compare ces deux journées')
        full.assert_called_once()
        self.assertEqual(result['response'], 'Analyse complète')

    def test_document_joint_conserve_parcours_import(self):
        from types import SimpleNamespace
        with patch('services.agent_v2.conversation.modele_agir', return_value=modele_decision(self.decision(), [])), \
             patch('services.agent_v2.agent.PlannerAgentV2.process_message_stream',
                   return_value=iter([{'type': 'done', 'response': 'Document traité'}])) as full:
            AgentConversation().process_message(self.user, 'Mon planning', attachment=SimpleNamespace(processed=True))
        full.assert_called_once()


class ConsultationDirecteHttpTests(TransactionTestCase):
    def setUp(self):
        from django.core.cache import cache
        from rest_framework.test import APIClient
        cache.clear()
        self.user = get_user_model().objects.create_user(username='lecture-http')
        profile = self.user.profile
        profile.ai_consent_at, profile.agent_v2 = timezone.now(), True
        profile.save(update_fields=['ai_consent_at', 'agent_v2'])
        self.client = APIClient()
        self.client.force_authenticate(self.user)
        self.jour = timezone.localdate() + timedelta(days=1)
        RecurringBlock.objects.create(user=self.user, title='Cours HTTP', block_type='course',
            day_of_week=self.jour.weekday(), start_time='09:00', end_time='10:00')
        self.payload = dict(mode='planning', objectif='Afficher la journée', consultation='demain',
            date_consultation=self.jour.isoformat(), affichage_direct=True)

    def verifier(self, stream):
        import asyncio
        import json
        from uuid import uuid4
        from core.models import ChatTurn
        identity, calls = str(uuid4()), []
        endpoint = '/api/chat/stream/' if stream else '/api/chat/'
        with patch('services.agent_v2.conversation.modele_agir',
                   return_value=modele_decision(self.payload, calls)), \
             patch('services.agent_v2.agent.modele_agir', side_effect=AssertionError('second modèle')):
            response = self.client.post(endpoint, dict(message='Mon planning de demain', turn_id=identity))
            self.assertEqual(response.status_code, 200)
            if stream:
                async def consume():
                    return [part async for part in response.streaming_content]
                text = ''.join(p.decode() if isinstance(p, bytes) else p for p in asyncio.run(consume()))
                events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: ')]
                result = next(e for e in events if e.get('type') == 'done')
                self.assertEqual(result['response'], ''.join(e['text'] for e in events if e.get('type') == 'delta'))
            else:
                result = response.data
            retry = self.client.post('/api/chat/', dict(message='Mon planning de demain', turn_id=identity))
            self.assertEqual(retry.status_code, 200)
        self.assertEqual(len(calls), 1)
        self.assertIn('Cours HTTP', result['response'])
        turn = ChatTurn.objects.get(user=self.user, turn_id=identity)
        self.assertEqual(turn.status, 'completed')
        self.assertEqual(turn.result['response'], result['response'])
        self.assertEqual(retry.data['response'], result['response'])
        budget = BudgetJetonsJournalier.objects.get(user=self.user)
        self.assertEqual((budget.jetons, budget.jetons_reserves), (120, 0))

    def test_http_reponse_durable_sans_double_appel(self):
        self.verifier(False)

    def test_sse_reponse_durable_sans_double_appel(self):
        self.verifier(True)
