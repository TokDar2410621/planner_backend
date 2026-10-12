"""La consultation décidée reste prioritaire sur les lectures auxiliaires."""
from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TransactionTestCase
from django.utils import timezone
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from core.models import ConversationMessage, RecurringBlock
from core.test_refonte_conversation import modele_decision
from services.agent_v2.conversation import AgentConversation
from services.agent_v2.redaction import ReponseDire


class ConsultationPrioritaireTests(TransactionTestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username='lecture-prioritaire')
        self.demain = timezone.localdate() + timedelta(days=1)
        RecurringBlock.objects.create(
            user=self.user, title='Cours de test', block_type='course',
            day_of_week=self.demain.weekday(), start_time='09:00', end_time='10:00')

    def executer(self, appels_outils, consultation='demain', date=None):
        requetes = []

        def planifier(messages, info):
            index = len(requetes)
            requetes.append(info)
            if index < len(appels_outils):
                nom, parametres = appels_outils[index]
                partie = ToolCallPart(nom, parametres)
            else:
                partie = ToolCallPart(info.output_tools[0].name, ReponseDire().model_dump())
            return ModelResponse(parts=[partie],
                                 usage=RequestUsage(input_tokens=100, output_tokens=20))

        decision = {'mode': 'planning', 'objectif': 'Voir la consultation demandée',
                    'consultation': consultation, 'autoriser_modifications': False,
                    'date_consultation': (date or self.demain).isoformat()}
        with patch('services.agent_v2.conversation.modele_agir',
                   return_value=modele_decision(decision, [])), \
             patch('services.agent_v2.agent.modele_agir', return_value=FunctionModel(planifier)), \
             patch.object(AgentConversation, '_sur_evenements', None), \
             patch('services.agent_v2.agent._jugement.juger', side_effect=AssertionError('nouveau jugement')):
            evenements = list(AgentConversation().process_message_stream(
                self.user, "Qu'est-ce que j'ai demain ?"))
        fin = next(e for e in evenements if e['type'] == 'done')
        self.assertEqual(fin['response'], ''.join(e['text'] for e in evenements if e['type'] == 'delta'))
        self.assertEqual(len(requetes), len(appels_outils) + 1)
        meta = ConversationMessage.objects.filter(user=self.user, role='assistant').latest('pk').metadata
        return fin['response'], meta

    def test_demain_reste_visible_apres_lecture_des_taches(self):
        texte, meta = self.executer([
            ('get_today_schedule', {'date': self.demain.isoformat()}),
            ('list_tasks', {}),
        ])
        self.assertIn('Cours de test', texte)
        self.assertIn('9 h', texte)
        self.assertIn('10 h', texte)
        self.assertNotIn('Ta liste de tâches est vide', texte)
        self.assertEqual([a['outil'] for a in meta['actions']], ['get_today_schedule', 'list_tasks'])
        self.assertEqual((meta['consultation'], meta['date_consultation']),
                         ('demain', self.demain.isoformat()))

    def test_autre_lecture_ne_dispense_pas_de_lire_la_date_demandee(self):
        texte, meta = self.executer([('list_tasks', {})])
        self.assertIn('Cours de test', texte)
        self.assertNotIn('Ta liste de tâches est vide', texte)
        self.assertEqual([a['outil'] for a in meta['actions']], ['list_tasks', 'get_today_schedule'])

    def test_jour_lu_different_declenche_la_bonne_lecture(self):
        texte, meta = self.executer([
            ('get_today_schedule', {'date': timezone.localdate().isoformat()}),
            ('list_tasks', {}),
        ])
        self.assertIn('Cours de test', texte)
        self.assertEqual([a['outil'] for a in meta['actions']].count('get_today_schedule'), 2)

    def test_date_du_contrat_prime_sur_les_mots_du_message(self):
        autre_date = self.demain + timedelta(days=1)
        texte, _ = self.executer([
            ('get_today_schedule', {'date': self.demain.isoformat()}),
        ], consultation='journee', date=autre_date)
        self.assertNotIn('Cours de test', texte)
        self.assertIn('Rien de prévu.', texte)

    def test_demande_de_taches_prime_sur_lecture_du_planning(self):
        texte, meta = self.executer([
            ('list_tasks', {}),
            ('get_today_schedule', {'date': self.demain.isoformat()}),
        ], consultation='taches')
        self.assertEqual(texte, 'Ta liste de tâches est vide.')
        self.assertEqual(len(meta['actions']), 2)

    def test_demande_de_semaine_prime_sur_lecture_des_taches(self):
        texte, meta = self.executer([
            ('get_week_schedule', {'start_date': self.demain.isoformat()}),
            ('list_tasks', {}),
        ], consultation='semaine')
        self.assertIn('Cours de test', texte)
        self.assertIn('Ta semaine', texte)
        self.assertNotIn('Ta liste de tâches est vide', texte)
        self.assertEqual(len(meta['actions']), 2)

    def test_lecture_de_secours_reste_limitee_a_lutilisateur(self):
        autre = get_user_model().objects.create_user(username='autre-lecture')
        RecurringBlock.objects.create(
            user=autre, title='Titre privé autre utilisateur', block_type='course',
            day_of_week=self.demain.weekday(), start_time='11:00', end_time='12:00')
        texte, meta = self.executer([])
        self.assertIn('Cours de test', texte)
        self.assertNotIn('Titre privé', texte)
        self.assertEqual([a['outil'] for a in meta['actions']], ['get_today_schedule'])

    def test_echec_lecture_demandee_ne_rend_pas_une_liste_sans_rapport(self):
        from services.agent.tools.base import ToolResult
        with patch('services.agent.tools.schedule.GetTodayScheduleTool.execute',
                   return_value=ToolResult(success=False, message='Lecture indisponible')):
            texte, meta = self.executer([('list_tasks', {})])
        self.assertNotIn('Ta liste de tâches est vide', texte)
        self.assertNotIn('Cours de test', texte)
        self.assertTrue(texte)
        self.assertEqual(len(meta['actions']), 2)
        self.assertFalse(meta['actions'][-1]['succes'])
