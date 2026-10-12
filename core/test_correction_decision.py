"""Réparation structurée bornée : SDK exécuté, données et modèles locaux."""
import asyncio
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TransactionTestCase, override_settings
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from core.models import BudgetJetonsJournalier
from services.agent_v2.conversation import AgentConversation


class ReparationDecisionTests(TransactionTestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username='decision-reparation')

    def executer(self, reponses):
        appels = []
        def repondre(messages, info):
            index = len(appels)
            appels.append(messages)
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                reponses[min(index, len(reponses) - 1)])],
                usage=RequestUsage(input_tokens=100, output_tokens=20))
        with patch('services.agent_v2.conversation.modele_agir', return_value=FunctionModel(repondre)), \
             patch('services.agent_v2.agent.PlannerAgentV2.process_message_stream',
                   return_value=iter([{'type': 'done', 'response': 'Planning traité'}])) as planning:
            instance = AgentConversation()
            resultat = instance.process_message(self.user,
                "Annule seulement le 2026-10-12 mon cours d'entriprise, garde les autres séances.")
        return instance, resultat, appels, planning

    def test_sortie_invalide_est_reparee_une_fois_avant_acces_planning(self):
        instance, resultat, appels, planning = self.executer([
            '{"mode":"planning","objectif":"Annuler une seule occurrence",'
            '"consultation":"autre","autoriser_modifications":true,'
            '"date_consultation":2026-10-12}',
            {'mode': 'planning', 'consultation': 'aucune',
             'objectif': 'Annuler une seule occurrence', 'autoriser_modifications': True},
        ])
        self.assertEqual(instance._decision_tour.mode, 'planning')
        planning.assert_called_once()
        self.assertEqual(len(appels), 2)
        self.assertTrue(any(getattr(part, 'part_kind', '') == 'retry-prompt'
                            for msg in appels[1] for part in msg.parts))
        self.assertEqual(BudgetJetonsJournalier.objects.get(user=self.user).jetons, 240)
        self.assertEqual(len(instance._trace.usages), 2)

    def test_deux_sorties_invalides_restent_sans_outil(self):
        instance, resultat, appels, planning = self.executer([
            {'mode': 'planning', 'objectif': ''},
        ])
        self.assertEqual(len(appels), 2)
        planning.assert_not_called()
        self.assertEqual(instance._decision_tour.mode, 'clarification')
        self.assertEqual(BudgetJetonsJournalier.objects.get(user=self.user).jetons, 240)

    def test_sortie_valide_ne_paie_pas_de_reparation(self):
        _, _, appels, planning = self.executer([
            {'mode': 'conversation', 'reponse': 'Salut !'},
        ])
        self.assertEqual(len(appels), 1)
        planning.assert_not_called()
        self.assertEqual(BudgetJetonsJournalier.objects.get(user=self.user).jetons, 120)

    @override_settings(AGENT_DECISION_DELAI=0.3)
    def test_reparation_partage_la_deadline_initiale(self):
        appels, annulations = [], []
        async def repondre(messages, info):
            appels.append(messages)
            if len(appels) == 1:
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                    {'mode': 'planning', 'objectif': ''})],
                    usage=RequestUsage(input_tokens=100, output_tokens=20))
            try:
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                annulations.append(True)
                raise
        with patch('services.agent_v2.conversation.modele_agir', return_value=FunctionModel(repondre)), \
             patch('services.agent_v2.agent.PlannerAgentV2.process_message_stream') as planning:
            instance = AgentConversation()
            instance.process_message(self.user, 'Annule le cours indiqué')
        self.assertEqual(len(appels), 2)
        self.assertEqual(annulations, [True])
        planning.assert_not_called()
        self.assertFalse(instance._trace.usages[-1]['usage_connu'])
        self.assertGreater(BudgetJetonsJournalier.objects.get(user=self.user).jetons, 120)
