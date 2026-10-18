from importlib import import_module

from django.apps import apps
from django.contrib.auth import get_user_model
from django.test import override_settings
from django.urls import reverse
from django.utils import timezone
from rest_framework.test import APITestCase

from core.models import BudgetJetonsJournalier, ConversationMessage


class ChatUsageTests(APITestCase):
    @override_settings(AGENT_V2_BUDGET_JETONS_JOUR=400000, AGENT_V2_GARDE_COUT_JOUR=800000)
    def test_jauge_ne_compte_pas_la_provision(self):
        user = get_user_model().objects.create_user(username='usage-jauge')
        self.client.force_authenticate(user=user)
        BudgetJetonsJournalier.objects.create(
            user=user, jour=timezone.localdate(), jetons=358000,
            jetons_mesures=5089, appels_non_mesures=6,
        )
        response = self.client.get(reverse('chat-usage'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['jetons_mesures'], 5089)
        self.assertEqual(response.data['limite_jetons'], 400000)
        self.assertEqual(response.data['appels_non_mesures'], 6)
        self.assertIsNone(response.data['bloque_par'])
        self.assertIn('reprise_a', response.data)

    def test_usage_exige_authentification(self):
        response = self.client.get(reverse('chat-usage'))
        self.assertIn(response.status_code, (401, 403))

    def test_reprise_des_mesures_anciennes_exclut_les_provisions(self):
        user = get_user_model().objects.create_user(username='legacy-usage')
        budget = BudgetJetonsJournalier.objects.create(
            user=user, jour=timezone.localdate(), jetons=358000,
        )
        ConversationMessage.objects.create(
            user=user, role='assistant', content='Terminé.',
            metadata={'usage': [
                {'usage_connu': True, 'total': 5089, 'succes': True},
                {'usage_connu': False, 'jetons_factures': 352911, 'succes': True},
            ]},
        )
        import_module('core.migrations.0046_budget_mesure').reprendre_mesures(apps, None)
        budget.refresh_from_db()
        self.assertEqual(budget.jetons_mesures, 5089)
        self.assertEqual(budget.appels_non_mesures, 1)
        self.assertTrue(budget.mesure_partielle)
