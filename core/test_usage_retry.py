from django.test import SimpleTestCase

from services.agent_v2.budget_tour import BudgetEpuise
from services.agent_v2.retry import est_reessayable


class UsageRetryTests(SimpleTestCase):
    def test_budget_local_ne_declenche_pas_dix_reprises(self):
        self.assertFalse(est_reessayable(BudgetEpuise('garde de coût atteint', 'garde_cout')))
        self.assertFalse(est_reessayable(BudgetEpuise('quota atteint', 'quota_mesure')))
