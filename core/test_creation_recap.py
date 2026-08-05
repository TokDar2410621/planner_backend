"""
Récap de création déterministe: après un create_block multi-jours, la réponse
annexe la liste RÉELLE des blocs créés (sourcée des args exécutés), pour que
toute erreur de compréhension du LLM soit immédiatement visible.
"""
from django.test import TestCase

from services.agent.agent import _creation_recap_footer


def _call(tool, args, success=True):
    return {"tool": tool, "args": args, "result": {"success": success, "data": {}}}


class CreationRecapFooterTest(TestCase):
    def test_multi_day_creation_gets_recap(self):
        calls = [
            _call("create_block", {"title": "Cours de maths", "days": [0], "start_time": "09:00", "end_time": "12:00"}),
            _call("create_block", {"title": "Cours de maths", "days": [1, 3], "start_time": "13:00", "end_time": "16:00"}),
            _call("create_block", {"title": "Travail resto", "days": [4], "start_time": "18:00", "end_time": "02:00"}),
        ]
        recap = _creation_recap_footer(calls)
        self.assertIn("Enregistré tel quel", recap)
        self.assertIn("Cours de maths : lun 09:00-12:00", recap)
        self.assertIn("Cours de maths : mar, jeu 13:00-16:00", recap)
        self.assertIn("Travail resto : ven 18:00-02:00", recap)

    def test_single_day_creation_no_recap(self):
        # 1 seul jour: le footer réalité couvre déjà; pas de doublon.
        calls = [_call("create_block", {"title": "Sport", "days": [2], "start_time": "17:00", "end_time": "18:00"})]
        self.assertEqual(_creation_recap_footer(calls), "")

    def test_failed_creation_ignored(self):
        calls = [
            _call("create_block", {"title": "A", "days": [0, 1], "start_time": "09:00", "end_time": "10:00"}, success=False),
        ]
        self.assertEqual(_creation_recap_footer(calls), "")

    def test_non_create_tools_ignored(self):
        calls = [_call("update_block", {"days": [0, 1]})]
        self.assertEqual(_creation_recap_footer(calls), "")
