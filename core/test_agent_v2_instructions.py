"""
Les instructions d'AGIR partent a CHAQUE requete, historique ou pas.

Banc du 2026-09-14: Agent(system_prompt=...) n'envoie le prompt systeme que
si l'historique est vide (pydantic-ai, _agent_graph). Sur un tour de suivi,
AGIR tournait sans date ni table de decision, et une puce « Jeudi » a place
une revision en 2025. Ce test aurait attrape le defaut.
"""
from unittest.mock import patch

from django.test import SimpleTestCase
from pydantic_ai.messages import (ModelRequest, ModelResponse, TextPart,
                                  UserPromptPart)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from services.agent_v2 import agent as module_agent
from services.agent_v2.agent import PlannerAgentV2
from services.agent_v2.registre import Registre


class InstructionsDeSuiviTests(SimpleTestCase):
    def test_la_date_atteint_le_modele_sur_un_tour_de_suivi(self):
        vues: list = []

        def repondre(messages, info: AgentInfo):
            vues.append(list(messages))
            return ModelResponse(parts=[TextPart(content="ok")])

        async def en_flux(messages, info: AgentInfo):
            # _agir passe un event_stream_handler: la requete est streamee.
            vues.append(list(messages))
            yield "ok"

        historique = [ModelRequest(parts=[UserPromptPart(content="place ma révision")]),
                      ModelResponse(parts=[TextPart(content="Quel jour ?")])]
        agent = PlannerAgentV2()
        agent._message_brut = "Jeudi"
        agent._tache = "t:1"
        with patch.object(module_agent, "modele_agir", return_value=FunctionModel(repondre, stream_function=en_flux)), \
             patch.object(module_agent, "prompt_agir", return_value="DATE: lundi 2026-09-14"), \
             patch.object(module_agent, "outils_pour", return_value=[]), \
             patch.object(PlannerAgentV2, "_historique", return_value=historique):
            agent._agir(None, "Jeudi", Registre())

        self.assertTrue(vues)
        derniere = [m for m in vues[-1] if isinstance(m, ModelRequest)][-1]
        self.assertIn("DATE: lundi 2026-09-14", derniere.instructions or "")
