"""
Le registre d'un tour: ce qui s'est VRAIMENT passe.

Ecrit par le runtime a chaque execution d'outil, jamais par le modele. Chaque
entree porte un identifiant que la phase DIRE devra citer pour avoir le droit
de parler d'une action.
"""
from __future__ import annotations

from dataclasses import dataclass

from services.agent.tools.base import ToolResult

# Les outils qui ECRIVENT. Sert a declencher la reconciliation et a distinguer
# lecture et mutation dans le bloc factuel. La liste de v1 (MUTATION_TOOLS)
# oublie organize_day, optimize_week et cancel_scheduled_block, qui ecrivent
# pourtant: on les inclut ici et un test verrouille la couverture.
OUTILS_DE_MUTATION = {
    "create_block", "update_block", "delete_block", "clear_all_blocks",
    "skip_block_occurrence", "restore_block_occurrence",
    "create_task", "update_task", "delete_task", "complete_task",
    "schedule_task_at", "cancel_scheduled_block",
    "optimize_week", "organize_day",
    "update_preferences", "create_goal", "update_goal",
}


@dataclass(frozen=True)
class Action:
    id: str
    outil: str
    parametres: dict
    succes: bool
    message: str
    donnees: dict

    @property
    def est_mutation(self) -> bool:
        return self.outil in OUTILS_DE_MUTATION


@dataclass(frozen=True)
class Ecart:
    id: str
    action_id: str
    description: str


class Registre:
    def __init__(self) -> None:
        self.actions: list[Action] = []
        self.ecarts: list[Ecart] = []
        self.budget_epuise: bool = False
        self._index: dict = {}

    def ajouter(self, outil: str, parametres: dict, resultat: ToolResult) -> Action:
        action = Action(
            id=f"a{len(self.actions) + 1}",
            outil=outil,
            parametres=dict(parametres or {}),
            succes=bool(resultat.success),
            message=resultat.message or "",
            donnees=dict(resultat.data or {}),
        )
        self.actions.append(action)
        self._index[action.id] = action
        return action

    def ajouter_ecart(self, action_id: str, description: str) -> Ecart:
        ecart = Ecart(id=f"e{len(self.ecarts) + 1}",
                      action_id=action_id, description=description)
        self.ecarts.append(ecart)
        self._index[ecart.id] = ecart
        return ecart

    def mutations(self) -> list[Action]:
        return [a for a in self.actions if a.est_mutation]

    def par_id(self, ident):
        if not ident or not isinstance(ident, str):
            return None
        return self._index.get(ident)

    def vide(self) -> bool:
        return not self.actions and not self.ecarts
