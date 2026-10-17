"""Outil delete_block."""
from django.contrib.auth.models import User
from pydantic import BaseModel, ConfigDict, Field

from core.models import RecurringBlock
from . import commun
from .base import ToolResult

NAME = "delete_block"
DESCRIPTION = "Supprime TOUTE la série d'un bloc récurrent. Résous le bloc par nom/jour/heure; ne demande jamais d'identifiant à l'utilisateur. Pour annuler un seul jour, utilise skip_block_occurrence, PAS delete_block."


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    block_id: int = Field(
        description="ID du bloc à supprimer",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    block_id = kwargs["block_id"]
    try:
        block = RecurringBlock.objects.get(id=block_id, user=user, active=True)
    except RecurringBlock.DoesNotExist:
        return ToolResult(success=False, data={}, message=f"Bloc #{block_id} introuvable.",
                          public="Je n'ai pas supprimé le créneau : je ne l'ai pas trouvé. Dis-moi lequel.")

    title = block.title
    # Lu AVANT la desactivation: le rendu nomme le jour et les heures.
    decrit = commun._block_to_dict(block)
    block.active = False
    block.save()
    return ToolResult(
        success=True,
        data={"deleted_id": block_id, "block": decrit},
        message=f"Bloc '{title}' supprimé.",
    )
