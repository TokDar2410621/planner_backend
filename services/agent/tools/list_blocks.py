"""Outil list_blocks."""
from django.contrib.auth.models import User
from pydantic import BaseModel, ConfigDict, Field

from core.models import RecurringBlock
from . import commun
from .base import ToolResult

NAME = "list_blocks"
DESCRIPTION = "Liste les blocs récurrents du planning de l'utilisateur. Peut filtrer par jour de la semaine (0=Lundi, 6=Dimanche) ou type de bloc."


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    day_of_week: commun.JourChaine = Field(
        default=None,
        description="Filtrer par jour. Nom de preference (\"samedi\"); les numeros restent acceptes (0=Lundi..6=Dimanche).",
    )
    block_type: commun.TypeBloc | None = Field(
        default=None,
        description="Filtrer par type de bloc",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    blocks = RecurringBlock.objects.filter(user=user, active=True)

    # Noms acceptes ici aussi: un filtre silencieusement faux rend une
    # reponse fausse sans le signaler, et melanger les conventions entre
    # outils reintroduirait la confusion qu'on supprime.
    jours = commun.normaliser_jours(kwargs.get("day_of_week"))
    if jours:
        blocks = blocks.filter(day_of_week=jours[0])

    block_type = kwargs.get("block_type")
    if block_type:
        blocks = blocks.filter(block_type=block_type)

    blocks = blocks.order_by("day_of_week", "start_time")
    block_list = [commun._block_to_dict(b) for b in blocks]

    return ToolResult(
        success=True,
        data={"blocks": block_list, "count": len(block_list)},
        message=f"{len(block_list)} bloc(s) trouvé(s).",
    )
