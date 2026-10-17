"""Outil clear_all_blocks."""
from django.contrib.auth.models import User
from django.db import transaction
from pydantic import BaseModel, ConfigDict, Field

from core.models import RecurringBlock
from .base import ToolResult

NAME = "clear_all_blocks"
DESCRIPTION = (
    "Archive TOUS les blocs récurrents du planning (soft-delete réversible: "
    "les blocs sont désactivés, pas détruits). Opération destructive à fort "
    "impact: elle DOIT être confirmée par l'utilisateur hors-bande (via un "
    "dialogue de confirmation côté frontend), jamais sur la seule initiative "
    "du modèle. N'utilise cet outil que si l'utilisateur le demande "
    "explicitement (ex: 'vide tout mon planning', 'recommencer à zéro')."
)
# Flag surfaced to the orchestration/frontend layer: this call must be gated
# by an explicit out-of-band human confirmation, not an LLM-supplied boolean.
REQUIRES_CONFIRMATION = True


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    confirm: bool = Field(
        description=(
            "Doit être true. Ne mets true que si l'utilisateur a "
            "explicitement confirmé vouloir tout archiver."
        ),
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    if not kwargs.get("confirm"):
        return ToolResult(
            success=False,
            message="Suppression annulée: confirm doit être true.",
            public="Je n'ai rien archivé : il me faut ta confirmation d'abord.",
        )

    # Soft-delete (active=False) au lieu d'un DELETE: réversible, restaurable.
    # Opération multi-lignes → atomique.
    with transaction.atomic():
        count = RecurringBlock.objects.filter(
            user=user, active=True
        ).update(active=False)
        # update() n'emet aucun signal: le reveil du telephone se demande
        # a la main, sinon les alarmes des blocs archives sonneraient encore.
        from services.apns import reveiller_utilisateur
        transaction.on_commit(lambda: reveiller_utilisateur(user, "planning"))
    return ToolResult(
        success=True,
        data={"deleted_count": count, "reversible": True},
        message=(
            f"{count} bloc(s) archivé(s). Planning vidé "
            f"(action réversible: les blocs sont désactivés, pas détruits)."
        ),
    )
