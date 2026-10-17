"""Outil send_notification."""
from django.contrib.auth.models import User
from pydantic import BaseModel, ConfigDict, Field

from core.models import PushSubscription
from .base import (
    ISSUE_TERMINEE, ISSUE_BESOIN_PRECISION, ISSUE_CAPACITE_ABSENTE, ISSUE_ECHEC,
    ToolResult,
)

NAME = "send_notification"
DESCRIPTION = (
    "Envoie immédiatement une notification push sur les appareils de "
    "l'utilisateur (PWA installée, notifications activées). Pour une info "
    "qu'il doit voir même hors de l'app; jamais pour répéter ta réponse "
    "du chat. Ne programme PAS de rappel futur."
)

# Mêmes bornes que la vue /api/push/send/: un seul contrat quel que soit le
# canal d'entrée (agent, REST direct, MCP). Les longueurs sont validées par
# le harnais contre le schéma avant execute.
TITLE_MAX_LENGTH = 100
BODY_MAX_LENGTH = 500


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str | None = Field(
        default=None, max_length=TITLE_MAX_LENGTH,
        description="Titre court (défaut: 'Planner AI')",
    )
    body: str = Field(
        max_length=BODY_MAX_LENGTH,
        description="Texte de la notification",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    from services.push import push_configured, send_to_user

    title = (kwargs.get("title") or "").strip() or "Planner AI"
    body = (kwargs.get("body") or "").strip()
    if not body:
        return ToolResult(
            success=False, data={},
            message="body requis: le texte de la notification.",
            issue=ISSUE_BESOIN_PRECISION, cause="texte_manquant",
            public="Il me faut le texte de la notification pour l'envoyer.",
        )

    if not push_configured():
        return ToolResult(
            success=False, data={},
            message="Web push non configuré côté serveur: rien n'est parti.",
            issue=ISSUE_CAPACITE_ABSENTE, cause="push_non_configure",
            public="Les notifications ne sont pas configurées : rien n'est parti.",
        )

    # Distinguer "aucun appareil abonné" (l'utilisateur n'a rien activé)
    # d'un échec d'envoi: le message doit guider vers la vraie action.
    if not PushSubscription.objects.filter(user=user).exists():
        return ToolResult(
            success=False,
            data={"sent": 0},
            message=(
                "Aucun appareil abonné aux notifications: "
                "active-les dans ton profil."
            ),
            issue=ISSUE_ECHEC, cause="aucun_appareil",
            public=("Aucun appareil n'est abonné aux notifications : "
                    "active-les dans ton profil pour recevoir celle-ci."),
        )

    sent = send_to_user(user, title, body, url="/")
    if sent == 0:
        return ToolResult(
            success=False, data={"sent": 0},
            message="L'envoi a échoué sur tous les appareils abonnés.",
            issue=ISSUE_ECHEC, cause="envoi_echoue",
            public="L'envoi a échoué sur tous les appareils abonnés.",
        )
    return ToolResult(
        success=True,
        data={"sent": sent},
        message=f"Notification envoyée sur {sent} appareil(s).",
        issue=ISSUE_TERMINEE, cause="",
        public=f"Notification envoyée sur {sent} appareil(s).",
    )
