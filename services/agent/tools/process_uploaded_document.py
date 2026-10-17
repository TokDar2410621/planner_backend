"""Outil process_uploaded_document."""
from django.contrib.auth.models import User
from pydantic import BaseModel, ConfigDict, Field

from core.models import UploadedDocument
from .base import (ToolResult, ISSUE_TERMINEE, ISSUE_ECHEC,
                   ISSUE_BESOIN_PRECISION, ISSUE_EFFET_A_VERIFIER)

NAME = "process_uploaded_document"
DESCRIPTION = "Récupère les données extraites du dernier document uploadé par l'utilisateur et les résume."


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_id: int | None = Field(
        default=None,
        description="ID du document (si absent, prend le dernier document uploadé)",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    doc_id = kwargs.get("document_id")

    if doc_id:
        try:
            doc = UploadedDocument.objects.get(id=doc_id, user=user)
        except UploadedDocument.DoesNotExist:
            return ToolResult(success=False, data={}, message=f"Document #{doc_id} introuvable.",
                              issue=ISSUE_ECHEC, cause="document_introuvable",
                              public="Je n'ai pas retrouvé ce document.")
    else:
        doc = UploadedDocument.objects.filter(user=user).order_by('-uploaded_at').first()
        if not doc:
            return ToolResult(success=False, data={}, message="Aucun document uploadé.",
                              issue=ISSUE_BESOIN_PRECISION, cause="aucun_document",
                              public="Aucun document n'a été téléversé pour le moment.")

    if doc.processing_error:
        # Un échec garde processed=False. Le détail historique peut contenir
        # une URL ou un secret fournisseur : aucun retour brut à l'agent.
        from services.document_processor import GENERIC_PROCESSING_ERROR
        return ToolResult(
            success=False,
            data={"document_id": doc.id, "status": "failed", "error": GENERIC_PROCESSING_ERROR},
            message=GENERIC_PROCESSING_ERROR,
            issue=ISSUE_ECHEC, cause="traitement_echoue",
            public="Le traitement du document a échoué.",
        )

    if not doc.processed:
        return ToolResult(
            success=False,
            data={"document_id": doc.id, "status": "processing"},
            message="Le document est encore en cours de traitement. Réessaie dans quelques secondes.",
            issue=ISSUE_EFFET_A_VERIFIER, cause="traitement_en_cours",
            public="Le document est encore en cours de traitement, réessaie dans quelques secondes.",
        )

    from services.agent_v2.importation import resume_import, message_import
    summary = resume_import(doc)
    pending = summary.get('precisions') or []
    to_confirm = summary.get('blocs_a_confirmer', 0)
    status = ('needs_clarification' if pending else
              'needs_confirmation' if to_confirm else 'processed')

    # Contrat d'issue (audit 2026-10-13): message_import est deja ecrit
    # pour etre lu par l'utilisateur, on le declare affichable.
    return ToolResult(
        success=True,
        data={
            "document_id": doc.id,
            "file_name": doc.file_name,
            "document_type": doc.document_type,
            "extracted_data": doc.extracted_data,
            "status": status,
            "pending_count": len(pending),
            "pending_entries": pending,
            "blocks_to_confirm": to_confirm,
            "import_summary": summary,
        },
        message=message_import(summary),
        issue=ISSUE_TERMINEE, cause="", public=message_import(summary),
    )
