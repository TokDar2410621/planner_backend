"""
Document processing tool for the Planner AI agent.
"""
from django.contrib.auth.models import User

from core.models import UploadedDocument
from .base import BaseTool, ToolResult


class ProcessUploadedDocumentTool(BaseTool):
    name = "process_uploaded_document"
    description = "Récupère les données extraites du dernier document uploadé par l'utilisateur et les résume."
    parameters = {
        "type": "object",
        "properties": {
            "document_id": {
                "type": "integer",
                "description": "ID du document (si absent, prend le dernier document uploadé)",
            },
        },
        "required": [],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        doc_id = kwargs.get("document_id")

        if doc_id:
            try:
                doc = UploadedDocument.objects.get(id=doc_id, user=user)
            except UploadedDocument.DoesNotExist:
                return ToolResult(success=False, data={}, message=f"Document #{doc_id} introuvable.")
        else:
            doc = UploadedDocument.objects.filter(user=user).order_by('-uploaded_at').first()
            if not doc:
                return ToolResult(success=False, data={}, message="Aucun document uploadé.")

        if not doc.processed:
            return ToolResult(
                success=False,
                data={"document_id": doc.id, "status": "processing"},
                message="Le document est encore en cours de traitement. Réessaie dans quelques secondes.",
            )

        if doc.processing_error:
            return ToolResult(
                success=False,
                data={"document_id": doc.id, "error": doc.processing_error},
                message=f"Erreur lors du traitement du document: {doc.processing_error}",
            )

        return ToolResult(
            success=True,
            data={
                "document_id": doc.id,
                "file_name": doc.file_name,
                "document_type": doc.document_type,
                "extracted_data": doc.extracted_data,
            },
            message=f"Document '{doc.file_name}' traité avec succès. Données extraites disponibles.",
        )
