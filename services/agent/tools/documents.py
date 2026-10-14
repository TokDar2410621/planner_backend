"""
Document processing tool for the Planner AI agent.
"""
import logging
from django.contrib.auth.models import User

from core.models import UploadedDocument
from .base import (BaseTool, ToolResult, ISSUE_TERMINEE, ISSUE_ECHEC,
                   ISSUE_BESOIN_PRECISION, ISSUE_EFFET_A_VERIFIER)

logger = logging.getLogger(__name__)


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


class ResolveDocumentEntryTool(BaseTool):
    name = 'resolve_document_entry'
    description = (
        "Complète une entrée documentaire en attente avec les précisions explicitement données "
        "par l'utilisateur. Lis process_uploaded_document pour obtenir document_id, source, index "
        "et missing_fields. Transmets seulement les champs manquants effectivement précisés, "
        "sans inventer heure, jour, date ou durée. Une réponse partielle conserve la question "
        "restante; une réponse complète crée le créneau, lie sa provenance et clôt la précision. "
        "Utilise cet outil pour compléter un import, afin de ne pas recréer l'entrée avec un autre outil."
    )
    parameters = {
        'type': 'object',
        'properties': {
            'document_id': {'type': 'integer', 'description': 'Document de la précision lue.'},
            'source': {'type': 'string', 'enum': ['courses', 'shifts', 'events']},
            'index': {'type': 'integer', 'minimum': 0, 'description': "Index stable de l'entrée dans le document."},
            'date': {'type': 'string', 'description': 'Date explicite YYYY-MM-DD, si manquante.'},
            'day': {'type': 'string', 'description': 'Jour hebdomadaire explicite, si manquant.'},
            'start_time': {'type': 'string', 'description': 'Heure de début fournie, HH:MM.'},
            'end_time': {'type': 'string', 'description': 'Heure de fin fournie, HH:MM.'},
        },
        'required': ['document_id', 'source', 'index'],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        from services.document_processor import DocumentProcessor, DocumentResolutionError
        from services.agent_v2.importation import resume_import, message_import, message_precision
        values = {key: value for key, value in kwargs.items()
                  if key not in ('document_id', 'source', 'index')}
        try:
            # La résolution relit et matérialise uniquement des données :
            # aucun client fournisseur ni nouvel appel d'extraction.
            processor = DocumentProcessor.__new__(DocumentProcessor)
            result = processor.resolve_entry(user, kwargs.get('document_id'),
                kwargs.get('source'), kwargs.get('index'), values)
        except DocumentResolutionError as exc:
            # Les messages de DocumentResolutionError sont deja des phrases
            # francaises sures pour l'utilisateur (voir document_processor).
            return ToolResult(success=False, data={'status': 'refused'}, message=str(exc),
                              public=f"La précision n'a pas été enregistrée : {exc}")
        except Exception as exc:
            logger.warning('Résolution documentaire échouée (%s)', type(exc).__name__)
            return ToolResult(success=False, data={'status': 'failed'},
                              message="La précision n'a pas pu être enregistrée. Réessaie.",
                              public="La précision n'a pas pu être enregistrée. Réessaie.")
        resolution = result['resolution']
        doc = UploadedDocument.objects.get(pk=resolution['document_id'], user=user)
        summary = resume_import(doc)
        ids = {t['id'] for t in resolution['targets'] if t['model'] == 'RecurringBlock'}
        date_ids = {t['id'] for t in resolution['targets'] if t['model'] == 'ScheduledBlock'}
        precisions = [p for p in summary['precisions'] if p['source'] == resolution['source']
                      and p['index'] == resolution['index']]
        a_confirmer = sum(t.get('status') == 'pending' for t in resolution['targets'])
        entry_summary = {**summary, 'blocs': [b for b in summary['blocs'] if b['id'] in ids],
                         'dates': [d for d in summary['dates'] if d['id'] in date_ids],
                         'en_attente': len(precisions) + a_confirmer,
                         'blocs_a_confirmer': a_confirmer, 'precisions': precisions, 'ignores': []}
        result.update(document_id=doc.pk, entry_summary=entry_summary,
                      pending_count=len(summary['precisions']))
        if resolution['pending']:
            message = 'Précision enregistrée. ' + message_precision(precisions[0])
        elif result['already_resolved']:
            message = 'Cette entrée est déjà résolue et son créneau a été vérifié.'
        else:
            message = message_import(entry_summary)
        return ToolResult(success=True, data=result, message=message)
