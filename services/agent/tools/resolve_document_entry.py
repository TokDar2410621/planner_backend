"""Outil resolve_document_entry."""
import logging
from typing import Literal

from django.contrib.auth.models import User
from pydantic import BaseModel, ConfigDict, Field

from core.models import UploadedDocument
from .base import ToolResult

logger = logging.getLogger(__name__)

NAME = "resolve_document_entry"
DESCRIPTION = (
    "Complète une entrée documentaire en attente avec les précisions explicitement données "
    "par l'utilisateur. Lis process_uploaded_document pour obtenir document_id, source, index "
    "et missing_fields. Transmets seulement les champs manquants effectivement précisés, "
    "sans inventer heure, jour, date ou durée. Une réponse partielle conserve la question "
    "restante; une réponse complète crée le créneau, lie sa provenance et clôt la précision. "
    "Utilise cet outil pour compléter un import, afin de ne pas recréer l'entrée avec un autre outil."
)

SourceDocumentaire = Literal["courses", "shifts", "events"]


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_id: int = Field(description="Document de la précision lue.")
    source: SourceDocumentaire
    index: int = Field(ge=0, description="Index stable de l'entrée dans le document.")
    date: str | None = Field(
        default=None,
        description="Date explicite YYYY-MM-DD, si manquante.",
    )
    day: str | None = Field(
        default=None,
        description="Jour hebdomadaire explicite, si manquant.",
    )
    start_time: str | None = Field(
        default=None,
        description="Heure de début fournie, HH:MM.",
    )
    end_time: str | None = Field(
        default=None,
        description="Heure de fin fournie, HH:MM.",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
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
