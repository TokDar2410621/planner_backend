"""Lire les rappels autonomes."""
from django.contrib.auth.models import User
from django.db.models import Q
from pydantic import BaseModel, Field

from core.models import Reminder
from .base import ToolResult

NAME = 'list_reminders'
DESCRIPTION = 'Liste les rappels vérifiés, y compris les manqués. Lis avant de reporter ou annuler une cible ambiguë.'


class Parametres(BaseModel):
    query: str | None = None
    limit: int = Field(default=30, ge=1, le=100)


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    qs = Reminder.objects.filter(user=user).exclude(status='cancelled')
    if kwargs.get('query'):
        qs = qs.filter(Q(title__icontains=kwargs['query']))
    entries = list(qs.order_by('due_at')[:kwargs.get('limit', 30)].values(
        'id', 'title', 'due_at', 'status', 'recurrence_id', 'occurrence_date'))
    for entry in entries:
        for key in ('due_at', 'occurrence_date'):
            if entry[key] is not None:
                entry[key] = entry[key].isoformat()
    return ToolResult(True, data={'reminders': entries, 'count': len(entries)},
                      message=f'{len(entries)} rappel(s) trouvé(s).')
