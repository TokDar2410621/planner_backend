"""Lire les series de taches et rappels."""
from django.contrib.auth.models import User
from pydantic import BaseModel, Field

from core.models import RecurrenceSeries
from .base import ToolResult

NAME = 'list_recurrences'
DESCRIPTION = 'Liste les séries récurrentes vérifiées avant une modification ou un arrêt.'


class Parametres(BaseModel):
    query: str | None = None
    limit: int = Field(default=30, ge=1, le=100)


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    qs = RecurrenceSeries.objects.filter(user=user, active=True)
    if kwargs.get('query'):
        qs = qs.filter(title__icontains=kwargs['query'])
    entries = list(qs.order_by('start_date')[:kwargs.get('limit', 30)].values(
        'id', 'title', 'kind', 'start_date', 'end_date', 'frequency',
        'interval', 'weekdays', 'day_of_month', 'due_time', 'alert_time'))
    return ToolResult(True, data={'series': entries, 'count': len(entries)},
                      message=f'{len(entries)} série(s) trouvée(s).')
