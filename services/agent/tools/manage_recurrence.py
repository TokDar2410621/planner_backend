"""Creer ou arreter une serie de taches ou de rappels."""
from datetime import date, time, timedelta
from typing import Literal

from django.contrib.auth.models import User
from django.db import transaction
from django.utils import timezone
from pydantic import BaseModel, Field

from core.models import RecurrenceSeries, Reminder, Task
from services.reminder_series import materialize, validate_series
from .base import ToolResult

NAME = 'manage_recurrence'
DESCRIPTION = (
    "Crée une série de tâches à cocher ou de rappels autonomes, ou arrête une "
    "série identifiée via list_recurrences. Chaque occurrence garde son état. "
    "Un créneau qui bloque du temps se crée avec create_block. Si la portée "
    "(cette occurrence ou toute la série) est ambiguë, demande avant d'agir."
)


class Parametres(BaseModel):
    action: Literal['create', 'update', 'stop']
    series_id: int | None = None
    kind: Literal['task', 'reminder'] | None = None
    title: str | None = Field(default=None, max_length=200)
    start_date: date | None = None
    end_date: date | None = None
    frequency: Literal['daily', 'weekly', 'monthly'] | None = None
    interval: int = Field(default=1, ge=1, le=365)
    weekdays: list[int] = Field(default_factory=list, description='0=lundi ... 6=dimanche')
    day_of_month: int | None = Field(default=None, ge=1, le=31)
    due_time: time | None = Field(default=None, description='Heure du rappel ou échéance de la tâche')
    alert_time: time | None = Field(default=None, description='Alerte plus tôt pour une tâche')


PARAMETERS = Parametres.model_json_schema()


@transaction.atomic
def execute(user: User, **kwargs) -> ToolResult:
    if kwargs['action'] in ('stop', 'update'):
        series = RecurrenceSeries.objects.filter(user=user, pk=kwargs.get('series_id')).first()
        if series is None:
            return ToolResult(False, message='Série introuvable.', public="Je n'ai pas trouvé cette série.")
        if kwargs['action'] == 'update':
            allowed = ('title', 'end_date', 'frequency', 'interval', 'weekdays',
                       'day_of_month', 'due_time', 'alert_time')
            changes = {key: kwargs[key] for key in allowed if key in kwargs}
            if not changes:
                return ToolResult(False, message='Aucun changement demandé.',
                                  public='Dis-moi ce que tu veux modifier dans la répétition.')
            proposed = {key: changes.get(key, getattr(series, key)) for key in allowed}
            try:
                validate_series(kind=series.kind, start_date=series.start_date,
                                end_date=proposed['end_date'], frequency=proposed['frequency'],
                                interval=proposed['interval'], weekdays=proposed['weekdays'],
                                day_of_month=proposed['day_of_month'],
                                due_time=proposed['due_time'], alert_time=proposed['alert_time'])
            except ValueError as error:
                return ToolResult(False, message=str(error), public=str(error))
            today = timezone.localdate()
            for key, value in changes.items():
                setattr(series, key, value)
            series.generated_through = today - timedelta(days=1)
            series.save(update_fields=[*changes, 'generated_through'])
            Reminder.objects.filter(recurrence=series, occurrence_date__gte=today,
                                    status='pending').delete()
            Task.objects.filter(recurrence=series, occurrence_date__gte=today,
                                completed=False).delete()
            count = materialize(series, today + timedelta(days=35))
            return ToolResult(True, data={'series': {'id': series.pk, 'title': series.title,
                              'kind': series.kind, 'active': True}, 'occurrences_created': count},
                              message=f"Série « {series.title} » modifiée.",
                              public=f"Répétition mise à jour : {count} occurrence(s) à venir.")
        series.active = False
        series.save(update_fields=['active'])
        today = timezone.localdate()
        Reminder.objects.filter(recurrence=series, occurrence_date__gte=today,
                                status='pending').update(status='cancelled')
        Task.objects.filter(recurrence=series, occurrence_date__gte=today,
                            completed=False).delete()
        return ToolResult(True, data={'series': {'id': series.pk, 'active': False}},
                          message=f"Série « {series.title} » arrêtée.", public='Série arrêtée.')
    required = ('kind', 'title', 'start_date', 'frequency')
    if any(not kwargs.get(key) for key in required):
        return ToolResult(False, message='Il manque type, titre, date ou fréquence.',
                          public='Précise le type, le titre, le début et la répétition.')
    if kwargs['kind'] == 'reminder' and not kwargs.get('due_time'):
        return ToolResult(False, message='Heure manquante.', public='À quelle heure veux-tu le rappel ?')
    try:
        validate_series(kind=kwargs['kind'], start_date=kwargs['start_date'],
                        end_date=kwargs.get('end_date'), frequency=kwargs['frequency'],
                        interval=kwargs.get('interval', 1), weekdays=kwargs.get('weekdays') or [],
                        day_of_month=kwargs.get('day_of_month'),
                        due_time=kwargs.get('due_time'), alert_time=kwargs.get('alert_time'))
    except ValueError as error:
        return ToolResult(False, message=str(error), public=str(error))
    if kwargs['start_date'] < timezone.localdate():
        return ToolResult(False, message='Début dans le passé.', public='Choisis une date de début à venir.')
    fields = {key: kwargs[key] for key in (
        'kind', 'title', 'start_date', 'end_date', 'frequency', 'interval',
        'weekdays', 'day_of_month', 'due_time', 'alert_time') if key in kwargs}
    series = RecurrenceSeries.objects.create(user=user, **fields)
    count = materialize(series, timezone.localdate() + timedelta(days=35))
    return ToolResult(True, data={'series': {'id': series.pk, 'title': series.title,
                      'kind': series.kind, 'active': True}, 'occurrences_created': count},
                      message=f"Série « {series.title} » créée ({count} occurrence(s) à venir).",
                      public=f"Répétition créée : {count} occurrence(s) à venir.")
