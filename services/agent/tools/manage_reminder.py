"""Creer, reporter ou annuler un rappel autonome."""
from datetime import datetime
from typing import Literal

from django.contrib.auth.models import User
from django.utils import timezone
from pydantic import BaseModel, Field

from core.models import Reminder
from .base import ToolResult

NAME = 'manage_reminder'
DESCRIPTION = (
    "Crée un rappel autonome à une date ET une heure explicites, ou reporte/annule "
    "un rappel vérifié via list_reminders. Si l'heure manque, demande-la avec "
    "poser_question ou present_form avant de créer. Une tâche à faire sans demande "
    "d'alerte est une tâche, pas un rappel. Pour une répétition, utilise manage_recurrence."
)


class Parametres(BaseModel):
    action: Literal['create', 'update', 'cancel']
    reminder_id: int | None = Field(default=None, description='ID lu dans list_reminders pour update/cancel')
    title: str | None = Field(default=None, max_length=200)
    due_at: str | None = Field(default=None, description='YYYY-MM-DDTHH:MM dans le fuseau America/Toronto')


PARAMETERS = Parametres.model_json_schema()


def _date(value: str):
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError('Indique une date et une heure valides.') from None
    if timezone.is_naive(dt):
        dt = timezone.make_aware(dt, timezone.get_default_timezone())
    if dt <= timezone.now():
        raise ValueError('Le rappel doit être dans le futur.')
    return dt


def execute(user: User, **kwargs) -> ToolResult:
    action = kwargs['action']
    if action == 'create':
        title = (kwargs.get('title') or '').strip()
        if not title or not kwargs.get('due_at'):
            return ToolResult(False, message="Le titre, le jour et l'heure du rappel sont nécessaires.",
                              public="Dis-moi quoi te rappeler, quel jour et à quelle heure.")
        try:
            due_at = _date(kwargs['due_at'])
        except ValueError as error:
            return ToolResult(False, message=str(error), public=str(error))
        reminder = Reminder.objects.create(user=user, title=title, due_at=due_at)
        return ToolResult(True, data={'reminder': {'id': reminder.pk, 'title': title,
                          'due_at': due_at.isoformat(), 'status': reminder.status}},
                          message=f"Rappel « {title} » enregistré pour {timezone.localtime(due_at):%d/%m/%Y à %H:%M}.",
                          public=f"Rappel enregistré pour le {timezone.localtime(due_at):%d/%m/%Y à %H:%M}.")

    reminder = Reminder.objects.filter(user=user, pk=kwargs.get('reminder_id')).first()
    if reminder is None:
        return ToolResult(False, message='Rappel introuvable.', public="Je n'ai pas trouvé ce rappel.")
    if action == 'cancel':
        if reminder.status == 'cancelled':
            return ToolResult(True, data={'reminder': {'id': reminder.pk, 'status': 'cancelled'}},
                              message='Rappel déjà annulé.', public='Ce rappel était déjà annulé.')
        reminder.status = 'cancelled'
        reminder.save(update_fields=['status'])
        return ToolResult(True, data={'reminder': {'id': reminder.pk, 'status': 'cancelled'}},
                          message=f"Rappel « {reminder.title} » annulé.", public='Rappel annulé.')
    if action == 'update':
        if not kwargs.get('due_at') and not kwargs.get('title'):
            return ToolResult(False, message='Aucun changement demandé.', public='Que veux-tu changer ?')
        if reminder.due_at <= timezone.now() and not kwargs.get('due_at'):
            return ToolResult(False, message='Échéance passée.',
                              public='Choisis une nouvelle date et une heure pour reporter ce rappel.')
        try:
            if kwargs.get('due_at'):
                reminder.due_at = _date(kwargs['due_at'])
        except ValueError as error:
            return ToolResult(False, message=str(error), public=str(error))
        if kwargs.get('title'):
            reminder.title = kwargs['title'].strip()
        reminder.status = 'pending'
        reminder.sent_at = None
        reminder.delivered_count = 0
        reminder.save()
        return ToolResult(True, data={'reminder': {'id': reminder.pk, 'title': reminder.title,
                          'due_at': reminder.due_at.isoformat(), 'status': reminder.status}},
                          message=f"Rappel « {reminder.title} » reporté.", public='Rappel mis à jour.')
    return ToolResult(False, message='Action inconnue.')
