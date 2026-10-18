"""Traite les alertes a echeance. A lancer environ toutes les 20 secondes."""
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from core.models import PushSendLog, Reminder, Task
from services.notification_delivery import has_device, send_visible


GRACE = timedelta(seconds=90)


class Command(BaseCommand):
    help = 'Envoie les rappels autonomes et alertes de taches a leur heure.'

    def handle(self, *args, **options):
        now = timezone.now()
        sent = missed = 0
        for pk in Reminder.objects.filter(status='pending', due_at__lte=now).values_list('pk', flat=True):
            with transaction.atomic():
                reminder = Reminder.objects.select_for_update().select_related('user').get(pk=pk)
                if reminder.status != 'pending':
                    continue
                if reminder.due_at < now - GRACE or not has_device(reminder.user):
                    reminder.status = 'missed'
                    missed += 1
                else:
                    count = send_visible(
                        reminder.user, 'Rappel Planner', reminder.title,
                        f'/schedule?view=reminders&reminder={reminder.pk}',
                    )
                    reminder.delivered_count = count
                    # Un echec reseau peut etre temporaire. Retenter pendant
                    # la fenetre de grace, puis marquer manque sans envoi tardif.
                    reminder.status = 'sent' if count else 'pending'
                    reminder.sent_at = now if count else None
                    sent += int(count > 0)
                reminder.save(update_fields=['status', 'delivered_count', 'sent_at'])

        tasks = Task.objects.filter(
            completed=False, alert_at__gt=now - GRACE, alert_at__lte=now,
        ).select_related('user').values_list('pk', flat=True)
        task_alerts = 0
        for pk in tasks:
            with transaction.atomic():
                task = Task.objects.select_for_update().select_related('user').get(pk=pk)
                if task.completed or not task.alert_at or task.alert_at > now:
                    continue
                log_fields = {
                    'user': task.user, 'kind': 'task_alert', 'ref': str(task.pk),
                    'date': timezone.localtime(task.alert_at).date(),
                }
                if not PushSendLog.objects.filter(**log_fields).exists() and has_device(task.user):
                    delivered = send_visible(
                        task.user, 'Tâche Planner', task.title,
                        f'/schedule?task={task.pk}',
                    )
                    if delivered:
                        PushSendLog.objects.create(**log_fields)
                        task_alerts += delivered
        self.stdout.write(
            f'Alertes: {sent} rappel(s) envoyes, {missed} manque(s), '
            f'{task_alerts} appareil(s) pour les taches.'
        )
