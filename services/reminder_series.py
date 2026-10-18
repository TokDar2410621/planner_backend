"""Materialisation des rappels et taches recurrents.

La serie ne remplace jamais ses anciennes occurrences. Chaque occurrence est
une ligne independante, donc une tache non terminee reste en retard quand la
suivante est creee.
"""
import calendar
from datetime import date, datetime, time, timedelta

from django.db import transaction
from django.utils import timezone

from core.models import RecurrenceSeries, Reminder, Task


def wall_datetime(day: date, hour: time) -> datetime:
    """Associe une date et une heure murale dans le fuseau du produit."""
    return timezone.make_aware(datetime.combine(day, hour), timezone.get_default_timezone())


def occurs_on(series: RecurrenceSeries, day: date) -> bool:
    """Vrai si la regle de la serie couvre le jour donne."""
    if day < series.start_date or (series.end_date and day > series.end_date):
        return False
    if series.frequency == 'daily':
        return (day - series.start_date).days % series.interval == 0
    if series.frequency == 'weekly':
        anchor = series.start_date - timedelta(days=series.start_date.weekday())
        week = (day - anchor).days // 7
        return week % series.interval == 0 and day.weekday() in (
            series.weekdays or [series.start_date.weekday()]
        )
    if series.frequency == 'monthly':
        months = (day.year - series.start_date.year) * 12 + day.month - series.start_date.month
        wanted = series.day_of_month or series.start_date.day
        return months % series.interval == 0 and day.day == min(
            wanted, calendar.monthrange(day.year, day.month)[1]
        )
    return False


def validate_series(*, kind: str, start_date: date, end_date: date | None,
                    frequency: str, interval: int, weekdays: list[int],
                    day_of_month: int | None, due_time: time | None,
                    alert_time: time | None) -> None:
    """Valide la forme de la regle sans interpreter le texte de l'utilisateur."""
    if kind not in ('task', 'reminder'):
        raise ValueError('Type de série inconnu.')
    if frequency not in ('daily', 'weekly', 'monthly'):
        raise ValueError('Fréquence inconnue.')
    if not 1 <= interval <= 365:
        raise ValueError('Intervalle hors limites.')
    if end_date and end_date < start_date:
        raise ValueError('La fin précède le début.')
    if not isinstance(weekdays, list) or any(
            type(day) is not int or day < 0 or day > 6 for day in weekdays):
        raise ValueError('Jour de semaine invalide.')
    if day_of_month is not None and not 1 <= day_of_month <= 31:
        raise ValueError('Jour du mois invalide.')
    if kind == 'reminder' and due_time is None:
        raise ValueError('Il faut une heure pour créer un rappel.')
    if alert_time and due_time and alert_time > due_time:
        raise ValueError("L'alerte ne peut pas suivre l'échéance.")


@transaction.atomic
def materialize(series: RecurrenceSeries, through: date) -> int:
    """Cree toutes les occurrences manquantes jusqu'a `through`, sans doublon."""
    series = RecurrenceSeries.objects.select_for_update().get(pk=series.pk)
    if not series.active:
        return 0
    first = max(series.start_date, (series.generated_through + timedelta(days=1))
                if series.generated_through else series.start_date)
    last = min(through, series.end_date) if series.end_date else through
    if first > last:
        return 0
    count = 0
    day = first
    while day <= last:
        if occurs_on(series, day):
            if series.kind == 'reminder':
                _, created = Reminder.objects.get_or_create(
                    recurrence=series, occurrence_date=day,
                    defaults={
                        'user': series.user, 'title': series.title,
                        'due_at': wall_datetime(day, series.due_time),
                    },
                )
            else:
                deadline = wall_datetime(day, series.due_time or time(23, 59, 59))
                alert_hour = series.alert_time or series.due_time
                _, created = Task.objects.get_or_create(
                    recurrence=series, occurrence_date=day,
                    defaults={
                        'user': series.user, 'title': series.title,
                        'deadline': deadline,
                        'alert_at': wall_datetime(day, alert_hour) if alert_hour else None,
                    },
                )
            count += int(created)
        day += timedelta(days=1)
    series.generated_through = last
    series.save(update_fields=['generated_through'])
    return count


def materialize_active(through: date | None = None) -> int:
    """Entretient toutes les series, meme sans ouverture de l'application."""
    horizon = through or (timezone.localdate() + timedelta(days=35))
    return sum(materialize(series, horizon) for series in
               RecurrenceSeries.objects.filter(active=True).iterator())
