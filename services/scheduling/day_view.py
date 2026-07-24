"""Emploi du temps EFFECTIF d'une journée (placement-aware, skip-aware).

Source de vérité factuelle partagée: les blocs souples apparaissent à leur heure
PLACÉE (place_day), les occurrences ignorées sont exclues, et les événements
datés (ScheduledBlock) sont inclus. Utilisé par le pied de page "planning réel"
de l'agent (anti créneau fantôme, OR2). La logique reflète celle de
GetTodayScheduleTool; garder les deux cohérentes.
"""
from core.models import RecurringBlock, ScheduledBlock
from services.scheduling.exceptions import skipped_block_ids
from services.scheduling.placement import place_day


def effective_day_blocks(user, target_date) -> list[dict]:
    """Retourne les blocs du jour à leur heure EFFECTIVE, triés par début.

    Chaque item: {title, start_time 'HH:MM', end_time 'HH:MM', kind}.
    """
    day_of_week = target_date.weekday()
    recurring = RecurringBlock.objects.filter(
        user=user, day_of_week=day_of_week, active=True
    ).exclude(id__in=skipped_block_ids(user, target_date))

    placements = {p["block_id"]: p for p in place_day(user, target_date)}

    entries = []
    for b in recurring:
        start_time = b.start_time.strftime("%H:%M")
        end_time = b.end_time.strftime("%H:%M")
        if b.is_flexible:
            placement = placements.get(b.id)
            if placement is None or placement["skipped"]:
                continue
            start_time = placement["start_time"]
            end_time = placement["end_time"]
        entries.append({
            "title": b.title,
            "start_time": start_time,
            "end_time": end_time,
            "kind": "recurring",
        })

    for sb in ScheduledBlock.objects.filter(
        user=user, date=target_date
    ).select_related("task"):
        entries.append({
            "title": sb.task.title,
            "start_time": sb.start_time.strftime("%H:%M"),
            "end_time": sb.end_time.strftime("%H:%M"),
            "kind": "scheduled",
        })

    entries.sort(key=lambda e: e["start_time"])
    return entries
