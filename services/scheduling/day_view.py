"""Emploi du temps EFFECTIF d'une journée (placement-aware, skip-aware).

Source de vérité factuelle partagée par les lectures jour, semaine et le pied
de page de l'agent. Les blocs souples apparaissent à leur heure placée.
"""
from core.models import ScheduledBlock
from services.scheduling.occurrences import recurring_occurrences
from services.scheduling.placement import place_day


def effective_day_entries(user, target_date) -> list[dict]:
    """Blocs effectifs avec identité typée, pour les outils jour et semaine."""
    recurring = recurring_occurrences(user, target_date)

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
            "id": b.id,
            "title": b.title,
            "start_time": start_time,
            "end_time": end_time,
            "type": "recurring",
            "block_type": b.block_type,
            "is_flexible": b.is_flexible,
            "location": b.location or None,
        })

    for sb in ScheduledBlock.objects.filter(
        user=user, date=target_date
    ).select_related("task"):
        entries.append({
            "id": sb.id,
            "title": sb.task.title,
            "start_time": sb.start_time.strftime("%H:%M"),
            "end_time": sb.end_time.strftime("%H:%M"),
            "type": "scheduled_task",
            "task_id": sb.task_id,
            "task_type": sb.task.task_type,
            "completed": sb.actually_completed or sb.task.completed,
        })

    entries.sort(key=lambda e: e["start_time"])
    return entries


def effective_day_blocks(user, target_date) -> list[dict]:
    """Projection compacte conservée pour le pied de page de l'agent."""
    return [
        {
            "title": entry["title"],
            "start_time": entry["start_time"],
            "end_time": entry["end_time"],
            "kind": "recurring" if entry["type"] == "recurring" else "scheduled",
        }
        for entry in effective_day_entries(user, target_date)
    ]
