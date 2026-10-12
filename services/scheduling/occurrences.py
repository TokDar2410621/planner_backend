"""Sélection des récurrences applicables à une date de début d'occurrence.

Cette couche lit uniquement les modèles. Elle ne dépend ni du rendu ni des
moteurs de placement, qui peuvent donc la partager sans appel récursif.
"""
from core.models import RecurringBlock, RecurringBlockOverride
from services.scheduling.exceptions import skipped_block_ids


def recurring_occurrences(user, target_date):
    """Occurrences visibles, actives, dans leurs bornes et non ignorées."""
    blocks = list(
        RecurringBlock.objects.filter(
            RecurringBlock.bounds_filter(target_date),
            user=user, active=True, day_of_week=target_date.weekday(),
        )
        .exclude(id__in=skipped_block_ids(user, target_date))
        .select_related("place")
        .order_by("id")
    )
    overrides = {
        override.recurring_block_id: override
        for override in RecurringBlockOverride.objects.filter(
            user=user, recurring_block__user=user, date=target_date,
            recurring_block_id__in=[block.pk for block in blocks],
        )
    }
    for block in blocks:
        override = overrides.get(block.pk)
        block.occurrence_date = target_date
        block.occurrence_override = override is not None
        if override is not None:
            # Instances fraîchement lues, réservées à la projection. Les outils
            # d'écriture relisent la série avant toute modification.
            block.start_time = override.start_time
            block.end_time = override.end_time
            block.duration_minutes = (
                (override.end_time.hour * 60 + override.end_time.minute)
                - (override.start_time.hour * 60 + override.start_time.minute)
            ) % 1440
            block.is_night_shift = override.end_time < override.start_time
    return blocks
