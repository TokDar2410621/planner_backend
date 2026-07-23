"""Résolution des occurrences ignorées (skip) d'un bloc récurrent — source unique.

Un `RecurringBlockException` marque qu'UNE occurrence d'un bloc récurrent
n'a pas lieu à une date donnée. Tous les sites qui matérialisent les
occurrences (agenda, créneaux libres, dispo partagée, auto-planif, iCal)
consultent ce module pour ne pas afficher/compter l'occurrence ignorée.
"""
from __future__ import annotations


def skipped_block_ids(user, target_date) -> set:
    """Ids des RecurringBlock que l'utilisateur a annulés pour `target_date`.

    `target_date` est la date de début de l'occurrence (jour de semaine ==
    day_of_week du bloc). Retourne un set vide si aucune exception.
    """
    from core.models import RecurringBlockException

    return set(
        RecurringBlockException.objects.filter(user=user, date=target_date)
        .values_list("recurring_block_id", flat=True)
    )
