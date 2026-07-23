"""Détection de chevauchement horaire — source de vérité unique.

Évite les "tâches sur tâches". Gère correctement les blocs qui traversent
minuit (night shift) en représentant chaque bloc hebdomadaire par un ou
plusieurs intervalles en "minutes de la semaine" (0..10080), où un bloc de
nuit déborde sur le jour suivant (avec rebouclage Dimanche -> Lundi).

Pur (pas d'I/O hormis la requête ORM dans find_recurring_conflicts), donc
facilement testable.
"""
from __future__ import annotations

from datetime import datetime, time
from typing import Iterable

MINUTES_PER_DAY = 1440
MINUTES_PER_WEEK = 7 * MINUTES_PER_DAY


def time_to_min(t: time) -> int:
    """Minutes depuis minuit (0..1439)."""
    return t.hour * 60 + t.minute


def parse_time(value) -> time:
    """Accepte un datetime.time ou une chaîne 'HH:MM' / 'HH:MM:SS'."""
    if isinstance(value, time):
        return value
    s = str(value).strip()
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Heure invalide: {value!r} (attendu 'HH:MM')")


def is_overnight(start: time, end: time, night_flag: bool = False) -> bool:
    """Un bloc traverse minuit si end <= start (ou si explicitement marqué)."""
    return bool(night_flag) or time_to_min(end) <= time_to_min(start)


def recurring_intervals(day_of_week: int, start: time, end: time,
                        night_flag: bool = False) -> list[tuple[int, int]]:
    """Intervalles [début, fin) en minutes de la semaine pour un bloc récurrent.

    - Bloc normal -> un seul intervalle dans le jour.
    - Bloc de nuit -> deux morceaux : [start, fin de journée) le jour J et
      [début de journée, end) le jour J+1 (rebouclage Dimanche->Lundi).
    """
    s = time_to_min(start)
    e = time_to_min(end)
    base = day_of_week * MINUTES_PER_DAY

    if not is_overnight(start, end, night_flag):
        if e == s:
            return []  # bloc de durée nulle -> aucun chevauchement
        return [(base + s, base + e)]

    # Overnight : déborde sur le jour suivant (rebouclage en fin de semaine).
    next_day = (day_of_week + 1) % 7
    next_base = next_day * MINUTES_PER_DAY
    first = (base + s, base + MINUTES_PER_DAY)
    second = (next_base, next_base + e)
    return [iv for iv in (first, second) if iv[1] > iv[0]]


def intervals_conflict(a: Iterable[tuple[int, int]],
                       b: Iterable[tuple[int, int]]) -> bool:
    """True si un intervalle de a chevauche un intervalle de b."""
    a = list(a)
    b = list(b)
    for a0, a1 in a:
        for b0, b1 in b:
            if a0 < b1 and b0 < a1:
                return True
    return False


def find_recurring_conflicts(
    user,
    day_of_week: int,
    start,
    end,
    night_flag: bool = False,
    exclude_id=None,
    *,
    new_is_flexible: bool = False,
):
    """Retourne les RecurringBlock actifs en chevauchement avec le bloc proposé.

    `start`/`end` peuvent être des time ou des chaînes 'HH:MM'. Gère l'overnight
    et le débordement sur le jour adjacent (donc on inspecte J-1, J, J+1).
    """
    if new_is_flexible:
        return []

    from core.models import RecurringBlock

    start_t = parse_time(start)
    end_t = parse_time(end)
    night = is_overnight(start_t, end_t, night_flag)
    new_iv = recurring_intervals(day_of_week, start_t, end_t, night)
    if not new_iv:
        return []

    # Un bloc de nuit déborde sur J+1 ; un bloc de nuit la veille déborde sur J.
    candidate_days = {
        day_of_week,
        (day_of_week + 1) % 7,
        (day_of_week - 1) % 7,
    }
    qs = RecurringBlock.objects.filter(
        user=user,
        active=True,
        day_of_week__in=candidate_days,
        flexibility='fixed',
    )
    if exclude_id is not None:
        qs = qs.exclude(id=exclude_id)

    conflicts = []
    for b in qs:
        b_iv = recurring_intervals(
            b.day_of_week, b.start_time, b.end_time, b.is_night_shift
        )
        if intervals_conflict(new_iv, b_iv):
            conflicts.append(b)
    return conflicts
