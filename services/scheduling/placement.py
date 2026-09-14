"""Pure placement engine for flexible recurring blocks.

This module decides where flexible recurring blocks sit on one concrete date,
using fixed commitments as hard walls. It performs ORM reads only and never
writes schedule state.
"""
from __future__ import annotations

from datetime import timedelta

from services.scheduling.exceptions import skipped_block_ids
from services.scheduling.overlap import (
    MINUTES_PER_DAY,
    is_overnight,
    time_to_min,
)

# Report du sommeil au lendemain. Cas fondateur (2026-09-14): quart fixe du
# jeudi 19:00-02:00 et sommeil souple 23:00-07:00. Le debut du sommeil tombe
# sous le quart; l'ancienne relocalisation intra-jour le posait a 11:00-19:00
# le jeudi, ce qui rendait la journee pleine (find_free_slots vide, refus
# « sommeil protege 11:00-24:00 »). La personne dort en realite apres le quart:
# vendredi 02:00-10:00. On reporte donc le sommeil au lendemain quand la fin du
# mur (trajet compris) plus la duree du sommeil tient avant midi. Au-dela (ex:
# quart 19:00-07:00, reveil a 15:00), on garde la relocalisation dans la
# journee du quart, comportement historique.
SEUIL_REPORT_LENDEMAIN = 12 * 60


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _clip_interval(
    start: int,
    end: int,
    day_start: int = 0,
    day_end: int = MINUTES_PER_DAY,
) -> tuple[int, int] | None:
    start = max(start, day_start)
    end = min(end, day_end)
    if end <= start:
        return None
    return start, end


def _fmt(minutes: int | None) -> str | None:
    if minutes is None:
        return None
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _overlaps(interval: tuple[int, int], intervals: list[tuple[int, int]]) -> bool:
    start, end = interval
    for other_start, other_end in intervals:
        if start < other_end and other_start < end:
            return True
    return False


def _result(
    block,
    *,
    start_min: int | None,
    end_min: int | None,
    preferred: bool,
    shrunk: bool,
    skipped: bool,
    overnight_kept: bool,
    start_time: str | None = None,
    end_time: str | None = None,
    reporte_au_lendemain: dict | None = None,
) -> dict:
    return {
        "block_id": block.id,
        "title": block.title,
        "block_type": block.block_type,
        "start_min": start_min,
        "end_min": end_min,
        "start_time": start_time if start_time is not None else _fmt(start_min),
        "end_time": end_time if end_time is not None else _fmt(end_min),
        "preferred": preferred,
        "shrunk": shrunk,
        "skipped": skipped,
        "overnight_kept": overnight_kept,
        "reporte_au_lendemain": reporte_au_lendemain,
    }


def _report_sommeil(
    murs: list[tuple[int, int]],
    sommeil_start: int,
    sommeil_end: int,
    duree: int,
) -> dict | None:
    """Fonction pure: le sommeil souple overnight doit-il passer au lendemain ?

    ``murs``: murs fixes overnight du jour, en (debut_min, fin_min_du_lendemain),
    trajet compris (debut moins l'aller, fin plus le retour). ``sommeil_start``
    et ``sommeil_end`` sont les heures stockees du sommeil (minutes depuis
    minuit, fin le lendemain). ``duree`` est la duree effective du sommeil.

    Retourne ``{"start_min": E, "end_min": E + duree}`` (minutes du lendemain)
    quand le debut du sommeil tombe dans la partie du soir d'un mur overnight
    qui finit a E le lendemain et que E + duree <= SEUIL_REPORT_LENDEMAIN.
    Exemple: quart 19:00-02:00, sommeil 23:00-07:00 -> 02:00-10:00. Sinon None
    (ex: quart 19:00-07:00, 07:00 + 8 h depasse midi).
    """
    if duree is None or duree <= 0:
        duree = (sommeil_end - sommeil_start) % MINUTES_PER_DAY
    if duree <= 0:
        return None
    fins = [
        fin
        for debut, fin in murs
        if debut <= sommeil_start < MINUTES_PER_DAY
    ]
    if not fins:
        return None
    fin_mur = max(0, max(fins))
    if fin_mur + duree > SEUIL_REPORT_LENDEMAIN:
        return None
    return {"start_min": fin_mur, "end_min": fin_mur + duree}


def _murs_de_nuit(blocs, profile, date) -> list[tuple[int, int]]:
    """Murs fixes overnight de ``date`` pour :func:`_report_sommeil`.

    ``blocs``: blocs recurrents actifs du jour de semaine de ``date``, deja
    purges des occurrences sautees. Les bornes start_date/end_date sont
    appliquees ici.
    """
    from services.commute import block_commute_minutes

    murs: list[tuple[int, int]] = []
    for bloc in blocs:
        if bloc.is_flexible or not bloc.active_on(date):
            continue
        if not is_overnight(bloc.start_time, bloc.end_time, bloc.is_night_shift):
            continue
        before, after = (
            block_commute_minutes(bloc, profile) if profile is not None else (0, 0)
        )
        murs.append(
            (time_to_min(bloc.start_time) - before, time_to_min(bloc.end_time) + after)
        )
    return murs


def _reports_du_jour(blocs, profile, date) -> dict[int, dict]:
    """Sommeils souples de ``date`` reportes au lendemain, par id de bloc.

    Source unique de la decision, partagee par place_day, solve_placement,
    solve_day et intervalles_sommeil_reporte, pour que le jour du quart et le
    lendemain voient toujours le meme report.
    """
    blocs = list(blocs)
    murs = _murs_de_nuit(blocs, profile, date)
    if not murs:
        return {}
    reports: dict[int, dict] = {}
    for bloc in blocs:
        if not bloc.is_flexible or bloc.block_type != "sleep":
            continue
        if not bloc.active_on(date):
            continue
        if not is_overnight(bloc.start_time, bloc.end_time, bloc.is_night_shift):
            continue
        report = _report_sommeil(
            murs,
            time_to_min(bloc.start_time),
            time_to_min(bloc.end_time),
            bloc.effective_duration_minutes(),
        )
        if report is not None:
            reports[bloc.id] = report
    return reports


def _blocs_du_jour(user, date) -> list:
    """Blocs recurrents actifs du jour de semaine de ``date``, occurrences
    sautees exclues, lieu precharge pour le calcul du trajet."""
    from core.models import RecurringBlock

    return list(
        RecurringBlock.objects.filter(
            user=user,
            active=True,
            day_of_week=date.weekday(),
        )
        .exclude(id__in=skipped_block_ids(user, date))
        .select_related("place")
        .order_by("id")
    )


def intervalles_sommeil_reporte(user, date) -> list[tuple[int, int]]:
    """Intervalles de ``date`` occupes par le sommeil reporte de la veille.

    NON RECURSIF: lit seulement les blocs de ``date - 1 jour`` (murs fixes
    overnight et sommeils souples, occurrences sautees et bornes respectees) et
    applique :func:`_report_sommeil`. N'appelle jamais place_day,
    occupied_intervals ni solve_placement. Intervalles bornes a [0, 1440],
    fusionnes et tries.
    """
    veille = date - timedelta(days=1)
    blocs = _blocs_du_jour(user, veille)
    if not blocs:
        return []
    profile = getattr(user, "profile", None)
    intervalles = []
    for report in _reports_du_jour(blocs, profile, veille).values():
        borne = _clip_interval(report["start_min"], report["end_min"])
        if borne is not None:
            intervalles.append(borne)
    return _merge_intervals(intervalles)


def fixed_busy_intervals(
    user,
    date,
    *,
    exclude_scheduled_id=None,
) -> list[tuple[int, int]]:
    """Return fixed hard-wall intervals for ``user`` on ``date``.

    Intervals are clipped to ``[0, 1440]``, merged, and sorted. Fixed recurring
    blocks count on their own start date, previous-day fixed overnight blocks
    spill into the current morning, and all scheduled blocks on the date count
    as walls. Flexible recurring blocks are intentionally excluded.
    """
    from core.models import RecurringBlock, ScheduledBlock
    from services.commute import block_commute_minutes, task_commute_minutes

    dow = date.weekday()
    prev_dow = (dow - 1) % 7
    prev_date = date - timedelta(days=1)
    skipped_today = skipped_block_ids(user, date)
    skipped_prev = skipped_block_ids(user, prev_date)
    profile = getattr(user, "profile", None)

    raw: list[tuple[int, int]] = []

    today_blocks = RecurringBlock.objects.filter(
        user=user,
        active=True,
        day_of_week=dow,
    ).exclude(id__in=skipped_today).select_related("place")
    for block in today_blocks:
        if block.is_flexible:
            continue
        start = time_to_min(block.start_time)
        end = time_to_min(block.end_time)
        before, after = (
            block_commute_minutes(block, profile) if profile is not None else (0, 0)
        )
        if is_overnight(block.start_time, block.end_time, block.is_night_shift):
            raw.append((start - before, MINUTES_PER_DAY))
        else:
            raw.append((start - before, end + after))

    previous_blocks = RecurringBlock.objects.filter(
        user=user,
        active=True,
        day_of_week=prev_dow,
    ).exclude(id__in=skipped_prev).select_related("place")
    for block in previous_blocks:
        if block.is_flexible:
            continue
        if is_overnight(block.start_time, block.end_time, block.is_night_shift):
            _, after = (
                block_commute_minutes(block, profile) if profile is not None else (0, 0)
            )
            raw.append((0, time_to_min(block.end_time) + after))

    from django.db.models import Q

    # Une tache TERMINEE ne bloque plus son creneau: cocher doit liberer la
    # place. Vecu (2026-08-13): « une tache d'une minute dans 6 minutes »
    # refusee parce qu'elle chevauchait une tache App Store cochee la veille,
    # dont le placement 18:40-20:40 comptait toujours comme un mur. On exclut
    # le placement coche (actually_completed) ET tout placement d'une tache
    # completee par un autre chemin (les deux drapeaux sont synchronises par
    # mark_completed, mais la ceinture ne coute qu'un OR).
    scheduled_qs = ScheduledBlock.objects.filter(user=user, date=date).exclude(
        Q(actually_completed=True) | Q(task__completed=True)
    )
    if exclude_scheduled_id is not None:
        scheduled_qs = scheduled_qs.exclude(id=exclude_scheduled_id)
    for block in scheduled_qs.select_related("task", "task__place"):
        start = time_to_min(block.start_time)
        end = time_to_min(block.end_time)
        before, after = (
            task_commute_minutes(block.task, profile) if profile is not None else (0, 0)
        )
        if end <= start:
            raw.append((start - before, MINUTES_PER_DAY))
        else:
            raw.append((start - before, end + after))

    clipped = [
        clipped_interval
        for start, end in raw
        if (clipped_interval := _clip_interval(start, end)) is not None
    ]
    return _merge_intervals(clipped)


def free_gaps(
    busy: list[tuple[int, int]],
    day_start: int,
    day_end: int,
) -> list[tuple[int, int]]:
    """Return the complement of ``busy`` within ``[day_start, day_end]``."""
    clipped_busy = [
        clipped_interval
        for start, end in busy
        if (clipped_interval := _clip_interval(start, end, day_start, day_end))
        is not None
    ]
    gaps: list[tuple[int, int]] = []
    cursor = day_start
    for start, end in _merge_intervals(clipped_busy):
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < day_end:
        gaps.append((cursor, day_end))
    return gaps


def occupied_intervals(
    user,
    date,
    day_start: int = 0,
    day_end: int = MINUTES_PER_DAY,
    *,
    exclude_scheduled_id=None,
) -> list[tuple[int, int]]:
    """Return fixed walls plus placed flexible blocks within ``[day_start, day_end]``.

    Le sommeil de la veille reporte apres un quart de nuit
    (:func:`intervalles_sommeil_reporte`) compte aussi comme occupe.
    """
    raw = list(fixed_busy_intervals(user, date, exclude_scheduled_id=exclude_scheduled_id))
    reporte = intervalles_sommeil_reporte(user, date)
    raw.extend(reporte)

    for placement in _placer_jour(user, date, day_start, day_end, reporte=reporte):
        if placement["skipped"]:
            continue

        start = placement["start_min"]
        end = placement["end_min"]
        if start is None or end is None:
            continue
        if end <= start:
            end = day_end

        clipped = _clip_interval(start, end, day_start, day_end)
        if clipped is not None:
            raw.append(clipped)

    clipped = [
        clipped_interval
        for start, end in raw
        if (clipped_interval := _clip_interval(start, end, day_start, day_end))
        is not None
    ]
    return _merge_intervals(clipped)


def open_intervals(
    user,
    date,
    day_start: int,
    day_end: int,
    *,
    exclude_scheduled_id=None,
) -> list[tuple[int, int]]:
    """Return the complement of placed occupancy within ``[day_start, day_end]``."""
    return free_gaps(
        occupied_intervals(
            user,
            date,
            day_start,
            day_end,
            exclude_scheduled_id=exclude_scheduled_id,
        ),
        day_start,
        day_end,
    )


def _sort_key(result: dict) -> tuple[int, int, int]:
    if result["skipped"]:
        return 1, MINUTES_PER_DAY + 1, result["block_id"]
    return 0, result["start_min"], result["block_id"]


def place_day(user, date, day_start=0, day_end=MINUTES_PER_DAY) -> list[dict]:
    """Place flexible recurring blocks for one user/date.

    Algorithm:
    1. Compute fixed hard walls with :func:`fixed_busy_intervals`.
    2. Collect active, non-skipped flexible recurring blocks for the date's
       weekday.
    3. Keep v1 scope deliberately intra-day: flexible blocks whose stored
       preferred slot crosses midnight are not relocated. They are emitted at
       their stored time with ``preferred=True`` and ``overnight_kept=True``.
       This means sleep stored as 23:00-07:00 stays overnight for now, while the
       common 00:00-07:00 sleep shape is movable because it fits inside the day.
    4. Place the remaining intra-day flexible blocks by duration descending,
       then preferred start ascending, so large blocks reserve space first.
    5. For each block, try its preferred interval. If that is unavailable, use
       the fitting free gap whose start is nearest to the preferred start, with
       an earliest-gap tiebreak, and place the interval inside that gap as close
       to the preferred start as possible. If no gap fully fits, fill the
       largest gap and mark the placement as shrunk. If no gap exists, mark the
       block skipped.

    Report au lendemain: un sommeil souple overnight dont le debut tombe sous
    un mur fixe overnight qui finit assez tot le lendemain (voir
    :func:`_report_sommeil` et SEUIL_REPORT_LENDEMAIN) n'est PAS relocalise
    dans la journee du quart. Il sort ``skipped=True`` sur sa date, sans
    heures, avec ``reporte_au_lendemain={"start_time", "end_time"}``; son
    morceau du matin (00:00 -> fin) reste occupe. Le lendemain, place_day et
    occupied_intervals comptent l'intervalle reporte comme occupe
    (:func:`intervalles_sommeil_reporte`).

    The function is deterministic and pure apart from ORM reads. It does not
    create, update, or delete schedule rows.
    """
    return _placer_jour(user, date, day_start, day_end)


def _placer_jour(user, date, day_start=0, day_end=MINUTES_PER_DAY, *, reporte=None) -> list[dict]:
    """Corps de :func:`place_day`. ``reporte`` evite de relire la veille quand
    l'appelant (occupied_intervals) l'a deja calculee."""
    fixed = fixed_busy_intervals(user, date)
    if reporte is None:
        reporte = intervalles_sommeil_reporte(user, date)
    taken = list(fixed) + list(reporte)

    blocs = _blocs_du_jour(user, date)
    flexible_blocks = [block for block in blocs if block.is_flexible]
    reports = (
        _reports_du_jour(blocs, getattr(user, "profile", None), date)
        if flexible_blocks
        else {}
    )

    overnight_results: list[dict] = []
    intraday_blocks = []
    for block in flexible_blocks:
        start = time_to_min(block.start_time)
        end = time_to_min(block.end_time)
        duration = block.effective_duration_minutes()
        if is_overnight(block.start_time, block.end_time, block.is_night_shift):
            duration = block.effective_duration_minutes()
            report = reports.get(block.id)
            if report is not None:
                # Sommeil apres le quart: il n'a pas lieu ce soir, il occupe le
                # lendemain matin (compte par intervalles_sommeil_reporte du
                # lendemain). Aucune relocalisation dans la journee du quart.
                overnight_results.append(
                    _result(
                        block,
                        start_min=None,
                        end_min=None,
                        preferred=False,
                        shrunk=False,
                        skipped=True,
                        overnight_kept=False,
                        reporte_au_lendemain={
                            "start_time": _fmt(report["start_min"]),
                            "end_time": _fmt(report["end_min"]),
                        },
                    )
                )
                morning_piece = _clip_interval(0, end, day_start, day_end)
                if morning_piece is not None:
                    taken.append(morning_piece)
                continue
            # Si l'heure de début stockée tombe DANS un mur fixe (ex: sommeil
            # 23:00-07:00 alors qu'un travail de nuit fixe couvre 19:00-07:00),
            # le bloc ne peut pas commencer là. On le relocalise dans la journée
            # comme un bloc intra-jour, au lieu de l'afficher par-dessus le mur
            # (c'était la limite v1: le sommeil restait collé à 23:00-07:00 sous
            # le quart). Si le début est libre, on garde le créneau overnight tel
            # quel (cas courant, y compris un léger rognage matinal déjà toléré).
            start_walled = _overlaps((start, start + 1), fixed)
            if start_walled and 0 < duration <= MINUTES_PER_DAY:
                intraday_blocks.append((block, duration, start))
                continue
            overnight_results.append(
                _result(
                    block,
                    start_min=start,
                    end_min=end,
                    start_time=block.start_time.strftime("%H:%M"),
                    end_time=block.end_time.strftime("%H:%M"),
                    preferred=True,
                    shrunk=False,
                    skipped=False,
                    overnight_kept=True,
                )
            )
            # Un souple overnight (sommeil 23:00-07:00) occupe CE jour aux DEUX
            # bouts: le soir (start->minuit) ET le matin (minuit->end). Sans le
            # morceau du matin, le glouton plaçait un autre souple en pleine nuit
            # (ex sport à 06:30, dans le sommeil).
            evening_piece = _clip_interval(start, MINUTES_PER_DAY, day_start, day_end)
            if evening_piece is not None:
                taken.append(evening_piece)
            morning_piece = _clip_interval(0, end, day_start, day_end)
            if morning_piece is not None:
                taken.append(morning_piece)
            continue

        if duration <= 0 or duration > MINUTES_PER_DAY or start + duration > MINUTES_PER_DAY:
            overnight_results.append(
                _result(
                    block,
                    start_min=None,
                    end_min=None,
                    preferred=False,
                    shrunk=False,
                    skipped=True,
                    overnight_kept=False,
                )
            )
            continue

        intraday_blocks.append((block, duration, start))

    intraday_blocks.sort(key=lambda item: (-item[1], item[2], item[0].id))

    placed_results: list[dict] = []
    for block, duration, preferred_start in intraday_blocks:
        preferred_interval = (preferred_start, preferred_start + duration)
        if (
            preferred_interval[0] >= day_start
            and preferred_interval[1] <= day_end
            and not _overlaps(preferred_interval, taken)
        ):
            start, end = preferred_interval
            placed_results.append(
                _result(
                    block,
                    start_min=start,
                    end_min=end,
                    preferred=True,
                    shrunk=False,
                    skipped=False,
                    overnight_kept=False,
                )
            )
            taken.append((start, end))
            continue

        gaps = free_gaps(_merge_intervals(taken), day_start, day_end)
        fitting_gaps = [
            (start, end)
            for start, end in gaps
            if end - start >= duration
        ]
        if fitting_gaps:
            gap_start, gap_end = min(
                fitting_gaps,
                key=lambda gap: (abs(gap[0] - preferred_start), gap[0]),
            )
            start = max(
                gap_start,
                min(preferred_start, gap_end - duration),
            )
            end = start + duration
            placed_results.append(
                _result(
                    block,
                    start_min=start,
                    end_min=end,
                    preferred=False,
                    shrunk=False,
                    skipped=False,
                    overnight_kept=False,
                )
            )
            taken.append((start, end))
            continue

        if gaps and block.block_type != "sleep":
            start, end = max(gaps, key=lambda gap: (gap[1] - gap[0], -gap[0]))
            placed_results.append(
                _result(
                    block,
                    start_min=start,
                    end_min=end,
                    preferred=False,
                    shrunk=True,
                    skipped=False,
                    overnight_kept=False,
                )
            )
            taken.append((start, end))
            continue

        placed_results.append(
            _result(
                block,
                start_min=None,
                end_min=None,
                preferred=False,
                shrunk=False,
                skipped=True,
                overnight_kept=False,
            )
        )

    return sorted(overnight_results + placed_results, key=_sort_key)
