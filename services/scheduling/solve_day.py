"""Solveur exact (CP-SAT / OR-Tools) de faisabilité et de placement d'une journée.

Complément CHIRURGICAL du glouton :func:`place_day`. Le glouton place au plus
proche de l'heure préférée puis rétrécit ou saute par optimum LOCAL — c'est la
source du faux « ça tient » d'OR2 (il tronque silencieusement un bloc de 2h en
1h). Ce solveur répond de façon EXACTE: soit tout rentre à pleine durée, soit il
dit précisément ce qui ne rentre pas. On l'utilise pour la question de
FAISABILITÉ (« est-ce que ces activités tiennent le jour D ? »), sans remplacer
le rendu (place_day reste la source d'affichage).

Lecture ORM seule; aucune écriture. Import d'ortools paresseux (dépendance lourde
isolée à l'appel).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from services.scheduling.overlap import MINUTES_PER_DAY, is_overnight, time_to_min
from services.scheduling.placement import fixed_busy_intervals


@dataclass
class _Candidate:
    key: str
    title: str
    duration: int
    preferred: Optional[int]  # minutes depuis minuit; None = aucune préférence


def _merge(intervals):
    """Fusionne des intervalles [s,e) qui se chevauchent/se touchent."""
    ivs = sorted((s, e) for s, e in intervals if e > s)
    out = []
    for s, e in ivs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _clip(intervals, day_start, day_end):
    res = []
    for s, e in intervals:
        s, e = max(s, day_start), min(e, day_end)
        if e > s:
            res.append((s, e))
    return res


def _flexible_candidates(user, date):
    """Blocs souples intra-jour du jour (candidats) + murs induits par les souples
    overnight (protégés, non relocalisés — parité place_day v1).

    Parité avec place_day / _window_conflict: un souple overnight du jour (ex
    sommeil 23:00-07:00) occupe CE jour aux DEUX bouts — le soir (start->minuit)
    ET le matin (minuit->end) — reconstruits depuis le bloc du JOUR MÊME (pas de
    la veille), exactement comme `sleep_pieces = [(ps,1440),(0,pe)]`.
    """
    from core.models import RecurringBlock
    from services.scheduling.exceptions import skipped_block_ids

    cands, extra_walls = [], []
    for b in RecurringBlock.objects.filter(
        user=user, active=True, day_of_week=date.weekday()
    ).exclude(id__in=skipped_block_ids(user, date)):
        if not b.is_flexible:
            continue
        start = time_to_min(b.start_time)
        duration = b.effective_duration_minutes()
        if is_overnight(b.start_time, b.end_time, b.is_night_shift):
            extra_walls.append((start, MINUTES_PER_DAY))       # soir
            extra_walls.append((0, time_to_min(b.end_time)))   # matin
            continue
        if duration <= 0 or start + duration > MINUTES_PER_DAY:
            continue
        cands.append(_Candidate(f"block:{b.id}", b.title, duration, start))

    return cands, extra_walls


def solve_day(user, date, extra=None, day_start=0, day_end=MINUTES_PER_DAY, time_limit=2.0):
    """Place les blocs souples + activités ``extra`` autour des murs fixes, à l'optimum.

    ``extra``: liste de dicts ``{title, duration_minutes, preferred_start?}`` =
    nouvelles activités à tester. ``preferred_start`` en minutes (optionnel).

    Retour: ``{feasible, placements, unplaced, solve_ms, status}`` où
    ``placements = [{key,title,start_min,end_min,kind}]`` (kind: 'flexible'|'extra'),
    ``unplaced = [{key,title,duration}]``, et ``feasible`` = toutes les activités
    ``extra`` tiennent à pleine durée.
    """
    from ortools.sat.python import cp_model

    walls = _merge(_clip(list(fixed_busy_intervals(user, date)), day_start, day_end))
    flex, extra_walls = _flexible_candidates(user, date)
    walls = _merge(walls + _clip(extra_walls, day_start, day_end))

    cands = list(flex)
    for i, a in enumerate(extra or []):
        try:
            dur = int(a.get("duration_minutes") or 0)
        except (TypeError, ValueError):
            dur = 0
        if dur <= 0:
            continue
        pref = a.get("preferred_start")
        cands.append(_Candidate(f"extra:{i}", a.get("title") or "Activité", dur,
                                int(pref) if pref is not None else None))

    model = cp_model.CpModel()
    intervals = [model.NewIntervalVar(ws, we - ws, we, f"wall{j}") for j, (ws, we) in enumerate(walls)]

    span = day_end - day_start
    # Paliers STRICTS: garder un bloc SOUPLE existant prime sur placer un extra,
    # qui prime sur minimiser la déviation. Bornes larges pour qu'aucun palier
    # inférieur (même sommé sur tous les candidats) ne compense le palier au-dessus:
    # un extra ne peut donc JAMAIS évincer un engagement existant (recréerait OR2).
    W_FLEX = 10 ** 12
    W_EXTRA = 10 ** 6
    present, start_var, obj = {}, {}, []
    for c in cands:
        if c.duration > span:
            present[c.key] = None  # ne rentrera jamais dans la fenêtre
            continue
        st = model.NewIntVar(day_start, day_end - c.duration, f"s_{c.key}")
        pr = model.NewBoolVar(f"p_{c.key}")
        intervals.append(model.NewOptionalIntervalVar(st, c.duration, st + c.duration, pr, f"iv_{c.key}"))
        present[c.key], start_var[c.key] = pr, st
        obj.append(pr * (W_EXTRA if c.key.startswith("extra:") else W_FLEX))
        if c.preferred is not None:
            pref = max(day_start, min(int(c.preferred), day_end))  # borné => AddAbsEquality faisable
            dev = model.NewIntVar(0, span, f"dev_{c.key}")
            model.AddAbsEquality(dev, st - pref)
            obj.append(-dev)

    model.AddNoOverlap(intervals)
    model.Maximize(sum(obj))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)

    # Garde: ne lire les variables que si le solveur a trouvé une solution.
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "feasible": False,
            "placements": [],
            "unplaced": [{"key": c.key, "title": c.title, "duration": c.duration} for c in cands],
            "solve_ms": round(solver.WallTime() * 1000),
            "status": solver.StatusName(status),
        }

    placements, unplaced = [], []
    for c in cands:
        pr = present.get(c.key)
        if pr is not None and solver.Value(pr):
            s = solver.Value(start_var[c.key])
            placements.append({
                "key": c.key, "title": c.title,
                "start_min": s, "end_min": s + c.duration,
                "kind": "extra" if c.key.startswith("extra:") else "flexible",
            })
        else:
            unplaced.append({"key": c.key, "title": c.title, "duration": c.duration})
    placements.sort(key=lambda p: p["start_min"])
    extra_unplaced = [u for u in unplaced if u["key"].startswith("extra:")]
    return {
        "feasible": not extra_unplaced,
        "placements": placements,
        "unplaced": unplaced,
        "solve_ms": round(solver.WallTime() * 1000),
        "status": solver.StatusName(status),
    }


def _placement_dict(block, start_min, end_min, *, preferred, shrunk, skipped,
                    overnight_kept, start_time=None, end_time=None):
    """Même forme que placement._result (contrat consommé par tout le rendu)."""
    def fmt(x):
        return f"{x // 60:02d}:{x % 60:02d}" if x is not None else None
    return {
        "block_id": block.id,
        "title": block.title,
        "block_type": block.block_type,
        "start_min": start_min,
        "end_min": end_min,
        "start_time": start_time if start_time is not None else fmt(start_min),
        "end_time": end_time if end_time is not None else fmt(end_min),
        "preferred": preferred,
        "shrunk": shrunk,
        "skipped": skipped,
        "overnight_kept": overnight_kept,
    }


def solve_placement(user, date, day_start=0, day_end=MINUTES_PER_DAY, time_limit=2.0):
    """Placement OPTIMAL des blocs souples d'un jour (candidat au remplacement du
    glouton place_day). Retour = MÊME forme que place_day (_result dicts).

    Le solveur place TOUS les blocs souples plaçables à pleine durée autour des
    murs fixes, au plus proche de leur heure préférée. Il ne RÉTRÉCIT jamais (un
    bloc rentre en entier ou est sauté — pas de troncature silencieuse); il évite
    les sauts que le glouton fait par optimum local. Overnight = gardé en place.
    """
    from ortools.sat.python import cp_model
    from core.models import RecurringBlock
    from services.scheduling.exceptions import skipped_block_ids

    fixed = _merge(_clip(list(fixed_busy_intervals(user, date)), day_start, day_end))
    skipped = skipped_block_ids(user, date)

    results, extra_walls, candidates = [], [], []
    for b in RecurringBlock.objects.filter(
        user=user, active=True, day_of_week=date.weekday()
    ).exclude(id__in=skipped):
        if not b.is_flexible:
            continue
        start = time_to_min(b.start_time)
        duration = b.effective_duration_minutes()
        if is_overnight(b.start_time, b.end_time, b.is_night_shift):
            results.append(_placement_dict(
                b, start, time_to_min(b.end_time), preferred=True, shrunk=False,
                skipped=False, overnight_kept=True,
                start_time=b.start_time.strftime("%H:%M"), end_time=b.end_time.strftime("%H:%M")))
            extra_walls.append((start, MINUTES_PER_DAY))
            extra_walls.append((0, time_to_min(b.end_time)))
            continue
        if duration <= 0 or start + duration > MINUTES_PER_DAY:
            results.append(_placement_dict(
                b, None, None, preferred=False, shrunk=False, skipped=True, overnight_kept=False))
            continue
        candidates.append((b, duration, start))

    walls = _merge(fixed + _clip(extra_walls, day_start, day_end))
    span = day_end - day_start

    model = cp_model.CpModel()
    intervals = [model.NewIntervalVar(ws, we - ws, we, f"w{j}") for j, (ws, we) in enumerate(walls)]
    present, start_var, obj = {}, {}, []
    W_PLACE = 10 ** 9  # placer prime toujours sur la déviation
    for b, dur, pref in candidates:
        if dur > span:
            present[b.id] = None
            continue
        st = model.NewIntVar(day_start, day_end - dur, f"s{b.id}")
        pr = model.NewBoolVar(f"p{b.id}")
        intervals.append(model.NewOptionalIntervalVar(st, dur, st + dur, pr, f"iv{b.id}"))
        present[b.id], start_var[b.id] = pr, st
        obj.append(pr * W_PLACE)
        p = max(day_start, min(pref, day_end))
        dev = model.NewIntVar(0, span, f"d{b.id}")
        model.AddAbsEquality(dev, st - p)
        obj.append(-dev)

    model.AddNoOverlap(intervals)
    model.Maximize(sum(obj))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)
    ok = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    for b, dur, pref in candidates:
        prv = present.get(b.id)
        if ok and prv is not None and solver.Value(prv):
            s = solver.Value(start_var[b.id])
            results.append(_placement_dict(
                b, s, s + dur, preferred=(s == pref), shrunk=False, skipped=False, overnight_kept=False))
        else:
            results.append(_placement_dict(
                b, None, None, preferred=False, shrunk=False, skipped=True, overnight_kept=False))

    results.sort(key=lambda r: (
        1 if r["skipped"] else 0,
        r["start_min"] if r["start_min"] is not None else MINUTES_PER_DAY + 1,
        r["block_id"],
    ))
    return results
