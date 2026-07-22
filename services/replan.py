"""
Partial replanning (spec §7) with automation modes (spec §8).

Replan flow, mode-aware:
  1. find the flexible ScheduledBlocks a delay invalidated (incomplete, starting
     before the resume time), one entry PER TASK (deduped);
  2. snapshot those TASKS' blocks today (BEFORE) — scoped, never the whole day;
  3. apply the adjustment (delete displaced, re-place the tasks after resume);
  4. snapshot the same tasks (AFTER) and measure how "important" the change is;
  5. decide by automation_mode: automatic / semi_auto+small -> keep (undoable);
     suggestion / semi_auto+important -> revert, return a proposal to confirm.

Correctness guarantees (from adversarial review):
  - snapshots + restore are SCOPED to the change's own tasks, so blocks the user
    added/moved for OTHER tasks between propose and apply/undo are never wiped;
  - restore is COMPLETION-AWARE: it never disturbs a task that is completed now
    (Task.completed owns the truth), and re-completes via ScheduledBlock.
    mark_completed so Task.completed / TaskHistory stay reconciled;
  - fixed events (RecurringBlock: cours/travail) are out of scope entirely.
"""
import logging
from datetime import datetime, time, timedelta

from django.db import transaction
from django.utils import timezone

from core.models import ScheduledBlock, SchedulePlanChange, Task
from services.ai_scheduler import AIScheduler

logger = logging.getLogger(__name__)


def _parse_hhmm(value):
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), '%H:%M').time()
    except (ValueError, AttributeError):
        return None


def _parse_time(value):
    """'HH:MM' or 'HH:MM:SS' -> time."""
    if isinstance(value, time):
        return value
    for fmt in ('%H:%M:%S', '%H:%M'):
        try:
            return datetime.strptime(value, fmt).time()
        except (ValueError, TypeError):
            continue
    return None


def _snapshot(user, day, task_ids):
    """JSON snapshot of `task_ids` ScheduledBlocks on `day` (scoped, not whole day)."""
    if not task_ids:
        return []
    rows = []
    for sb in ScheduledBlock.objects.filter(
        user=user, date=day, task_id__in=list(task_ids)
    ).select_related('task'):
        rows.append({
            'task_id': sb.task_id,
            'start_time': sb.start_time.strftime('%H:%M:%S'),
            'end_time': sb.end_time.strftime('%H:%M:%S'),
            'actually_completed': sb.actually_completed,
            'actual_duration_minutes': sb.actual_duration_minutes,
        })
    return rows


def _scope_of(*snapshots):
    """Union of task_ids across snapshots (the tasks a change owns)."""
    ids = set()
    for snap in snapshots:
        for b in snap or []:
            if b.get('task_id') is not None:
                ids.add(b['task_id'])
    return ids


def _restore(user, day, blocks_json, scope_task_ids):
    """Replace ONLY the scope tasks' blocks on `day` with the snapshot.

    Completion-aware: tasks completed now are left untouched (their blocks own
    the completion truth); recreated blocks that were completed are reconciled
    through ScheduledBlock.mark_completed so Task/TaskHistory stay in sync.
    """
    scope = {t for t in scope_task_ids if t is not None}
    if not scope:
        return 0
    completed_now = set(
        Task.objects.filter(id__in=scope, user=user, completed=True)
        .values_list('id', flat=True)
    )
    touch = scope - completed_now
    if not touch:
        return 0

    ScheduledBlock.objects.filter(user=user, date=day, task_id__in=list(touch)).delete()
    restored = 0
    for b in blocks_json or []:
        tid = b.get('task_id')
        if tid not in touch:
            continue
        task = Task.objects.filter(id=tid, user=user).first()
        if task is None:
            continue  # task deleted meanwhile: do not resurrect
        start = _parse_time(b.get('start_time'))
        end = _parse_time(b.get('end_time'))
        if start is None or end is None:
            continue
        sb = ScheduledBlock.objects.create(
            user=user, task=task, date=day, start_time=start, end_time=end,
        )
        restored += 1
        if b.get('actually_completed'):
            # Reconcile via the single source of truth, not a raw field write.
            sb.mark_completed(actual_minutes=b.get('actual_duration_minutes'))
    return restored


def _minutes(hhmm):
    t = _parse_time(hhmm)
    return t.hour * 60 + t.minute if t else 0


def _is_important(moved, unplaced, threshold_minutes):
    """A change is important if something can't fit, or a block shifts a lot."""
    if unplaced:
        return True
    for m in moved:
        if abs(_minutes(m['now']) - _minutes(m['was'])) > threshold_minutes:
            return True
    return False


@transaction.atomic
def replan_after_delay(user, resume_time=None, delay_minutes=None, apply=None):
    """Mode-aware partial replan. Returns applied(bool), moved, unplaced, message, token, mode."""
    today = timezone.localdate()

    if isinstance(resume_time, str):
        resume_time = _parse_hhmm(resume_time)
    if resume_time is None:
        base = timezone.localtime()
        if delay_minutes:
            base = base + timedelta(minutes=int(delay_minutes))
        resume_time = base.time()

    profile = user.profile
    mode = profile.automation_mode
    threshold = profile.auto_apply_threshold_minutes or 0

    # Flexible, not-yet-done blocks today that the delay invalidated — deduped
    # to one entry PER TASK (a task is re-placed once, never N times).
    displaced_qs = ScheduledBlock.objects.filter(
        user=user, date=today, actually_completed=False, start_time__lt=resume_time,
    ).select_related('task')

    seen = set()
    tasks_to_replace = []
    moved_out = []
    for b in displaced_qs:
        if b.task_id not in seen:
            seen.add(b.task_id)
            tasks_to_replace.append(b.task)
            moved_out.append({
                'task_id': b.task_id, 'title': b.task.title,
                'was': b.start_time.strftime('%H:%M'),
            })

    # Nothing to move -> no change recorded, no token.
    if not tasks_to_replace:
        return {
            'applied': True, 'mode': mode,
            'resume_time': resume_time.strftime('%H:%M'),
            'moved': [], 'unplaced': [],
            'message': _build_message(resume_time, [], [], applied=True),
            'token': None,
        }

    scope_ids = list(seen)
    before = _snapshot(user, today, scope_ids)  # original positions, pre-mutation

    # Mutate: free the displaced blocks, re-place their tasks after resume.
    for b in displaced_qs:
        b.delete()
    scheduler = AIScheduler()
    created = scheduler.generate_schedule(
        user, tasks=tasks_to_replace, start_date=today, num_days=1,
        earliest_start={today: resume_time},
    )
    created_by_task = {c.task_id: c for c in created}
    moved = []
    for m in moved_out:
        c = created_by_task.get(m['task_id'])
        if c:
            moved.append({**m, 'now': c.start_time.strftime('%H:%M')})
    unplaced = scheduler.last_unplaced

    after = _snapshot(user, today, scope_ids)
    important = _is_important(moved, unplaced, threshold)

    if apply is None:
        apply = mode == 'automatic' or (mode == 'semi_auto' and not important)

    message = _build_message(resume_time, moved, unplaced, applied=apply)

    if apply:
        change = SchedulePlanChange.objects.create(
            user=user, date=today, before=before, after=after,
            moved=moved, unplaced=unplaced, message=message, status='applied',
        )
        return {
            'applied': True, 'mode': mode, 'resume_time': resume_time.strftime('%H:%M'),
            'moved': moved, 'unplaced': unplaced, 'message': message,
            'token': str(change.token),
        }

    # Not applied: revert to BEFORE (scoped) and record a proposal.
    _restore(user, today, before, scope_ids)
    change = SchedulePlanChange.objects.create(
        user=user, date=today, before=before, after=after,
        moved=moved, unplaced=unplaced, message=message, status='proposed',
    )
    return {
        'applied': False, 'mode': mode, 'resume_time': resume_time.strftime('%H:%M'),
        'moved': moved, 'unplaced': unplaced, 'message': message,
        'token': str(change.token),
    }


def _get_change(user, token, status):
    from django.core.exceptions import ValidationError
    try:
        return SchedulePlanChange.objects.filter(
            user=user, token=token, status=status
        ).first()
    except (ValidationError, ValueError):
        return None


@transaction.atomic
def apply_proposal(user, token):
    """Apply a previously-proposed change verbatim (its stored AFTER state)."""
    change = _get_change(user, token, 'proposed')
    if change is None:
        return None
    scope = _scope_of(change.before, change.after)
    _restore(user, change.date, change.after, scope)
    change.status = 'applied'
    change.save(update_fields=['status'])
    return {
        'applied': True, 'token': str(change.token),
        'moved': change.moved, 'unplaced': change.unplaced, 'message': change.message,
    }


@transaction.atomic
def undo_change(user, token):
    """Revert an applied change back to its BEFORE state."""
    change = _get_change(user, token, 'applied')
    if change is None:
        return None
    scope = _scope_of(change.before, change.after)
    restored = _restore(user, change.date, change.before, scope)
    change.status = 'undone'
    change.save(update_fields=['status'])
    return {'undone': True, 'token': str(change.token), 'restored_blocks': restored}


def _build_message(resume_time, moved, unplaced, applied=True):
    r = resume_time.strftime('%H:%M')
    verb = "j'ai déplacé" if applied else "je propose de déplacer"
    parts = []
    if moved:
        items = ', '.join(f"{m['title']} ({m['was']} → {m['now']})" for m in moved)
        parts.append(f"Reprise à {r} : {verb} {items}.")
    if unplaced:
        titles = ', '.join(u['title'] for u in unplaced)
        parts.append(
            f"Je n'ai pas pu recaser {titles} aujourd'hui — il n'y a plus assez "
            f"de temps libre après {r}."
        )
    if not parts:
        parts.append(f"Rien à déplacer : ton planning après {r} tient toujours.")
    return ' '.join(parts)
