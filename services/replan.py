"""
Partial replanning (spec §7).

"Je termine 30 minutes plus tard" / "je reprends à 15h30". We do NOT rebuild the
whole day. We:
  1. lock fixed events (RecurringBlock) — never touched here;
  2. keep already-started / completed activities;
  3. free only the flexible ScheduledBlocks that the delay made impossible
     (those starting before the resume time);
  4. re-place their tasks in the remaining gaps AFTER the resume time, same day
     first (earliest_start floors out the elapsed part).

Returns a structured diff + a plain explanation so the UI can show "what changed
and why" (spec §7/§11).
"""
import logging
from datetime import datetime, timedelta

from django.db import transaction
from django.utils import timezone

from core.models import ScheduledBlock
from services.ai_scheduler import AIScheduler

logger = logging.getLogger(__name__)


def _parse_hhmm(value):
    """'HH:MM' -> time, else None."""
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), '%H:%M').time()
    except (ValueError, AttributeError):
        return None


@transaction.atomic
def replan_after_delay(user, resume_time=None, delay_minutes=None):
    """Re-place today's flexible blocks that a delay pushed out of the way.

    Args:
        user: owner
        resume_time: a ``datetime.time`` or 'HH:MM' — when the user is free again
        delay_minutes: alternatively, minutes from now

    Returns a dict: resume_time, moved[], unplaced[], message.
    """
    today = timezone.localdate()

    if isinstance(resume_time, str):
        resume_time = _parse_hhmm(resume_time)
    if resume_time is None:
        base = timezone.localtime()
        if delay_minutes:
            base = base + timedelta(minutes=int(delay_minutes))
        resume_time = base.time()

    # Flexible, not-yet-done blocks today whose slot the delay invalidated.
    displaced = list(
        ScheduledBlock.objects.filter(
            user=user, date=today, actually_completed=False,
            start_time__lt=resume_time,
        ).select_related('task')
    )

    moved_out = []
    tasks_to_replace = []
    for b in displaced:
        moved_out.append({
            'task_id': b.task_id,
            'title': b.task.title,
            'was': b.start_time.strftime('%H:%M'),
        })
        tasks_to_replace.append(b.task)
        b.delete()  # freed; kept blocks (start >= resume) stay put and block their slot

    scheduler = AIScheduler()
    created = scheduler.generate_schedule(
        user,
        tasks=tasks_to_replace,
        start_date=today,
        num_days=1,
        earliest_start={today: resume_time},
    )
    created_by_task = {c.task_id: c for c in created}

    moved = []
    for m in moved_out:
        c = created_by_task.get(m['task_id'])
        if c:
            moved.append({**m, 'now': c.start_time.strftime('%H:%M')})

    return {
        'resume_time': resume_time.strftime('%H:%M'),
        'moved': moved,
        'unplaced': scheduler.last_unplaced,
        'message': _build_message(resume_time, moved, scheduler.last_unplaced),
    }


def _build_message(resume_time, moved, unplaced):
    """Human explanation of the replan (spec §7 example style)."""
    r = resume_time.strftime('%H:%M')
    parts = []
    if moved:
        items = ', '.join(f"{m['title']} ({m['was']} → {m['now']})" for m in moved)
        parts.append(f"Reprise à {r} : j'ai déplacé {items}.")
    if unplaced:
        titles = ', '.join(u['title'] for u in unplaced)
        parts.append(
            f"Je n'ai pas pu recaser {titles} aujourd'hui — il n'y a plus assez "
            f"de temps libre après {r}."
        )
    if not parts:
        parts.append(f"Rien à déplacer : ton planning après {r} tient toujours.")
    return ' '.join(parts)
