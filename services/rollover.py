"""
Structural forgiveness (pardon par le mécanisme, pas par le copywriting).

A ScheduledBlock left in the past and never completed is NOT a 'late' failure to
paint red — it is simply work that still needs a slot. This deletes those stale
past blocks and re-places their tasks into today's (and the week's) free time,
never touching completed work. A skipped block silently reappears instead of
accumulating as visible guilt debt (the #1 driver of planner-app abandonment).
"""
import logging

from django.db import transaction
from django.utils import timezone

from core.models import ScheduledBlock
from services.ai_scheduler import AIScheduler

logger = logging.getLogger(__name__)


@transaction.atomic
def roll_over_missed(user) -> dict:
    """Re-place tasks of past incomplete blocks into today+. Returns a report."""
    today = timezone.localdate()
    now_t = timezone.localtime().time()

    missed = list(
        ScheduledBlock.objects.filter(
            user=user, date__lt=today, actually_completed=False
        ).select_related('task')
    )

    seen = set()
    tasks = []
    meta = []
    for b in missed:
        if b.task_id in seen or b.task.completed:
            continue  # dedupe + never resurrect an already-done task
        seen.add(b.task_id)
        tasks.append(b.task)
        meta.append({'task_id': b.task_id, 'title': b.task.title, 'was': str(b.date)})

    # The stale past blocks never happened — remove them (no red, no debt).
    ScheduledBlock.objects.filter(id__in=[b.id for b in missed]).delete()

    if not tasks:
        return {'rolled': 0, 'items': [], 'unplaced': []}

    scheduler = AIScheduler()
    created = scheduler.generate_schedule(
        user, tasks=tasks, start_date=today, num_days=7,
        earliest_start={today: now_t},
    )
    by_task = {c.task_id: c for c in created}
    items = []
    for m in meta:
        c = by_task.get(m['task_id'])
        items.append({
            **m,
            'now': f"{c.date} {c.start_time.strftime('%H:%M')}" if c else None,
        })
    return {'rolled': len(items), 'items': items, 'unplaced': scheduler.last_unplaced}
