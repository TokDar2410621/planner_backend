"""
RecurringBlock maintenance: remove exact-duplicate blocks.

Duplicates accumulate when the same schedule is declared across several chat
sessions or re-uploaded. A duplicate = same (day_of_week, start_time, end_time,
block_type); we keep the oldest (smallest id) and drop the rest. Pure enough to
unit-test, reused by the management command and the /deduplicate/ endpoint.
"""
from collections import defaultdict


def dedupe_recurring_blocks(user) -> dict:
    """Delete exact-duplicate recurring blocks for `user`. Returns a report."""
    from core.models import RecurringBlock

    blocks = list(RecurringBlock.all_objects.filter(user=user))
    groups = defaultdict(list)
    for b in blocks:
        groups[(b.day_of_week, b.start_time, b.end_time, b.block_type)].append(b)

    to_delete = []
    details = []
    for (day, start, end, btype), bs in groups.items():
        if len(bs) <= 1:
            continue
        bs.sort(key=lambda x: x.id)  # keep the oldest
        keep, extra = bs[0], bs[1:]
        details.append({
            'day_of_week': day,
            'start_time': start.strftime('%H:%M'),
            'end_time': end.strftime('%H:%M'),
            'block_type': btype,
            'kept_id': keep.id,
            'kept_title': keep.title,
            'removed': [{'id': x.id, 'title': x.title} for x in extra],
        })
        to_delete.extend(x.id for x in extra)

    if to_delete:
        RecurringBlock.all_objects.filter(id__in=to_delete, user=user).delete()

    return {'removed': len(to_delete), 'duplicate_groups': details}
