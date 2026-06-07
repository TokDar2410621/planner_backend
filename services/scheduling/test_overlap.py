"""Tests du garde-fou anti-chevauchement ("tâches sur tâches")."""
from datetime import time

import pytest

from services.scheduling.overlap import (
    intervals_conflict,
    is_overnight,
    parse_time,
    recurring_intervals,
)


# --- Unitaires purs (sans DB) ------------------------------------------------

def test_parse_time():
    assert parse_time("09:00") == time(9, 0)
    assert parse_time("9:05") == time(9, 5)
    assert parse_time(time(14, 30)) == time(14, 30)
    with pytest.raises(ValueError):
        parse_time("pas une heure")


def test_is_overnight():
    assert is_overnight(time(22, 0), time(6, 0)) is True
    assert is_overnight(time(9, 0), time(17, 0)) is False
    assert is_overnight(time(9, 0), time(9, 0), True) is True


def test_normal_overlap():
    a = recurring_intervals(0, time(9, 0), time(11, 0))
    b = recurring_intervals(0, time(10, 0), time(12, 0))
    assert intervals_conflict(a, b)


def test_adjacent_no_overlap():
    a = recurring_intervals(0, time(9, 0), time(10, 0))
    b = recurring_intervals(0, time(10, 0), time(11, 0))
    assert not intervals_conflict(a, b)


def test_different_days_no_overlap():
    a = recurring_intervals(0, time(9, 0), time(11, 0))
    b = recurring_intervals(1, time(9, 0), time(11, 0))
    assert not intervals_conflict(a, b)


def test_overnight_spills_into_next_day():
    # Lundi 22:00 -> 06:00 chevauche Mardi 05:00 -> 07:00
    a = recurring_intervals(0, time(22, 0), time(6, 0), True)
    b = recurring_intervals(1, time(5, 0), time(7, 0))
    assert intervals_conflict(a, b)


def test_overnight_no_conflict_same_evening():
    # Lundi 22:00 -> 06:00 ne chevauche pas Lundi 18:00 -> 20:00
    a = recurring_intervals(0, time(22, 0), time(6, 0), True)
    b = recurring_intervals(0, time(18, 0), time(20, 0))
    assert not intervals_conflict(a, b)


def test_week_wraps_sunday_to_monday():
    # Dimanche 23:00 -> 01:00 chevauche Lundi 00:30 -> 02:00
    a = recurring_intervals(6, time(23, 0), time(1, 0), True)
    b = recurring_intervals(0, time(0, 30), time(2, 0))
    assert intervals_conflict(a, b)


# --- Outils de l'agent (DB) --------------------------------------------------

def test_create_block_skips_overlap(user):
    from core.models import RecurringBlock
    from services.agent.tools.blocks import CreateBlockTool

    tool = CreateBlockTool()
    r1 = tool.execute(user, title="A", block_type="work", days=[0],
                      start_time="09:00", end_time="11:00")
    assert r1.success
    r2 = tool.execute(user, title="B", block_type="work", days=[0],
                      start_time="10:00", end_time="12:00")
    assert r2.data["skipped"]
    assert not r2.data["created"]
    assert RecurringBlock.objects.filter(
        user=user, day_of_week=0, active=True
    ).count() == 1


def test_create_block_nightshift_overlap_not_bypassed(user):
    """Régression : un bloc de nuit ne doit PLUS contourner le contrôle."""
    from services.agent.tools.blocks import CreateBlockTool

    tool = CreateBlockTool()
    tool.execute(user, title="Soir", block_type="work", days=[0],
                 start_time="20:00", end_time="23:00")
    r = tool.execute(user, title="Nuit", block_type="work", days=[0],
                     start_time="22:00", end_time="06:00", is_night_shift=True)
    assert r.data["skipped"]
    assert not r.data["created"]


def test_update_block_rejects_overlap(user):
    from core.models import RecurringBlock
    from services.agent.tools.blocks import CreateBlockTool, UpdateBlockTool

    create = CreateBlockTool()
    create.execute(user, title="A", block_type="work", days=[0],
                   start_time="09:00", end_time="10:00")
    r = create.execute(user, title="B", block_type="work", days=[0],
                       start_time="11:00", end_time="12:00")
    bid = r.data["created"][0]["id"]

    res = UpdateBlockTool().execute(user, block_id=bid,
                                    start_time="09:30", end_time="10:30")
    assert not res.success
    # Inchangé
    assert RecurringBlock.objects.get(id=bid).start_time == time(11, 0)


# --- Auto-scheduler (DB) -----------------------------------------------------

def test_scheduler_does_not_stack_tasks(user):
    """Deux tâches planifiées en appels séparés ne doivent pas se chevaucher."""
    from core.models import ScheduledBlock, Task
    from services.ai_scheduler import AIScheduler

    t1 = Task.objects.create(user=user, title="T1",
                             estimated_duration_minutes=120, priority=8)
    t2 = Task.objects.create(user=user, title="T2",
                             estimated_duration_minutes=120, priority=8)

    sched = AIScheduler()
    sched.generate_schedule(user, tasks=[t1], num_days=1)
    sched.generate_schedule(user, tasks=[t2], num_days=1)

    blocks = list(ScheduledBlock.objects.filter(user=user).order_by("date", "start_time"))
    assert len(blocks) == 2
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            a, b = blocks[i], blocks[j]
            if a.date == b.date:
                assert a.end_time <= b.start_time or b.end_time <= a.start_time, \
                    "Deux blocs planifiés se chevauchent (tâche sur tâche)"
