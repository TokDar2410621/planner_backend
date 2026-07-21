"""
Regression tests for group A8 (agent tools hardening).

Covers:
- D5: enum/choice + max_length enforced at the tool layer before create()/save();
  multi-row create wrapped in transaction.atomic() (no partial rows on failure).
- S10: destructive tools require confirmation / are soft + reversible.

Tools are called directly with a test user (no LLM in the loop).
"""
from datetime import time
from unittest.mock import patch

import pytest

from core.models import RecurringBlock, Task
from services.agent.tools.blocks import CreateBlockTool, ClearAllBlocksTool
from services.agent.tools.tasks import CreateTaskTool, DeleteTaskTool


# --- D5: enum validation at the tool layer -------------------------------

@pytest.mark.django_db
def test_create_block_rejects_out_of_enum_block_type(user):
    """An out-of-enum block_type is rejected and NOT written to the DB."""
    tool = CreateBlockTool()
    result = tool.execute(
        user,
        title="Bloc bidon",
        block_type="not_a_real_type",  # not in BLOCK_TYPE_CHOICES
        days=[0],
        start_time="09:00",
        end_time="10:00",
    )

    assert result.success is False
    assert "block_type" in result.message
    # Nothing persisted.
    assert RecurringBlock.objects.filter(user=user).count() == 0


@pytest.mark.django_db
def test_create_task_rejects_out_of_enum_task_type(user):
    """An out-of-enum task_type is rejected and NOT written to the DB."""
    tool = CreateTaskTool()
    result = tool.execute(user, title="T", task_type="wizardry")

    assert result.success is False
    assert "task_type" in result.message
    assert Task.objects.filter(user=user).count() == 0


@pytest.mark.django_db
def test_create_block_rejects_overlong_title(user):
    """A title longer than max_length is rejected before create()."""
    tool = CreateBlockTool()
    result = tool.execute(
        user,
        title="x" * 5000,
        block_type="work",
        days=[0],
        start_time="09:00",
        end_time="10:00",
    )

    assert result.success is False
    assert "title" in result.message
    assert RecurringBlock.objects.filter(user=user).count() == 0


@pytest.mark.django_db
def test_create_block_valid_input_still_works(user):
    """A valid create is unaffected by the new validation."""
    tool = CreateBlockTool()
    result = tool.execute(
        user,
        title="Travail",
        block_type="work",
        days=[0, 1],
        start_time="09:00",
        end_time="10:00",
    )

    assert result.success is True
    assert RecurringBlock.objects.filter(user=user, active=True).count() == 2


# --- D5: multi-row create is atomic --------------------------------------

@pytest.mark.django_db
def test_create_block_multiday_failure_midway_leaves_no_partial_rows(user):
    """If a create fails midway through a multi-day create, nothing persists."""
    tool = CreateBlockTool()

    real_create = RecurringBlock.objects.create
    calls = {"n": 0}

    def flaky_create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom on the second day")
        return real_create(**kwargs)

    with patch.object(RecurringBlock.objects, "create", side_effect=flaky_create):
        result = tool.execute(
            user,
            title="Travail",
            block_type="work",
            days=[0, 1],  # day 0 succeeds, day 1 raises
            start_time="09:00",
            end_time="10:00",
        )

    assert result.success is False
    assert calls["n"] == 2  # it did attempt the second create
    # The first (successful) create must have been rolled back.
    assert RecurringBlock.objects.filter(user=user).count() == 0


# --- S10: destructive tools gated / reversible ---------------------------

@pytest.mark.django_db
def test_delete_task_requires_confirmation(user):
    """DeleteTaskTool refuses to delete without explicit confirm=true."""
    task = Task.objects.create(user=user, title="Ne me supprime pas")
    tool = DeleteTaskTool()

    # Without confirm: refused, task still present.
    result = tool.execute(user, task_id=task.id)
    assert result.success is False
    assert Task.objects.filter(id=task.id).exists()

    # Explicit confirm=False also refused.
    result = tool.execute(user, task_id=task.id, confirm=False)
    assert result.success is False
    assert Task.objects.filter(id=task.id).exists()


@pytest.mark.django_db
def test_delete_task_declares_requires_confirmation(user):
    """DeleteTaskTool advertises it needs out-of-band confirmation + schema requires confirm."""
    tool = DeleteTaskTool()
    assert getattr(tool, "requires_confirmation", False) is True
    assert "confirm" in tool.parameters["required"]


@pytest.mark.django_db
def test_delete_task_with_confirmation_deletes(user):
    """With confirm=true the task is deleted (scoped to the owner)."""
    task = Task.objects.create(user=user, title="Adieu")
    tool = DeleteTaskTool()

    result = tool.execute(user, task_id=task.id, confirm=True)
    assert result.success is True
    assert not Task.objects.filter(id=task.id).exists()


@pytest.mark.django_db
def test_delete_task_is_user_scoped(user, django_user_model):
    """A user cannot delete another user's task even with confirm=true."""
    other = django_user_model.objects.create_user(username="other", password="x")
    other_task = Task.objects.create(user=other, title="Tâche de l'autre")
    tool = DeleteTaskTool()

    result = tool.execute(user, task_id=other_task.id, confirm=True)
    assert result.success is False
    assert Task.objects.filter(id=other_task.id).exists()


@pytest.mark.django_db
def test_clear_all_blocks_is_soft_and_reversible(user):
    """ClearAllBlocksTool soft-deletes (active=False), rows are not destroyed."""
    b1 = RecurringBlock.objects.create(
        user=user, title="A", block_type="work", day_of_week=0,
        start_time=time(9, 0), end_time=time(10, 0),
    )
    b2 = RecurringBlock.objects.create(
        user=user, title="B", block_type="work", day_of_week=1,
        start_time=time(9, 0), end_time=time(10, 0),
    )
    tool = ClearAllBlocksTool()

    result = tool.execute(user, confirm=True)
    assert result.success is True
    assert result.data.get("reversible") is True
    # Rows still exist in the DB (soft-delete), just deactivated -> restorable.
    assert RecurringBlock.objects.filter(user=user).count() == 2
    assert RecurringBlock.objects.filter(user=user, active=True).count() == 0
    b1.refresh_from_db()
    b2.refresh_from_db()
    assert b1.active is False and b2.active is False


@pytest.mark.django_db
def test_clear_all_blocks_requires_confirm(user):
    """Without confirm=true, ClearAllBlocksTool does nothing."""
    RecurringBlock.objects.create(
        user=user, title="A", block_type="work", day_of_week=0,
        start_time=time(9, 0), end_time=time(10, 0),
    )
    tool = ClearAllBlocksTool()

    result = tool.execute(user)
    assert result.success is False
    assert RecurringBlock.objects.filter(user=user, active=True).count() == 1
    assert getattr(tool, "requires_confirmation", False) is True
