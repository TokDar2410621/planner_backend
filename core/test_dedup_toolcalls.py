"""
Regression: the agentic loop must never execute the same tool call twice in one
message (the bug that created a task 3x from one "ajoute une tâche" request).
"""
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase

from core.models import Task
from services.llm.base import FunctionCall, LLMResponse


class _AlwaysDuplicatesProvider:
    """Simulates a provider (e.g. Gemini) that re-emits the SAME tool call on
    every turn because it never 'sees' the result. Without a guard this loops
    and writes duplicates."""

    def __init__(self):
        self.turns = 0

    def is_available(self):
        return True

    def generate_with_history(self, messages, tools, system_prompt):
        self.turns += 1
        args = {"title": "Dedup test task", "estimated_duration_minutes": 60}
        return LLMResponse(
            function_calls=[FunctionCall(name="create_task", args=args, call_id="create_task")],
            raw_content=[{"type": "tool_use", "name": "create_task", "input": args}],
        )


class ToolCallDedupTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('dedup_user', password='pw-dedup-123')

    def test_repeated_tool_call_executes_once(self):
        from services.agent.agent import PlannerAgent

        provider = _AlwaysDuplicatesProvider()
        with patch('services.agent.agent.get_provider', return_value=provider):
            agent = PlannerAgent()
            agent.process_message(self.user, "Ajoute une tache: reviser")

        # The task is created exactly ONCE despite the model repeating the call.
        self.assertEqual(
            Task.objects.filter(user=self.user, title="Dedup test task").count(),
            1,
            msg="Duplicate tool calls must not create duplicate tasks",
        )
        # The loop stops early instead of burning all MAX_TOOL_TURNS turns.
        self.assertLessEqual(provider.turns, PlannerAgent.MAX_TOOL_TURNS)
        self.assertGreaterEqual(provider.turns, 2)


class CreateTaskIdempotencyTest(TestCase):
    """create_task must not duplicate an already-active task with the same title
    (guards against the model re-creating a task mentioned earlier)."""

    def setUp(self):
        self.user = User.objects.create_user('idem_user', password='pw-idem-1234')

    def test_same_title_not_duplicated(self):
        from services.agent.tools.tasks import CreateTaskTool
        tool = CreateTaskTool()
        r1 = tool.execute(self.user, title="Réviser les maths")
        r2 = tool.execute(self.user, title="  réviser les maths ")  # different case/space
        self.assertTrue(r1.success)
        self.assertTrue(r2.success)  # returns the existing one, not an error
        self.assertEqual(
            Task.objects.filter(user=self.user, completed=False).count(), 1
        )

    def test_completed_task_does_not_block_new_one(self):
        from services.agent.tools.tasks import CreateTaskTool
        tool = CreateTaskTool()
        tool.execute(self.user, title="Courses")
        Task.objects.filter(user=self.user, title="Courses").update(completed=True)
        tool.execute(self.user, title="Courses")  # a new active one is allowed
        self.assertEqual(Task.objects.filter(user=self.user, title="Courses").count(), 2)


class CreateTaskDeadlineTest(TestCase):
    """A string deadline ('YYYY-MM-DD') must not crash serialization: the bug
    made create_task return an error ('str' object has no attribute 'isoformat')
    even though the task was created, so the model apologized."""

    def setUp(self):
        self.user = User.objects.create_user('deadline_user', password='pw-dl-12345')

    def test_string_deadline_succeeds_and_is_aware(self):
        from django.utils import timezone as tz
        from services.agent.tools.tasks import CreateTaskTool

        r = CreateTaskTool().execute(self.user, title="Réviser les maths", deadline="2026-07-22")
        self.assertTrue(r.success, msg=f"tool reported failure: {r.message}")
        self.assertTrue(r.data["task"]["deadline"].startswith("2026-07-22"))

        task = Task.objects.get(id=r.data["task"]["id"])
        self.assertIsNotNone(task.deadline)
        self.assertFalse(tz.is_naive(task.deadline))  # aware under USE_TZ
