"""QA #2 deterministic planner/agent-contract regressions.

These tests cover invariants from ``serie-tests-planner-ia`` that do not need a
live LLM: metamorphic idempotence, locked-block sanctuaries, and one-off event
upserts exposed through the agent tools.
"""
from datetime import date, time
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from rest_framework.test import APIClient

from core.models import RecurringBlock, ScheduledBlock, Task
from services.agent.agent import PlannerAgent
from services.agent.tools.blocks import CreateBlockTool, UpdateBlockTool
from services.agent.tools.schedule import ScheduleTaskAtTool
from services.llm.base import FunctionCall, LLMProvider, LLMResponse
from services.replan import replan_after_delay
from services.scheduling.placement import fixed_busy_intervals, place_day


MONDAY = date(2026, 7, 27)
WED = date(2026, 7, 29)


class GenerateIdempotenceTests(TestCase):
    """M05: two generate calls against the same state must not duplicate tasks."""

    def setUp(self):
        self.user = User.objects.create_user("qa2_generate", password="pw")
        self.client = APIClient()
        self.client.force_authenticate(self.user)
        self.task = Task.objects.create(
            user=self.user,
            title="Lecture",
            estimated_duration_minutes=60,
            priority=5,
        )

    def test_schedule_generate_does_not_recreate_already_scheduled_task(self):
        first = self.client.post(
            reverse("schedule-generate"),
            {"start_date": MONDAY.isoformat()},
            format="json",
        )
        self.assertEqual(first.status_code, 200, first.content)
        self.assertEqual(len(first.data["created_blocks"]), 1)
        original = ScheduledBlock.objects.get(user=self.user, task=self.task)
        original_slot = (original.date, original.start_time, original.end_time)

        second = self.client.post(
            reverse("schedule-generate"),
            {"start_date": MONDAY.isoformat()},
            format="json",
        )

        self.assertEqual(second.status_code, 200, second.content)
        self.assertEqual(second.data["created_blocks"], [])
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user, task=self.task).count(), 1)
        original.refresh_from_db()
        self.assertEqual((original.date, original.start_time, original.end_time), original_slot)


class ScheduleTaskAtIdempotenceTests(TestCase):
    """N06/M05 contract: repeating or refining the same one-off event is an upsert."""

    def setUp(self):
        self.user = User.objects.create_user("qa2_oneoff", password="pw")
        self.target_date = MONDAY
        self.tool = ScheduleTaskAtTool()

    def test_same_one_off_event_is_idempotent_not_a_conflict(self):
        first = self.tool.execute(
            self.user,
            title="Lecture",
            date=self.target_date.isoformat(),
            start_time="09:00",
            end_time="09:30",
        )
        self.assertTrue(first.success, first.message)

        second = self.tool.execute(
            self.user,
            title="Lecture",
            date=self.target_date.isoformat(),
            start_time="09:00",
            end_time="09:30",
        )

        self.assertTrue(second.success, second.message)
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 1)
        self.assertEqual(
            second.data["scheduled_block"]["id"],
            first.data["scheduled_block"]["id"],
        )

    def test_same_one_off_event_duration_refinement_updates_existing_block(self):
        first = self.tool.execute(
            self.user,
            title="Lecture",
            date=self.target_date.isoformat(),
            start_time="09:00",
            end_time="09:30",
        )
        self.assertTrue(first.success, first.message)

        refined = self.tool.execute(
            self.user,
            title="Lecture",
            date=self.target_date.isoformat(),
            start_time="09:00",
            end_time="09:45",
        )

        self.assertTrue(refined.success, refined.message)
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 1)
        block = ScheduledBlock.objects.get(user=self.user)
        self.assertEqual(block.start_time, time(9, 0))
        self.assertEqual(block.end_time, time(9, 45))


class ReplanLockedBlockTests(TestCase):
    """M06: replan may move displaced flexibles, never locked blocks."""

    def test_replan_preserves_locked_block_strictly(self):
        user = User.objects.create_user("qa2_locked", password="pw")
        user.profile.automation_mode = "automatic"
        user.profile.save()
        today = timezone.localdate()
        locked_task = Task.objects.create(user=user, title="Sport verrouille", estimated_duration_minutes=60)
        moving_task = Task.objects.create(user=user, title="Lecture", estimated_duration_minutes=60)
        locked = ScheduledBlock.objects.create(
            user=user,
            task=locked_task,
            date=today,
            start_time=time(9, 0),
            end_time=time(10, 0),
            locked=True,
        )
        ScheduledBlock.objects.create(
            user=user,
            task=moving_task,
            date=today,
            start_time=time(8, 0),
            end_time=time(9, 0),
        )

        result = replan_after_delay(user, resume_time="11:00")

        self.assertTrue(result["applied"])
        locked.refresh_from_db()
        self.assertEqual((locked.date, locked.start_time, locked.end_time), (today, time(9, 0), time(10, 0)))
        self.assertFalse(any(m["task_id"] == locked_task.id for m in result["moved"]))


class PlacementMetamorphicTests(TestCase):
    """M03/M04/M07: irrelevant metadata and shorter durations cannot add conflicts."""

    def test_renaming_flexible_block_does_not_change_placement_math(self):
        user = User.objects.create_user("qa2_rename", password="pw")
        block = RecurringBlock.objects.create(
            user=user,
            title="Lecture",
            block_type="revision",
            day_of_week=MONDAY.weekday(),
            start_time=time(9, 0),
            end_time=time(10, 0),
        )
        RecurringBlock.objects.create(
            user=user,
            title="Cours",
            block_type="course",
            day_of_week=MONDAY.weekday(),
            start_time=time(9, 30),
            end_time=time(11, 0),
        )
        before = next(p for p in place_day(user, MONDAY) if p["block_id"] == block.id)

        block.title = "Activite A"
        block.save(update_fields=["title"])
        after = next(p for p in place_day(user, MONDAY) if p["block_id"] == block.id)

        self.assertEqual(
            (after["start_min"], after["end_min"], after["preferred"], after["shrunk"], after["skipped"]),
            (before["start_min"], before["end_min"], before["preferred"], before["shrunk"], before["skipped"]),
        )

    def test_reducing_flexible_duration_does_not_create_overlap(self):
        user = User.objects.create_user("qa2_reduce", password="pw")
        block = RecurringBlock.objects.create(
            user=user,
            title="Revision",
            block_type="revision",
            day_of_week=MONDAY.weekday(),
            start_time=time(11, 0),
            end_time=time(12, 0),
            duration_minutes=60,
        )
        RecurringBlock.objects.create(
            user=user,
            title="Cours",
            block_type="course",
            day_of_week=MONDAY.weekday(),
            start_time=time(11, 30),
            end_time=time(13, 0),
        )
        longer = next(p for p in place_day(user, MONDAY) if p["block_id"] == block.id)
        self.assertFalse(longer["skipped"])

        block.duration_minutes = 30
        block.save(update_fields=["duration_minutes"])
        shorter = next(p for p in place_day(user, MONDAY) if p["block_id"] == block.id)

        self.assertFalse(shorter["skipped"])
        self.assertFalse(shorter["shrunk"])
        self.assertEqual(shorter["end_min"] - shorter["start_min"], 30)


class _StaticProvider(LLMProvider):
    def __init__(self, text):
        self.text = text

    def generate(self, prompt, tools=None, system_prompt=None):
        return LLMResponse(text=self.text)

    def generate_with_history(self, messages, tools=None, system_prompt=None):
        return LLMResponse(text=self.text)

    def is_available(self):
        return True

    @property
    def name(self):
        return "static"


class _ScriptedProvider(LLMProvider):
    """Rend une séquence de réponses (un tour d'outil, puis un tour de texte)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0

    def _next(self):
        r = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return r

    def generate(self, prompt, tools=None, system_prompt=None):
        return self._next()

    def generate_with_history(self, messages, tools=None, system_prompt=None):
        return self._next()

    def is_available(self):
        return True

    @property
    def name(self):
        return "scripted"


class AgentActionTruthTests(TestCase):
    """Conversation contract: a write ATTEMPTED this turn that fails must never be
    spun as a success; but a true recap of a PAST action (no tool this turn) passes."""

    def test_failed_write_claimed_as_success_is_replaced(self):
        user = User.objects.create_user("qa2_truth", password="pw")
        # Tour 1: le LLM tente create_block avec un type invalide -> échec (success=False).
        # Tour 2: il prétend malgré tout avoir bloqué le créneau.
        responses = [
            LLMResponse(function_calls=[FunctionCall(
                name="create_block",
                args={"title": "Lecture", "block_type": "invalid_type",
                      "days": [0], "start_time": "09:00", "end_time": "10:00"},
                call_id="c1",
            )]),
            LLMResponse(text="J'ai bloqué ta lecture lundi de 09h00 à 10h00."),
        ]
        provider = _ScriptedProvider(responses)

        with patch("services.agent.agent.get_provider", return_value=provider):
            result = PlannerAgent().process_message(user, "Bloque ma lecture lundi de 9h à 10h")

        # La garde REMPLACE le faux succès: la phrase mensongère ("bloqué") ne
        # subsiste pas (sinon message contradictoire), et rien n'est créé.
        self.assertIn("échoué", result["response"])
        self.assertNotIn("bloqué", result["response"])
        self.assertEqual(RecurringBlock.objects.filter(user=user).count(), 0)

    def test_past_recap_without_write_is_left_intact(self):
        # Régression (revue adversariale): un récapitulatif VRAI d'une action
        # passée n'appelle aucun outil ce tour -> la garde ne doit PAS le démentir.
        user = User.objects.create_user("qa2_recap", password="pw")
        provider = _StaticProvider("Oui, j'ai créé ton bloc sport hier à 18h.")

        with patch("services.agent.agent.get_provider", return_value=provider):
            result = PlannerAgent().process_message(user, "Tu as bien créé mon bloc sport hier ?")

        self.assertEqual(result["response"], "Oui, j'ai créé ton bloc sport hier à 18h.")

    def test_plain_proposal_without_write_is_left_as_proposal(self):
        user = User.objects.create_user("qa2_proposal", password="pw")
        provider = _StaticProvider("Je propose de bloquer ta lecture lundi de 09h00 à 10h00.")

        with patch("services.agent.agent.get_provider", return_value=provider):
            result = PlannerAgent().process_message(user, "Propose un creneau")

        self.assertEqual(result["response"], "Je propose de bloquer ta lecture lundi de 09h00 à 10h00.")


class RecurringProtectionToolTests(TestCase):
    """OR5 contract: user protection rules must become real fixed walls."""

    def test_update_block_can_lock_flexible_recurring_sport(self):
        user = User.objects.create_user("qa2_recurring_lock", password="pw")
        sport = RecurringBlock.objects.create(
            user=user,
            title="Sport du mercredi",
            block_type="sport",
            day_of_week=2,
            start_time=time(18, 0),
            end_time=time(19, 0),
        )
        self.assertTrue(sport.is_flexible)

        locked = UpdateBlockTool().execute(user, block_id=sport.id, flexibility="fixed")

        self.assertTrue(locked.success, locked.message)
        sport.refresh_from_db()
        self.assertFalse(sport.is_flexible)
        self.assertEqual(fixed_busy_intervals(user, WED), [(18 * 60, 19 * 60)])
        self.assertEqual(place_day(user, WED), [])

        conflict = CreateBlockTool().execute(
            user,
            title="Travail conflit",
            block_type="work",
            days=[2],
            start_time="18:00",
            end_time="20:00",
        )

        self.assertFalse(conflict.success)
        self.assertEqual(conflict.data["created"], [])
        self.assertIn("Chevauchement", conflict.data["skipped"][0]["reason"])
