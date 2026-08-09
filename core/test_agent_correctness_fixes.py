"""Correctness fixes surfaced by the DeepSeek prod stress-test:
- create_block must never create an EXACT duplicate (flexible blocks bypassed
  find_recurring_conflicts, so "move by recreating" left duplicates).
- update_block can move a block to another DAY in place (no delete+create).
- the false-success guard's completion regex must catch DeepSeek phrasings
  ("c'est fait ... est créé", "c'est bloqué", "tout est calé") without matching
  proposals or denials.
"""
from datetime import time

from django.contrib.auth.models import User
from django.test import TestCase

from core.models import RecurringBlock
from services.agent.agent import _claims_completed_mutation
from services.agent.tools.blocks import CreateBlockTool, UpdateBlockTool


class CreateBlockDedupTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("dedup", password="pw-123456")
        self.tool = CreateBlockTool()

    def _create(self, title, bt, day, s, e, flex=None):
        kw = {"title": title, "block_type": bt, "days": [day],
              "start_time": s, "end_time": e}
        if flex:
            kw["flexibility"] = flex
        return self.tool.execute(self.user, **kw)

    def test_exact_duplicate_flexible_is_skipped(self):
        r1 = self._create("Déjeuner", "meal", 0, "12:00", "12:30")
        self.assertTrue(r1.success)
        r2 = self._create("Déjeuner", "meal", 0, "12:00", "12:30")
        self.assertFalse(r2.success)  # nothing created
        self.assertEqual(
            RecurringBlock.objects.filter(user=self.user, title="Déjeuner").count(), 1)

    def test_exact_duplicate_fixed_course_is_skipped(self):
        self._create("Programmation", "course", 0, "08:00", "10:00")
        self._create("Programmation", "course", 0, "08:00", "10:00")
        self.assertEqual(
            RecurringBlock.objects.filter(user=self.user, title="Programmation").count(), 1)

    def test_same_title_different_time_is_allowed(self):
        self._create("Sport", "sport", 0, "08:00", "09:00")
        r2 = self._create("Sport", "sport", 0, "18:00", "19:00")
        self.assertTrue(r2.success)
        self.assertEqual(RecurringBlock.objects.filter(user=self.user, title="Sport").count(), 2)


class UpdateBlockMoveTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("mover", password="pw-123456")

    def _block(self, title, bt, day, s, e):
        return RecurringBlock.objects.create(
            user=self.user, title=title, block_type=bt, day_of_week=day,
            start_time=time.fromisoformat(s), end_time=time.fromisoformat(e))

    def test_move_to_another_day_in_place(self):
        b = self._block("Programmation", "course", 0, "08:00", "10:00")
        r = UpdateBlockTool().execute(self.user, block_id=b.id, day_of_week=1)
        self.assertTrue(r.success, r.message)
        b.refresh_from_db()
        self.assertEqual(b.day_of_week, 1)  # moved to mardi
        # no duplicate created
        self.assertEqual(RecurringBlock.objects.filter(user=self.user).count(), 1)

    def test_move_onto_a_fixed_conflict_is_rejected(self):
        self._block("Reseaux", "course", 2, "08:00", "11:00")  # mercredi fixed wall
        b = self._block("Sport", "sport", 0, "08:30", "10:00")
        # make Sport fixed so it would conflict
        b.flexibility = "fixed"; b.save()
        r = UpdateBlockTool().execute(self.user, block_id=b.id, day_of_week=2, flexibility="fixed")
        self.assertFalse(r.success)
        b.refresh_from_db()
        self.assertEqual(b.day_of_week, 0)  # unchanged

    def test_invalid_day_rejected(self):
        b = self._block("X", "other", 0, "08:00", "09:00")
        r = UpdateBlockTool().execute(self.user, block_id=b.id, day_of_week=9)
        self.assertFalse(r.success)


class CompletionClaimRegexTests(TestCase):
    def test_matches_deepseek_phrasings(self):
        for txt in [
            "C'est fait : Entraînement mercredi 8h30-10h est créé.",
            "C'est bloqué ! Rendez-vous dentiste 18h-19h.",
            "Tout est calé : 9h de sommeil.",
            "Ton cours est déplacé au mardi.",
            "j'ai ajouté ton cours de maths",
            "je t'ai planifié 2h de révision",
        ]:
            self.assertTrue(_claims_completed_mutation(txt), f"devrait matcher: {txt}")

    def test_does_not_match_proposals_or_denials(self):
        for txt in [
            "je propose de créer un bloc demain, ça te va ?",
            "Je n'ai rien modifié du tout.",
            "Voici tes créneaux libres lundi.",
            "Veux-tu que je bloque ce créneau ?",
            "Ton lundi est déjà vide, rien à changer.",
        ]:
            self.assertFalse(_claims_completed_mutation(txt), f"ne devrait PAS matcher: {txt}")
