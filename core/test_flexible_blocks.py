from datetime import time

from django.contrib.auth.models import User
from django.test import TestCase

from core.models import RecurringBlock
from services.agent.tools.blocks import CreateBlockTool
from services.scheduling.overlap import find_recurring_conflicts


class FlexibleRecurringBlockTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(
            username='flexuser',
            password='pw-flex-123456',
        )

    def test_default_flexibility_for_block_types(self):
        for block_type in ('work', 'course'):
            self.assertEqual(
                RecurringBlock.default_flexibility_for(block_type),
                RecurringBlock.FLEXIBILITY_FIXED,
            )

        for block_type in ('sleep', 'meal', 'sport', 'project', 'revision', 'other'):
            self.assertEqual(
                RecurringBlock.default_flexibility_for(block_type),
                RecurringBlock.FLEXIBILITY_FLEXIBLE,
            )

    def test_is_flexible_uses_explicit_value_or_type_default(self):
        self.assertTrue(
            RecurringBlock(block_type='work', flexibility='flexible').is_flexible
        )
        self.assertFalse(
            RecurringBlock(block_type='sleep', flexibility='fixed').is_flexible
        )
        self.assertTrue(
            RecurringBlock(block_type='sleep', flexibility=None).is_flexible
        )
        self.assertFalse(
            RecurringBlock(block_type='course', flexibility=None).is_flexible
        )

    def test_effective_duration_minutes(self):
        normal = RecurringBlock(start_time=time(9, 0), end_time=time(11, 0))
        self.assertEqual(normal.effective_duration_minutes(), 120)

        overnight = RecurringBlock(start_time=time(23, 0), end_time=time(7, 0))
        self.assertEqual(overnight.effective_duration_minutes(), 480)

        override = RecurringBlock(
            start_time=time(9, 0),
            end_time=time(11, 0),
            duration_minutes=45,
        )
        self.assertEqual(override.effective_duration_minutes(), 45)

    def test_save_sets_default_flexibility(self):
        sleep = RecurringBlock.objects.create(
            user=self.user,
            title='Sommeil',
            block_type='sleep',
            day_of_week=0,
            start_time=time(0, 0),
            end_time=time(7, 0),
        )
        work = RecurringBlock.objects.create(
            user=self.user,
            title='Travail',
            block_type='work',
            day_of_week=0,
            start_time=time(9, 0),
            end_time=time(17, 0),
        )

        self.assertEqual(sleep.flexibility, RecurringBlock.FLEXIBILITY_FLEXIBLE)
        self.assertEqual(work.flexibility, RecurringBlock.FLEXIBILITY_FIXED)

    def test_find_recurring_conflicts_respects_flexibility(self):
        RecurringBlock.objects.create(
            user=self.user,
            title='Sommeil',
            block_type='sleep',
            day_of_week=6,
            start_time=time(0, 0),
            end_time=time(7, 0),
        )

        conflicts = find_recurring_conflicts(
            self.user,
            5,
            time(19, 0),
            time(7, 0),
            night_flag=True,
        )
        self.assertEqual(conflicts, [])

        course = RecurringBlock.objects.create(
            user=self.user,
            title='Cours',
            block_type='course',
            day_of_week=6,
            start_time=time(1, 0),
            end_time=time(3, 0),
        )
        conflicts = find_recurring_conflicts(
            self.user,
            6,
            time(2, 0),
            time(4, 0),
        )
        self.assertEqual([block.id for block in conflicts], [course.id])

        conflicts = find_recurring_conflicts(
            self.user,
            6,
            time(2, 0),
            time(4, 0),
            new_is_flexible=True,
        )
        self.assertEqual(conflicts, [])

    def test_create_block_tool_fixed_work_can_overlap_flexible_sleep(self):
        RecurringBlock.objects.create(
            user=self.user,
            title='Sommeil',
            block_type='sleep',
            day_of_week=6,
            start_time=time(0, 0),
            end_time=time(7, 0),
        )

        result = CreateBlockTool().execute(
            self.user,
            title='Travail',
            block_type='work',
            days=[5],
            start_time='19:00',
            end_time='07:00',
        )

        self.assertTrue(result.success)
        self.assertEqual(len(result.data['created']), 1)
        self.assertEqual(result.data['skipped'], [])

        work = RecurringBlock.objects.get(user=self.user, title='Travail')
        self.assertEqual(work.flexibility, RecurringBlock.FLEXIBILITY_FIXED)
        self.assertTrue(work.is_night_shift)
