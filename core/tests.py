"""
Tests for Planner AI backend.
"""
from datetime import time, timedelta
from unittest.mock import patch, MagicMock

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase

from .models import (
    UserProfile,
    UploadedDocument,
    RecurringBlock,
    Task,
    ScheduledBlock,
    ConversationMessage,
)
from services.ai_scheduler import AIScheduler, TimeSlot


class UserProfileModelTest(TestCase):
    """Tests for UserProfile model."""

    def test_profile_created_on_user_creation(self):
        """Test that UserProfile is auto-created when User is created."""
        user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.assertTrue(hasattr(user, 'profile'))
        self.assertIsInstance(user.profile, UserProfile)

    def test_profile_default_values(self):
        """Test default values for UserProfile."""
        user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        profile = user.profile
        self.assertEqual(profile.min_sleep_hours, 7)
        self.assertEqual(profile.max_deep_work_hours_per_day, 4)
        self.assertEqual(profile.peak_productivity_time, 'morning')
        self.assertFalse(profile.onboarding_completed)


class TaskModelTest(TestCase):
    """Tests for Task model."""

    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )

    def test_task_priority_clamping(self):
        """Test that task priority is clamped between 1 and 10."""
        task = Task.objects.create(
            user=self.user,
            title='Test task',
            priority=15
        )
        self.assertEqual(task.priority, 10)

        task.priority = -5
        task.save()
        self.assertEqual(task.priority, 1)

    def test_task_ordering(self):
        """Test that tasks are ordered by priority and deadline."""
        now = timezone.now()
        Task.objects.create(
            user=self.user,
            title='Low priority',
            priority=3
        )
        Task.objects.create(
            user=self.user,
            title='High priority',
            priority=9
        )
        Task.objects.create(
            user=self.user,
            title='Medium with deadline',
            priority=5,
            deadline=now + timedelta(days=1)
        )

        tasks = Task.objects.filter(user=self.user)
        self.assertEqual(tasks[0].title, 'High priority')


class RecurringBlockModelTest(TestCase):
    """Tests for RecurringBlock model."""

    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )

    def test_recurring_block_creation(self):
        """Test creating a recurring block."""
        block = RecurringBlock.objects.create(
            user=self.user,
            title='Math Class',
            block_type='course',
            day_of_week=0,
            start_time=time(9, 0),
            end_time=time(10, 30),
            location='Room A101'
        )
        self.assertEqual(str(block), 'Math Class - Lundi 09:00:00')


class AISchedulerTest(TestCase):
    """Tests for AIScheduler service."""

    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.scheduler = AIScheduler()

    def test_calculate_task_priority(self):
        """Test task priority calculation."""
        now = timezone.now()

        # High priority task with deadline today
        urgent_task = Task.objects.create(
            user=self.user,
            title='Urgent task',
            priority=10,
            deadline=now,
            task_type='deep_work'
        )

        # Low priority task without deadline
        normal_task = Task.objects.create(
            user=self.user,
            title='Normal task',
            priority=5,
            task_type='shallow'
        )

        urgent_score = self.scheduler._calculate_task_priority(
            urgent_task, now.date()
        )
        normal_score = self.scheduler._calculate_task_priority(
            normal_task, now.date()
        )

        self.assertGreater(urgent_score, normal_score)

    def test_time_slot_overlap(self):
        """Test TimeSlot overlap detection."""
        slot = TimeSlot(
            date=timezone.now().date(),
            start_time=time(9, 0),
            end_time=time(12, 0),
            energy_level='high',
            duration_minutes=180
        )

        # Overlapping times
        self.assertTrue(slot.overlaps(time(10, 0), time(11, 0)))
        self.assertTrue(slot.overlaps(time(8, 0), time(10, 0)))

        # Non-overlapping times
        self.assertFalse(slot.overlaps(time(12, 0), time(14, 0)))
        self.assertFalse(slot.overlaps(time(7, 0), time(9, 0)))


class AuthAPITest(APITestCase):
    """Tests for authentication endpoints."""

    def test_user_registration(self):
        """Test user registration endpoint."""
        url = reverse('register')
        data = {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'securepassword123',
            'password_confirm': 'securepassword123',
        }
        response = self.client.post(url, data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('tokens', response.data)
        self.assertIn('access', response.data['tokens'])

    def test_user_registration_password_mismatch(self):
        """Test registration fails with password mismatch."""
        url = reverse('register')
        data = {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'securepassword123',
            'password_confirm': 'differentpassword',
        }
        response = self.client.post(url, data)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_user_login(self):
        """Test user login endpoint."""
        User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        url = reverse('login')
        data = {
            'username': 'testuser',
            'password': 'testpass123',
        }
        response = self.client.post(url, data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('tokens', response.data)

    def test_user_login_invalid_credentials(self):
        """Test login fails with invalid credentials."""
        url = reverse('login')
        data = {
            'username': 'nonexistent',
            'password': 'wrongpassword',
        }
        response = self.client.post(url, data)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class TaskAPITest(APITestCase):
    """Tests for task endpoints."""

    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.client.force_authenticate(user=self.user)

    def test_create_task(self):
        """Test creating a task."""
        url = reverse('task-list')
        data = {
            'title': 'New task',
            'description': 'Task description',
            'task_type': 'deep_work',
            'priority': 7,
        }
        response = self.client.post(url, data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['title'], 'New task')

    def test_list_tasks(self):
        """Test listing tasks."""
        Task.objects.create(user=self.user, title='Task 1')
        Task.objects.create(user=self.user, title='Task 2')

        url = reverse('task-list')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['results']), 2)

    def test_complete_task(self):
        """Test completing a task."""
        task = Task.objects.create(user=self.user, title='Task to complete')
        url = reverse('task-complete', kwargs={'pk': task.id})
        response = self.client.post(url, {'actual_duration_minutes': 45})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['completed'])

    def test_filter_tasks_by_completed(self):
        """Test filtering tasks by completed status."""
        Task.objects.create(user=self.user, title='Pending', completed=False)
        Task.objects.create(user=self.user, title='Done', completed=True)

        url = reverse('task-list') + '?completed=false'
        response = self.client.get(url)
        self.assertEqual(len(response.data['results']), 1)
        self.assertEqual(response.data['results'][0]['title'], 'Pending')


class ScheduleAPITest(APITestCase):
    """Tests for schedule endpoints."""

    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.client.force_authenticate(user=self.user)

    def test_get_schedule(self):
        """Test getting weekly schedule."""
        RecurringBlock.objects.create(
            user=self.user,
            title='Math',
            block_type='course',
            day_of_week=0,
            start_time=time(9, 0),
            end_time=time(10, 0)
        )

        url = reverse('schedule')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('recurring_blocks', response.data)
        self.assertEqual(len(response.data['recurring_blocks']), 1)

    def test_generate_schedule(self):
        """Test schedule generation."""
        Task.objects.create(
            user=self.user,
            title='Task to schedule',
            estimated_duration_minutes=60
        )

        url = reverse('schedule-generate')
        response = self.client.post(url, {'force': True})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('created_blocks', response.data)


class HealthCheckTest(APITestCase):
    """Tests for health check endpoint."""

    def test_health_check(self):
        """Test health check returns healthy status."""
        url = reverse('health_check')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'healthy')


class ProfileAPITest(APITestCase):
    """Tests for profile endpoints."""

    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.client.force_authenticate(user=self.user)

    def test_get_profile(self):
        """Test getting user profile."""
        url = reverse('profile')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['min_sleep_hours'], 7)

    def test_update_profile(self):
        """Test updating user profile."""
        url = reverse('profile')
        data = {
            'peak_productivity_time': 'evening',
            'max_deep_work_hours_per_day': 6,
        }
        response = self.client.patch(url, data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['peak_productivity_time'], 'evening')
        self.assertEqual(response.data['max_deep_work_hours_per_day'], 6)

    def test_onboarding_status(self):
        """Test getting onboarding status."""
        url = reverse('onboarding-status')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(response.data['completed'])
        self.assertEqual(response.data['current_step'], 0)
