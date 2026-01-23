"""
Management command to create a demo user with sample data.
"""
from datetime import time, timedelta

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from django.utils import timezone

from core.models import (
    UserProfile,
    RecurringBlock,
    Task,
    ConversationMessage,
)


class Command(BaseCommand):
    help = 'Create a demo user with sample data for testing'

    def add_arguments(self, parser):
        parser.add_argument(
            '--username',
            type=str,
            default='demo',
            help='Username for demo user (default: demo)'
        )
        parser.add_argument(
            '--password',
            type=str,
            default='demo123',
            help='Password for demo user (default: demo123)'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Delete existing demo user if exists'
        )

    def handle(self, *args, **options):
        username = options['username']
        password = options['password']
        force = options['force']

        # Check if user exists
        if User.objects.filter(username=username).exists():
            if force:
                self.stdout.write(f'Deleting existing user "{username}"...')
                User.objects.filter(username=username).delete()
            else:
                self.stdout.write(
                    self.style.WARNING(
                        f'User "{username}" already exists. Use --force to recreate.'
                    )
                )
                return

        # Create user
        self.stdout.write(f'Creating demo user "{username}"...')
        user = User.objects.create_user(
            username=username,
            email=f'{username}@example.com',
            password=password,
            first_name='Demo',
            last_name='User'
        )

        # Update profile
        profile = user.profile
        profile.min_sleep_hours = 7
        profile.peak_productivity_time = 'morning'
        profile.max_deep_work_hours_per_day = 4
        profile.transport_time_minutes = 15
        profile.onboarding_completed = True
        profile.onboarding_step = 3
        profile.save()

        # Create sample recurring blocks (course schedule)
        self.stdout.write('Creating sample recurring blocks...')

        courses = [
            # Monday
            {'title': 'Mathématiques', 'day': 0, 'start': '09:00', 'end': '10:30', 'location': 'A-101'},
            {'title': 'Informatique', 'day': 0, 'start': '14:00', 'end': '16:00', 'location': 'B-205'},
            # Tuesday
            {'title': 'Physique', 'day': 1, 'start': '10:00', 'end': '12:00', 'location': 'C-110'},
            # Wednesday
            {'title': 'Mathématiques', 'day': 2, 'start': '09:00', 'end': '10:30', 'location': 'A-101'},
            {'title': 'TP Informatique', 'day': 2, 'start': '14:00', 'end': '17:00', 'location': 'Labo-1'},
            # Thursday
            {'title': 'Anglais', 'day': 3, 'start': '11:00', 'end': '12:30', 'location': 'D-301'},
            # Friday
            {'title': 'Projet tutoré', 'day': 4, 'start': '09:00', 'end': '12:00', 'location': 'Salle projet'},
        ]

        for course in courses:
            start_parts = course['start'].split(':')
            end_parts = course['end'].split(':')
            RecurringBlock.objects.create(
                user=user,
                title=course['title'],
                block_type='course',
                day_of_week=course['day'],
                start_time=time(int(start_parts[0]), int(start_parts[1])),
                end_time=time(int(end_parts[0]), int(end_parts[1])),
                location=course['location'],
            )

        # Create sample work shifts
        work_shifts = [
            {'day': 5, 'start': '10:00', 'end': '18:00'},  # Saturday
            {'day': 6, 'start': '14:00', 'end': '22:00'},  # Sunday
        ]

        for shift in work_shifts:
            start_parts = shift['start'].split(':')
            end_parts = shift['end'].split(':')
            RecurringBlock.objects.create(
                user=user,
                title='Travail',
                block_type='work',
                day_of_week=shift['day'],
                start_time=time(int(start_parts[0]), int(start_parts[1])),
                end_time=time(int(end_parts[0]), int(end_parts[1])),
            )

        # Create sample tasks
        self.stdout.write('Creating sample tasks...')

        now = timezone.now()
        tasks_data = [
            {
                'title': 'Réviser le chapitre 5 de maths',
                'task_type': 'deep_work',
                'deadline': now + timedelta(days=3),
                'estimated_duration_minutes': 90,
                'priority': 8,
                'related_course': 'Mathématiques',
            },
            {
                'title': 'Finir le TP informatique',
                'task_type': 'deep_work',
                'deadline': now + timedelta(days=5),
                'estimated_duration_minutes': 120,
                'priority': 9,
                'related_course': 'Informatique',
            },
            {
                'title': 'Lire les notes de physique',
                'task_type': 'shallow',
                'deadline': now + timedelta(days=2),
                'estimated_duration_minutes': 45,
                'priority': 6,
                'related_course': 'Physique',
            },
            {
                'title': 'Préparer présentation anglais',
                'task_type': 'deep_work',
                'deadline': now + timedelta(days=7),
                'estimated_duration_minutes': 60,
                'priority': 7,
                'related_course': 'Anglais',
            },
            {
                'title': 'Acheter fournitures',
                'task_type': 'errand',
                'deadline': now + timedelta(days=4),
                'estimated_duration_minutes': 30,
                'priority': 4,
            },
            {
                'title': 'Revoir exercices corrigés',
                'task_type': 'shallow',
                'deadline': None,
                'estimated_duration_minutes': 40,
                'priority': 5,
                'related_course': 'Mathématiques',
            },
        ]

        for task_data in tasks_data:
            Task.objects.create(user=user, **task_data)

        # Create sample conversation messages
        self.stdout.write('Creating sample conversation...')

        messages = [
            ('assistant', "Salut! Je suis ton assistant de planification. Comment puis-je t'aider?"),
            ('user', "J'ai un exam de maths lundi prochain"),
            ('assistant', "J'ai noté ton examen de maths pour lundi. Veux-tu que je t'aide à planifier des créneaux de révision?"),
        ]

        for role, content in messages:
            ConversationMessage.objects.create(
                user=user,
                role=role,
                content=content,
            )

        self.stdout.write(
            self.style.SUCCESS(
                f'\nDemo user created successfully!\n'
                f'Username: {username}\n'
                f'Password: {password}\n'
                f'- {len(courses) + len(work_shifts)} recurring blocks\n'
                f'- {len(tasks_data)} tasks\n'
                f'- {len(messages)} conversation messages'
            )
        )
