"""
Models for Planner AI backend.
"""
from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


class UserProfile(models.Model):
    """Extended user profile with planning preferences."""

    PRODUCTIVITY_CHOICES = [
        ('morning', 'Matin'),
        ('afternoon', 'Après-midi'),
        ('evening', 'Soir'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    min_sleep_hours = models.PositiveIntegerField(default=7)
    post_night_shift_wake_time = models.TimeField(null=True, blank=True)
    peak_productivity_time = models.CharField(
        max_length=20,
        choices=PRODUCTIVITY_CHOICES,
        default='morning'
    )
    max_deep_work_hours_per_day = models.PositiveIntegerField(default=4)
    transport_time_minutes = models.PositiveIntegerField(default=0)
    onboarding_completed = models.BooleanField(default=False)
    onboarding_step = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Profile de {self.user.username}"

    class Meta:
        verbose_name = "Profil utilisateur"
        verbose_name_plural = "Profils utilisateurs"


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """Create a UserProfile when a User is created."""
    if created:
        UserProfile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """Save the UserProfile when the User is saved."""
    if hasattr(instance, 'profile'):
        instance.profile.save()


class UploadedDocument(models.Model):
    """Document uploaded by user (PDF, image)."""

    DOCUMENT_TYPE_CHOICES = [
        ('course_schedule', 'Horaire de cours'),
        ('work_schedule', 'Horaire de travail'),
        ('other', 'Autre'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='documents')
    file = models.FileField(upload_to='documents/')
    document_type = models.CharField(
        max_length=20,
        choices=DOCUMENT_TYPE_CHOICES,
        default='other'
    )
    extracted_data = models.JSONField(default=dict, blank=True)
    processed = models.BooleanField(default=False)
    processing_error = models.TextField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.get_document_type_display()} - {self.user.username}"

    class Meta:
        verbose_name = "Document uploadé"
        verbose_name_plural = "Documents uploadés"
        ordering = ['-uploaded_at']


class RecurringBlock(models.Model):
    """Recurring time block (course, work, sleep, etc.)."""

    BLOCK_TYPE_CHOICES = [
        ('course', 'Cours'),
        ('work', 'Travail'),
        ('sleep', 'Sommeil'),
        ('meal', 'Repas'),
        ('sport', 'Sport'),
        ('project', 'Projet'),
        ('other', 'Autre'),
    ]

    DAY_CHOICES = [
        (0, 'Lundi'),
        (1, 'Mardi'),
        (2, 'Mercredi'),
        (3, 'Jeudi'),
        (4, 'Vendredi'),
        (5, 'Samedi'),
        (6, 'Dimanche'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='recurring_blocks')
    title = models.CharField(max_length=200)
    block_type = models.CharField(max_length=20, choices=BLOCK_TYPE_CHOICES)
    day_of_week = models.PositiveIntegerField(choices=DAY_CHOICES)
    start_time = models.TimeField()
    end_time = models.TimeField()
    location = models.CharField(max_length=200, blank=True)
    is_night_shift = models.BooleanField(default=False)
    source_document = models.ForeignKey(
        UploadedDocument,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='recurring_blocks'
    )
    active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} - {self.get_day_of_week_display()} {self.start_time}"

    class Meta:
        verbose_name = "Bloc récurrent"
        verbose_name_plural = "Blocs récurrents"
        ordering = ['day_of_week', 'start_time']


class Task(models.Model):
    """Task to be scheduled."""

    TASK_TYPE_CHOICES = [
        ('deep_work', 'Travail en profondeur'),
        ('shallow', 'Tâche légère'),
        ('errand', 'Course/Admin'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='tasks')
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    deadline = models.DateTimeField(null=True, blank=True)
    estimated_duration_minutes = models.PositiveIntegerField(null=True, blank=True)
    task_type = models.CharField(
        max_length=20,
        choices=TASK_TYPE_CHOICES,
        default='shallow'
    )
    priority = models.PositiveIntegerField(default=5)  # 1-10
    related_course = models.CharField(max_length=100, blank=True)
    completed = models.BooleanField(default=False)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        status = "✓" if self.completed else "○"
        return f"{status} {self.title}"

    def save(self, *args, **kwargs):
        # Clamp priority between 1 and 10
        if self.priority < 1:
            self.priority = 1
        elif self.priority > 10:
            self.priority = 10
        super().save(*args, **kwargs)

    class Meta:
        verbose_name = "Tâche"
        verbose_name_plural = "Tâches"
        ordering = ['-priority', 'deadline', '-created_at']


class ScheduledBlock(models.Model):
    """Scheduled instance of a task."""

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='scheduled_blocks')
    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name='scheduled_blocks')
    date = models.DateField()
    start_time = models.TimeField()
    end_time = models.TimeField()
    actually_completed = models.BooleanField(default=False)
    actual_duration_minutes = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.task.title} - {self.date} {self.start_time}"

    class Meta:
        verbose_name = "Bloc planifié"
        verbose_name_plural = "Blocs planifiés"
        ordering = ['date', 'start_time']


class RecurringBlockCompletion(models.Model):
    """Tracks completion status of recurring blocks on specific dates."""

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='recurring_completions')
    recurring_block = models.ForeignKey(
        RecurringBlock,
        on_delete=models.CASCADE,
        related_name='completions'
    )
    date = models.DateField()
    completed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.recurring_block.title} - {self.date}"

    class Meta:
        verbose_name = "Complétion bloc récurrent"
        verbose_name_plural = "Complétions blocs récurrents"
        unique_together = ['recurring_block', 'date']
        ordering = ['-date', '-completed_at']


class ConversationMessage(models.Model):
    """Message in the chat conversation."""

    ROLE_CHOICES = [
        ('user', 'Utilisateur'),
        ('assistant', 'Assistant'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    attachment = models.ForeignKey(
        UploadedDocument,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='messages'
    )
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"[{self.role}] {preview}"

    class Meta:
        verbose_name = "Message"
        verbose_name_plural = "Messages"
        ordering = ['created_at']
