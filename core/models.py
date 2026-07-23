"""
Models for Planner AI backend.
"""
import uuid
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

    AUTOMATION_CHOICES = [
        ('suggestion', 'Suggestion (tout est proposé)'),
        ('semi_auto', 'Semi-automatique (petits changements auto)'),
        ('automatic', 'Automatique (déplace, explique, annulable)'),
    ]

    LLM_CHOICES = [
        ('gemini', 'Gemini (Google)'),
        ('claude', 'Claude (Anthropic)'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    avatar_url = models.URLField(max_length=500, blank=True, null=True)
    preferred_llm = models.CharField(
        max_length=20,
        choices=LLM_CHOICES,
        default='gemini',
        help_text="Modèle IA préféré pour le chat"
    )
    min_sleep_hours = models.PositiveIntegerField(default=7)
    post_night_shift_wake_time = models.TimeField(null=True, blank=True)
    peak_productivity_time = models.CharField(
        max_length=20,
        choices=PRODUCTIVITY_CHOICES,
        default='morning'
    )
    max_deep_work_hours_per_day = models.PositiveIntegerField(default=4)
    transport_time_minutes = models.PositiveIntegerField(default=0)
    prep_time_minutes = models.PositiveIntegerField(
        default=15,
        help_text="Temps de préparation avant de partir (minutes).",
    )
    safety_margin_minutes = models.PositiveIntegerField(
        default=10,
        help_text="Marge de sécurité contre les retards (minutes).",
    )
    automation_mode = models.CharField(
        max_length=12,
        choices=AUTOMATION_CHOICES,
        default='semi_auto',
        help_text="Niveau d'automatisation des réajustements (spec §8).",
    )
    auto_apply_threshold_minutes = models.PositiveIntegerField(
        default=60,
        help_text="En semi-auto: un déplacement au-delà de ce seuil est 'important' (proposé, pas appliqué).",
    )
    onboarding_completed = models.BooleanField(default=False)
    onboarding_step = models.PositiveIntegerField(default=0)
    energy_levels = models.JSONField(
        default=dict,
        blank=True,
        help_text="Niveaux d'énergie par créneau (ex: {'morning': 'high', 'evening': 'low'})"
    )
    notification_preferences = models.JSONField(
        default=dict,
        blank=True,
        help_text="Préférences de notifications (canaux, horaires, types activés)"
    )
    public_planning_enabled = models.BooleanField(
        default=False,
        help_text="Autorise le partage public du planning via /planning/<username>/"
    )
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
    file_name = models.CharField(max_length=255, blank=True)
    content_hash = models.CharField(max_length=64, blank=True, db_index=True)
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


class UserPlace(models.Model):
    """A place the user goes to (work, school, gym...) with its travel time.

    Phase 1 of the planning engine: travel_minutes is the user-declared usual
    trip duration from home. A later phase can refresh it live from a maps API
    (traffic-aware) without changing the scheduling formula that consumes it:

        heure_de_depart      = début_événement - trajet - marge_de_sécurité
        début_indisponibilité = heure_de_depart - temps_de_préparation
    """

    KIND_CHOICES = [
        ('home', 'Maison'),
        ('school', 'École'),
        ('work', 'Travail'),
        ('other', 'Autre'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='places')
    name = models.CharField(max_length=100)
    kind = models.CharField(max_length=10, choices=KIND_CHOICES, default='other')
    address = models.CharField(max_length=300, blank=True)
    travel_minutes = models.PositiveIntegerField(
        default=0,
        help_text="Durée habituelle du trajet depuis la maison (minutes).",
    )
    # Coordonnées géocodées depuis `address` (rappels basés sur la localisation,
    # Phase 0). Peuplées au mieux via Nominatim; null si géocodage indisponible.
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def has_coordinates(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    def __str__(self):
        return f"{self.name} ({self.travel_minutes} min) - {self.user.username}"

    class Meta:
        verbose_name = "Lieu"
        verbose_name_plural = "Lieux"
        ordering = ['name']
        constraints = [
            models.UniqueConstraint(fields=['user', 'name'], name='userplace_user_name_unique'),
        ]


class VisibleRecurringBlockManager(models.Manager):
    """Default manager that hides blocks awaiting confirmation.

    Extraction can create low-confidence blocks with status='pending'. They must
    NOT appear in the planning, the iCal feed, reminders, public shares, or the
    agent context until the user confirms them. Making this the default manager
    excludes pending blocks from every existing ``RecurringBlock.objects`` query
    by construction, so no read site can accidentally leak them. Use
    ``RecurringBlock.all_objects`` to reach pending blocks (confirm/reject flow).
    """

    def get_queryset(self):
        return super().get_queryset().exclude(status=RecurringBlock.STATUS_PENDING)


class RecurringBlock(models.Model):
    """Recurring time block (course, work, sleep, etc.)."""

    STATUS_ACTIVE = 'active'
    STATUS_PENDING = 'pending'
    STATUS_CHOICES = [
        (STATUS_ACTIVE, 'Actif'),
        (STATUS_PENDING, 'À confirmer'),
    ]

    BLOCK_TYPE_CHOICES = [
        ('course', 'Cours'),
        ('work', 'Travail'),
        ('sleep', 'Sommeil'),
        ('meal', 'Repas'),
        ('sport', 'Sport'),
        ('project', 'Projet'),
        ('revision', 'Révision'),
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
    place = models.ForeignKey(
        UserPlace,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='recurring_blocks',
        help_text="Lieu de l'activité ; son travel_minutes pilote l'heure limite de départ.",
    )
    is_night_shift = models.BooleanField(default=False)
    source_document = models.ForeignKey(
        UploadedDocument,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='recurring_blocks'
    )
    active = models.BooleanField(default=True)
    status = models.CharField(
        max_length=10,
        choices=STATUS_CHOICES,
        default=STATUS_ACTIVE,
        db_index=True,
        help_text="'pending' = extrait mais en attente de confirmation utilisateur.",
    )
    confidence = models.FloatField(
        null=True,
        blank=True,
        help_text="Confiance d'extraction 0-1 (null si créé manuellement).",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    # Default manager hides pending blocks; all_objects sees everything.
    objects = VisibleRecurringBlockManager()
    all_objects = models.Manager()

    def __str__(self):
        return f"{self.title} - {self.get_day_of_week_display()} {self.start_time}"

    class Meta:
        verbose_name = "Bloc récurrent"
        verbose_name_plural = "Blocs récurrents"
        ordering = ['day_of_week', 'start_time']
        # Cascade deletes / integrity must see pending blocks too.
        base_manager_name = 'all_objects'
        indexes = [
            models.Index(
                fields=['user', 'day_of_week', 'active'],
                name='recurblock_user_day_active_idx',
            ),
        ]


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
    # Lieu du rendez-vous (rappels de départ basés sur la localisation, Phase 0/1):
    # une tâche "Réunion 14h à l'UQAC" porte un UserPlace -> trajet + coords.
    place = models.ForeignKey(
        UserPlace,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='tasks',
    )
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

    def _ensure_task_history(self, actual_minutes=None):
        """
        Idempotently ensure a TaskHistory row exists for this completed task.

        Dedup key is (user, task_title, completed_at). Because completed_at is
        set once and preserved (never overwritten once present), this stays
        stable across repeated calls, so no duplicate history is created.
        """
        from django.utils import timezone

        completed_at = self.completed_at or timezone.now()
        already = TaskHistory.objects.filter(
            user=self.user,
            task_title=self.title,
            completed_at=completed_at,
        ).exists()
        if already:
            return

        block = self.scheduled_blocks.order_by('date', 'start_time').first()

        if actual_minutes is not None:
            actual = actual_minutes
        elif block is not None and block.actual_duration_minutes is not None:
            actual = block.actual_duration_minutes
        elif self.estimated_duration_minutes is not None:
            actual = self.estimated_duration_minutes
        else:
            actual = 0

        if block is not None:
            day_of_week = block.date.weekday()
            scheduled_start = block.start_time
        else:
            day_of_week = timezone.localtime(completed_at).weekday()
            scheduled_start = None

        TaskHistory.objects.create(
            user=self.user,
            task_title=self.title,
            task_type=self.task_type,
            estimated_duration_minutes=self.estimated_duration_minutes,
            actual_duration_minutes=actual,
            scheduled_start_time=scheduled_start,
            day_of_week=day_of_week,
            completed_at=completed_at,
        )

    def mark_completed(self, actual_minutes=None):
        """
        Single source of truth for completing a Task.

        Reconciles all four completion representations:
          - Task.completed / Task.completed_at
          - ScheduledBlock.actually_completed (+ actual_duration_minutes)
          - a TaskHistory row

        Idempotent: repeated calls do not duplicate history nor overwrite an
        existing completed_at timestamp.
        """
        from django.utils import timezone

        updated_fields = []
        if not self.completed:
            self.completed = True
            updated_fields.append('completed')
        if self.completed_at is None:
            self.completed_at = timezone.now()
            updated_fields.append('completed_at')
        if updated_fields:
            self.save(update_fields=updated_fields)

        for block in self.scheduled_blocks.all():
            changed = []
            if not block.actually_completed:
                block.actually_completed = True
                changed.append('actually_completed')
            if actual_minutes is not None and block.actual_duration_minutes is None:
                block.actual_duration_minutes = actual_minutes
                changed.append('actual_duration_minutes')
            if changed:
                block.save(update_fields=changed)

        self._ensure_task_history(actual_minutes=actual_minutes)
        return self

    class Meta:
        verbose_name = "Tâche"
        verbose_name_plural = "Tâches"
        ordering = ['-priority', 'deadline', '-created_at']
        constraints = [
            models.CheckConstraint(
                condition=models.Q(priority__gte=1) & models.Q(priority__lte=10),
                name='task_priority_between_1_and_10',
            ),
        ]


class ScheduledBlock(models.Model):
    """Scheduled instance of a task."""

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='scheduled_blocks')
    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name='scheduled_blocks')
    date = models.DateField()
    start_time = models.TimeField()
    end_time = models.TimeField()
    actually_completed = models.BooleanField(default=False)
    actual_duration_minutes = models.PositiveIntegerField(null=True, blank=True)
    locked = models.BooleanField(
        default=False,
        help_text="Sanctuaire : la replanification ne déplace jamais ce bloc.",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.task.title} - {self.date} {self.start_time}"

    def mark_completed(self, actual_minutes=None):
        """
        Single source of truth for completing a ScheduledBlock.

        Sets actually_completed (+ optional actual_duration_minutes) and
        delegates to the parent Task so Task.completed / completed_at and a
        TaskHistory row stay reconciled. Idempotent.
        """
        changed = []
        if not self.actually_completed:
            self.actually_completed = True
            changed.append('actually_completed')
        if actual_minutes is not None and self.actual_duration_minutes is None:
            self.actual_duration_minutes = actual_minutes
            changed.append('actual_duration_minutes')
        if changed:
            self.save(update_fields=changed)

        self.task.mark_completed(actual_minutes=actual_minutes)
        return self

    class Meta:
        verbose_name = "Bloc planifié"
        verbose_name_plural = "Blocs planifiés"
        ordering = ['date', 'start_time']
        indexes = [
            models.Index(fields=['user', 'date'], name='schedblock_user_date_idx'),
        ]


class SchedulePlanChange(models.Model):
    """A reversible schedule adjustment (spec §8 automation modes).

    Stores the day's ScheduledBlocks BEFORE and the computed target AFTER, so a
    change can be: proposed (suggestion mode / important change in semi-auto),
    applied (automatic / small semi-auto), and undone. `before`/`after` are lists
    of {task_id, start_time, end_time, actually_completed, actual_duration_minutes}.
    """

    STATUS_CHOICES = [
        ('proposed', 'Proposé'),
        ('applied', 'Appliqué'),
        ('undone', 'Annulé'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='plan_changes')
    token = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    date = models.DateField()
    before = models.JSONField(default=list)
    after = models.JSONField(default=list)
    moved = models.JSONField(default=list, blank=True)
    unplaced = models.JSONField(default=list, blank=True)
    message = models.TextField(blank=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='proposed')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"PlanChange {self.token} [{self.status}] - {self.user.username}"

    class Meta:
        verbose_name = "Changement de planning"
        verbose_name_plural = "Changements de planning"
        ordering = ['-created_at']
        indexes = [models.Index(fields=['user', 'status'])]


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


class RecurringBlockException(models.Model):
    """Une occurrence d'un bloc récurrent ignorée/annulée pour UNE date précise.

    Marqueur NÉGATIF par date ("cette occurrence n'a pas lieu"), miroir de
    RecurringBlockCompletion (marqueur POSITIF "fait"). Modèle séparé exprès:
    les compteurs streak/progress/analytics comptent les complétions et ne
    doivent JAMAIS être pollués par des skips.

    La `date` est la date de DÉBUT de l'occurrence, c.-à-d. la date dont le jour
    de semaine == recurring_block.day_of_week (pour un bloc de nuit, c'est le
    soir où il commence, pas le matin où il finit).
    """

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='recurring_exceptions')
    recurring_block = models.ForeignKey(
        RecurringBlock,
        on_delete=models.CASCADE,
        related_name='exceptions'
    )
    date = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"SKIP {self.recurring_block.title} - {self.date}"

    class Meta:
        verbose_name = "Exception bloc récurrent"
        verbose_name_plural = "Exceptions blocs récurrents"
        unique_together = ['recurring_block', 'date']
        ordering = ['-date', '-created_at']
        indexes = [models.Index(fields=['user', 'date'])]


class TaskHistory(models.Model):
    """Historical data for completed tasks - used for AI predictions."""

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='task_history')
    task_title = models.CharField(max_length=200)
    task_type = models.CharField(max_length=20)
    estimated_duration_minutes = models.PositiveIntegerField(null=True, blank=True)
    actual_duration_minutes = models.PositiveIntegerField()
    scheduled_start_time = models.TimeField(null=True, blank=True)
    day_of_week = models.PositiveIntegerField()  # 0-6
    completed_at = models.DateTimeField()
    energy_level = models.CharField(max_length=10, blank=True)  # high, medium, low
    was_rescheduled = models.BooleanField(default=False)
    reschedule_count = models.PositiveIntegerField(default=0)

    class Meta:
        verbose_name = "Historique tâche"
        verbose_name_plural = "Historique tâches"
        ordering = ['-completed_at']
        indexes = [
            models.Index(fields=['user', 'task_type']),
            models.Index(fields=['user', 'task_title']),
        ]

    def __str__(self):
        return f"{self.task_title} - {self.actual_duration_minutes}min"


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
        indexes = [
            models.Index(fields=['user', 'created_at'], name='convmsg_user_created_idx'),
        ]


class Goal(models.Model):
    """User goal (short or long term)."""

    GOAL_TYPE_CHOICES = [
        ('short_term', 'Court terme'),
        ('long_term', 'Long terme'),
    ]

    STATUS_CHOICES = [
        ('active', 'Actif'),
        ('completed', 'Terminé'),
        ('paused', 'En pause'),
        ('cancelled', 'Annulé'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='goals')
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    goal_type = models.CharField(max_length=20, choices=GOAL_TYPE_CHOICES, default='short_term')
    deadline = models.DateField(null=True, blank=True)
    progress = models.PositiveIntegerField(default=0)  # 0-100
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.title} ({self.get_status_display()})"

    def save(self, *args, **kwargs):
        # Clamp progress between 0 and 100
        if self.progress is None or self.progress < 0:
            self.progress = 0
        elif self.progress > 100:
            self.progress = 100
        super().save(*args, **kwargs)

    class Meta:
        verbose_name = "Objectif"
        verbose_name_plural = "Objectifs"
        ordering = ['-created_at']
        constraints = [
            models.CheckConstraint(
                condition=models.Q(progress__gte=0) & models.Q(progress__lte=100),
                name='goal_progress_between_0_and_100',
            ),
        ]


class SharedSchedule(models.Model):
    """Shareable link for a user's schedule."""

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='shared_schedules')
    share_token = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    title = models.CharField(max_length=100, blank=True, default='Mon planning')
    is_active = models.BooleanField(default=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    include_tasks = models.BooleanField(default=False)  # Include scheduled tasks or just recurring
    created_at = models.DateTimeField(auto_now_add=True)
    view_count = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"Partage {self.share_token} - {self.user.username}"

    @property
    def share_url(self):
        return f"/shared/{self.share_token}"

    def is_valid(self):
        """Check if the share link is still valid."""
        if not self.is_active:
            return False
        if self.expires_at:
            from django.utils import timezone
            return timezone.now() < self.expires_at
        return True

    class Meta:
        verbose_name = "Planning partagé"
        verbose_name_plural = "Plannings partagés"
        ordering = ['-created_at']


class Connection(models.Model):
    """A connection (friend link) between two users.

    Planner's structural superpower: it knows the REAL free slots of several
    users, so it can propose co-presence / body-doubling / study groups on the
    actual common availability (fin de cours, retour maison). A directed request
    (from_user -> to_user) that becomes mutual once accepted.
    """

    STATUS_CHOICES = [
        ('pending', 'En attente'),
        ('accepted', 'Accepté'),
    ]

    from_user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sent_connections')
    to_user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='received_connections')
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    accepted_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.from_user.username} -> {self.to_user.username} ({self.status})"

    class Meta:
        verbose_name = "Connexion"
        verbose_name_plural = "Connexions"
        constraints = [
            models.UniqueConstraint(fields=['from_user', 'to_user'], name='connection_unique_pair'),
        ]
        indexes = [models.Index(fields=['to_user', 'status'])]


class CalendarFeed(models.Model):
    """A stable, token-secured iCal (.ics) subscription URL for a user.

    Unlike SharedSchedule (a one-off HTML share), this is a long-lived feed the
    user pastes into Google/Apple/Outlook Calendar once; those apps re-fetch it
    on their own cadence, so Planner blocks stay synced. Rotating the token
    (regenerate) instantly revokes every existing subscription.
    """
    user = models.OneToOneField(
        User, on_delete=models.CASCADE, related_name='calendar_feed'
    )
    token = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    include_tasks = models.BooleanField(
        default=True,
        help_text="Inclure les tâches planifiées et les échéances, pas seulement les blocs récurrents.",
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_accessed_at = models.DateTimeField(null=True, blank=True)
    access_count = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"CalendarFeed {self.token} - {self.user.username}"

    @property
    def feed_path(self):
        return f"/api/calendar/{self.token}.ics"

    class Meta:
        verbose_name = "Flux calendrier"
        verbose_name_plural = "Flux calendrier"


class WebhookEndpoint(models.Model):
    """A user-registered outbound webhook (n8n, Zapier, Make, custom).

    When Planner events fire (task created/completed, block completed, reminder),
    a signed JSON POST is sent to `url`. This is the generic automation bridge:
    the user wires Planner into any no-code/low-code flow without Planner needing
    a per-integration connector.
    """
    EVENT_CHOICES = [
        ('task.created', 'Tâche créée'),
        ('task.completed', 'Tâche complétée'),
        ('block.completed', 'Bloc récurrent complété'),
        ('schedule.generated', 'Horaire généré'),
        ('reminder.sent', 'Rappel envoyé'),
    ]

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='webhook_endpoints'
    )
    url = models.URLField(max_length=1000)
    secret = models.CharField(
        max_length=64, blank=True,
        help_text="Signe chaque payload en HMAC-SHA256 (header X-Planner-Signature).",
    )
    events = models.JSONField(
        default=list, blank=True,
        help_text="Liste d'événements souscrits; vide = tous.",
    )
    description = models.CharField(max_length=200, blank=True)
    active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_triggered_at = models.DateTimeField(null=True, blank=True)
    last_status = models.CharField(max_length=40, blank=True)
    failure_count = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"Webhook {self.user.username} -> {self.url[:50]}"

    def wants(self, event: str) -> bool:
        """True if this endpoint should receive `event` (empty list = all)."""
        return self.active and (not self.events or event in self.events)

    class Meta:
        verbose_name = "Webhook"
        verbose_name_plural = "Webhooks"
        ordering = ['-created_at']
        indexes = [models.Index(fields=['user', 'active'])]


class PushSubscription(models.Model):
    """A Web Push (VAPID) subscription for a user's browser/PWA/device.

    Used by the Planner PWA and by downstream apps (e.g. the tenant-management
    app) to receive push notifications. One user can have several (multi-device).
    """
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='push_subscriptions'
    )
    endpoint = models.URLField(max_length=1000, unique=True)
    p256dh = models.CharField(max_length=255)  # client public key
    auth = models.CharField(max_length=255)    # client auth secret
    user_agent = models.CharField(max_length=300, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [models.Index(fields=['user'])]

    def __str__(self):
        return f"PushSub {self.user.username} {self.endpoint[:40]}"
