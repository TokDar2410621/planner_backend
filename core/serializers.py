"""
Serializers for Planner AI backend.
"""
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.utils import timezone
from rest_framework import serializers

from .models import (
    UserProfile,
    UserPlace,
    UploadedDocument,
    RecurringBlock,
    RecurringBlockCompletion,
    RecurringBlockException,
    Task,
    ScheduledBlock,
    ConversationMessage,
    SharedSchedule,
)


class UserProfileSerializer(serializers.ModelSerializer):
    """Serializer for UserProfile model.

    B23: exposes the ``energy_levels`` and ``notification_preferences``
    JSONFields (added by T1) as read/write fields so the frontend
    ``PATCH /profile/`` actually persists them. They are declared explicitly
    (rather than relying on ``Meta.fields`` auto-mapping) and dropped in
    ``get_fields`` when the underlying model does not yet carry them, so the
    serializer stays usable against an older schema without raising.
    """

    energy_levels = serializers.JSONField(required=False)
    notification_preferences = serializers.JSONField(required=False)

    class Meta:
        model = UserProfile
        fields = [
            'avatar_url',
            'preferred_llm',
            'min_sleep_hours',
            'post_night_shift_wake_time',
            'peak_productivity_time',
            'max_deep_work_hours_per_day',
            'transport_time_minutes',
            'prep_time_minutes',
            'safety_margin_minutes',
            'automation_mode',
            'auto_apply_threshold_minutes',
            'energy_levels',
            'notification_preferences',
            'onboarding_completed',
            'onboarding_step',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['created_at', 'updated_at']

    def get_fields(self):
        fields = super().get_fields()
        model_field_names = {f.name for f in UserProfile._meta.get_fields()}
        for name in ('energy_levels', 'notification_preferences'):
            if name not in model_field_names:
                fields.pop(name, None)
        return fields


class UserSerializer(serializers.ModelSerializer):
    """Serializer for User model with profile."""

    profile = UserProfileSerializer(read_only=True)

    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'date_joined', 'profile']
        read_only_fields = ['id', 'date_joined']


class UserRegistrationSerializer(serializers.ModelSerializer):
    """Serializer for user registration."""

    password = serializers.CharField(write_only=True, validators=[validate_password])
    password_confirm = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'password_confirm', 'first_name', 'last_name']

    def validate_email(self, value):
        # Django's User has no unique email; enforce it here so a second account
        # can't be created on the same address (root cause of the Google-login
        # 409 "plusieurs comptes partagent cet email"). Case-insensitive.
        if value and User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError(
                "Un compte existe déjà avec cet email. Connectez-vous."
            )
        return value

    def validate(self, attrs):
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError({
                'password_confirm': "Les mots de passe ne correspondent pas."
            })
        return attrs

    def create(self, validated_data):
        validated_data.pop('password_confirm')
        user = User.objects.create_user(**validated_data)
        return user


class UploadedDocumentSerializer(serializers.ModelSerializer):
    """Serializer for UploadedDocument model."""

    # Client-facing aliases for document_type. The /chat/ upload path is lenient
    # (it stores the raw value and the processor normalizes it afterwards), so a
    # client sending document_type="schedule" works there but used to 400 here.
    # Normalizing before ChoiceField validation gives the /documents/ upload path
    # the same contract. Unknown values still fail validation as before.
    _DOC_TYPE_ALIASES = {
        'schedule': 'course_schedule',
        'course': 'course_schedule',
        'cours': 'course_schedule',
        'course_schedule': 'course_schedule',
        'horaire': 'course_schedule',
        'work': 'work_schedule',
        'travail': 'work_schedule',
        'work_schedule': 'work_schedule',
        'other': 'other',
        'autre': 'other',
    }

    def to_internal_value(self, data):
        # Multipart payloads arrive as an immutable QueryDict; copy before edit.
        if hasattr(data, 'copy'):
            data = data.copy()
        raw = data.get('document_type')
        if raw:
            data['document_type'] = self._DOC_TYPE_ALIASES.get(
                str(raw).strip().lower(), raw
            )
        return super().to_internal_value(data)

    class Meta:
        model = UploadedDocument
        fields = [
            'id',
            'file',
            'document_type',
            'extracted_data',
            'processed',
            'processing_error',
            'uploaded_at',
        ]
        read_only_fields = ['id', 'extracted_data', 'processed', 'processing_error', 'uploaded_at']


class UserPlaceSerializer(serializers.ModelSerializer):
    """Serializer for UserPlace (travel-time engine + geo coordinates)."""

    class Meta:
        model = UserPlace
        fields = ['id', 'name', 'kind', 'address', 'travel_minutes',
                  'latitude', 'longitude', 'created_at']
        # lat/lng are populated server-side by geocoding, never client-supplied.
        read_only_fields = ['id', 'latitude', 'longitude', 'created_at']


class RecurringBlockSerializer(serializers.ModelSerializer):
    """Serializer for RecurringBlock model."""

    day_of_week_display = serializers.CharField(source='get_day_of_week_display', read_only=True)
    block_type_display = serializers.CharField(source='get_block_type_display', read_only=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Scope `place` choices to the requesting user's own places (no IDOR).
        request = self.context.get('request')
        if request is not None and 'place' in self.fields:
            user = getattr(request, 'user', None)
            if user is not None and user.is_authenticated:
                self.fields['place'].queryset = UserPlace.objects.filter(user=user)

    class Meta:
        model = RecurringBlock
        fields = [
            'id',
            'title',
            'block_type',
            'block_type_display',
            'day_of_week',
            'day_of_week_display',
            'start_time',
            'end_time',
            'location',
            'place',
            'is_night_shift',
            'source_document',
            'active',
            'status',
            'confidence',
            'created_at',
        ]
        read_only_fields = [
            'id', 'created_at', 'day_of_week_display', 'block_type_display',
            'status', 'confidence',
        ]

    def validate(self, attrs):
        start_time = attrs.get('start_time') or (
            self.instance.start_time if self.instance is not None else None
        )
        end_time = attrs.get('end_time') or (
            self.instance.end_time if self.instance is not None else None
        )
        is_night_shift = attrs.get('is_night_shift')
        if is_night_shift is None and self.instance is not None:
            is_night_shift = self.instance.is_night_shift
        block_type = attrs.get('block_type') or (
            self.instance.block_type if self.instance is not None else None
        )
        day_of_week = attrs.get('day_of_week')
        if day_of_week is None and self.instance is not None:
            day_of_week = self.instance.day_of_week

        if start_time and end_time:
            # Allow overnight blocks for night shifts
            if not is_night_shift and start_time >= end_time:
                raise serializers.ValidationError({
                    'end_time': "L'heure de fin doit être après l'heure de début."
                })

        request = self.context.get('request')
        user = getattr(request, 'user', None) if request is not None else None

        # Ne contrôle le chevauchement QUE si un champ pertinent pour l'horaire
        # change (ou à la création). Un PATCH de métadonnées seules (title, active,
        # location, place) ne doit JAMAIS être bloqué par un chevauchement
        # préexistant: des blocs qui se chevauchent existent en base (l'extraction
        # de documents et les outils agent créent via l'ORM, hors ce serializer),
        # et l'utilisateur doit pouvoir les renommer/désactiver pour les résoudre.
        target_active = attrs.get('active')
        if target_active is None:
            target_active = self.instance.active if self.instance is not None else True

        active_changed_to_true = (
            self.instance is not None
            and attrs.get('active') is True
            and not self.instance.active
        )
        scheduling_fields = {'start_time', 'end_time', 'day_of_week', 'block_type', 'is_night_shift'}
        is_create = self.instance is None
        scheduling_changed = (
            is_create
            or active_changed_to_true
            or bool(scheduling_fields & set(attrs))
        )

        if (
            user is not None
            and user.is_authenticated
            and target_active
            and scheduling_changed
            and start_time
            and end_time
            and day_of_week is not None
            and block_type
        ):
            from services.scheduling.overlap import find_recurring_conflicts

            # Flexibilité RÉSOLUE: si le type change (ou création), suit le défaut
            # du NOUVEAU type; sinon respecte l'instance (override éventuel). Évite
            # le trou (sport->work chevauchant qui passait) et la sur-restriction
            # (work->sport chevauchant qui était refusé alors que le souple cède).
            if is_create or 'block_type' in attrs:
                is_flexible = (
                    RecurringBlock.default_flexibility_for(block_type)
                    == RecurringBlock.FLEXIBILITY_FLEXIBLE
                )
            else:
                is_flexible = self.instance.is_flexible
            conflicts = find_recurring_conflicts(
                user,
                day_of_week,
                start_time,
                end_time,
                bool(is_night_shift),
                exclude_id=getattr(self.instance, 'id', None),
                new_is_flexible=is_flexible,
            )
            if conflicts:
                overlap = conflicts[0]
                raise serializers.ValidationError({
                    'non_field_errors': [
                        (
                            f"Chevauchement avec '{overlap.title}' "
                            f"({overlap.start_time.strftime('%H:%M')}-"
                            f"{overlap.end_time.strftime('%H:%M')})."
                        )
                    ]
                })
        return attrs


class RecurringBlockCompletionSerializer(serializers.ModelSerializer):
    """Serializer for RecurringBlockCompletion model."""

    class Meta:
        model = RecurringBlockCompletion
        fields = ['id', 'recurring_block', 'date', 'completed_at']
        read_only_fields = ['id', 'completed_at']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # S7: scope the recurring_block choices to blocks owned by the
        # requesting user so a user cannot create a completion that
        # references another user's block (IDOR).
        request = self.context.get('request')
        if request is not None and 'recurring_block' in self.fields:
            user = getattr(request, 'user', None)
            if user is not None and user.is_authenticated:
                self.fields['recurring_block'].queryset = (
                    RecurringBlock.objects.filter(user=user)
                )

    def validate_recurring_block(self, value):
        # Defense in depth: even if the field queryset was not scoped
        # (e.g. serializer used without request context), reject blocks
        # that belong to another user.
        request = self.context.get('request')
        if request is not None:
            user = getattr(request, 'user', None)
            if user is not None and user.is_authenticated and value.user_id != user.id:
                raise serializers.ValidationError(
                    "Ce bloc récurrent ne vous appartient pas."
                )
        return value


class RecurringBlockExceptionSerializer(serializers.ModelSerializer):
    """Serializer for RecurringBlockException (a skipped/cancelled occurrence)."""

    class Meta:
        model = RecurringBlockException
        fields = ['id', 'recurring_block', 'date', 'created_at']
        read_only_fields = ['id', 'created_at']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # S7: scope recurring_block choices to the requesting user's blocks (IDOR).
        request = self.context.get('request')
        if request is not None and 'recurring_block' in self.fields:
            user = getattr(request, 'user', None)
            if user is not None and user.is_authenticated:
                self.fields['recurring_block'].queryset = (
                    RecurringBlock.objects.filter(user=user)
                )

    def validate_recurring_block(self, value):
        request = self.context.get('request')
        if request is not None:
            user = getattr(request, 'user', None)
            if user is not None and user.is_authenticated and value.user_id != user.id:
                raise serializers.ValidationError(
                    "Ce bloc récurrent ne vous appartient pas."
                )
        return value


class TaskSerializer(serializers.ModelSerializer):
    """Serializer for Task model."""

    task_type_display = serializers.CharField(source='get_task_type_display', read_only=True)
    # Nested read-only venue so the frontend can show distance / departure for a
    # located task (ex: "Réunion 14h à l'UQAC"). `place` (write) sets the FK.
    place_detail = UserPlaceSerializer(source='place', read_only=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Scope `place` choices to the requesting user's own places (no IDOR).
        request = self.context.get('request')
        if request is not None and 'place' in self.fields:
            user = getattr(request, 'user', None)
            if user is not None and user.is_authenticated:
                self.fields['place'].queryset = UserPlace.objects.filter(user=user)

    class Meta:
        model = Task
        fields = [
            'id',
            'title',
            'description',
            'deadline',
            'estimated_duration_minutes',
            'task_type',
            'task_type_display',
            'priority',
            'related_course',
            'place',
            'place_detail',
            'completed',
            'completed_at',
            'created_at',
        ]
        read_only_fields = ['id', 'completed_at', 'created_at', 'task_type_display', 'place_detail']

    def validate_priority(self, value):
        if value < 1 or value > 10:
            raise serializers.ValidationError("La priorité doit être entre 1 et 10.")
        return value

    def validate_place(self, value):
        # Defense in depth against IDOR even without request-scoped queryset.
        request = self.context.get('request')
        if value is not None and request is not None:
            user = getattr(request, 'user', None)
            if user is not None and user.is_authenticated and value.user_id != user.id:
                raise serializers.ValidationError("Ce lieu ne vous appartient pas.")
        return value


class TaskCompleteSerializer(serializers.Serializer):
    """Serializer for completing a task."""

    actual_duration_minutes = serializers.IntegerField(required=False, min_value=1)


class ScheduledBlockSerializer(serializers.ModelSerializer):
    """Serializer for ScheduledBlock model."""

    task_title = serializers.CharField(source='task.title', read_only=True)
    task = TaskSerializer(read_only=True)

    class Meta:
        model = ScheduledBlock
        fields = [
            'id',
            'task',
            'task_title',
            'date',
            'start_time',
            'end_time',
            'actually_completed',
            'actual_duration_minutes',
            'locked',
            'created_at',
        ]
        read_only_fields = ['id', 'task', 'task_title', 'created_at']

    def validate(self, attrs):
        start_time = attrs.get('start_time')
        end_time = attrs.get('end_time')

        if start_time and end_time and start_time >= end_time:
            raise serializers.ValidationError({
                'end_time': "L'heure de fin doit être après l'heure de début."
            })
        return attrs

    def update(self, instance, validated_data):
        """Auto-update end_time when start_time changes (for drag & drop)."""
        from datetime import datetime, timedelta

        new_start_time = validated_data.get('start_time')
        new_end_time = validated_data.get('end_time')

        # If start_time is being changed but end_time is not provided
        if new_start_time and not new_end_time:
            # Calculate the original duration
            old_start = datetime.combine(datetime.today(), instance.start_time)
            old_end = datetime.combine(datetime.today(), instance.end_time)
            duration = old_end - old_start

            # If duration is invalid (negative or zero), use task's estimated duration
            if duration.total_seconds() <= 0:
                task_duration = instance.task.estimated_duration_minutes or 60
                duration = timedelta(minutes=task_duration)

            # Apply the duration to the new start time
            new_start = datetime.combine(datetime.today(), new_start_time)
            new_end = new_start + duration
            validated_data['end_time'] = new_end.time()

        return super().update(instance, validated_data)


class ConversationMessageSerializer(serializers.ModelSerializer):
    """Serializer for ConversationMessage model."""

    class Meta:
        model = ConversationMessage
        fields = ['id', 'role', 'content', 'attachment', 'metadata', 'created_at']
        read_only_fields = ['id', 'role', 'created_at']


class ChatInputSerializer(serializers.Serializer):
    """Serializer for chat input."""

    message = serializers.CharField(max_length=5000)


class ChatResponseSerializer(serializers.Serializer):
    """Serializer for chat response."""

    response = serializers.CharField()
    extracted_data = serializers.DictField(required=False)
    tasks_created = TaskSerializer(many=True, required=False)


class ScheduleGenerateSerializer(serializers.Serializer):
    """Serializer for schedule generation request."""

    start_date = serializers.DateField(required=False)
    force = serializers.BooleanField(default=False)


class ScheduleResponseSerializer(serializers.Serializer):
    """Serializer for schedule response."""

    recurring_blocks = RecurringBlockSerializer(many=True)
    scheduled_tasks = ScheduledBlockSerializer(many=True)
    unscheduled_tasks = TaskSerializer(many=True)


class OnboardingStatusSerializer(serializers.Serializer):
    """Serializer for onboarding status."""

    completed = serializers.BooleanField()
    current_step = serializers.IntegerField()
    total_steps = serializers.IntegerField()
    next_action = serializers.CharField()


class SharedScheduleSerializer(serializers.ModelSerializer):
    """Serializer for SharedSchedule model."""

    share_url = serializers.CharField(read_only=True)
    is_valid = serializers.SerializerMethodField()

    class Meta:
        model = SharedSchedule
        fields = [
            'id',
            'share_token',
            'title',
            'is_active',
            'expires_at',
            'include_tasks',
            'created_at',
            'view_count',
            'share_url',
            'is_valid',
        ]
        read_only_fields = ['id', 'share_token', 'created_at', 'view_count', 'share_url', 'is_valid']

    def get_is_valid(self, obj):
        return obj.is_valid()


class CreateShareSerializer(serializers.Serializer):
    """Serializer for creating a share link."""

    title = serializers.CharField(max_length=100, required=False, default='Mon planning')
    expires_in_days = serializers.IntegerField(required=False, min_value=1, max_value=365)
    include_tasks = serializers.BooleanField(default=False)
