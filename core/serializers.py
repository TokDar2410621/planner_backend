"""
Serializers for Planner AI backend.
"""
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.utils import timezone
from rest_framework import serializers

from .models import (
    UserProfile,
    UploadedDocument,
    RecurringBlock,
    RecurringBlockCompletion,
    Task,
    ScheduledBlock,
    ConversationMessage,
    SharedSchedule,
)


class UserProfileSerializer(serializers.ModelSerializer):
    """Serializer for UserProfile model."""

    class Meta:
        model = UserProfile
        fields = [
            'avatar_url',
            'min_sleep_hours',
            'post_night_shift_wake_time',
            'peak_productivity_time',
            'max_deep_work_hours_per_day',
            'transport_time_minutes',
            'onboarding_completed',
            'onboarding_step',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['created_at', 'updated_at']


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


class RecurringBlockSerializer(serializers.ModelSerializer):
    """Serializer for RecurringBlock model."""

    day_of_week_display = serializers.CharField(source='get_day_of_week_display', read_only=True)
    block_type_display = serializers.CharField(source='get_block_type_display', read_only=True)

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
            'is_night_shift',
            'source_document',
            'active',
            'created_at',
        ]
        read_only_fields = ['id', 'created_at', 'day_of_week_display', 'block_type_display']

    def validate(self, attrs):
        start_time = attrs.get('start_time')
        end_time = attrs.get('end_time')

        if start_time and end_time:
            # Allow overnight blocks for night shifts
            if not attrs.get('is_night_shift') and start_time >= end_time:
                raise serializers.ValidationError({
                    'end_time': "L'heure de fin doit être après l'heure de début."
                })
        return attrs


class RecurringBlockCompletionSerializer(serializers.ModelSerializer):
    """Serializer for RecurringBlockCompletion model."""

    class Meta:
        model = RecurringBlockCompletion
        fields = ['id', 'recurring_block', 'date', 'completed_at']
        read_only_fields = ['id', 'completed_at']


class TaskSerializer(serializers.ModelSerializer):
    """Serializer for Task model."""

    task_type_display = serializers.CharField(source='get_task_type_display', read_only=True)

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
            'completed',
            'completed_at',
            'created_at',
        ]
        read_only_fields = ['id', 'completed_at', 'created_at', 'task_type_display']

    def validate_priority(self, value):
        if value < 1 or value > 10:
            raise serializers.ValidationError("La priorité doit être entre 1 et 10.")
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
