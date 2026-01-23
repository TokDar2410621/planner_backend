"""
Admin configuration for Planner AI models.
"""
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User

from .models import (
    UserProfile,
    UploadedDocument,
    RecurringBlock,
    Task,
    ScheduledBlock,
    ConversationMessage,
)


class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'Profil'


class UserAdmin(BaseUserAdmin):
    inlines = (UserProfileInline,)


# Re-register UserAdmin
admin.site.unregister(User)
admin.site.register(User, UserAdmin)


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'peak_productivity_time', 'onboarding_completed', 'created_at']
    list_filter = ['peak_productivity_time', 'onboarding_completed']
    search_fields = ['user__username', 'user__email']


@admin.register(UploadedDocument)
class UploadedDocumentAdmin(admin.ModelAdmin):
    list_display = ['user', 'document_type', 'processed', 'uploaded_at']
    list_filter = ['document_type', 'processed']
    search_fields = ['user__username']
    readonly_fields = ['extracted_data', 'uploaded_at']


@admin.register(RecurringBlock)
class RecurringBlockAdmin(admin.ModelAdmin):
    list_display = ['title', 'user', 'block_type', 'day_of_week', 'start_time', 'end_time', 'active']
    list_filter = ['block_type', 'day_of_week', 'active', 'is_night_shift']
    search_fields = ['title', 'user__username']


@admin.register(Task)
class TaskAdmin(admin.ModelAdmin):
    list_display = ['title', 'user', 'task_type', 'priority', 'deadline', 'completed']
    list_filter = ['task_type', 'completed', 'priority']
    search_fields = ['title', 'user__username', 'related_course']


@admin.register(ScheduledBlock)
class ScheduledBlockAdmin(admin.ModelAdmin):
    list_display = ['task', 'user', 'date', 'start_time', 'end_time', 'actually_completed']
    list_filter = ['date', 'actually_completed']
    search_fields = ['task__title', 'user__username']


@admin.register(ConversationMessage)
class ConversationMessageAdmin(admin.ModelAdmin):
    list_display = ['user', 'role', 'short_content', 'created_at']
    list_filter = ['role', 'created_at']
    search_fields = ['user__username', 'content']

    def short_content(self, obj):
        return obj.content[:100] + "..." if len(obj.content) > 100 else obj.content
    short_content.short_description = 'Contenu'
