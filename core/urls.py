"""
URL configuration for core app.
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import (
    # Auth
    RegisterView,
    LoginView,
    MeView,
    # Profile
    ProfileView,
    OnboardingStatusView,
    # Chat
    ChatView,
    # ViewSets
    DocumentViewSet,
    RecurringBlockViewSet,
    RecurringBlockCompletionViewSet,
    TaskViewSet,
    # Schedule
    ScheduleView,
    ScheduleGenerateView,
    ScheduledBlockView,
    # Conversation
    ConversationView,
)

router = DefaultRouter()
router.register(r'documents', DocumentViewSet, basename='document')
router.register(r'recurring-blocks', RecurringBlockViewSet, basename='recurring-block')
router.register(r'recurring-completions', RecurringBlockCompletionViewSet, basename='recurring-completion')
router.register(r'tasks', TaskViewSet, basename='task')

urlpatterns = [
    # Auth endpoints
    path('auth/register/', RegisterView.as_view(), name='register'),
    path('auth/login/', LoginView.as_view(), name='login'),
    path('auth/me/', MeView.as_view(), name='me'),

    # Profile endpoints
    path('profile/', ProfileView.as_view(), name='profile'),
    path('profile/onboarding-status/', OnboardingStatusView.as_view(), name='onboarding-status'),

    # Chat endpoint
    path('chat/', ChatView.as_view(), name='chat'),

    # Schedule endpoints
    path('schedule/', ScheduleView.as_view(), name='schedule'),
    path('schedule/generate/', ScheduleGenerateView.as_view(), name='schedule-generate'),
    path('schedule/<int:block_id>/', ScheduledBlockView.as_view(), name='schedule-block'),

    # Conversation history
    path('conversations/', ConversationView.as_view(), name='conversations'),

    # Router URLs (documents, recurring-blocks, tasks)
    path('', include(router.urls)),
]
