"""
URL configuration for core app.
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import (
    # Auth
    RegisterView,
    LoginView,
    CheckEmailView,
    MeView,
    GoogleAuthView,
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
    # AI Insights
    SuggestionsView,
    PatternsView,
    PredictDurationView,
    ConflictsView,
    SmartRescheduleView,
    NaturalLanguageScheduleView,
    # Share
    ShareScheduleView,
    ShareScheduleDetailView,
    PublicScheduleView,
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
    path('auth/check-email/', CheckEmailView.as_view(), name='check-email'),
    path('auth/google/', GoogleAuthView.as_view(), name='google-auth'),
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

    # AI Insights endpoints
    path('insights/suggestions/', SuggestionsView.as_view(), name='suggestions'),
    path('insights/patterns/', PatternsView.as_view(), name='patterns'),
    path('insights/predict-duration/', PredictDurationView.as_view(), name='predict-duration'),
    path('insights/conflicts/', ConflictsView.as_view(), name='conflicts'),
    path('insights/smart-reschedule/', SmartRescheduleView.as_view(), name='smart-reschedule'),
    path('insights/natural-language/', NaturalLanguageScheduleView.as_view(), name='natural-language'),

    # Share endpoints
    path('schedule/share/', ShareScheduleView.as_view(), name='share-schedule'),
    path('schedule/share/<int:share_id>/', ShareScheduleDetailView.as_view(), name='share-schedule-detail'),
    path('shared/<uuid:share_token>/', PublicScheduleView.as_view(), name='public-schedule'),

    # Router URLs (documents, recurring-blocks, tasks)
    path('', include(router.urls)),
]
