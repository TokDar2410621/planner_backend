"""
API Views for Planner AI backend.
"""
import logging
from datetime import date, timedelta

from django.contrib.auth.models import User
from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken

from .models import (
    UserProfile,
    UploadedDocument,
    RecurringBlock,
    RecurringBlockCompletion,
    Task,
    ScheduledBlock,
    ConversationMessage,
    TaskHistory,
    SharedSchedule,
)
from .serializers import (
    UserSerializer,
    UserRegistrationSerializer,
    UserProfileSerializer,
    UploadedDocumentSerializer,
    RecurringBlockSerializer,
    RecurringBlockCompletionSerializer,
    TaskSerializer,
    TaskCompleteSerializer,
    ScheduledBlockSerializer,
    ConversationMessageSerializer,
    ChatInputSerializer,
    ChatResponseSerializer,
    ScheduleGenerateSerializer,
    ScheduleResponseSerializer,
    OnboardingStatusSerializer,
    SharedScheduleSerializer,
    CreateShareSerializer,
)
from services import DocumentProcessor, ChatEngine, AIScheduler
from services.ai_insights import AIInsightsService

logger = logging.getLogger(__name__)


class HealthCheckView(APIView):
    """Health check endpoint."""

    permission_classes = [AllowAny]

    def get(self, request):
        """Return API health status."""
        return Response({
            'status': 'healthy',
            'timestamp': timezone.now().isoformat(),
        })


# ============== Auth Views ==============

class RegisterView(APIView):
    """User registration endpoint."""

    permission_classes = [AllowAny]

    def post(self, request):
        """Register a new user."""
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            refresh = RefreshToken.for_user(user)
            return Response({
                'user': UserSerializer(user).data,
                'tokens': {
                    'refresh': str(refresh),
                    'access': str(refresh.access_token),
                }
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(APIView):
    """User login endpoint."""

    permission_classes = [AllowAny]

    def post(self, request):
        """Login and return JWT tokens."""
        username = request.data.get('username')
        password = request.data.get('password')

        if not username or not password:
            return Response(
                {'error': 'Username et password requis.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return Response(
                {'error': 'Identifiants invalides.'},
                status=status.HTTP_401_UNAUTHORIZED
            )

        if not user.check_password(password):
            return Response(
                {'error': 'Identifiants invalides.'},
                status=status.HTTP_401_UNAUTHORIZED
            )

        refresh = RefreshToken.for_user(user)
        return Response({
            'user': UserSerializer(user).data,
            'tokens': {
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }
        })


class MeView(APIView):
    """Get current user information."""

    def get(self, request):
        """Return current user data."""
        return Response(UserSerializer(request.user).data)


class GoogleAuthView(APIView):
    """Google OAuth2 authentication endpoint."""

    permission_classes = [AllowAny]

    def post(self, request):
        """Verify Google ID token and login/register user."""
        from django.conf import settings
        import requests

        credential = request.data.get('credential')
        if not credential:
            return Response(
                {'error': 'Google credential requis.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Verify the Google ID token
        try:
            # Verify token with Google
            google_response = requests.get(
                f'https://oauth2.googleapis.com/tokeninfo?id_token={credential}'
            )

            if google_response.status_code != 200:
                return Response(
                    {'error': 'Token Google invalide.'},
                    status=status.HTTP_401_UNAUTHORIZED
                )

            google_data = google_response.json()

            # Verify the token is for our app
            if google_data.get('aud') != settings.GOOGLE_CLIENT_ID:
                return Response(
                    {'error': 'Token non autorise pour cette application.'},
                    status=status.HTTP_401_UNAUTHORIZED
                )

            email = google_data.get('email')
            if not email:
                return Response(
                    {'error': 'Email non fourni par Google.'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Get or create user
            user, created = User.objects.get_or_create(
                email=email,
                defaults={
                    'username': email.split('@')[0],
                    'first_name': google_data.get('given_name', ''),
                    'last_name': google_data.get('family_name', ''),
                }
            )

            # Handle username conflict for new users
            if created:
                # Make username unique if needed
                base_username = email.split('@')[0]
                username = base_username
                counter = 1
                while User.objects.filter(username=username).exclude(id=user.id).exists():
                    username = f"{base_username}{counter}"
                    counter += 1
                user.username = username
                user.save()

            # Update profile with Google data (avatar, name)
            profile = user.profile
            picture_url = google_data.get('picture')
            if picture_url:
                profile.avatar_url = picture_url
                profile.save()

            # Update user name if not set
            if not user.first_name and google_data.get('given_name'):
                user.first_name = google_data.get('given_name', '')
                user.last_name = google_data.get('family_name', '')
                user.save()

            # Generate JWT tokens
            refresh = RefreshToken.for_user(user)
            return Response({
                'user': UserSerializer(user).data,
                'tokens': {
                    'refresh': str(refresh),
                    'access': str(refresh.access_token),
                },
                'created': created,
            })

        except Exception as e:
            logger.error(f"Google auth error: {e}")
            return Response(
                {'error': 'Erreur lors de l\'authentification Google.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# ============== Profile Views ==============

class ProfileView(APIView):
    """User profile management."""

    def get(self, request):
        """Get user profile."""
        return Response(UserProfileSerializer(request.user.profile).data)

    def patch(self, request):
        """Update user profile."""
        serializer = UserProfileSerializer(
            request.user.profile,
            data=request.data,
            partial=True
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class OnboardingStatusView(APIView):
    """Get onboarding status."""

    def get(self, request):
        """Return onboarding status."""
        profile = request.user.profile
        total_steps = 3

        if profile.onboarding_completed:
            next_action = "completed"
        elif profile.onboarding_step == 0:
            next_action = "upload_schedule"
        elif profile.onboarding_step == 1:
            next_action = "confirm_schedule"
        else:
            next_action = "set_preferences"

        data = {
            'completed': profile.onboarding_completed,
            'current_step': profile.onboarding_step,
            'total_steps': total_steps,
            'next_action': next_action,
        }
        return Response(OnboardingStatusSerializer(data).data)


# ============== Chat View ==============

class ChatView(APIView):
    """Chat endpoint for conversational AI."""

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        """Send a message and get AI response."""
        message = request.data.get('message', '')
        attachment_file = request.FILES.get('attachment')

        if not message and not attachment_file:
            return Response(
                {'error': 'Message ou fichier requis.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Handle file upload
        attachment = None
        if attachment_file:
            doc_type = request.data.get('document_type', 'other')
            doc = UploadedDocument.objects.create(
                user=request.user,
                file=attachment_file,
                document_type=doc_type,
            )
            attachment = doc

            # Process document
            try:
                processor = DocumentProcessor()
                processor.process_document(doc)
            except Exception as e:
                logger.error(f"Document processing error: {e}")

        # Generate chat response
        engine = ChatEngine()
        result = engine.generate_response(request.user, message or "J'ai uploadé un document.", attachment)

        response_data = {
            'response': result['response'],
        }
        if result.get('extracted_data'):
            response_data['extracted_data'] = result['extracted_data']
        if result.get('tasks_created'):
            response_data['tasks_created'] = TaskSerializer(result['tasks_created'], many=True).data
        if result.get('quick_replies'):
            response_data['quick_replies'] = result['quick_replies']
        if result.get('interactive_inputs'):
            response_data['interactive_inputs'] = result['interactive_inputs']

        return Response(response_data)


# ============== Document Views ==============

class DocumentViewSet(viewsets.ModelViewSet):
    """ViewSet for document management."""

    serializer_class = UploadedDocumentSerializer
    parser_classes = [MultiPartParser, FormParser]

    def get_queryset(self):
        return UploadedDocument.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        doc = serializer.save(user=self.request.user)

        # Process document after upload
        try:
            processor = DocumentProcessor()
            processor.process_document(doc)
        except Exception as e:
            logger.error(f"Document processing error: {e}")


# ============== RecurringBlock Views ==============

class RecurringBlockViewSet(viewsets.ModelViewSet):
    """ViewSet for recurring blocks."""

    serializer_class = RecurringBlockSerializer

    def get_queryset(self):
        return RecurringBlock.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=False, methods=['delete'])
    def clear_all(self, request):
        """Delete all recurring blocks for the current user."""
        deleted_count, _ = RecurringBlock.objects.filter(user=request.user).delete()

        # Also delete associated completions
        RecurringBlockCompletion.objects.filter(user=request.user).delete()

        # Reset onboarding to allow re-setup
        profile = request.user.profile
        profile.onboarding_completed = False
        profile.onboarding_step = 0
        profile.save()

        return Response({
            'message': f'{deleted_count} blocs supprimés',
            'deleted_count': deleted_count
        }, status=status.HTTP_200_OK)


# ============== RecurringBlockCompletion Views ==============

class RecurringBlockCompletionViewSet(viewsets.ModelViewSet):
    """ViewSet for recurring block completions."""

    serializer_class = RecurringBlockCompletionSerializer
    http_method_names = ['get', 'post', 'delete']

    def get_queryset(self):
        queryset = RecurringBlockCompletion.objects.filter(user=self.request.user)

        # Filter by date range
        start_date = self.request.query_params.get('start_date')
        end_date = self.request.query_params.get('end_date')

        if start_date:
            queryset = queryset.filter(date__gte=start_date)
        if end_date:
            queryset = queryset.filter(date__lte=end_date)

        return queryset

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    def create(self, request, *args, **kwargs):
        """Create or return existing completion (idempotent)."""
        recurring_block_id = request.data.get('recurring_block')
        date_str = request.data.get('date')

        # Check if completion already exists
        existing = RecurringBlockCompletion.objects.filter(
            user=request.user,
            recurring_block_id=recurring_block_id,
            date=date_str
        ).first()

        if existing:
            return Response(
                RecurringBlockCompletionSerializer(existing).data,
                status=status.HTTP_200_OK
            )

        return super().create(request, *args, **kwargs)

    @action(detail=False, methods=['delete'])
    def uncomplete(self, request):
        """Remove a completion by block and date."""
        recurring_block_id = request.query_params.get('recurring_block')
        date_str = request.query_params.get('date')

        if not recurring_block_id or not date_str:
            return Response(
                {'error': 'recurring_block et date requis.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        deleted, _ = RecurringBlockCompletion.objects.filter(
            user=request.user,
            recurring_block_id=recurring_block_id,
            date=date_str
        ).delete()

        if deleted:
            return Response(status=status.HTTP_204_NO_CONTENT)
        return Response(
            {'error': 'Complétion non trouvée.'},
            status=status.HTTP_404_NOT_FOUND
        )


# ============== Task Views ==============

class TaskViewSet(viewsets.ModelViewSet):
    """ViewSet for task management."""

    serializer_class = TaskSerializer

    def get_queryset(self):
        queryset = Task.objects.filter(user=self.request.user)

        # Filter by completed status
        completed = self.request.query_params.get('completed')
        if completed is not None:
            queryset = queryset.filter(completed=completed.lower() == 'true')

        # Filter by deadline
        has_deadline = self.request.query_params.get('has_deadline')
        if has_deadline is not None:
            if has_deadline.lower() == 'true':
                queryset = queryset.filter(deadline__isnull=False)
            else:
                queryset = queryset.filter(deadline__isnull=True)

        return queryset

    def perform_create(self, serializer):
        task = serializer.save(user=self.request.user)

        # Auto-schedule the task if it has a deadline or estimated duration
        if task.deadline or task.estimated_duration_minutes:
            try:
                scheduler = AIScheduler()
                scheduled_blocks = scheduler.generate_schedule(
                    user=self.request.user,
                    tasks=[task],
                    num_days=7
                )
                if scheduled_blocks:
                    logger.info(f"Auto-scheduled task '{task.title}' into {len(scheduled_blocks)} block(s)")
            except Exception as e:
                logger.error(f"Error auto-scheduling task '{task.title}': {e}")
                # Don't fail task creation if scheduling fails

    @action(detail=True, methods=['post'])
    def complete(self, request, pk=None):
        """Mark a task as completed."""
        task = self.get_object()

        serializer = TaskCompleteSerializer(data=request.data)
        if serializer.is_valid():
            task.completed = True
            task.completed_at = timezone.now()

            # Update scheduled block if exists
            actual_duration = serializer.validated_data.get('actual_duration_minutes')
            scheduled_time = None
            was_rescheduled = False
            reschedule_count = 0

            if actual_duration:
                scheduled_block = task.scheduled_blocks.filter(
                    actually_completed=False
                ).first()
                if scheduled_block:
                    scheduled_block.actually_completed = True
                    scheduled_block.actual_duration_minutes = actual_duration
                    scheduled_block.save()
                    scheduled_time = scheduled_block.start_time

                    # Check if task was rescheduled
                    reschedule_count = task.scheduled_blocks.count() - 1
                    was_rescheduled = reschedule_count > 0

            # Record task history for AI predictions
            if actual_duration:
                insights = AIInsightsService()
                insights.record_task_completion(
                    user=request.user,
                    task=task,
                    actual_duration=actual_duration,
                    scheduled_time=scheduled_time,
                    was_rescheduled=was_rescheduled,
                    reschedule_count=reschedule_count
                )

            task.save()
            return Response(TaskSerializer(task).data)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# ============== Schedule Views ==============

class ScheduleView(APIView):
    """View for schedule management."""

    def get(self, request):
        """Get weekly schedule."""
        start_date_str = request.query_params.get('start_date')
        if start_date_str:
            try:
                start_date = date.fromisoformat(start_date_str)
            except ValueError:
                return Response(
                    {'error': 'Format de date invalide. Utilisez YYYY-MM-DD.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
        else:
            start_date = timezone.now().date()

        end_date = start_date + timedelta(days=7)

        # Get recurring blocks for the week
        recurring_blocks = RecurringBlock.objects.filter(
            user=request.user,
            active=True
        )

        # Get scheduled task blocks
        scheduled_tasks = ScheduledBlock.objects.filter(
            user=request.user,
            date__gte=start_date,
            date__lt=end_date
        )

        # Get unscheduled tasks
        scheduled_task_ids = scheduled_tasks.values_list('task_id', flat=True)
        unscheduled_tasks = Task.objects.filter(
            user=request.user,
            completed=False
        ).exclude(id__in=scheduled_task_ids)

        # Get recurring block completions for the week
        recurring_completions = RecurringBlockCompletion.objects.filter(
            user=request.user,
            date__gte=start_date,
            date__lt=end_date
        )

        data = {
            'recurring_blocks': RecurringBlockSerializer(recurring_blocks, many=True).data,
            'scheduled_tasks': ScheduledBlockSerializer(scheduled_tasks, many=True).data,
            'unscheduled_tasks': TaskSerializer(unscheduled_tasks, many=True).data,
            'recurring_completions': RecurringBlockCompletionSerializer(recurring_completions, many=True).data,
        }

        return Response(data)


class ScheduleGenerateView(APIView):
    """Generate a new schedule."""

    def post(self, request):
        """Generate schedule for user's tasks."""
        serializer = ScheduleGenerateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        start_date = serializer.validated_data.get('start_date') or timezone.now().date()
        force = serializer.validated_data.get('force', False)

        scheduler = AIScheduler()
        created_blocks = scheduler.generate_schedule(
            request.user,
            start_date=start_date,
            force=force
        )

        return Response({
            'created_blocks': ScheduledBlockSerializer(created_blocks, many=True).data,
            'count': len(created_blocks),
        })


class ScheduledBlockView(APIView):
    """Update a scheduled block (for drag & drop)."""

    def patch(self, request, block_id):
        """Update a scheduled block."""
        try:
            block = ScheduledBlock.objects.get(id=block_id, user=request.user)
        except ScheduledBlock.DoesNotExist:
            return Response(
                {'error': 'Bloc non trouvé.'},
                status=status.HTTP_404_NOT_FOUND
            )

        serializer = ScheduledBlockSerializer(block, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# ============== Conversation Views ==============

class ConversationView(APIView):
    """Get conversation history."""

    def get(self, request):
        """Return conversation messages."""
        limit = request.query_params.get('limit', 50)
        try:
            limit = int(limit)
        except ValueError:
            limit = 50

        messages = ConversationMessage.objects.filter(
            user=request.user
        ).order_by('-created_at')[:limit]

        # Reverse to get chronological order
        messages = list(reversed(messages))

        return Response(ConversationMessageSerializer(messages, many=True).data)


# ============== AI Insights Views ==============

class SuggestionsView(APIView):
    """Get AI-powered proactive suggestions."""

    def get(self, request):
        """Get suggestions for today or a specific date."""
        date_str = request.query_params.get('date')
        if date_str:
            try:
                target_date = date.fromisoformat(date_str)
            except ValueError:
                return Response(
                    {'error': 'Format de date invalide. Utilisez YYYY-MM-DD.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
        else:
            target_date = timezone.now().date()

        limit = int(request.query_params.get('limit', 5))

        insights = AIInsightsService()
        suggestions = insights.get_proactive_suggestions(
            request.user,
            target_date=target_date,
            limit=limit
        )

        return Response({
            'suggestions': [
                {
                    'type': s.type,
                    'message': s.message,
                    'task_id': s.task_id,
                    'action': s.action,
                    'metadata': s.metadata,
                }
                for s in suggestions
            ],
            'date': target_date.isoformat(),
        })


class PatternsView(APIView):
    """Get user productivity patterns."""

    def get(self, request):
        """Analyze and return user patterns."""
        insights = AIInsightsService()
        patterns = insights.analyze_user_patterns(request.user)
        return Response(patterns)


class PredictDurationView(APIView):
    """Predict task duration based on history."""

    def post(self, request):
        """Predict duration for a task."""
        task_title = request.data.get('title', '')
        task_type = request.data.get('task_type', 'shallow')
        scheduled_time = request.data.get('scheduled_time')

        if scheduled_time:
            try:
                from datetime import time as dt_time
                parts = scheduled_time.split(':')
                scheduled_time = dt_time(int(parts[0]), int(parts[1]))
            except (ValueError, IndexError):
                scheduled_time = None

        insights = AIInsightsService()
        prediction = insights.predict_duration(
            request.user,
            task_title=task_title,
            task_type=task_type,
            scheduled_time=scheduled_time
        )

        return Response(prediction)


class ConflictsView(APIView):
    """Detect scheduling conflicts."""

    def get(self, request):
        """Get conflicts for the next N days."""
        date_str = request.query_params.get('date')
        if date_str:
            try:
                target_date = date.fromisoformat(date_str)
            except ValueError:
                return Response(
                    {'error': 'Format de date invalide. Utilisez YYYY-MM-DD.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
        else:
            target_date = timezone.now().date()

        days_ahead = int(request.query_params.get('days', 7))

        insights = AIInsightsService()
        conflicts = insights.detect_conflicts(
            request.user,
            target_date=target_date,
            days_ahead=days_ahead
        )

        return Response({
            'conflicts': [
                {
                    'type': c.type,
                    'severity': c.severity,
                    'message': c.message,
                    'blocks_involved': c.blocks_involved,
                    'suggested_resolution': c.suggested_resolution,
                }
                for c in conflicts
            ],
            'count': len(conflicts),
        })


class SmartRescheduleView(APIView):
    """Handle smart rescheduling when tasks overflow."""

    def post(self, request):
        """Reschedule affected blocks after a task overflows."""
        block_id = request.data.get('block_id')
        actual_end_time = request.data.get('actual_end_time')

        if not block_id or not actual_end_time:
            return Response(
                {'error': 'block_id et actual_end_time requis.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            from datetime import time as dt_time
            parts = actual_end_time.split(':')
            end_time = dt_time(int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            return Response(
                {'error': 'Format de temps invalide. Utilisez HH:MM.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        insights = AIInsightsService()
        result = insights.smart_reschedule(
            request.user,
            overflowed_block_id=int(block_id),
            actual_end_time=end_time
        )

        if 'error' in result:
            return Response(result, status=status.HTTP_400_BAD_REQUEST)

        return Response(result)


class NaturalLanguageScheduleView(APIView):
    """Parse and execute natural language scheduling requests."""

    def post(self, request):
        """Parse a natural language request and optionally execute it."""
        message = request.data.get('message', '')
        execute = request.data.get('execute', False)

        if not message:
            return Response(
                {'error': 'Message requis.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        insights = AIInsightsService()
        parsed = insights.parse_scheduling_request(request.user, message)

        response_data = {
            'parsed': parsed,
            'executed': False,
        }

        if execute and parsed.get('task_title'):
            result = insights.execute_scheduling_request(request.user, parsed)
            response_data['execution_result'] = result
            response_data['executed'] = True

        return Response(response_data)


# ============== Share Schedule Views ==============

class ShareScheduleView(APIView):
    """Create and manage schedule share links."""

    def get(self, request):
        """List all active share links for the user."""
        shares = SharedSchedule.objects.filter(user=request.user, is_active=True)
        return Response(SharedScheduleSerializer(shares, many=True).data)

    def post(self, request):
        """Create a new share link."""
        serializer = CreateShareSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        expires_at = None
        if serializer.validated_data.get('expires_in_days'):
            expires_at = timezone.now() + timedelta(days=serializer.validated_data['expires_in_days'])

        share = SharedSchedule.objects.create(
            user=request.user,
            title=serializer.validated_data.get('title', 'Mon planning'),
            expires_at=expires_at,
            include_tasks=serializer.validated_data.get('include_tasks', False),
        )

        return Response(SharedScheduleSerializer(share).data, status=status.HTTP_201_CREATED)


class ShareScheduleDetailView(APIView):
    """Manage a specific share link."""

    def delete(self, request, share_id):
        """Deactivate a share link."""
        try:
            share = SharedSchedule.objects.get(id=share_id, user=request.user)
        except SharedSchedule.DoesNotExist:
            return Response(
                {'error': 'Lien de partage non trouve.'},
                status=status.HTTP_404_NOT_FOUND
            )

        share.is_active = False
        share.save()
        return Response({'message': 'Lien de partage desactive.'})


class PublicScheduleView(APIView):
    """View a shared schedule (public, no auth required)."""

    permission_classes = [AllowAny]

    def get(self, request, share_token):
        """Get shared schedule data."""
        try:
            share = SharedSchedule.objects.get(share_token=share_token)
        except SharedSchedule.DoesNotExist:
            return Response(
                {'error': 'Lien de partage invalide.'},
                status=status.HTTP_404_NOT_FOUND
            )

        if not share.is_valid():
            return Response(
                {'error': 'Ce lien de partage a expire ou a ete desactive.'},
                status=status.HTTP_410_GONE
            )

        # Increment view count
        share.view_count += 1
        share.save(update_fields=['view_count'])

        # Get recurring blocks
        recurring_blocks = RecurringBlock.objects.filter(
            user=share.user,
            active=True
        )

        response_data = {
            'title': share.title,
            'owner': share.user.first_name or share.user.username,
            'recurring_blocks': RecurringBlockSerializer(recurring_blocks, many=True).data,
            'scheduled_tasks': [],
        }

        # Include tasks if enabled
        if share.include_tasks:
            today = timezone.now().date()
            end_date = today + timedelta(days=7)
            scheduled_tasks = ScheduledBlock.objects.filter(
                user=share.user,
                date__gte=today,
                date__lt=end_date
            )
            response_data['scheduled_tasks'] = ScheduledBlockSerializer(scheduled_tasks, many=True).data

        return Response(response_data)
