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
)
from services import DocumentProcessor, ChatEngine, AIScheduler

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
            if actual_duration:
                scheduled_block = task.scheduled_blocks.filter(
                    actually_completed=False
                ).first()
                if scheduled_block:
                    scheduled_block.actually_completed = True
                    scheduled_block.actual_duration_minutes = actual_duration
                    scheduled_block.save()

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
