"""
API Views for Planner AI backend.
"""
import logging
from datetime import date, timedelta

from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db.models import Q
from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from rest_framework.authtoken.models import Token
from rest_framework_simplejwt.tokens import RefreshToken

from .models import (
    UserProfile,
    UploadedDocument,
    RecurringBlock,
    RecurringBlockCompletion,
    RecurringBlockException,
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
    UserPlaceSerializer,
    RecurringBlockSerializer,
    RecurringBlockCompletionSerializer,
    RecurringBlockExceptionSerializer,
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
from services import DocumentProcessor, AIScheduler
from services.ai_insights import AIInsightsService
from services.agent import PlannerAgent

logger = logging.getLogger(__name__)


def _as_bool(value):
    """Coerce a request value (bool, int, or string) to a boolean.

    Form-encoded payloads deliver booleans as strings ("true", "1"), while
    JSON payloads deliver real booleans; both must be interpreted the same way.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ('true', '1', 'yes', 'on')


def _as_int_or_none(value):
    """Coerce a request value to an int, or None when empty/invalid."""
    if value is None or value == '':
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class CheckEmailRateThrottle(ScopedRateThrottle):
    """Scoped throttle for CheckEmailView (S8: account-existence oracle).

    Uses the ``check_email`` scope that A1 wires into ``DEFAULT_THROTTLE_RATES``.
    Falls back to a conservative default if that scope has not been configured
    so the endpoint never raises ImproperlyConfigured (500) at runtime.
    """

    def get_rate(self):
        try:
            return super().get_rate()
        except ImproperlyConfigured:
            return '10/min'


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

        # Use Django's authenticate() so the auth backend enforces is_active:
        # a deactivated account must never receive tokens (B24).
        user = authenticate(request, username=username, password=password)
        if user is None:
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


class CheckEmailView(APIView):
    """Check if email exists in database for lazy auth flow."""

    permission_classes = [AllowAny]
    throttle_classes = [CheckEmailRateThrottle]
    throttle_scope = 'check_email'

    def post(self, request):
        """Check if email exists and return status."""
        email = request.data.get('email', '').strip().lower()

        if not email:
            return Response(
                {'error': 'Email requis.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Check if user exists with this email
        exists = User.objects.filter(email=email).exists()

        return Response({
            'exists': exists,
            'email': email,
        })


class MeView(APIView):
    """Get current user information."""

    def get(self, request):
        """Return current user data."""
        return Response(UserSerializer(request.user).data)


class McpTokenView(APIView):
    """Long-lived per-user API token for the MCP server.

    Authenticated via the normal session (JWT). GET returns the caller's token
    (creating it on first use); POST rotates it. The MCP server then calls the
    Planner API on the user's behalf with 'Authorization: Token <key>'.
    """

    def get(self, request):
        token, _ = Token.objects.get_or_create(user=request.user)
        return Response({"token": token.key, "username": request.user.username})

    def post(self, request):
        Token.objects.filter(user=request.user).delete()
        token = Token.objects.create(user=request.user)
        return Response({"token": token.key, "username": request.user.username})


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

            # S6: reject the token unless Google asserts the email is verified.
            # Google tokeninfo returns email_verified as the string "true".
            # Without this check an attacker can mint a token for an unverified
            # address and get auto-linked to a victim's account (takeover).
            email_verified = google_data.get('email_verified')
            if email_verified not in (True, 'true', 'True'):
                return Response(
                    {'error': 'Adresse email Google non vérifiée.'},
                    status=status.HTTP_401_UNAUTHORIZED
                )

            # Shared account resolution (dedupe-aware): a Google-verified email
            # has one owner, so duplicates are the same person -> sign into the
            # account holding their data instead of hard-blocking. See
            # services/social_login.resolve_social_user (also used by Apple).
            from services.social_login import resolve_social_user

            user, created = resolve_social_user(
                email,
                first_name=google_data.get('given_name', ''),
                last_name=google_data.get('family_name', ''),
            )

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


class AppleAuthView(APIView):
    """Sign in with Apple: verify the identity token, then login/register."""

    permission_classes = [AllowAny]

    def post(self, request):
        from services.apple_auth import apple_configured, verify_apple_identity_token
        from services.social_login import resolve_social_user

        if not apple_configured():
            return Response(
                {'error': "Connexion Apple non configurée."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        id_token = request.data.get('id_token') or request.data.get('identityToken')
        if not id_token:
            return Response({'error': 'id_token requis.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            claims = verify_apple_identity_token(id_token)
        except Exception as e:  # noqa: BLE001 - never leak the raw reason
            logger.warning("Apple auth failed: %s", e)
            return Response({'error': 'Token Apple invalide.'}, status=status.HTTP_401_UNAUTHORIZED)

        email = claims.get('email')
        if not email:
            return Response(
                {'error': 'Email non fourni par Apple.'}, status=status.HTTP_400_BAD_REQUEST
            )
        # Apple sends email_verified as bool True or the string "true" (private
        # relay addresses are verified). Reject anything else.
        if claims.get('email_verified') not in (True, 'true', 'True'):
            return Response(
                {'error': 'Adresse email Apple non vérifiée.'},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        # Apple only sends the name on the FIRST authorization, in the client
        # payload (not the token). Accept it if present.
        name = request.data.get('name') or {}
        first_name = (name.get('firstName') or name.get('given_name') or '') if isinstance(name, dict) else ''
        last_name = (name.get('lastName') or name.get('family_name') or '') if isinstance(name, dict) else ''

        user, created = resolve_social_user(email, first_name=first_name, last_name=last_name)

        refresh = RefreshToken.for_user(user)
        return Response({
            'user': UserSerializer(user).data,
            'tokens': {'refresh': str(refresh), 'access': str(refresh.access_token)},
            'created': created,
        })


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
        logger.debug("Chat POST received from user %s", request.user.id)
        message = request.data.get('message', '')
        attachment_file = request.FILES.get('attachment')
        logger.debug(
            "Chat message len=%s attachment=%s",
            len(message) if message else 0,
            bool(attachment_file),
        )

        if not message and not attachment_file:
            return Response(
                {'error': 'Message ou fichier requis.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Handle file upload
        attachment = None
        if attachment_file:
            logger.debug(
                "Chat attachment name=%s size=%s",
                attachment_file.name,
                attachment_file.size,
            )

            # S4: validate the upload (size + extension allowlist + magic bytes)
            # BEFORE persisting it, so a smuggled SVG/HTML/executable payload or
            # an oversized file never reaches storage or the processor.
            from core.validators import validate_upload_file
            from rest_framework.exceptions import ValidationError as DRFValidationError
            try:
                validate_upload_file(attachment_file)
            except DRFValidationError as exc:
                return Response(
                    {'error': 'Fichier invalide.', 'detail': exc.detail},
                    status=status.HTTP_400_BAD_REQUEST
                )

            doc_type = request.data.get('document_type', 'other')
            doc = UploadedDocument.objects.create(
                user=request.user,
                file=attachment_file,
                document_type=doc_type,
            )
            attachment = doc
            logger.debug("Chat document created id=%s", doc.id)

            # Process document ASYNCHRONOUSLY to avoid timeout
            try:
                processor = DocumentProcessor()
                processor.process_document_async(doc.id)
                logger.debug("Chat async processing started for doc %s", doc.id)
            except Exception as e:
                logger.error(f"Document processing error: {e}")

        # Generate chat response via PlannerAgent
        try:
            agent = PlannerAgent()
            result = agent.process_message(request.user, message or "J'ai uploadé un document.", attachment)
            logger.info(f"Agent response generated: {result.get('response', '')[:100]}...")
        except Exception as e:
            # Log the detail server-side; never leak the raw exception string
            # (paths, provider errors, tokens) to the client.
            logger.error(f"PlannerAgent error: {e}", exc_info=True)
            return Response(
                {'error': 'Erreur interne lors du traitement du message.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        response_data = {
            'response': result['response'],
        }
        if result.get('quick_replies'):
            response_data['quick_replies'] = result['quick_replies']
        if result.get('interactive_inputs'):
            response_data['interactive_inputs'] = result['interactive_inputs']
        if result.get('blocks_created'):
            response_data['blocks_created'] = result['blocks_created']
        if result.get('tasks_created'):
            response_data['tasks_created'] = result['tasks_created']

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

        # Process document ASYNCHRONOUSLY after upload
        try:
            processor = DocumentProcessor()
            processor.process_document_async(doc.id)
        except Exception as e:
            logger.error(f"Document processing error: {e}")


# ============== UserPlace Views ==============

class UserPlaceViewSet(viewsets.ModelViewSet):
    """CRUD for the user's places (travel-time engine, Phase 1).

    Each place carries the usual trip duration from home; the scheduler derives
    the latest-departure / unavailability window from it for any recurring
    block attached to the place.
    """

    serializer_class = UserPlaceSerializer

    def get_queryset(self):
        from core.models import UserPlace
        return UserPlace.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


# ============== RecurringBlock Views ==============

class RecurringBlockViewSet(viewsets.ModelViewSet):
    """ViewSet for recurring blocks."""

    serializer_class = RecurringBlockSerializer

    def get_queryset(self):
        # Default manager hides 'pending' blocks, so the normal list/detail
        # endpoints only ever expose confirmed blocks.
        return RecurringBlock.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=False, methods=['get'])
    def pending(self, request):
        """List blocks awaiting confirmation (low-confidence extractions)."""
        qs = RecurringBlock.all_objects.filter(
            user=request.user, status=RecurringBlock.STATUS_PENDING
        )
        return Response(RecurringBlockSerializer(qs, many=True).data)

    @action(detail=True, methods=['post'])
    def confirm(self, request, pk=None):
        """Confirm a pending block -> it becomes active (visible in the planning)."""
        block = RecurringBlock.all_objects.filter(user=request.user, pk=pk).first()
        if block is None:
            return Response({'error': 'Bloc introuvable.'}, status=status.HTTP_404_NOT_FOUND)
        if block.status != RecurringBlock.STATUS_ACTIVE:
            block.status = RecurringBlock.STATUS_ACTIVE
            block.save(update_fields=['status'])
        return Response(RecurringBlockSerializer(block).data)

    @action(detail=True, methods=['post'])
    def reject(self, request, pk=None):
        """Reject a pending (or any) block -> deleted."""
        block = RecurringBlock.all_objects.filter(user=request.user, pk=pk).first()
        if block is None:
            return Response({'error': 'Bloc introuvable.'}, status=status.HTTP_404_NOT_FOUND)
        block.delete()
        return Response({'deleted': 1})

    @action(detail=False, methods=['post'])
    def deduplicate(self, request):
        """Remove exact-duplicate recurring blocks (same day/time/type)."""
        from services.blocks_maintenance import dedupe_recurring_blocks
        return Response(dedupe_recurring_blocks(request.user))

    @action(detail=False, methods=['post'])
    def confirm_all(self, request):
        """Confirm every pending block at once (bulk 1-tap accept)."""
        updated = RecurringBlock.all_objects.filter(
            user=request.user, status=RecurringBlock.STATUS_PENDING
        ).update(status=RecurringBlock.STATUS_ACTIVE)
        return Response({'confirmed': updated})

    @action(detail=False, methods=['delete'])
    def clear_all(self, request):
        """Delete all recurring blocks for the current user (incl. pending)."""
        deleted_count, _ = RecurringBlock.all_objects.filter(user=request.user).delete()

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


# ============== RecurringBlockException Views ==============

class RecurringBlockExceptionViewSet(viewsets.ModelViewSet):
    """ViewSet for recurring block exceptions (skipped/cancelled occurrences)."""

    serializer_class = RecurringBlockExceptionSerializer
    http_method_names = ['get', 'post', 'delete']

    def get_queryset(self):
        queryset = RecurringBlockException.objects.filter(user=self.request.user)
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
        """Create or return existing exception (idempotent)."""
        recurring_block_id = request.data.get('recurring_block')
        date_str = request.data.get('date')

        existing = RecurringBlockException.objects.filter(
            user=request.user,
            recurring_block_id=recurring_block_id,
            date=date_str
        ).first()

        if existing:
            return Response(
                RecurringBlockExceptionSerializer(existing).data,
                status=status.HTTP_200_OK
            )

        return super().create(request, *args, **kwargs)

    @action(detail=False, methods=['delete'])
    def restore(self, request):
        """Remove an exception by block and date (un-skip the occurrence)."""
        recurring_block_id = request.query_params.get('recurring_block')
        date_str = request.query_params.get('date')

        if not recurring_block_id or not date_str:
            return Response(
                {'error': 'recurring_block et date requis.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        deleted, _ = RecurringBlockException.objects.filter(
            user=request.user,
            recurring_block_id=recurring_block_id,
            date=date_str
        ).delete()

        if deleted:
            return Response(status=status.HTTP_204_NO_CONTENT)
        return Response(
            {'error': 'Exception non trouvée.'},
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

        # Fan out to any automation the user wired up (n8n, etc.).
        try:
            from services.webhooks import dispatch
            dispatch(self.request.user, "task.created", {
                "task": {
                    "id": task.id,
                    "title": task.title,
                    "deadline": task.deadline.isoformat() if task.deadline else None,
                    "priority": task.priority,
                    "task_type": task.task_type,
                },
            })
        except Exception as e:
            logger.error(f"Webhook dispatch (task.created) failed: {e}")

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
        """Mark a task as completed.

        D9: routes through ``Task.mark_completed`` (the canonical sync helper
        added by T1) so ``Task.completed`` / ``completed_at``, the active
        ``ScheduledBlock.actually_completed`` and the ``TaskHistory`` record
        never diverge. The helper is idempotent and records history itself, so
        the view no longer touches those representations directly. A fallback
        replicates the same reconciliation on schemas without the helper.
        """
        task = self.get_object()

        serializer = TaskCompleteSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        actual_duration = serializer.validated_data.get('actual_duration_minutes')

        if hasattr(task, 'mark_completed'):
            task.mark_completed(actual_minutes=actual_duration)
        else:
            # Fallback for environments without the T1 sync helpers: keep the
            # exact prior behaviour (mark task, sync the active block, record
            # history) so the four representations stay consistent.
            task.completed = True
            task.completed_at = timezone.now()

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

                    reschedule_count = task.scheduled_blocks.count() - 1
                    was_rescheduled = reschedule_count > 0

                AIInsightsService().record_task_completion(
                    user=request.user,
                    task=task,
                    actual_duration=actual_duration,
                    scheduled_time=scheduled_time,
                    was_rescheduled=was_rescheduled,
                    reschedule_count=reschedule_count,
                )

            task.save()

        try:
            from services.webhooks import dispatch
            dispatch(request.user, "task.completed", {
                "task": {
                    "id": task.id,
                    "title": task.title,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "actual_duration_minutes": actual_duration,
                },
            })
        except Exception as e:
            logger.error(f"Webhook dispatch (task.completed) failed: {e}")

        return Response(TaskSerializer(task).data)


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

        # Get recurring block exceptions (skipped occurrences) for the week
        recurring_exceptions = RecurringBlockException.objects.filter(
            user=request.user,
            date__gte=start_date,
            date__lt=end_date
        )

        data = {
            'recurring_blocks': RecurringBlockSerializer(recurring_blocks, many=True).data,
            'scheduled_tasks': ScheduledBlockSerializer(scheduled_tasks, many=True).data,
            'unscheduled_tasks': TaskSerializer(unscheduled_tasks, many=True).data,
            'recurring_completions': RecurringBlockCompletionSerializer(recurring_completions, many=True).data,
            'recurring_exceptions': RecurringBlockExceptionSerializer(recurring_exceptions, many=True).data,
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
            # Spec §10: never hide an impossible plan — surface what did not
            # fit and why so the UI can propose compromises.
            'unplaced': getattr(scheduler, 'last_unplaced', []),
        })


class ScheduleReplanView(APIView):
    """Partial replan after a delay (spec §7).

    Body: {"resume_time": "HH:MM"} or {"delay_minutes": 30}. Locks fixed events
    and already-started activities, moves only the displaced flexible blocks
    after the resume time, and explains what changed.
    """

    def post(self, request):
        from services.replan import replan_after_delay

        resume_time = request.data.get('resume_time')
        delay_minutes = request.data.get('delay_minutes')
        if delay_minutes is not None:
            try:
                delay_minutes = int(delay_minutes)
            except (TypeError, ValueError):
                return Response(
                    {'error': 'delay_minutes doit être un entier.'},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        result = replan_after_delay(
            request.user, resume_time=resume_time, delay_minutes=delay_minutes
        )
        return Response(result)


class ScheduleProposalApplyView(APIView):
    """Apply a previously-proposed schedule change (spec §8 suggestion/semi-auto)."""

    def post(self, request):
        from services.replan import apply_proposal
        token = request.data.get('token')
        if not token:
            return Response({'error': 'token requis.'}, status=status.HTTP_400_BAD_REQUEST)
        result = apply_proposal(request.user, token)
        if result is None:
            return Response(
                {'error': 'Proposition introuvable ou déjà traitée.'},
                status=status.HTTP_404_NOT_FOUND,
            )
        return Response(result)


class ScheduleUndoView(APIView):
    """Undo an applied schedule change (spec §8: automatique = annulable)."""

    def post(self, request):
        from services.replan import undo_change
        token = request.data.get('token')
        if not token:
            return Response({'error': 'token requis.'}, status=status.HTTP_400_BAD_REQUEST)
        result = undo_change(request.user, token)
        if result is None:
            return Response(
                {'error': 'Changement introuvable ou déjà annulé.'},
                status=status.HTTP_404_NOT_FOUND,
            )
        return Response(result)


class ScheduleRollOverView(APIView):
    """Roll missed blocks forward (structural forgiveness — no red, no debt)."""

    def post(self, request):
        from services.rollover import roll_over_missed
        return Response(roll_over_missed(request.user))


class ScheduleResetDayView(APIView):
    """'Remettre à plat': rebase today on reality (bankruptcy antidote)."""

    def post(self, request):
        today = timezone.localdate()
        now_t = timezone.localtime().time()
        # Drop today's not-yet-done blocks, keep completed AND locked (sanctuary)
        # work, then regenerate around what remains.
        ScheduledBlock.objects.filter(
            user=request.user, date=today, actually_completed=False, locked=False
        ).delete()
        scheduler = AIScheduler()
        created = scheduler.generate_schedule(
            request.user, start_date=today, num_days=1,
            earliest_start={today: now_t},
        )
        return Response({
            'created_blocks': ScheduledBlockSerializer(created, many=True).data,
            'count': len(created),
            'unplaced': getattr(scheduler, 'last_unplaced', []),
        })


class StreakView(APIView):
    """Forgiving elastic streak (days you followed OR adjusted your plan)."""

    def get(self, request):
        from services.streak import compute_streak
        return Response(compute_streak(request.user))


class WeeklySummaryView(APIView):
    """Positive peak-end weekly summary (accomplishments only, tied to a stake)."""

    def get(self, request):
        from services.progress import weekly_summary
        return Response(weekly_summary(request.user))


class ScheduledBlockView(APIView):
    """Update a scheduled block (for drag & drop)."""

    def patch(self, request, block_id):
        """Update a scheduled block.

        D9: when the PATCH marks the block completed, the completion is applied
        through ``ScheduledBlock.mark_completed`` (the canonical sync helper
        added by T1) instead of a raw field write, so the parent ``Task`` and
        the ``TaskHistory`` record stay in sync. Other field updates (drag &
        drop of ``start_time`` / ``end_time``) still flow through the serializer.
        """
        try:
            block = ScheduledBlock.objects.get(id=block_id, user=request.user)
        except ScheduledBlock.DoesNotExist:
            return Response(
                {'error': 'Bloc non trouvé.'},
                status=status.HTTP_404_NOT_FOUND
            )

        data = {key: request.data[key] for key in request.data}
        completing = _as_bool(data.get('actually_completed'))
        completion_minutes = None
        if completing:
            # Route the completion transition through the helper rather than the
            # serializer so Task + TaskHistory are reconciled canonically.
            data.pop('actually_completed', None)
            completion_minutes = _as_int_or_none(data.pop('actual_duration_minutes', None))

        serializer = ScheduledBlockSerializer(block, data=data, partial=True)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        serializer.save()

        if completing:
            block.refresh_from_db()
            if hasattr(block, 'mark_completed'):
                block.mark_completed(actual_minutes=completion_minutes)
            else:
                # Fallback: preserve prior behaviour (flag-only write) on
                # schemas without the T1 sync helpers.
                block.actually_completed = True
                if completion_minutes is not None:
                    block.actual_duration_minutes = completion_minutes
                block.save()
            block.refresh_from_db()

        return Response(ScheduledBlockSerializer(block).data)


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


class PublicPlanningByUsernameView(APIView):
    """View a user's planning by username (public, no auth required)."""

    permission_classes = [AllowAny]

    def get(self, request, username):
        """Get a user's public planning data by username.

        S2: this endpoint used to leak every user's schedule AND room/location
        by username enumeration, bypassing the token-based share mechanism. It
        is now gated behind an explicit per-user opt-in
        (UserProfile.public_planning_enabled). Users who did not opt in are
        indistinguishable from a non-existent user (404 either way), so the
        endpoint can no longer be used to enumerate accounts. When a user did
        opt in, exact location/room info is stripped from the response.
        """
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return Response(
                {'error': 'Utilisateur introuvable.'},
                status=status.HTTP_404_NOT_FOUND
            )

        profile = getattr(user, 'profile', None)
        if profile is None or not getattr(profile, 'public_planning_enabled', False):
            # Do not reveal whether the account exists or simply did not opt in.
            return Response(
                {'error': 'Utilisateur introuvable.'},
                status=status.HTTP_404_NOT_FOUND
            )

        recurring_blocks = RecurringBlock.objects.filter(user=user, active=True)
        blocks_data = RecurringBlockSerializer(recurring_blocks, many=True).data

        # Strip exact location/room info from the public payload.
        for block in blocks_data:
            block.pop('location', None)

        return Response({
            'title': f'Planning de {user.first_name or user.username}',
            'owner': user.first_name or user.username,
            'username': user.username,
            'avatar_url': getattr(profile, 'avatar_url', None),
            'recurring_blocks': blocks_data,
            'scheduled_tasks': [],
        })


# ============== Web Push (VAPID) ==============
class PushPublicKeyView(APIView):
    """Return the public VAPID key so a client can subscribe."""

    def get(self, request):
        from django.conf import settings as _s
        return Response({"publicKey": _s.VAPID_PUBLIC_KEY})


class PushSubscribeView(APIView):
    """Register (or refresh) a Web Push subscription for the current user."""

    def post(self, request):
        from core.models import PushSubscription
        endpoint = request.data.get("endpoint")
        keys = request.data.get("keys") or {}
        p256dh = keys.get("p256dh") or request.data.get("p256dh")
        auth = keys.get("auth") or request.data.get("auth")
        if not (endpoint and p256dh and auth):
            return Response(
                {"error": "endpoint, keys.p256dh et keys.auth requis."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        _, created = PushSubscription.objects.update_or_create(
            endpoint=endpoint,
            defaults={
                "user": request.user,
                "p256dh": p256dh,
                "auth": auth,
                "user_agent": request.META.get("HTTP_USER_AGENT", "")[:300],
            },
        )
        return Response(
            {"ok": True, "created": created},
            status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
        )


class PushUnsubscribeView(APIView):
    def post(self, request):
        from core.models import PushSubscription
        endpoint = request.data.get("endpoint")
        if not endpoint:
            return Response({"error": "endpoint requis."}, status=status.HTTP_400_BAD_REQUEST)
        deleted, _ = PushSubscription.objects.filter(
            user=request.user, endpoint=endpoint
        ).delete()
        return Response({"deleted": deleted})


class PushTestView(APIView):
    """Send a test push to the current user's devices."""

    def post(self, request):
        from services.push import push_configured, send_to_user
        if not push_configured():
            return Response(
                {"error": "Web push non configuré (VAPID)."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        sent = send_to_user(
            request.user, "Planner AI",
            request.data.get("message") or "Notification de test ✅", url="/",
        )
        return Response({"sent": sent})


# ============== iCal calendar feed ==============
class CalendarFeedView(APIView):
    """Manage the current user's stable iCal subscription URL.

    GET  -> feed metadata + absolute https/webcal URLs (creates the feed lazily).
    POST -> rotate the token (revokes every existing calendar subscription).
    PATCH-> toggle include_tasks / is_active.
    """

    def _payload(self, request, feed):
        https_url = request.build_absolute_uri(feed.feed_path)
        return {
            "token": str(feed.token),
            "include_tasks": feed.include_tasks,
            "is_active": feed.is_active,
            "url": https_url,
            "webcal_url": https_url.replace("https://", "webcal://").replace(
                "http://", "webcal://"
            ),
            "access_count": feed.access_count,
            "last_accessed_at": feed.last_accessed_at,
        }

    def get(self, request):
        from core.models import CalendarFeed
        feed, _ = CalendarFeed.objects.get_or_create(user=request.user)
        return Response(self._payload(request, feed))

    def post(self, request):
        import uuid as _uuid
        from core.models import CalendarFeed
        feed, _ = CalendarFeed.objects.get_or_create(user=request.user)
        feed.token = _uuid.uuid4()
        feed.is_active = True
        feed.save(update_fields=["token", "is_active"])
        return Response(self._payload(request, feed))

    def patch(self, request):
        from core.models import CalendarFeed
        feed, _ = CalendarFeed.objects.get_or_create(user=request.user)
        if "include_tasks" in request.data:
            feed.include_tasks = bool(request.data["include_tasks"])
        if "is_active" in request.data:
            feed.is_active = bool(request.data["is_active"])
        feed.save(update_fields=["include_tasks", "is_active"])
        return Response(self._payload(request, feed))


class CalendarICSView(APIView):
    """Public, token-secured .ics feed that calendar apps subscribe to."""

    permission_classes = [AllowAny]
    authentication_classes = []

    def get(self, request, token):
        from django.http import HttpResponse
        from django.db.models import F
        from core.models import CalendarFeed
        from services.ical import build_calendar

        try:
            feed = CalendarFeed.objects.select_related("user").get(token=token)
        except (CalendarFeed.DoesNotExist, ValueError, ValidationError):
            return Response(
                {"error": "Flux calendrier introuvable."},
                status=status.HTTP_404_NOT_FOUND,
            )
        if not feed.is_active:
            return Response(
                {"error": "Ce flux a été désactivé."}, status=status.HTTP_410_GONE
            )

        ics = build_calendar(feed.user, include_tasks=feed.include_tasks)

        CalendarFeed.objects.filter(id=feed.id).update(
            access_count=F("access_count") + 1, last_accessed_at=timezone.now()
        )

        resp = HttpResponse(ics, content_type="text/calendar; charset=utf-8")
        resp["Content-Disposition"] = 'inline; filename="planner.ics"'
        resp["Cache-Control"] = "no-cache, max-age=0"
        return resp


# ============== Outbound webhooks (n8n / automation) ==============
class WebhookEndpointView(APIView):
    """List and create the current user's outbound webhooks."""

    def get(self, request):
        from core.models import WebhookEndpoint
        hooks = WebhookEndpoint.objects.filter(user=request.user)
        return Response([_webhook_dict(h) for h in hooks])

    def post(self, request):
        import secrets as _secrets
        from core.models import WebhookEndpoint

        url = (request.data.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            return Response(
                {"error": "Une URL http(s) valide est requise."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        valid_events = {c[0] for c in WebhookEndpoint.EVENT_CHOICES}
        events = request.data.get("events") or []
        if not isinstance(events, list) or any(e not in valid_events for e in events):
            return Response(
                {"error": f"events doit être une liste parmi {sorted(valid_events)}."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        hook = WebhookEndpoint.objects.create(
            user=request.user,
            url=url,
            secret=request.data.get("secret") or _secrets.token_hex(16),
            events=events,
            description=(request.data.get("description") or "")[:200],
        )
        return Response(_webhook_dict(hook, reveal_secret=True), status=status.HTTP_201_CREATED)


class WebhookEndpointDetailView(APIView):
    """Update or delete a specific webhook."""

    def _get(self, request, hook_id):
        from core.models import WebhookEndpoint
        return WebhookEndpoint.objects.filter(id=hook_id, user=request.user).first()

    def patch(self, request, hook_id):
        hook = self._get(request, hook_id)
        if hook is None:
            return Response({"error": "Webhook introuvable."}, status=status.HTTP_404_NOT_FOUND)
        if "active" in request.data:
            hook.active = bool(request.data["active"])
        if "url" in request.data:
            hook.url = request.data["url"]
        if "events" in request.data:
            hook.events = request.data["events"]
        hook.save()
        return Response(_webhook_dict(hook))

    def delete(self, request, hook_id):
        hook = self._get(request, hook_id)
        if hook is None:
            return Response({"error": "Webhook introuvable."}, status=status.HTTP_404_NOT_FOUND)
        hook.delete()
        return Response({"deleted": 1})


class WebhookTestView(APIView):
    """Fire a sample event to the user's webhooks synchronously and report status."""

    def post(self, request):
        from services.webhooks import dispatch_sync
        results = dispatch_sync(
            request.user,
            "task.completed",
            {"test": True, "task": {"id": 0, "title": "Webhook de test Planner AI"}},
        )
        return Response(
            {"delivered_to": len(results), "results": [{"id": i, "status": s} for i, s in results]}
        )


# ============== Social co-presence ==============
class SocialConnectView(APIView):
    """Send a connection (friend) request by username; auto-accepts a reverse one."""

    def post(self, request):
        from core.models import Connection
        username = (request.data.get('username') or '').strip()
        if not username:
            return Response({'error': 'username requis.'}, status=status.HTTP_400_BAD_REQUEST)
        target = User.objects.filter(username__iexact=username).first()
        if target is None:
            return Response({'error': 'Utilisateur introuvable.'}, status=status.HTTP_404_NOT_FOUND)
        if target.id == request.user.id:
            return Response({'error': 'Impossible de se connecter à soi-même.'}, status=status.HTTP_400_BAD_REQUEST)

        # If they already sent me a request, accept it (mutual).
        reverse_c = Connection.objects.filter(from_user=target, to_user=request.user).first()
        if reverse_c:
            if reverse_c.status != 'accepted':
                reverse_c.status = 'accepted'
                reverse_c.accepted_at = timezone.now()
                reverse_c.save(update_fields=['status', 'accepted_at'])
            return Response({'status': 'accepted', 'connection_id': reverse_c.id})

        conn, created = Connection.objects.get_or_create(from_user=request.user, to_user=target)
        return Response(
            {'status': conn.status, 'connection_id': conn.id},
            status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
        )


class SocialConnectionsView(APIView):
    """List accepted friends + incoming pending requests."""

    def get(self, request):
        from core.models import Connection
        accepted = Connection.objects.filter(status='accepted').filter(
            Q(from_user=request.user) | Q(to_user=request.user)
        ).select_related('from_user', 'to_user')
        friends = []
        for c in accepted:
            other = c.to_user if c.from_user_id == request.user.id else c.from_user
            friends.append({'user_id': other.id, 'username': other.username, 'connection_id': c.id})
        incoming = Connection.objects.filter(
            to_user=request.user, status='pending'
        ).select_related('from_user')
        pending = [
            {'connection_id': c.id, 'from_user_id': c.from_user_id, 'from_username': c.from_user.username}
            for c in incoming
        ]
        return Response({'friends': friends, 'pending_incoming': pending})


class SocialAcceptView(APIView):
    """Accept a received pending connection request."""

    def post(self, request, connection_id):
        from core.models import Connection
        conn = Connection.objects.filter(
            id=connection_id, to_user=request.user, status='pending'
        ).first()
        if conn is None:
            return Response({'error': 'Demande introuvable.'}, status=status.HTTP_404_NOT_FOUND)
        conn.status = 'accepted'
        conn.accepted_at = timezone.now()
        conn.save(update_fields=['status', 'accepted_at'])
        return Response({'status': 'accepted', 'connection_id': conn.id})


class SocialAvailabilityView(APIView):
    """Common free slots with an accepted friend on a date (co-presence)."""

    def get(self, request):
        from core.models import Connection
        from services.social import common_free

        friend_id = request.query_params.get('friend')
        if not friend_id:
            return Response({'error': 'friend requis.'}, status=status.HTTP_400_BAD_REQUEST)
        is_friend = Connection.objects.filter(status='accepted').filter(
            Q(from_user=request.user, to_user_id=friend_id)
            | Q(from_user_id=friend_id, to_user=request.user)
        ).exists()
        if not is_friend:
            return Response(
                {'error': "Cette personne n'est pas dans tes amis."},
                status=status.HTTP_403_FORBIDDEN,
            )
        friend = User.objects.filter(id=friend_id).first()
        if friend is None:
            return Response({'error': 'Introuvable.'}, status=status.HTTP_404_NOT_FOUND)

        date_str = request.query_params.get('date')
        try:
            target = date.fromisoformat(date_str) if date_str else timezone.localdate()
        except ValueError:
            return Response({'error': 'date invalide (YYYY-MM-DD).'}, status=status.HTTP_400_BAD_REQUEST)

        slots = common_free(request.user, friend, target)
        return Response({'friend': friend.username, 'date': str(target), 'common_free': slots})


def _webhook_dict(hook, reveal_secret=False):
    data = {
        "id": hook.id,
        "url": hook.url,
        "events": hook.events,
        "description": hook.description,
        "active": hook.active,
        "created_at": hook.created_at,
        "last_triggered_at": hook.last_triggered_at,
        "last_status": hook.last_status,
        "failure_count": hook.failure_count,
        "has_secret": bool(hook.secret),
    }
    if reveal_secret:
        # Returned once, at creation, so the user can configure their receiver.
        data["secret"] = hook.secret
    return data
