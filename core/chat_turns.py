"""Identité et admission HTTP du tour. Aucun verrou ne traverse l'appel agent."""
import hashlib
import json
from datetime import timedelta
from uuid import UUID, uuid4

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from rest_framework.response import Response

from core.models import ChatTurn


def _fingerprint(request):
    attachment = request.FILES.get('attachment')
    file_data = None
    if attachment is not None:
        digest = hashlib.sha256()
        position = attachment.tell()
        for chunk in attachment.chunks():
            digest.update(chunk)
        attachment.seek(position)
        file_data = [attachment.name, attachment.size, attachment.content_type, digest.hexdigest()]
    # SSE diffère toujours les suggestions; ce détail de transport ne change
    # pas l'instruction. Les champs de confirmation, eux, font partie du tour.
    payload = {
        'message': request.data.get('message', ''),
        'document_type': request.data.get('document_type', 'other'),
        'tap_demande': request.data.get('tap_demande', ''),
        'tap_option': request.data.get('tap_option', ''),
        'attachment': file_data,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def refresh_turn(turn):
    # Au-delà du budget maximal, un worker peut être mort. Ne jamais supposer
    # que ses effets ont été annulés et ne jamais réadmettre le même UUID.
    cutoff = timezone.now() - timedelta(seconds=getattr(settings, 'CHAT_TURN_STALE_SECONDS', 900))
    ChatTurn.objects.filter(pk=turn.pk, status__in=['accepted', 'running'], updated_at__lt=cutoff).update(
        status='interrupted', updated_at=timezone.now(),
    )
    turn.refresh_from_db()
    return turn


def snapshot(turn):
    refresh_turn(turn)
    data = {'turn_id': str(turn.turn_id), 'status': turn.status}
    if turn.status in ('completed', 'failed'):
        data['result'] = turn.result
    if turn.status == 'interrupted':
        data['error'] = "Le traitement a été interrompu. Vérifie ton planning avant de formuler une nouvelle demande."
    return data


def replay(turn):
    data = snapshot(turn)
    if turn.status in ('completed', 'failed'):
        return Response(turn.result, status=turn.http_status)
    if turn.status == 'interrupted':
        return Response({**data, 'code': 'turn_interrupted'}, status=409)
    response = Response({**data, 'code': 'turn_pending'}, status=202)
    response['Retry-After'] = '2'
    return response


def admit(request):
    """Retourne (nouveau tour, None) ou (None, réponse du tour déjà admis).

    La contrainte unique arbitre les processus concurrents. Le claim conditionnel
    reste dans une transaction courte, achevée avant l'upload et le modèle.
    Sans UUID client, on génère une identité nouvelle pour compatibilité seulement.
    """
    try:
        turn_id = UUID(str(request.data['turn_id'])) if 'turn_id' in request.data else uuid4()
    except (ValueError, TypeError, AttributeError):
        return None, Response({'code': 'invalid_turn_id', 'error': 'Identifiant de demande invalide.'}, status=400)
    fingerprint = _fingerprint(request)
    with transaction.atomic():
        turn, created = ChatTurn.objects.get_or_create(
            user=request.user, turn_id=turn_id, defaults={'payload_hash': fingerprint},
        )
        if created:
            claimed = ChatTurn.objects.filter(pk=turn.pk, status='accepted').update(
                status='running', updated_at=timezone.now(),
            )
            if claimed:
                turn.status = 'running'
                return turn, None
    if turn.payload_hash != fingerprint:
        return None, Response({
            'turn_id': str(turn_id), 'code': 'turn_payload_mismatch',
            'error': 'Cet identifiant appartient à une autre demande.',
        }, status=409)
    return None, replay(turn)


def finish(turn, result, http_status=200):
    result = {**result, 'turn_id': str(turn.turn_id)}
    persisted = {key: value for key, value in result.items() if key != 'raisonnement'}
    # Un worker encore vivant peut finir après le constat « interrompu ». Son
    # résultat reste récupérable. Il demeure le seul propriétaire de ce tour.
    ChatTurn.objects.filter(pk=turn.pk).update(
        status='completed' if http_status < 400 else 'failed', result=persisted,
        http_status=http_status, updated_at=timezone.now(),
    )
    return result


def fail(turn):
    return finish(turn, {
        'code': 'turn_failed',
        'error': 'Le traitement a échoué. Vérifie ton planning avant de refaire la demande.',
    }, 500)
