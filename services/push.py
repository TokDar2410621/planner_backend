"""
Web Push (VAPID) delivery.

Sends push notifications to a user's stored PushSubscription rows. Expired /
gone subscriptions (404/410) are pruned automatically. Degrades silently when
VAPID keys are not configured.
"""
import json
import logging

from django.conf import settings

logger = logging.getLogger(__name__)

try:
    from pywebpush import WebPushException, webpush
except ImportError:  # pragma: no cover - pywebpush optional at import time
    webpush = None
    WebPushException = Exception


def push_configured() -> bool:
    return bool(
        webpush and settings.VAPID_PRIVATE_KEY and settings.VAPID_PUBLIC_KEY
    )


def _vapid_claims() -> dict:
    return {"sub": settings.VAPID_SUBJECT}


def send_web_push(subscription, title: str, body: str, url: str = "/", extra: dict | None = None) -> bool:
    """Send one push to one PushSubscription. Returns True on success.

    Prunes the subscription on 404/410 (browser unsubscribed / endpoint gone).
    """
    if not push_configured():
        logger.info("Web push not configured; skipping.")
        return False

    payload = {"title": title, "body": body, "url": url}
    if extra:
        payload.update(extra)

    sub_info = {
        "endpoint": subscription.endpoint,
        "keys": {"p256dh": subscription.p256dh, "auth": subscription.auth},
    }
    try:
        webpush(
            subscription_info=sub_info,
            data=json.dumps(payload),
            vapid_private_key=settings.VAPID_PRIVATE_KEY,
            vapid_claims=_vapid_claims(),
        )
        return True
    except WebPushException as e:  # noqa: BLE001
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (404, 410):
            logger.info("Pruning gone push subscription %s", subscription.endpoint[:40])
            subscription.delete()
        else:
            logger.warning("Web push failed (%s): %s", status, e)
        return False
    except Exception as e:  # noqa: BLE001
        logger.error("Web push error: %s", e)
        return False


def send_to_user(user, title: str, body: str, url: str = "/", extra: dict | None = None) -> int:
    """Send a push to every subscription of a user. Returns the count delivered."""
    from core.models import PushSubscription

    sent = 0
    for sub in PushSubscription.objects.filter(user=user):
        if send_web_push(sub, title, body, url=url, extra=extra):
            sent += 1
    return sent
