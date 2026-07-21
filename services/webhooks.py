"""
Outbound webhook dispatch (n8n / Zapier / Make / custom).

Fire-and-forget: Planner events POST a signed JSON payload to every matching
WebhookEndpoint. The HTTP call runs in a daemon thread so it never blocks the
request/response cycle (mirrors the async document-processing pattern). Each
payload is HMAC-SHA256 signed with the endpoint's secret so the receiver can
verify authenticity.

Payload shape:
    { "event": "task.completed", "timestamp": "<ISO8601 UTC>", "data": {...} }
Header:
    X-Planner-Event: <event>
    X-Planner-Signature: sha256=<hex hmac of the raw body>
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading

import requests
from django.utils import timezone

logger = logging.getLogger(__name__)

_TIMEOUT = 8  # seconds; receivers (n8n) should ack fast


def _sign(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _post(endpoint_id: int, url: str, secret: str, event: str, body: bytes) -> None:
    from django.db.models import F

    from core.models import WebhookEndpoint

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "PlannerAI-Webhook/1.0",
        "X-Planner-Event": event,
    }
    if secret:
        headers["X-Planner-Signature"] = _sign(secret, body)

    status = "error"
    ok = False
    try:
        resp = requests.post(url, data=body, headers=headers, timeout=_TIMEOUT)
        status = f"{resp.status_code}"
        ok = 200 <= resp.status_code < 300
    except requests.RequestException as e:
        status = f"error: {type(e).__name__}"
        logger.warning("Webhook %s POST failed: %s", endpoint_id, e)

    # Best-effort bookkeeping; never raise back into the caller.
    try:
        updates = {"last_triggered_at": timezone.now(), "last_status": status[:40]}
        WebhookEndpoint.objects.filter(id=endpoint_id).update(**updates)
        if not ok:
            WebhookEndpoint.objects.filter(id=endpoint_id).update(
                failure_count=F("failure_count") + 1
            )
    except Exception:  # pragma: no cover - bookkeeping must not crash the thread
        logger.exception("Webhook %s bookkeeping failed", endpoint_id)


def dispatch(user, event: str, data: dict) -> int:
    """Queue delivery of `event` to all of `user`'s matching endpoints.

    Returns the number of endpoints the event was dispatched to (queued, not
    confirmed delivered). Safe to call from any request path or command.
    """
    from core.models import WebhookEndpoint

    endpoints = [
        e
        for e in WebhookEndpoint.objects.filter(user=user, active=True)
        if e.wants(event)
    ]
    if not endpoints:
        return 0

    payload = {
        "event": event,
        "timestamp": timezone.now().isoformat(),
        "data": data,
    }
    body = json.dumps(payload, default=str, ensure_ascii=False).encode("utf-8")

    for e in endpoints:
        threading.Thread(
            target=_post,
            args=(e.id, e.url, e.secret, event, body),
            daemon=True,
        ).start()

    return len(endpoints)


def dispatch_sync(user, event: str, data: dict) -> list:
    """Synchronous variant for tests / cron where blocking is fine.

    Returns a list of (endpoint_id, status_str). Used by the "test webhook"
    endpoint so the user gets an immediate delivery result.
    """
    from core.models import WebhookEndpoint

    endpoints = [
        e
        for e in WebhookEndpoint.objects.filter(user=user, active=True)
        if e.wants(event)
    ]
    payload = {
        "event": event,
        "timestamp": timezone.now().isoformat(),
        "data": data,
    }
    body = json.dumps(payload, default=str, ensure_ascii=False).encode("utf-8")

    results = []
    for e in endpoints:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "PlannerAI-Webhook/1.0",
            "X-Planner-Event": event,
        }
        if e.secret:
            headers["X-Planner-Signature"] = _sign(e.secret, body)
        try:
            resp = requests.post(e.url, data=body, headers=headers, timeout=_TIMEOUT)
            results.append((e.id, str(resp.status_code)))
            WebhookEndpoint.objects.filter(id=e.id).update(
                last_triggered_at=timezone.now(), last_status=str(resp.status_code)[:40]
            )
        except requests.RequestException as ex:
            results.append((e.id, f"error: {type(ex).__name__}"))
    return results
