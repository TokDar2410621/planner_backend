"""Cerveau quotidien proactif.

Transforme Planner d'« assistant qui répond » en « assistant qui pilote » : une
fois par jour et par utilisateur, il lit l'état réel du planning et en tire les
1-3 choses qui méritent l'attention — sans qu'aucun message ne soit envoyé par
l'utilisateur :

  1. Pardon structurel : les blocs passés non complétés sont reprogrammés
     automatiquement (roll_over_missed — zéro rouge, zéro dette visible).
  2. Conflits à venir (3 jours) via le détecteur produit (fixe/fixe seulement).
  3. Échéances proches : tâches (72 h) et objectifs actifs (7 jours).

Le brief est calculé UNE fois par (user, jour) et mis en cache dans
DailyBrainReport (idempotent, y compris entre le worker et l'endpoint GET
/insights/daily-brief/). Le push n'est envoyé que s'il y a quelque chose
d'actionnable : le silence vaut mieux que le bruit.
"""
import logging
from datetime import timedelta

from django.db import IntegrityError
from django.utils import timezone

from core.models import DailyBrainReport, Goal, Task
from services.push import push_configured, send_to_user
from services.rollover import roll_over_missed

logger = logging.getLogger(__name__)

MAX_ITEMS = 3


def build_daily_brief(user, target_date=None) -> dict:
    """Calcule (ou relit) le brief du jour pour ``user``. Idempotent par (user, date).

    ATTENTION effet de bord assumé (aligné produit « rétention lot 1 ») : le
    premier calcul du jour APPLIQUE le roll-over des blocs ratés — comme le fait
    déjà POST /schedule/rollover/ à l'ouverture de l'app.
    """
    today = target_date or timezone.localdate()
    existing = DailyBrainReport.objects.filter(user=user, date=today).first()
    if existing:
        return existing.payload

    items = []

    # 1. Pardon structurel : reprogramme les blocs ratés.
    try:
        roll = roll_over_missed(user)
        if roll.get("rolled"):
            items.append({
                "type": "rollover",
                "text": f"{roll['rolled']} bloc(s) raté(s) reprogrammé(s) automatiquement.",
            })
    except Exception:  # noqa: BLE001 - un signal raté ne tue pas le brief
        logger.warning("DailyBrain rollover failed for user %s", user.id, exc_info=True)

    # 2. Conflits à venir (3 jours) — le détecteur produit (fixe/fixe seulement).
    try:
        from services.ai_insights import AIInsightsService
        conflicts = AIInsightsService().detect_conflicts(user, days_ahead=3)
        if conflicts:
            items.append({
                "type": "conflict",
                "text": (f"{len(conflicts)} conflit(s) d'horaire dans les 3 prochains "
                         f"jours: {conflicts[0].message}"),
            })
    except Exception:  # noqa: BLE001
        logger.warning("DailyBrain conflicts failed for user %s", user.id, exc_info=True)

    # 3. Échéances proches : tâches (72 h) puis objectifs actifs (7 jours).
    now = timezone.now()
    due_tasks = (Task.objects
                 .filter(user=user, completed=False, deadline__isnull=False,
                         deadline__gte=now, deadline__lte=now + timedelta(days=3))
                 .order_by("deadline")[:MAX_ITEMS])
    for t in due_tasks:
        local = timezone.localtime(t.deadline)
        items.append({
            "type": "deadline",
            "text": f"Échéance proche: « {t.title} » le {local.strftime('%d/%m à %H:%M')}.",
        })
    due_goals = (Goal.objects
                 .filter(user=user, status="active", deadline__isnull=False,
                         deadline__gte=today, deadline__lte=today + timedelta(days=7))
                 .order_by("deadline")[:2])
    for g in due_goals:
        items.append({
            "type": "goal",
            "text": (f"Objectif « {g.title} » à échéance le {g.deadline.strftime('%d/%m')} "
                     f"(progression {g.progress}%)."),
        })

    payload = {"date": today.isoformat(), "items": items[:MAX_ITEMS + 2]}
    try:
        DailyBrainReport.objects.create(user=user, date=today, payload=payload)
    except IntegrityError:
        # Course worker/endpoint: l'autre écrivain a gagné, sa version fait foi.
        existing = DailyBrainReport.objects.filter(user=user, date=today).first()
        if existing:
            return existing.payload
    return payload


def push_daily_brief(user, payload) -> int:
    """Pousse le brief s'il est actionnable. Retourne le nombre de devices touchés."""
    items = payload.get("items") or []
    if not items or not push_configured():
        return 0
    title = "Planner — ton brief du jour"
    body = " • ".join(i.get("text", "") for i in items[:MAX_ITEMS])[:180]
    try:
        return send_to_user(user, title, body, url="/chat")
    except Exception:  # noqa: BLE001 - le push ne doit jamais casser le worker
        logger.warning("DailyBrain push failed for user %s", user.id, exc_info=True)
        return 0
