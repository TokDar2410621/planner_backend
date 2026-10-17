"""Outil cancel_scheduled_block."""
from datetime import datetime, time, timedelta

from django.contrib.auth.models import User
from pydantic import BaseModel, ConfigDict, Field

from core.models import ScheduledBlock, Task
from .base import ToolResult

NAME = "cancel_scheduled_block"
DESCRIPTION = (
    "Annule/supprime un événement PONCTUEL déjà planifié à une date précise "
    "(ex: 'annule mon rdv optométriste','enlève mon étude de demain', 'supprime "
    "le prochain événement'). C'est l'outil pour un one-off créé par "
    "schedule_task_at. NE l'utilise PAS pour un bloc RÉCURRENT: pour sauter UN "
    "jour d'une série -> skip_block_occurrence; pour supprimer toute la série -> "
    "delete_block. Résous par date (+ titre si plusieurs événements ce jour-là); "
    "ne demande jamais d'identifiant. Si plusieurs événements distincts existent "
    "ce jour-là sans titre fourni, l'outil renvoie la liste et te demande lequel."
)


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: str = Field(
        description="Date de l'événement (YYYY-MM-DD).",
    )
    title: str | None = Field(
        default=None,
        description="Titre pour lever l'ambiguïté si plusieurs événements ce jour-là (optionnel).",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    try:
        target_date = datetime.strptime(kwargs["date"], "%Y-%m-%d").date()
    except (ValueError, KeyError, TypeError):
        return ToolResult(success=False, data={}, message="Format de date invalide (attendu YYYY-MM-DD).",
                          public="Je n'ai rien annulé : la date est invalide. Utilise le format AAAA-MM-JJ.")

    title = (kwargs.get("title") or "").strip()
    qs = ScheduledBlock.objects.filter(user=user, date=target_date).select_related("task")
    if title:
        qs = qs.filter(task__title__icontains=title)
    blocks = list(qs)
    if not blocks:
        suffix = f" pour « {title} »" if title else ""
        return ToolResult(
            success=False, data={},
            message=f"Aucun événement ponctuel trouvé le {target_date.isoformat()}{suffix}.",
            public="Je n'ai rien annulé : aucun événement ponctuel ce jour-là. Vérifie la date.",
        )

    distinct = sorted({(b.task.title if b.task else "") for b in blocks})
    if len(distinct) > 1 and not title:
        listing = ", ".join(t for t in distinct if t)
        return ToolResult(
            success=False, data={"candidates": distinct},
            message=f"Plusieurs événements le {target_date.isoformat()}: {listing}. Lequel veux-tu annuler ?",
        )

    # Inclut la queue d'un événement ponctuel qui traverse minuit (morceau
    # 00:00 du lendemain de la même tâche), pour ne pas laisser de résidu.
    task_ids = {b.task_id for b in blocks if b.task_id}
    tail = ScheduledBlock.objects.filter(
        user=user, date=target_date + timedelta(days=1),
        task_id__in=task_ids, start_time=time(0, 0),
    )
    blocks += list(tail)

    cancelled = []
    for b in blocks:
        cancelled.append({
            "title": (b.task.title if b.task else ""),
            "date": b.date.isoformat(),
            "start_time": b.start_time.strftime("%H:%M"),
            "end_time": b.end_time.strftime("%H:%M"),
        })
        b.delete()

    # Supprime la tâche devenue orpheline (one-off pur: plus aucun créneau,
    # non complétée). Une tâche encore planifiée ailleurs ou complétée reste.
    for tid in task_ids:
        t = Task.objects.filter(id=tid, user=user).first()
        if t and not t.completed and not ScheduledBlock.objects.filter(task=t).exists():
            t.delete()

    label = cancelled[0]["title"] or "événement"
    extra = f" ({len(cancelled)} créneaux)" if len(cancelled) > 1 else ""
    return ToolResult(
        success=True,
        data={"cancelled": cancelled},
        message=f"Annulé: {label} le {target_date.isoformat()}{extra}.",
    )
