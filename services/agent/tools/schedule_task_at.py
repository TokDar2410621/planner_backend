"""Outil schedule_task_at."""
from datetime import datetime, time, timedelta
from typing import Annotated, Literal

from django.contrib.auth.models import User
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from core.models import ScheduledBlock, Task
from services.scheduling.overlap import parse_time
from .base import ISSUE_ECHEC, ToolResult
from .commun import DAY_NAMES, _time_to_minutes, _window_conflict

NAME = "schedule_task_at"
DESCRIPTION = (
    "Planifie un événement PONCTUEL daté à une heure précise (ex: 'atelier ce "
    "dimanche 10h-12h', 'rdv mardi 14h-15h'). C'est l'outil pour un événement UNIQUE "
    "sur une date donnée: il crée directement le créneau (verrouillé, la "
    "replanification ne le bouge pas). N'utilise PAS create_block (qui crée une "
    "habitude répétée CHAQUE semaine) pour un événement ponctuel. Un jour nommé "
    "sans mot de récurrence (« repas mercredi 12h », sans « chaque » ni « tous les ») "
    "est un événement unique: c'est cet outil, à la prochaine date de ce jour. Sans heure donnée: "
    "pour une activité souple (lecture, étude, sport), choisis un créneau libre "
    "(find_free_slots); pour un rendez-vous, un cours, un quart, une réunion ou une "
    "leçon fixé par quelqu'un d'autre sans heure de début ou de fin, ou sans date, "
    "n'appelle pas cet outil: demande-lui (present_form ou present_choices). Une heure "
    "donnée par l'utilisateur ne se change jamais sans lui demander. Pour AJUSTER un événement déjà planifié (durée/heure: "
    "'finalement 45 min'), ré-appelle schedule_task_at avec le même titre et la "
    "même date: ça MET À JOUR l'événement (upsert), ce n'est pas un doublon. "
    "Gère aussi un événement qui traverse minuit (ex: quart de nuit PONCTUEL "
    "'ce lundi 22h-06h'). Pour un quart de nuit RÉCURRENT (chaque semaine), "
    "utilise plutôt create_block."
)

# Source de verite : les choix du modele. Le Literal ci-dessous est le
# miroir type du schema ; l'assertion garde les deux synchronises.
VALID_TASK_TYPES = {c[0] for c in Task.TASK_TYPE_CHOICES}
TypeTache = Literal["deep_work", "shallow", "errand"]
assert set(TypeTache.__args__) == VALID_TASK_TYPES, (
    "derive entre le schema type et Task.TASK_TYPE_CHOICES"
)


def _coercer_type_tache(v):
    """Un type invalide (ex: le LLM envoie 'work') vaut 'shallow', pas un rejet.

    Comportement historique de l'outil : la coercition silencieuse évite de
    casser la boucle de l'agent pour une simple étiquette.
    """
    if v is None:
        return None
    return v if v in VALID_TASK_TYPES else "shallow"


TypeTacheSouple = Annotated[TypeTache, BeforeValidator(_coercer_type_tache)]


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(
        description="Titre de l'événement (ex: 'Atelier').",
    )
    date: str = Field(
        description="Date de l'événement (YYYY-MM-DD). Déduis-la de la DATE du jour.",
    )
    start_time: str = Field(
        description="Heure de début (HH:MM).",
    )
    end_time: str = Field(
        description="Heure de fin (HH:MM). Peut passer minuit pour un événement ponctuel qui traverse la nuit (ex: 22:00 -> 06:00).",
    )
    task_type: TypeTacheSouple | None = Field(
        default=None,
        description="Type (optionnel, défaut shallow). Une valeur inconnue vaut shallow.",
    )
    priority: int | None = Field(
        default=None, ge=1, le=10,
        description="Priorité 1-10 (optionnel).",
    )
    description: str | None = Field(
        default=None,
        description="Détails (optionnel).",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    title = (kwargs.get("title") or "").strip()
    if not title:
        return ToolResult(success=False, data={}, message="Titre requis.",
                          public="Je n'ai rien planifié : le titre est manquant. Dis-moi le titre de l'événement.")

    try:
        target_date = datetime.strptime(kwargs["date"], "%Y-%m-%d").date()
    except (ValueError, KeyError, TypeError):
        # Audit 2026-10-13: une date illisible envoyee par le modele est
        # un probleme de preparation interne. Le message public ne doit
        # pas accuser l'utilisateur d'une date qu'il n'a pas saisie.
        return ToolResult(success=False, data={}, message="Format de date invalide. Utilise YYYY-MM-DD.",
                          issue=ISSUE_ECHEC, cause="date_illisible",
                          public="Je n'ai pas pu vérifier ce créneau : la date était illisible.")

    try:
        start_t = parse_time(kwargs["start_time"])
        end_t = parse_time(kwargs["end_time"])
    except (ValueError, KeyError):
        return ToolResult(success=False, data={}, message=(
            f"Heure invalide (attendu HH:MM) : début={kwargs.get('start_time')!r}, "
            f"fin={kwargs.get('end_time')!r}. Corrige le format et réessaie."),
            public=("Je n'ai pas planifié l'événement : l'heure indiquée est invalide. "
                    "Dis-moi l'heure de début et l'heure de fin."))

    s = _time_to_minutes(start_t)
    e = _time_to_minutes(end_t)
    if s == e:
        return ToolResult(success=False, data={}, message="La durée doit être non nulle (fin différente du début).",
                          public="Je n'ai pas planifié l'événement : la fin doit être différente du début. Dis-moi l'heure de fin.")

    # task_type est OPTIONNEL: un type invalide (ex: le LLM envoie 'work' pour
    # un quart de nuit) ne doit PAS faire échouer toute la planification.
    task_type = kwargs.get("task_type") or "shallow"
    if task_type not in VALID_TASK_TYPES:
        task_type = "shallow"

    next_date = target_date + timedelta(days=1)
    # Un événement qui traverse minuit (ex: quart de nuit 22h-06h) est stocké
    # en DEUX morceaux within-day, frontend-safe: [start, 23:59] sur la date +
    # [00:00, end] le lendemain. Chaque morceau garde fin > début (jamais
    # end<=start, que fixed_busy_intervals lirait comme un mur toute la
    # journée). "Fin à minuit exact" (end=00:00) n'est PAS un passage de
    # minuit: c'est un même-jour clampé à 23:59, sans morceau du lendemain.
    crosses_midnight = e < s and e > 0
    main_end_t = end_t if e > s else time(23, 59)
    main_win_end = e if e > s else 24 * 60  # le conflit du soir couvre jusqu'à minuit
    has_main = _time_to_minutes(main_end_t) > s

    # Morceaux verrouillés du MÊME événement, à exclure des conflits / à
    # réconcilier. La queue overnight commence TOUJOURS à 00:00: on la cible
    # précisément pour ne pas toucher un autre événement du lendemain.
    task = Task.objects.filter(user=user, completed=False, title__iexact=title).first()
    existing_main = existing_tail = None
    if task is not None:
        existing_main = ScheduledBlock.objects.filter(
            user=user, task=task, date=target_date, locked=True,
        ).order_by("start_time", "id").first()
        existing_tail = ScheduledBlock.objects.filter(
            user=user, task=task, date=next_date, locked=True, start_time=time(0, 0),
        ).order_by("id").first()

    if has_main:
        c = _window_conflict(user, target_date, s, main_win_end,
                             existing_main.id if existing_main is not None else None)
        if c is not None:
            return c
    if crosses_midnight:
        c = _window_conflict(user, next_date, 0, e,
                             existing_tail.id if existing_tail is not None else None)
        if c is not None:
            return c

    # Réutilise une tâche active du même titre (idempotence), sinon la crée.
    if task is None:
        task = Task.objects.create(
            user=user,
            title=title,
            task_type=task_type,
            priority=kwargs.get("priority", 5),
            description=kwargs.get("description", ""),
        )

    # Morceau principal (jour de départ): met à jour l'existant (garde l'id,
    # upsert N06) sinon crée.
    sb = existing_main
    if sb is not None:
        changed_fields = []
        if sb.start_time != start_t:
            sb.start_time = start_t
            changed_fields.append("start_time")
        if sb.end_time != main_end_t:
            sb.end_time = main_end_t
            changed_fields.append("end_time")
        if not sb.locked:
            sb.locked = True
            changed_fields.append("locked")
        if changed_fields:
            sb.save(update_fields=changed_fields)
    else:
        sb = ScheduledBlock.objects.create(
            user=user, task=task, date=target_date,
            start_time=start_t, end_time=main_end_t, locked=True,
        )

    # Morceau du lendemain UNIQUEMENT si l'événement traverse minuit. Sinon on
    # SUPPRIME une éventuelle queue orpheline d'une version overnight
    # précédente (sinon bloc fantôme verrouillé le matin suivant, immunisé au
    # replan).
    if crosses_midnight:
        if existing_tail is not None:
            if existing_tail.end_time != end_t:
                existing_tail.end_time = end_t
                existing_tail.save(update_fields=["end_time"])
        else:
            ScheduledBlock.objects.create(
                user=user, task=task, date=next_date,
                start_time=time(0, 0), end_time=end_t, locked=True,
            )
    elif existing_tail is not None:
        existing_tail.delete()

    span = f" (traverse minuit, jusqu'au {DAY_NAMES[next_date.weekday()]})" if crosses_midnight else ""
    return ToolResult(
        success=True,
        data={"scheduled_block": {
            "id": sb.id,
            "title": task.title,
            "date": target_date.isoformat(),
            "start_time": start_t.strftime("%H:%M"),
            "end_time": end_t.strftime("%H:%M"),
            "overnight": crosses_midnight,
        }},
        message=(
            f"'{task.title}' planifié le {DAY_NAMES[target_date.weekday()]} "
            f"{target_date.isoformat()} de {start_t.strftime('%H:%M')} à "
            f"{end_t.strftime('%H:%M')}{span}."
        ),
    )
