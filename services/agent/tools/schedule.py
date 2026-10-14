"""
Schedule query tools for the Planner AI agent.
"""
from datetime import date, datetime, time, timedelta

from django.contrib.auth.models import User
from django.db import transaction
from django.utils import timezone

from core.models import RecurringBlock, ScheduledBlock, Task
from services.scheduling.day_view import effective_day_entries
from services.scheduling.occurrences import recurring_occurrences
from services.scheduling.overlap import parse_time
from services.scheduling.placement import (
    fixed_busy_intervals,
    intervalles_sommeil_reporte,
    occupied_intervals,
    open_intervals,
    place_day,
)
from services.scheduling.solve_day import (
    empreinte_entrees_jour, empreinte_entrees_semaine, solve_day, solve_placement,
)
from .base import BaseTool, ToolResult, validate_choice
from .base import (ISSUE_TERMINEE, ISSUE_BESOIN_PRECISION, ISSUE_ECHEC)

DAY_NAMES = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
VALID_TASK_TYPES = {c[0] for c in Task.TASK_TYPE_CHOICES}

# Fenêtre "éveillée" par défaut pour les créneaux libres proposés (7h-23h).
DAY_START_MIN = 7 * 60
DAY_END_MIN = 23 * 60


def _time_to_minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def _minutes_to_str(minutes: int) -> str:
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def _free_slots_from_intervals(intervals, min_duration: int) -> list:
    """(start,end)-minutes -> dicts de créneaux libres, filtrés par durée min."""
    out = []
    for s, e in intervals:
        duration = e - s
        if duration >= min_duration:
            out.append({
                "start_time": _minutes_to_str(s),
                "end_time": _minutes_to_str(e),
                "duration_minutes": duration,
            })
    return out


class GetTodayScheduleTool(BaseTool):
    name = "get_today_schedule"
    description = "Récupère le planning EFFECTIF d'un jour (blocs récurrents aux heures PLACÉES, tâches planifiées, créneaux libres). Consulte-le avant d'affirmer où se trouve une activité ou si elle a bougé, et parle des heures effectives, jamais de mémoire ni d'après l'historique (un bloc souple peut être placé à une autre heure que son heure habituelle)."
    parameters = {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "Date au format YYYY-MM-DD (défaut: aujourd'hui)",
            },
        },
        "required": [],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        target_date_str = kwargs.get("date")
        if target_date_str:
            try:
                target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
            except ValueError:
                return ToolResult(success=False, data={}, message="Format de date invalide. Utilise YYYY-MM-DD.",
                              issue=ISSUE_ECHEC, cause="date_illisible",
                              public="Je n'ai pas pu vérifier ce créneau : la date était illisible.")
        else:
            target_date = timezone.localdate()

        # Day of week (0=Monday)
        day_of_week = target_date.weekday()
        day_name = DAY_NAMES[day_of_week]

        blocks = effective_day_entries(user, target_date)

        # Créneaux libres 7h-23h via la logique unique overnight-aware (compte
        # correctement un quart de nuit du jour ET le débordement de la veille,
        # + les blocs déjà planifiés). Corrige le faux "libre 7h-23h".
        intervals = open_intervals(user, target_date, DAY_START_MIN, DAY_END_MIN)
        # AUJOURD'HUI, le passé n'est pas libre, comme dans find_free_slots.
        # Vécu (banc du 2026-09-15, 16:31): la ligne « Libre : 15 h 50 à 18 h »
        # de cette lecture est devenue un bouton « 15 h 50 à 18 h ».
        now_local = timezone.localtime()
        if target_date == now_local.date():
            now_min = ((now_local.hour * 60 + now_local.minute + 4) // 5) * 5
            intervals = [(max(s, now_min), e) for s, e in intervals if e > now_min]
        free_slots = _free_slots_from_intervals(intervals, 30)

        return ToolResult(
            success=True,
            data={
                "date": target_date.isoformat(),
                "day_name": day_name,
                "blocks": blocks,
                "free_slots": free_slots,
                "total_blocks": len(blocks),
                "total_free_minutes": sum(s["duration_minutes"] for s in free_slots),
            },
            message=f"Planning du {day_name} {target_date.isoformat()}: {len(blocks)} bloc(s), {len(free_slots)} créneau(x) libre(s).",
        )


class GetWeekScheduleTool(BaseTool):
    name = "get_week_schedule"
    description = "Récupère un résumé du planning de la semaine : nombre de blocs par jour, heures occupées, et jours les plus chargés."
    parameters = {
        "type": "object",
        "properties": {
            "start_date": {
                "type": "string",
                "description": "Date de début de la semaine (YYYY-MM-DD, défaut: lundi de la semaine en cours)",
            },
        },
        "required": [],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        start_str = kwargs.get("start_date")
        if start_str:
            try:
                start = datetime.strptime(start_str, "%Y-%m-%d").date()
            except ValueError:
                return ToolResult(success=False, data={}, message="Format de date invalide.")
        else:
            today = timezone.localdate()
            start = today - timedelta(days=today.weekday())  # Monday

        days_summary = []
        total_hours = 0

        for i in range(7):
            day = start + timedelta(days=i)
            day_of_week = day.weekday()

            detail = effective_day_entries(user, day)

            day_minutes = 0
            block_list = []
            for b in detail:
                start_h, start_m = map(int, b["start_time"].split(":"))
                end_h, end_m = map(int, b["end_time"].split(":"))
                start_min = start_h * 60 + start_m
                end_min = end_h * 60 + end_m
                duration = end_min - start_min if end_min > start_min else (24 * 60 - start_min + end_min)
                day_minutes += duration
                block_list.append(f"{b['title']} ({b['start_time']}-{b['end_time']})")

            total_hours += day_minutes / 60

            days_summary.append({
                "date": day.isoformat(),
                "day_name": DAY_NAMES[day_of_week],
                "block_count": len(detail),
                "occupied_hours": round(day_minutes / 60, 1),
                "blocks": block_list,
                "detail": detail,
            })

        return ToolResult(
            success=True,
            data={
                "week_start": start.isoformat(),
                "days": days_summary,
                "total_hours": round(total_hours, 1),
            },
            message=f"Semaine du {start.isoformat()}: {round(total_hours, 1)}h planifiées sur 7 jours.",
        )


class FindFreeSlotsTool(BaseTool):
    name = "find_free_slots"
    description = "Trouve les créneaux libres d'un jour (overnight-aware: un quart de nuit occupe la soirée, le travail de la veille occupe le matin), avec une durée minimum optionnelle. Appelle-le AVANT de déclarer un jour 'libre' ou de choisir un créneau toi-même. Activité souple sans heure (lecture, étude, sport): choisis un créneau libre. Pour un rendez-vous, un cours, un quart, une réunion ou une leçon fixé par quelqu'un d'autre sans heure de début ou de fin, ou sans date: ne choisis pas à sa place, demande-lui (present_form ou present_choices)."
    parameters = {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "Date au format YYYY-MM-DD",
            },
            "min_duration_minutes": {
                "type": "integer",
                "description": "Durée minimum du créneau en minutes (défaut: 30)",
                "minimum": 15,
            },
        },
        "required": ["date"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        try:
            target_date = datetime.strptime(kwargs["date"], "%Y-%m-%d").date()
        except ValueError:
            return ToolResult(success=False, data={}, message="Format de date invalide.")

        day_of_week = target_date.weekday()
        min_duration = kwargs.get("min_duration_minutes", 30)

        # Créneaux libres 7h-23h via la logique unique overnight-aware: un quart
        # de nuit (ex: 19h-07h) occupe bien la soirée, le débordement de la veille
        # occupe le matin, et les blocs déjà planifiés comptent aussi.
        intervals = open_intervals(user, target_date, DAY_START_MIN, DAY_END_MIN)

        # AUJOURD'HUI, le passé n'est pas libre. Vécu (2026-08-13, 19:48):
        # l'agent proposait « 07:00 à 08:00 » pour une tâche « dans 6
        # minutes ». On rogne à l'heure murale courante, arrondie aux 5 min
        # supérieures; le filtre de durée minimum s'applique après rognage.
        now_local = timezone.localtime()
        if target_date == now_local.date():
            now_min = ((now_local.hour * 60 + now_local.minute + 4) // 5) * 5
            intervals = [(max(s, now_min), e) for s, e in intervals if e > now_min]

        free = _free_slots_from_intervals(intervals, min_duration)

        return ToolResult(
            success=True,
            data={
                "date": target_date.isoformat(),
                "day_name": DAY_NAMES[day_of_week],
                "free_slots": free,
                "total_free_minutes": sum(s["duration_minutes"] for s in free),
            },
            message=f"{len(free)} créneau(x) libre(s) le {DAY_NAMES[day_of_week]} {target_date.isoformat()} (min {min_duration}min).",
        )


def _first_conflicting_item(user, target_date, s, e, exclude_scheduled_id=None):
    """(titre, start_min, end_min) du premier élément REEL qui chevauche
    [s,e], ou None. fixed_busy_intervals renvoie l'union FUSIONNÉE de la
    journée (vécu: refus « chevauche 08:00-21:30 » pour un trou de 15 min,
    illisible et faux): on nomme l'occupation précise à la place."""
    from django.db.models import Q
    from core.models import RecurringBlock, ScheduledBlock
    items = []
    # Meme regle que fixed_busy_intervals: une tache terminee n'occupe plus
    # son creneau, donc elle ne peut pas non plus etre NOMMEE comme conflit.
    qs = ScheduledBlock.objects.filter(user=user, date=target_date).exclude(
        Q(actually_completed=True) | Q(task__completed=True)
    ).select_related('task')
    if exclude_scheduled_id is not None:
        qs = qs.exclude(id=exclude_scheduled_id)
    for sb in qs:
        bs = sb.start_time.hour * 60 + sb.start_time.minute
        be = sb.end_time.hour * 60 + sb.end_time.minute
        if be <= bs:
            be = 24 * 60
        items.append((sb.task.title, bs, be))
    for b in recurring_occurrences(user, target_date):
        if b.is_flexible and not b.occurrence_override:
            continue
        bs = b.start_time.hour * 60 + b.start_time.minute
        be = b.end_time.hour * 60 + b.end_time.minute
        if be <= bs:
            be = 24 * 60
        items.append((b.title, bs, be))
    hits = [it for it in items if s < it[2] and it[1] < e]
    return min(hits, key=lambda it: it[1]) if hits else None


def _window_conflict(user, target_date, s, e, exclude_scheduled_id=None):
    """ToolResult d'erreur si la fenêtre [s,e] (minutes) du jour target_date
    chevauche un mur fixe ou le sommeil protégé; sinon None. Réutilisable pour
    chaque moitié d'un événement ponctuel qui traverse minuit."""
    for bs, be in fixed_busy_intervals(user, target_date, exclude_scheduled_id=exclude_scheduled_id):
        if s < be and bs < e:
            item = _first_conflicting_item(user, target_date, s, e, exclude_scheduled_id)
            titre = None
            if item:
                title, ibs, ibe = item
                label = f"« {title} » ({_minutes_to_str(ibs)}-{_minutes_to_str(ibe)})"
                bs, be = ibs, ibe
                titre = title
            else:
                label = f"une occupation existante ({_minutes_to_str(bs)}-{_minutes_to_str(be)})"
            return ToolResult(
                success=False,
                data={"conflict": {
                    "start_time": _minutes_to_str(bs),
                    "end_time": _minutes_to_str(be),
                    "titre": titre,
                    "sommeil": False,
                }},
                message=(
                    f"Ce créneau ({_minutes_to_str(s)}-{_minutes_to_str(e)}) chevauche "
                    f"{label}. Choisis un autre horaire libre."
                ),
            )
    # Le sommeil reporte au lendemain d'un quart de nuit (placement.py) occupe
    # le matin de ce jour: il est protege comme le sommeil place.
    for bs, be in intervalles_sommeil_reporte(user, target_date):
        if s < be and bs < e:
            return ToolResult(
                success=False,
                data={"conflict": {
                    "start_time": _minutes_to_str(bs),
                    "end_time": _minutes_to_str(be),
                    "titre": None,
                    "sommeil": True,
                }},
                message=(
                    f"Ce créneau ({_minutes_to_str(s)}-{_minutes_to_str(e)}) "
                    f"chevauche le sommeil protégé "
                    f"({_minutes_to_str(bs)}-{_minutes_to_str(be)}). "
                    f"Choisis un autre horaire libre."
                ),
            )
    for placement in place_day(user, target_date):
        if placement["skipped"] or placement["block_type"] != "sleep":
            continue
        if placement.get("occurrence_override"):
            continue
        ps = placement["start_min"]
        pe = placement["end_min"]
        if ps is None or pe is None:
            continue
        sleep_pieces = [(ps, pe)] if pe > ps else [(ps, 24 * 60), (0, pe)]
        for bs, be in sleep_pieces:
            if s < be and bs < e:
                conflict_start, conflict_end = bs, be
                for os, oe in occupied_intervals(user, target_date, 0, 24 * 60):
                    if s < oe and os < e:
                        conflict_start, conflict_end = os, oe
                        break
                return ToolResult(
                    success=False,
                    data={"conflict": {
                        "start_time": _minutes_to_str(conflict_start),
                        "end_time": _minutes_to_str(conflict_end),
                        "titre": None,
                        "sommeil": True,
                    }},
                    message=(
                        f"Ce créneau ({_minutes_to_str(s)}-{_minutes_to_str(e)}) "
                        f"chevauche le sommeil protégé "
                        f"({_minutes_to_str(conflict_start)}-{_minutes_to_str(conflict_end)}). "
                        f"Choisis un autre horaire libre."
                    ),
                )
    return None


class ScheduleTaskAtTool(BaseTool):
    name = "schedule_task_at"
    description = (
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
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Titre de l'événement (ex: 'Atelier')."},
            "date": {"type": "string", "description": "Date de l'événement (YYYY-MM-DD). Déduis-la de la DATE du jour."},
            "start_time": {"type": "string", "description": "Heure de début (HH:MM)."},
            "end_time": {"type": "string", "description": "Heure de fin (HH:MM). Peut passer minuit pour un événement ponctuel qui traverse la nuit (ex: 22:00 -> 06:00)."},
            "task_type": {
                "type": "string",
                "enum": ["deep_work", "shallow", "errand"],
                "description": "Type (optionnel, défaut shallow).",
            },
            "priority": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Priorité 1-10 (optionnel)."},
            "description": {"type": "string", "description": "Détails (optionnel)."},
        },
        "required": ["title", "date", "start_time", "end_time"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        title = (kwargs.get("title") or "").strip()
        if not title:
            return ToolResult(success=False, data={}, message="Titre requis.")

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
            return ToolResult(success=False, data={}, message="Heure invalide (attendu HH:MM).")

        s = _time_to_minutes(start_t)
        e = _time_to_minutes(end_t)
        if s == e:
            return ToolResult(success=False, data={}, message="La durée doit être non nulle (fin différente du début).")

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


class CheckFeasibilityTool(BaseTool):
    name = "check_feasibility"
    description = (
        "Vérifie EXACTEMENT (solveur) si un ensemble d'activités tient dans une "
        "journée donnée, autour de l'emploi du temps existant (murs fixes + "
        "sommeil protégé). À utiliser AVANT d'affirmer qu'une charge 'tient', "
        "surtout sur une journée chargée: ne juge JAMAIS la faisabilité de tête. "
        "Renvoie ce qui rentre (avec un créneau suggéré) et ce qui ne rentre pas. "
        "Ex: 3h libres fragmentées en 3x1h ne peuvent PAS accueillir un bloc de 2h."
    )
    parameters = {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "Jour à tester (YYYY-MM-DD). Déduis-le de la DATE du jour."},
            "activities": {
                "type": "array",
                "description": "Activités à faire tenir ce jour-là.",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Nom de l'activité."},
                        "duration_minutes": {"type": "integer", "description": "Durée en minutes."},
                    },
                    "required": ["title", "duration_minutes"],
                },
            },
        },
        "required": ["date", "activities"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        try:
            target_date = datetime.strptime(kwargs["date"], "%Y-%m-%d").date()
        except (ValueError, KeyError, TypeError):
            return ToolResult(success=False, data={}, message="Format de date invalide. Utilise YYYY-MM-DD.",
                              issue=ISSUE_ECHEC, cause="date_illisible",
                              public="Je n'ai pas pu vérifier ce créneau : la date était illisible.")

        extra = []
        for a in kwargs.get("activities") or []:
            if not isinstance(a, dict):
                continue
            try:
                dur = int(a.get("duration_minutes") or 0)
            except (TypeError, ValueError):
                dur = 0
            if dur <= 0:
                continue
            extra.append({"title": a.get("title") or "Activité", "duration_minutes": dur})
        if not extra:
            return ToolResult(success=False, data={}, message="Aucune activité valide à tester (titre + durée en minutes).",
                              issue=ISSUE_BESOIN_PRECISION, cause="activites_invalides",
                              public="Je n'ai rien à vérifier : il me faut au moins une activité avec un titre et une durée.")

        # Fenêtre éveillée 07h-23h (comme find_free_slots) pour ne pas proposer un
        # créneau en pleine nuit; le sommeil reste un mur si l'utilisateur en a un.
        result = solve_day(user, target_date, extra=extra,
                           day_start=DAY_START_MIN, day_end=DAY_END_MIN)
        placed = [
            {"title": p["title"], "start_time": _minutes_to_str(p["start_min"]), "end_time": _minutes_to_str(p["end_min"])}
            for p in result["placements"] if p["kind"] == "extra"
        ]
        unplaced = [
            {"title": x["title"], "duration_minutes": x["duration"]}
            for x in result["unplaced"] if x["key"].startswith("extra:")
        ]
        day_name = DAY_NAMES[target_date.weekday()]
        placed_str = ", ".join(f"{p['title']} {p['start_time']}-{p['end_time']}" for p in placed) or "rien"
        # C'est une VÉRIFICATION: rien n'est créé. L'agent doit ensuite créer le
        # faisable (schedule_task_at/create_block), pas dire "j'ai planifié".
        head = "Vérification (rien n'est encore créé):"
        if result["feasible"]:
            msg = f"{head} tout tient le {day_name} {target_date.isoformat()}: {placed_str}. Crée-le avec schedule_task_at."
        else:
            unplaced_str = ", ".join(f"{x['title']} ({x['duration_minutes']}min)" for x in unplaced)
            msg = (
                f"{head} tout ne tient PAS le {day_name} {target_date.isoformat()}. "
                f"Rentre: {placed_str}. Ne rentre pas: {unplaced_str}. Propose une option (créer ce qui rentre, autre jour, durée réduite)."
            )
        # Contrat d'issue (audit 2026-10-13): le `msg` ci-dessus est deja
        # ecrit pour etre lu, on le declare affichable tel quel.
        public = (
            f"Vérification du {day_name} {target_date.isoformat()} : "
            + (f"tout rentre ({placed_str})."
               if result["feasible"] else
               f"tout ne rentre pas. Rentre : {placed_str}. "
               f"Ne rentre pas : {unplaced_str}.")
        )
        return ToolResult(
            success=True,
            data={
                "feasible": result["feasible"],
                "placed": placed,
                "unplaced": unplaced,
                "date": target_date.isoformat(),
            },
            message=msg,
            issue=ISSUE_TERMINEE, cause="", public=public,
        )


def _mettre_en_forme_jour(arrangement):
    """Decoupe + listing d'un arrangement existant (sans solveur)."""
    placed = [r for r in arrangement if not r["skipped"] and not r["overnight_kept"]]
    overnight = [r for r in arrangement if r["overnight_kept"]]
    skipped = [r for r in arrangement if r["skipped"] and not r.get("reporte_au_lendemain")]
    listing = ", ".join(f"{r['title']} {r['start_time']}-{r['end_time']}" for r in placed) or "rien à replacer"
    return placed, overnight, skipped, listing


def _executer_organize_day(user, target_date, apply, arrangement, expected_entries=None):
    """Corps d'OrganizeDayTool sur un arrangement (deja resolu par l'appelant).

    Séparé du solveur pour réutiliser la proposition mise en cache.
    """
    placed, overnight, skipped, listing = _mettre_en_forme_jour(arrangement)
    head = "Proposition (rien changé)"
    msg = f"{head} le {DAY_NAMES[target_date.weekday()]} {target_date.isoformat()}: {listing}."
    if skipped:
        msg += " Non placé (journée trop pleine): " + ", ".join(r["title"] for r in skipped) + "."
    moved = []
    if apply:
        try:
            moved = _appliquer_periode(user, target_date, [arrangement], expected_entries)[0]
        except ValueError as exc:
            return ToolResult(success=False, data={"applied": False, "reason": str(exc),
                                                   "date": target_date.isoformat()},
                              message="Cet arrangement ne peut plus être appliqué tel quel. Rien n'a changé. Demande une nouvelle proposition.")
        placed, overnight, skipped, listing = _mettre_en_forme_jour(arrangement)
        msg = f"Journée réorganisée le {target_date.isoformat()}: {listing}."
    return ToolResult(
        success=True,
        data={
            "applied": apply,
            "date": target_date.isoformat(),
            "placed": [{"block_id": r["block_id"], "title": r["title"], "start_time": r["start_time"], "end_time": r["end_time"]} for r in placed],
            "overnight_kept": [{"block_id": r["block_id"], "title": r["title"], "start_time": r["start_time"], "end_time": r["end_time"]} for r in overnight],
            "skipped": [{"title": r["title"]} for r in skipped],
            "moved": moved,
            "moved_count": len(moved),
        },
        message=msg,
    )


class OrganizeDayTool(BaseTool):
    name = "organize_day"
    description = (
        "Propose un placement des blocs SOUPLES "
        "d'une journée (sport, révision, repas, sommeil...) autour des murs fixes "
        "(cours, travail) et du sommeil protégé, via un solveur exact. À utiliser "
        "quand l'utilisateur demande d'optimiser / réorganiser / mieux agencer sa "
        "journée, ou quand le placement laisse des trous. apply=false PROPOSE "
        "l'arrangement sans rien changer (défaut). Après confirmation, apply=true "
        "enregistre ces horaires pour cette date seulement et conserve la série."
    )
    parameters = {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "Jour à réorganiser (YYYY-MM-DD). Déduis-le de la DATE du jour."},
            "apply": {"type": "boolean", "description": "false = proposer (défaut). true = appliquer à cette date après confirmation."},
        },
        "required": ["date"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        try:
            target_date = datetime.strptime(kwargs["date"], "%Y-%m-%d").date()
        except (ValueError, KeyError, TypeError):
            return ToolResult(success=False, data={}, message="Format de date invalide. Utilise YYYY-MM-DD.",
                              issue=ISSUE_ECHEC, cause="date_illisible",
                              public="Je n'ai pas pu vérifier ce créneau : la date était illisible.")
        apply = bool(kwargs.get("apply", False))
        expected_entries = kwargs.pop("_expected_entries", None) or empreinte_entrees_jour(user, target_date)
        # Kwarg PRIVE (agent v2): arrangement deja resolu et valide. Evite de
        # re-resoudre a l'apply quand le plan vient d'etre verifie a
        # l'identique. Jamais fourni par le modele ni v1.
        arrangement = kwargs.pop("_arrangement", None)
        if arrangement is None:
            arrangement = solve_placement(user, target_date)  # fenêtre pleine journée
        return _executer_organize_day(user, target_date, apply, arrangement, expected_entries)


def _mettre_en_forme_semaine(user, start, arrangements):
    """Mise en forme (days_data, lines, skipped_total) d'arrangements existants.

    Meme code que la boucle de _resoudre_semaine, sans le solveur: sert quand
    l'agent v2 fournit un arrangement deja resolu et valide (_arrangements).
    """
    days_data = []
    lines = []
    skipped_total = 0
    for offset, arrangement in enumerate(arrangements):
        day = start + timedelta(days=offset)
        placed = [r for r in arrangement if not r["skipped"] and not r["overnight_kept"]]
        overnight = [r for r in arrangement if r["overnight_kept"]]
        skipped = [r for r in arrangement if r["skipped"] and not r.get("reporte_au_lendemain")]

        skipped_total += len(skipped)
        days_data.append({
            "date": day.isoformat(),
            "placed": [{"block_id": r["block_id"], "title": r["title"], "start_time": r["start_time"], "end_time": r["end_time"]} for r in placed],
            "overnight_kept": [{"block_id": r["block_id"], "title": r["title"], "start_time": r["start_time"], "end_time": r["end_time"]} for r in overnight],
            "skipped": [{"title": r["title"]} for r in skipped],
            "moved": [],
        })
        if placed or skipped:
            seg = ", ".join(f"{r['title']} {r['start_time']}-{r['end_time']}" for r in placed) or "aucun bloc"
            line = f"{DAY_NAMES[day.weekday()]} {seg}"
            if skipped:
                line += " (non placé: " + ", ".join(r["title"] for r in skipped) + ")"
            lines.append(line)
    return days_data, lines, skipped_total


def _resoudre_semaine(user, start):
    """Les 7 solve_placement de la semaine + mise en forme.

    Retourne (days_data, lines, skipped_total, arrangements). Les arrangements
    bruts (avec block_id) sont conserves pour l'apply et pour le cache v2;
    days_data seul ne suffirait pas (pas de block_id).
    """
    arrangements = [solve_placement(user, start + timedelta(days=offset)) for offset in range(7)]
    days_data, lines, skipped_total = _mettre_en_forme_semaine(user, start, arrangements)
    return days_data, lines, skipped_total, arrangements


def _appliquer_periode(user, start, arrangements, expected_entries=None):
    """Valide puis écrit atomiquement les occurrences, et relit le résultat."""
    from core.models import RecurringBlockOverride

    if len(arrangements) not in (1, 7):
        raise ValueError("invalid_period")
    with transaction.atomic():
        User.objects.select_for_update().get(pk=user.pk)
        list(RecurringBlock.objects.select_for_update().filter(user=user))
        fingerprint = empreinte_entrees_jour if len(arrangements) == 1 else empreinte_entrees_semaine
        if expected_entries is not None and expected_entries != fingerprint(user, start):
            raise ValueError("stale_plan")
        pending, moved_days = [], []
        for offset, arrangement in enumerate(arrangements):
            day = start + timedelta(days=offset)
            blocks = {b.pk: b for b in recurring_occurrences(user, day) if b.is_flexible}
            if {r.get("block_id") for r in arrangement} != set(blocks) or len(arrangement) != len(blocks):
                raise ValueError("invalid_occurrences")
            before = {b["id"]: b for b in effective_day_entries(user, day) if b["type"] == "recurring"}
            busy = list(fixed_busy_intervals(user, day)) + list(intervalles_sommeil_reporte(user, day))
            for row in arrangement:
                if row.get("overnight_kept") and not row.get("occurrence_override"):
                    busy.extend([(row["start_min"], 1440), (0, row["end_min"])])
            moved = []
            for row in arrangement:
                block = blocks[row["block_id"]]
                if row.get("skipped"):
                    if row.get("reporte_au_lendemain"):
                        continue
                    raise ValueError("unplaced_occurrences")
                if row.get("overnight_kept") or block.occurrence_override:
                    if (row.get("start_time"), row.get("end_time")) != (block.start_time.strftime("%H:%M"), block.end_time.strftime("%H:%M")):
                        raise ValueError("invalid_occurrence_times")
                    continue
                s, e = row.get("start_min"), row.get("end_min")
                if not isinstance(s, int) or not isinstance(e, int) or not 0 <= s < e <= 1440:
                    raise ValueError("invalid_occurrence_times")
                if e - s != block.effective_duration_minutes():
                    raise ValueError("invalid_occurrence_duration")
                if any(s < be and bs < e for bs, be in busy):
                    raise ValueError("occurrence_conflict")
                busy.append((s, e))
                new_start, new_end = time(s // 60, s % 60), time((e // 60) % 24, e % 60)
                if row["start_time"] != _minutes_to_str(s) or row["end_time"] not in {_minutes_to_str(e), new_end.strftime("%H:%M")}:
                    raise ValueError("invalid_occurrence_times")
                row["end_time"] = new_end.strftime("%H:%M")
                pending.append((block.pk, day, new_start, new_end))
                previous = before.get(block.pk) or {}
                if (previous.get("start_time"), previous.get("end_time")) != (row["start_time"], row["end_time"]):
                    moved.append({k: row[k] for k in ("block_id", "title", "start_time", "end_time")})
            moved_days.append(moved)
        for block_id, day, new_start, new_end in pending:
            RecurringBlockOverride.objects.update_or_create(
                recurring_block_id=block_id, date=day,
                defaults={"user": user, "start_time": new_start, "end_time": new_end},
            )
        for offset, arrangement in enumerate(arrangements):
            effective = {b["id"]: b for b in effective_day_entries(user, start + timedelta(days=offset))
                         if b["type"] == "recurring"}
            for row in arrangement:
                if row.get("skipped"):
                    continue
                observed = effective.get(row["block_id"])
                if observed is None or any(observed[key] != row[key] for key in ("title", "start_time", "end_time")):
                    raise ValueError("occurrence_postcondition_failed")
        return moved_days


def _resultat_semaine(*, apply, start, days_data, lines, moved_total, skipped_total):
    head = "Semaine réorganisée" if apply else "Proposition pour la semaine (rien n'est encore changé)"
    msg = f"{head}: " + ("; ".join(lines) if lines else "rien à replacer, tout est déjà bien agencé") + "."
    if skipped_total:
        msg += f" {skipped_total} activité(s) ne rentrent pas entièrement (journées trop pleines)."
    return ToolResult(
        success=True,
        data={"applied": apply, "start_date": start.isoformat(),
              "days": days_data, "moved_count": moved_total, "skipped_count": skipped_total},
        message=msg,
    )


class OptimizeWeekTool(BaseTool):
    name = "optimize_week"
    description = (
        "Résout et RÉORGANISE de façon OPTIMALE les blocs SOUPLES de TOUTE LA "
        "SEMAINE (7 jours consécutifs), jour par jour, autour des murs fixes "
        "(cours, travail) et du sommeil protégé, via le solveur exact. À utiliser "
        "quand l'utilisateur demande d'optimiser / réorganiser / mieux agencer sa "
        "SEMAINE (pour UN seul jour → organize_day). apply=false PROPOSE le plan "
        "sans rien changer (défaut); apply=true l'APPLIQUE en ajustant les heures "
        "des occurrences souples sur ces 7 dates seulement. Les horaires des "
        "séries et des semaines suivantes restent les mêmes. Ne touche jamais aux blocs fixes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "start_date": {"type": "string", "description": "Premier jour de la passe (YYYY-MM-DD). Défaut: aujourd'hui."},
            "apply": {"type": "boolean", "description": "true = appliquer (déplacer les blocs); false = proposer seulement (défaut)."},
        },
        "required": [],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        raw = kwargs.get("start_date")
        if raw:
            try:
                start = datetime.strptime(raw, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                return ToolResult(success=False, data={}, message="Format de date invalide. Utilise YYYY-MM-DD.",
                              issue=ISSUE_ECHEC, cause="date_illisible",
                              public="Je n'ai pas pu vérifier ce créneau : la date était illisible.")
        else:
            start = timezone.localdate()
        apply = bool(kwargs.get("apply", False))
        expected_entries = kwargs.pop("_expected_entries", None) or empreinte_entrees_semaine(user, start)
        # Kwarg PRIVE (agent v2): arrangement deja resolu et valide par la
        # garde. Evite de re-resoudre 7 jours a l'apply quand le plan vient
        # d'etre verifie a l'identique. Jamais fourni par le modele ni v1.
        arrangements = kwargs.pop("_arrangements", None)
        if arrangements is None:
            days_data, lines, skipped_total, arrangements = _resoudre_semaine(user, start)
        else:
            days_data, lines, skipped_total = _mettre_en_forme_semaine(user, start, arrangements)

        moved_total = 0
        if apply:
            try:
                moved_days = _appliquer_periode(user, start, arrangements, expected_entries)
            except ValueError as exc:
                return ToolResult(success=False, data={"applied": False, "reason": str(exc),
                                                       "start_date": start.isoformat()},
                                  message="Ce plan ne peut plus être appliqué tel quel. Rien n'a changé. Demande une nouvelle proposition.")
            days_data, lines, skipped_total = _mettre_en_forme_semaine(user, start, arrangements)
            for jour, moved in zip(days_data, moved_days):
                jour["moved"] = moved
                moved_total += len(moved)

        return _resultat_semaine(
            apply=apply, start=start, days_data=days_data, lines=lines,
            moved_total=moved_total, skipped_total=skipped_total,
        )


class CancelScheduledBlockTool(BaseTool):
    name = "cancel_scheduled_block"
    description = (
        "Annule/supprime un événement PONCTUEL déjà planifié à une date précise "
        "(ex: 'annule mon rdv optométriste','enlève mon étude de demain', 'supprime "
        "le prochain événement'). C'est l'outil pour un one-off créé par "
        "schedule_task_at. NE l'utilise PAS pour un bloc RÉCURRENT: pour sauter UN "
        "jour d'une série -> skip_block_occurrence; pour supprimer toute la série -> "
        "delete_block. Résous par date (+ titre si plusieurs événements ce jour-là); "
        "ne demande jamais d'identifiant. Si plusieurs événements distincts existent "
        "ce jour-là sans titre fourni, l'outil renvoie la liste et te demande lequel."
    )
    parameters = {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "Date de l'événement (YYYY-MM-DD)."},
            "title": {"type": "string", "description": "Titre pour lever l'ambiguïté si plusieurs événements ce jour-là (optionnel)."},
        },
        "required": ["date"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        try:
            target_date = datetime.strptime(kwargs["date"], "%Y-%m-%d").date()
        except (ValueError, KeyError, TypeError):
            return ToolResult(success=False, data={}, message="Format de date invalide (attendu YYYY-MM-DD).")

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
