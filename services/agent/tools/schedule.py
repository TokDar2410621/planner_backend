"""
Schedule query tools for the Planner AI agent.
"""
from datetime import date, datetime, time, timedelta

from django.contrib.auth.models import User
from django.utils import timezone

from core.models import RecurringBlock, ScheduledBlock, Task
from services.scheduling.exceptions import skipped_block_ids
from services.scheduling.overlap import parse_time
from services.scheduling.placement import (
    fixed_busy_intervals,
    occupied_intervals,
    open_intervals,
    place_day,
)
from services.scheduling.solve_day import solve_day, solve_placement
from .base import BaseTool, ToolResult, validate_choice

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
    description = "Récupère le planning EFFECTIF d'un jour (blocs récurrents aux heures PLACÉES, tâches planifiées, créneaux libres). Consulte-le avant d'affirmer où se trouve une activité ou si elle a bougé, et parle des heures effectives — jamais de mémoire ni d'après l'historique (un bloc souple peut être placé à une autre heure que son heure habituelle)."
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
                return ToolResult(success=False, data={}, message="Format de date invalide. Utilise YYYY-MM-DD.")
        else:
            target_date = timezone.localdate()

        # Day of week (0=Monday)
        day_of_week = target_date.weekday()
        day_name = DAY_NAMES[day_of_week]

        # Recurring blocks for this day (moins les occurrences ignorées ce jour-là)
        recurring = RecurringBlock.objects.filter(
            user=user, day_of_week=day_of_week, active=True
        ).exclude(
            id__in=skipped_block_ids(user, target_date)
        ).order_by("start_time")

        placements = {
            p["block_id"]: p
            for p in place_day(user, target_date)
        }
        blocks = []
        for b in recurring:
            start_time = b.start_time.strftime("%H:%M")
            end_time = b.end_time.strftime("%H:%M")
            if b.is_flexible:
                placement = placements.get(b.id)
                if placement is None or placement["skipped"]:
                    continue
                start_time = placement["start_time"]
                end_time = placement["end_time"]

            blocks.append({
                "id": b.id,
                "title": b.title,
                "type": "recurring",
                "block_type": b.block_type,
                "start_time": start_time,
                "end_time": end_time,
                "location": b.location or None,
            })

        # Scheduled tasks for this day
        scheduled = ScheduledBlock.objects.filter(
            user=user, date=target_date
        ).select_related("task").order_by("start_time")

        for sb in scheduled:
            blocks.append({
                "id": sb.id,
                "title": sb.task.title,
                "type": "scheduled_task",
                "task_type": sb.task.task_type,
                "start_time": sb.start_time.strftime("%H:%M"),
                "end_time": sb.end_time.strftime("%H:%M"),
                "completed": sb.actually_completed,
            })

        # Sort all blocks by start time
        blocks.sort(key=lambda x: x["start_time"])

        # Créneaux libres 7h-23h via la logique unique overnight-aware (compte
        # correctement un quart de nuit du jour ET le débordement de la veille,
        # + les blocs déjà planifiés). Corrige le faux "libre 7h-23h".
        free_slots = _free_slots_from_intervals(
            open_intervals(user, target_date, DAY_START_MIN, DAY_END_MIN), 30
        )

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

            blocks = RecurringBlock.objects.filter(
                user=user, day_of_week=day_of_week, active=True
            ).exclude(id__in=skipped_block_ids(user, day))

            day_minutes = 0
            block_list = []
            for b in blocks:
                start_min = _time_to_minutes(b.start_time)
                end_min = _time_to_minutes(b.end_time)
                duration = end_min - start_min if end_min > start_min else (24 * 60 - start_min + end_min)
                day_minutes += duration
                block_list.append(f"{b.title} ({b.start_time.strftime('%H:%M')}-{b.end_time.strftime('%H:%M')})")

            total_hours += day_minutes / 60

            days_summary.append({
                "date": day.isoformat(),
                "day_name": DAY_NAMES[day_of_week],
                "block_count": blocks.count(),
                "occupied_hours": round(day_minutes / 60, 1),
                "blocks": block_list,
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
    description = "Trouve les créneaux libres d'un jour (overnight-aware: un quart de nuit occupe la soirée, le travail de la veille occupe le matin), avec une durée minimum optionnelle. Appelle-le AVANT de déclarer un jour 'libre' ou de choisir un créneau toi-même (ne demande pas l'heure à l'utilisateur si tu peux la trancher)."
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
        free = _free_slots_from_intervals(
            open_intervals(user, target_date, DAY_START_MIN, DAY_END_MIN), min_duration
        )

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


def _window_conflict(user, target_date, s, e, exclude_scheduled_id=None):
    """ToolResult d'erreur si la fenêtre [s,e] (minutes) du jour target_date
    chevauche un mur fixe ou le sommeil protégé; sinon None. Réutilisable pour
    chaque moitié d'un événement ponctuel qui traverse minuit."""
    for bs, be in fixed_busy_intervals(user, target_date, exclude_scheduled_id=exclude_scheduled_id):
        if s < be and bs < e:
            return ToolResult(
                success=False,
                data={"conflict": {"start_time": _minutes_to_str(bs), "end_time": _minutes_to_str(be)}},
                message=(
                    f"Ce créneau ({_minutes_to_str(s)}-{_minutes_to_str(e)}) chevauche une "
                    f"occupation existante ({_minutes_to_str(bs)}-{_minutes_to_str(be)}). "
                    f"Choisis un autre horaire libre."
                ),
            )
    for placement in place_day(user, target_date):
        if placement["skipped"] or placement["block_type"] != "sleep":
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
        "Planifie un événement PONCTUEL daté à une heure précise (ex: 'lecture ce "
        "samedi 9h-11h', 'rdv mardi 14h-15h'). C'est l'outil pour un événement UNIQUE "
        "sur une date donnée: il crée directement le créneau (verrouillé, la "
        "replanification ne le bouge pas). N'utilise PAS create_block (qui crée une "
        "habitude répétée CHAQUE semaine) pour un événement ponctuel. Si l'utilisateur "
        "ne donne pas d'heure, choisis toi-même un créneau libre (find_free_slots) au "
        "lieu de lui demander. Pour AJUSTER un événement déjà planifié (durée/heure: "
        "'finalement 45 min'), ré-appelle schedule_task_at avec le même titre et la "
        "même date: ça MET À JOUR l'événement (upsert), ce n'est pas un doublon. "
        "Gère aussi un événement qui traverse minuit (ex: quart de nuit PONCTUEL "
        "'ce lundi 22h-06h'). Pour un quart de nuit RÉCURRENT (chaque semaine), "
        "utilise plutôt create_block."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Titre de l'événement (ex: 'Lecture')."},
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
            return ToolResult(success=False, data={}, message="Format de date invalide. Utilise YYYY-MM-DD.")

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
            return ToolResult(success=False, data={}, message="Format de date invalide. Utilise YYYY-MM-DD.")

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
            return ToolResult(success=False, data={}, message="Aucune activité valide à tester (titre + durée en minutes).")

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
        return ToolResult(
            success=True,
            data={
                "feasible": result["feasible"],
                "placed": placed,
                "unplaced": unplaced,
                "date": target_date.isoformat(),
            },
            message=msg,
        )


class OrganizeDayTool(BaseTool):
    name = "organize_day"
    description = (
        "Résout et RÉORGANISE de façon OPTIMALE le placement des blocs SOUPLES "
        "d'une journée (sport, révision, repas, sommeil...) autour des murs fixes "
        "(cours, travail) et du sommeil protégé, via un solveur exact. À utiliser "
        "quand l'utilisateur demande d'optimiser / réorganiser / mieux agencer sa "
        "journée, ou quand le placement laisse des trous. apply=false PROPOSE "
        "l'arrangement sans rien changer (défaut); apply=true l'APPLIQUE en ajustant "
        "les heures des blocs souples. Ne touche jamais aux blocs fixes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "Jour à réorganiser (YYYY-MM-DD). Déduis-le de la DATE du jour."},
            "apply": {"type": "boolean", "description": "true = appliquer (déplacer les blocs); false = proposer seulement (défaut)."},
        },
        "required": ["date"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        try:
            target_date = datetime.strptime(kwargs["date"], "%Y-%m-%d").date()
        except (ValueError, KeyError, TypeError):
            return ToolResult(success=False, data={}, message="Format de date invalide. Utilise YYYY-MM-DD.")
        apply = bool(kwargs.get("apply", False))

        arrangement = solve_placement(user, target_date)  # fenêtre pleine journée
        placed = [r for r in arrangement if not r["skipped"] and not r["overnight_kept"]]
        overnight = [r for r in arrangement if r["overnight_kept"]]
        skipped = [r for r in arrangement if r["skipped"]]

        moved = []
        if apply:
            for r in placed:
                block = RecurringBlock.objects.filter(id=r["block_id"], user=user).first()
                if block is None:
                    continue
                new_start = time(r["start_min"] // 60, r["start_min"] % 60)
                new_end = time(r["end_min"] // 60, r["end_min"] % 60)
                if block.start_time != new_start or block.end_time != new_end:
                    block.start_time = new_start
                    block.end_time = new_end
                    block.save(update_fields=["start_time", "end_time"])
                    moved.append({"title": block.title, "start_time": r["start_time"], "end_time": r["end_time"]})

        listing = ", ".join(f"{r['title']} {r['start_time']}-{r['end_time']}" for r in placed) or "rien à replacer"
        head = "J'ai réorganisé" if apply else "Proposition (rien changé)"
        msg = f"{head} le {DAY_NAMES[target_date.weekday()]} {target_date.isoformat()}: {listing}."
        if skipped:
            msg += " Non placé (journée trop pleine): " + ", ".join(r["title"] for r in skipped) + "."
        return ToolResult(
            success=True,
            data={
                "applied": apply,
                "date": target_date.isoformat(),
                "placed": [{"title": r["title"], "start_time": r["start_time"], "end_time": r["end_time"]} for r in placed],
                "overnight_kept": [{"title": r["title"], "start_time": r["start_time"], "end_time": r["end_time"]} for r in overnight],
                "skipped": [{"title": r["title"]} for r in skipped],
                "moved": moved,
            },
            message=msg,
        )
