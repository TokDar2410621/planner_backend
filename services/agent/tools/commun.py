"""Helpers partages des outils fonctions.

Regroupe les helpers qui etaient eparpilles dans les anciens modules
d'outils (blocks.py, tasks.py, schedule.py, goals.py) et importes
ailleurs (v1, v2, reconciliation, tests). Les outils fonctions
l'importent via `from .commun import ...`.
"""
from datetime import date, datetime, time, timedelta
from typing import Annotated, Literal
import unicodedata

from django.contrib.auth.models import User
from django.db import transaction
from django.utils import timezone
from django.utils.dateparse import parse_date
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from core.models import Goal, RecurringBlock, RecurringBlockException, ScheduledBlock, Task, UserPlace
from services.scheduling.day_view import effective_day_entries
from services.scheduling.occurrences import recurring_occurrences
from services.scheduling.overlap import (
    find_recurring_conflicts,
    is_overnight,
    parse_time,
)
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
from .base import ToolResult


def _parse_occurrence(kwargs):
    """(target_date, error) — parse la date d'occurrence. error=ToolResult|None.

    Le block_type est desormais valide par le harnais contre le schema
    type avant execute : il n'y a plus de validate_choice ici.
    """
    try:
        target_date = datetime.strptime(kwargs["date"], "%Y-%m-%d").date()
    except (ValueError, KeyError, TypeError):
        return None, ToolResult(success=False, data={}, message="Format de date invalide. Utilise YYYY-MM-DD.",
                                public="Je n'ai rien changé : la date est invalide. Utilise le format AAAA-MM-JJ.")
    return target_date, None

# ---- blocs (ex blocks.py) ----
DAY_NAMES = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']


_JOURS_PAR_NOM = {
    'lundi': 0, 'mardi': 1, 'mercredi': 2, 'jeudi': 3,
    'vendredi': 4, 'samedi': 5, 'dimanche': 6,
}


def normaliser_jours(jours):
    """Rend une liste d'entiers 0..6, a partir de noms, de numeros ou des deux.

    Une valeur incomprise est IGNOREE, jamais devinee: deviner rouvrirait par
    un autre chemin le defaut qu'on vient de fermer.
    """
    import unicodedata

    if jours is None:
        return []
    if isinstance(jours, (str, int)):
        jours = [jours]

    sortie = []
    for brut in jours:
        valeur = None
        if isinstance(brut, bool):
            continue
        if isinstance(brut, int):
            valeur = brut
        elif isinstance(brut, str):
            plat = unicodedata.normalize('NFKD', brut.strip().lower())
            plat = plat.encode('ascii', 'ignore').decode('ascii')
            if plat.isdigit():
                valeur = int(plat)
            else:
                valeur = _JOURS_PAR_NOM.get(plat)
        if valeur is not None and 0 <= valeur <= 6 and valeur not in sortie:
            sortie.append(valeur)
    return sortie


def _coercer_jour(v):
    """Un numero de jour devient sa chaine avant validation.

    Le schema annonce des NOMS (la convention), mais normaliser_jours sait
    lire les numeros : l'entier est coercé en chaine ici pour que les deux
    passent la validation sans changer le type annonce.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return str(v)
    return v


# Champ jour : le schema dit 'string', la validation accepte aussi les entiers.
JourChaine = Annotated[str, BeforeValidator(_coercer_jour)]


VALID_BLOCK_TYPES = {c[0] for c in RecurringBlock.BLOCK_TYPE_CHOICES}
VALID_FLEXIBILITIES = {c[0] for c in RecurringBlock.FLEXIBILITY_CHOICES}
TITLE_MAX_LENGTH = RecurringBlock._meta.get_field("title").max_length
LOCATION_MAX_LENGTH = RecurringBlock._meta.get_field("location").max_length


# --- miroirs types des enums (source de verite : les CHOICES des modeles) ---
# _BLOCK_TYPE_ENUM est defini plus bas (extrait de blocks.py) ; les Literals
# sont le miroir type utilise dans les schemas pydantic, avec garde-fou.
TypeBloc = Literal["course", "work", "sleep", "meal", "sport", "project", "revision", "other"]
FlexibiliteBloc = Literal["fixed", "flexible"]


def _block_to_dict(block: RecurringBlock) -> dict:
    return {
        "id": block.id,
        "title": block.title,
        "block_type": block.block_type,
        "day_of_week": block.day_of_week,
        "day_name": DAY_NAMES[block.day_of_week],
        "start_time": block.start_time.strftime("%H:%M"),
        "end_time": block.end_time.strftime("%H:%M"),
        "flexibility": block.flexibility,
        "location": block.location or None,
        "is_night_shift": block.is_night_shift,
        "start_date": block.start_date.isoformat() if block.start_date else None,
        "end_date": block.end_date.isoformat() if block.end_date else None,
    }


_BLOCK_TYPE_ENUM = ["course", "work", "sleep", "meal", "sport", "project", "revision", "other"]

assert set(TypeBloc.__args__) == set(_BLOCK_TYPE_ENUM) == VALID_BLOCK_TYPES, \
    "derive entre le schema type et RecurringBlock.BLOCK_TYPE_CHOICES"
assert set(FlexibiliteBloc.__args__) == VALID_FLEXIBILITIES, \
    "derive entre le schema type et RecurringBlock.FLEXIBILITY_CHOICES"


class ParametresOccurrence(BaseModel):
    """Champs partages des outils d'occurrence (ex _OCCURRENCE_PARAMS).

    La `date` est resolue par le modele lui-meme a partir de la DATE du
    jour (system prompt), jamais un ID.
    """
    model_config = ConfigDict(extra="ignore")

    date: str = Field(
        description="Date de l'occurrence (YYYY-MM-DD). Déduis-la toi-même de la DATE du jour (ex: 'ce vendredi').",
    )
    block_type: TypeBloc | None = Field(
        default=None,
        description="Type du bloc concerné ce jour-là (ex: 'work'). Fortement recommandé pour cibler le bon bloc.",
    )
    title: str | None = Field(
        default=None,
        description="Titre (ou fragment) du bloc, si plusieurs blocs partagent le même jour.",
    )


class ChampsLieu(BaseModel):
    """Champs lieu partages create/update task (ex _PLACE_PARAMS)."""
    model_config = ConfigDict(extra="ignore")

    place_name: str | None = Field(
        default=None,
        description="Nom du lieu du rendez-vous si la tâche a lieu quelque part (ex: 'UQAC', 'Clinique'). Déclenche les rappels de départ.",
    )
    place_address: str | None = Field(
        default=None,
        description="Adresse du lieu (optionnel), pour le géocodage et un trajet plus précis.",
    )
    travel_minutes: int | None = Field(
        default=None, ge=0,
        description="Durée habituelle du trajet vers ce lieu en minutes (optionnel).",
    )


def _resolve_day_blocks(user, target_date, block_type=None, title=None):
    """Blocs récurrents actifs dont le jour == weekday(target_date), filtrés."""
    qs = RecurringBlock.objects.filter(
        user=user, active=True, day_of_week=target_date.weekday()
    )
    if block_type:
        qs = qs.filter(block_type=block_type)
    if title:
        qs = qs.filter(title__icontains=title)
    return list(qs.order_by("start_time"))


def _candidates_payload(blocks, dow):
    return [
        {
            "title": b.title,
            "day_name": DAY_NAMES[dow],
            "start_time": b.start_time.strftime("%H:%M"),
            "end_time": b.end_time.strftime("%H:%M"),
            "block_type": b.block_type,
        }
        for b in blocks
    ]


# ---- taches (ex tasks.py) ----
VALID_TASK_TYPES = {c[0] for c in Task.TASK_TYPE_CHOICES}
TASK_TITLE_MAX_LENGTH = Task._meta.get_field("title").max_length


def _resolve_place(user, place_name=None, place_address=None, travel_minutes=None):
    """Trouve ou crée un UserPlace (par nom) et le géocode. None si rien fourni.

    Permet à l'agent de poser un lieu sur une tâche depuis le langage naturel
    ("Réunion 14h à l'UQAC") sans que l'utilisateur ne manipule d'ID: le lieu
    est réutilisé s'il existe (contrainte unique user+name), géocodé au mieux
    (adresse sinon nom) via Nominatim.
    """
    name = (place_name or place_address or "").strip()
    if not name:
        return None
    name = name[:100]
    place, _created = UserPlace.objects.get_or_create(
        user=user,
        name=name,
        defaults={
            "address": (place_address or "").strip()[:300],
            "travel_minutes": travel_minutes or 0,
        },
    )
    # Complète adresse/trajet si fournis et encore absents.
    changed = False
    if place_address and not place.address:
        place.address = place_address.strip()[:300]
        changed = True
    if travel_minutes and not place.travel_minutes:
        place.travel_minutes = travel_minutes
        changed = True
    if changed:
        place.save()
    # Géocode si pas encore de coordonnées (best-effort, gratuit).
    if not place.has_coordinates:
        from services.geocoding import geocode_address
        coords = geocode_address(place.address or place.name)
        if coords:
            place.latitude, place.longitude = coords
            place.save(update_fields=["latitude", "longitude"])
    return place


def _parse_deadline(value):
    """Parse a deadline provided by the LLM (usually 'YYYY-MM-DD') into an aware
    datetime. Returns None if empty/unparseable.

    Without this, ``Task.objects.create(deadline='2026-07-22')`` keeps the raw
    string on the in-memory instance, so serializing it crashes with
    "'str' object has no attribute 'isoformat'" and the tool wrongly reports a
    failure (the model then apologizes even though the task was created). It also
    stored a naive datetime under USE_TZ.
    """
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    elif isinstance(value, str):
        v = value.strip()
        dt = None
        try:
            dt = datetime.fromisoformat(v)
        except ValueError:
            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    dt = datetime.strptime(v, fmt)
                    break
                except ValueError:
                    continue
        if dt is None:
            return None
    else:
        return None
    if timezone.is_naive(dt):
        dt = timezone.make_aware(dt, timezone.get_current_timezone())
    return dt


def _deadline_iso(deadline):
    """Serialize a deadline that may be a datetime or (defensively) a raw str."""
    if not deadline:
        return None
    return deadline.isoformat() if hasattr(deadline, "isoformat") else str(deadline)


def _task_to_dict(task: Task) -> dict:
    d = {
        "id": task.id,
        "title": task.title,
        "description": task.description or None,
        "task_type": task.task_type,
        "priority": task.priority,
        "deadline": _deadline_iso(task.deadline),
        "alert_at": _deadline_iso(task.alert_at),
        "recurrence_id": task.recurrence_id,
        "occurrence_date": task.occurrence_date.isoformat() if task.occurrence_date else None,
        "estimated_duration_minutes": task.estimated_duration_minutes,
        "completed": task.completed,
    }
    if task.place_id:
        d["place"] = {
            "name": task.place.name,
            "address": task.place.address or None,
            "has_coordinates": task.place.has_coordinates,
            "travel_minutes": task.place.travel_minutes,
        }
    return d


# ---- horaires (ex schedule.py) ----
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
                              message="Cet arrangement ne peut plus être appliqué tel quel. Rien n'a changé. Demande une nouvelle proposition.",
                              public="Je n'ai pas appliqué ce plan : il n'est plus valable tel quel. Demande une nouvelle proposition.")
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


# ---- objectifs (ex goals.py) ----
def _parse_goal_deadline(value):
    """Parse a 'YYYY-MM-DD' deadline to a date.

    Returns (date_or_None, error_or_None). A DateField assigned a raw string
    stays a string in memory until reloaded, so `_goal_to_dict` used to call
    `.isoformat()` on a str and crash. Parse before write.
    """
    if value in (None, ""):
        return None, None
    if not isinstance(value, str):
        return value, None
    d = parse_date(value.strip())
    if d is None:
        return None, "Format de deadline invalide (attendu YYYY-MM-DD)."
    return d, None


VALID_GOAL_TYPES = {c[0] for c in Goal.GOAL_TYPE_CHOICES}
VALID_GOAL_STATUSES = {c[0] for c in Goal.STATUS_CHOICES}
GOAL_TITLE_MAX_LENGTH = Goal._meta.get_field("title").max_length


def _goal_to_dict(goal: Goal) -> dict:
    return {
        "id": goal.id,
        "title": goal.title,
        "description": goal.description or None,
        "goal_type": goal.goal_type,
        "deadline": (
            goal.deadline.isoformat() if hasattr(goal.deadline, "isoformat")
            else (goal.deadline or None)
        ),
        # Observable quand l'objectif a des taches liees, manuel sinon.
        "progress": goal.progres_effectif(),
        "status": goal.status,
        "created_at": goal.created_at.isoformat(),
    }


# ---- interface (ex interactive.py) ----
def _fold(text) -> str:
    """Normalise un texte pour comparaison (utilise par present_form et
    present_choices)."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).strip().lower()


def _iso_date(value):
    """Normalise une date ISO (YYYY-MM-DD) ou rend None (utilise par
    present_form et present_choices)."""
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip()).isoformat()
    except ValueError:
        return None


def inline_refs(schema):
    """Remplace les $ref/$defs d'un schéma pydantic par leur contenu inline,
    comme les anciens schémas écrits à la main (le modèle et les tests
    lisent inputs.items.properties directement)."""
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if set(node.keys()) == {"$ref"}:
                return resolve(dict(defs[node["$ref"].split("/")[-1]]))
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)
