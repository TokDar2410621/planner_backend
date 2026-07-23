"""
Task tools for the Planner AI agent.
"""
from datetime import date, datetime

from django.contrib.auth.models import User
from django.utils import timezone

from core.models import Task, UserPlace
from .base import BaseTool, ToolResult, validate_choice, validate_max_length

# Enforced at the tool layer (not just in the JSON schema) before any write.
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


# Champs de lieu partagés par create_task / update_task (langage naturel, pas d'ID).
_PLACE_PARAMS = {
    "place_name": {
        "type": "string",
        "description": "Nom du lieu du rendez-vous si la tâche a lieu quelque part (ex: 'UQAC', 'Clinique'). Déclenche les rappels de départ.",
    },
    "place_address": {
        "type": "string",
        "description": "Adresse du lieu (optionnel), pour le géocodage et un trajet plus précis.",
    },
    "travel_minutes": {
        "type": "integer",
        "description": "Durée habituelle du trajet vers ce lieu en minutes (optionnel).",
        "minimum": 0,
    },
}


class ListTasksTool(BaseTool):
    name = "list_tasks"
    description = "Liste les tâches de l'utilisateur. Peut filtrer par statut (complétée ou non), type, ou limite."
    parameters = {
        "type": "object",
        "properties": {
            "completed": {
                "type": "boolean",
                "description": "Filtrer par statut (true=terminées, false=en cours)",
            },
            "task_type": {
                "type": "string",
                "description": "Filtrer par type",
                "enum": ["deep_work", "shallow", "errand"],
            },
            "limit": {
                "type": "integer",
                "description": "Nombre max de tâches à retourner (défaut: 20)",
                "minimum": 1,
                "maximum": 50,
            },
        },
        "required": [],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        tasks = Task.objects.filter(user=user)

        completed = kwargs.get("completed")
        if completed is not None:
            tasks = tasks.filter(completed=completed)

        task_type = kwargs.get("task_type")
        if task_type:
            tasks = tasks.filter(task_type=task_type)

        limit = kwargs.get("limit", 20)
        tasks = tasks[:limit]
        task_list = [_task_to_dict(t) for t in tasks]

        return ToolResult(
            success=True,
            data={"tasks": task_list, "count": len(task_list)},
            message=f"{len(task_list)} tâche(s) trouvée(s).",
        )


class CreateTaskTool(BaseTool):
    name = "create_task"
    description = (
        "Crée une nouvelle tâche pour l'utilisateur. Si la tâche a lieu quelque "
        "part (rendez-vous, réunion, cours ponctuel), renseigne place_name (et "
        "place_address si connue) pour activer les rappels de départ."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Titre de la tâche",
            },
            "task_type": {
                "type": "string",
                "description": "Type de tâche (deep_work=travail concentré, shallow=tâche légère, errand=course/admin)",
                "enum": ["deep_work", "shallow", "errand"],
            },
            "priority": {
                "type": "integer",
                "description": "Priorité de 1 (basse) à 10 (haute). Défaut: 5",
                "minimum": 1,
                "maximum": 10,
            },
            "description": {
                "type": "string",
                "description": "Description détaillée (optionnel)",
            },
            "deadline": {
                "type": "string",
                "description": "Date limite au format YYYY-MM-DD (optionnel)",
            },
            "estimated_duration_minutes": {
                "type": "integer",
                "description": "Durée estimée en minutes (optionnel)",
                "minimum": 5,
            },
            **_PLACE_PARAMS,
        },
        "required": ["title"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        task_type = kwargs.get("task_type", "shallow")
        title = kwargs["title"]

        # Validation choices/max_length AVANT create() (le schéma JSON n'est
        # qu'indicatif; SQLite n'applique ni choices ni max_length).
        err = (
            validate_choice(task_type, VALID_TASK_TYPES, "task_type")
            or validate_max_length(title, TASK_TITLE_MAX_LENGTH, "title")
        )
        if err:
            return ToolResult(success=False, data={}, message=err)

        # Idempotency: guard against the model re-creating a task it already
        # mentioned earlier in the conversation. If an ACTIVE (non-completed)
        # task with the same title already exists, return it instead of
        # duplicating.
        place = _resolve_place(
            user,
            kwargs.get("place_name"),
            kwargs.get("place_address"),
            kwargs.get("travel_minutes"),
        )

        existing = Task.objects.filter(
            user=user, completed=False, title__iexact=title.strip()
        ).first()
        if existing is not None:
            # N'attache le lieu à la tâche existante que s'il n'en a pas encore un.
            if place is not None and existing.place_id is None:
                existing.place = place
                existing.save(update_fields=["place"])
            return ToolResult(
                success=True,
                data={"task": _task_to_dict(existing)},
                message=f"Tâche '{existing.title}' déjà présente (non dupliquée).",
            )

        task = Task.objects.create(
            user=user,
            title=kwargs["title"],
            task_type=kwargs.get("task_type", "shallow"),
            priority=kwargs.get("priority", 5),
            description=kwargs.get("description", ""),
            deadline=_parse_deadline(kwargs.get("deadline")),
            estimated_duration_minutes=kwargs.get("estimated_duration_minutes"),
            place=place,
        )
        loc = f" à {place.name}" if place else ""
        return ToolResult(
            success=True,
            data={"task": _task_to_dict(task)},
            message=f"Tâche '{task.title}'{loc} créée (priorité {task.priority}).",
        )


class UpdateTaskTool(BaseTool):
    name = "update_task"
    description = "Modifie une tâche existante."
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {"type": "integer", "description": "ID de la tâche à modifier"},
            "title": {"type": "string", "description": "Nouveau titre"},
            "priority": {"type": "integer", "minimum": 1, "maximum": 10},
            "deadline": {"type": "string", "description": "Nouvelle deadline (YYYY-MM-DD)"},
            "description": {"type": "string"},
            "task_type": {"type": "string", "enum": ["deep_work", "shallow", "errand"]},
            **_PLACE_PARAMS,
        },
        "required": ["task_id"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        task_id = kwargs["task_id"]
        try:
            task = Task.objects.get(id=task_id, user=user)
        except Task.DoesNotExist:
            return ToolResult(success=False, data={}, message=f"Tâche #{task_id} introuvable.")

        # Validation choices/max_length AVANT save().
        err = (
            validate_choice(kwargs.get("task_type"), VALID_TASK_TYPES, "task_type")
            or validate_max_length(kwargs.get("title"), TASK_TITLE_MAX_LENGTH, "title")
        )
        if err:
            return ToolResult(success=False, data={}, message=err)

        for field in ["title", "priority", "description", "task_type"]:
            if field in kwargs and kwargs[field] is not None:
                setattr(task, field, kwargs[field])
        if kwargs.get("deadline") is not None:
            task.deadline = _parse_deadline(kwargs["deadline"])

        if kwargs.get("place_name") or kwargs.get("place_address"):
            place = _resolve_place(
                user,
                kwargs.get("place_name"),
                kwargs.get("place_address"),
                kwargs.get("travel_minutes"),
            )
            if place is not None:
                task.place = place

        task.save()
        return ToolResult(
            success=True,
            data={"task": _task_to_dict(task)},
            message=f"Tâche '{task.title}' mise à jour.",
        )


class DeleteTaskTool(BaseTool):
    name = "delete_task"
    description = (
        "Supprime définitivement une tâche (hard delete, cascade sur les blocs "
        "planifiés associés). Opération destructive et IRRÉVERSIBLE: elle DOIT "
        "être confirmée par l'utilisateur hors-bande (dialogue de confirmation "
        "côté frontend), jamais sur la seule initiative du modèle. Passe "
        "confirm=true uniquement après une confirmation explicite de "
        "l'utilisateur. Pour marquer une tâche faite, préfère complete_task."
    )
    # Flag surfaced to the orchestration/frontend layer: this call must be gated
    # by an explicit out-of-band human confirmation, not an LLM-supplied boolean.
    requires_confirmation = True
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {"type": "integer", "description": "ID de la tâche à supprimer"},
            "confirm": {
                "type": "boolean",
                "description": (
                    "Doit être true. Ne mets true que si l'utilisateur a "
                    "explicitement confirmé vouloir supprimer cette tâche."
                ),
            },
        },
        "required": ["task_id", "confirm"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        task_id = kwargs["task_id"]

        # Le modèle Task n'a pas de champ de soft-delete: la suppression est
        # irréversible. On exige donc une confirmation explicite avant de
        # détruire quoi que ce soit (S10).
        if not kwargs.get("confirm"):
            return ToolResult(
                success=False,
                data={"requires_confirmation": True},
                message=(
                    "Suppression annulée: confirm doit être true et la "
                    "suppression doit être confirmée explicitement par "
                    "l'utilisateur (action irréversible)."
                ),
            )

        try:
            task = Task.objects.get(id=task_id, user=user)
        except Task.DoesNotExist:
            return ToolResult(success=False, data={}, message=f"Tâche #{task_id} introuvable.")

        title = task.title
        task.delete()
        return ToolResult(
            success=True,
            data={"deleted_id": task_id},
            message=f"Tâche '{title}' supprimée.",
        )


class CompleteTaskTool(BaseTool):
    name = "complete_task"
    description = "Marque une tâche comme terminée."
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {"type": "integer", "description": "ID de la tâche à compléter"},
        },
        "required": ["task_id"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        task_id = kwargs["task_id"]
        try:
            task = Task.objects.get(id=task_id, user=user)
        except Task.DoesNotExist:
            return ToolResult(success=False, data={}, message=f"Tâche #{task_id} introuvable.")

        task.completed = True
        task.completed_at = timezone.now()
        task.save()
        return ToolResult(
            success=True,
            data={"task": _task_to_dict(task)},
            message=f"Tâche '{task.title}' marquée comme terminée.",
        )
