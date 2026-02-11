"""
RecurringBlock tools for the Planner AI agent.
"""
from django.contrib.auth.models import User

from core.models import RecurringBlock
from .base import BaseTool, ToolResult

DAY_NAMES = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']


def _block_to_dict(block: RecurringBlock) -> dict:
    return {
        "id": block.id,
        "title": block.title,
        "block_type": block.block_type,
        "day_of_week": block.day_of_week,
        "day_name": DAY_NAMES[block.day_of_week],
        "start_time": block.start_time.strftime("%H:%M"),
        "end_time": block.end_time.strftime("%H:%M"),
        "location": block.location or None,
        "is_night_shift": block.is_night_shift,
    }


class ListBlocksTool(BaseTool):
    name = "list_blocks"
    description = "Liste les blocs récurrents du planning de l'utilisateur. Peut filtrer par jour de la semaine (0=Lundi, 6=Dimanche) ou type de bloc."
    parameters = {
        "type": "object",
        "properties": {
            "day_of_week": {
                "type": "integer",
                "description": "Filtrer par jour (0=Lundi, 1=Mardi, ..., 6=Dimanche)",
                "minimum": 0,
                "maximum": 6,
            },
            "block_type": {
                "type": "string",
                "description": "Filtrer par type de bloc",
                "enum": ["course", "work", "sleep", "meal", "sport", "project", "revision", "other"],
            },
        },
        "required": [],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        blocks = RecurringBlock.objects.filter(user=user, active=True)

        day = kwargs.get("day_of_week")
        if day is not None:
            blocks = blocks.filter(day_of_week=day)

        block_type = kwargs.get("block_type")
        if block_type:
            blocks = blocks.filter(block_type=block_type)

        blocks = blocks.order_by("day_of_week", "start_time")
        block_list = [_block_to_dict(b) for b in blocks]

        return ToolResult(
            success=True,
            data={"blocks": block_list, "count": len(block_list)},
            message=f"{len(block_list)} bloc(s) trouvé(s).",
        )


class CreateBlockTool(BaseTool):
    name = "create_block"
    description = "Crée un ou plusieurs blocs récurrents dans le planning. Spécifie les jours comme une liste (ex: [0,1,2,3,4] pour lundi à vendredi). Un bloc séparé sera créé pour chaque jour."
    parameters = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Nom du bloc (ex: 'Travail', 'Cours de maths')",
            },
            "block_type": {
                "type": "string",
                "description": "Type du bloc",
                "enum": ["course", "work", "sleep", "meal", "sport", "project", "revision", "other"],
            },
            "days": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 6},
                "description": "Jours de la semaine (0=Lundi, 1=Mardi, ..., 6=Dimanche)",
            },
            "start_time": {
                "type": "string",
                "description": "Heure de début au format HH:MM (ex: '09:00')",
            },
            "end_time": {
                "type": "string",
                "description": "Heure de fin au format HH:MM (ex: '17:00')",
            },
            "location": {
                "type": "string",
                "description": "Lieu (optionnel)",
            },
            "is_night_shift": {
                "type": "boolean",
                "description": "Si le bloc traverse minuit (ex: travail 22h-06h)",
            },
        },
        "required": ["title", "block_type", "days", "start_time", "end_time"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        title = kwargs["title"]
        block_type = kwargs["block_type"]
        days = kwargs["days"]
        start_time = kwargs["start_time"]
        end_time = kwargs["end_time"]
        location = kwargs.get("location", "")
        is_night_shift = kwargs.get("is_night_shift", False)

        # Auto-detect night shift
        if not is_night_shift and end_time < start_time:
            is_night_shift = True

        created = []
        skipped = []

        for day in days:
            if day < 0 or day > 6:
                skipped.append({"day": day, "reason": "Jour invalide"})
                continue

            # Check for overlap
            existing = RecurringBlock.objects.filter(
                user=user, day_of_week=day, active=True,
                start_time__lt=end_time, end_time__gt=start_time,
            )
            if not is_night_shift and existing.exists():
                overlap = existing.first()
                skipped.append({
                    "day": day,
                    "day_name": DAY_NAMES[day],
                    "reason": f"Chevauchement avec '{overlap.title}' ({overlap.start_time.strftime('%H:%M')}-{overlap.end_time.strftime('%H:%M')})",
                })
                continue

            block = RecurringBlock.objects.create(
                user=user,
                title=title,
                block_type=block_type,
                day_of_week=day,
                start_time=start_time,
                end_time=end_time,
                location=location,
                is_night_shift=is_night_shift,
            )
            created.append(_block_to_dict(block))

        day_names = [DAY_NAMES[d] for d in days if 0 <= d <= 6]
        msg = f"{len(created)} bloc(s) créé(s): {title} ({start_time}-{end_time})"
        if created:
            msg += f" les {', '.join(d['day_name'] for d in created)}"
        if skipped:
            msg += f". {len(skipped)} sauté(s): " + "; ".join(s.get("reason", "") for s in skipped)

        return ToolResult(
            success=len(created) > 0,
            data={"created": created, "skipped": skipped},
            message=msg,
        )


class UpdateBlockTool(BaseTool):
    name = "update_block"
    description = "Modifie un bloc récurrent existant (titre, horaires, lieu, etc.)."
    parameters = {
        "type": "object",
        "properties": {
            "block_id": {
                "type": "integer",
                "description": "ID du bloc à modifier",
            },
            "title": {"type": "string", "description": "Nouveau titre"},
            "start_time": {"type": "string", "description": "Nouvelle heure de début (HH:MM)"},
            "end_time": {"type": "string", "description": "Nouvelle heure de fin (HH:MM)"},
            "location": {"type": "string", "description": "Nouveau lieu"},
            "block_type": {
                "type": "string",
                "enum": ["course", "work", "sleep", "meal", "sport", "project", "revision", "other"],
            },
        },
        "required": ["block_id"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        block_id = kwargs["block_id"]
        try:
            block = RecurringBlock.objects.get(id=block_id, user=user, active=True)
        except RecurringBlock.DoesNotExist:
            return ToolResult(success=False, data={}, message=f"Bloc #{block_id} introuvable.")

        for field in ["title", "start_time", "end_time", "location", "block_type"]:
            if field in kwargs and kwargs[field] is not None:
                setattr(block, field, kwargs[field])

        block.save()
        return ToolResult(
            success=True,
            data={"block": _block_to_dict(block)},
            message=f"Bloc '{block.title}' mis à jour.",
        )


class DeleteBlockTool(BaseTool):
    name = "delete_block"
    description = "Supprime un bloc récurrent du planning."
    parameters = {
        "type": "object",
        "properties": {
            "block_id": {
                "type": "integer",
                "description": "ID du bloc à supprimer",
            },
        },
        "required": ["block_id"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        block_id = kwargs["block_id"]
        try:
            block = RecurringBlock.objects.get(id=block_id, user=user, active=True)
        except RecurringBlock.DoesNotExist:
            return ToolResult(success=False, data={}, message=f"Bloc #{block_id} introuvable.")

        title = block.title
        block.active = False
        block.save()
        return ToolResult(
            success=True,
            data={"deleted_id": block_id},
            message=f"Bloc '{title}' supprimé.",
        )


class ClearAllBlocksTool(BaseTool):
    name = "clear_all_blocks"
    description = "Supprime TOUS les blocs récurrents du planning. Action irréversible. N'utilise cet outil que si l'utilisateur le demande explicitement (ex: 'vide tout mon planning', 'recommencer à zéro')."
    parameters = {
        "type": "object",
        "properties": {
            "confirm": {
                "type": "boolean",
                "description": "Doit être true pour confirmer la suppression totale",
            },
        },
        "required": ["confirm"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        if not kwargs.get("confirm"):
            return ToolResult(
                success=False,
                message="Suppression annulée: confirm doit être true.",
            )

        count = RecurringBlock.objects.filter(user=user, active=True).update(active=False)
        return ToolResult(
            success=True,
            data={"deleted_count": count},
            message=f"{count} bloc(s) supprimé(s). Planning vidé.",
        )
