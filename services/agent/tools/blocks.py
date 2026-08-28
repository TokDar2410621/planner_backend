"""
RecurringBlock tools for the Planner AI agent.
"""
from datetime import datetime

from django.contrib.auth.models import User
from django.db import transaction

from core.models import RecurringBlock, RecurringBlockException
from services.scheduling.overlap import (
    find_recurring_conflicts,
    is_overnight,
    parse_time,
)
from .base import BaseTool, ToolResult, validate_choice, validate_max_length

DAY_NAMES = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

# Le modele peut nommer les jours au lieu de les numeroter.
#
# POURQUOI. L'audit de production du 2026-08-19 decrivait un decalage de +1
# jour a l'import: des matchs annonces samedi stockes dimanche, l'utilisateur
# finissant par vider ses 28 blocs. Le defaut a ete REPRODUIT le 2026-08-28.
# L'outil n'y est pour rien (days=[5] stocke bien samedi) et le schema n'est
# pas ambigu (il dit 0=Lundi..6=Dimanche): c'est le modele qui retombe sur la
# convention americaine dimanche=0, et seulement sur les jours de fin de
# semaine. Une convention bien documentee ne suffit donc pas; on retire
# l'occasion de se tromper.
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

# Enforced at the tool layer (not just in the JSON schema) before any write.
VALID_BLOCK_TYPES = {c[0] for c in RecurringBlock.BLOCK_TYPE_CHOICES}
VALID_FLEXIBILITIES = {c[0] for c in RecurringBlock.FLEXIBILITY_CHOICES}
TITLE_MAX_LENGTH = RecurringBlock._meta.get_field("title").max_length
LOCATION_MAX_LENGTH = RecurringBlock._meta.get_field("location").max_length


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
    description = "Crée un ou plusieurs blocs récurrents HEBDOMADAIRES (une habitude qui revient chaque semaine: cours, travail, sport, sommeil...). Jours = liste (ex: [0,1,2,3,4] = lundi à vendredi); un bloc séparé par jour. Appelle-le DÈS que l'utilisateur décrit un horaire habituel. N'utilise JAMAIS create_block pour un événement unique daté ('ce samedi', une date précise) -> schedule_task_at. Gère un quart de nuit récurrent qui traverse minuit (ex: travail 22:00-06:00 chaque lundi): mets start_time/end_time tels quels, le passage de minuit est détecté automatiquement. Les conflits sont détectés et renvoyés automatiquement."
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
                "items": {"type": "string"},
                "description": "Jours de la semaine. Utilise de preference les NOMS: [\"samedi\", \"dimanche\"]. Les numeros restent acceptes (0=Lundi, 1=Mardi, ..., 6=Dimanche) mais les noms evitent toute confusion de convention.",
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
            "flexibility": {
                "type": "string",
                "enum": ["fixed", "flexible"],
                "description": "fixed si le bloc ne doit pas bouger; flexible sinon.",
            },
            "is_night_shift": {
                "type": "boolean",
                "description": "Si le bloc traverse minuit (ex: travail 22h-06h)",
            },
            "start_date": {
                "type": "string",
                "description": "Premiere date d'application YYYY-MM-DD (ex: entrainements qui commencent le 24 aout). Omettre si des maintenant.",
            },
            "end_date": {
                "type": "string",
                "description": "Derniere date d'application YYYY-MM-DD (fin de session/saison). Omettre si sans fin.",
            },
        },
        "required": ["title", "block_type", "days", "start_time", "end_time"],
    }

    def execute(self, user: User, **kwargs) -> ToolResult:
        title = kwargs["title"]
        block_type = kwargs["block_type"]
        # Noms ou numeros: le modele se trompait sur samedi et dimanche.
        days = normaliser_jours(kwargs["days"])
        start_time = kwargs["start_time"]
        end_time = kwargs["end_time"]
        location = kwargs.get("location", "")
        flexibility = kwargs.get("flexibility")
        is_night_shift = kwargs.get("is_night_shift", False)

        from datetime import date as dt_date
        bounds = {}
        for key in ("start_date", "end_date"):
            raw = str(kwargs.get(key) or "").strip()
            if raw:
                try:
                    bounds[key] = dt_date.fromisoformat(raw[:10])
                except ValueError:
                    return ToolResult(success=False, data={}, message=f"{key} invalide: attendu YYYY-MM-DD.")
        if bounds.get("start_date") and bounds.get("end_date") and bounds["end_date"] < bounds["start_date"]:
            return ToolResult(success=False, data={}, message="end_date avant start_date.")

        # Validation choices/max_length AVANT toute écriture (le schéma JSON
        # n'est qu'indicatif pour le LLM; SQLite n'applique pas ces contraintes).
        err = (
            validate_choice(block_type, VALID_BLOCK_TYPES, "block_type")
            or validate_choice(flexibility, VALID_FLEXIBILITIES, "flexibility")
            or validate_max_length(title, TITLE_MAX_LENGTH, "title")
            or validate_max_length(location, LOCATION_MAX_LENGTH, "location")
        )
        if err:
            return ToolResult(success=False, data={}, message=err)

        # Parse + valide les heures une seule fois (format HH:MM).
        try:
            start_t = parse_time(start_time)
            end_t = parse_time(end_time)
        except ValueError as e:
            return ToolResult(success=False, data={}, message=str(e))

        # Détection night shift basée sur les minutes (pas une comparaison de chaînes).
        is_night_shift = is_overnight(start_t, end_t, is_night_shift)
        nf = (
            flexibility == RecurringBlock.FLEXIBILITY_FLEXIBLE
            if flexibility
            else RecurringBlock.default_flexibility_for(block_type) == RecurringBlock.FLEXIBILITY_FLEXIBLE
        )

        created = []
        skipped = []

        # Opération multi-lignes: tout ou rien. Si une création échoue en cours
        # de route, aucun bloc partiel ne subsiste (D5: transaction.atomic).
        try:
            with transaction.atomic():
                for day in days:
                    if day < 0 or day > 6:
                        skipped.append({"day": day, "reason": "Jour invalide"})
                        continue

                    # Anti-doublon EXACT, quelle que soit la flexibilité: un bloc
                    # souple (sommeil, repas...) échappe à find_recurring_conflicts
                    # (new_is_flexible -> []), donc rien n'empêchait de recréer un
                    # "Déjeuner"/"Sommeil"/cours identique. Cause des doublons vus
                    # quand le LLM "déplace"/"annule" en recréant sans supprimer.
                    # SANS block_type: un import classe « Entraînements » en
                    # course/other, l'agent en sport — même titre, même jour,
                    # mêmes heures = même bloc (vécu: paire dupliquée e2e).
                    if RecurringBlock.objects.filter(
                        user=user,
                        active=True,
                        day_of_week=day,
                        title=title,
                        start_time=start_t,
                        end_time=end_t,
                    ).exists():
                        skipped.append({
                            "day": day,
                            "day_name": DAY_NAMES[day],
                            "reason": f"'{title}' existe déjà le {DAY_NAMES[day]} à {start_time} (aucun doublon créé)",
                        })
                        continue

                    # Contrôle de chevauchement (gère l'overnight, AUCUN contournement).
                    conflicts = find_recurring_conflicts(
                        user,
                        day,
                        start_t,
                        end_t,
                        is_night_shift,
                        new_is_flexible=nf,
                    )
                    if conflicts:
                        overlap = conflicts[0]
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
                        start_time=start_t,
                        end_time=end_t,
                        flexibility=flexibility,
                        location=location,
                        is_night_shift=is_night_shift,
                        **bounds,
                    )
                    created.append(_block_to_dict(block))
        except Exception as e:
            # Rollback déjà effectué par le context manager: aucune ligne partielle.
            return ToolResult(
                success=False,
                data={},
                message=f"Création annulée (aucun bloc créé): {e}",
            )

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
    description = "Modifie un bloc récurrent existant (titre, horaires, JOUR, lieu, flexibilité). Résous le bloc par son nom/jour/heure via list_blocks; ne demande JAMAIS un identifiant à l'utilisateur. Pour DÉPLACER un bloc vers un autre jour (ex: 'déplace mon cours du lundi au mardi'), passe day_of_week (0=Lundi..6=Dimanche): c'est un déplacement PROPRE en place. N'utilise JAMAIS delete_block+create_block pour déplacer (ça crée des doublons). flexibility='fixed' verrouille/protège le bloc; flexibility='flexible' le rend déplaçable. Pour annuler/ignorer UN SEUL jour, n'utilise PAS update_block -> skip_block_occurrence."
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
            "flexibility": {
                "type": "string",
                "enum": ["fixed", "flexible"],
                "description": "fixed pour verrouiller le bloc recurrent, flexible pour le rendre deplacable.",
            },
            "day_of_week": {
                "type": "integer",
                "minimum": 0,
                "maximum": 6,
                "description": "Nouveau jour (0=Lundi..6=Dimanche) pour DÉPLACER le bloc vers ce jour, proprement, sans le recréer.",
            },
            "start_date": {
                "type": "string",
                "description": "Premiere date d'application YYYY-MM-DD; chaine vide '' pour retirer la borne.",
            },
            "end_date": {
                "type": "string",
                "description": "Derniere date d'application YYYY-MM-DD (ex: la session finit le 15 octobre); chaine vide '' pour retirer la borne.",
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

        # Validation choices/max_length AVANT save() (idem CreateBlockTool).
        err = (
            validate_choice(kwargs.get("block_type"), VALID_BLOCK_TYPES, "block_type")
            or validate_choice(kwargs.get("flexibility"), VALID_FLEXIBILITIES, "flexibility")
            or validate_max_length(kwargs.get("title"), TITLE_MAX_LENGTH, "title")
            or validate_max_length(kwargs.get("location"), LOCATION_MAX_LENGTH, "location")
        )
        if err:
            return ToolResult(success=False, data={}, message=err)

        # Heures effectives après modification (parse + valide si fournies).
        try:
            new_start = parse_time(kwargs["start_time"]) if kwargs.get("start_time") else block.start_time
            new_end = parse_time(kwargs["end_time"]) if kwargs.get("end_time") else block.end_time
        except ValueError as e:
            return ToolResult(success=False, data={}, message=str(e))

        # Déplacement inter-jours propre (pas de delete+create qui duplique).
        new_day = kwargs.get("day_of_week")
        if new_day is None:
            new_day = block.day_of_week
        elif not (0 <= new_day <= 6):
            return ToolResult(success=False, data={}, message="day_of_week invalide (0=Lundi..6=Dimanche).")

        night = is_overnight(new_start, new_end, block.is_night_shift)
        new_block_type = kwargs.get("block_type") or block.block_type
        if kwargs.get("flexibility") is not None:
            new_flexibility = kwargs["flexibility"]
        elif kwargs.get("block_type") is not None:
            new_flexibility = RecurringBlock.default_flexibility_for(new_block_type)
        else:
            new_flexibility = block.flexibility or RecurringBlock.default_flexibility_for(new_block_type)
        new_is_flexible = new_flexibility == RecurringBlock.FLEXIBILITY_FLEXIBLE

        # Contrôle de chevauchement sur le jour CIBLE (en s'excluant soi-même).
        conflicts = find_recurring_conflicts(
            user,
            new_day,
            new_start,
            new_end,
            night,
            exclude_id=block.id,
            new_is_flexible=new_is_flexible,
        )
        if conflicts:
            o = conflicts[0]
            return ToolResult(
                success=False,
                data={},
                message=(
                    f"Modification annulée: chevauchement avec '{o.title}' "
                    f"({o.start_time.strftime('%H:%M')}-{o.end_time.strftime('%H:%M')})."
                ),
            )

        if kwargs.get("title") is not None:
            block.title = kwargs["title"]
        if kwargs.get("location") is not None:
            block.location = kwargs["location"]
        if kwargs.get("block_type") is not None:
            block.block_type = kwargs["block_type"]
        from datetime import date as dt_date
        for key in ("start_date", "end_date"):
            if kwargs.get(key) is None:
                continue
            raw = str(kwargs[key]).strip()
            if raw == "":
                setattr(block, key, None)
            else:
                try:
                    setattr(block, key, dt_date.fromisoformat(raw[:10]))
                except ValueError:
                    return ToolResult(success=False, data={}, message=f"{key} invalide: attendu YYYY-MM-DD.")
        if block.start_date and block.end_date and block.end_date < block.start_date:
            return ToolResult(success=False, data={}, message="end_date avant start_date.")
        block.flexibility = new_flexibility
        block.start_time = new_start
        block.end_time = new_end
        block.is_night_shift = night
        block.day_of_week = new_day

        block.save()
        return ToolResult(
            success=True,
            data={"block": _block_to_dict(block)},
            message=f"Bloc '{block.title}' mis à jour.",
        )


class DeleteBlockTool(BaseTool):
    name = "delete_block"
    description = "Supprime TOUTE la série d'un bloc récurrent. Résous le bloc par nom/jour/heure; ne demande jamais d'identifiant à l'utilisateur. Pour annuler un seul jour, utilise skip_block_occurrence, PAS delete_block."
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
    description = (
        "Archive TOUS les blocs récurrents du planning (soft-delete réversible: "
        "les blocs sont désactivés, pas détruits). Opération destructive à fort "
        "impact: elle DOIT être confirmée par l'utilisateur hors-bande (via un "
        "dialogue de confirmation côté frontend), jamais sur la seule initiative "
        "du modèle. N'utilise cet outil que si l'utilisateur le demande "
        "explicitement (ex: 'vide tout mon planning', 'recommencer à zéro')."
    )
    # Flag surfaced to the orchestration/frontend layer: this call must be gated
    # by an explicit out-of-band human confirmation, not an LLM-supplied boolean.
    requires_confirmation = True
    parameters = {
        "type": "object",
        "properties": {
            "confirm": {
                "type": "boolean",
                "description": (
                    "Doit être true. Ne mets true que si l'utilisateur a "
                    "explicitement confirmé vouloir tout archiver."
                ),
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

        # Soft-delete (active=False) au lieu d'un DELETE: réversible, restaurable.
        # Opération multi-lignes → atomique.
        with transaction.atomic():
            count = RecurringBlock.objects.filter(
                user=user, active=True
            ).update(active=False)
        return ToolResult(
            success=True,
            data={"deleted_count": count, "reversible": True},
            message=(
                f"{count} bloc(s) archivé(s). Planning vidé "
                f"(action réversible: les blocs sont désactivés, pas détruits)."
            ),
        )


_BLOCK_TYPE_ENUM = ["course", "work", "sleep", "meal", "sport", "project", "revision", "other"]

# Champ `date` partagé par skip/restore: le modèle résout la date lui-même à
# partir de la DATE du jour (présente dans le system prompt), jamais un ID.
_OCCURRENCE_PARAMS = {
    "type": "object",
    "properties": {
        "date": {
            "type": "string",
            "description": "Date de l'occurrence (YYYY-MM-DD). Déduis-la toi-même de la DATE du jour (ex: 'ce vendredi').",
        },
        "block_type": {
            "type": "string",
            "enum": _BLOCK_TYPE_ENUM,
            "description": "Type du bloc concerné ce jour-là (ex: 'work'). Fortement recommandé pour cibler le bon bloc.",
        },
        "title": {
            "type": "string",
            "description": "Titre (ou fragment) du bloc, si plusieurs blocs partagent le même jour.",
        },
    },
    "required": ["date"],
}


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


def _parse_occurrence(kwargs):
    """(target_date, error) — parse la date + valide block_type. error=ToolResult|None."""
    try:
        target_date = datetime.strptime(kwargs["date"], "%Y-%m-%d").date()
    except (ValueError, KeyError, TypeError):
        return None, ToolResult(success=False, data={}, message="Format de date invalide. Utilise YYYY-MM-DD.")
    err = validate_choice(kwargs.get("block_type"), VALID_BLOCK_TYPES, "block_type")
    if err:
        return None, ToolResult(success=False, data={}, message=err)
    return target_date, None


class SkipBlockOccurrenceTool(BaseTool):
    name = "skip_block_occurrence"
    description = (
        "Ignore/annule UNE SEULE occurrence d'un bloc récurrent pour une date précise "
        "(ex: 'ce vendredi je ne travaille pas', 'pas de sport demain'). Le bloc récurrent "
        "reste en place les autres semaines. Résous toi-même le bon bloc avec la date + le "
        "type (et le titre si plusieurs blocs le même jour). Ne demande JAMAIS d'identifiant. "
        "N'utilise ni delete_block ni update_block pour annuler un seul jour."
    )
    parameters = _OCCURRENCE_PARAMS

    def execute(self, user: User, **kwargs) -> ToolResult:
        target_date, err = _parse_occurrence(kwargs)
        if err:
            return err

        dow = target_date.weekday()
        block_type = kwargs.get("block_type")
        matches = _resolve_day_blocks(user, target_date, block_type, kwargs.get("title"))

        if not matches:
            kind = f" '{block_type}'" if block_type else ""
            return ToolResult(
                success=False,
                data={},
                message=f"Aucun bloc récurrent{kind} le {DAY_NAMES[dow]} {target_date.isoformat()}. Rien à ignorer.",
            )
        if len(matches) > 1:
            candidates = _candidates_payload(matches, dow)
            listing = "; ".join(f"{c['title']} ({c['start_time']}-{c['end_time']})" for c in candidates)
            return ToolResult(
                success=False,
                data={"candidates": candidates},
                message=f"Plusieurs blocs le {DAY_NAMES[dow]}: {listing}. Précise lequel (par titre).",
            )

        block = matches[0]
        RecurringBlockException.objects.get_or_create(
            user=user, recurring_block=block, date=target_date
        )
        return ToolResult(
            success=True,
            data={"date": target_date.isoformat(), "title": block.title, "block_type": block.block_type},
            message=f"'{block.title}' ignoré le {DAY_NAMES[dow]} {target_date.isoformat()}. Le bloc récurrent reste actif les autres semaines.",
        )


class RestoreBlockOccurrenceTool(BaseTool):
    name = "restore_block_occurrence"
    description = (
        "Rétablit une occurrence précédemment ignorée d'un bloc récurrent (annule un skip). "
        "Ex: 'finalement je travaille ce vendredi'. Résous le bon bloc par date + type; "
        "ne demande jamais d'identifiant."
    )
    parameters = _OCCURRENCE_PARAMS

    def execute(self, user: User, **kwargs) -> ToolResult:
        target_date, err = _parse_occurrence(kwargs)
        if err:
            return err

        dow = target_date.weekday()
        block_type = kwargs.get("block_type")
        matches = _resolve_day_blocks(user, target_date, block_type, kwargs.get("title"))

        if not matches:
            kind = f" '{block_type}'" if block_type else ""
            return ToolResult(
                success=False,
                data={},
                message=f"Aucun bloc récurrent{kind} le {DAY_NAMES[dow]} {target_date.isoformat()}.",
            )
        if len(matches) > 1:
            candidates = _candidates_payload(matches, dow)
            listing = "; ".join(f"{c['title']} ({c['start_time']}-{c['end_time']})" for c in candidates)
            return ToolResult(
                success=False,
                data={"candidates": candidates},
                message=f"Plusieurs blocs le {DAY_NAMES[dow]}: {listing}. Précise lequel (par titre).",
            )

        block = matches[0]
        deleted, _ = RecurringBlockException.objects.filter(
            user=user, recurring_block=block, date=target_date
        ).delete()
        if deleted:
            msg = f"'{block.title}' rétabli le {DAY_NAMES[dow]} {target_date.isoformat()}."
        else:
            msg = f"'{block.title}' a bien lieu le {DAY_NAMES[dow]} {target_date.isoformat()} (aucune annulation à retirer)."
        return ToolResult(
            success=True,
            data={"date": target_date.isoformat(), "title": block.title, "restored": bool(deleted)},
            message=msg,
        )
