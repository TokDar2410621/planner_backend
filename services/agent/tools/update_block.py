"""Outil update_block."""
from django.contrib.auth.models import User
from pydantic import BaseModel, ConfigDict, Field

from core.models import RecurringBlock
from services.scheduling.overlap import find_recurring_conflicts, is_overnight, parse_time
from . import commun
from .base import ToolResult

NAME = "update_block"
DESCRIPTION = "Modifie un bloc récurrent existant (titre, horaires, JOUR, lieu, flexibilité). Résous le bloc par son nom/jour/heure via list_blocks; ne demande JAMAIS un identifiant à l'utilisateur. Pour DÉPLACER un bloc vers un autre jour (ex: 'déplace mon atelier du lundi au mardi'), passe day_of_week (0=Lundi..6=Dimanche): c'est un déplacement PROPRE en place. N'utilise JAMAIS delete_block+create_block pour déplacer (ça crée des doublons). flexibility='fixed' verrouille/protège le bloc; flexibility='flexible' le rend déplaçable. Pour annuler/ignorer UN SEUL jour, n'utilise PAS update_block -> skip_block_occurrence."


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    block_id: int = Field(
        description="ID du bloc à modifier",
    )
    title: str | None = Field(
        default=None,
        max_length=commun.TITLE_MAX_LENGTH,
        description="Nouveau titre",
    )
    start_time: str | None = Field(
        default=None,
        description="Nouvelle heure de début (HH:MM)",
    )
    end_time: str | None = Field(
        default=None,
        description="Nouvelle heure de fin (HH:MM)",
    )
    location: str | None = Field(
        default=None,
        max_length=commun.LOCATION_MAX_LENGTH,
        description="Nouveau lieu",
    )
    block_type: commun.TypeBloc | None = None
    flexibility: commun.FlexibiliteBloc | None = Field(
        default=None,
        description="fixed pour verrouiller le bloc recurrent, flexible pour le rendre deplacable.",
    )
    day_of_week: commun.JourChaine = Field(
        default=None,
        description="Nouveau jour pour DÉPLACER le bloc, proprement, sans le recréer. Utilise de préférence le NOM: \"samedi\". Les numéros restent acceptés (0=Lundi..6=Dimanche).",
    )
    start_date: str | None = Field(
        default=None,
        description="Premiere date d'application YYYY-MM-DD; chaine vide '' pour retirer la borne.",
    )
    end_date: str | None = Field(
        default=None,
        description="Derniere date d'application YYYY-MM-DD (ex: la session finit le 15 octobre); chaine vide '' pour retirer la borne.",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    block_id = kwargs["block_id"]
    try:
        block = RecurringBlock.objects.get(id=block_id, user=user, active=True)
    except RecurringBlock.DoesNotExist:
        return ToolResult(success=False, data={}, message=f"Bloc #{block_id} introuvable.",
                          public="Je n'ai pas modifié le créneau : je ne l'ai pas trouvé. Dis-moi lequel.")

    # Validation choices/max_length: appliquee par le harnais contre le
    # schema type (Literal + max_length) avant execute.

    # L'etat AVANT modification, pour que le rendu dise ce qui a change.
    avant = {
        "title": block.title,
        "day_of_week": block.day_of_week,
        "start_time": block.start_time.strftime("%H:%M"),
        "end_time": block.end_time.strftime("%H:%M"),
        "flexibility": block.flexibility,
    }

    # Heures effectives après modification (parse + valide si fournies).
    try:
        new_start = parse_time(kwargs["start_time"]) if kwargs.get("start_time") else block.start_time
        new_end = parse_time(kwargs["end_time"]) if kwargs.get("end_time") else block.end_time
    except ValueError as e:
        return ToolResult(success=False, data={}, message=str(e),
                          public=("Je n'ai pas modifié le créneau : l'heure indiquée est invalide. "
                                  "Dis-moi l'heure de début et l'heure de fin."))

    # Déplacement inter-jours propre (pas de delete+create qui duplique).
    new_day = kwargs.get("day_of_week")
    if new_day is None:
        new_day = block.day_of_week
    else:
        # Meme correctif que create_block: le modele confond samedi et
        # dimanche avec la convention americaine. On accepte les noms, et
        # on REFUSE ce qu'on ne comprend pas plutot que de deviner: un
        # bloc deplace au hasard est pire qu'un refus, l'utilisateur ne le
        # verrait pas passer.
        compris = commun.normaliser_jours(new_day)
        if not compris:
            return ToolResult(success=False, data={}, message=(
                f"day_of_week invalide : {new_day!r} "
                "(nom de jour ou 0=Lundi..6=Dimanche). Corrige et réessaie."),
                public=("Je n'ai pas modifié le créneau : le jour indiqué n'est pas reconnu. "
                        "Utilise un nom de jour, par exemple lundi."))
        new_day = compris[0]

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
            data={"conflit": {
                "titre": o.title,
                "debut": o.start_time.strftime('%H:%M'),
                "fin": o.end_time.strftime('%H:%M'),
                "jour": new_day,
            }},
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
                libelle = {"start_date": "date de début", "end_date": "date de fin"}[key]
                return ToolResult(success=False, data={}, message=f"{key} invalide: attendu YYYY-MM-DD.",
                                  public=f"Je n'ai pas modifié le créneau : la {libelle} est invalide. Utilise le format AAAA-MM-JJ.")
    if block.start_date and block.end_date and block.end_date < block.start_date:
        return ToolResult(success=False, data={}, message=(
            f"Impossible : la date de fin ({block.end_date.isoformat()}) est avant "
            f"la date de début ({block.start_date.isoformat()}). "
            "Si la fin vient d'une réponse de l'utilisateur sans année, elle est "
            "peut-être dans le passé : demande-lui de confirmer l'année."),
            public=("Je n'ai pas modifié le créneau : la date de fin est avant la date de début. "
                    "Vérifie l'année de la date de fin."))
    block.flexibility = new_flexibility
    block.start_time = new_start
    block.end_time = new_end
    block.is_night_shift = night
    block.day_of_week = new_day

    block.save()
    return ToolResult(
        success=True,
        data={"block": commun._block_to_dict(block), "avant": avant},
        message=f"Bloc '{block.title}' mis à jour.",
    )
