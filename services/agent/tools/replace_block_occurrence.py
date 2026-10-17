"""Outil replace_block_occurrence."""
from django.contrib.auth.models import User
from pydantic import BaseModel, ConfigDict, Field

from core.models import RecurringBlock, RecurringBlockException
from . import commun
from .base import ToolResult

NAME = "replace_block_occurrence"
DESCRIPTION = (
    "Remplace UNE occurrence d'un bloc récurrent, à une date précise, par un "
    "événement ponctuel : un examen à la place du cours, une réunion à la place "
    "du quart, une sortie à la place du gym. Fait les DEUX gestes d'un coup et "
    "dans le bon ordre : l'occurrence de ce jour-là est annulée (les autres "
    "semaines restent en place), puis le remplaçant prend le créneau libéré. "
    "Sans heures données, il reprend celles du bloc remplacé. N'enchaîne jamais "
    "skip_block_occurrence puis schedule_task_at pour ça, et n'utilise jamais "
    "update_block, qui renommerait ou déplacerait TOUTE la série. Résous "
    "toi-même le bloc visé avec la date, le type et le titre; ne demande jamais "
    "d'identifiant."
)


class Parametres(commun.ParametresOccurrence):
    replacement_title: str = Field(
        description="Titre de ce qui prend la place (ex: 'Examen').",
    )
    start_time: str | None = Field(
        default=None,
        description="Heure de début du remplaçant (HH:MM). Par défaut celle du bloc remplacé.",
    )
    end_time: str | None = Field(
        default=None,
        description="Heure de fin du remplaçant (HH:MM). Par défaut celle du bloc remplacé.",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    from django.db import transaction

    from . import schedule_task_at as _m

    remplacant = (kwargs.get("replacement_title") or "").strip()
    if not remplacant:
        return ToolResult(success=False, data={},
                          message="Titre du remplaçant requis.",
                          public="Je n'ai rien remplacé : le titre du remplaçant est manquant. Dis-moi ce qui prend la place.")

    target_date, err = commun._parse_occurrence(kwargs)
    if err:
        return err

    dow = target_date.weekday()
    block_type = kwargs.get("block_type")
    matches = commun._resolve_day_blocks(user, target_date, block_type, kwargs.get("title"))
    if not matches:
        kind = f" '{block_type}'" if block_type else ""
        return ToolResult(
            success=False,
            data={},
            message=(f"Aucun bloc récurrent{kind} le {commun.DAY_NAMES[dow]} "
                     f"{target_date.isoformat()}. Rien à remplacer."),
            public="Je n'ai rien remplacé : aucun créneau ce jour-là. Vérifie le jour ou le titre.",
        )
    if len(matches) > 1:
        candidates = commun._candidates_payload(matches, dow)
        listing = "; ".join(f"{c['title']} ({c['start_time']}-{c['end_time']})"
                            for c in candidates)
        return ToolResult(
            success=False,
            data={"candidates": candidates},
            message=(f"Plusieurs blocs le {commun.DAY_NAMES[dow]}: {listing}. "
                     "Précise lequel (par titre)."),
        )

    block = matches[0]
    debut = (kwargs.get("start_time") or "").strip() or block.start_time.strftime("%H:%M")
    fin = (kwargs.get("end_time") or "").strip() or block.end_time.strftime("%H:%M")

    # L'ordre compte: libérer AVANT de placer. Dans l'autre sens, le bloc
    # qu'on remplace est lui-même compté comme un conflit et le placement
    # échoue (parcours vécu le 2026-09-29).
    with transaction.atomic():
        RecurringBlockException.objects.get_or_create(
            user=user, recurring_block=block, date=target_date)
        from . import harnais
        place = harnais.appeler(_m, user, {
            "title": remplacant,
            "date": target_date.isoformat(),
            "start_time": debut,
            "end_time": fin,
            "description": kwargs.get("description", ""),
        })
        if not place.success:
            # Tout ou rien: sans remplaçant placé, l'occurrence reste.
            transaction.set_rollback(True)

    if not place.success:
        return ToolResult(
            success=False,
            data=dict(place.data or {}),
            message=(f"{remplacant} n'a pas pu prendre la place de "
                     f"'{block.title}': {place.message}"),
            # La raison du placement echoue est celle de l'outil interne :
            # on la relaie, les branches specifiques (conflit, ...) restent prioritaires.
            public=place.public,
        )

    donnees = dict(place.data or {})
    donnees["date"] = target_date.isoformat()
    donnees["remplace"] = {
        "title": block.title,
        "block_type": block.block_type,
        "start_time": block.start_time.strftime("%H:%M"),
        "end_time": block.end_time.strftime("%H:%M"),
    }
    return ToolResult(
        success=True,
        data=donnees,
        message=(f"'{remplacant}' prend la place de '{block.title}' le "
                 f"{commun.DAY_NAMES[dow]} {target_date.isoformat()} de {debut} à "
                 f"{fin}. Le bloc récurrent reste actif les autres semaines."),
    )
