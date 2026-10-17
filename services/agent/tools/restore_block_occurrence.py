"""Outil restore_block_occurrence."""
from django.contrib.auth.models import User

from core.models import RecurringBlockException
from . import commun
from .base import ToolResult

NAME = "restore_block_occurrence"
DESCRIPTION = (
    "Rétablit une occurrence précédemment ignorée d'un bloc récurrent (annule un skip). "
    "Ex: 'finalement je travaille ce vendredi'. Résous le bon bloc par date + type; "
    "ne demande jamais d'identifiant."
)


class Parametres(commun.ParametresOccurrence):
    pass


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
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
            message=f"Aucun bloc récurrent{kind} le {commun.DAY_NAMES[dow]} {target_date.isoformat()}.",
            public="Je n'ai rien rétabli : aucun créneau ce jour-là. Vérifie le jour ou le titre.",
        )
    if len(matches) > 1:
        candidates = commun._candidates_payload(matches, dow)
        listing = "; ".join(f"{c['title']} ({c['start_time']}-{c['end_time']})" for c in candidates)
        return ToolResult(
            success=False,
            data={"candidates": candidates},
            message=f"Plusieurs blocs le {commun.DAY_NAMES[dow]}: {listing}. Précise lequel (par titre).",
        )

    block = matches[0]
    deleted, _ = RecurringBlockException.objects.filter(
        user=user, recurring_block=block, date=target_date
    ).delete()
    if deleted:
        msg = f"'{block.title}' rétabli le {commun.DAY_NAMES[dow]} {target_date.isoformat()}."
    else:
        msg = f"'{block.title}' a bien lieu le {commun.DAY_NAMES[dow]} {target_date.isoformat()} (aucune annulation à retirer)."
    return ToolResult(
        success=True,
        data={"date": target_date.isoformat(), "title": block.title, "restored": bool(deleted)},
        message=msg,
    )
