"""Outil skip_block_occurrence."""
from django.contrib.auth.models import User

from core.models import RecurringBlockException
from . import commun
from .base import ToolResult

NAME = "skip_block_occurrence"
DESCRIPTION = (
    "Ignore/annule UNE SEULE occurrence d'un bloc récurrent pour une date précise "
    "(ex: 'ce vendredi je ne travaille pas', 'pas de sport demain'). Le bloc récurrent "
    "reste en place les autres semaines. Résous toi-même le bon bloc avec la date + le "
    "type (et le titre si plusieurs blocs le même jour). Ne demande JAMAIS d'identifiant. "
    "N'utilise ni delete_block ni update_block pour annuler un seul jour."
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
            message=f"Aucun bloc récurrent{kind} le {commun.DAY_NAMES[dow]} {target_date.isoformat()}. Rien à ignorer.",
            public="Je n'ai rien annulé : aucun créneau ce jour-là. Vérifie le jour ou le titre.",
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
    RecurringBlockException.objects.get_or_create(
        user=user, recurring_block=block, date=target_date
    )
    return ToolResult(
        success=True,
        data={"date": target_date.isoformat(), "title": block.title,
              "block_type": block.block_type,
              # Le creneau libere: le code demande ce qui prend sa place
              # quand le message annoncait un remplacant.
              "start_time": block.start_time.strftime("%H:%M"),
              "end_time": block.end_time.strftime("%H:%M")},
        message=f"'{block.title}' ignoré le {commun.DAY_NAMES[dow]} {target_date.isoformat()}. Le bloc récurrent reste actif les autres semaines.",
    )
