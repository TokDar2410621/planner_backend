"""Outil detect_conflicts."""
from django.contrib.auth.models import User
from pydantic import BaseModel, ConfigDict, Field

from core.models import RecurringBlock
from .base import ISSUE_TERMINEE, ToolResult
from . import commun
from .commun import DAY_NAMES, _time_to_minutes

NAME = "detect_conflicts"
DESCRIPTION = "Détecte les chevauchements et conflits dans le planning de l'utilisateur."


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    day_of_week: commun.JourChaine = Field(
        default=None,
        description="Vérifier un jour spécifique. Nom de préférence (\"samedi\"); les numéros restent acceptés (0=Lundi..6=Dimanche). Si absent, vérifie tous les jours.",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    # Nom ou numero, comme partout ailleurs depuis le correctif du
    # decalage +1: une convention par outil serait une invitation a se
    # tromper.
    from .commun import normaliser_jours
    jours = normaliser_jours(kwargs.get("day_of_week"))
    blocks = RecurringBlock.objects.filter(user=user, active=True)

    if jours:
        days_to_check = [jours[0]]
    else:
        days_to_check = range(7)

    conflicts = []

    for d in days_to_check:
        day_blocks = list(blocks.filter(day_of_week=d).order_by("start_time"))

        for i in range(len(day_blocks)):
            for j in range(i + 1, len(day_blocks)):
                a = day_blocks[i]
                b = day_blocks[j]

                a_start = _time_to_minutes(a.start_time)
                a_end = _time_to_minutes(a.end_time)
                b_start = _time_to_minutes(b.start_time)
                b_end = _time_to_minutes(b.end_time)

                # Handle non-night-shift blocks
                if not a.is_night_shift and not b.is_night_shift:
                    if a_start < b_end and b_start < a_end:
                        overlap_start = max(a_start, b_start)
                        overlap_end = min(a_end, b_end)
                        conflicts.append({
                            "day": d,
                            "day_name": DAY_NAMES[d],
                            "block_a": {"id": a.id, "title": a.title, "time": f"{a.start_time.strftime('%H:%M')}-{a.end_time.strftime('%H:%M')}"},
                            "block_b": {"id": b.id, "title": b.title, "time": f"{b.start_time.strftime('%H:%M')}-{b.end_time.strftime('%H:%M')}"},
                            "overlap_minutes": overlap_end - overlap_start,
                        })

    # Contrat d'issue (audit 2026-10-13): les titres sont des donnees
    # lues, on les nomme sans rien inventer.
    if conflicts:
        lignes = "\n".join(
            f"- {c['day_name']} : {c['block_a']['title']} "
            f"({c['block_a']['time']}) chevauche {c['block_b']['title']} "
            f"({c['block_b']['time']})"
            for c in conflicts)
        public = f"{len(conflicts)} conflit(s) dans ton horaire :\n{lignes}"
    else:
        public = "Aucun conflit détecté dans ton horaire."
    return ToolResult(
        success=True,
        data={"conflicts": conflicts, "count": len(conflicts)},
        message=f"{len(conflicts)} conflit(s) détecté(s)." if conflicts else "Aucun conflit détecté.",
        issue=ISSUE_TERMINEE, cause="", public=public,
    )
