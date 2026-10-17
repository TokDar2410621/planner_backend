"""Outil check_feasibility."""
from datetime import datetime

from django.contrib.auth.models import User
from pydantic import BaseModel, ConfigDict, Field

from services.scheduling.solve_day import solve_day
from .base import (
    ISSUE_TERMINEE, ISSUE_BESOIN_PRECISION, ISSUE_ECHEC, ToolResult,
)
from . import commun
from .commun import DAY_NAMES, DAY_START_MIN, DAY_END_MIN, _minutes_to_str

NAME = "check_feasibility"
DESCRIPTION = (
    "Vérifie EXACTEMENT (solveur) si un ensemble d'activités tient dans une "
    "journée donnée, autour de l'emploi du temps existant (murs fixes + "
    "sommeil protégé). À utiliser AVANT d'affirmer qu'une charge 'tient', "
    "surtout sur une journée chargée: ne juge JAMAIS la faisabilité de tête. "
    "Renvoie ce qui rentre (avec un créneau suggéré) et ce qui ne rentre pas. "
    "Ex: 3h libres fragmentées en 3x1h ne peuvent PAS accueillir un bloc de 2h."
)


class Activite(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(
        description="Nom de l'activité.",
    )
    duration_minutes: int = Field(
        description="Durée en minutes.",
    )


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: str = Field(
        description="Jour à tester (YYYY-MM-DD). Déduis-le de la DATE du jour.",
    )
    activities: list[Activite] = Field(
        description="Activités à faire tenir ce jour-là.",
    )


PARAMETERS = commun.inline_refs(Parametres.model_json_schema())


def execute(user: User, **kwargs) -> ToolResult:
    try:
        target_date = datetime.strptime(kwargs["date"], "%Y-%m-%d").date()
    except (ValueError, KeyError, TypeError):
        return ToolResult(success=False, data={}, message="Format de date invalide. Utilise YYYY-MM-DD.",
                          issue=ISSUE_ECHEC, cause="date_illisible",
                          public="Je n'ai pas pu vérifier ce créneau : la date était illisible.")

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
        return ToolResult(success=False, data={}, message="Aucune activité valide à tester (titre + durée en minutes).",
                          issue=ISSUE_BESOIN_PRECISION, cause="activites_invalides",
                          public="Je n'ai rien à vérifier : il me faut au moins une activité avec un titre et une durée.")

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
    # Contrat d'issue (audit 2026-10-13): le `msg` ci-dessus est deja
    # ecrit pour etre lu, on le declare affichable tel quel.
    public = (
        f"Vérification du {day_name} {target_date.isoformat()} : "
        + (f"tout rentre ({placed_str})."
           if result["feasible"] else
           f"tout ne rentre pas. Rentre : {placed_str}. "
           f"Ne rentre pas : {unplaced_str}.")
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
        issue=ISSUE_TERMINEE, cause="", public=public,
    )
