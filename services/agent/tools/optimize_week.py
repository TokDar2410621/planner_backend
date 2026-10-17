"""Outil optimize_week."""
from datetime import datetime

from django.contrib.auth.models import User
from django.utils import timezone
from pydantic import BaseModel, ConfigDict, Field

from services.scheduling.solve_day import empreinte_entrees_semaine
from .base import ISSUE_ECHEC, ToolResult
from .commun import (
    _resoudre_semaine, _mettre_en_forme_semaine, _appliquer_periode,
    _resultat_semaine,
)

NAME = "optimize_week"
DESCRIPTION = (
    "Résout et RÉORGANISE de façon OPTIMALE les blocs SOUPLES de TOUTE LA "
    "SEMAINE (7 jours consécutifs), jour par jour, autour des murs fixes "
    "(cours, travail) et du sommeil protégé, via le solveur exact. À utiliser "
    "quand l'utilisateur demande d'optimiser / réorganiser / mieux agencer sa "
    "SEMAINE (pour UN seul jour → organize_day). apply=false PROPOSE le plan "
    "sans rien changer (défaut); apply=true l'APPLIQUE en ajustant les heures "
    "des occurrences souples sur ces 7 dates seulement. Les horaires des "
    "séries et des semaines suivantes restent les mêmes. Ne touche jamais aux blocs fixes."
)


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start_date: str | None = Field(
        default=None,
        description="Premier jour de la passe (YYYY-MM-DD). Défaut: aujourd'hui.",
    )
    apply: bool | None = Field(
        default=None,
        description="true = appliquer (déplacer les blocs); false = proposer seulement (défaut).",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    raw = kwargs.get("start_date")
    if raw:
        try:
            start = datetime.strptime(raw, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return ToolResult(success=False, data={}, message="Format de date invalide. Utilise YYYY-MM-DD.",
                          issue=ISSUE_ECHEC, cause="date_illisible",
                          public="Je n'ai pas pu vérifier ce créneau : la date était illisible.")
    else:
        start = timezone.localdate()
    apply = bool(kwargs.get("apply", False))
    expected_entries = kwargs.pop("_expected_entries", None) or empreinte_entrees_semaine(user, start)
    # Kwarg PRIVE (agent v2): arrangement deja resolu et valide par la
    # garde. Evite de re-resoudre 7 jours a l'apply quand le plan vient
    # d'etre verifie a l'identique. Jamais fourni par le modele ni v1.
    arrangements = kwargs.pop("_arrangements", None)
    if arrangements is None:
        days_data, lines, skipped_total, arrangements = _resoudre_semaine(user, start)
    else:
        days_data, lines, skipped_total = _mettre_en_forme_semaine(user, start, arrangements)

    moved_total = 0
    if apply:
        try:
            moved_days = _appliquer_periode(user, start, arrangements, expected_entries)
        except ValueError as exc:
            return ToolResult(success=False, data={"applied": False, "reason": str(exc),
                                                   "start_date": start.isoformat()},
                              message="Ce plan ne peut plus être appliqué tel quel. Rien n'a changé. Demande une nouvelle proposition.",
                              public="Je n'ai pas appliqué ce plan : il n'est plus valable tel quel. Demande une nouvelle proposition.")
        days_data, lines, skipped_total = _mettre_en_forme_semaine(user, start, arrangements)
        for jour, moved in zip(days_data, moved_days):
            jour["moved"] = moved
            moved_total += len(moved)

    return _resultat_semaine(
        apply=apply, start=start, days_data=days_data, lines=lines,
        moved_total=moved_total, skipped_total=skipped_total,
    )
