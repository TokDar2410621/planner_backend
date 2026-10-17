"""Outil organize_day."""
from datetime import datetime

from django.contrib.auth.models import User
from pydantic import BaseModel, ConfigDict, Field

from services.scheduling.solve_day import empreinte_entrees_jour
from . import commun
from .base import ISSUE_ECHEC, ToolResult
from .commun import _executer_organize_day

NAME = "organize_day"
DESCRIPTION = (
    "Propose un placement des blocs SOUPLES "
    "d'une journée (sport, révision, repas, sommeil...) autour des murs fixes "
    "(cours, travail) et du sommeil protégé, via un solveur exact. À utiliser "
    "quand l'utilisateur demande d'optimiser / réorganiser / mieux agencer sa "
    "journée, ou quand le placement laisse des trous. apply=false PROPOSE "
    "l'arrangement sans rien changer (défaut). Après confirmation, apply=true "
    "enregistre ces horaires pour cette date seulement et conserve la série."
)


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: str = Field(
        description="Jour à réorganiser (YYYY-MM-DD). Déduis-le de la DATE du jour.",
    )
    apply: bool | None = Field(
        default=None,
        description="false = proposer (défaut). true = appliquer à cette date après confirmation.",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    try:
        target_date = datetime.strptime(kwargs["date"], "%Y-%m-%d").date()
    except (ValueError, KeyError, TypeError):
        return ToolResult(success=False, data={}, message="Format de date invalide. Utilise YYYY-MM-DD.",
                          issue=ISSUE_ECHEC, cause="date_illisible",
                          public="Je n'ai pas pu vérifier ce créneau : la date était illisible.")
    apply = bool(kwargs.get("apply", False))
    expected_entries = kwargs.pop("_expected_entries", None) or empreinte_entrees_jour(user, target_date)
    # Kwarg PRIVE (agent v2): arrangement deja resolu et valide. Evite de
    # re-resoudre a l'apply quand le plan vient d'etre verifie a
    # l'identique. Jamais fourni par le modele ni v1.
    arrangement = kwargs.pop("_arrangement", None)
    if arrangement is None:
        arrangement = commun.solve_placement(user, target_date)  # fenêtre pleine journée
    return _executer_organize_day(user, target_date, apply, arrangement, expected_entries)
