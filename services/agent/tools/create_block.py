"""Outil create_block."""
from django.contrib.auth.models import User
from django.db import transaction
from pydantic import BaseModel, ConfigDict, Field

from core.models import RecurringBlock
from services.scheduling.overlap import (
    find_recurring_conflicts,
    is_overnight,
    parse_time,
)
from . import commun
from .base import ToolResult

NAME = "create_block"
DESCRIPTION = "Crée un ou plusieurs blocs récurrents HEBDOMADAIRES (une habitude qui revient chaque semaine: cours, travail, sport, sommeil...). Jours = liste (ex: [0,1,2,3,4] = lundi à vendredi); un bloc séparé par jour. Appelle-le quand l'utilisateur décrit un horaire qui revient (mot de récurrence dit, ou un cours, un quart) avec ses jours, son début et sa fin. Un jour nommé sans mot de récurrence (« chaque », « tous les », « les lundis ») désigne un seul jour: c'est un événement unique -> schedule_task_at. Activité souple sans heure (sport, étude): choisis un créneau libre. Pour un rendez-vous, un cours, un quart, une réunion ou une leçon fixé par quelqu'un d'autre sans heure de début ou de fin: n'appelle pas cet outil, demande-lui (present_form ou present_choices). N'utilise JAMAIS create_block pour un événement unique daté ('ce samedi', une date précise) -> schedule_task_at. Gère un quart de nuit récurrent qui traverse minuit (ex: travail 22:00-06:00 chaque lundi): mets start_time/end_time tels quels, le passage de minuit est détecté automatiquement. Les conflits sont détectés et renvoyés automatiquement."


class Parametres(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(
        max_length=commun.TITLE_MAX_LENGTH,
        description="Nom du bloc (ex: 'Travail', 'Cours de biologie')",
    )
    block_type: commun.TypeBloc = Field(
        description="Type du bloc",
    )
    days: list[commun.JourChaine] = Field(
        description="Jours de la semaine. Utilise de preference les NOMS: [\"samedi\", \"dimanche\"]. Les numeros restent acceptes (0=Lundi, 1=Mardi, ..., 6=Dimanche) mais les noms evitent toute confusion de convention.",
    )
    start_time: str = Field(
        description="Heure de début au format HH:MM (ex: '09:00')",
    )
    end_time: str = Field(
        description="Heure de fin au format HH:MM (ex: '17:00')",
    )
    location: str | None = Field(
        default=None,
        max_length=commun.LOCATION_MAX_LENGTH,
        description="Lieu (optionnel)",
    )
    flexibility: commun.FlexibiliteBloc | None = Field(
        default=None,
        description="fixed si le bloc ne doit pas bouger; flexible sinon.",
    )
    is_night_shift: bool | None = Field(
        default=None,
        description="Si le bloc traverse minuit (ex: travail 22h-06h)",
    )
    start_date: str | None = Field(
        default=None,
        description="Premiere date d'application YYYY-MM-DD (ex: entrainements qui commencent le 24 aout). Omettre si des maintenant.",
    )
    end_date: str | None = Field(
        default=None,
        description="Derniere date d'application YYYY-MM-DD (fin de session/saison). Omettre si sans fin.",
    )


PARAMETERS = Parametres.model_json_schema()


def execute(user: User, **kwargs) -> ToolResult:
    title = kwargs["title"]
    block_type = kwargs["block_type"]
    # Noms ou numeros: le modele se trompait sur samedi et dimanche.
    days = commun.normaliser_jours(kwargs["days"])
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
                libelle = {"start_date": "date de début", "end_date": "date de fin"}[key]
                return ToolResult(success=False, data={}, message=f"{key} invalide: attendu YYYY-MM-DD.",
                                  public=f"Je n'ai pas créé le créneau : la {libelle} est invalide. Utilise le format AAAA-MM-JJ.")
    if bounds.get("start_date") and bounds.get("end_date") and bounds["end_date"] < bounds["start_date"]:
        return ToolResult(success=False, data={}, message=(
            f"Impossible : la date de fin ({bounds['end_date'].isoformat()}) est avant "
            f"la date de début ({bounds['start_date'].isoformat()}). "
            "Si la fin vient d'une réponse de l'utilisateur sans année, elle est "
            "peut-être dans le passé : demande-lui de confirmer l'année."),
            public=("Je n'ai pas créé le créneau : la date de fin est avant la date de début. "
                    "Vérifie l'année de la date de fin."))

    # Validation choices/max_length: appliquee par le harnais contre le
    # schema type (Literal + max_length) avant execute.

    # Parse + valide les heures une seule fois (format HH:MM).
    try:
        start_t = parse_time(start_time)
        end_t = parse_time(end_time)
    except ValueError as e:
        return ToolResult(success=False, data={}, message=str(e),
                          public=("Je n'ai pas créé le créneau : l'heure indiquée est invalide. "
                                  "Dis-moi l'heure de début et l'heure de fin."))

    # Détection night shift basée sur les minutes (pas une comparaison de chaînes).
    is_night_shift = is_overnight(start_t, end_t, is_night_shift)
    nf = (
        flexibility == RecurringBlock.FLEXIBILITY_FLEXIBLE
        if flexibility
        else RecurringBlock.default_flexibility_for(block_type) == RecurringBlock.FLEXIBILITY_FLEXIBLE
    )

    created = []
    skipped = []
    # Donnees structurees pour le rendu cote utilisateur: le motif et les
    # heures, a cote de la raison ecrite pour le modele (inchangee).
    debut_hhmm = start_t.strftime("%H:%M")
    fin_hhmm = end_t.strftime("%H:%M")

    # Opération multi-lignes: tout ou rien. Si une création échoue en cours
    # de route, aucun bloc partiel ne subsiste (D5: transaction.atomic).
    try:
        with transaction.atomic():
            for day in days:
                if day < 0 or day > 6:
                    skipped.append({
                        "day": day, "reason": "Jour invalide",
                        "motif": "jour_invalide", "titre": title,
                        "debut": debut_hhmm, "fin": fin_hhmm,
                    })
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
                        "day_name": commun.DAY_NAMES[day],
                        "reason": f"'{title}' existe déjà le {commun.DAY_NAMES[day]} à {start_time} (aucun doublon créé)",
                        "motif": "doublon",
                        "titre": title,
                        "debut": debut_hhmm,
                        "fin": fin_hhmm,
                        "avec": {"titre": title, "debut": debut_hhmm, "fin": fin_hhmm},
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
                        "day_name": commun.DAY_NAMES[day],
                        "reason": f"Chevauchement avec '{overlap.title}' ({overlap.start_time.strftime('%H:%M')}-{overlap.end_time.strftime('%H:%M')})",
                        "motif": "chevauchement",
                        "titre": title,
                        "debut": debut_hhmm,
                        "fin": fin_hhmm,
                        "avec": {
                            "titre": overlap.title,
                            "debut": overlap.start_time.strftime('%H:%M'),
                            "fin": overlap.end_time.strftime('%H:%M'),
                        },
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
                created.append(commun._block_to_dict(block))
    except Exception as e:
        # Rollback déjà effectué par le context manager: aucune ligne partielle.
        return ToolResult(
            success=False,
            data={},
            message=f"Création annulée (aucun bloc créé): {e}",
            public=("Je n'ai pas créé le créneau : une erreur technique est survenue, "
                    "aucun créneau n'a été créé. Réessaie."),
        )

    day_names = [commun.DAY_NAMES[d] for d in days if 0 <= d <= 6]
    msg = f"{len(created)} bloc(s) créé(s): {title} ({start_time}-{end_time})"
    if created:
        msg += f" les {', '.join(d['day_name'] for d in created)}"
    if skipped:
        msg += f". {len(skipped)} sauté(s): " + "; ".join(s.get("reason", "") for s in skipped)

    return ToolResult(
        success=len(created) > 0,
        data={"created": created, "skipped": skipped},
        message=msg,
        # Ne rend que si rien n'a ete cree ni saute (aucun jour valide) :
        # les sauts passent par la branche dediee du rendu.
        public=("Je n'ai pas créé le créneau : aucun jour valide. "
                "Utilise des noms de jour, par exemple lundi."),
    )
