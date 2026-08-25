"""
Relire ce qu'on vient d'ecrire, et comparer par du code.

Aucun outil ne relit la base apres ecriture: leur `data` est construit depuis
l'instance en memoire. C'est exactement pourquoi cette etape existe.

Les chemins de date ci-dessous viennent du code REEL, outil par outil. Une
seule erreur de chemin rendrait le detecteur muet en production tout en
laissant ses tests au vert.
"""
from __future__ import annotations

import logging

from django.contrib.auth.models import User

from services.agent.tools import execute_tool
from services.agent_v2.registre import Registre

logger = logging.getLogger(__name__)

LECTURES: dict[str, str] = {
    "create_block": "list_blocks",
    "update_block": "list_blocks",
    "delete_block": "list_blocks",
    "clear_all_blocks": "list_blocks",
    "skip_block_occurrence": "get_today_schedule",
    "restore_block_occurrence": "get_today_schedule",
    "create_task": "list_tasks",
    "update_task": "list_tasks",
    "delete_task": "list_tasks",
    "complete_task": "list_tasks",
    "schedule_task_at": "get_week_schedule",
    "cancel_scheduled_block": "get_week_schedule",
    "optimize_week": "get_week_schedule",
    "organize_day": "get_today_schedule",
    "update_preferences": "get_preferences",
    "create_goal": "list_goals",
    "update_goal": "list_goals",
}

# Ou trouver la date obtenue, par outil. Chemins verifies dans le code v1.
CHEMINS_DATE: dict[str, tuple] = {
    "schedule_task_at": ("scheduled_block", "date"),
    "skip_block_occurrence": ("date",),
    "restore_block_occurrence": ("date",),
    "organize_day": ("date",),
    "optimize_week": ("start_date",),
}


def _lire(donnees: dict, chemin: tuple):
    valeur = donnees
    for cle in chemin:
        if not isinstance(valeur, dict):
            return None
        valeur = valeur.get(cle)
    return valeur


def reconcilier(user: User, registre: Registre) -> dict:
    """Relit l'etat mute. Hors du budget d'etapes du modele."""
    besoins = {LECTURES[a.outil] for a in registre.mutations() if a.outil in LECTURES}
    etat: dict = {}
    for lecture in sorted(besoins):
        try:
            etat[lecture] = execute_tool(lecture, user, {}).data
        except Exception as e:  # noqa: BLE001 - une relecture ratee ne casse pas le tour
            logger.warning("Reconciliation %s impossible: %s", lecture, e)
    return etat


def detecter_ecarts(registre: Registre) -> None:
    """Compare l'intention machine au resultat reel."""
    for action in registre.actions:
        if not action.succes or not action.est_mutation:
            continue

        chemin = CHEMINS_DATE.get(action.outil)
        if chemin:
            demandee = action.parametres.get("date") or action.parametres.get("start_date")
            obtenue = _lire(action.donnees, chemin)
            if demandee and obtenue and str(demandee) != str(obtenue):
                registre.ajouter_ecart(
                    action.id, f"date demandee {demandee}, date obtenue {obtenue}")

        # Succes SANS mutation reelle: verifie dans le code v1, quatre cas.
        if action.outil == "create_task" and "non dupliquee" in action.message.lower():
            registre.ajouter_ecart(action.id, "tache deja presente, rien n'a ete cree")
        if action.outil in ("optimize_week", "organize_day") and action.donnees.get("applied") is False:
            registre.ajouter_ecart(action.id, "plan seulement propose, rien n'a ete applique")
        if action.outil == "update_preferences" and not action.donnees.get("updated_fields"):
            registre.ajouter_ecart(action.id, "aucune preference n'a change")
        if action.outil == "restore_block_occurrence" and action.donnees.get("restored") is False:
            registre.ajouter_ecart(action.id, "aucune occurrence sautee a restaurer")
