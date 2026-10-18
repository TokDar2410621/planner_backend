"""Registre des outils fonctions de l'agent Planner.

Chaque outil est un module qui expose NAME, DESCRIPTION, PARAMETERS et
``execute(user, **kwargs) -> ToolResult`` (style OpenCode : pas de
classes, zero indirection). PARAMETERS est genere depuis
``module.Parametres`` (pydantic, source de verite typee).

Le harnais (harnais.py) applique autour de chaque appel : crochets avant
execution, validation du schema type, contrat d'erreur type. La v1 passe
par ``execute_tool`` ; la v2 travaille directement sur les modules via
le harnais.
"""
import logging

from . import harnais
from .base import ToolResult

from . import list_blocks
from . import create_block
from . import update_block
from . import delete_block
from . import clear_all_blocks
from . import skip_block_occurrence
from . import replace_block_occurrence
from . import restore_block_occurrence
from . import list_tasks
from . import create_task
from . import update_task
from . import delete_task
from . import complete_task
from . import get_today_schedule
from . import get_week_schedule
from . import find_free_slots
from . import schedule_task_at
from . import check_feasibility
from . import optimize_week
from . import organize_day
from . import cancel_scheduled_block
from . import get_preferences
from . import update_preferences
from . import list_goals
from . import create_goal
from . import update_goal
from . import suggest_schedule_optimization
from . import detect_conflicts
from . import get_productivity_stats
from . import send_notification
from . import manage_reminder
from . import list_reminders
from . import manage_recurrence
from . import list_recurrences
from . import read_planning_skill
from . import process_uploaded_document
from . import resolve_document_entry
from . import present_form
from . import present_choices
from . import poser_question

logger = logging.getLogger(__name__)

# Tous les outils, dans l'ordre historique de l'ancien registre.
ALL_OUTILS = [
    # Blocs
    list_blocks,
    create_block,
    update_block,
    delete_block,
    clear_all_blocks,
    skip_block_occurrence,
    replace_block_occurrence,
    restore_block_occurrence,
    # Taches
    list_tasks,
    create_task,
    update_task,
    delete_task,
    complete_task,
    # Horaires
    get_today_schedule,
    get_week_schedule,
    find_free_slots,
    schedule_task_at,
    check_feasibility,
    optimize_week,
    organize_day,
    cancel_scheduled_block,
    # Preferences
    get_preferences,
    update_preferences,
    # Objectifs
    list_goals,
    create_goal,
    update_goal,
    # Planification
    suggest_schedule_optimization,
    detect_conflicts,
    # Analyse
    get_productivity_stats,
    # Notifications
    send_notification,
    manage_reminder,
    list_reminders,
    manage_recurrence,
    list_recurrences,
    read_planning_skill,
    # Documents : lecture et cloture des precisions d'un import.
    process_uploaded_document,
    resolve_document_entry,
    # Interface interactive
    present_form,
    present_choices,
    poser_question,
]

# Index par nom pour recherche rapide.
OUTIL_MAP = {module.NAME: module for module in ALL_OUTILS}
assert len(OUTIL_MAP) == len(ALL_OUTILS), "doublon de NAME dans le registre"

# Noms historiques conserves pour les importeurs (v1, v2, tests).
ALL_TOOLS = ALL_OUTILS
TOOL_MAP = OUTIL_MAP

# Outils que seul l'agent v2 sait relayer. present_choices et poser_question
# rangent leur question dans une DEMANDE que v2 transforme en
# done.quick_replies; v1 n'a pas ce relais et poserait une question sans
# boutons. Ils restent dans ALL_OUTILS (v2 les expose via
# services/agent_v2/outils.py) mais sortent de toute liste offerte a v1.
V2_SEULEMENT = {"present_choices", "poser_question",
                "manage_reminder", "list_reminders", "manage_recurrence",
                "list_recurrences", "read_planning_skill",
                # Outil compose ajoute pour la boucle unique: v2 le rend en une
                # ligne de faits, v1 ne le connait pas.
                "replace_block_occurrence", "resolve_document_entry"}


def get_tools_for_claude() -> list[dict]:
    """Definitions d'outils au format du modele (v1: V2_SEULEMENT exclus)."""
    return [harnais.en_claude(module) for module in ALL_OUTILS
            if module.NAME not in V2_SEULEMENT]


def execute_tool(tool_name: str, user, args: dict) -> ToolResult:
    """Execute un outil par son nom. Chemin de la v1, via le harnais."""
    module = OUTIL_MAP.get(tool_name)
    if module is None:
        return ToolResult(
            success=False,
            data={},
            message=f"Outil inconnu: {tool_name}",
        )
    return harnais.appeler(module, user, args or {})
