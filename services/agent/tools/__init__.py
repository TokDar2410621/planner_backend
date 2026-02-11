"""
Tool registry for the Planner AI agent.

Collects all tools, converts to Claude format, and dispatches execution.
"""
import json
import logging
from typing import Optional

from django.contrib.auth.models import User

from .base import BaseTool, ToolResult
from .blocks import ListBlocksTool, CreateBlockTool, UpdateBlockTool, DeleteBlockTool, ClearAllBlocksTool
from .tasks import ListTasksTool, CreateTaskTool, UpdateTaskTool, DeleteTaskTool, CompleteTaskTool
from .schedule import GetTodayScheduleTool, GetWeekScheduleTool, FindFreeSlotsTool
from .preferences import GetPreferencesTool, UpdatePreferencesTool
from .goals import ListGoalsTool, CreateGoalTool, UpdateGoalTool
from .planning import SuggestOptimizationTool, DetectConflictsTool
from .analytics import GetProductivityStatsTool
from .interactive import PresentFormTool, PresentQuickRepliesTool

logger = logging.getLogger(__name__)

# All available tools
ALL_TOOLS: list[BaseTool] = [
    # Blocks
    ListBlocksTool(),
    CreateBlockTool(),
    UpdateBlockTool(),
    DeleteBlockTool(),
    ClearAllBlocksTool(),
    # Tasks
    ListTasksTool(),
    CreateTaskTool(),
    UpdateTaskTool(),
    DeleteTaskTool(),
    CompleteTaskTool(),
    # Schedule
    GetTodayScheduleTool(),
    GetWeekScheduleTool(),
    FindFreeSlotsTool(),
    # Preferences
    GetPreferencesTool(),
    UpdatePreferencesTool(),
    # Goals
    ListGoalsTool(),
    CreateGoalTool(),
    UpdateGoalTool(),
    # Planning
    SuggestOptimizationTool(),
    DetectConflictsTool(),
    # Analytics
    GetProductivityStatsTool(),
    # Interactive UI
    PresentFormTool(),
    PresentQuickRepliesTool(),
]

# Index by name for fast lookup
TOOL_MAP: dict[str, BaseTool] = {tool.name: tool for tool in ALL_TOOLS}


def get_tools_for_claude() -> list[dict]:
    """Convert all tools to Claude's tool format."""
    return [tool.to_claude_format() for tool in ALL_TOOLS]


def execute_tool(tool_name: str, user: User, args: dict) -> ToolResult:
    """Execute a tool by name with the given arguments."""
    tool = TOOL_MAP.get(tool_name)
    if not tool:
        return ToolResult(
            success=False,
            data={},
            message=f"Outil inconnu: {tool_name}",
        )
    try:
        return tool.execute(user, **args)
    except Exception as e:
        logger.error(f"Tool execution error ({tool_name}): {e}", exc_info=True)
        return ToolResult(
            success=False,
            data={},
            message=f"Erreur lors de l'exécution de {tool_name}: {str(e)}",
        )
