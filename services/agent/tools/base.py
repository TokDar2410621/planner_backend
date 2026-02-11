"""
Base tool class for the Planner AI agent.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from django.contrib.auth.models import User


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    data: dict = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict:
        """Serialize for sending back to the LLM."""
        return {
            "success": self.success,
            "data": self.data,
            "message": self.message,
        }

    def to_string(self) -> str:
        """Convert to a string for the LLM to read."""
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class BaseTool(ABC):
    """Abstract base class for all agent tools."""

    name: str
    description: str
    parameters: dict  # JSON Schema format

    @abstractmethod
    def execute(self, user: User, **kwargs) -> ToolResult:
        """Execute the tool and return a result."""
        pass

    def to_claude_format(self) -> dict:
        """Convert tool definition to Claude's expected format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }
