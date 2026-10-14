"""
Base tool class for the Planner AI agent.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from django.contrib.auth.models import User


# Issues metier partagees (audit 2026-10-13). Le booleen `success` ne decrit
# pas l'issue metier: une confirmation en attente, une cible absente et une
# panne valent toutes success=False. Reference: OpenCode Tool.Result, qui
# porte a la fois un `output` structure et un `content` lisible ecrit par
# l'outil lui-meme.
ISSUE_TERMINEE = "terminee"
ISSUE_PARTIELLE = "partielle"
ISSUE_AUCUN_CHANGEMENT = "aucun_changement"
ISSUE_BESOIN_PRECISION = "besoin_precision"
ISSUE_BESOIN_CONFIRMATION = "besoin_confirmation"
ISSUE_CAPACITE_ABSENTE = "capacite_absente"
ISSUE_ECHEC = "echec"
ISSUE_EFFET_A_VERIFIER = "effet_a_verifier"

ISSUES_METIER = frozenset({
    ISSUE_TERMINEE, ISSUE_PARTIELLE, ISSUE_AUCUN_CHANGEMENT,
    ISSUE_BESOIN_PRECISION, ISSUE_BESOIN_CONFIRMATION,
    ISSUE_CAPACITE_ABSENTE, ISSUE_ECHEC, ISSUE_EFFET_A_VERIFIER,
})


@dataclass
class ToolResult:
    """Result of a tool execution.

    `success`/`data`/`message` restent pour compatibilite. `issue` dit
    l'issue metier (voir ISSUES_METIER), `cause` un code de cause stable,
    `public` l'explication affichable ecrite PAR L'OUTIL: le rendu la relaie
    telle quelle, il ne l'invente pas et ne recopie jamais `message` brut
    (qui peut contenir une exception ou une instruction).
    """
    success: bool
    data: dict = field(default_factory=dict)
    message: str = ""
    issue: str = ""
    cause: str = ""
    public: str = ""

    def to_dict(self) -> dict:
        """Serialize for sending back to the LLM."""
        return {
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "issue": self.issue,
            "cause": self.cause,
            "public": self.public,
        }

    def to_string(self) -> str:
        """Convert to a string for the LLM to read."""
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


def validate_choice(value: Any, allowed, field_name: str) -> str | None:
    """Enforce a choices/enum constraint at the tool layer.

    The JSON schema `enum` is only advisory for the LLM; a hallucinated or
    injected tool-call can still pass an out-of-enum value. SQLite does not
    enforce `choices`, so bad data would land silently in dev (and raise a
    DataError in Postgres prod). Return an error message if invalid, else None.
    """
    if value is not None and value not in allowed:
        return (
            f"{field_name} invalide: '{value}'. "
            f"Valeurs autorisées: {', '.join(str(a) for a in sorted(allowed))}."
        )
    return None


def validate_max_length(value: Any, max_length: int, field_name: str) -> str | None:
    """Enforce a CharField max_length at the tool layer.

    SQLite silently accepts over-length strings (D1); Postgres raises DataError.
    Validate before create()/save() so behavior is consistent. Returns an error
    message if the string is too long, else None.
    """
    if isinstance(value, str) and len(value) > max_length:
        return (
            f"{field_name} trop long ({len(value)} caractères, "
            f"maximum {max_length})."
        )
    return None


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
