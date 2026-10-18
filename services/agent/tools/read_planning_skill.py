"""Charge un guide produit court uniquement quand la demande le requiert."""
from pathlib import Path
from typing import Literal

from django.contrib.auth.models import User
from pydantic import BaseModel

from .base import ToolResult

NAME = 'read_planning_skill'
DESCRIPTION = (
    "Lis le guide de décision rappels/tâches/récurrence quand la demande porte "
    "sur ces sujets. Le guide détaille les règles que le prompt permanent résume."
)


class Parametres(BaseModel):
    guide: Literal['rappels-et-taches', 'repetition-et-portee']


PARAMETERS = Parametres.model_json_schema()
ROOT = Path(__file__).resolve().parents[2] / 'agent_v2' / 'skills'


def execute(user: User, **kwargs) -> ToolResult:
    guide = kwargs['guide']
    content = (ROOT / f'{guide}.md').read_text(encoding='utf-8')
    return ToolResult(True, data={'guide': guide, 'content': content},
                      message=content)
