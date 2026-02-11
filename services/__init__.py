"""
Services package for Planner AI backend.

Architecture:
- PlannerAgent (services.agent): Agentic AI loop with tool use
- DocumentProcessor: Async document processing
- AIScheduler: Schedule generation

LLM Providers (services.llm):
- ClaudeProvider: Anthropic Claude API (primary)
"""
from .document_processor import DocumentProcessor
from .ai_scheduler import AIScheduler
from .calendar_sync import CalendarSync

__all__ = [
    'DocumentProcessor',
    'AIScheduler',
    'CalendarSync',
]
