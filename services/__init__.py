"""
Services package for Planner AI backend.

Architecture:
- ChatOrchestrator: Lightweight coordinator (NEW - recommended)
- ChatEngine: Legacy monolith (still functional for backward compatibility)

Specialized Services:
- SecurityService: Input validation, rate limiting
- OnboardingService: New user setup flow
- PlanningService: Schedule management

LLM Providers (services.llm):
- GeminiProvider: Google Gemini API
- (Future: OpenAIProvider, ClaudeProvider, etc.)
"""
from .document_processor import DocumentProcessor
from .chat_engine import ChatEngine
from .ai_scheduler import AIScheduler
from .calendar_sync import CalendarSync

# New refactored services
from .security import SecurityService
from .onboarding import OnboardingService
from .planning import PlanningService
from .chat_orchestrator import ChatOrchestrator

__all__ = [
    # Core services
    'DocumentProcessor',
    'AIScheduler',
    'CalendarSync',
    # Chat (new + legacy)
    'ChatOrchestrator',  # NEW - recommended
    'ChatEngine',        # Legacy - backward compatible
    # Specialized
    'SecurityService',
    'OnboardingService',
    'PlanningService',
]
