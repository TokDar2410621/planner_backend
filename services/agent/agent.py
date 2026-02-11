"""
PlannerAgent - The AI engine for Planner AI.

Implements a multi-turn agentic loop where the LLM can call tools,
see results, and decide what to do next.
"""
import json
import logging
from typing import Optional

from django.contrib.auth.models import User

from core.models import ConversationMessage, UploadedDocument
from services.llm.claude import ClaudeProvider
from services.llm.base import LLMResponse

from .context_builder import build_context
from .system_prompt import build_system_prompt
from .tools import get_tools_for_claude, execute_tool

logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    The main AI agent for Planner AI.

    Uses Claude with tools in a multi-turn loop:
    1. Build rich context + system prompt
    2. Send message to Claude with all tools
    3. If Claude calls tools → execute them → feed results back → loop
    4. When Claude responds with text → return to user
    """

    MAX_TOOL_TURNS = 8

    def __init__(self):
        self.llm = ClaudeProvider()

    def process_message(
        self,
        user: User,
        message: str,
        attachment: Optional[UploadedDocument] = None,
    ) -> dict:
        """
        Process a user message through the agentic loop.

        Returns:
            {
                "response": str,           # The AI's text response
                "quick_replies": list,      # Contextual quick reply buttons
                "blocks_created": list,     # IDs of blocks created
                "tasks_created": list,      # Tasks created
            }
        """
        if not self.llm.is_available():
            return {
                "response": "Service IA non disponible. Vérifie la configuration de la clé API.",
                "quick_replies": [],
            }

        # 1. Save user message
        ConversationMessage.objects.create(
            user=user, role="user", content=message
        )

        # 2. Build context and system prompt
        context = build_context(user)
        system_prompt = build_system_prompt(user, context)

        # 3. Get conversation history
        history = self._get_conversation_history(user, limit=20)

        # 4. Get tools in Claude format
        tools = get_tools_for_claude()

        # 5. Add current message
        # Handle attachment context
        if attachment:
            msg_content = f"{message}\n\n[Document uploadé: {attachment.file_name} (type: {attachment.document_type})]"
        else:
            msg_content = message

        history.append({"role": "user", "content": msg_content})

        # 6. Agentic loop
        final_text = ""
        tool_calls_made = []
        interactive_inputs = None  # Captured from present_form tool
        ai_quick_replies = None    # Captured from present_quick_replies tool

        for turn in range(self.MAX_TOOL_TURNS):
            logger.info(f"Agent turn {turn + 1}/{self.MAX_TOOL_TURNS}")

            response = self.llm.generate_with_history(
                messages=history,
                tools=tools,
                system_prompt=system_prompt,
            )

            if not response.has_function_calls:
                # No tool calls - we have the final response
                final_text = response.text
                break

            # Process tool calls
            # Add assistant message with tool calls to history (raw content)
            history.append({
                "role": "assistant",
                "content": response.raw_content,
            })

            # Execute each tool and build tool results
            tool_results = []
            for fc in response.function_calls:
                logger.info(f"Executing tool: {fc.name}({fc.args})")
                result = execute_tool(fc.name, user, fc.args)
                logger.info(f"Tool result: {result.message}")

                tool_calls_made.append({
                    "tool": fc.name,
                    "args": fc.args,
                    "result": result.to_dict(),
                })

                # Capture interactive UI data from tools
                if fc.name == "present_form" and result.success:
                    interactive_inputs = result.data.get("interactive_inputs")
                if fc.name == "present_quick_replies" and result.success:
                    ai_quick_replies = result.data.get("quick_replies")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": fc.call_id,
                    "content": result.to_string(),
                })

            # Add tool results as a user message (Claude's convention)
            history.append({
                "role": "user",
                "content": tool_results,
            })

            # Capture any text from intermediate turns
            if response.text:
                final_text = response.text

        else:
            # Hit max turns - use whatever text we have
            if not final_text:
                final_text = "J'ai effectué plusieurs actions. Voici un résumé de ce que j'ai fait."
                for tc in tool_calls_made:
                    final_text += f"\n- {tc['result'].get('message', tc['tool'])}"

        # 7. Save assistant response
        ConversationMessage.objects.create(
            user=user, role="assistant", content=final_text
        )

        # 8. Build response
        # Use AI-generated quick replies if available, otherwise auto-generate
        quick_replies = ai_quick_replies or self._generate_quick_replies(tool_calls_made, context)

        result = {
            "response": final_text,
            "quick_replies": quick_replies,
            "blocks_created": [
                tc["result"]["data"].get("created", [])
                for tc in tool_calls_made
                if tc["tool"] == "create_block"
            ],
            "tasks_created": [
                tc["result"]["data"].get("task", {})
                for tc in tool_calls_made
                if tc["tool"] == "create_task"
            ],
        }

        # Add interactive inputs if the AI presented a form
        if interactive_inputs:
            result["interactive_inputs"] = interactive_inputs

        return result

    def _get_conversation_history(self, user: User, limit: int = 20) -> list[dict]:
        """Get recent conversation history formatted for Claude."""
        messages = ConversationMessage.objects.filter(
            user=user
        ).order_by("-created_at")[:limit]

        history = []
        for msg in reversed(messages):
            role = msg.role
            if role == "model":
                role = "assistant"
            history.append({
                "role": role,
                "content": msg.content,
            })

        # Ensure history starts with user message (Claude requirement)
        if history and history[0]["role"] != "user":
            history = history[1:]

        # Ensure alternating roles
        cleaned = []
        last_role = None
        for msg in history:
            if msg["role"] == last_role:
                # Merge consecutive same-role messages
                if cleaned:
                    cleaned[-1]["content"] += "\n" + msg["content"]
                continue
            cleaned.append(msg)
            last_role = msg["role"]

        return cleaned

    def _generate_quick_replies(self, tool_calls: list, context: dict) -> list[dict]:
        """Generate contextual quick reply buttons based on what just happened."""
        replies = []

        tool_names = [tc["tool"] for tc in tool_calls]

        if "create_block" in tool_names:
            replies.append({"label": "📅 Voir mon planning", "value": "Montre-moi mon planning de la semaine"})
            replies.append({"label": "🔍 Vérifier les conflits", "value": "Est-ce qu'il y a des conflits dans mon planning ?"})

        if "create_task" in tool_names:
            replies.append({"label": "📋 Mes tâches", "value": "Liste toutes mes tâches en attente"})

        if not tool_calls:
            # No tools used - suggest common actions
            if context["total_blocks"] == 0:
                replies.append({"label": "🏗️ Configurer mon planning", "value": "Je veux configurer mon emploi du temps"})
            else:
                replies.append({"label": "📊 Mes stats", "value": "Montre-moi mes statistiques de productivité"})
                replies.append({"label": "💡 Suggestions", "value": "Tu as des suggestions pour améliorer mon planning ?"})

        if context["tasks"]["pending_count"] > 0 and "list_tasks" not in tool_names:
            replies.append({"label": "✅ Mes tâches", "value": "Quelles tâches je dois faire ?"})

        # Limit to 3 quick replies
        return replies[:3]
