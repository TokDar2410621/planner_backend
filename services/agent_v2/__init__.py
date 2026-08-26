"""
Agent v2: la boucle ou le recit d'action est vrai par construction.
Voir docs/superpowers/specs/2026-08-24-agent-v2-design.md.

Ce paquet vit A COTE de services/agent/, qui sert la production. Aucun import
d'ici ne doit modifier le comportement de v1.
"""

from services.agent_v2.agent import PlannerAgentV2  # noqa: E402,F401

__all__ = ["PlannerAgentV2"]
