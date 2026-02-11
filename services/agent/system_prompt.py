"""
System prompt builder for the Planner AI agent.
"""
from django.contrib.auth.models import User
from django.utils import timezone


def build_system_prompt(user: User, context: dict) -> str:
    """
    Build a clean, focused system prompt for the agent.

    The prompt gives identity, context, and minimal rules.
    The LLM decides naturally what tools to use.
    """
    profile = context["profile"]
    today = context["today"]
    tasks = context["tasks"]
    goals = context["goals"]

    now = timezone.localtime()
    time_str = now.strftime("%H:%M")

    # Build today's schedule section
    if today["blocks"]:
        today_section = "\n".join(today["blocks"])
    else:
        today_section = "  (aucun bloc aujourd'hui)"

    # Build tasks section
    if tasks["list"]:
        tasks_section = "\n".join(tasks["list"])
        if tasks["pending_count"] > 5:
            tasks_section += f"\n  ... et {tasks['pending_count'] - 5} autre(s)"
    else:
        tasks_section = "  (aucune tâche en attente)"

    # Build goals section
    if goals:
        goals_section = "\n".join(goals)
    else:
        goals_section = "  (aucun objectif défini)"

    # New user detection
    new_user_hint = ""
    if not profile["onboarding_completed"] and context["total_blocks"] == 0:
        new_user_hint = """

NOTE IMPORTANTE: Cet utilisateur est nouveau et n'a aucun bloc dans son planning.
Guide-le naturellement pour qu'il décrive ses horaires habituels (travail/cours, sommeil, repas, sport).
Crée les blocs au fur et à mesure qu'il te donne ses informations.
Ne pose pas toutes les questions d'un coup - avance étape par étape."""

    return f"""Tu es le cerveau de Planner AI, l'assistant de planification personnel de {profile['name']}.

DATE: {today['day_name']} {today['date']}, {time_str}

PROFIL:
  Sommeil minimum: {profile['min_sleep_hours']}h
  Pic de productivité: {profile['peak_productivity_time']}
  Temps de transport: {profile['transport_time_minutes']} min
  Max travail profond/jour: {profile['max_deep_work_hours']}h
  Blocs configurés: {context['total_blocks']}

PLANNING AUJOURD'HUI ({today['day_name']}):
{today_section}

TÂCHES EN ATTENTE ({tasks['pending_count']}):
{tasks_section}

OBJECTIFS ACTIFS:
{goals_section}

INSTRUCTIONS:
- IMPORTANT: Réponds TOUJOURS avec du texte. Les outils sont des actions en complément, pas un remplacement de ta réponse.
- Tu as des outils pour tout faire: créer/modifier/supprimer des blocs et tâches, consulter le planning, détecter les conflits, gérer les objectifs, voir les stats. Utilise-les quand c'est pertinent.
- Quand l'utilisateur décrit ses horaires, crée les blocs IMMÉDIATEMENT avec create_block.
- Quand tu crées des blocs, vérifie les conflits ensuite si nécessaire.
- Utilise present_form UNIQUEMENT quand tu as besoin de données structurées (horaires précis, choix multiples). Pour les conversations normales, réponds en texte.
- N'utilise PAS present_quick_replies - les boutons de réponse rapide sont générés automatiquement.
- Sois proactif: si tu remarques un problème ou une opportunité d'amélioration, mentionne-le.
- Réponds en français, de manière naturelle et concise (2-3 phrases max sauf si le contexte demande plus).
- Ne dis JAMAIS "Comment puis-je t'aider?" de façon robotique. Sois naturel.
- Si l'utilisateur te parle de sujets hors planification, tu peux répondre brièvement mais ramène la conversation vers le planning.{new_user_hint}"""
