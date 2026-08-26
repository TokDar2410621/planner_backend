"""
Épreuve « ton et format »: la seule notée par un juge LLM, sur grille stricte.

Le juge est un modèle DIFFÉRENT de celui qui répond (Gemini juge, DeepSeek
répond) et il ne voit jamais quel agent a produit le texte: il note des
réponses anonymes sur cinq critères binaires.
"""
from __future__ import annotations

import json
import re

from .harness import Note, Pilote, monde_neuf

GRILLE = """Tu notes des réponses d'un assistant de planification québécois.
Critères, chacun vaut 1 point, réponds STRICTEMENT par un JSON.

1. tutoiement: la réponse tutoie (jamais « vous »).
2. concision: pas de délayage, pas de répétition de la question, va au fait.
3. non_culpabilisant: aucun reproche sur ce qui a été raté; ton qui pardonne.
4. concret: nomme des heures, des titres, des jours plutôt que du vague.
5. pas_de_promesse: n'annonce aucun travail futur ou « en cours ».

Format EXACT: {"tutoiement":0|1,"concision":0|1,"non_culpabilisant":0|1,"concret":0|1,"pas_de_promesse":0|1}
"""

MESSAGES = [
    "J'ai rien fait de ma journée, j'ai tout raté.",
    "C'est quoi mon planning demain ?",
    "Annule mon cours de maths de lundi.",
]


def epreuve_ton(p: Pilote) -> Note:
    n = Note("Ton et format", 0, 0)
    m = monde_neuf("ton")
    m.bloc("Mathématiques", 0, "09:00", "12:00")
    m.bloc("Physique", 2, "09:00", "12:00")

    for msg in MESSAGES:
        t = p.envoyer(m.user, msg)
        n.tours.append(t)
        scores = _juger(msg, t.reponse)
        for critere, valeur in scores.items():
            n.point(bool(valeur), f"{critere} sur « {msg[:32]}… »", 10 / (5 * len(MESSAGES)))
    return n


def _juger(question: str, reponse: str) -> dict:
    from services.llm import get_provider
    if not reponse:
        return {c: 0 for c in ("tutoiement", "concision", "non_culpabilisant",
                               "concret", "pas_de_promesse")}
    prompt = (f"{GRILLE}\n\nQuestion de l'utilisateur:\n{question}\n\n"
              f"Réponse à noter:\n{reponse[:1500]}\n\nJSON:")
    try:
        # Le juge est Gemini, jamais le modèle qui a répondu.
        res = get_provider("gemini").generate(prompt)
        brut = getattr(res, "text", "") or ""
        trouve = re.search(r"\{[^{}]+\}", brut)
        return json.loads(trouve.group(0)) if trouve else {}
    except Exception as e:  # noqa: BLE001
        # Un echec de PARSING n'est pas une reponse mauvaise: sans ce dict
        # complet, possibles restait a zero et l'epreuve rendait 0/10 sans
        # qu'aucune ligne du rapport ne l'explique. Vecu: la note de ton a
        # derive de 9,3 a 8,0 sur trois passages sans cause visible.
        print(f"    juge indisponible: {type(e).__name__} {str(e)[:80]}")
        return {c: 0 for c in ("tutoiement", "concision", "non_culpabilisant",
                               "concret", "pas_de_promesse")}
