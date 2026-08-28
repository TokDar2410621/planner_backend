"""
La partie factuelle de la reponse est rendue par du CODE, pas par un modele.

Granularite (decision du 2026-08-24): une ligne par action jusqu'a 5, puis
groupement des REUSSITES par outil avec un compte. Les echecs et les ecarts
restent detailles un par un quel que soit le volume, parce qu'un refus noye
dans un total est exactement le defaut qu'on corrige.
"""
from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, Field

from services.agent_v2.registre import Registre

SEUIL_GROUPEMENT = 5


class ActionCitee(BaseModel):
    ref: str = Field(description="Identifiant EXACT d'une entree du registre, par exemple a1")
    phrase: str = Field(description="Une phrase courte au sujet de CETTE action")


class ReponseDire(BaseModel):
    ouverture: str = Field(default="", description="Une phrase d'accroche, AUCUNE affirmation d'action")
    actions: list[ActionCitee] = Field(default_factory=list)
    suite: str = Field(default="", description="Une phrase de suite, AUCUNE affirmation d'action")


def bloc_factuel(registre: Registre) -> str:
    """Le compte rendu deterministe de ce qui s'est passe."""
    reussites = [a for a in registre.actions if a.succes and a.est_mutation]
    echecs = [a for a in registre.actions if not a.succes]
    interrompu = registre.budget_epuise or getattr(registre, "boucle_interrompue", False)
    if not reussites and not echecs and not registre.ecarts and not interrompu:
        return ""

    lignes: list[str] = []
    if len(reussites) <= SEUIL_GROUPEMENT:
        lignes += [f"- {a.message}" for a in reussites]
    else:
        par_outil: dict[str, list] = defaultdict(list)
        for a in reussites:
            par_outil[a.outil].append(a)
        for outil, actions in par_outil.items():
            lignes.append(f"- {len(actions)} x {outil}")

    # Jamais groupes: un refus ou un ecart se lit en toutes lettres.
    lignes += [f"- Refus: {a.message}" for a in echecs]
    lignes += [f"- Ecart: {e.description}" for e in registre.ecarts]
    if registre.budget_epuise:
        lignes.append("- Traitement interrompu: la limite d'etapes du tour a ete atteinte.")
    if getattr(registre, "boucle_interrompue", False):
        # Dit a l'utilisateur, pas seulement journalise: un tour tronque sans
        # explication ressemble a une panne, et il a le droit de savoir que
        # l'agent tournait en rond plutot que de travailler.
        lignes.append(
            "- Traitement interrompu: je repetais la meme action sans progresser.")
    return "\n".join(lignes)


def assembler(brut: ReponseDire, registre: Registre) -> tuple[str, int]:
    """Assemble la reponse finale et compte les actions rejetees.

    Une action dont la reference n'existe pas dans le registre est SUPPRIMEE.
    C'est le point ou le mensonge meurt.
    """
    rejetees = 0
    phrases: list[str] = []
    for citee in brut.actions:
        if registre.par_id(citee.ref) is None:
            rejetees += 1
            continue
        if citee.phrase.strip():
            phrases.append(citee.phrase.strip())

    morceaux = [m for m in [brut.ouverture.strip()] + phrases if m]
    faits = bloc_factuel(registre)
    if faits:
        morceaux.append(faits)
    if brut.suite.strip():
        morceaux.append(brut.suite.strip())
    return "\n\n".join(morceaux), rejetees
