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


# Les lectures qui montrent un horaire. Leur resultat contient deja tout ce
# qu'il faut pour ecrire la liste; on ne demande donc pas au modele de la
# recopier de memoire.
LECTURES_D_HORAIRE = ("get_week_schedule", "get_today_schedule", "list_blocks")


def _liste_de_lecture(action) -> list[str]:
    """La liste des blocs vus, rendue par du CODE.

    Defaut remonte par Darius le 2026-08-30: a « qu'est-ce que j'ai cette
    semaine ? », l'agent repondait « deux grosses journees (lundi avec cours
    et quart au depanneur) et trois jours bien degages ». Des noms noyes dans
    une prose, aucune heure, rien a lire.

    La cause tenait a la portee de la garantie: `bloc_factuel` ne rend que les
    MUTATIONS. Sur un tour de lecture il est vide, donc tout ce que
    l'utilisateur lit est de la prose du modele, et un modele resume.

    Or les outils rendent deja la matiere exacte: `get_week_schedule` donne
    `days[].blocks` sous la forme « Cours de geologie (09:00-12:00) ». On la
    met en page, on n'en fabrique rien.
    """
    donnees = action.donnees or {}
    lignes: list[str] = []

    if action.outil == "get_week_schedule":
        for jour in donnees.get("days") or []:
            blocs = jour.get("blocks") or []
            if not blocs:
                continue
            lignes.append(f"- {jour.get('day_name', '?')} : " + ", ".join(blocs))
        return lignes

    if action.outil == "get_today_schedule":
        blocs = donnees.get("blocks") or []
        jour = donnees.get("day_name", "Aujourd'hui")
        for b in blocs:
            if isinstance(b, str):
                lignes.append(f"- {jour} : {b}")
            else:
                titre = b.get("title", "?")
                debut, fin = b.get("start_time", ""), b.get("end_time", "")
                lignes.append(f"- {jour} : {titre} ({debut}-{fin})")
        return lignes

    if action.outil == "list_blocks":
        # Groupe par jour, dans l'ordre de la semaine: une liste a plat obligerait
        # a la trier de tete.
        par_jour: dict[str, list[str]] = defaultdict(list)
        ordre: list[str] = []
        for b in donnees.get("blocks") or []:
            jour = b.get("day_name") or "?"
            if jour not in par_jour:
                ordre.append(jour)
            debut, fin = b.get("start_time", ""), b.get("end_time", "")
            par_jour[jour].append(f"{b.get('title', '?')} ({debut}-{fin})")
        for jour in ordre:
            lignes.append(f"- {jour} : " + ", ".join(par_jour[jour]))
        return lignes

    return lignes


def bloc_lecture(registre: Registre) -> str:
    """Ce que l'agent a VU, quand il n'a rien fait d'autre que regarder.

    Uniquement sur un tour SANS mutation: si quelque chose a change, c'est le
    changement qui compte et le compte rendu le dit deja. Empiler les deux
    noierait l'important.

    On prend la DERNIERE lecture reussie: si l'agent a relu apres coup, c'est
    la vue la plus recente qui fait foi.
    """
    if any(a.succes and a.est_mutation for a in registre.actions):
        return ""
    lectures = [a for a in registre.actions
                if a.succes and a.outil in LECTURES_D_HORAIRE]
    if not lectures:
        return ""
    for action in reversed(lectures):
        lignes = _liste_de_lecture(action)
        if lignes:
            return "\n".join(lignes)
    return ""


def bloc_factuel(registre: Registre) -> str:
    """Le compte rendu deterministe de ce qui s'est passe."""
    reussites = [a for a in registre.actions if a.succes and a.est_mutation]
    echecs = [a for a in registre.actions if not a.succes]
    interrompu = registre.budget_epuise or getattr(registre, "boucle_interrompue", False)
    lecture = bloc_lecture(registre)
    if not reussites and not echecs and not registre.ecarts and not interrompu:
        return lecture

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
