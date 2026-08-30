"""
La partie factuelle de la reponse est rendue par du CODE, pas par un modele.

Granularite (decision du 2026-08-24): une ligne par action jusqu'a 5, puis
groupement des REUSSITES par outil avec un compte. Les echecs et les ecarts
restent detailles un par un quel que soit le volume, parce qu'un refus noye
dans un total est exactement le defaut qu'on corrige.
"""
from __future__ import annotations

import re
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


# ── La section RESTE: demande contre place, rendu par du code ─────────────
#
# Piece de la spec (6.x, « comparateur de quantite et section RESTE »)
# signalee absente des la verification du plan du 2026-08-24, et reclamee par
# les faits le 2026-08-30: quand l'utilisateur demande 6 h et que l'agent en
# place 2, le compte rendu annonce un succes et rien ne nomme le manque. Le
# modele ne peut pas etre charge de ce calcul: c'est une soustraction, elle
# se rend par du code.

_NOMBRES_EN_MOTS = {
    "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
    "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10,
}
_HEURES_DEMANDEES = re.compile(
    r"\b(\d+(?:[.,]\d+)?|" + "|".join(_NOMBRES_EN_MOTS) + r")\s*(?:h\b|heures?\b)",
    re.IGNORECASE)
_COMPTE_DEMANDE = re.compile(
    r"\b(\d+|" + "|".join(_NOMBRES_EN_MOTS) + r")\s*"
    r"(?:blocs?|s[ée]ances?|sessions?|cr[ée]neaux?|entra[iî]nements?)\b",
    re.IGNORECASE)

_CREATEURS = ("create_block", "schedule_task_at")


def _en_nombre(brut: str) -> float:
    brut = brut.lower().replace(",", ".")
    return _NOMBRES_EN_MOTS.get(brut, 0) or float(brut)


def _minutes(debut: str, fin: str, overnight: bool = False) -> int:
    h1, m1 = int(debut[:2]), int(debut[3:5])
    h2, m2 = int(fin[:2]), int(fin[3:5])
    duree = (h2 * 60 + m2) - (h1 * 60 + m1)
    if duree <= 0 or overnight:
        duree += 24 * 60
        duree %= 24 * 60
    return duree


def _fmt_minutes(minutes: int) -> str:
    h, m = divmod(max(0, minutes), 60)
    if h and m:
        return f"{h} h {m:02d}"
    if h:
        return f"{h} h"
    return f"{m} min"


def _creations_du_tour(registre: Registre) -> tuple[int, int]:
    """(nombre cree, minutes creees), dedoublonne par id.

    Le dedoublonnage par id est essentiel: un rejeu idempotent inscrit la
    MEME action une seconde fois au registre, et une somme naive compterait
    double ce qui n'a ete cree qu'une fois.
    """
    vus: set = set()
    compte = 0
    minutes = 0
    for a in registre.actions:
        if not a.succes or a.outil not in _CREATEURS:
            continue
        donnees = a.donnees or {}
        entrees = list(donnees.get("created") or [])
        sb = donnees.get("scheduled_block")
        if sb:
            entrees.append(sb)
        for e in entrees:
            cle = (a.outil, e.get("id"))
            if e.get("id") is not None and cle in vus:
                continue
            vus.add(cle)
            compte += 1
            debut, fin = e.get("start_time"), e.get("end_time")
            if debut and fin:
                minutes += _minutes(debut, fin, bool(e.get("overnight") or e.get("is_night_shift")))
    return compte, minutes


def bloc_reste(message: str, registre: Registre) -> str:
    """Nomme le manque quand on a place moins que demande. Sinon, rien.

    Trois conditions, toutes necessaires:
    - le message porte une quantite explicite (heures ou nombre d'elements);
    - le tour a reellement CREE quelque chose (sur une suppression ou un
      refus, comparer n'aurait aucun sens et le succes n'est pas annonce);
    - le place est strictement sous le demande. Quand tout rentre, la ligne
      se tait: annoncer qu'il ne manque rien serait du bruit.
    """
    if not message:
        return ""
    compte, minutes = _creations_du_tour(registre)
    if compte == 0:
        return ""

    m_heures = _HEURES_DEMANDEES.search(message)
    if m_heures:
        demande_min = int(round(_en_nombre(m_heures.group(1)) * 60))
        if 0 < minutes < demande_min:
            return (f"- Demandé : {_fmt_minutes(demande_min)}. "
                    f"Placé : {_fmt_minutes(minutes)}. "
                    f"Il manque {_fmt_minutes(demande_min - minutes)}.")
        return ""

    m_compte = _COMPTE_DEMANDE.search(message)
    if m_compte:
        demande_n = int(_en_nombre(m_compte.group(1)))
        if 0 < compte < demande_n:
            manque = demande_n - compte
            return (f"- Demandé : {demande_n}. Créé : {compte}. "
                    f"Il en manque {manque}.")
    return ""
