"""
La partie factuelle de la reponse est rendue par du CODE, pas par un modele.

Depuis le 2026-09-14, UN SEUL NARRATEUR. Avant, DIRE ecrivait une phrase par
action ET le code imprimait une ligne pour la meme action: « J'ai mis a jour
le bloc Gym » trois fois, puis « Bloc Gym mis a jour » trois fois, sans heure.
Desormais le compte rendu des actions vient uniquement de rendu.py (lot b4),
et la prose de DIRE ne le repete jamais: les phrases `actions` ne sont plus
rendues, elles ne servent qu'a compter les references inventees.

Un tour s'affiche en trois sections, dans l'ordre ou elles sont streamees:
FAITS (code), PROSE (ouverture + suite de DIRE, epurees), QUESTION (une seule
par tour, choisie par agent.py selon PRIORITE).
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from services.agent_v2.registre import Registre

SEUIL_GROUPEMENT = 5

# Les cinq lectures dont le resultat merite une liste rendue par le code. Un
# tour ou l'une d'elles a reussi sans que rien ne s'affiche est compte dans
# read_without_list: c'est le defaut « que voici » suivi de rien.
LECTURES_RENDUES = ("get_week_schedule", "get_today_schedule", "list_blocks",
                    "find_free_slots", "list_tasks")


class ActionCitee(BaseModel):
    ref: str = Field(description="Identifiant EXACT d'une entree du registre, par exemple a1")
    phrase: str = Field(description="Une phrase courte au sujet de CETTE action")


class ReponseDire(BaseModel):
    ouverture: str = Field(
        default="",
        description="Facultative, au plus 12 mots. Répond d'abord. Aucune action "
                    "affirmée, aucun fait déjà affiché répété.")
    suite: str = Field(
        default="",
        description="Facultative, une phrase. Aucune action affirmée. Pas de question ici.")
    question: str = Field(
        default="",
        description="Au plus UNE question, finit par '?'. Vide si le code pose déjà une question.")
    options: list[str] = Field(
        default_factory=list,
        description="0, ou 2 à 4 réponses courtes à `question`, tirées des vraies "
                    "entités du registre (créneaux, blocs, jours).")
    refs: list[str] = Field(
        default_factory=list,
        description="Références du registre dont ouverture ou suite parlent, ex. a1.")
    # Garde pour compatibilite: un modele qui la remplit encore voit ses
    # references verifiees, mais ses phrases ne sont JAMAIS rendues.
    actions: list[ActionCitee] = Field(
        default_factory=list, description="Obsolète: laisse vide.")


# ── Couture avec rendu.py ────────────────────────────────────────────────


def _charger_rendu():  # SEAM-INTEGRATION
    try:
        from services.agent_v2 import rendu
        return rendu
    except ImportError:
        return None


# Les lectures qui montrent un horaire, pour l'ancien rendu seulement.
LECTURES_D_HORAIRE = ("get_week_schedule", "get_today_schedule", "list_blocks")  # SEAM-INTEGRATION


def _liste_de_lecture(action) -> list[str]:  # SEAM-INTEGRATION
    """Ancien rendu d'une lecture, garde tant que rendu.py n'est pas fusionne."""
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


def _bloc_lecture_ancien(registre: Registre) -> str:  # SEAM-INTEGRATION
    if any(a.succes and a.est_mutation for a in registre.actions):
        return ""
    lectures = [a for a in registre.actions
                if a.succes and a.outil in LECTURES_D_HORAIRE]
    for action in reversed(lectures):
        lignes = _liste_de_lecture(action)
        if lignes:
            return "\n".join(lignes)
    return ""


def _bloc_factuel_ancien(registre: Registre) -> str:  # SEAM-INTEGRATION
    reussites = [a for a in registre.actions if a.succes and a.est_mutation]
    echecs = [a for a in registre.actions if not a.succes]
    interrompu = registre.budget_epuise or getattr(registre, "boucle_interrompue", False)
    lecture = _bloc_lecture_ancien(registre)
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
    lignes += [f"- Refus: {a.message}" for a in echecs]
    lignes += [f"- Ecart: {e.description}" for e in registre.ecarts]
    if registre.budget_epuise:
        lignes.append("- Traitement interrompu: la limite d'etapes du tour a ete atteinte.")
    if getattr(registre, "boucle_interrompue", False):
        lignes.append(
            "- Traitement interrompu: je repetais la meme action sans progresser.")
    return "\n".join(lignes)


def bloc_lecture(registre: Registre, aujourdhui=None) -> str:
    """Ce que l'agent a VU, rendu par du code, sur un tour sans mutation."""
    r = _charger_rendu()
    if r is None:
        return _bloc_lecture_ancien(registre)
    return r.rendre_lecture(registre, aujourdhui)


def bloc_factuel(registre: Registre, aujourdhui=None, cles_posees=None) -> str:
    """Le compte rendu deterministe du tour: les faits, sinon la lecture.

    `cles_posees` dit quelles actions retenues sont couvertes par la question
    du tour: rendu.py tait celles-la et donne une ligne aux autres.
    """
    r = _charger_rendu()
    if r is None:
        return _bloc_factuel_ancien(registre)
    return (r.rendre_faits(registre, aujourdhui, cles_posees)
            or r.rendre_lecture(registre, aujourdhui))


def question_code(demandes: list[dict], aujourdhui=None) -> tuple[str, list[dict], list[str]]:
    """(question, chips avec leur option, cles rendues) pour les demandes du tour."""
    if not demandes:
        return "", [], []
    r = _charger_rendu()
    if r is None:  # SEAM-INTEGRATION
        premiere = demandes[0]
        return ("Tu confirmes ?", [
            {"label": "Oui, confirme", "value": "Oui, je confirme.", "option": "confirmer"},
            {"label": "Non, garde tout", "value": "Non, ne change rien.", "option": "annuler"},
        ], [premiere.get("cle")] if premiere.get("cle") else [])
    return r.rendre_demandes(demandes, aujourdhui)


_MARQUEURS_REPLI = (  # SEAM-INTEGRATION
    ("anglais", re.compile(r"\b(?:created|skipped|block|tool)\b")),
    ("compte_outil", re.compile(r"\d+ x [a-z_]+")),
    ("date_iso", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("ecart", re.compile(r"\b[EeÉé]cart\b")),
    ("heure_hhmm", re.compile(r"\b\d{2}:\d{2}\b")),
    ("id_interne", re.compile(r"#\d+")),
    ("nom_outil", re.compile(r"\b[a-z]+_[a-z_]+\b")),
    ("pluriel_machine", re.compile(r"\(s\)")),
    ("ref_registre", re.compile(r"\(\s*[ae]\d+\s*\)")),
    ("refus", re.compile(r"\b[Rr]efus\b")),
)


def marqueurs_bruts(texte: str) -> list[str]:
    """Les traces de texte ecrit pour le modele dans ce que lit l'utilisateur."""
    r = _charger_rendu()
    if r is not None:
        return list(r.marqueurs_bruts(texte or ""))
    return sorted({nom for nom, motif in _MARQUEURS_REPLI if motif.search(texte or "")})  # SEAM-INTEGRATION


# ── Composition de la reponse ────────────────────────────────────────────

_FIN_DE_PHRASE = re.compile(r"(?<=[.!?…])\s+")
_VOICI = re.compile(r"\b(?:que\s+)?voici\b", re.IGNORECASE)
# Un compte nu: « 3 créneaux », « deux blocs ». « un bloc » n'est pas un
# compte, et le retirer tuerait des offres legitimes.
_COMPTE_NU = re.compile(
    r"\b(?:\d+|deux|trois|quatre|cinq|six|sept|huit|neuf|dix)\s+"
    r"(?:blocs?|cr[ée]neaux?|t[âa]ches?)\b",
    re.IGNORECASE)
OPTIONS_MAX = 4


@dataclass
class Composition:
    faits: str = ""
    prose: str = ""
    question: str = ""
    chips: list[dict] = field(default_factory=list)
    motif: str = ""
    demandes: list[dict] = field(default_factory=list)
    cles_posees: list[str] = field(default_factory=list)
    rejetees: int = 0
    lecture_sans_liste: bool = False

    @property
    def sections(self) -> list[str]:
        return [s for s in (self.faits, self.prose, self.question) if s]

    @property
    def texte(self) -> str:
        return "\n\n".join(self.sections)


def _phrases(texte: str) -> list[str]:
    return [p.strip() for p in _FIN_DE_PHRASE.split(texte or "") if p.strip()]


def _sans_annonce_vide(texte: str) -> tuple[str, int]:
    """Retire « que voici » et les comptes nus quand rien n'est affiche."""
    gardees: list[str] = []
    retirees = 0
    for phrase in _phrases(texte):
        if _VOICI.search(phrase) or _COMPTE_NU.search(phrase):
            retirees += 1
            continue
        gardees.append(phrase)
    return " ".join(gardees), retirees


def _references_rejetees(brut, registre: Registre) -> int:
    rejetees = 0
    for ref in getattr(brut, "refs", None) or []:
        if registre.par_id(ref) is None:
            rejetees += 1
    for citee in getattr(brut, "actions", None) or []:
        if registre.par_id(getattr(citee, "ref", None)) is None:
            rejetees += 1
    return rejetees


def composer(brut: ReponseDire | None, registre: Registre, faits: str,
             question_code: dict | None) -> Composition:
    """Assemble les trois sections a partir d'une sortie DIRE deja epuree.

    (b) une seule reference inconnue et TOUT ce que DIRE a ecrit tombe:
        ouverture, suite, question et options. Un redacteur qui invente une
        action n'est pas cru sur le reste.
    (c) les options ne partent qu'avec une question et seulement a 2, 3 ou 4.
    (e) sans faits affiches, une phrase qui annonce une liste (« que voici »,
        « 3 créneaux ») est retiree: elle promettrait ce qui ne suit pas.
    (f) si le code pose deja une question, celle de DIRE est ecartee.
    Les phrases de `actions` ne sont jamais rendues (un seul narrateur).
    """
    faits = faits or ""
    rejetees = _references_rejetees(brut, registre) if brut is not None else 0

    ouverture = suite = question = ""
    options: list = []
    if brut is not None and not rejetees:
        ouverture = (getattr(brut, "ouverture", "") or "").strip()
        suite = (getattr(brut, "suite", "") or "").strip()
        question = (getattr(brut, "question", "") or "").strip()
        options = list(getattr(brut, "options", None) or [])

    lecture_sans_liste = False
    if not faits:
        ouverture, n1 = _sans_annonce_vide(ouverture)
        suite, n2 = _sans_annonce_vide(suite)
        lecture_sans_liste = bool(n1 or n2)
    prose = " ".join(p for p in (ouverture, suite) if p).strip()

    if question_code:
        return Composition(
            faits=faits,
            prose=prose,
            question=(question_code.get("question") or "").strip(),
            chips=[dict(c) for c in question_code.get("chips") or []],
            motif=question_code.get("motif") or "",
            demandes=list(question_code.get("demandes") or []),
            cles_posees=list(question_code.get("cles_posees") or []),
            rejetees=rejetees,
            lecture_sans_liste=lecture_sans_liste,
        )

    propres: list[str] = []
    for option in options:
        texte = str(option or "").strip()
        if texte and texte not in propres:
            propres.append(texte)
    propres = propres[:OPTIONS_MAX]
    if not question or len(propres) < 2:
        propres = []
    return Composition(
        faits=faits,
        prose=prose,
        question=question,
        chips=[{"label": o, "value": o} for o in propres],
        motif="dire" if question else "",
        rejetees=rejetees,
        lecture_sans_liste=lecture_sans_liste,
    )


def assembler(brut: ReponseDire, registre: Registre) -> tuple[str, int]:
    """Le texte final (faits, prose, question) et le nombre de references rejetees."""
    compo = composer(brut, registre, bloc_factuel(registre), None)
    return compo.texte, compo.rejetees


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
