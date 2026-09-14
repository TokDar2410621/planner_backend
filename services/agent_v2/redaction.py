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
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from services.agent_v2.registre import Registre

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


def _charger_rendu():
    """rendu.py, charge par une fonction pour que les tests puissent le simuler."""
    from services.agent_v2 import rendu
    return rendu


def bloc_lecture(registre: Registre, aujourdhui=None) -> str:
    """Ce que l'agent a VU, rendu par du code, sur un tour sans mutation."""
    return _charger_rendu().rendre_lecture(registre, aujourdhui)


def bloc_factuel(registre: Registre, aujourdhui=None, cles_posees=None) -> str:
    """Le compte rendu deterministe du tour: les faits, sinon la lecture.

    `cles_posees` dit quelles actions retenues sont couvertes par la question
    du tour: rendu.py tait celles-la et donne une ligne aux autres.
    """
    r = _charger_rendu()
    return (r.rendre_faits(registre, aujourdhui, cles_posees)
            or r.rendre_lecture(registre, aujourdhui))


def question_code(demandes: list[dict], aujourdhui=None) -> tuple[str, list[dict], list[str]]:
    """(question, chips avec leur option, cles rendues) pour les demandes du tour."""
    if not demandes:
        return "", [], []
    return _charger_rendu().rendre_demandes(demandes, aujourdhui)


def marqueurs_bruts(texte: str) -> list[str]:
    """Les traces de texte ecrit pour le modele dans ce que lit l'utilisateur."""
    return list(_charger_rendu().marqueurs_bruts(texte or ""))



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


_VOCABULAIRE_INTERNE = re.compile(
    r"\b(?:flexibles?|verrouill\w*|port[ée]e|clarifi\w*)\b", re.IGNORECASE)


def _sans_vocabulaire_interne(texte: str) -> str:
    return " ".join(p for p in _phrases(texte) if not _VOCABULAIRE_INTERNE.search(p))


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

    # Le vocabulaire interne (« flexible », « verrouiller », « portee »,
    # « clarifier ») ne parle pas a l'utilisateur (banc du 2026-09-14).
    ouverture = _sans_vocabulaire_interne(ouverture)
    suite = _sans_vocabulaire_interne(suite)
    if _VOCABULAIRE_INTERNE.search(question):
        question, options = "", []
    options = [o for o in options if not _VOCABULAIRE_INTERNE.search(str(o or ""))]

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
# Une DUREE, jamais une heure d'horloge (revue du 2026-09-14: « de 14 h à
# 16 h » donnait « Il manque 12 h », et « 2026-09-17\nHeure du rendez-vous »
# lisait 17 h parce que \s traversait le saut de ligne). Trois formes
# comptent: « N heures », « N h de/d' » + nom, « pendant N h ». Un nombre colle
# a un chiffre, un tiret, deux-points ou une barre (date, plage) ne compte pas,
# ni un « N h » precede de de/à/vers/dès/avant/après ou suivi d'une autre heure.
_NOMBRE = r"(\d+(?:[.,]\d+)?|" + "|".join(_NOMBRES_EN_MOTS) + r")"
_HEURES_DEMANDEES = re.compile(
    r"(?<![\w:/.,\-–])" + _NOMBRE + r"[ \t]*(heures?\b|h\b)",
    re.IGNORECASE)
_AVANT_HORLOGE = re.compile(
    r"(?:\b(?:de|des|dès|a|à|vers|avant|apres|après|jusqu'?(?:a|à)|entre|et|ou|midi|minuit)"
    r"|[\-–])[ \t]*$",
    re.IGNORECASE)
_APRES_HORLOGE = re.compile(r"^[ \t]*(?:\d|[aà][ \t]+\d|[\-–]|jusqu)", re.IGNORECASE)
_SUIVI_DE_NOM = re.compile(r"^[ \t]*(?:de\b|d['’])", re.IGNORECASE)
_AVANT_DUREE = re.compile(r"\b(?:pendant|durant|environ|au total|en tout|total de)[ \t]*$", re.IGNORECASE)
_COMPTE_DEMANDE = re.compile(
    r"(?<![\w:/.,\-–])(\d+|" + "|".join(_NOMBRES_EN_MOTS) + r")[ \t]*"
    r"(?:blocs?|s[ée]ances?|sessions?|cr[ée]neaux?|entra[iî]nements?)\b",
    re.IGNORECASE)
_RESUME_DE_FORMULAIRE = re.compile(r"^\s*voici mes r[ée]ponses", re.IGNORECASE)


def _duree_demandee(message: str):
    """Le premier « N h » du message qui est une vraie duree, ou None."""
    for m in _HEURES_DEMANDEES.finditer(message):
        avant, apres = message[:m.start()], message[m.end():]
        if _AVANT_HORLOGE.search(avant) or _APRES_HORLOGE.match(apres):
            continue
        unite = m.group(2).lower()
        if unite.startswith("heure") or _SUIVI_DE_NOM.match(apres) or _AVANT_DUREE.search(avant):
            return m
    return None

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
    if not message or _RESUME_DE_FORMULAIRE.match(message):
        # Un formulaire rempli porte des dates et des heures, jamais une
        # quantite demandee.
        return ""
    compte, minutes = _creations_du_tour(registre)
    if compte == 0:
        return ""

    m_heures = _duree_demandee(message)
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
