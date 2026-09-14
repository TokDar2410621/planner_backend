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
import unicodedata
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


def bloc_factuel(registre: Registre, aujourdhui=None, cles_posees=None,
                 sans_lecture: bool = False) -> str:
    """Le compte rendu deterministe du tour: les faits, sinon la lecture.

    `cles_posees` dit quelles actions retenues sont couvertes par la question
    du tour: rendu.py tait celles-la et donne une ligne aux autres.
    `sans_lecture`: la lecture n'a servi qu'a preparer un formulaire ou un
    choix, elle ne se deverse pas au-dessus (banc du round 4, s02-1).
    """
    r = _charger_rendu()
    faits = r.rendre_faits(registre, aujourdhui, cles_posees)
    if faits and not sans_lecture and _abandons_seuls(registre):
        # Revue du round 6: la ligne d'abandon (D2) remplacait la lecture que
        # le message demandait (« c'est quoi mon horaire demain ? »). Elle la
        # precede maintenant.
        lecture = r.rendre_lecture(registre, aujourdhui)
        return f"{faits}\n\n{lecture}" if lecture else faits
    if faits or sans_lecture:
        return faits or ""
    return r.rendre_lecture(registre, aujourdhui)


def _abandons_seuls(registre: Registre) -> bool:
    """Les seuls faits du tour sont des demandes laissees tombees par le code:
    aucune autre mutation (reussie ou retenue), aucun ecart, aucun arret."""
    abandons = {id(a) for a in registre.actions
                if (a.donnees or {}).get("abandonnee_par_le_code")}
    if not abandons:
        return False
    if any(a.est_mutation for a in registre.actions if id(a) not in abandons):
        return False
    if any(getattr(e, "genre", None) for e in registre.ecarts):
        return False
    return not (registre.budget_epuise or getattr(registre, "boucle_interrompue", False))


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


# « jeudi 17 sept. ou tous les jeudis ? » est UNE phrase: un point
# d'abreviation de mois ou de jour suivi d'une minuscule ne la coupe pas.
# Sans cette couture, la question restait a moitie dans la prose (round 4).
_ABREVIATION_EN_QUEUE = re.compile(
    r"\b(?:janv|f[ée]vr|avr|juil|sept|oct|nov|d[ée]c|lun|mar|mer|jeu|ven|sam|dim)\.$",
    re.IGNORECASE)
_DEBUT_MINUSCULE = re.compile(r"^[a-zàâçéèêëîïôûùüÿœ0-9]")


def _phrases(texte: str) -> list[str]:
    morceaux = [p.strip() for p in _FIN_DE_PHRASE.split(texte or "") if p.strip()]
    phrases: list[str] = []
    for morceau in morceaux:
        if phrases and _ABREVIATION_EN_QUEUE.search(phrases[-1]) \
                and _DEBUT_MINUSCULE.match(morceau):
            phrases[-1] = f"{phrases[-1]} {morceau}"
        else:
            phrases.append(morceau)
    return phrases


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


# « bloc » et « formulaire » s'y ajoutent: « Remplis le formulaire »,
# « le bloc de sommeil chevauche ton quart » (banc du 2026-09-14, round 2).
_VOCABULAIRE_INTERNE = re.compile(
    r"\b(?:flexibles?|verrouill\w*|port[ée]e|clarifi\w*|blocs?|formulaires?)\b", re.IGNORECASE)


def _sans_vocabulaire_interne(texte: str) -> str:
    return " ".join(p for p in _phrases(texte) if not _VOCABULAIRE_INTERNE.search(p))


# ── Tiret long, mecanique, questions hors champ (round 4) ──────────────────

# U+2014 ecrit par echappement: la regle zero tiret long vaut aussi pour ce
# fichier. Le premier devient « : », les suivants « ; », un tiret en tete ou
# en queue disparait.
_TIRET_LONG = re.compile("\\s*\u2014\\s*")


def sans_tiret_long(texte):
    """Retire tout tiret long d'un texte montre a l'utilisateur."""
    if not isinstance(texte, str) or "\u2014" not in texte:
        return texte
    rang = {"n": 0}

    def _remplacer(m):
        if m.start() == 0 or m.end() == len(texte):
            return " "
        rang["n"] += 1
        return " : " if rang["n"] == 1 else " ; "

    return " ".join(_TIRET_LONG.sub(_remplacer, texte).split(" ")).strip()


_FIN_QUESTION = re.compile(r"\?[\s\"'»)\]]*$")


def _est_question(phrase: str) -> bool:
    return bool(_FIN_QUESTION.search(phrase or ""))


def contient_question(texte: str) -> bool:
    """Une phrase du texte finit-elle par « ? » ?"""
    return any(_est_question(p) for p in _phrases(texte or ""))


# La mecanique de l'interface decrite a l'utilisateur (banc du round 3):
# « Remplis ce qui te convient », « le tout est pré-rempli », « Réponds « Tous
# les jeudis » », « ajuste les jours si besoin », « touche un des boutons ».
# Revue de lisibilite du round 4: « Ta journée est bien remplie » et « au
# champ de tir » tombaient. Seuls l'imperatif « remplis » et le champ d'une
# saisie (« le champ », « ces champs ») restent de la mecanique.
_MECANIQUE = re.compile(
    r"\brempli(?:s|r)\b|\bpr[ée][- ]?rempli\w*"
    r"|\br[ée]ponds?\s*(?:[«\"“]|par\b|avec\b)"
    r"|\bboutons?\b|\bcoch(?:e|es|er|ez|ée|ées)\b|\bclique\w*|\bappuie\w*\s+sur\b"
    r"|\bci-(?:dessous|dessus)\b|\bajuste\w*\b[^.?!]*\bsi\s+besoin\b"
    r"|\bs[ée]lectionne\w*|\b(?:le|les|ce|ces|chaque|un|des)\s+champs?\b"
    # Banc du round 5, cinq tours: « dans tes réponses » (s02-1), « Choisis
    # tes trois jours » (s06-1), « déjà affiché » et « dans la liste »
    # (s08-1), « l'étendue » (p2-1), « choisis parmi les moments libres
    # proposés » (p3-1). « choisis » ne compte qu'a l'imperatif: « si tu
    # choisis le matin » reste.
    r"|\bdans\s+(?:tes|ta|les|ces|mes)\s+r[ée]ponses?\b"
    r"|(?<!tu\s)\bchoisis\b"
    r"|\bd[ée]j[àa]\s+affich\w*|\b(?:est|sont)\s+affich[ée]\w*"
    r"|\bdans\s+(?:la|cette|ta|les|ces)\s+listes?\b"
    r"|\bl['’]\s*[ée]tendue\b"
    r"|\b(?:moments?|cr[ée]neaux|options|choix|heures|jours|plages?)\s+(?:\w+\s+)?propos[ée]e?s?\b",
    re.IGNORECASE)
# Une absence affirmee alors que le code affiche la liste lue (banc du round
# 4, s03-1). Round 6 (D7): la phrase ne tombe que si elle nomme un TITRE que
# la liste affiche. « Tu n'as pas d'examen lundi. » est une vraie reponse et
# doit survivre; « Il n'y a pas de cours de maths » au-dessus de « Calcul
# différentiel » survit aussi, faute de meme titre (ecart accepte: le code
# ne devine pas qu'un cours de maths est ce calcul).
_ABSENCE = re.compile(
    r"\bil\s+n['’]?\s*y\s+a\s+(?:pas|aucun\w*|rien)\b|\btu\s+n['’]?\s*as\s+(?:pas|aucun\w*)\b"
    r"|\bn['’]?\s*(?:appara[iî]\w*|figure\w*)\s+pas\b|\baucun\w*\s+\w+\s+(?:dans|à|a)\s+ton\b",
    re.IGNORECASE)
# Une demande a l'imperatif, seconde question deguisee quand le code demande
# deja: « Dis-moi aussi vers quel jour tu veux le déplacer. »
_DEMANDE_EN_PROSE = re.compile(
    r"^(?:et\s+|alors\s+|sinon\s+)?(?:dis|donne|indique|pr[ée]cise|confirme)[- ]moi\b",
    re.IGNORECASE)


_BLOC_TEXTE = re.compile(r"^(.*\S)\s\(\d{1,2}:\d{2}-\d{1,2}:\d{2}\)$")


def _plat(texte: str) -> str:
    """Minuscules, sans accents ni ponctuation, espaces simples."""
    sans_accents = unicodedata.normalize("NFKD", texte or "").encode("ascii", "ignore").decode("ascii")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", sans_accents.lower()).split())


def _collecter_titres(valeur, sortie: set) -> None:
    if isinstance(valeur, dict):
        for cle, v in valeur.items():
            if cle in ("title", "titre") and isinstance(v, str):
                sortie.add(v)
            else:
                _collecter_titres(v, sortie)
    elif isinstance(valeur, (list, tuple)):
        for v in valeur:
            _collecter_titres(v, sortie)
    elif isinstance(valeur, str):
        m = _BLOC_TEXTE.match(valeur.strip())
        if m:
            sortie.add(m.group(1))


def _titres_affiches(registre: Registre, faits: str) -> set[str]:
    """Les titres (aplatis) lus ce tour ET visibles dans la liste affichee."""
    candidats: set = set()
    for a in registre.actions:
        if a.succes and a.outil in LECTURES_RENDUES:
            _collecter_titres(a.donnees, candidats)
    for ligne in (faits or "").splitlines():
        if "·" in ligne:
            candidats.add(ligne.rsplit("·", 1)[1])
    affiche = f" {_plat(faits)} "
    return {t for t in map(_plat, candidats) if len(t) >= 3 and f" {t} " in affiche}


def _absence_contredite(phrase: str, titres: set[str]) -> bool:
    if not titres or not _ABSENCE.search(phrase or ""):
        return False
    plate = f" {_plat(phrase)} "
    return any(f" {t} " in plate for t in titres)


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
        ouverture = sans_tiret_long((getattr(brut, "ouverture", "") or "").strip())
        suite = sans_tiret_long((getattr(brut, "suite", "") or "").strip())
        question = sans_tiret_long((getattr(brut, "question", "") or "").strip())
        options = [sans_tiret_long(o) for o in list(getattr(brut, "options", None) or [])]

    # Le vocabulaire interne (« flexible », « verrouiller », « portee »,
    # « clarifier ») ne parle pas a l'utilisateur (banc du 2026-09-14), ni la
    # mecanique de l'interface (banc du round 3).
    ouverture = _sans_vocabulaire_interne(ouverture)
    suite = _sans_vocabulaire_interne(suite)
    if _VOCABULAIRE_INTERNE.search(question) or _MECANIQUE.search(question):
        question, options = "", []
    options = [o for o in options if not _VOCABULAIRE_INTERNE.search(str(o or ""))]

    lecture_sans_liste = False
    if not faits:
        ouverture, n1 = _sans_annonce_vide(ouverture)
        suite, n2 = _sans_annonce_vide(suite)
        lecture_sans_liste = bool(n1 or n2)
    elif any(a.succes and a.outil in LECTURES_RENDUES for a in registre.actions):
        titres = _titres_affiches(registre, faits)
        ouverture = " ".join(p for p in _phrases(ouverture) if not _absence_contredite(p, titres))
        suite = " ".join(p for p in _phrases(suite) if not _absence_contredite(p, titres))
        if _absence_contredite(question, titres):
            question, options = "", []

    # Une question par reponse, et dans son champ (lot 3d, banc du round 3).
    # Une question ecrite en prose sort de la prose: elle devient LA question
    # s'il n'y en a aucune, sinon elle tombe. Quand le code, un formulaire ou
    # un choix porte deja la question, toute question de DIRE tombe, et ses
    # demandes a l'imperatif (« Dis-moi aussi... ») avec.
    questions_en_prose: list[str] = []

    def _garder(texte: str) -> str:
        gardees = []
        for phrase in _phrases(texte):
            if _MECANIQUE.search(phrase):
                continue
            if _est_question(phrase):
                questions_en_prose.append(phrase)
                continue
            if question_code and _DEMANDE_EN_PROSE.match(phrase):
                continue
            gardees.append(phrase)
        return " ".join(gardees)

    ouverture = _garder(ouverture)
    suite = _garder(suite)
    prose = " ".join(p for p in (ouverture, suite) if p).strip()

    if question_code:
        return Composition(
            faits=faits,
            prose=prose,
            question=sans_tiret_long((question_code.get("question") or "").strip()),
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
    if not question and questions_en_prose:
        # Les options de DIRE repondaient a SON champ question, vide ici:
        # elles ne suivent pas une question venue de la prose.
        question, propres = questions_en_prose[0], []
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
