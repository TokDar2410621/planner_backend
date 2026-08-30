"""
Le canal libre de DIRE: le mesurer, et depuis le 2026-08-30, le FERMER.

La garantie structurelle protege les actions referencees. Les champs libres
ouverture et suite, eux, n'etaient qu'observes: la decision de bascule du
2026-08-30 a chiffre ce trou (verite 15,9/15,9/20,0, item « ecrit=False,
annonce=True »), et c'est lui seul qui a coute la bascule des 185 comptes.

`epurer_reponse` applique donc a la prose la meme doctrine qu'aux actions
citees: une PHRASE qui presente une action comme faite, en cours ou a venir
est supprimee, pas corrigee. La phrase, jamais le champ entier: l'accroche
legitime qui l'entoure survit. Le contenu supprime n'est jamais journalise.

Le detecteur a ete reecrit apres une contre-expertise adversariale du meme
jour (trois angles, douze defauts prouves par execution). Ce qu'elle a
appris, et que cette version encode:

- les CLITIQUES defont un motif naif: « je l'ai deplace », « je vais en
  creer un » sont les formulations les plus naturelles du francais;
- le mensonge sans sujet passe par le PASSIF et le NOMINAL: « ton planning a
  ete reorganise », « le bloc est cree », « c'est fait », « Planning mis a
  jour! »;
- les radicaux larges tuent des phrases legitimes: cal\\w* matchait
  « calendrier » et « calculer », cre\\w* matchait « creuser » et
  « creation »; les racines exigent desormais une TERMINAISON verbale;
- une OFFRE n'est pas une affirmation: « Veux-tu que je m'occupe de
  deplacer ton examen? » doit survivre, c'est le geste central du champ
  suite. Les regles de futur, de present et de prise en charge se taisent
  dans une phrase interrogative ou apres « que je »;
- le detecteur travaille PAR PHRASE, comme la guillotine: normaliser le
  champ entier faisait matcher « Je vais bien. Organiser ta semaine est mon
  travail. » a cheval sur deux phrases.
"""
from __future__ import annotations

import re
import unicodedata

from services.agent_v2.redaction import ReponseDire

# ── Vocabulaire ──────────────────────────────────────────────────────────
# Racines d'action avec TERMINAISONS verbales explicites. `cal` sans garde
# matchait « calendrier »; `cre` matchait « creuser » et « creation »;
# `deplac` matchait « deplacement »; `annul` matchait « annulation ». Les
# terminaisons (e, es, é, ée, és, ées, er, ons, ent) excluent tous ces noms
# et les verbes hors sujet, verifie par la contre-expertise du 2026-08-30.
_RACINES = (
    "organis", "reorganis", "supprim", "effac", "vid", "enlev", "retir",
    "ajout", "cre", "deplac", "modifi", "annul", "planifi", "programm",
    "cal", "plac", "replac", "optimis", "restaur", "termin", "complet",
    "boug", "chang", "arrang", "configur", "liber", "decal", "reserv",
)
_TERM_CONJ = r"(?:e|es|ee|ees|er|ez|ons|ent)\b"
_TERM_PART = r"(?:e|ee|es|ees)\b"
_VERBE = r"(?:%s)%s" % ("|".join(_RACINES), _TERM_CONJ)
_PARTICIPE = r"(?:%s)%s" % ("|".join(_RACINES), _TERM_PART)
_LOCUTIONS = r"(?:mis(?:e|es)?\s+a\s+jour|mettre\s+a\s+jour|mis(?:e|es)?\s+en\s+place)"

# Adverbes toleres entre l'auxiliaire et le verbe. PLUS de possessifs ici:
# « j'ai ton calendrier sous les yeux » n'affirme rien, et c'est ton/ta/tes
# qui le faisait mourir.
_ADV = r"(?:bien\s+|deja\s+|tout\s+|aussi\s+|donc\s+)*"
# Clitiques objets, avant l'auxiliaire (« je l'ai deplace ») ou avant
# l'infinitif (« je vais le deplacer », « je vais en creer un »).
_CLIT = r"(?:l\s*'\s*|le\s+|la\s+|les\s+|leur\s+|lui\s+|en\s+|y\s+|te\s+|t\s*'\s*|me\s+|m\s*'\s*|nous\s+|vous\s+|se\s+|s\s*'\s*)"

_ACTION = rf"(?:{_VERBE}|{_LOCUTIONS})"

_REGLES = (
    # « j'ai deplace », « je l'ai deplace », « je te l'ai cale ». Deux tetes:
    # l'elision « j'ai » n'admet pas de clitique, la forme pleine « je » les
    # admet tous, et c'est eux qui defaisaient la premiere version.
    ("passe", re.compile(
        rf"\b(?:j\s*'\s*ai|je\s+(?:{_CLIT})+ai)\s+{_ADV}(?:{_CLIT})*{_ACTION}")),
    # « je viens de reorganiser », « je viens d'en creer un »
    ("passe_recent", re.compile(
        rf"\bje\s+viens\s+d[e']\s*(?:{_CLIT})*(?:(?:{'|'.join(_RACINES)})er\b|{_LOCUTIONS})")),
    # « je vais deplacer », « je vais le deplacer », « je vais en creer »
    ("futur", re.compile(
        rf"\bje\s+vais\s+{_ADV}(?:{_CLIT})*(?:(?:{'|'.join(_RACINES)})er\b|{_LOCUTIONS})")),
    # « je supprime le doublon et j'ajoute le nouveau »
    ("present", re.compile(
        rf"\bj(?:e\s+|\s*'\s*)(?:{_CLIT})*{_VERBE}")),
    ("en_cours", re.compile(
        rf"\bje\s+suis\s+en\s+train\s+d[e']\s*(?:{_CLIT})*(?:(?:{'|'.join(_RACINES)})er\b|{_LOCUTIONS})")),
    ("prise_en_charge", re.compile(r"\bje\s+m\s*'?\s*occupe\s+de\b")),
    # Sans sujet: « a ete reorganise », « est cree », « sont supprimes »
    ("resultat", re.compile(
        rf"\b(?:a|ont|est|sont|etait|etaient)\s+(?:ete\s+)?{_ADV}(?:{_PARTICIPE}|{_LOCUTIONS})")),
    # « c'est fait », « voila, c'est regle », « mission accomplie »
    ("cest_fait", re.compile(
        r"\bc\s*'?\s*est\s+(?:fait|regle|corrige|arrange|termine|en\s+place|bon\s+c\s*'?\s*est\s+fait)\b"
        r"|\bmission\s+accomplie\b")),
)

# Une exclamation nominale est un mensonge complet a elle seule:
# « Planning mis a jour! », « Termine! », « Fait! », « Voila, bloc cree. »
_NOMINALE = re.compile(
    rf"^(?:et\s+)?(?:voila\s*,?\s*)?(?:planning\s+|horaire\s+|bloc\s+|tache\s+|semaine\s+|cours\s+)?"
    rf"(?:{_PARTICIPE}|{_LOCUTIONS}|fait|regle|termine|corrige|arrange)\s*$")

# Les regles d'INTENTION se taisent dans une offre: phrase interrogative, ou
# subordonnee en « que je » (« veux-tu que je m'occupe de... »). Le passe et
# le resultat restent actifs partout: une question n'excuse pas une
# affirmation d'action deja faite.
_REGLES_D_INTENTION = {"futur", "present", "prise_en_charge", "passe_recent"}
_MARQUE_OFFRE = re.compile(
    r"\b(?:veux|voudrais|souhaites?|aimerais|peux|pourrais|dois|devrais)\s*-?\s*(?:tu|je|on)\b"
    r"|\bque\s+j(?:e\b|\s*')")

_APOSTROPHES = str.maketrans({
    "‘": "'",
    "’": "'",
    "‛": "'",
    "ʼ": "'",
    "`": "'",
    "´": "'",
})


def _normaliser(texte) -> str:
    if not texte or not isinstance(texte, str):
        return ""
    texte = texte.translate(_APOSTROPHES)
    plat = (
        unicodedata.normalize("NFKD", texte)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return re.sub(r"[^a-z0-9'?]+", " ", plat)


_FIN_DE_PHRASE = re.compile(r"(?<=[.!?…])\s+")


def _phrases(texte: str) -> list[str]:
    return [p for p in _FIN_DE_PHRASE.split(texte) if p.strip()]


def _fuites_d_une_phrase(phrase: str) -> list[str]:
    plat = _normaliser(phrase)
    if not plat.strip():
        return []
    interrogative = "?" in plat or bool(_MARQUE_OFFRE.search(plat))
    sans_marque = plat.replace("?", " ")
    fuites = []
    for nom, regle in _REGLES:
        if interrogative and nom in _REGLES_D_INTENTION:
            continue
        if regle.search(sans_marque):
            fuites.append(nom)
    if _NOMINALE.match(sans_marque.strip()):
        fuites.append("nominale")
    return fuites


def fuite_lexicale(texte) -> list[str]:
    """Detecte une affirmation d'action, PHRASE par PHRASE.

    Travailler par phrase est ce qui aligne le detecteur et la guillotine:
    normaliser le champ entier faisait matcher deux phrases innocentes a
    cheval (« Je vais bien. Organiser ta semaine est mon travail. »).
    """
    if not texte or not isinstance(texte, str):
        return []
    fuites: list[str] = []
    for phrase in _phrases(texte):
        for f in _fuites_d_une_phrase(phrase):
            if f not in fuites:
                fuites.append(f)
    return fuites


# Un fragment orphelin qui suivait une phrase supprimee: « ... si tu
# confirmes, bien sur. » sans sa principale. On le supprime avec elle.
_ORPHELIN = re.compile(
    r"^(?:si|et|mais|donc|car|ou|puis|alors|ensuite|sinon|comme)\b", re.IGNORECASE)


def epurer_reponse(reponse: ReponseDire) -> tuple[ReponseDire, int]:
    """Retire de la prose toute phrase qui affirme une action.

    Rend la reponse epuree et le nombre de phrases supprimees. Ne touche pas
    aux actions structurees: elles ont leur propre garde (les references
    inconnues meurent dans assembler).
    """
    supprimees = 0
    champs: dict[str, str] = {}
    for champ in ("ouverture", "suite"):
        gardees: list[str] = []
        precedente_supprimee = False
        for phrase in _phrases(getattr(reponse, champ, "") or ""):
            nette = phrase.strip()
            if _fuites_d_une_phrase(nette):
                supprimees += 1
                precedente_supprimee = True
                continue
            if precedente_supprimee and _ORPHELIN.match(nette):
                # Subordonnee detachee par « ... »: seule, elle n'a plus de
                # tete et mutile la voix davantage qu'elle ne la sert.
                supprimees += 1
                continue
            precedente_supprimee = False
            gardees.append(nette)
        champs[champ] = " ".join(gardees)
    if not supprimees:
        return reponse, 0
    return reponse.model_copy(update=champs), supprimees


def fuites_reponse(reponse: ReponseDire) -> list[str]:
    """Observe seulement ouverture et suite, pas les actions structurees."""
    fuites: list[str] = []
    for champ in ("ouverture", "suite"):
        for fuite in fuite_lexicale(getattr(reponse, champ, "")):
            fuites.append(f"{champ}:{fuite}")
    return fuites
