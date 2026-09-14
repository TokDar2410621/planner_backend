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

Le 2026-09-14, l'enquete « l'agent ne demande pas » a montre l'autre face:
la regle « resultat » restait active dans les questions et tuait de vraies
clarifications (« Ton cours est place a quelle heure ? »), la regle du
present tuait « Dis-moi a quelle heure commence ton quart et je le cree. »,
et la regle de l'orphelin emportait la question suivante. `fuite_question`
tranche desormais les phrases qui finissent par « ? », les champs question
et options: une premiere personne au passe reste toujours une fuite, un
participe de mutation n'en est une que sans marque interrogative ni offre.
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
# Clarification conditionnelle: « Dis-moi a quelle heure commence ton quart
# et je le cree. » L'action est suspendue a la reponse, ce n'est pas une
# affirmation. Seules les regles d'INTENTION se taisent, comme pour une offre.
_CONDITIONNELLE = re.compile(
    r"^(?:et\s+|alors\s+|sinon\s+)?(?:dis|donne|indique|precise|confirme)\s+moi\b.*\bet\s+je\b")

# ── Questions ────────────────────────────────────────────────────────────
# Une premiere personne au passe ou en cours est une fuite PARTOUT, question
# comprise: « Tu gardes le bloc Gym que j'ai ajoute ? » affirme l'ajout.
_PREMIERE_PERSONNE = re.compile(
    r"(?<![a-z])(?:j'ai|je\s+t'ai|que\s+j'ai|je\s+viens\s+de|j'ai\s+deja|c'est\s+fait|voila\s+qui\s+est)\b")
# « j'ai besoin de », « j'ai une question »: aucune action, on les neutralise
# avant la recherche pour ne pas tuer une clarification ordinaire.
_PREMIERE_PERSONNE_NEUTRE = re.compile(
    r"(?<![a-z])j'ai\s+(?=besoin\s+d|une\s+(?:petite\s+|derniere\s+)?question|un\s+doute)")

# Marques qui font d'une proposition une vraie question sur l'etat du monde:
# « Ton cours est place a quelle heure ? », « annule ou juste decale ? ».
_MARQUE_INTERROGATIVE = re.compile(
    r"\b(?:(?:quel|quelle|quels|quelles|quand|combien|lequel|laquelle|lesquels|lesquelles"
    r"|ou|comment|pourquoi)\b|a\s+quelle\s+heure\b|pour\s+quel|est\s+ce\s+qu)")
_OFFRE_EN_TETE = re.compile(
    r"^(?:et\s+|alors\s+|sinon\s+|ou\s+|donc\s+)?"
    r"(?:veux\s+tu|tu\s+veux|voudrais\s+tu|souhaites\s+tu|preferes\s+tu|je\s+peux|est\s+ce\s+que\s+je)\b"
    r"|" + _CONDITIONNELLE.pattern)

# Participe NU d'une mutation, sans auxiliaire: « Ton cours deplace a 14 h te
# convient ? ». Liste plus etroite que _RACINES: « programme », « vide »,
# « complete » ou « change » sont aussi des noms et des adjectifs courants.
_RACINES_PARTICIPE_NU = (
    "deplac", "replac", "plac", "ajout", "supprim", "cre", "reorganis", "organis",
    "annul", "effac", "modifi", "planifi", "decal", "boug", "retir", "enlev",
    "restaur", "optimis", "configur", "cal",
)
_PARTICIPE_NU = re.compile(
    r"(?<![a-z'])(?:%s)(?:e|ee|es|ees)\b" % "|".join(_RACINES_PARTICIPE_NU))
# Un mot qui precede un verbe conjugue ou un nom, pas un participe:
# « je le place », « que je deplace », « une place », « des places ».
_AVANT_VERBE_OU_NOM = {
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "le", "la", "les", "y", "en", "me", "te", "se", "lui", "leur", "ne",
    "un", "une", "des", "de", "du", "ta", "ma", "sa", "ton", "mon", "son",
    "tes", "mes", "ses", "ce", "cet", "cette", "ces", "notre", "votre", "leurs",
}
_CLAUSE = re.compile(r"[,;:]")
_FIN_QUESTION = re.compile(r"\?[\s\"'»)\]]*$")

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
    interrogative = (
        "?" in plat
        or bool(_MARQUE_OFFRE.search(plat))
        or bool(_CONDITIONNELLE.match(plat.strip()))
    )
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


def _participe_nu(clause: str) -> bool:
    """Un participe de mutation qui n'est ni un verbe conjugue ni un nom."""
    for trouve in _PARTICIPE_NU.finditer(clause):
        avant = clause[:trouve.start()].split()
        if not avant or avant[-1] not in _AVANT_VERBE_OU_NOM:
            return True
    return False


def _resultat_sans_marque(phrase: str) -> bool:
    """La regle « resultat », clause par clause, exemptee par une vraie
    marque interrogative ou une offre en tete de clause.

    Le decoupage se fait sur la phrase BRUTE: _normaliser efface les
    virgules, et « Ton cours deplace, ou autre chose ? » passait entier grace
    au « ou » de la seconde clause.
    """
    regle_resultat = dict(_REGLES)["resultat"]
    for brute in _CLAUSE.split(phrase):
        clause = " ".join(_normaliser(brute).replace("?", " ").split())
        if not clause:
            continue
        if not (regle_resultat.search(clause) or _participe_nu(clause)):
            continue
        if _MARQUE_INTERROGATIVE.search(clause) or _OFFRE_EN_TETE.match(clause):
            continue
        return True
    return False


def fuite_question(texte) -> list[str]:
    """Les regles qui fuient dans une question, un champ question ou une
    option. [] veut dire propre.

    - « premiere_personne »: j'ai, je t'ai, que j'ai, je viens de, c'est
      fait, voila qui est. Toujours une fuite, meme dans une question.
    - toute la guillotine existante, phrase par phrase (une phrase qui ne
      finit pas par « ? » garde ses regles d'intention);
    - « resultat » (passif ou participe nu d'une mutation), SAUF dans une
      clause qui porte une marque interrogative (quel, quand, ou, a quelle
      heure, est-ce que...) ou qui commence par une offre (veux-tu, tu veux,
      je peux, dis-moi ... et je).
    """
    if not texte or not isinstance(texte, str):
        return []
    fuites: list[str] = []

    def _noter(nom: str) -> None:
        if nom not in fuites:
            fuites.append(nom)

    for phrase in _phrases(texte):
        plat = _normaliser(phrase)
        if not plat.strip():
            continue
        if _PREMIERE_PERSONNE.search(_PREMIERE_PERSONNE_NEUTRE.sub("il faut ", plat)):
            _noter("premiere_personne")
        for nom in _fuites_d_une_phrase(phrase):
            if nom != "resultat":
                _noter(nom)
        if _resultat_sans_marque(phrase):
            _noter("resultat")
    return fuites


def _finit_par_question(phrase: str) -> bool:
    return bool(_FIN_QUESTION.search(phrase or ""))


def _fuites_de_prose(phrase: str) -> list[str]:
    """Le meme arbitre pour le detecteur et la guillotine: une phrase qui
    finit par « ? » passe par fuite_question, les autres par la guillotine."""
    if _finit_par_question(phrase):
        return fuite_question(phrase)
    return _fuites_d_une_phrase(phrase)


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
        for f in _fuites_de_prose(phrase):
            if f not in fuites:
                fuites.append(f)
    return fuites


# Un fragment orphelin qui suivait une phrase supprimee: « ... si tu
# confirmes, bien sur. » sans sa principale. On le supprime avec elle. Une
# question n'est JAMAIS un orphelin: « Et pour la duree, 1 h te va ? » tient
# debout seule.
_ORPHELIN = re.compile(
    r"^(?:si|et|mais|donc|car|ou|puis|alors|ensuite|sinon|comme)\b", re.IGNORECASE)


def _champs_du_schema(reponse) -> dict:
    return getattr(type(reponse), "model_fields", None) or {}


def epurer_reponse(reponse: ReponseDire) -> tuple[ReponseDire, int]:
    """Retire de la prose toute phrase qui affirme une action.

    Rend la reponse epuree et le nombre d'elements supprimes: phrases de
    ouverture et suite, question, options. Une question qui fuit emporte ses
    options avec elle; une option qui fuit part seule. question et options ne
    sont traites que s'ils existent dans le schema de la reponse. Ne touche
    pas aux actions structurees: elles ont leur propre garde (les references
    inconnues meurent dans assembler).
    """
    schema = _champs_du_schema(reponse)
    supprimees = 0
    champs: dict = {}
    for champ in ("ouverture", "suite"):
        gardees: list[str] = []
        precedente_supprimee = False
        coupees = 0
        for phrase in _phrases(getattr(reponse, champ, "") or ""):
            nette = phrase.strip()
            if _fuites_de_prose(nette):
                coupees += 1
                precedente_supprimee = True
                continue
            if (precedente_supprimee and _ORPHELIN.match(nette)
                    and not _finit_par_question(nette)):
                # Subordonnee detachee par « ... »: seule, elle n'a plus de
                # tete et mutile la voix davantage qu'elle ne la sert.
                coupees += 1
                continue
            precedente_supprimee = False
            gardees.append(nette)
        if coupees:
            supprimees += coupees
            champs[champ] = " ".join(gardees)

    if "question" in schema:
        question = getattr(reponse, "question", "") or ""
        if question.strip() and fuite_question(question):
            supprimees += 1
            champs["question"] = ""
            if "options" in schema and getattr(reponse, "options", None):
                champs["options"] = []
    if "options" in schema and "options" not in champs:
        options = list(getattr(reponse, "options", None) or [])
        gardees_opt = [o for o in options
                       if not (isinstance(o, str) and fuite_question(o))]
        if len(gardees_opt) != len(options):
            supprimees += len(options) - len(gardees_opt)
            champs["options"] = gardees_opt

    if not supprimees:
        return reponse, 0
    return reponse.model_copy(update=champs), supprimees


def fuites_reponse(reponse: ReponseDire) -> list[str]:
    """Observe ouverture, suite, question et options (ces deux-la seulement
    s'ils existent dans le schema), pas les actions structurees."""
    schema = _champs_du_schema(reponse)
    fuites: list[str] = []
    for champ in ("ouverture", "suite"):
        for fuite in fuite_lexicale(getattr(reponse, champ, "")):
            fuites.append(f"{champ}:{fuite}")
    if "question" in schema:
        for fuite in fuite_question(getattr(reponse, "question", "") or ""):
            fuites.append(f"question:{fuite}")
    if "options" in schema:
        vues: list[str] = []
        for option in getattr(reponse, "options", None) or []:
            for fuite in fuite_question(option):
                if fuite not in vues:
                    vues.append(fuite)
        fuites.extend(f"options:{fuite}" for fuite in vues)
    return fuites
