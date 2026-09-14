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
    # Futur simple: « je m'occuperai de te trouver les creneaux », « je le
    # placerai » (banc du round 3, s06-1).
    ("futur_simple", re.compile(
        rf"\bje\s+(?:{_CLIT})*(?:(?:{'|'.join(_RACINES)})erai|occuperai|trouverai|mettrai|remettrai)\b")),
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
_REGLES_D_INTENTION = {"futur", "futur_simple", "present", "prise_en_charge", "passe_recent"}
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
# Revue de verite du round 4: « Je l'ai mis jeudi », « je te les ai
# inscrits » et « C'est noté pour jeudi » passaient (clitiques et
# participes hors des racines).
_JE_VIENS_DE = (
    r"je\s+(?:(?:l|t|m|s)'\s*|(?:le|la|les|lui|leur|te|en|y|me)\s+)*viens\s+(?:\w+\s+){0,2}d(?:e\b|')")
_PREMIERE_PERSONNE = re.compile(
    # Revue de verite du round 6: « que je viens d'ajouter » et « je viens
    # tout juste de placer » passaient (elision et mots intercales).
    r"(?<![a-z])(?:j'ai|je\s+t'ai|que\s+j'ai|j'ai\s+deja|c'est\s+fait|voila\s+qui\s+est"
    r"|" + _JE_VIENS_DE +
    r"|je\s+(?:(?:l|t|m|s)'\s*|(?:le|la|les|lui|leur|te|vous|nous|en|y|me)\s+)+ai"
    r"|c'est\s+(?:note|confirme|inscrit|enregistre|reserve))\b")
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
    # Determinants interrogatifs et indefinis: « quelle place », « une autre place ».
    "quel", "quelle", "quels", "quelles", "autre", "autres", "chaque", "aucune", "aucun",
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
    # Revue de verite du round 3: le participe NU d'une mutation n'etait
    # cherche que dans les questions. « Gym retire pour jeudi. » passait en
    # ouverture pendant que le code demandait s'il fallait le retirer.
    if "resultat" not in fuites and _resultat_sans_marque(phrase):
        fuites.append("resultat")
    return fuites


# Un imperatif en tete de proposition, suivi d'un determinant: « Place ma
# revision jeudi. », « Ajoute le gym a 9 h. » C'est l'utilisateur qui parle
# (valeur d'une puce), pas un participe qui raconte une action faite.
_DETERMINANT_APRES = re.compile(
    r"^\s+(?:ma|ta|mon|ton|mes|tes|le|la|les|l'|un|une|des|du|ce|cet|cette|ces|notre|votre|nos|vos|moi|toi|y|en)\b")


def _imperatif_en_tete(clause: str, debut: int, fin: int) -> bool:
    return not clause[:debut].split() and bool(_DETERMINANT_APRES.match(clause[fin:]))


def _participe_nu(clause: str) -> bool:
    """Un participe de mutation qui n'est ni un verbe conjugue ni un nom."""
    for trouve in _PARTICIPE_NU.finditer(clause):
        avant = clause[:trouve.start()].split()
        if _imperatif_en_tete(clause, trouve.start(), trouve.end()):
            continue
        if not avant or avant[-1] not in _AVANT_VERBE_OU_NOM:
            return True
    return False


# Revue de verite du 2026-09-14: exempter toute la clause des qu'elle portait
# un mot interrogatif laissait passer « Ton cours est deplace a 14 h ou tu
# preferes 15 h ? ». L'exemption est desormais PAR PARTICIPE: seul le mot
# interrogatif qui GOUVERNE le participe (juste apres, au plus une
# preposition entre les deux) en fait une vraie question sur l'etat.
_PARTICIPE_OU_LOCUTION = re.compile(
    r"(?<![a-z'])(?:(?:%s)(?:e|ee|es|ees)|mis(?:e|es)?\s+a\s+jour|mis(?:e|es)?\s+en\s+place)\b"
    % "|".join(sorted(set(_RACINES) | set(_RACINES_PARTICIPE_NU), key=len, reverse=True)))
_AUXILIAIRE_AVANT = re.compile(
    r"\b(a|ont|est|sont|etait|etaient)\s+((?:ete\s+)?)((?:(?:bien|deja|tout|aussi|donc)\s+)*)$")
_GOUVERNE = re.compile(
    r"^(?:(?:a|pour|vers|en|le|la|des|depuis|jusqu'a|jusqu a|jusque)\s+)?"
    r"(quel|quelle|quels|quelles|quand|combien|lequel|laquelle|lesquels|lesquelles"
    r"|ou|comment|pourquoi)\b(?:\s+(\S+))?")
_PRONOMS_SUJETS = {"je", "j'", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles", "t'", "c'est"}
# « maintenant que ton quart est place », « vu que le gym est cale »: une
# subordonnee qui pose l'action comme acquise, quelle que soit la suite.
_ACQUIS = re.compile(
    r"\b(?:maintenant|vu|puisque|puis|depuis|une\s+fois|comme)\s*(?:qu'|que\s+|qu\s+)?.*"
    r"\b(?:a|ont|est|sont)\s+(?:ete\s+)?(?:(?:bien|deja|tout|aussi|donc)\s+)*"
    r"(?:(?:%s)(?:e|ee|es|ees)\b|mis(?:e|es)?\s+a\s+jour)" % "|".join(_RACINES))
_EST_CE_QUE_EN_TETE = re.compile(r"^(?:et\s+|alors\s+|donc\s+)?est\s+ce\s+qu")
# Un passif hypothetique ou a l'infinitif n'affirme rien: « Tu veux qu'il
# soit deplace ? », « Ca doit etre place avant midi ? ».
_HYPOTHETIQUE = {"soit", "soient", "sois", "etre"}


def _participe_exempte(clause: str, debut: int, fin: int) -> bool:
    """Ce participe precis est-il ce que la question demande ?"""
    avant = clause[:debut]
    mots_avant = avant.split()
    if mots_avant and mots_avant[-1] in _HYPOTHETIQUE:
        return True
    # Alternative d'une vraie question: « annule ou juste decale ? ».
    if re.search(r"\bou\s+(?:juste|seulement|plutot|bien|simplement)?\s*$", avant):
        return True
    aux = _AUXILIAIRE_AVANT.search(avant)
    passe_accompli = bool(aux and (aux.group(2) or aux.group(1) in ("a", "ont")))
    confirmation = bool(aux and aux.group(3)) or " deja" in f" {clause}"
    if _EST_CE_QUE_EN_TETE.match(clause) and aux and not confirmation:
        # « Est-ce que ton horaire a change ? », question fermee sur l'etat.
        return True
    if passe_accompli or confirmation:
        # « a ete deplace a quelle heure ? » affirme le deplacement.
        return False
    gouverne = _GOUVERNE.match(clause[fin:].strip())
    if not gouverne:
        return False
    if gouverne.group(1) == "ou":
        suivant = gouverne.group(2) or ""
        # « deplace a 14 h ou tu preferes » : le « ou » ouvre une autre
        # proposition, il ne questionne pas le participe.
        if suivant in _PRONOMS_SUJETS or suivant.startswith(("j'", "t'")):
            return False
    return True


def _resultat_sans_marque(phrase: str) -> bool:
    """La regle « resultat », clause par clause et PARTICIPE par participe.

    Un participe de mutation (lie a un auxiliaire ou nu) est une fuite, sauf
    quand la question porte sur lui: mot interrogatif qui le gouverne
    (« place a quelle heure », « place quand »), alternative (« annule ou
    juste decale »), « est-ce que » en tete d'une question fermee, ou passif
    hypothetique (« qu'il soit deplace »). Une marque ailleurs dans la clause
    (« ou tu preferes », « quel autre bloc ») n'exempte rien, et « maintenant
    que X est place » est une affirmation sans condition.

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
        if _ACQUIS.search(clause):
            return True
        for trouve in _PARTICIPE_OU_LOCUTION.finditer(clause):
            mots_avant = clause[:trouve.start()].split()
            lie = bool(_AUXILIAIRE_AVANT.search(clause[:trouve.start()]))
            nu = not mots_avant or mots_avant[-1] not in _AVANT_VERBE_OU_NOM
            if not (lie or nu):
                continue  # verbe conjugue ou nom: « je le place », « une place »
            if not lie and _imperatif_en_tete(clause, trouve.start(), trouve.end()):
                continue
            if not _participe_exempte(clause, trouve.start(), trouve.end()):
                return True
    return False


# « Je supprime ton cours de jeudi, ca te va ? »: un present d'action suivi
# d'une demande d'aval est une annonce, pas une offre.
_AVAL_EN_QUEUE = re.compile(
    r"^(?:ok|okay|ca\s+te\s+va|ca\s+va|ca\s+marche|d'accord|c'est\s+bon|tu\s+es\s+d'accord|parfait)$")


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
        clauses = [c for c in _CLAUSE.split(phrase) if _normaliser(c).replace("?", " ").strip()]
        if len(clauses) >= 2:
            queue = " ".join(_normaliser(clauses[-1]).replace("?", " ").split())
            tete = " ".join(_normaliser(" ".join(clauses[:-1])).replace("?", " ").split())
            if _AVAL_EN_QUEUE.match(queue):
                for nom in ("present", "futur"):
                    if dict(_REGLES)[nom].search(tete):
                        _noter(nom)
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


# Une tete de question ou d'offre, en DEBUT de proposition.
_TETE_DEMANDE = re.compile(
    r"^(?:et\s+|alors\s+|sinon\s+|ou\s+|donc\s+|mais\s+)?(?:"
    r"(?:quel|quelle|quels|quelles|quand|combien|lequel|laquelle|lesquels|lesquelles"
    r"|ou|comment|pourquoi)\b"
    r"|a\s+quelle\s+heure\b|pour\s+quel|est\s+ce\s+qu"
    r"|(?:veux|voudrais|souhaites|preferes|aimerais|peux|pourrais|dois)\s+tu\b"
    r"|tu\s+(?:veux|voudrais|preferes|aimerais|souhaites|peux)\b"
    r"|ca\s+te\s+va\b)")
# Ce qu'un brouillon ne peut JAMAIS porter, meme sous une tete de question.
_AFFIRMATION_DE_BROUILLON = re.compile(
    r"(?<![a-z])(?:j'ai|je\s+(?:(?:l|t|m|s)'\s*|(?:le|la|les|lui|leur|te|vous|nous|en|y|me)\s+)+ai"
    r"|" + _JE_VIENS_DE +
    r"|c'est\s+(?:note|bon|confirme|inscrit|reserve|enregistre|regle)"
    r"|maintenant|desormais|ne\s+figure\s+plus|plus\s+de)\b")


def _demande_structurelle(phrase: str) -> bool:
    """Revue de verite du round 4: le filtre lexical laissait passer « Je l'ai
    mis jeudi a 9 h, veux-tu que je change ? ». Round 6 (D4): la derniere
    proposition n'est plus exemptee. « Tu veux un rappel, ton gym est jeudi
    a 9 h ? » affirmait un etat derriere une tete de question. Une phrase ne
    passe que si CHAQUE proposition est une tete de question, d'offre ou de
    conditionnelle. Une vraie question ecartee ici ne coute rien: DIRE peut
    poser la sienne, et le code pose les questions gardees."""
    clauses = [" ".join(_normaliser(c).replace("?", " ").split()) for c in _CLAUSE.split(phrase)]
    clauses = [c for c in clauses if c]
    if not clauses:
        return False

    def _tete(c):
        return bool(_TETE_DEMANDE.match(c) or _OFFRE_EN_TETE.match(c) or _CONDITIONNELLE.match(c))

    return all(_tete(c) for c in clauses)


def questions_et_offres(texte) -> str:
    """Ce qui, d'un brouillon d'AGIR, peut entrer au brief de DIRE.

    Seulement les phrases qui DEMANDENT par leur STRUCTURE (voir
    _demande_structurelle), sans aucune marque d'action acquise, et que
    fuite_question juge propres. Toute phrase declarative tombe: c'est la que
    le modele raconte ses actions, y compris celles qu'une garde a retenues
    (revues de verite des rounds 3 et 4).
    """
    if not texte or not isinstance(texte, str):
        return ""
    gardees: list[str] = []
    for phrase in _phrases(texte):
        nette = phrase.strip()
        plat = _normaliser(nette).replace("?", " ").strip()
        if not plat:
            continue
        if not (_finit_par_question(nette) or _OFFRE_EN_TETE.match(plat)
                or _CONDITIONNELLE.match(plat)):
            continue
        if not _demande_structurelle(nette) or _AFFIRMATION_DE_BROUILLON.search(plat):
            continue
        if not fuite_question(nette):
            gardees.append(nette)
    return " ".join(gardees)


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
