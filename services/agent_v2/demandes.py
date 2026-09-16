"""
Les DEMANDES: comment une garde du code pose une question, et comment le tour
suivant lit la reponse.

Une demande naît quand le code retient une action (suppression, ajouts en
serie, plan de la semaine) ou quand une heure dite par l'utilisateur est
refusee. Elle voyage dans ToolResult.data["demande"], b6 la persiste sur le
message assistant du tour (metadata["demandes"], avec ses puces), et le tour
suivant la relit ici.

Deux regles de fond, verifiees par les tests de core/test_agent_v2_gardes.py:

1. La demande elle-meme n'autorise JAMAIS l'action. Le message qui demande de
   supprimer ne peut pas etre sa propre confirmation: il n'existe aucune
   demande en attente au tour de la requete.
2. La reponse se lit PAR DEMANDE, jamais sur une liste. Une puce « Tous les
   jeudis » repond a la question de portee, pas a la confirmation d'un
   planning vide pose au meme tour.
3. Round 6: une suppression ne se tranche que sur la puce EXACTE. Une reponse
   libre ne peut que fermer la demande (garder, annuler).

Tout est pur sauf demandes_en_attente, qui lit la conversation.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import date, datetime, timedelta

from django.utils import timezone

# Au-dela, la question est perimee: une puce touchee le lendemain ne supprime
# plus rien, elle fait reposer la question.
FENETRE_ATTENTE = timedelta(minutes=30)

LEXIQUE_SUPPRESSION = (
    r"\b(supprim\w*|effac\w*|enlev\w*|retir\w*|annul\w*|vide[rz]?|debarrasse\w*)\b"
)
_LEXIQUE_SUPPRESSION = re.compile(LEXIQUE_SUPPRESSION)

_JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_MOIS = {
    "janv": 1, "fevr": 2, "mars": 3, "avr": 4, "mai": 5, "juin": 6,
    "juil": 7, "aout": 8, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_RE_MOIS = "|".join(_MOIS)


# --------------------------------------------------------------- normalisation

def sans_accents(texte) -> str:
    """Minuscules, accents retires, apostrophe typographique ramenee a '."""
    if not isinstance(texte, str):
        return ""
    texte = texte.replace("’", "'").replace("ʼ", "'")
    plat = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode("ascii")
    return plat.lower()


def normaliser(texte) -> str:
    """Forme de comparaison exacte (R1): ponctuation retiree, espaces reduits."""
    plat = sans_accents(texte)
    plat = re.sub(r"[^\w\s]", " ", plat)
    return re.sub(r"\s+", " ", plat).strip()


def _plat(texte) -> str:
    """Forme de lecture (R2, R3): garde l'apostrophe et le trait d'union, qui
    portent « d'accord », « vas-y » et « celui-la »."""
    plat = sans_accents(texte)
    plat = re.sub(r"[^\w\s'-]", " ", plat)
    return re.sub(r"\s+", " ", plat).strip()


# ---------------------------------------------------------------------- cles

def cle_demande(nom: str, identite: dict) -> str:
    """Cle d'IDENTITE de la cible. Jamais un argument cosmetique ni `confirm`:
    une confirmation qui basculerait la cle ne retrouverait plus sa question."""
    brut = nom + json.dumps(identite, sort_keys=True, default=str)
    return hashlib.sha1(brut.encode("utf-8")).hexdigest()[:12]


def construire_demande(type, motif, outil, parametres, cible, options, cle) -> dict:  # noqa: A002
    return {
        "type": type,
        "motif": motif,
        "cle": cle,
        "outil": outil,
        "parametres": dict(parametres or {}),
        "cible": dict(cible or {}),
        "options": list(options or []),
        "emise_le": timezone.now().isoformat(),
    }


# ------------------------------------------------------------------- attente

def _moment(valeur):
    if not isinstance(valeur, str) or not valeur:
        return None
    try:
        moment = datetime.fromisoformat(valeur)
    except ValueError:
        return None
    if timezone.is_naive(moment):
        moment = timezone.make_aware(moment, timezone.get_current_timezone())
    return moment


def demandes_en_attente(user, maintenant=None) -> list[dict]:
    """Les demandes auxquelles le message COURANT peut repondre.

    Lecture par pk decroissant: [0] le message courant (sauve avant AGIR),
    [1] la reponse de l'assistant, [2] le message utilisateur qu'elle traitait.
    Toute autre forme ne vaut rien: un tour intercale ou deux tours chevauches
    (un tour abandonne qui sauve sa reponse apres le message suivant) rendent
    la question inutilisable, et le pire effet est une question reposee.
    """
    from core.models import ConversationMessage

    lignes = list(
        ConversationMessage.objects.filter(user=user).order_by("-pk")[:3]
    )
    if len(lignes) < 3:
        return []
    courant, assistant, precedent = lignes
    if courant.role != "user" or assistant.role != "assistant" or precedent.role != "user":
        return []
    meta = assistant.metadata if isinstance(assistant.metadata, dict) else {}
    if meta.get("en_reponse_a") != precedent.pk:
        return []
    maintenant = maintenant or timezone.now()
    valides = []
    for demande in meta.get("demandes") or []:
        if not isinstance(demande, dict) or not demande.get("cle"):
            continue
        emise = _moment(demande.get("emise_le"))
        if emise is None:
            continue
        if emise > maintenant + timedelta(minutes=1):
            continue
        if maintenant - emise >= FENETRE_ATTENTE:
            continue
        valides.append(demande)
    return valides


# ------------------------------------------------------------ lecture reponse
#
# Round 6 (D1). Les rounds 1 a 5 lisaient la reponse libre: oui en tete, portee
# nommee (« tous les jeudis », « juste celui-la »), reponse nue, verbe de garde.
# Chaque revue trouvait une tournure mal lue qui supprimait, et chaque
# rafistolage ouvrait un trou neuf. La surface est retiree:
#
# 1. Une option destructive (serie, occurrence, confirmer) ne sort QUE d'une
#    puce exacte de la demande: egalite normalisee (casse, accents, espaces,
#    ponctuation) avec sa valeur ou son libelle.
# 2. Une reponse libre ne peut produire que « annuler », et seulement quand
#    elle ne dit rien d'autre que garder ou annuler. Tout le reste ne donne
#    aucune option: le code repose la question une fois, puis l'abandonne.

MOTIFS_LECTURE_LIBRE = {"destructif", "creation_en_masse", "optimisation", "portee_jour"}
# Round 8 (F4): un oui clair confirme ce qui ne detruit rien. Seule la creation
# en masse en fait partie. « heure_refusee » n'a pas d'option « confirmer »:
# un oui ne dit pas QUEL creneau, et en choisir un changerait l'heure a la
# place de l'utilisateur (I3). Suppressions, vidage, annulations, arrets et
# plan de la semaine restent a la puce exacte (I1).
MOTIFS_OUI_LIBRE = {"creation_en_masse"}
_OUI = {"oui", "ok", "okay", "ouais", "yes", "go", "continue", "vas-y", "d'accord", "daccord"}
_POLITESSE = {"merci", "stp", "svp", "s'il", "te", "vous", "plait"}


def oui_clair(message_brut) -> bool:
    """Le message n'est-il qu'un oui, avec au plus de la politesse autour ?"""
    if not isinstance(message_brut, str) or "?" in message_brut:
        return False
    mots = [m for m in (x.strip("'-") for x in re.findall(r"[a-z0-9'-]+", _plat(message_brut))) if m]
    return (any(m in _OUI for m in mots)
            and all(m in _OUI or m in _POLITESSE for m in mots))


def _ids_options(demande: dict) -> set:
    return {
        o.get("id") for o in demande.get("options") or []
        if isinstance(o, dict) and o.get("id")
    }


def puce_touchee(message_brut, demande: dict):
    """L'option dont le message est la puce EXACTE, ou None.

    « Tous les jeudis ? » n'est pas la puce « Tous les jeudis »: normaliser
    effacerait le « ? » d'un utilisateur encore hesitant (revue du round 4).
    """
    if not isinstance(demande, dict) or not isinstance(message_brut, str):
        return None
    ids = _ids_options(demande)
    cible = normaliser(message_brut)
    if not ids or not cible:
        return None
    interrogatif = "?" in message_brut
    for puce in demande.get("chips") or []:
        if not isinstance(puce, dict):
            continue
        option = puce.get("option")
        if option not in ids:
            continue
        for champ in ("value", "label"):
            texte = puce.get(champ)
            if interrogatif and "?" not in str(texte or ""):
                continue
            if normaliser(texte) == cible:
                return option
    return None


# -- garder ou annuler, en texte libre

_MARQUE_GARDE = re.compile(r"\b(?:gard\w*|laiss\w*|conserv\w*|annul\w*|non|nan|finalement|arret\w*)\b")
_VERBE_SUPPRESSION = re.compile(r"\b(?:supprim\w*|effac\w*|enlev\w*|retir\w*|vide[rz]?|debarrass\w*)\b")
# Ces verbes ne gardent que nies (« ne touche pas », « change rien »).
_VERBE_A_NIER = re.compile(r"\b(?:touch\w*|chang\w*|boug\w*)\b")
_CHANGE_D_AVIS = re.compile(r"\bchang\w*\s+d'?\s*avis\b")
_NEGATIONS_DU_VERBE = {"pas", "rien", "aucun", "aucune", "jamais", "pu", "plus", "ne"}
# « non pas tous les jeudis », « pas juste celui-la »: la portee elle-meme est
# niee, l'utilisateur en veut peut-etre une autre.
_PORTEE_NIEE = re.compile(
    r"\b(?:pas|jamais|non)\s+(?:tous|toutes|tout|chaque|la serie|juste|seulement|"
    r"ce|cet|cette|celui|celle)\b")
# Une portee d'une fois (« laisse ce jeudi », « garde celui-la ») laisse
# entendre qu'une autre occurrence part: on ne ferme pas la demande.
_OCCURRENCE = re.compile(
    r"\b(?:seulement|juste|cette fois|celui-la|celle-la|"
    r"ce (?:lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche))\b")
# « laisse tomber le cours » peut vouloir dire « supprime le cours ».
_TOMBER_AVEC_OBJET = re.compile(
    r"\blaiss\w*[\s-]+tomber\s+(?:le|la|les|l'|ce|cet|cette|ces|mon|ma|mes|ton|ta|tes|"
    r"son|sa|ses|un|une)\b")
_VOCABULAIRE_DE_GARDE = {
    "non", "nan", "ne", "n", "pas", "rien", "plus", "pu", "jamais", "tomber",
    "le", "la", "les", "l", "lui", "leur", "y", "en", "ce", "ca", "c", "ces",
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses", "moi",
    "tous", "toutes", "tout", "toute", "chaque", "semaine", "semaines", "serie",
    "toujours", "definitivement", "reste", "restent", "rester", "comme", "est",
    "je", "j", "ai", "veux", "voudrais", "prefere", "te", "dis", "a", "au", "aux",
    "de", "des", "du", "d", "avis", "alors", "bon", "ben", "euh", "hmm", "tant", "pis",
    "merci", "stp", "svp", "s", "il", "plait", "finalement", "encore",
}


def _mots(texte: str) -> list[str]:
    return [m for m in re.split(r"[\s-]+", texte) if m]


def _verbe_nie(plat: str, m) -> bool:
    if plat[:m.start()].endswith("n'"):
        return True
    avant = [x.strip("'") for x in _mots(plat[:m.start()])][-2:]
    apres = [x.strip("'") for x in _mots(plat[m.end():])][:2]
    return bool(_NEGATIONS_DU_VERBE.intersection(avant + apres))


_VERBE_DE_GARDE = re.compile(r"(?:gard|laiss|conserv|arret)")
# Avant un verbe de garde, « rien » est l'objet du verbe precedent (« n'efface
# rien, garde-le »): il ne nie pas la garde.
_NEGATIONS_AVANT_GARDE = {"ne", "pas", "plus", "pu", "jamais"}
_NEGATIONS_APRES_GARDE = {"pas", "rien", "plus", "pu", "jamais", "aucun", "aucune"}


def _garde_nie(plat: str, m) -> bool:
    if plat[:m.start()].endswith("n'"):
        return True
    avant = [x.strip("'") for x in _mots(plat[:m.start()])][-2:]
    apres = [x.strip("'") for x in _mots(plat[m.end():])][:2]
    return bool(_NEGATIONS_AVANT_GARDE.intersection(avant)
                or _NEGATIONS_APRES_GARDE.intersection(apres))


def _jetons(texte: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", sans_accents(texte))


def _meme_mot(a: str, b: str) -> bool:
    if a == b:
        return True
    return len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4]


def _jours_permis(jour) -> set:
    """Les noms de jour qu'une reponse peut porter: celui de la cible
    seulement. « garde-le vendredi » ne repond pas a une question sur jeudi."""
    if isinstance(jour, int) and 0 <= jour <= 6:
        return {_JOURS[jour], _JOURS[jour] + "s"}
    return set()


def annulation_libre(message_brut, demande: dict) -> bool:
    """Le message ne dit-il QUE garder ou annuler, sans rien nommer d'autre ?

    Vrai ferme la demande sur « annuler »: rien ne s'execute. Tout doute rend
    Faux, donc aucune option.
    """
    if not isinstance(message_brut, str) or "?" in message_brut or not isinstance(demande, dict):
        return False
    plat = _plat(message_brut)
    if not plat or _PORTEE_NIEE.search(plat) or _OCCURRENCE.search(plat):
        return False
    if _TOMBER_AVEC_OBJET.search(plat):
        return False
    # « J'annule le dentiste ? » « annule »: la c'est un oui. Ni l'un ni
    # l'autre ne se lit en texte libre.
    annuler_est_l_action = demande.get("outil") == "cancel_scheduled_block"
    garde = False
    for m in _VERBE_SUPPRESSION.finditer(plat):
        if not _verbe_nie(plat, m):
            return False
        garde = True
    for m in _VERBE_A_NIER.finditer(plat):
        if _CHANGE_D_AVIS.match(plat, m.start()):
            garde = True
            continue
        if not _verbe_nie(plat, m):
            return False
        garde = True
    for m in _MARQUE_GARDE.finditer(plat):
        if annuler_est_l_action and m.group(0).startswith("annul"):
            return False
        # Round 8: « ne le garde pas », « garde rien », « laisse pas »,
        # « n'arrete pas » disent l'inverse: la demande se repose.
        if _VERBE_DE_GARDE.match(m.group(0)) and _garde_nie(plat, m):
            return False
        garde = True
    if not garde:
        return False
    cible = demande.get("cible") or {}
    titre = [m for m in _jetons(cible.get("titre") or "") if len(m) >= 3]
    jours = _jours_permis(cible.get("jour"))
    for mot in _jetons(plat):
        if (mot in _VOCABULAIRE_DE_GARDE or mot in jours
                or _MARQUE_GARDE.fullmatch(mot) or _VERBE_SUPPRESSION.fullmatch(mot)
                or _VERBE_A_NIER.fullmatch(mot) or any(_meme_mot(mot, t) for t in titre)):
            continue
        return False
    return True


def option_choisie(message_brut: str, demande: dict,
                   tap: dict | None = None) -> str | None:
    """L'option que CE message choisit pour CETTE demande, ou None.

    Jamais evaluee sur une liste: chaque demande ne connait que ses propres
    puces. Seule la puce exacte donne une option destructive; une reponse
    libre ne donne au mieux que « annuler ».

    `tap` est le postback structure du front ({"demande": cle, "option": id})
    envoye quand l'utilisateur touche une puce: l'egalite d'identifiants
    remplace la comparaison de texte. Il n'ouvre aucune surface nouvelle
    (equivalent byte-exact de taper la puce) et ne vaut que pour la demande
    dont il porte la cle.
    """
    if not isinstance(demande, dict) or not isinstance(message_brut, str):
        return None
    ids = _ids_options(demande)
    if not ids:
        return None
    if (isinstance(tap, dict) and tap.get("demande")
            and tap.get("demande") == demande.get("cle")
            and tap.get("option") in ids):
        return tap["option"]
    choix = puce_touchee(message_brut, demande)
    if (choix is None and "confirmer" in ids
            and demande.get("motif") in MOTIFS_OUI_LIBRE and oui_clair(message_brut)):
        choix = "confirmer"
    if (choix is None and "annuler" in ids
            and demande.get("motif") in MOTIFS_LECTURE_LIBRE
            and annulation_libre(message_brut, demande)):
        choix = "annuler"
    return choix if choix in ids else None


# -- reposer ou abandonner (D2): cette lecture n'autorise JAMAIS rien

# Les mots qu'une reponse floue peut porter sans nommer autre chose que sa
# cible: portee, politesse, hesitation, pronoms.
_VOCABULAIRE_DE_REPONSE = _VOCABULAIRE_DE_GARDE | {
    "oui", "ok", "okay", "ouais", "yes", "confirme", "confirmer", "vas", "go", "est", "bon",
    "accord", "daccord", "parfait", "super", "bien", "sur", "vraiment", "absolument",
    "fait", "exactement", "allez", "fais", "applique", "continue", "ajouts", "plan",
    "certain", "certaine", "sure", "donc", "plutot", "seulement", "juste", "cette", "fois",
    "celui", "celle", "cet", "mais", "ou", "et", "bof", "sais", "completement", "prochain",
    "prochaine", "occurrence",
    # Noms generiques et petits nombres: « Oui, supprime ces trois blocs. »
    # ne nomme aucun AUTRE element; le oui en tete ne suffit plus (round 6).
    "bloc", "blocs", "creneau", "creneaux", "element", "elements", "evenement",
    "evenements", "ceux", "celles", "ci", "un", "une", "deux", "trois", "quatre", "cinq",
}

# Une reponse qui ouvre une AUTRE demande n'est pas une reponse floue a la
# question en attente (revue du round 4: « c'est quoi mon horaire demain ? »,
# « merci, bonne nuit »).
_NOUVELLE_REQUETE = re.compile(
    r"\b(?:ajout\w*|cree\w*|creer|mets|met|place\w*|planifi\w*|deplac\w*|bouge\w*|"
    r"montre\w*|affiche\w*|horaire|planning|agenda|quoi|quel\w*|quand|combien|"
    r"merci|bonne|bonjour|salut|allo)\b")
# Round 8 (F6): « efface tout », « vide tout », « supprime tous mes blocs »,
# « supprime tout ce jeudi » ouvrent une nouvelle demande destructive. AGIR la
# sert, sous ses propres gardes; rien ne s'execute par cette lecture. « tous
# les jeudis » reste une portee, donc une reponse floue.
_NOUVELLE_DESTRUCTION = re.compile(
    r"\b(?:supprim\w*|effac\w*|enlev\w*|retir\w*|vide[rz]?)\s+"
    r"(?:tout\b|(?:tous|toutes)\s+(?!les\s+(?:" + "|".join(_JOURS) + r")s?\b))")


def _nomme_rien_d_autre(plat: str, demande: dict | None) -> bool:
    """La reponse ne nomme que sa cible: aucun titre ni jour d'un autre
    element (« supprime mon gym tous les jeudis » repondu a la question sur
    le quart est une nouvelle requete)."""
    cible = (demande or {}).get("cible") or {}
    titre = [m for m in _jetons(cible.get("titre") or "") if len(m) >= 3]
    permis_jours = _jours_permis(cible.get("jour")) or (set(_JOURS) | {j + "s" for j in _JOURS})
    for mot in _jetons(plat):
        if mot in _VOCABULAIRE_DE_REPONSE or mot in permis_jours:
            continue
        if _VERBE_SUPPRESSION.fullmatch(mot) or re.fullmatch(r"annul\w*", mot):
            continue
        if any(_meme_mot(mot, t) for t in titre):
            continue
        return False
    return True


def reponse_plausible(message_brut, demande: dict) -> bool:
    """Le message peut-il etre une reponse, meme floue, a CETTE demande ?

    Ne sert qu'a choisir entre reposer la question et l'abandonner (D2), et a
    dire si le tour peut se passer d'AGIR (D6). Une erreur ici coute une
    question reposee ou abandonnee, jamais une action.
    """
    plat = _plat(message_brut)
    if not plat or _NOUVELLE_REQUETE.search(plat) or _NOUVELLE_DESTRUCTION.search(plat):
        return False
    # Revue du round 6: un oui ou un non en tete ne suffit plus. « non,
    # supprime plutot mon gym » est une correction qui porte une nouvelle
    # requete; lue comme reponse, elle faisait sauter AGIR (D6). Le message
    # entier ne doit nommer que la cible.
    return _nomme_rien_d_autre(plat, demande)


# ------------------------------------------------------------- jours, heures

_RE_JOUR_VISE = re.compile(
    r"\b(" + "|".join(_JOURS) + r")s?\b"
    r"|\bapres-demain\b|\bdemain\b|\baujourd'?hui\b|\bce soir\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}(?:er)?\s+(?:" + _RE_MOIS + r")\w*"
    r"|\b\d{1,2}/\d{1,2}\b"
)


def jour_vise(message_brut) -> bool:
    """Le message nomme-t-il un jour (nom, demain, une date) ?"""
    return bool(_RE_JOUR_VISE.search(sans_accents(message_brut)))


def suppression_demandee(message_brut) -> bool:
    return bool(_LEXIQUE_SUPPRESSION.search(sans_accents(message_brut)))


def _dates_nommees(message_brut, aujourdhui: date) -> list[date]:
    """Les dates que le message nomme, dans l'ordre de preference: date
    explicite, puis relative, puis nom de jour (prochaine occurrence)."""
    plat = sans_accents(message_brut)
    dates: list[date] = []
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", plat):
        try:
            dates.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
    for m in re.finditer(r"\b(\d{1,2})(?:er)?\s+(" + _RE_MOIS + r")\w*", plat):
        jour_num, mois = int(m.group(1)), _MOIS[m.group(2)]
        try:
            candidate = date(aujourdhui.year, mois, jour_num)
        except ValueError:
            continue
        if candidate < aujourdhui - timedelta(days=60):
            try:
                candidate = date(aujourdhui.year + 1, mois, jour_num)
            except ValueError:
                continue
        dates.append(candidate)
    for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})\b", plat):
        try:
            dates.append(date(aujourdhui.year, int(m.group(2)), int(m.group(1))))
        except ValueError:
            pass
    if re.search(r"\bapres-demain\b", plat):
        dates.append(aujourdhui + timedelta(days=2))
    elif re.search(r"\bdemain\b", plat):
        dates.append(aujourdhui + timedelta(days=1))
    if re.search(r"\baujourd'?hui\b|\bce soir\b", plat):
        dates.append(aujourdhui)
    for m in re.finditer(r"\b(" + "|".join(_JOURS) + r")s?\b", plat):
        dow = _JOURS.index(m.group(1))
        dates.append(prochaine_occurrence(dow, aujourdhui))
    return dates


def prochaine_occurrence(dow: int, aujourdhui: date | None = None) -> date:
    """Prochaine date (aujourd'hui compris) dont le jour de semaine vaut dow
    (0 = lundi)."""
    aujourdhui = aujourdhui or timezone.localdate()
    return aujourdhui + timedelta(days=(dow - aujourdhui.weekday()) % 7)


def date_visee(message_brut, dow_bloc: int, aujourdhui: date | None = None) -> date:
    """La date d'occurrence que vise le message pour un bloc du jour dow_bloc.

    Une date nommee qui ne tombe pas le jour du bloc est ignoree: sauter
    l'occurrence d'un autre jour echouerait. On retombe alors sur la
    prochaine occurrence du jour du bloc.
    """
    aujourdhui = aujourdhui or timezone.localdate()
    for candidate in _dates_nommees(message_brut, aujourdhui):
        if candidate.weekday() == dow_bloc:
            return candidate
    return prochaine_occurrence(dow_bloc, aujourdhui)


_RE_HEURE = re.compile(
    # [ \t]* et non \s*: « 2026-09-17\nHeure du rendez-vous » se lisait 17:00.
    r"(?<![\d:-])([01]?\d|2[0-3])[ \t]*(?:heures?|h)(?![a-z])[ \t]*([0-5]\d)?(?!\d)"
    r"|(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?!\d)"
    # « apres midi » et « avant midi » sans trait d'union ne sont pas 12 h.
    r"|(?<![-\w])(?<!apres )(?<!avant )(midi|minuit)\b"
)
# « pour » n'en fait pas partie: « Va pour 11 h 50 » est la valeur d'une puce.
_AVANT_DUREE = re.compile(r"(pendant|durant|dure|duree de)\s*$")
# Round 9: « 2 h de l'apres-midi », « 2 h de la nuit » donnent une heure, pas
# une duree. La lecture stricte (H+12) est choisie par outils._lectures_d_heure.
_APRES_DUREE = re.compile(
    r"^\s*(de\s(?!l'?\s*(?:apres[- ]midi|aprem)\b|la\s+(?:nuit|soiree|matinee)\b)"
    r"|d'(?!\s*(?:apres[- ]midi|aprem)\b)|par (jour|semaine|soir|seance)\b)")
# Une reponse de formulaire « Durée: 1 h » ou « Temps d'étude total: 4 h »
# donne une DUREE. Lue comme 01:00 ou 04:00, elle faisait retenir les ajouts
# du formulaire d'etude (banc du 2026-09-14, s02-2). Le libelle se lit sur la
# meme ligne, avant les deux-points.
# « Heures d'étude: 4 h » (banc du round 4, s02-2) est un libelle de duree;
# « Heure du rendez-vous: 14 h », au singulier, reste une heure.
_LIBELLE_DUREE = re.compile(
    r"(?:^|\n)[^\n:]*\b(dure\w*|temps|total\w*|combien|longueur|nombre d'heures|"
    r"heures par|heures d'|heures de|volume)(?:\b|(?<='))[^\n:]*:\s*$")


def heures_dites(message_brut) -> list[str]:
    """Les heures d'horloge que l'utilisateur a donnees, en "HH:MM".

    « 10h30 », « 10 h 30 », « 10:30 » et « 14h » comptent; une duree
    (« pendant 2 h », « 2 h de lecture ») ne compte pas.
    """
    heures: list[str] = []
    for valeur, _debut in heures_dites_positions(message_brut):
        if valeur not in heures:
            heures.append(valeur)
    return heures


def heures_dites_positions(message_brut) -> list[tuple[str, int]]:
    """Comme heures_dites, avec la position de chaque heure dans le texte
    SANS ACCENTS (meme longueur utile pour decouper en propositions)."""
    plat = sans_accents(message_brut)
    sortie: list[tuple[str, int]] = []
    for m in _RE_HEURE.finditer(plat):
        if m.group(5):
            valeur = "12:00" if m.group(5) == "midi" else "00:00"
        else:
            if m.group(1) is not None:
                h, mn = int(m.group(1)), int(m.group(2) or 0)
                if _APRES_DUREE.match(plat[m.end():]):
                    continue
            else:
                h, mn = int(m.group(3)), int(m.group(4))
            if _AVANT_DUREE.search(plat[:m.start()]) or _LIBELLE_DUREE.search(plat[:m.start()]):
                continue
            valeur = f"{h:02d}:{mn:02d}"
        sortie.append((valeur, m.start()))
    return sortie
