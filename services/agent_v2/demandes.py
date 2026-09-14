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

_TETE_OUI = re.compile(
    r"^(je confirme|c'est bon|c est bon|d'accord|d accord|daccord|vas-y|vas y|"
    r"oui|okay|ok|confirme|go|applique)(?![\w'-])"
)
_TETE_NON = re.compile(r"^(non|annul\w*|laisse\w*|arret\w*|pas)(?![\w'-])")
_RESTE_INTERDIT = re.compile(
    r"supprim|effac|enlev|retir|annul|vid|ajout|cre[eé]|mets?\b|place|deplac|"
    r"change|modifi|bouge|aussi|mais|sauf"
)
_OCC = re.compile(
    r"\b(seulement|juste|cette fois|celui-la|celle-la|"
    r"ce (lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche))\b"
)
_SER = re.compile(r"\b(tous les|toutes les|chaque|la serie|toujours|definitivement)\b")
_NEGATIONS = {"pas", "non", "jamais"}
_ANNULER_PORTEE = re.compile(r"^(non|laisse|garde tout|ne change rien)\b")

_MOTIFS_OUI_NON = {"destructif", "creation_en_masse", "optimisation"}


def _ids_options(demande: dict) -> set:
    return {
        o.get("id") for o in demande.get("options") or []
        if isinstance(o, dict) and o.get("id")
    }


def _r1(message_brut: str, demande: dict, ids: set):
    cible = normaliser(message_brut)
    if not cible:
        return None
    for puce in demande.get("chips") or []:
        if not isinstance(puce, dict):
            continue
        option = puce.get("option")
        if option not in ids:
            continue
        for champ in ("value", "label"):
            if normaliser(puce.get(champ)) == cible:
                return option
    return None


# Revue du 2026-09-14: « oui pour jeudi seulement » confirmait la SERIE, parce
# que le reste ne contenait aucun verbe interdit. Le reste d'un oui ne peut
# plus porter que de la politesse: toute portee, tout jour, toute date ou tout
# autre mot fait reposer la question.
_RESTE_POLI = {
    "oui", "ok", "okay", "ouais", "yes", "je", "confirme", "confirmer", "vas-y", "vas", "y",
    "go", "c'est", "c", "est", "bon", "d'accord", "d", "accord", "daccord", "merci", "stp",
    "svp", "s'il", "te", "plait", "parfait", "super", "bien", "sur", "vraiment", "absolument",
    "tout", "a", "fait", "exactement", "allez", "fais-le", "fais", "le", "applique", "continue",
    "les", "ajouts", "plan", "la", "certain", "certaine", "sure",
}


def _reste_poli(reste: str) -> bool:
    mots = [m.strip("-'") for m in reste.split()]
    return all(m in _RESTE_POLI for m in mots if m)


def _r2(plat: str):
    if not plat or len(plat.split()) > 8:
        return None
    oui = _TETE_OUI.match(plat)
    if oui:
        reste = plat[oui.end():]
        if _RESTE_INTERDIT.search(reste) or not _reste_poli(reste):
            return None
        return "confirmer"
    if _TETE_NON.match(plat):
        return "annuler"
    return None


def _r3(plat: str):
    occ = _OCC.search(plat)
    series = list(_SER.finditer(plat))
    niee = False
    for m in series:
        avant = plat[:m.start()].split()[-3:]
        if _NEGATIONS.intersection(avant):
            niee = True
    if series and occ:
        # Les deux portees sont nommees (« pas tous les jeudis, juste
        # celui-la »): on repose la question plutot que de trancher.
        return None
    if series:
        return None if niee else "serie"
    if occ:
        return "occurrence"
    if _ANNULER_PORTEE.match(plat):
        return "annuler"
    return None


def option_choisie(message_brut: str, demande: dict) -> str | None:
    """L'option que CE message choisit pour CETTE demande, ou None.

    Jamais evaluee sur une liste: chaque demande ne connait que ses propres
    puces et ses propres options. Tout doute rend None, donc une question
    reposee, jamais une action.
    """
    if not isinstance(demande, dict) or not isinstance(message_brut, str):
        return None
    ids = _ids_options(demande)
    if not ids:
        return None
    choix = _r1(message_brut, demande, ids)
    if choix is None:
        motif = demande.get("motif")
        plat = _plat(message_brut)
        if motif in _MOTIFS_OUI_NON:
            choix = _r2(plat)
        elif motif == "portee_jour":
            choix = _r3(plat)
    return choix if choix in ids else None


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
    r"|(?<![-\w])(midi|minuit)\b"
)
# « pour » n'en fait pas partie: « Va pour 11 h 50 » est la valeur d'une puce.
_AVANT_DUREE = re.compile(r"(pendant|durant|dure|duree de)\s*$")
_APRES_DUREE = re.compile(r"^\s*(de\s|d'|par (jour|semaine|soir|seance)\b)")
# Une reponse de formulaire « Durée: 1 h » ou « Temps d'étude total: 4 h »
# donne une DUREE. Lue comme 01:00 ou 04:00, elle faisait retenir les ajouts
# du formulaire d'etude (banc du 2026-09-14, s02-2). Le libelle se lit sur la
# meme ligne, avant les deux-points.
_LIBELLE_DUREE = re.compile(
    r"(?:^|\n)[^\n:]*\b(dure\w*|temps|total\w*|combien|longueur|nombre d'heures|"
    r"heures par|volume)\b[^\n:]*:\s*$")


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
