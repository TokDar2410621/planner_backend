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

MOTIFS_LECTURE_LIBRE = {"destructif", "creation_en_masse", "optimisation",
                        "portee_jour", "portee_changement"}
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


_NON = {"non", "no", "nan"}


def non_clair(message_brut) -> bool:
    """Miroir de oui_clair: le message n'est-il qu'un non, avec au plus de
    la politesse autour ?"""
    if not isinstance(message_brut, str) or "?" in message_brut:
        return False
    mots = [m for m in (x.strip("'-") for x in re.findall(r"[a-z0-9'-]+", _plat(message_brut))) if m]
    return (any(m in _NON for m in mots)
            and all(m in _NON or m in _POLITESSE for m in mots))


def _polarite_label(label) -> str | None:
    """La polarite d'un libelle de puce, ou None si elle n'est pas lisible
    sans ambiguite. Strict: seuls les libelles qui SONT un oui ou un non
    (avec au plus de la politesse) comptent."""
    plat = _plat(label).strip() if isinstance(label, str) else ""
    if not plat:
        return None
    mots = [m for m in (x.strip("'-") for x in re.findall(r"[a-z0-9'-]+", plat)) if m]
    if (any(m in _OUI for m in mots)
            and all(m in _OUI or m in _POLITESSE for m in mots)):
        return "oui"
    if (any(m in _NON for m in mots)
            and all(m in _NON or m in _POLITESSE for m in mots)):
        return "non"
    return None


def _oui_non_binaire(message_brut, demande):
    """Question libre binaire (exactement 2 options, une oui et une non):
    un oui/non clair en texte libre vaut la puce touchee.

    Exempte de la regle du round 6 (D1): les options d'une question libre
    ne portent aucun effet, un tap n'execute jamais d'outil. Le choix repart
    vers AGIR comme message (« CHOISI PAR L'UTILISATEUR »), jamais comme
    execution; le pire cas est une interpretation, pas une mutation."""
    options = [o for o in demande.get("options") or []
               if isinstance(o, dict) and o.get("id")]
    if len(options) != 2:
        return None
    polarites = {}
    for option in options:
        polarite = _polarite_label(option.get("libelle"))
        if polarite is None or polarite in polarites:
            return None
        polarites[polarite] = option.get("id")
    if set(polarites) != {"oui", "non"}:
        return None
    if oui_clair(message_brut):
        return polarites["oui"]
    if non_clair(message_brut):
        return polarites["non"]
    return None


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


# -- lire l'intention: la couche de jugement, pas des regex
#
# 2026-09-29 (Darius): jamais de vocabulaire ecrit a la main, de listes de
# declencheurs ou de regex pour detecter l'intention utilisateur afin de
# declencher ou restreindre une action. L'intention peut etre tout autre.
# Les anciennes regex (_MARQUE_GARDE, _VERBE_SUPPRESSION, _NOUVELLE_REQUETE,
# _LEXIQUE_SUPPRESSION, ...) sont supprimees. Le code pose des questions
# typees a services/agent_v2/jugement.py (Jev, repli LLM) et ne lit que des
# decisions typees. Les politiques restent dans le code: une option
# destructive ne provient que d'une puce exacte ou d'un postback; une
# reponse libre ne donne au mieux que « confirmer » (motifs non destructifs)
# ou « annuler »; incertain/indisponible repose la question, jamais d'action.

from services.agent_v2 import jugement as _jugement


def _etat_lecture(message_brut: str, demande: dict) -> dict:
    """L'etat passe au juge: le message brut, la question posee, son sujet."""
    cible = demande.get("cible") or {}
    morceaux = [str(cible.get("titre") or "")]
    jour = cible.get("jour")
    if isinstance(jour, int) and 0 <= jour <= 6:
        morceaux.append(_JOURS[jour])
    if cible.get("date"):
        morceaux.append(str(cible.get("date")))
    return {
        "message": message_brut,
        "question": (demande.get("question") or "").strip(),
        "sujet": " ".join(m for m in morceaux if m).strip(),
        "motif": demande.get("motif"),
    }


def _jugements(message_brut: str, demande: dict) -> dict:
    """Un seul appel au juge par (message, demande): l'intention, plus le
    choix libre quand le motif le demande (choix_modele, question_libre: des
    options sans effet). La portee d'une suppression ne se lit jamais en
    texte libre (D1): pas de question de portee ici. Le cache de
    jugement.juger rend les appels repetes gratuits dans le tour."""
    etat = _etat_lecture(message_brut, demande)
    questions = {"intention": _jugement.q_intention(
        etat["question"], etat["sujet"])}
    motif = demande.get("motif")
    if motif in ("choix_modele", "question_libre"):
        options = {}
        for opt in demande.get("options") or []:
            if not isinstance(opt, dict) or opt.get("id") == "annuler":
                continue
            oid = opt.get("id")
            options[oid] = str(opt.get("libelle") or opt.get("valeur")
                               or opt.get("value") or oid)
        if options:
            questions["choix"] = _jugement.q_choix(options)
    return _jugement.juger(etat, questions)


def _intention(message_brut: str, demande: dict) -> str:
    """Ce que le message fait face a la question en attente, selon le juge.

    'accepte' | 'refuse' | 'precise' | 'nouvelle_requete' | 'incertain'.
    Incertain couvre aussi l'indisponible: le juge n'a pas tranche.
    """
    res = _jugements(message_brut, demande).get("intention") or {}
    if res.get("statut") != _jugement.STATUT_DECISION:
        return "incertain"
    return res.get("valeur") or "incertain"


def _noul(message_brut, qid: str, fabrique, prudent: bool) -> tuple[bool, bool]:
    """Un jugement noul: (valeur, tranchee). Tranchee est Faux quand le juge
    est incertain ou indisponible; la valeur vaut alors le repli prudent."""
    res = _jugement.juger(
        {"message": message_brut}, {qid: fabrique()}).get(qid) or {}
    if res.get("statut") == _jugement.STATUT_DECISION:
        return bool(res.get("valeur")), True
    return prudent, False


def annulation_libre(message_brut, demande: dict) -> bool:
    """Le juge dit-il que le message refuse ce que la question propose ?

    Vrai ferme la demande sur « annuler »: rien ne s'execute. Tout doute rend
    Faux, donc aucune option. Pour « J'annule le dentiste ? », « annule »
    seul se lit « accepte » (un oui ambigu), jamais « refuse »: c'est
    l'intention qui desambigue, pas un mot banni.
    """
    if (not isinstance(message_brut, str) or "?" in message_brut
            or not isinstance(demande, dict)):
        return False
    # « J'annule le dentiste ? » « annule »: la c'est un oui. Ni l'un ni
    # l'autre ne se lit en texte libre.
    return (_intention(message_brut, demande) == "refuse"
            and "annuler" in _ids_options(demande)
            and demande.get("motif") in MOTIFS_LECTURE_LIBRE)


def _choix_juge(message_brut: str, demande: dict, ids: set) -> str | None:
    """L'option que le juge lit dans une reponse libre, ou None.

    Politique inchangee, seul le lecteur change (D1, round 6): une option
    qui detruit (confirmer une suppression, occurrence/serie d'une portee)
    ne provient que d'une puce exacte ou d'un postback, jamais d'ici. Une
    reponse libre ne donne au mieux que « confirmer » (motifs non
    destructifs), « annuler », ou le choix juge pour les motifs sans effet
    (choix_modele, question_libre). « J'annule le dentiste ? » + « annule »
    ne se lit pas en texte libre.
    """
    motif = demande.get("motif")
    intention = _intention(message_brut, demande)
    if demande.get("outil") == "cancel_scheduled_block" and intention == "accepte":
        # « annule » face a « J'annule le dentiste ? » est un oui ambigu,
        # pas un refus lisible. Un refus clair (« non, garde-le ») passe
        # par la voie « refuse » ci-dessous.
        return None
    if intention == "accepte":
        # Un oui libre ne confirme que ce qui ne detruit rien (I1/I3).
        if "confirmer" in ids and motif in MOTIFS_OUI_LIBRE:
            return "confirmer"
        return None
    if intention == "refuse":
        return "annuler" if annulation_libre(message_brut, demande) else None
    if intention == "precise" and motif in ("choix_modele", "question_libre"):
        res = _jugements(message_brut, demande).get("choix") or {}
        if (res.get("statut") == _jugement.STATUT_DECISION
                and res.get("valeur") in ids):
            return res["valeur"]
    return None


def classification_reponse(message_brut, demande: dict) -> str:
    """'reponse' | 'nouvelle_requete' | 'incertain'.

    Ne sert qu'a choisir entre reposer la question et l'abandonner (D2), et a
    dire si le tour peut se passer d'AGIR (D6). Une erreur ici coute une
    question reposee ou abandonnee, jamais une action.
    """
    if not isinstance(message_brut, str) or not isinstance(demande, dict):
        return "incertain"
    intention = _intention(message_brut, demande)
    if intention == "nouvelle_requete":
        return "nouvelle_requete"
    if intention == "incertain":
        return "incertain"
    return "reponse"


def option_choisie(message_brut: str, demande: dict,
                   tap: dict | None = None) -> str | None:
    """L'option que CE message choisit pour CETTE demande, ou None.

    Jamais evaluee sur une liste: chaque demande ne connait que ses propres
    puces. Seule la puce exacte donne une option destructive; une reponse
    libre ne donne au mieux que « annuler » (ou « confirmer » pour les motifs
    non destructifs), lue par le juge.

    Exception: pour le motif « question_libre » (options sans effet), une
    question binaire oui/non accepte aussi un oui/non clair en texte libre.

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
    if choix is None:
        choix = _choix_juge(message_brut, demande, ids)
    if choix is None and demande.get("motif") == "question_libre":
        choix = _oui_non_binaire(message_brut, demande)
    return choix if choix in ids else None


# -- reposer ou abandonner (D2): cette lecture n'autorise JAMAIS rien


def reponse_plausible(message_brut, demande: dict) -> bool:
    """Le message peut-il etre une reponse, meme floue, a CETTE demande ?

    Ne sert qu'a choisir entre reposer la question et l'abandonner (D2), et a
    dire si le tour peut se passer d'AGIR (D6). Une erreur ici coute une
    question reposee ou abandonnee, jamais une action. Seule une nouvelle
    requete tranchee par le juge rend Faux; un doute repose la question.
    """
    return classification_reponse(message_brut, demande) != "nouvelle_requete"


# ------------------------------------------------------------- jours, heures


def jour_vise(message_brut) -> bool:
    """Le message vise-t-il un jour precis (nom, demain, une date) ?

    Juge par noul. En cas de doute (incertain/indisponible), rend Vrai: le
    code pose la question de portee plutot que de supposer toute la serie.
    """
    valeur, _ = _noul(message_brut, "jour_vise", _jugement.q_jour_vise, True)
    return valeur


def jour_vise_tranche(message_brut) -> tuple[bool, bool]:
    """(vise_un_jour, tranche). Tranche est Faux si le juge n'a pas decide."""
    return _noul(message_brut, "jour_vise", _jugement.q_jour_vise, True)


def suppression_demandee(message_brut) -> bool:
    """Le message demande-t-il de SUPPRIMER quelque chose ?

    Juge par noul. En cas de doute (incertain/indisponible), rend Vrai: un
    doute se traite comme une suppression possible, jamais comme un feu vert.
    """
    valeur, _ = _noul(message_brut, "suppression", _jugement.q_suppression,
                      True)
    return valeur


def suppression_tranchee(message_brut) -> tuple[bool, bool]:
    """(suppression_visee, tranche). Tranche est Faux si le juge n'a pas
    decide: pour les comparateurs, qui ne comparent que du tranche."""
    return _noul(message_brut, "suppression", _jugement.q_suppression, True)


def serie_entiere_visee(message_brut) -> bool:
    """La personne vise-t-elle clairement TOUTE la serie ?

    Vrai seulement sur une decision nette du juge : c'est ce qui DISPENSE de
    poser la question de portee. Juge muet ou incertain, on pose la question.
    """
    message = str(message_brut or "").strip()
    if not message:
        return False
    rep = _jugement.juger(
        message,
        {"portee": _jugement.q_portee_changement()}).get("portee") or {}
    return (rep.get("statut") == _jugement.STATUT_DECISION
            and rep.get("valeur") == "serie"
            and float(rep.get("confiance") or 0) >= 0.9)


def saut_suspect(message_brut) -> bool:
    """Un saut d'occurrence qui ressemble a une suppression large (« efface
    tout jeudi »). Le saut unique explicite (« saute mon gym demain ») passe.

    Juge par choice. En cas de doute (incertain/indisponible), rend Vrai: le
    code pose la question de portee plutot que de laisser passer un saut
    ambigu.
    """
    res = _jugement.juger(
        {"message": message_brut},
        {"saut": _jugement.q_saut_ou_suppression()}).get("saut") or {}
    if res.get("statut") == _jugement.STATUT_DECISION:
        return res.get("valeur") == "suppression_large"
    return True


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
