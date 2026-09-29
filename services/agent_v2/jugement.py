"""
La couche de jugement : des decisions typees au lieu des regex d'intention.

Principe (2026-09-29, Darius) : jamais de vocabulaire ecrit a la main, de
listes de declencheurs ou de regex pour detecter l'intention utilisateur afin
de declencher ou restreindre une action. L'intention peut etre tout autre.
Le code pose donc des QUESTIONS typees (choice/score/noul) sur l'etat du tour,
et un modele de decision y repond. Le code ne lit plus les mots, il lit des
decisions.

Fournisseur principal : Jev (TypeSafe AI), modele de decision non
autoregressif. POST {JEV_API_URL} (defaut
https://api.typesafe.ai/v1/systemone, d'apres la doc officielle
docs.typesafe.ai),
Bearer JEV_API_KEY, body {model, state, questions}. Reponse
{model, answers: {qid: {choice|score|noul, confidence, probabilities}}, usage}.

Repli sans cle : un LLM en sortie structuree (meme pattern que lecture.py
LIRE, pydantic-ai, output_type). Si ca echoue aussi : indisponible.

Regle d'or : un echec de validation ou un timeout rend INDISPONIBLE, jamais
une approbation par defaut. Sur une action destructive, un jugement incertain
ou indisponible impose le comportement prudent (demander), jamais une
execution silencieuse. C'est l'appelant qui applique cette politique ; ici on
ne fait que dire ce qu'on sait, avec un niveau de confiance.
"""
from __future__ import annotations

import hashlib
import json
import os
import logging
from typing import Any, Optional

from django.conf import settings

logger = logging.getLogger(__name__)

# -- types de questions ----------------------------------------------------
TYPE_NOUL = "noul"
TYPE_CHOICE = "choice"
TYPE_SCORE = "score"
TYPES_QUESTION = frozenset({TYPE_NOUL, TYPE_CHOICE, TYPE_SCORE})

# -- statuts d'une reponse ---------------------------------------------------
STATUT_DECISION = "decision"       # confiance >= seuil : le code peut trancher
STATUT_INCERTAIN = "incertain"     # reponse valide mais confiance trop basse
STATUT_INDISPONIBLE = "indisponible"  # Jev puis le repli ont echoue


def _cle_jev() -> str:
    # Convention officielle des SDK TypeSafe: TYPESAFE_API_KEY. JEV_API_KEY
    # reste le nom historique du projet, prioritaire s'il est defini.
    return (getattr(settings, "JEV_API_KEY", "")
            or os.environ.get("TYPESAFE_API_KEY", "") or "")


def _url_jev() -> str:
    # Endpoint officiel verifie sur docs.typesafe.ai (quickstart + API
    # reference). thejevai.com est un tiers non affilie: ne pas l'utiliser.
    return getattr(settings, "JEV_API_URL",
                   "https://api.typesafe.ai/v1/systemone") or \
        "https://api.typesafe.ai/v1/systemone"


def _modele_jev() -> str:
    return getattr(settings, "JEV_MODEL", "jev-latest") or "jev-latest"


def _delai_jev() -> float:
    try:
        return max(0.5, float(getattr(settings, "JEV_TIMEOUT", 2.5)))
    except (TypeError, ValueError):
        return 2.5


def _seuil_decision() -> float:
    """Confiance minimale pour qu'une reponse devienne une decision."""
    try:
        seuil = float(getattr(settings, "JUGEMENT_SEUIL_DECISION", 0.8))
    except (TypeError, ValueError):
        return 0.8
    return min(1.0, max(0.0, seuil))


def _repli_llm_actif() -> bool:
    return str(getattr(settings, "JUGEMENT_REPLI_LLM", "1")).lower() not in (
        "0", "false", "non")


# -- questions pretes a l'emploi ----------------------------------------------
#
# Les instructions sont en francais (le repo est francophone, l'etat juge est
# du francais quebecois). Si le banc montre que Jev juge moins bien le
# francais que l'anglais, ces instructions passeront a l'anglais ; l'etat
# restera en francais.

def q_suppression() -> dict:
    """Noul : le message demande-t-il de SUPPRIMER quelque chose ?"""
    return {
        "type": TYPE_NOUL,
        "instructions": (
            "Ce message demande-t-il de SUPPRIMER quelque chose (un cours, "
            "un bloc, une tache, un evenement, des donnees) ? Reponds oui "
            "seulement si la suppression est l'action demandee par le message."
        ),
        "criteria": {
            "oui": "le message demande une suppression",
            "non": "le message ne demande pas de suppression",
        },
    }


def q_jour_vise() -> dict:
    """Noul : le message vise-t-il un jour precis pour son action ?"""
    return {
        "type": TYPE_NOUL,
        "instructions": (
            "Le message vise-t-il un jour precis (nom de jour, demain, "
            "apres-demain, une date) pour l'action qu'il demande ? Reponds "
            "oui seulement si un jour est nomme ou clairement designe."
        ),
        "criteria": {
            "oui": "un jour precis est vise",
            "non": "aucun jour precis n'est vise",
        },
    }


def q_intention(question_posee: str, cible: str = "") -> dict:
    """Choice : que fait ce message face a la question en attente ?

    Remplace les regex _MARQUE_GARDE, _VERBE_SUPPRESSION, _NOUVELLE_REQUETE,
    _NOUVELLE_DESTRUCTION, _CHANGE_D_AVIS, _PORTEE_NIEE, _OCCURRENCE,
    _TOMBER_AVEC_OBJET et les listes _OUI/_NON/_POLITESSE.
    """
    precision = f' (au sujet de : {cible})' if cible else ""
    return {
        "type": TYPE_CHOICE,
        "instructions": (
            f"Une question a ete posee a l'utilisateur : "
            f"\u00ab {question_posee} \u00bb{precision}. "
            "Que fait son message en reponse ? Ne devine pas au-dela de ce "
            "qui est ecrit : en cas de doute, choisis incertain."
        ),
        "options": {
            "confirme": "il confirme, dit oui, accepte",
            "annule": ("il annule, garde, laisse tomber : il ne veut rien "
                       "changer"),
            "precise": ("il repond a la question : il choisit une option ou "
                        "donne la precision demandee"),
            "nouvelle_requete": ("il fait une nouvelle demande sans rapport "
                                 "avec la question posee"),
            "incertain": "impossible a trancher",
        },
    }


def q_portee() -> dict:
    """Choice : quelle portee l'utilisateur choisit-il pour la suppression ?"""
    return {
        "type": TYPE_CHOICE,
        "instructions": (
            "Pour la suppression en question, quelle portee l'utilisateur "
            "choisit-il ? En cas de doute, choisis incertain."
        ),
        "options": {
            "occurrence": "cette occurrence seulement (une fois, ce jour-la)",
            "serie": "toute la serie (toujours, chaque semaine)",
            "annuler": "annuler : ne rien supprimer",
            "incertain": "impossible a trancher",
        },
    }


def q_saut_ou_suppression() -> dict:
    """Choice : le message veut-il sauter une occurrence, ou supprimer large ?

    Remplace _saut_suspect (outils.py) et ses regex _TOUT/_UNE_SEULE_FOIS.
    """
    return {
        "type": TYPE_CHOICE,
        "instructions": (
            "L'utilisateur parle de sauter ou d'enlever quelque chose un jour "
            "donne. S'agit-il vraiment d'un saut ponctuel (une seule "
            "occurrence, un jour precis), ou veut-il en fait supprimer "
            "largement (tout, tous les jours, toute la serie) ? En cas de "
            "doute, choisis incertain."
        ),
        "options": {
            "saut_ponctuel": "sauter une seule occurrence, un jour precis",
            "suppression_large": "supprimer largement : tout, tous, la serie",
            "incertain": "impossible a trancher",
        },
    }


def q_choix(options: dict[str, str]) -> dict:
    """Choice : quelle option l'utilisateur choisit-il, parmi celles-ci ?

    `options` : {id: libelle}. Pour les questions libres (present_choices)
    dont les options ne portent aucun effet.
    """
    return {
        "type": TYPE_CHOICE,
        "instructions": (
            "Des options ont ete proposees a l'utilisateur. Laquelle "
            "choisit-il dans son message ? En cas de doute, choisis incertain."
        ),
        "options": {cle: libelle for cle, libelle in options.items()}
        | {"incertain": "impossible a trancher"},
    }


# -- appel Jev -----------------------------------------------------------------

class _ReponseInvalide(Exception):
    """La reponse Jev ne respecte pas le contrat : on la jette entiere."""


def _question_jev(question: dict) -> dict:
    """La question au format fil de Jev."""
    qtype = question.get("type")
    if qtype not in TYPES_QUESTION:
        raise _ReponseInvalide(f"type de question inconnu: {qtype!r}")
    corps = {"type": qtype, "instructions": question.get("instructions") or ""}
    if qtype == TYPE_CHOICE:
        options = question.get("options") or {}
        if not isinstance(options, dict) or not (2 <= len(options) <= 255):
            raise _ReponseInvalide("choice: 2 a 255 options requises")
        corps["criteria"] = {k: str(v) for k, v in options.items()}
    elif qtype == TYPE_SCORE:
        niveaux = question.get("niveaux") or []
        if not isinstance(niveaux, list) or not (2 <= len(niveaux) <= 10):
            raise _ReponseInvalide("score: 2 a 10 niveaux requis")
        corps["criteria"] = [str(n) for n in niveaux]
    else:  # noul
        criteria = question.get("criteria") or {}
        if criteria:
            if not isinstance(criteria, dict):
                raise _ReponseInvalide("noul: criteria doit etre un objet")
            corps["criteria"] = {k: str(v) for k, v in criteria.items()}
    return corps


def _est_nombre(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _valider_reponse(donnees: Any, questions: dict) -> dict:
    """Validation stricte au bord reseau. Tout ecart -> _ReponseInvalide.

    Rend {qid: {"valeur", "confiance", "probabilites"}} sans statut : le
    statut vient de la politique de confiance, plus bas.
    """
    if not isinstance(donnees, dict) or not isinstance(
            donnees.get("answers"), dict):
        raise _ReponseInvalide("reponse sans objet 'answers'")
    reponses = donnees["answers"]
    sortie = {}
    for qid, question in questions.items():
        if qid not in reponses or not isinstance(reponses[qid], dict):
            raise _ReponseInvalide(f"reponse manquante pour {qid!r}")
        rep = reponses[qid]
        qtype = question["type"]
        if qtype == TYPE_CHOICE:
            options = question["options"]
            choix = rep.get("choice")
            if choix not in options:
                raise _ReponseInvalide(
                    f"{qid}: choix {choix!r} hors des options declarees")
            probas = rep.get("probabilities") or {}
            if not isinstance(probas, dict):
                raise _ReponseInvalide(f"{qid}: probabilities invalide")
            for k, p in probas.items():
                if k not in options or not _est_nombre(p) or not 0.0 <= p <= 1.0:
                    raise _ReponseInvalide(f"{qid}: probabilite invalide {k!r}")
            confiance = rep.get("confidence")
            if not _est_nombre(confiance) or not 0.0 <= confiance <= 1.0:
                raise _ReponseInvalide(f"{qid}: confidence invalide")
            sortie[qid] = {"valeur": choix, "confiance": float(confiance),
                           "probabilites": {k: float(v) for k, v in probas.items()}}
        elif qtype == TYPE_NOUL:
            p = rep.get("noul")
            if not _est_nombre(p) or not 0.0 <= p <= 1.0:
                raise _ReponseInvalide(f"{qid}: noul invalide")
            p = float(p)
            confiance = rep.get("confidence")
            if _est_nombre(confiance) and 0.0 <= confiance <= 1.0:
                conf = float(confiance)
            else:
                conf = max(p, 1.0 - p)
            sortie[qid] = {"valeur": p >= 0.5, "confiance": conf,
                           "probabilites": {"oui": p, "non": 1.0 - p}}
        else:  # score
            niveaux = question["niveaux"]
            score = rep.get("score")
            if not _est_nombre(score) or not 0.0 <= score <= len(niveaux) - 1:
                raise _ReponseInvalide(f"{qid}: score invalide")
            confiance = rep.get("confidence")
            if not _est_nombre(confiance) or not 0.0 <= confiance <= 1.0:
                raise _ReponseInvalide(f"{qid}: confidence invalide")
            sortie[qid] = {"valeur": float(score), "confiance": float(confiance),
                           "probabilites": rep.get("probabilities")
                           if isinstance(rep.get("probabilities"), dict) else None}
    return sortie


def _appeler_jev(etat: dict, questions: dict) -> dict:
    """Un appel Jev, valide strictement. Leve en cas de probleme."""
    import requests

    corps = {
        "model": _modele_jev(),
        "state": etat,
        "questions": {qid: _question_jev(q) for qid, q in questions.items()},
    }
    # La cle ne sort que dans l'en-tete Authorization, jamais dans les logs.
    reponse = requests.post(
        _url_jev(), json=corps,
        headers={"Authorization": f"Bearer {_cle_jev()}",
                 "Content-Type": "application/json"},
        timeout=_delai_jev(),
    )
    reponse.raise_for_status()
    return _valider_reponse(reponse.json(), questions)


# -- repli LLM ------------------------------------------------------------------
#
# Meme pattern que lecture.py LIRE : pydantic-ai, sortie structuree,
# output_retries=0 (une sortie invalide rend indisponible, sans second appel
# paye), jamais d'historique.

def _questions_texte(questions: dict) -> str:
    morceaux = []
    for qid, q in questions.items():
        morceaux.append(f"[{qid}] ({q['type']}) {q.get('instructions') or ''}")
        if q["type"] == TYPE_CHOICE:
            for cle, desc in (q.get("options") or {}).items():
                morceaux.append(f"  - {cle}: {desc}")
        elif q["type"] == TYPE_SCORE:
            for i, niv in enumerate(q.get("niveaux") or []):
                morceaux.append(f"  - {i}: {niv}")
        elif q.get("criteria"):
            morceaux.append(f"  criteres: {q['criteria']}")
    return "\n".join(morceaux)


_PROMPT_REPLI = """Tu es un juge d'intention, pas un assistant conversationnel.
On te donne un ETAT (un message d'utilisateur et son contexte) et des
QUESTIONS typees. Pour chaque question, rends :
- valeur : la cle de l'option choisie (choice), "oui"/"non" (noul), ou le
  numero du niveau (score, en texte) ;
- confiance : un nombre entre 0 et 1 (0.5 = pile ou face, 1 = certain).
Ne rends QUE du JSON valide, rien d'autre. En cas de doute, confiance basse.
"""


def _juger_llm(etat: dict, questions: dict) -> Optional[dict]:
    """Le repli sans cle Jev. Rend None si indisponible."""
    from pydantic import BaseModel, Field

    from services.agent_v2.modeles import modele_agir

    class _Une(BaseModel):
        valeur: str
        confiance: float = Field(ge=0.0, le=1.0)

    class _Reponses(BaseModel):
        reponses: dict[str, _Une]

    try:
        modele = modele_agir()
    except RuntimeError:
        logger.warning("jugement: aucun fournisseur LLM pour le repli")
        return None

    from pydantic_ai import Agent
    agent = Agent(modele, output_type=_Reponses, output_retries=0)
    etat_texte = json.dumps(etat, ensure_ascii=False, sort_keys=True,
                            default=str)
    try:
        resultat = agent.run_sync(
            f"ETAT:\n{etat_texte}\n\nQUESTIONS:\n{_questions_texte(questions)}",
            instructions=_PROMPT_REPLI,
        )
    except Exception as e:  # noqa: BLE001 - le repli ne casse jamais un tour
        logger.warning("jugement: repli LLM echoue (%s)", type(e).__name__)
        return None

    sortie = {}
    try:
        donnees = resultat.output
        for qid, question in questions.items():
            une = donnees.reponses.get(qid)
            if une is None:
                raise _ReponseInvalide(f"reponse manquante pour {qid!r}")
            qtype = question["type"]
            valeur = une.valeur
            if qtype == TYPE_CHOICE:
                if valeur not in (question.get("options") or {}):
                    raise _ReponseInvalide(
                        f"{qid}: choix {valeur!r} hors options")
            elif qtype == TYPE_NOUL:
                if valeur == "oui":
                    valeur = True
                elif valeur == "non":
                    valeur = False
                else:
                    raise _ReponseInvalide(f"{qid}: noul attend oui/non")
            else:  # score
                try:
                    niveau = float(valeur)
                except (TypeError, ValueError):
                    raise _ReponseInvalide(f"{qid}: score non numerique")
                if not 0.0 <= niveau <= len(question["niveaux"]) - 1:
                    raise _ReponseInvalide(f"{qid}: score hors echelle")
                valeur = niveau
            sortie[qid] = {"valeur": valeur,
                           "confiance": float(une.confiance),
                           "probabilites": None}
    except _ReponseInvalide as e:
        logger.warning("jugement: repli LLM invalide (%s)", e)
        return None
    return sortie


# -- politique de confiance ------------------------------------------------------

def _appliquer_seuil(brutes: dict) -> dict:
    """Decision si confiance >= seuil, incertain sinon."""
    seuil = _seuil_decision()
    sortie = {}
    for qid, rep in brutes.items():
        statut = (STATUT_DECISION if rep["confiance"] >= seuil
                  else STATUT_INCERTAIN)
        sortie[qid] = {**rep, "statut": statut}
    return sortie


def _indisponibles(questions: dict) -> dict:
    return {qid: {"valeur": None, "confiance": 0.0, "probabilites": None,
                  "statut": STATUT_INDISPONIBLE}
            for qid in questions}


# -- cache -------------------------------------------------------------------------
#
# Un jugement est une fonction pure de (etat, questions) : on le memoise par
# tour et entre les tours. Borne, vidangeable pour les tests.

_CACHE: dict[str, dict] = {}
_CACHE_MAX = 512


def _cle_cache(etat: dict, questions: dict) -> str:
    canonique = json.dumps({"etat": etat, "questions": questions},
                           ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(canonique.encode("utf-8")).hexdigest()


def vider_cache() -> None:
    _CACHE.clear()


# -- entree publique -----------------------------------------------------------------

def juger(etat: dict | str, questions: dict) -> dict:
    """Juge l'etat avec les questions typees.

    `etat` : dict (contexte JSON) ou str (message brut, mis sous "message").
    `questions` : {qid: {"type", "instructions", "options"/"niveaux"/"criteria"}}.
    Rend {qid: {"valeur", "confiance", "probabilites", "statut"}}, statut dans
    {"decision", "incertain", "indisponible"}.

    Ne leve jamais : tout echec rend indisponible. Un indisponible n'est
    JAMAIS une approbation : l'appelant applique le comportement prudent.
    """
    if isinstance(etat, str):
        etat = {"message": etat}
    if not isinstance(questions, dict) or not questions:
        return {}
    try:
        for qid, q in questions.items():
            if not isinstance(q, dict) or q.get("type") not in TYPES_QUESTION:
                raise _ReponseInvalide(f"question {qid!r} mal formee")
        cle = _cle_cache(etat, questions)
        if cle in _CACHE:
            return _CACHE[cle]
        brutes = _juger_sans_cache(etat, questions)
        resultats = _appliquer_seuil(brutes)
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[cle] = resultats
        return resultats
    except Exception as e:  # noqa: BLE001 - juger ne casse jamais un tour
        logger.warning("jugement: echec (%s), indisponible", type(e).__name__)
        return _indisponibles(questions)


def _juger_sans_cache(etat: dict, questions: dict) -> dict:
    """Jev si la cle est configuree, sinon le repli LLM. Leve si indisponible."""
    if _cle_jev():
        try:
            return _appeler_jev(etat, questions)
        except _ReponseInvalide as e:
            logger.warning("jugement: reponse Jev invalide (%s)", e)
        except Exception as e:  # noqa: BLE001 - timeout, 429, reseau...
            logger.warning("jugement: Jev injoignable (%s)", type(e).__name__)
    if _repli_llm_actif():
        repli = _juger_llm(etat, questions)
        if repli is not None:
            return repli
    raise _ReponseInvalide("aucun juge disponible")
