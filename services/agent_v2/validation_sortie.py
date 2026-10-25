"""Retour au modèle des rejets déterministes de sa réponse finale."""

from services.agent_v2.mesure import verifier_prose
from services.agent_v2.redaction import ReponseDire, composer, contient_question, sans_tiret_long


_OUTILS_QUESTION = frozenset({
    "ask_question_custom", "present_form", "present_choices", "poser_question",
})


def question_sans_outil(texte: str, registre) -> bool:
    """Une question libre ne peut pas remplacer une demande interactive suivie."""
    if not contient_question(texte):
        return False
    return not any(
        (action.succes and action.outil in _OUTILS_QUESTION)
        or isinstance((action.donnees or {}).get("demande"), dict)
        for action in registre.actions
    )


def jugement_exige_outil(verdict: dict) -> bool:
    """Un verdict incertain ne transforme pas une relance sociale en formulaire."""
    return (verdict.get("statut") == "decision"
            and verdict.get("valeur") is True)


def _simple(texte: str) -> str:
    return " ".join((texte or "").split())


def retour_validation(texte: str, registre, *, faits_controle: str = "",
                      question_exige_outil: bool = False) -> str | None:
    """Décrit au modèle ce que le rendu retirerait, sans lui confier l'autorité.

    La même vérification reste exécutée après la boucle. Cette fonction ne
    prétend pas vérifier sémantiquement toute phrase libre : elle transforme
    seulement les rejets que le code sait déjà constater en retour exploitable.
    """
    if not isinstance(texte, str) or not texte.strip():
        return None

    if question_exige_outil and question_sans_outil(texte, registre):
        return (
            "VALIDATION DE LA RÉPONSE : tu demandes une précision en prose. "
            "Appelle ask_question_custom pour poser cette question et recueillir "
            "une réponse structurée. Si plusieurs valeurs manquent, utilise "
            "kind=form. Attends la réponse de l'utilisateur avant toute action "
            "qui dépend de ces valeurs."
        )

    brut = ReponseDire(ouverture=texte.strip())
    # Même contrat que le rendu final : sans action ni faits du code, la
    # réponse conversationnelle du modèle est le seul contenu disponible.
    propre, _, fuites = verifier_prose(
        brut, registre, sans_faits=bool(registre.vide() and not faits_controle))
    candidat = composer(propre, registre, faits_controle, None)
    texte_retenu = " ".join(p for p in (candidat.prose, candidat.question) if p)
    if _simple(texte_retenu) == _simple(sans_tiret_long(texte)):
        return None

    mutation = any(a.succes and a.est_mutation for a in registre.actions)
    if mutation:
        etats = []
        for action in registre.actions:
            if not action.est_mutation:
                continue
            statut = (action.recu or {}).get("statut") or (
                "echec" if not action.succes else "a_verifier")
            etats.append(f"{action.id}: {action.outil}, {statut}")
        bilan = "; ".join(etats[-8:]) or "aucun reçu confirmé"
        return (
            "VALIDATION DE LA RÉPONSE : le code aurait retiré ton texte sur un "
            "tour de modification. Ne présente aucun effet que le reçu relu "
            f"ne confirme. État des reçus : {bilan}. "
            "Le code affichera les effets confirmés. Garde seulement une "
            "question ou une offre utile sans affirmation d'effet, s'il y "
            "en a une. "
            "Tu peux appeler un outil de lecture si l'état réel manque, puis "
            "répondre à la demande. Ne répète pas la même écriture sans lecture."
        )

    raison = ", ".join(fuites[:3]) if fuites else "texte retiré au rendu"
    if faits_controle:
        raison = "lecture non demandée ou répétée dans la réponse"
    return (
        "VALIDATION DE LA RÉPONSE : une partie de ton texte serait supprimée "
        f"par le code ({raison}). Relis le résultat de l'outil, corrige ta "
        "réponse à la question précise, ou appelle un autre outil si une "
        "donnée manque. N'annonce pas une action qui n'a pas réussi."
    )
