"""
Les boutons que le CODE impose, quoi que raconte le modele.

Deux familles, heritees de v1 ou la regle de prompt seule etait une loterie
(1 tirage sur 3 omettait la question de fin de recurrence, vecu e2e):

1. Fin de recurrence: l'import de CE tour a cree des blocs sans end_date, le
   code pose « jusqu'a quand ? » avec ses deux reponses.
2. Planification ambigue: un schedule_task_at bloque par conflit, une
   consultation sans ecriture, ou une demande explicite sur une fenetre
   occupee -> 2-3 creneaux LIBRES en boutons. Le calcul est delegue au helper
   de v1 (_ambiguous_scheduling_chips); le registre est traduit dans le format
   qu'il attend plutot que de dupliquer sa logique. Deux gardes s'ajoutent
   AVANT lui (_creneaux_envisageables): une mutation reussie coupe tout, et
   une consultation sans intention de planifier reste une lecture.

Le frontend affiche done.quick_replies des qu'elle est non vide: c'est le seul
canal par lequel v2 peut GARANTIR un bouton.
"""
from __future__ import annotations

import unicodedata

from services.agent_v2.registre import Registre


def appels_outils(registre: Registre) -> list[dict]:
    """Le registre au format des helpers de v1: {tool, args, result}."""
    return [
        {
            "tool": a.outil,
            "args": a.parametres,
            "result": {"success": a.succes, "data": a.donnees},
        }
        for a in registre.actions
    ]


def _fin_de_recurrence(attachment, texte: str):
    """(texte, chips) si l'import a laisse des blocs sans date de fin, sinon None.

    Meme requete, meme phrase et memes deux chips que v1. Les chips sont
    forcees MEME si le texte pose deja la question: les suggestions du modele
    partent souvent sur autre chose (vecu: « Voir mon agenda ») et
    l'utilisateur se retrouve a taper ce qu'un tap aurait du regler.
    """
    from core.models import RecurringBlock

    open_ended = list(RecurringBlock.objects.filter(
        source_document=attachment, end_date__isnull=True))
    if not open_ended:
        return None
    titles = sorted({b.title for b in open_ended})
    label = titles[0] if len(titles) == 1 else " et ".join(titles[:2])
    if "jusqu" not in texte.lower():
        texte += (
            f"\n\n⏳ « {label} » n'a pas de date de fin pour l'instant : "
            "jusqu'à quand veux-tu le garder à l'horaire ?"
        )
    chips = [
        {"label": "🏁 Je te donne la date de fin",
         "value": f"Je vais te donner la date de fin pour {label}."},
        {"label": "♾️ Pas de fin prévue",
         "value": f"{label} n'a pas de date de fin, garde-le tel quel."},
    ]
    return texte, chips


def _intention_de_planifier(message: str) -> bool:
    """Le message BRUT porte-t-il un verbe de planification ?

    Meme regex et meme normalisation (accents retires, minuscules) que la
    troisieme jambe de v1: les deux lectures du message doivent concorder.
    """
    if not message:
        return False
    from services.agent.agent import _SCHED_INTENT_RE

    norm = unicodedata.normalize("NFKD", message).encode("ascii", "ignore").decode("ascii")
    return bool(_SCHED_INTENT_RE.search(norm.lower()))


def _creneaux_envisageables(message: str, registre: Registre) -> bool:
    """Faut-il seulement consulter le helper de v1 ? Deux gardes qu'il n'a pas.

    Un schedule_task_at bloque par conflit passe toujours: c'est la premiere
    jambe de v1, et le helper tranche seul (seul un schedule_task_at reussi
    ensuite l'annule; une autre mutation reussie ne compte pas pour lui).

    Sans conflit:
    - une mutation reussie ce tour coupe tout. OUTILS_DE_MUTATION de v2 est
      plus large que MUTATION_TOOLS de v1, qui ignore organize_day,
      optimize_week et cancel_scheduled_block: un organize_day reussi suivi
      d'un find_free_slots passait la deuxieme jambe de v1 et forcait des
      chips sur un tour qui avait deja ecrit;
    - une consultation reussie (find_free_slots) ne force des chips que si
      l'utilisateur voulait PLANIFIER (verbe dans le message brut) ou si une
      ecriture a ete tentee ce tour. « Quand suis-je libre demain ? » est une
      lecture: un tap y creerait un evenement que personne n'a demande.
    """
    actions = registre.actions
    if any(a.outil == "schedule_task_at" and not a.succes and a.donnees.get("conflict")
           for a in actions):
        return True
    if any(a.succes and a.est_mutation for a in actions):
        return False
    if any(a.outil == "find_free_slots" and a.succes for a in actions):
        tentee = any(a.outil == "schedule_task_at" for a in actions)
        return tentee or _intention_de_planifier(message)
    return True


def boutons_forces(user, message: str, attachment, registre: Registre,
                   texte: str, attachment_traite_ce_tour: bool) -> tuple[str, list[dict]]:
    """Rend (texte eventuellement complete, chips) ou (texte, []) si rien a forcer.

    `message` est le message BRUT de l'utilisateur, pas sa version enrichie du
    contexte document: la troisieme jambe de v1 y cherche un verbe de
    planification et deux heures, et un horaire importe en est plein.

    `attachment_traite_ce_tour` reproduit attachment_processed_this_turn de
    v1. Chez v1 le drapeau ne passe a True qu'a un seul endroit: dans la
    boucle d'attente, quand `attachment.processed` bascule de False a True
    apres un refresh_from_db. Il reste donc False si le document etait DEJA
    traite a l'arrivee du message (import d'un tour precedent) ou s'il ne
    finit pas dans la borne. `_contexte_document` de v2 execute la meme
    boucle sur le meme objet (refresh_from_db a chaque tic, sortie des que
    processed) et rien d'autre dans le tour ne recharge cet objet:
    _build_attachment_context lit ses attributs, _recent_import_context
    requete une autre instance. Une fois le generateur draine,
    `attachment.processed` vaut donc True exactement dans les cas ou il l'etait
    au depart ou a bascule pendant l'attente, et
    `not deja_traite and attachment.processed` egale le drapeau de v1, borne
    a zero comprise (boucle vide des deux cotes).

    Priorite identique a v1: la fin de recurrence d'abord, les creneaux
    ensuite, jamais les deux.

    Les creneaux sont ecrits dans le texte a la suite de la phrase, libelles
    tels que fournis par le helper, separes par ', '. Le texte rendu est a la
    fois affiche et persiste, et les chips ne le sont pas: sans les libelles,
    l'historique du tour suivant contiendrait une annonce tronquee et le
    modele ne saurait pas de quoi « le deuxieme » est la reponse.
    """
    if attachment_traite_ce_tour and attachment is not None:
        force = _fin_de_recurrence(attachment, texte)
        if force is not None:
            return force

    if not _creneaux_envisageables(message, registre):
        return texte, []

    from services.agent.agent import _ambiguous_scheduling_chips

    ambigu = _ambiguous_scheduling_chips(user, appels_outils(registre), message)
    if ambigu:
        phrase, chips = ambigu
        libelles = ", ".join(chip["label"] for chip in chips)
        return f"{texte}\n\n{phrase} {libelles}", chips
    return texte, []
