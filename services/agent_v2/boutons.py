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

Deux sorties sur le meme calcul:
- `question_forcee` (2026-09-14) rend {question, chips, motif}: la question
  seule, sans les libelles, et des creneaux aux heures humaines (« 15 h a
  16 h »). C'est la forme que le narrateur place dans la section QUESTION.
- `boutons_forces` garde le contrat historique (texte complete, libelles v1).

Une question forcee que l'utilisateur a ignoree deux fois de suite (il a
repondu autre chose que ses boutons) n'est plus reposee: la reposer une
troisieme fois, c'est le harceler sur un sujet qu'il a laisse de cote.
"""
from __future__ import annotations

import re
import unicodedata

from services.agent_v2.registre import Registre
from services.agent_v2.rendu import date_courte, plage


MOTIF_FIN_RECURRENCE = "fin_recurrence"
MOTIF_CRENEAUX = "creneaux"

_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_PLAGE_V1 = re.compile(r"(\d{2}:\d{2})\s*[–-]\s*(\d{2}:\d{2})")
_HEURES_VALEUR_V1 = re.compile(r"de (\d{2}:\d{2}) à (\d{2}:\d{2})")
_TITRE_V1 = re.compile(r"«\s*(.+?)\s*»")


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


def _fin_de_recurrence(attachment):
    """(question, chips) si l'import a laisse des blocs sans date de fin, sinon None.

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
    if len(titles) == 1:
        label = titles[0]
        question = (
            f"« {label} » n'a pas de date de fin pour l'instant : "
            "jusqu'à quand veux-tu le garder à l'horaire ?"
        )
        chips = [
            {"label": "🏁 Je te donne la date de fin",
             "value": f"Je vais te donner la date de fin pour {label}."},
            {"label": "♾️ Pas de fin prévue",
             "value": f"{label} n'a pas de date de fin, garde-le tel quel."},
        ]
        return question, chips

    # Plusieurs blocs: v1 collait deux titres entre guillemets avec un verbe
    # au singulier (« Design d'interfaces et Economie globale » n'a pas de
    # date de fin... le garder). Vu sur un import de cinq cours le
    # 2026-09-01: faux en nombre et muet sur les trois autres.
    apercu = ", ".join(titles[:3]) + ("…" if len(titles) > 3 else "")
    question = (
        f"Tes {len(open_ended)} activités importées ({apercu}) n'ont pas de "
        "date de fin pour l'instant : jusqu'à quand veux-tu les garder à "
        "l'horaire ?"
    )
    chips = [
        {"label": "🏁 Je te donne la date de fin",
         "value": "Je vais te donner la date de fin pour ces activités importées."},
        {"label": "♾️ Pas de fin prévue",
         "value": "Ces activités importées n'ont pas de date de fin, garde-les telles quelles."},
    ]
    return question, chips


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


def _quand(iso: str) -> str:
    """« aujourd'hui », « demain » ou « le jeu. 24 sept. »."""
    court = date_courte(iso)
    return court if court in ("aujourd'hui", "demain", "hier") else f"le {court}"


def _chip_humaine(chip: dict) -> dict:
    """Un chip de creneau v1 (« 🕐 15:00–16:00 ») aux heures humaines.

    La valeur, qui s'affiche comme message de l'utilisateur au tap, perd elle
    aussi ses dates ISO et ses HH:MM. Un chip illisible reste tel quel.
    """
    label, valeur = chip.get("label", ""), chip.get("value", "")
    heures = _PLAGE_V1.search(label) or _HEURES_VALEUR_V1.search(valeur)
    jour = _ISO.search(valeur)
    if not heures or not jour:
        return dict(chip)
    lisible = plage(heures.group(1), heures.group(2))
    quand = _quand(jour.group(1))
    titre = _TITRE_V1.search(valeur)
    if titre:
        nouvelle = f"Planifie « {titre.group(1)} » {quand} de {lisible}."
    else:
        nouvelle = f"Va pour {lisible} {quand}."
    return {"label": lisible, "value": nouvelle}


def _question_creneaux(phrase_v1: str, chips_v1: list[dict]) -> str:
    jour = next((m.group(1) for m in (_ISO.search(c.get("value", "")) for c in chips_v1) if m),
                None)
    quand = f" {_quand(jour)}" if jour else ""
    if "pris" in phrase_v1:
        return f"Ce créneau est pris. Quel créneau libre te va{quand} ?"
    return f"Quel créneau libre te va{quand} ?"


def _normaliser_reponse(texte) -> str:
    return " ".join(str(texte or "").split()).casefold()


def _ignoree_deux_fois(user, motif: str) -> bool:
    """Les deux dernieres questions de ce motif sont-elles restees sans tap ?

    On lit les messages persistes: chaque message assistant suivi
    IMMEDIATEMENT d'un message utilisateur forme une paire. Sur les deux
    dernieres paires, si les deux assistants portaient ce question_motif et
    que la reponse n'etait aucune des valeurs de leurs boutons, la question
    a ete ignoree deux fois.
    """
    if user is None or getattr(user, "pk", None) is None:
        return False
    from core.models import ConversationMessage

    recents = list(ConversationMessage.objects.filter(user=user).order_by("-pk")[:12])
    recents.reverse()
    paires = [
        (message, recents[rang + 1])
        for rang, message in enumerate(recents[:-1])
        if message.role == "assistant" and recents[rang + 1].role == "user"
    ]
    if len(paires) < 2:
        return False
    for assistant, reponse in paires[-2:]:
        meta = assistant.metadata if isinstance(assistant.metadata, dict) else {}
        if meta.get("question_motif") != motif:
            return False
        valeurs = {
            _normaliser_reponse(chip.get("value"))
            for chip in meta.get("quick_replies") or []
            if isinstance(chip, dict)
        }
        if _normaliser_reponse(reponse.content) in valeurs:
            return False
    return True


def _calcul_force(user, message: str, attachment, registre: Registre,
                  attachment_traite_ce_tour: bool) -> dict | None:
    """Le calcul commun aux deux sorties.

    Priorite identique a v1: la fin de recurrence d'abord, les creneaux
    ensuite, jamais les deux. Une question ignoree deux fois se tait et
    laisse sa place a la suivante.
    """
    if attachment_traite_ce_tour and attachment is not None:
        fin = _fin_de_recurrence(attachment)
        if fin is not None and not _ignoree_deux_fois(user, MOTIF_FIN_RECURRENCE):
            question, chips = fin
            return {"motif": MOTIF_FIN_RECURRENCE, "question": question,
                    "chips": chips, "chips_v1": chips, "phrase_v1": None}

    if not _creneaux_envisageables(message, registre):
        return None

    from services.agent.agent import _ambiguous_scheduling_chips

    ambigu = _ambiguous_scheduling_chips(user, appels_outils(registre), message)
    if not ambigu:
        return None
    phrase, chips_v1 = ambigu
    if not chips_v1 or _ignoree_deux_fois(user, MOTIF_CRENEAUX):
        return None
    return {
        "motif": MOTIF_CRENEAUX,
        "question": _question_creneaux(phrase, chips_v1),
        "chips": [_chip_humaine(chip) for chip in chips_v1],
        "chips_v1": chips_v1,
        "phrase_v1": phrase,
    }


def question_forcee(user, message: str, attachment, registre: Registre,
                    attachment_traite_ce_tour: bool) -> dict | None:
    """{"question", "chips": [{"label", "value"}], "motif"} ou None.

    `message` est le message BRUT de l'utilisateur, pas sa version enrichie
    du contexte document (voir boutons_forces). La question ne contient pas
    les libelles des boutons: le narrateur les rend a part, et l'historique
    les retrouve dans les metadonnees du message.
    """
    calcul = _calcul_force(user, message, attachment, registre, attachment_traite_ce_tour)
    if calcul is None:
        return None
    return {
        "question": calcul["question"],
        "chips": [dict(chip) for chip in calcul["chips"]],
        "motif": calcul["motif"],
    }


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

    Contrat historique, construit sur le meme calcul que question_forcee: les
    creneaux sont ecrits dans le texte a la suite de la phrase, libelles v1
    tels que fournis par le helper, separes par ', '.
    """
    calcul = _calcul_force(user, message, attachment, registre, attachment_traite_ce_tour)
    if calcul is None:
        return texte, []
    if calcul["motif"] == MOTIF_FIN_RECURRENCE:
        if "jusqu" not in texte.lower():
            texte += f"\n\n⏳ {calcul['question']}"
        return texte, calcul["chips_v1"]
    libelles = ", ".join(chip["label"] for chip in calcul["chips_v1"])
    return f"{texte}\n\n{calcul['phrase_v1']} {libelles}", calcul["chips_v1"]
