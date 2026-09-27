"""
Le registre d'un tour: ce qui s'est VRAIMENT passe.

Ecrit par le runtime a chaque execution d'outil, jamais par le modele. Chaque
entree porte un identifiant que la phase DIRE devra citer pour avoir le droit
de parler d'une action.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from services.agent.tools.base import ToolResult

# Les outils qui ECRIVENT. Sert a declencher la reconciliation et a distinguer
# lecture et mutation dans le bloc factuel. La liste de v1 (MUTATION_TOOLS)
# oublie organize_day, optimize_week et cancel_scheduled_block, qui ecrivent
# pourtant: on les inclut ici et un test verrouille la couverture.
OUTILS_DE_MUTATION = {
    "create_block", "update_block", "delete_block", "clear_all_blocks",
    "skip_block_occurrence", "restore_block_occurrence",
    "create_task", "update_task", "delete_task", "complete_task",
    "schedule_task_at", "cancel_scheduled_block",
    "optimize_week", "organize_day",
    "update_preferences", "create_goal", "update_goal",
    # Pas un outil du modele: l'import d'un document envoye, fait par le
    # systeme et inscrit par importation.py. C'est une ecriture en base
    # comme une autre, et le bloc factuel doit la raconter.
    "import_document",
}


# Les outils du modele autorises a tourner EN PARALLELE dans un meme lot
# d'appels (sequential=False). Audit du 2026-09-27, outil par outil:
# aucun n'ecrit en base dans son execute(), aucun ne mute l'etat du tour
# (etat.attente, etat.cache) dans _executer_appel: _analyser ne renvoie
# de garde que pour les outils destructifs, _garde_creations et
# _titre_du_formulaire rendent la main sauf pour les createurs (verrou
# du tour tenu), _abandon_cible_changee n'est joignable que sur les
# motifs portee_jour/destructif, _apres ne touche que schedule_task_at,
# create_block et update_block, et _consigner passe par le registre
# thread-safe et une file d'envoi thread-safe. send_notification a un
# effet externe (push) mais sans etat partage: le parallelisme ne change
# que le moment de l'envoi, pas sa semantique (aucune idempotence avant
# comme apres). present_form/present_choices rendent leur demande dans
# les donnees de l'action, consommees une par une au rendu.
#
# REGLE: tout nouvel outil du modele est SEQUENTIEL par defaut. On ne
# l'ajoute ici qu'apres audit de son execute() et de son passage dans
# _executer_appel. Un oubli coute du temps (lot sequentiel), jamais un
# bug de concurrence.
OUTILS_PARALLELES = frozenset({
    "list_blocks", "list_tasks",
    "get_today_schedule", "get_week_schedule", "find_free_slots",
    "check_feasibility", "detect_conflicts", "get_productivity_stats",
    "suggest_schedule_optimization",
    "get_preferences", "list_goals",
    "send_notification",
    "present_form", "present_choices",
})


@dataclass(frozen=True)
class Action:
    id: str
    outil: str
    parametres: dict
    succes: bool
    message: str
    donnees: dict

    @property
    def est_mutation(self) -> bool:
        return self.outil in OUTILS_DE_MUTATION


@dataclass(frozen=True)
class Ecart:
    """Un ecart entre l'intention et le resultat relu.

    `description` reste ecrite pour le MODELE (brief de DIRE). L'utilisateur,
    lui, lit une phrase construite par rendu.py depuis `genre` et `donnees`:
    « Ecart: CREE mais dans le passe » a atteint des utilisateurs dans la
    moitie des tours de production relus le 2026-09-13.

    Genres connus: passe, date_differente, tache_existante, plan_propose,
    preferences_inchangees, rien_a_restaurer. Un genre vide ne se rend pas.
    """
    id: str
    action_id: str
    description: str
    genre: str = ""
    donnees: dict = field(default_factory=dict)


# Deux appels identiques peuvent etre legitimes: relire apres avoir ecrit est
# un bon reflexe. Trois d'affilee ne le sont plus. Seuil choisi bas parce que
# la litterature de 2026 (taxonomie MAST, 1600+ traces) montre que les boucles
# d'actions identiques absorbent plus d'un quart des echecs sur certains bancs.
REPETITIONS_TOLEREES = 3


def _empreinte(outil: str, parametres: dict) -> str:
    """Identifie une action par son intention, pas par sa forme.

    Les cles sont triees: un dict n'a pas d'ordre stable d'un appel a l'autre,
    et sans normalisation le detecteur raterait la repetition la plus banale.
    """
    import json

    return f"{outil}:{json.dumps(parametres or {}, sort_keys=True, default=str)}"


def boucle_detectee(registre: "Registre") -> bool:
    """Le meme appel, a l'identique, REPETITIONS_TOLEREES fois de suite ?

    On regarde la fin du registre et non tout l'historique: trois creations de
    blocs differents ne sont pas une boucle, c'est un import d'horaire.
    """
    if len(registre.actions) < REPETITIONS_TOLEREES:
        return False
    derniers = registre.actions[-REPETITIONS_TOLEREES:]
    empreintes = {_empreinte(a.outil, a.parametres) for a in derniers}
    return len(empreintes) == 1


class Registre:
    def __init__(self) -> None:
        self.actions: list[Action] = []
        self.ecarts: list[Ecart] = []
        self.budget_epuise: bool = False
        self.delai_depasse: bool = False
        self.boucle_interrompue: bool = False
        self._index: dict = {}
        # Les outils de LECTURE peuvent desormais s'executer en parallele
        # (pydantic-ai dispatche les appels batchés via asyncio.create_task):
        # l'attribution des ids et l'indexation doivent rester atomiques.
        # Ordre des verrous: verrou du tour (outils.py) PUIS celui-ci,
        # jamais l'inverse.
        self._verrou = threading.Lock()

    def ajouter(self, outil: str, parametres: dict, resultat: ToolResult) -> Action:
        with self._verrou:
            action = Action(
                id=f"a{len(self.actions) + 1}",
                outil=outil,
                parametres=dict(parametres or {}),
                succes=bool(resultat.success),
                message=resultat.message or "",
                donnees=dict(resultat.data or {}),
            )
            self.actions.append(action)
            self._index[action.id] = action
            return action

    def ajouter_ecart(self, action_id: str, description: str,
                      genre: str = "", donnees: dict | None = None) -> Ecart:
        with self._verrou:
            ecart = Ecart(id=f"e{len(self.ecarts) + 1}",
                          action_id=action_id, description=description,
                          genre=genre or "", donnees=dict(donnees or {}))
            self.ecarts.append(ecart)
            self._index[ecart.id] = ecart
            return ecart

    def mutations(self) -> list[Action]:
        return [a for a in self.actions if a.est_mutation]

    def par_id(self, ident):
        if not ident or not isinstance(ident, str):
            return None
        return self._index.get(ident)

    def vide(self) -> bool:
        return not self.actions and not self.ecarts
