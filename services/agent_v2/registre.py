"""
Le registre d'un tour: ce qui s'est VRAIMENT passe.

Ecrit par le runtime a chaque execution d'outil, jamais par le modele. Chaque
entree porte un identifiant que la phase DIRE devra citer pour avoir le droit
de parler d'une action.
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field, replace
from typing import Literal, TypedDict

from services.agent.tools.base import ToolResult

# Les outils qui ECRIVENT. Sert a declencher la reconciliation et a distinguer
# lecture et mutation dans le bloc factuel. La liste de v1 (MUTATION_TOOLS)
# oublie organize_day, optimize_week et cancel_scheduled_block, qui ecrivent
# pourtant: on les inclut ici et un test verrouille la couverture.
OUTILS_DE_MUTATION = {
    "create_block", "update_block", "delete_block", "clear_all_blocks",
    "skip_block_occurrence", "restore_block_occurrence",
    "replace_block_occurrence",
    "create_task", "update_task", "delete_task", "complete_task",
    "schedule_task_at", "cancel_scheduled_block",
    "optimize_week", "organize_day",
    "update_preferences", "create_goal", "update_goal",
    # Pas un outil du modele: l'import d'un document envoye, fait par le
    # systeme et inscrit par importation.py. C'est une ecriture en base
    # comme une autre, et le bloc factuel doit la raconter.
    "import_document",
    "resolve_document_entry",
    "manage_reminder", "manage_recurrence",
}


@dataclass(frozen=True)
class Postcondition:
    """Comparaison exécutée par le runtime sur une cible de l'utilisateur."""

    entite: str
    cible: dict
    attendu: dict
    observe: dict | None
    conforme: bool


class RecuMutation(TypedDict, total=False):
    cle_operation: str
    statut: Literal["a_verifier", "pending_confirmation", "confirme", "sans_mutation", "contradictoire", "echec"]
    user_id: int
    source: Literal["base"]
    postconditions: list[dict]


@dataclass(frozen=True)
class Action:
    id: str
    outil: str
    parametres: dict
    succes: bool
    message: str
    donnees: dict
    # Contrat d'issue metier (audit 2026-10-13, ref OpenCode Tool.Result):
    # l'outil ecrit son issue, sa cause et son explication affichable; le
    # registre les transporte jusqu'au rendu et au journal, sans les
    # reinterpreter. Chaines vides = outil non migre, comportement inchange.
    issue: str = ""
    cause: str = ""
    public: str = ""
    # Duree d'execution de l'outil en millisecondes (journal, audit
    # 2026-10-13). 0.0 = non mesuree (chemins par le code, refus de garde).
    duree_ms: float = 0.0
    # Recu d'idempotence (boucle unique, 2026-09-29). `cle_operation` identifie
    # l'intention metier (tache + outil + empreinte des parametres); `recu`
    # porte le statut de confirmation. Une mutation dont le resultat est
    # ambigu (timeout, exception transitoire) n'est ni un succes ni un echec:
    # elle est `pending_confirmation` et doit etre reconciliee par requete
    # avant toute reemission. Le modele ne reexecute jamais a l'aveugle.
    cle_operation: str = ""
    recu: RecuMutation = field(default_factory=dict)

    @property
    def est_mutation(self) -> bool:
        return self.outil in OUTILS_DE_MUTATION

    @property
    def en_attente_confirmation(self) -> bool:
        return bool((self.donnees or {}).get("pending_confirmation"))

    @property
    def etat_verifie(self) -> bool:
        """Un résultat déclaré réussi ne constitue pas une relecture."""
        preuves = self.recu.get("postconditions", [])
        return bool(self.succes and self.recu.get("source") == "base"
                    and self.recu.get("user_id") is not None
                    and self.recu.get("statut") in ("confirme", "sans_mutation")
                    and preuves and all(p.get("conforme") is True for p in preuves))


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
        self.motif_budget: str = "etapes"
        self.delai_depasse: bool = False
        self.boucle_interrompue: bool = False
        self._index: dict = {}
        # Les outils de LECTURE peuvent desormais s'executer en parallele
        # (pydantic-ai dispatche les appels batchés via asyncio.create_task):
        # l'attribution des ids et l'indexation doivent rester atomiques.
        # Ordre des verrous: verrou du tour (outils.py) PUIS celui-ci,
        # jamais l'inverse.
        self._verrou = threading.Lock()

    def ajouter(self, outil: str, parametres: dict, resultat: ToolResult,
                cle_operation: str = "", duree_ms: float = 0.0) -> Action:
        with self._verrou:
            donnees = dict(resultat.data or {})
            recu = {}
            if outil in OUTILS_DE_MUTATION:
                # L'exécution ouvre une demande de vérification. Seule la
                # réconciliation peut y joindre les postconditions relues.
                recu = {
                    "cle_operation": cle_operation or "",
                    "statut": ("pending_confirmation"
                               if donnees.get("pending_confirmation")
                               else ("a_verifier" if resultat.success else "echec")),
                }
                objet_id = donnees.get("block_id") or donnees.get("task_id")
                if objet_id is not None:
                    recu["objet_id"] = objet_id
            action = Action(
                id=f"a{len(self.actions) + 1}",
                outil=outil,
                parametres=dict(parametres or {}),
                succes=bool(resultat.success),
                message=resultat.message or "",
                donnees=donnees,
                cle_operation=cle_operation or "",
                recu=recu,
                issue=getattr(resultat, "issue", "") or "",
                cause=getattr(resultat, "cause", "") or "",
                public=getattr(resultat, "public", "") or "",
                duree_ms=float(duree_ms or 0.0),
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

    def recus_confirmes(self) -> list[Action]:
        """Les ecritures prouvees: succes + recu confirme. Seule source de
        verite de la phase de rendu."""
        return [a for a in self.actions
                if a.est_mutation and a.etat_verifie
                and a.recu.get("statut") == "confirme"]

    def verifier(self, action_id: str, user_id: int,
                 preuves: list[Postcondition], *, sans_mutation: bool = False) -> None:
        """Consigne les seules preuves structurées issues de la relecture."""
        with self._verrou:
            action = self._index.get(action_id)
            if not isinstance(action, Action) or not action.est_mutation:
                return
            statut = "a_verifier"
            if preuves:
                statut = ("sans_mutation" if sans_mutation else "confirme") \
                    if all(p.conforme for p in preuves) else "contradictoire"
            recu = RecuMutation(cle_operation=action.cle_operation, statut=statut,
                                user_id=user_id, source="base",
                                postconditions=[asdict(p) for p in preuves])
            nouvelle = replace(action, recu=recu)
            self.actions[self.actions.index(action)] = nouvelle
            self._index[action.id] = nouvelle

    def en_attente(self, cle_operation: str) -> Action | None:
        """Une tentative ambigue (pending_confirmation) pour cette cle ?"""
        if not cle_operation:
            return None
        for a in reversed(self.actions):
            if a.cle_operation == cle_operation and a.en_attente_confirmation:
                return a
        return None

    def par_id(self, ident):
        if not ident or not isinstance(ident, str):
            return None
        return self._index.get(ident)

    def vide(self) -> bool:
        return not self.actions and not self.ecarts
