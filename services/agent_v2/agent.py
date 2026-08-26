"""
La boucle: AGIR, RECONCILIER, DIRE.

La difference de fond avec v1 tient en une phrase: le recit d'action n'est plus
produit par le modele. AGIR outille et alimente un registre ecrit par le
runtime; le code rend un compte rendu factuel depuis ce registre; DIRE ne fait
qu'enrober, et toute phrase citant une action qui n'existe pas est supprimee a
l'assemblage.

La surface publique porte les QUATRE points d'entree que core/views.py et le
banc exigent. views.py:861 lit result['response'] par indexation DIRECTE: une
cle manquante rend un 500 a l'utilisateur.
"""
from __future__ import annotations

import logging
from typing import Optional

from django.contrib.auth.models import User
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.usage import UsageLimits

from core.models import ConversationMessage, UploadedDocument
from services.agent_v2.modeles import REGLAGES_DIRE, modele_agir, modele_dire
from services.agent_v2.outils import outils_pour
from services.agent_v2.prompts import PROMPT_DIRE, prompt_agir
from services.agent_v2.reconciliation import detecter_ecarts, reconcilier
from services.agent_v2.redaction import ReponseDire, assembler, bloc_factuel
from services.agent_v2.registre import Registre

logger = logging.getLogger(__name__)

BUDGET_ETAPES = 10
HISTORIQUE_MAX = 20


class PlannerAgentV2:
    """Le nom est fixe par benchmarks/harness.py, qui l'importe tel quel."""

    def __init__(self, user: Optional[User] = None):
        self.user = user

    # ------------------------------------------------------------------ AGIR

    def _agir(self, user: User, message: str, registre: Registre) -> str:
        """Laisse le modele outiller. Rend son raisonnement, jamais persiste.

        Le registre est alimente par l'adaptateur d'outils a chaque execution:
        cette methode ne l'ecrit pas elle-meme, et c'est voulu. Une action ne
        peut entrer dans le registre qu'en ayant reellement ete executee.
        """
        agent = Agent(
            modele_agir(),
            system_prompt=prompt_agir(user),
            tools=outils_pour(user, registre, message),
        )
        try:
            resultat = agent.run_sync(
                message,
                message_history=self._historique(user),
                usage_limits=UsageLimits(request_limit=BUDGET_ETAPES),
            )
        except UsageLimitExceeded:
            # Le tour est tronque, pas rate: les outils deja executes ont
            # ecrit. Le bloc factuel le dira, c'est tout l'interet du registre.
            registre.budget_epuise = True
            return ""
        return self._raisonnement(resultat)

    # ------------------------------------------------------------------ DIRE

    def _dire(self, user: User, message: str, registre: Registre,
              etat: dict, faits: str) -> ReponseDire:
        """Redige, sans outil. REGLAGES_DIRE coupe le raisonnement: verifie par
        sonde, DeepSeek refuse tool_choice=required en mode thinking, or c'est
        ainsi que PydanticAI force une sortie structuree."""
        agent = Agent(
            modele_dire(),
            output_type=ReponseDire,
            system_prompt=PROMPT_DIRE,
            model_settings=REGLAGES_DIRE,
        )
        return agent.run_sync(self._brief_dire(message, registre, etat, faits)).output

    @staticmethod
    def _brief_dire(message: str, registre: Registre, etat: dict, faits: str) -> str:
        lignes = [f"MESSAGE DE L'UTILISATEUR:\n{message}", ""]
        if registre.actions:
            lignes.append("REGISTRE DU TOUR (seules ces references existent):")
            for a in registre.actions:
                etiquette = "OK" if a.succes else "ECHEC"
                lignes.append(f"  {a.id} [{etiquette}] {a.outil}: {a.message}")
        else:
            lignes.append("REGISTRE DU TOUR: VIDE. Tu n'as rien accompli.")
        for e in registre.ecarts:
            lignes.append(f"  {e.id} [ECART] {e.description}")
        if faits:
            lignes += ["", "COMPTE RENDU DEJA AFFICHE (ne le repete pas):", faits]
        if etat:
            lignes += ["", f"ETAT RELU APRES ECRITURE: {list(etat)}"]
        return "\n".join(lignes)

    # ---------------------------------------------------------------- PUBLIC

    def process_message_stream(
        self,
        user: User,
        message: str,
        attachment: Optional[UploadedDocument] = None,
        *,
        use_streaming: bool = True,
        generate_quick_replies: bool = False,
    ):
        """Contrat SSE identique a v1: status, delta, done. Le done fait
        AUTORITE et le client remplace toujours la bulle par son response."""
        self.user = user

        # Persiste d'abord, puis exclut CETTE ligne de l'historique par son id.
        # v1 devait s'en remettre a un filet (B9: message sauve, relu, puis
        # rajoute, donc duplique a chaque requete); ici la duplication est
        # structurellement impossible.
        courant = ConversationMessage.objects.create(
            user=user, role="user", content=message)
        self._exclu = courant.pk

        registre = Registre()
        yield {"type": "status", "text": "Je regarde ton planning..."}

        raisonnement = ""
        try:
            raisonnement = self._agir(user, message, registre) or ""
        except Exception as e:  # noqa: BLE001
            # Une panne d'AGIR ne doit pas effacer ce que les outils ont deja
            # ecrit: le registre survit et le tour continue vers DIRE.
            logger.error("AGIR a echoue: %s", e, exc_info=True)

        etat: dict = {}
        if registre.mutations():
            yield {"type": "status", "text": "Je relis ton planning..."}
            etat = reconcilier(user, registre)
            detecter_ecarts(registre)

        faits = bloc_factuel(registre)
        if faits:
            # Les faits partent AVANT la redaction: ils sont deja vrais, et
            # l'utilisateur n'a pas a attendre l'enrobage pour les voir.
            yield {"type": "delta", "text": faits}

        texte = ""
        try:
            brut = self._dire(user, message, registre, etat, faits)
            texte, rejetees = assembler(brut, registre)
            if rejetees:
                # La mesure du canal que la garantie structurelle protege.
                logger.warning(
                    "DIRE a cite %d action(s) inexistante(s), supprimee(s)", rejetees)
        except Exception as e:  # noqa: BLE001
            logger.error("DIRE a echoue: %s", e, exc_info=True)
            texte = self._repli(faits)

        ConversationMessage.objects.create(
            user=user, role="assistant", content=texte)

        yield {
            "type": "done",
            "response": texte,
            "quick_replies": [],
            "blocks_created": self._crees(registre, "create_block", "created"),
            "tasks_created": self._crees(registre, "create_task", "task"),
            "raisonnement": raisonnement,
        }

    def process_message(
        self,
        user: User,
        message: str,
        attachment: Optional[UploadedDocument] = None,
        generate_quick_replies: bool = True,
    ) -> dict:
        """Enveloppe non streamee: draine le flux, seule source de verite."""
        done: dict = {}
        for event in self.process_message_stream(
            user, message, attachment,
            use_streaming=False,
            generate_quick_replies=generate_quick_replies,
        ):
            if event.get("type") == "done":
                done = {k: v for k, v in event.items() if k != "type"}
        return done

    def quick_replies_for(
        self, user: User, user_message: str, assistant_response: str,
    ) -> list[dict]:
        """Une vue l'appelle et avale les exceptions: sans cette methode, les
        chips disparaitraient en silence pour tout compte bascule.

        Les suggestions n'ont rien a voir avec la verite d'action, et v1 les
        rend bien: on delegue plutot que de dupliquer."""
        try:
            from services.agent.agent import PlannerAgent
            return PlannerAgent().quick_replies_for(
                user, user_message, assistant_response) or []
        except Exception:  # noqa: BLE001 - une suggestion ne remonte jamais d'erreur
            logger.debug("Suggestions indisponibles", exc_info=True)
            return []

    # ------------------------------------------------------------------ util

    def _historique(self, user: User) -> list:
        lignes = (ConversationMessage.objects
                  .filter(user=user).exclude(pk=getattr(self, "_exclu", None))
                  .order_by("-created_at")[:HISTORIQUE_MAX])
        messages = []
        for ligne in reversed(list(lignes)):
            if ligne.role == "user":
                messages.append(ModelRequest(parts=[UserPromptPart(content=ligne.content)]))
            else:
                messages.append(ModelResponse(parts=[TextPart(content=ligne.content)]))
        return messages

    @staticmethod
    def _raisonnement(resultat) -> str:
        """Le raisonnement est ephemere: affiche, jamais persiste."""
        morceaux = []
        try:
            for message in resultat.all_messages():
                for part in getattr(message, "parts", []):
                    if type(part).__name__ == "ThinkingPart":
                        morceaux.append(getattr(part, "content", "") or "")
        except Exception:  # noqa: BLE001 - un volet d'affichage ne casse pas un tour
            logger.debug("Raisonnement illisible", exc_info=True)
        return "\n".join(m for m in morceaux if m)

    @staticmethod
    def _repli(faits: str) -> str:
        """DIRE est tombe. Se taire laisserait l'utilisateur croire que rien
        n'a eu lieu, alors que son planning a peut-etre change."""
        if faits:
            return f"{faits}\n\nJe n'ai pas pu rediger de reponse complete, mais voici ce qui a ete fait."
        return "Je n'ai pas reussi a traiter ta demande. Peux-tu reformuler ?"

    @staticmethod
    def _crees(registre: Registre, outil: str, cle: str) -> list:
        sortie: list = []
        for action in registre.actions:
            if action.outil != outil or not action.succes:
                continue
            valeur = action.donnees.get(cle)
            if isinstance(valeur, list):
                sortie.extend(valeur)
            elif valeur:
                sortie.append(valeur)
        return sortie
