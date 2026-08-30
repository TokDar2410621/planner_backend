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

import json
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from django.conf import settings
from django.contrib.auth.models import User
from django.db import close_old_connections
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.usage import UsageLimits

from core.models import ConversationMessage, UploadedDocument
from services.agent_v2.mesure import epurer_reponse, fuites_reponse
from services.agent_v2.modeles import REGLAGES_DIRE, modele_agir, modele_dire
from services.agent_v2.outils import outils_pour
from services.agent_v2.prompts import PROMPT_DIRE, prompt_agir
from services.agent_v2.reconciliation import detecter_ecarts, reconcilier
from services.agent_v2.redaction import (ReponseDire, assembler,
                                          bloc_factuel, bloc_reste)
from services.agent_v2.registre import Registre

logger = logging.getLogger(__name__)

BUDGET_ETAPES = 10
HISTORIQUE_MAX = 20

# POOL DE THREADS REUTILISES, et le mot « reutilises » porte tout le poids.
#
# AGIR doit tourner a cote du generateur pour qu'on puisse emettre pendant
# qu'il travaille. La premiere version creait un thread NEUF par tour: mesure
# du 2026-08-28, le DEUXIEME tour d'un meme processus se bloque
# indefiniment, alors que trois tours dans le thread principal passent sans
# rien. Quatre tours sur un thread reutilise passent aussi, et douze tours
# concurrents sur un pool de quatre egalement.
#
# Consequence pratique: ne JAMAIS remplacer ce pool par un threading.Thread
# cree a la volee. Le symptome est un blocage silencieux au second message,
# donc invisible sur un essai unique.
_POOL_AGIR = ThreadPoolExecutor(
    max_workers=getattr(settings, "AGENT_V2_THREADS", 8),
    thread_name_prefix="agir",
)

# Le drainage attend par tranches plutot qu'indefiniment: si le thread meurt
# sans poser sa sentinelle, un get() sans delai gelerait la connexion SSE
# jusqu'au timeout du serveur.
ATTENTE_PENSEE = 0.5


def _cout(resultat, duree: float) -> dict:
    """Ce qu'une phase a reellement coute: allers-retours, jetons, secondes.

    `usage()` peut manquer selon le fournisseur. On ne fait alors pas semblant
    de savoir: zero, et la ligne de log le montrera tel quel.
    """
    vide = {"etapes": 0, "entree": 0, "sortie": 0, "raisonnement": 0,
            "cache": 0, "duree": duree}
    try:
        u = resultat.usage()
    except Exception:  # noqa: BLE001
        return vide
    # `details` porte ce que le fournisseur ajoute. Sur DeepSeek: les jetons de
    # RAISONNEMENT, et le partage entre succes et echecs de cache de prefixe.
    #
    # Ces deux nombres repondent a la question ouverte. Sonde du 2026-08-29:
    # au deuxieme appel d'une conversation, 7680 jetons d'entree sur 7781
    # etaient des succes de cache. Les trente schemas d'outils sont donc bien
    # renvoyes a chaque etape, mais ni refactures ni retraites: la piste du
    # contexte qui gonfle ne tient pas, et c'est le raisonnement qu'il faut
    # regarder.
    details = getattr(u, "details", None) or {}
    return {
        "etapes": getattr(u, "requests", 0) or 0,
        "entree": getattr(u, "input_tokens", 0) or 0,
        "sortie": getattr(u, "output_tokens", 0) or 0,
        "raisonnement": details.get("reasoning_tokens", 0) or 0,
        "cache": details.get("prompt_cache_hit_tokens", 0) or 0,
        "duree": duree,
    }
# Une lecture de semaine chargee depasse largement ce volume; on tronque plutot
# que de laisser un seul outil manger le contexte de la redaction.
EXTRAIT_MAX = 4000


def _extrait(donnees: dict) -> str:
    brut = json.dumps(donnees, ensure_ascii=False, default=str)
    if len(brut) <= EXTRAIT_MAX:
        return brut
    return f"{brut[:EXTRAIT_MAX]}... (tronque)"


class PlannerAgentV2:
    """Le nom est fixe par benchmarks/harness.py, qui l'importe tel quel."""

    def __init__(self, user: Optional[User] = None):
        self.user = user
        # Poses ici et pas seulement dans le flux: _agir et _historique sont
        # appelables directement (banc, tests), et une instance a demi
        # initialisee leverait un AttributeError loin de sa cause.
        self._tache: str = ""
        self._exclu: Optional[int] = None
        self._file_pensees: Optional[queue.Queue] = None

    def pousser_pensee(self, texte: str) -> None:
        """Emet un fragment de raisonnement vers le flux, s'il y a un flux.

        Sans file (appel direct depuis le banc ou un test), l'appel est sans
        effet: la primitive ne doit jamais imposer au reste du code de savoir
        s'il tourne dans un contexte streame.
        """
        if self._file_pensees is not None and texte:
            self._file_pensees.put(("thinking", texte))

    def signaler_outil(self, action) -> None:
        """Diffuse un appel d'outil vers le flux, s'il y a un flux.

        On envoie de quoi AFFICHER (nom, succes, message rendu par l'outil),
        jamais le dictionnaire d'arguments: il peut porter du contenu
        utilisateur, et le flux part vers le client.
        """
        if self._file_pensees is None or action is None:
            return
        self._file_pensees.put(("tool", {
            "id": action.id,
            "name": action.outil,
            "ok": bool(action.succes),
            "message": action.message or "",
        }))

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
            tools=outils_pour(user, registre, message, tache=self._tache,
                              signaler=self.signaler_outil),
        )
        async def sur_evenements(_contexte, evenements):
            """Capte les fragments de raisonnement AU FIL de leur production.

            run_sync execute le graphe en entier, donc tous les outils
            tournent; run_stream_sync s'arreterait a la premiere sortie
            « finale » et sauterait les appels suivants, ce qui serait faux
            ici. Le handler est le seul moyen d'avoir les deux.
            """
            async for evenement in evenements:
                delta = getattr(evenement, "delta", None)
                fragment = getattr(delta, "content_delta", None)
                if fragment:
                    self.pousser_pensee(fragment)

        try:
            depart = time.perf_counter()
            resultat = agent.run_sync(
                message,
                message_history=self._historique(user),
                usage_limits=UsageLimits(request_limit=BUDGET_ETAPES),
                event_stream_handler=sur_evenements,
            )
            self._cout_agir = _cout(resultat, time.perf_counter() - depart)
        except UsageLimitExceeded:
            # Le tour est tronque, pas rate: les outils deja executes ont
            # ecrit. Le bloc factuel le dira, c'est tout l'interet du registre.
            registre.budget_epuise = True
            self._cout_agir = {"etapes": BUDGET_ETAPES, "entree": 0, "sortie": 0,
                               "duree": time.perf_counter() - depart}
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
            output_retries=1,
        )

        tentatives = {"n": 0}

        @agent.output_validator
        def _sans_affirmation_d_action(sortie: ReponseDire) -> ReponseDire:
            # SECONDE CHANCE avant la guillotine, et JAMAIS plus. Contre-
            # expertise du 2026-08-30: lever encore a la recidive faisait
            # exploser run_sync (UnexpectedModelBehavior), la reponse entiere
            # partait au repli et la guillotine ne coupait jamais. Ici la
            # recidive est LIVREE a l'assemblage, qui supprime les phrases
            # fautives: la verite ne depend pas de la cooperation du modele.
            tentatives["n"] += 1
            fuites = fuites_reponse(sortie)
            if fuites and tentatives["n"] <= 1:
                champs = ", ".join(sorted({f.split(":", 1)[0] for f in fuites}))
                # Le message NE montre PAS comment reformuler l'affirmation:
                # la premiere version listait les tournures interdites et
                # poussait le modele vers le passif « a ete deplace »,
                # indetectable a l'epoque. On demande de RETIRER, pas de
                # deguiser.
                raise ModelRetry(
                    f"Les champs {champs} presentent une action comme faite, "
                    "en cours ou a venir. SUPPRIME ces phrases: toute action "
                    "reelle se cite uniquement dans `actions` avec sa "
                    "reference. N'evoque aucune action dans la prose, sous "
                    "aucune forme ni aucun temps."
                )
            return sortie
        brief = self._brief_dire(message, registre, etat, faits)

        # DIRE passe par le MEME pool qu'AGIR, et ce n'est pas un detail de
        # style. Mesure du 2026-08-28: appeler run_sync depuis le thread
        # principal apres qu'un thread du pool en a fait un bloque
        # indefiniment. Le tour se figeait juste apres le compte rendu
        # factuel, donc apres avoir tout affiche, ce qui rendait le defaut
        # particulierement trompeur. La seule configuration verifiee est
        # « tous les appels au modele sur le pool ».
        def _rediger():
            close_old_connections()
            try:
                depart_dire = time.perf_counter()
                sortie = agent.run_sync(brief)
                self._cout_dire = _cout(sortie, time.perf_counter() - depart_dire)
                return sortie.output
            finally:
                close_old_connections()

        return _POOL_AGIR.submit(_rediger).result()

    @staticmethod
    def _brief_dire(message: str, registre: Registre, etat: dict, faits: str) -> str:
        lignes = [f"MESSAGE DE L'UTILISATEUR:\n{message}", ""]
        if registre.actions:
            lignes.append("REGISTRE DU TOUR (seules ces references existent):")
            for a in registre.actions:
                etiquette = "OK" if a.succes else "ECHEC"
                lignes.append(f"  {a.id} [{etiquette}] {a.outil}: {a.message}")
                # Le CONTENU des lectures, sans quoi DIRE ne peut repondre a
                # « c'est quoi mon planning ? »: il saurait qu'un outil a
                # tourne sans savoir ce qu'il a renvoye (defaut observe le
                # 2026-08-25 sur un tour reel). Les MUTATIONS en sont exclues:
                # leur recit reste tenu par le bloc factuel et la validation
                # des references, et deverser leurs donnees brutes rouvrirait
                # le canal que la garantie structurelle ferme.
                if a.succes and not a.est_mutation and a.donnees:
                    lignes.append(f"       donnees: {_extrait(a.donnees)}")
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
        """Contrat SSE additif: status, thinking, delta, done. Le done fait
        AUTORITE et le client remplace toujours la bulle par son response."""
        self.user = user

        # Persiste d'abord, puis exclut CETTE ligne de l'historique par son id.
        # v1 devait s'en remettre a un filet (B9: message sauve, relu, puis
        # rajoute, donc duplique a chaque requete); ici la duplication est
        # structurellement impossible.
        courant = ConversationMessage.objects.create(
            user=user, role="user", content=message)
        self._exclu = courant.pk
        # Identifie CE tour pour les cles d'idempotence: deux tours
        # distincts peuvent legitimement refaire la meme action, un meme
        # tour rejoue ne doit l'executer qu'une fois.
        self._tache = f"{user.pk}:{courant.pk}"

        registre = Registre()

        # Le document et l'import recent DOIVENT entrer dans le message vu par
        # AGIR. Sans cela, un horaire envoye est perdu en silence et l'agent
        # decrit le planning deja en base en laissant croire qu'il a lu le
        # document (defaut observe le 2026-08-26 sur un tour reel). L'envoi
        # d'horaire est le premier chemin d'entree du produit.
        message_enrichi = message
        for evenement, complement in self._contexte_document(user, attachment):
            if evenement:
                yield evenement
            if complement:
                message_enrichi = f"{message_enrichi}\n\n{complement}"

        yield {"type": "status", "text": "R\u00e9flexion..."}

        # AGIR tourne dans un THREAD pour qu'on puisse emettre pendant qu'il
        # travaille. Mesure du 2026-08-28: sur une demande multi-etapes il
        # occupe 15 s des 25 s du tour, et l'utilisateur n'avait rien a lire
        # pendant ce temps. Le raisonnement etait bien capte, mais emis apres
        # coup: il decrivait une reflexion deja terminee.
        raisonnement, panne = "", None
        self._file_pensees = queue.Queue()

        def travailler():
            nonlocal raisonnement, panne
            # Ce thread vit hors du cycle de requete Django, qui ferme les
            # connexions: on s'en charge des deux cotes.
            close_old_connections()
            try:
                raisonnement = self._agir(user, message_enrichi, registre) or ""
            except Exception as e:  # noqa: BLE001
                panne = e
            finally:
                close_old_connections()
                self._file_pensees.put(None)  # sentinelle de fin

        futur = _POOL_AGIR.submit(travailler)
        fragments = 0
        while True:
            try:
                element = self._file_pensees.get(timeout=ATTENTE_PENSEE)
            except queue.Empty:
                # Filet: si le thread s'est termine sans poser sa sentinelle
                # (arret brutal du worker), on sort au lieu d'attendre pour
                # toujours et de geler la connexion SSE.
                if futur.done():
                    break
                continue
            if element is None:
                break
            fragments += 1
            genre, charge = element
            # La file transporte deux formes: un fragment de raisonnement
            # (texte nu) et un appel d'outil (dictionnaire deja pret).
            yield ({"type": genre, "text": charge} if isinstance(charge, str)
                   else {"type": genre, **charge})
        futur.result()  # remonte une panne du pool lui-meme, pas d'AGIR
        self._file_pensees = None

        # Repli pour les fournisseurs qui ne streament pas leurs deltas: sans
        # lui, leur raisonnement n'atteindrait le client que dans la charge
        # utile finale, et le volet resterait vide tout le tour. On perd le
        # gain de latence, jamais l'information.
        if raisonnement and not fragments:
            yield {"type": "thinking", "text": raisonnement}

        if panne is not None:
            # Une panne d'AGIR ne doit pas effacer ce que les outils ont deja
            # ecrit: le registre survit et le tour continue vers DIRE.
            logger.error("AGIR a echoue: %s", panne, exc_info=panne)

        etat: dict = {}
        if registre.mutations():
            yield {"type": "status", "text": "Je relis ton planning..."}
            etat = reconcilier(user, registre)
            detecter_ecarts(registre)

        faits = bloc_factuel(registre)
        # La section RESTE: demande contre place, une soustraction rendue par
        # du code. Elle rejoint les faits AVANT la redaction et le flux: le
        # manque se nomme au meme instant que le succes qu'il tempere.
        reste = bloc_reste(message, registre)
        if reste:
            faits = f"{faits}\n{reste}" if faits else reste
        if faits:
            # Les faits partent AVANT la redaction: ils sont deja vrais, et
            # l'utilisateur n'a pas a attendre l'enrobage pour les voir.
            yield {"type": "delta", "text": faits}

        texte = ""
        rejetees = 0
        supprimees = 0
        fuites: list[str] = []
        try:
            brut = self._dire(user, message, registre, etat, faits)
            # Fuites APRES la seconde chance du validateur: ce compteur dit
            # ce que le modele persiste a affirmer, pas ce qui part.
            fuites = fuites_reponse(brut)
            brut, supprimees = epurer_reponse(brut)
            texte, rejetees = assembler(brut, registre)
            if not texte.strip():
                # Tout etait mensonge: le compte rendu factuel reste la seule
                # chose vraie a dire.
                texte = self._repli(faits)
        except Exception as e:  # noqa: BLE001
            logger.error("DIRE a echoue: %s", e, exc_info=True)
            texte = self._repli(faits)

        # Une seule ligne par tour, mais pas toujours au meme niveau: une
        # reference rejetee est un mensonge que la garantie structurelle vient
        # d'attraper, et une fuite est une affirmation d'action dans le canal
        # qu'elle ne protege pas. Ce sont LES deux signaux du projet; en INFO
        # ils se noieraient dans le bruit et personne ne les verrait passer.
        cout_agir = getattr(self, "_cout_agir", None) or {}
        cout_dire = getattr(self, "_cout_dire", None) or {}
        anormal = bool(rejetees or fuites)
        logger.log(
            logging.WARNING if anormal else logging.INFO,
            "agent_v2 tour actions=%d rejetees=%d fuites=%d supprimees=%d ecarts=%d%s"
            " agir=%.1fs/%dep/%d->%dj/r%d/c%d dire=%.1fs/%dep/%d->%dj/r%d",
            len(registre.actions),
            rejetees,
            len(fuites),
            supprimees,
            len(registre.ecarts),
            f" types={','.join(fuites)}" if fuites else "",
            # CHRONO. Il entre dans LA ligne du tour plutot que d'en ouvrir une
            # seconde: le contrat est une ligne agregee par tour, et c'est
            # aussi ce qui permet de correler duree et verite sans jointure.
            # En production le 2026-08-29, la mediane etait de 57 s et un tour
            # a atteint 397 s sans qu'aucun log ne dise ou passait le temps: on
            # ne pouvait qu'inferer des ecarts entre lignes httpx.
            cout_agir.get("duree", 0.0), cout_agir.get("etapes", 0),
            cout_agir.get("entree", 0), cout_agir.get("sortie", 0),
            cout_agir.get("raisonnement", 0), cout_agir.get("cache", 0),
            cout_dire.get("duree", 0.0), cout_dire.get("etapes", 0),
            cout_dire.get("entree", 0), cout_dire.get("sortie", 0),
            cout_dire.get("raisonnement", 0),
        )


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

    @staticmethod
    def _contexte_document(user: User, attachment):
        """Rend des couples (evenement a emettre, complement de message).

        Les deux formateurs viennent de v1 et sont repris tels quels: ils
        portent des consignes produit affinees par l'audit (ne jamais nier un
        import, ne jamais promettre un resume qu'on ne livrera pas, presenter
        les blocs comme le RESULTAT de l'import). Les reecrire les ferait
        deriver.
        """
        from services.agent.agent import PlannerAgent
        v1 = PlannerAgent()

        if attachment is not None:
            if not attachment.processed:
                # Meme attente cooperative que v1: gunicorn tourne en workers
                # gevent avec monkey.patch_all(), ce sommeil ne suspend que
                # cette conversation. Une reponse vraie en un tour vaut mieux
                # qu'une promesse cassee en trois secondes.
                attente = getattr(settings, "ATTACHMENT_WAIT_SECONDS", 45)
                if attente:
                    yield {"type": "status", "text": "J'analyse ton document..."}, None
                    for tic in range(int(attente * 2)):
                        time.sleep(0.5)
                        attachment.refresh_from_db()
                        if attachment.processed:
                            break
                        if tic and tic % 16 == 0:
                            yield ({"type": "status",
                                    "text": "J'analyse ton document... (presque fini)"}, None)
            yield None, v1._build_attachment_context(attachment)

        # Vaut AVEC ou SANS piece jointe: un « c'est bon ? » au tour suivant
        # doit voir l'import, sinon l'agent invente une limitation et dit a
        # l'utilisateur l'inverse de la verite.
        recent = v1._recent_import_context(user)
        if recent:
            yield None, recent

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
