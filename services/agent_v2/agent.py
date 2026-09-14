"""
La boucle: AGIR, RECONCILIER, DIRE.

La difference de fond avec v1 tient en une phrase: le recit d'action n'est plus
produit par le modele. AGIR outille et alimente un registre ecrit par le
runtime; le code rend un compte rendu factuel depuis ce registre; DIRE ne fait
qu'enrober, et toute sortie citant une action qui n'existe pas est ecartee a
l'assemblage.

Depuis le 2026-09-14 (lots 1 a 3), un tour s'affiche en trois sections
streamees dans leur ordre final: FAITS (code), PROSE (DIRE epuree), QUESTION
(une seule, choisie par PRIORITE). done.response est exactement la
concatenation des deltas, et le message persiste porte ses boutons et ses
demandes en metadonnees pour que le tour suivant sache a quoi l'utilisateur
repond.

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
from pydantic_ai.messages import (ModelRequest, ModelResponse, PartDeltaEvent,
                                  PartStartEvent, TextPart, ThinkingPart,
                                  ThinkingPartDelta, UserPromptPart)
from pydantic_ai.usage import UsageLimits

from core.models import ConversationMessage, UploadedDocument
from services.agent_v2.mesure import epurer_reponse, fuites_reponse, questions_et_offres
from services.agent_v2.modeles import REGLAGES_DIRE, modele_agir, modele_dire
from services.agent_v2.outils import outils_pour
from services.agent_v2.prompts import PROMPT_DIRE, prompt_agir
from services.agent_v2.reconciliation import detecter_ecarts, reconcilier
from services.agent_v2.importation import inscrire_import
from services.agent_v2.redaction import (LECTURES_RENDUES, ReponseDire,
                                          bloc_factuel, bloc_reste, composer,
                                          contient_question, marqueurs_bruts,
                                          question_code, sans_tiret_long)
from services.agent_v2.registre import Registre

logger = logging.getLogger(__name__)

BUDGET_ETAPES = 10
HISTORIQUE_MAX = 20

# Une question par tour, la premiere presente dans cet ordre. Les gardes du
# code passent avant tout: une suppression retenue sans sa question serait un
# refus muet. Le formulaire passe avant le choix du modele, et DIRE ferme la
# marche.
PRIORITE = [
    "portee_jour", "destructif", "heure_refusee", "creation_en_masse",
    "optimisation", "formulaire", "choix_modele", "chevauchement",
    "fin_recurrence", "creneaux", "dire",
]
CHIPS_MAX = 4

REPLI_PROSE = "Voici ce qui a changé. Dis-moi si tu veux autre chose."
REPLI_PROSE_LECTURE = "Dis-moi si tu veux autre chose."
REPLI_QUESTION = "Je n'ai pas compris. Tu veux ajouter, déplacer ou voir quelque chose ?"
PROSE_FORMULAIRE = "Il me manque quelques précisions."
PROSE_REPRISE = "Il me faut juste cette précision avant de toucher à ton horaire."
PROSE_ANNULEE = "D'accord, je ne change rien."

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


# ── Chargeurs de boutons.py et outils.py (simulables par les tests) ──────


def _charger_question_forcee():
    """boutons.question_forcee, charge par une fonction pour que les tests la simulent."""
    from services.agent_v2.boutons import question_forcee
    return question_forcee


def _charger_appliquer_choix():
    """outils.appliquer_choix_en_attente, charge par une fonction pour les tests."""
    from services.agent_v2.outils import appliquer_choix_en_attente
    return appliquer_choix_en_attente


def _charger_tour_decide():
    """outils.tour_entierement_decide_par_le_code, ou None s'il n'existe pas.

    Contrat du round 6 avec les gardes: vrai quand le message ne fait que
    repondre (ou echouer a repondre) a une demande en attente, sans nouvelle
    requete. Absente, la fonction ne decide rien et AGIR tourne comme avant.
    """
    from services.agent_v2 import outils
    return getattr(outils, "tour_entierement_decide_par_le_code", None)


def _abandonnee(action) -> bool:
    return bool((action.donnees or {}).get("abandonnee_par_le_code"))


def _garde_a_retenu(registre: Registre) -> bool:
    """Une garde du code a-t-elle retenu une action ce tour ?"""
    for a in registre.actions:
        if a.succes or _abandonnee(a):
            continue
        d = a.donnees or {}
        if (isinstance(d.get("demande"), dict) or d.get("needs_confirmation")
                or d.get("requires_confirmation") or d.get("heure_dite")):
            return True
    return False


def _brouillon_interdit(registre: Registre) -> bool:
    """Le brouillon d'AGIR est-il ecarte du brief ce tour ?

    Revue de verite du round 4: _garde_a_retenu ignorait une date passee, une
    erreur d'outil et les autres echecs, et le brouillon racontait l'action
    ratee (« Je l'ai mis jeudi a 9 h, veux-tu... ? »). Toute action en echec
    ce tour, retenue ou non, ecarte le brouillon entier.
    """
    # Une demande laissee tombee par le code (D2) n'est pas un echec du tour:
    # son brouillon peut encore porter la question d'une nouvelle requete.
    return _garde_a_retenu(registre) or any(
        not a.succes and a.outil != "import_recent" and not _abandonnee(a)
        for a in registre.actions)


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
BROUILLON_MAX = 1500
ECHANGE_MAX = 400


def _extrait(donnees: dict) -> str:
    brut = json.dumps(donnees, ensure_ascii=False, default=str)
    if len(brut) <= EXTRAIT_MAX:
        return brut
    return f"{brut[:EXTRAIT_MAX]}... (tronque)"


def _json_sur(valeur):
    """Les metadonnees passent en JSONField: tout ce qui n'est pas natif devient texte."""
    return json.loads(json.dumps(valeur, ensure_ascii=False, default=str))


def _chips_propres(chips, garder_option: bool) -> list[dict]:
    propres: list[dict] = []
    for chip in chips or []:
        if not isinstance(chip, dict) or not chip.get("label") or not chip.get("value"):
            continue
        propre = {"label": sans_tiret_long(str(chip["label"])),
                  "value": sans_tiret_long(str(chip["value"]))}
        if garder_option and chip.get("option") is not None:
            propre["option"] = chip["option"]
        propres.append(propre)
    return propres[:CHIPS_MAX]


def _rang(motif) -> int:
    # Un motif inconnu est traite comme informatif, au rang du chevauchement.
    return PRIORITE.index(motif) if motif in PRIORITE else PRIORITE.index("chevauchement")


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
        self._message_brut: Optional[str] = None
        self._brouillon_agir: str = ""

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

    async def _sur_evenements(self, _contexte, evenements) -> None:
        """Capte le RAISONNEMENT au fil de sa production, et rien d'autre.

        run_sync execute le graphe en entier, donc tous les outils tournent;
        run_stream_sync s'arreterait a la premiere sortie « finale » et
        sauterait les appels suivants. Le handler est le seul moyen d'avoir
        les deux.

        Deux defauts corriges le 2026-09-14: la version precedente lisait
        `content_delta` sur tout evenement, donc les brouillons en francais
        d'AGIR (TextPartDelta) se melaient au raisonnement anglais du volet;
        et elle ignorait PartStartEvent, qui porte le PREMIER fragment de
        chaque partie (d'ou « user wants » sans « The »).
        """
        async for evenement in evenements:
            if isinstance(evenement, PartStartEvent):
                if isinstance(evenement.part, ThinkingPart):
                    self.pousser_pensee(evenement.part.content)
            elif isinstance(evenement, PartDeltaEvent):
                if isinstance(evenement.delta, ThinkingPartDelta):
                    self.pousser_pensee(evenement.delta.content_delta)

    def _agir(self, user: User, message: str, registre: Registre) -> str:
        """Laisse le modele outiller. Rend son raisonnement, jamais persiste.

        Le registre est alimente par l'adaptateur d'outils a chaque execution:
        cette methode ne l'ecrit pas elle-meme, et c'est voulu. Une action ne
        peut entrer dans le registre qu'en ayant reellement ete executee.

        Le texte final d'AGIR est garde dans `_brouillon_agir`: c'est la que
        vivent ses questions de clarification, que DIRE doit reprendre.
        """
        self._brouillon_agir = ""
        # Les regles de la garde (confirmation, heure dite, portee d'un jour)
        # lisent le message TAPE, jamais sa version enrichie du document.
        brut = self._message_brut if self._message_brut is not None else message
        outils = outils_pour(user, registre, message_du_tour=message,
                             tache=self._tache, signaler=self.signaler_outil,
                             message_brut=brut)
        # instructions= et non system_prompt=: pydantic-ai n'ajoute les system
        # prompts QUE si l'historique est vide. Sur tout tour de suivi, AGIR
        # tournait sans date, sans table de decision ni semaine type, et a
        # place une revision en 2025 (banc du 2026-09-14). Les instructions
        # partent a chaque requete.
        agent = Agent(
            modele_agir(),
            instructions=prompt_agir(user),
            tools=outils,
        )

        depart = time.perf_counter()
        try:
            resultat = agent.run_sync(
                message,
                message_history=self._historique(user),
                usage_limits=UsageLimits(request_limit=BUDGET_ETAPES),
                event_stream_handler=self._sur_evenements,
            )
            self._cout_agir = _cout(resultat, time.perf_counter() - depart)
        except UsageLimitExceeded:
            # Le tour est tronque, pas rate: les outils deja executes ont
            # ecrit. Le bloc factuel le dira, c'est tout l'interet du registre.
            registre.budget_epuise = True
            self._cout_agir = {"etapes": BUDGET_ETAPES, "entree": 0, "sortie": 0,
                               "duree": time.perf_counter() - depart}
            return ""
        sortie = getattr(resultat, "output", "")
        self._brouillon_agir = sortie.strip() if isinstance(sortie, str) else ""
        return self._raisonnement(resultat)

    # ------------------------------------------------------------------ DIRE

    def _dire(self, user: User, message: str, registre: Registre,
              etat: dict, faits: str, brouillon: str = "",
              question_code: str = "", historique_court: str = "") -> ReponseDire:
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
                # On demande de RETIRER, pas de deguiser en passif. Mais on dit
                # aussi ce qui reste permis: la version precedente faisait
                # fuir les questions et les offres avec les affirmations, et
                # l'agent ne demandait plus rien.
                raise ModelRetry(
                    f"Les champs {champs} presentent une action comme deja "
                    "faite ou en cours. SUPPRIME ces phrases: les actions "
                    "reelles sont deja affichees par le code. Les questions et "
                    "les offres restent permises, par exemple « Veux-tu que je "
                    "le deplace a 14 h ? » ou « Donne-moi l'heure et je le "
                    "place. »"
                )
            return sortie
        brief = self._brief_dire(message, registre, etat, faits,
                                 brouillon=brouillon, question_code=question_code,
                                 historique_court=historique_court)

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
    def _brief_dire(message: str, registre: Registre, etat: dict, faits: str,
                    brouillon: str = "", question_code: str = "",
                    historique_court: str = "") -> str:
        lignes: list[str] = []
        if historique_court:
            lignes += ["DEUX DERNIERS ECHANGES (contexte, ne les repete pas):",
                       historique_court, ""]
        lignes += [f"MESSAGE DE L'UTILISATEUR:\n{message}", ""]
        if registre.actions:
            lignes.append("REGISTRE DU TOUR (seules ces references existent):")
            for a in registre.actions:
                if a.outil == "import_recent":
                    # Un import d'un tour precedent: DIRE s'en sert pour
                    # repondre, mais le citer a chaque tour pendant vingt
                    # minutes redisait « c'est importe » a toute question.
                    etiquette = "CONTEXTE (ne pas citer)"
                else:
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
            lignes += ["", "COMPTE RENDU DEJA AFFICHE (ne repete ni ses noms, ni ses "
                       "heures, ni ses nombres):", faits]
        if question_code:
            lignes += ["", "QUESTION DEJA POSEE PAR LE CODE (laisse question et "
                       "options vides):", question_code]
        # Le brouillon d'AGIR est le canal ou le modele raconte ses actions,
        # y compris celles qu'une garde a retenues (revue de verite du round
        # 3). Seules ses questions et ses offres propres entrent au brief, et
        # rien du tout quand une garde a retenu une action ce tour: le code
        # pose alors la question.
        brouillon = "" if _brouillon_interdit(registre) else questions_et_offres(brouillon)
        if brouillon:
            extrait = brouillon if len(brouillon) <= BROUILLON_MAX \
                else f"{brouillon[:BROUILLON_MAX]}... (tronque)"
            lignes += ["", "BROUILLON D'AGIR (ses questions et offres seulement):", extrait]
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
        """Contrat SSE additif: status, thinking, tool, delta, done. Les deltas
        arrivent dans l'ordre final (faits, prose, question) et done.response
        en est exactement la concatenation."""
        self.user = user
        depart_tour = time.perf_counter()
        self._cout_agir, self._cout_dire = {}, {}
        # Le message TAPE, avant tout enrichissement: c'est lui que lisent la
        # garde et les boutons forces.
        self._message_brut = message
        self._brouillon_agir = ""
        self._journaliser_reponse_formulaire(user, message)

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
        # Lu AVANT l'attente: seul un document qui bascule pendant ce tour
        # compte comme « traite ce tour » (equivalence avec v1 detaillee dans
        # la docstring de boutons_forces).
        deja_traite = attachment is not None and attachment.processed
        for evenement, complement in self._contexte_document(user, attachment):
            if evenement:
                yield evenement
            if complement:
                message_enrichi = f"{message_enrichi}\n\n{complement}"
        attachment_traite_ce_tour = (
            attachment is not None and not deja_traite and attachment.processed)

        # L'import fait par le systeme entre au registre AVANT AGIR, comme
        # une action deja accomplie. Sans cela DIRE lit « registre vide » et
        # propose de renvoyer l'horaire qui vient d'etre importe (deux tours
        # reels le 2026-09-01). Detail dans importation.py.
        try:
            inscrire_import(registre, user, attachment)
        except Exception:  # noqa: BLE001 - un recap absent vaut mieux qu'un tour tombe
            logger.error("Import du document non inscrit au registre", exc_info=True)

        # La reponse a une question du tour precedent s'execute par le CODE,
        # avant AGIR: un tap sur « Tous les jeudis » supprime la serie sans
        # qu'un modele ait a le refaire (ni a pouvoir le rater). AGIR recoit
        # le bilan pour ne pas le rejouer; le message brut reste intact.
        self._file_pensees = queue.Queue()
        choix: list = []
        try:
            choix = list(_charger_appliquer_choix()(
                user, registre, message, tache=self._tache,
                signaler=self.signaler_outil) or [])
        except Exception:  # noqa: BLE001 - un choix non applique se redemande
            logger.error("Choix en attente non appliques", exc_info=True)
            choix = []
        yield from self._vider_file()
        choix = [c for c in choix if isinstance(c, dict)]
        choix_code = sum(1 for c in choix if c.get("action_id"))
        # Une reponse floue a une question gardee ne perd pas la demande: les
        # gardes la reposent UNE fois et l'inscrivent au registre avec
        # reposee_par_le_code (contrat du round 6). La voix ne relit plus
        # l'attente elle-meme: une seule source decide de reposer.
        reemises = [a.donnees["demande"] for a in registre.actions
                    if (a.donnees or {}).get("reposee_par_le_code")
                    and isinstance(a.donnees.get("demande"), dict)]
        resumes = [str(c["resume"]) for c in choix if c.get("resume")]
        if resumes:
            message_enrichi = (f"{message_enrichi}\n\nSUITE AU CHOIX DE L'UTILISATEUR:\n- "
                               + "\n- ".join(resumes))

        # CHEMIN RAPIDE (D6). Banc du round 5, s05-2: le code avait repose la
        # demande a 0,06 s, puis AGIR a raisonne 105,6 s sans outil et son
        # brouillon a ete jete. Quand le message ne fait que repondre (ou ne
        # pas repondre) a une demande en attente, le tour se construit depuis
        # la decision du code: ni AGIR, ni DIRE.
        # Round 8: une piece jointe est toujours une demande pour AGIR, meme
        # quand le texte tape ne fait que repondre.
        par_le_code = attachment is None and self._tour_decide(registre, message)
        raisonnement, panne = "", None
        if par_le_code:
            self._file_pensees = None
        else:
            yield {"type": "status", "text": "Réflexion..."}
            raisonnement, panne = yield from self._agir_en_fond(user, message_enrichi, registre)

        if panne is not None:
            # Une panne d'AGIR ne doit pas effacer ce que les outils ont deja
            # ecrit: le registre survit et le tour continue vers DIRE.
            logger.error("AGIR a echoue: %s", panne, exc_info=panne)

        etat: dict = {}
        if registre.mutations():
            yield {"type": "status", "text": "Je relis ton planning..."}
            etat = reconcilier(user, registre)
            detecter_ecarts(registre)

        # La question du tour est choisie AVANT les faits: les actions
        # retenues que cette question couvre n'ont pas a etre redites, les
        # autres recoivent leur ligne « pas encore ».
        gagnant = self._choisir_question(
            user, message, attachment, registre, attachment_traite_ce_tour,
            reemises=reemises, sans_forcee=par_le_code)
        par_demande = bool(gagnant) and gagnant.get("source") == "demande"
        cles_posees = set(gagnant.get("cles_posees") or []) if par_demande else set()

        # Une lecture qui n'a servi qu'a preparer un formulaire ou un choix ne
        # se deverse pas au-dessus de la question (banc du round 4, s02-1).
        sans_lecture = bool(gagnant) and (
            gagnant.get("source") == "formulaire" or gagnant.get("motif") == "choix_modele"
        ) and not any(a.succes and a.est_mutation for a in registre.actions)
        faits = bloc_factuel(registre, cles_posees=cles_posees, sans_lecture=sans_lecture)
        # La section RESTE: demande contre place, une soustraction rendue par
        # du code. Elle rejoint les faits AVANT la redaction et le flux: le
        # manque se nomme au meme instant que le succes qu'il tempere.
        reste = bloc_reste(message, registre)
        if reste:
            faits = f"{faits}\n{reste}" if faits else reste
        faits = sans_tiret_long(faits or "")

        emis: list[str] = []
        if faits:
            # Les faits partent AVANT la redaction: ils sont deja vrais, et
            # l'utilisateur n'a pas a attendre l'enrobage pour les voir.
            emis.append(faits)
            yield {"type": "delta", "text": faits}

        question_deja = ""
        if gagnant:
            question_deja = gagnant.get("question") or "(un formulaire est affiché)"

        supprimees = 0
        fuites: list[str] = []
        panne_dire = False
        if par_le_code:
            compo = composer(None, registre, faits, gagnant)
        else:
            try:
                brut = self._dire(user, message, registre, etat, faits,
                                  brouillon="" if reemises else self._brouillon_agir,
                                  question_code=question_deja,
                                  historique_court=self._deux_derniers_echanges(user))
                # Fuites APRES la seconde chance du validateur: ce compteur dit
                # ce que le modele persiste a affirmer, pas ce qui part.
                fuites = fuites_reponse(brut)
                brut, supprimees = epurer_reponse(brut)
                compo = composer(brut, registre, faits, gagnant)
            except Exception as e:  # noqa: BLE001
                logger.error("DIRE a echoue: %s", e, exc_info=True)
                panne_dire = True
                compo = composer(None, registre, faits, gagnant)

        # Zero tiret long dans ce que lit l'utilisateur, quelle que soit la
        # source (banc du round 3, s06-1).
        prose, question = sans_tiret_long(compo.prose), sans_tiret_long(compo.question)
        motif, chips = compo.motif, compo.chips
        mutation_reussie = any(a.succes and a.est_mutation for a in registre.actions)
        if reemises and par_demande and cles_posees & {d.get("cle") for d in reemises}:
            # Revue de lisibilite du round 4: DIRE lisait une reponse de garde
            # et ecrivait « D'accord, je garde ta chimie. » juste avant la
            # question de suppression reposee. Sur ce tour, le code parle seul.
            # Round 6 (D5): « avant de toucher a ton horaire » serait faux
            # apres une mutation reussie; les faits parlent, puis la question.
            prose = "" if mutation_reussie else PROSE_REPRISE
        annulee = (any(c.get("decision_code") == "annulee" for c in choix)
                   or any((a.donnees or {}).get("decision_code") == "annulee"
                          for a in registre.actions))
        if par_le_code and annulee and not faits and not prose and not question:
            # Une garde fermee par « laisse faire »: une ligne, pas un silence.
            prose = PROSE_ANNULEE
        formulaire = gagnant.get("interactive_inputs") if gagnant and \
            gagnant.get("source") == "formulaire" else None
        if panne_dire and faits and not prose:
            # DIRE est tombe. Se taire laisserait croire que rien n'a eu lieu.
            prose = REPLI_PROSE if mutation_reussie else REPLI_PROSE_LECTURE
        if formulaire and not faits and not prose:
            prose = PROSE_FORMULAIRE
        if not faits and not prose and not question and not formulaire:
            question, motif, chips = REPLI_QUESTION, "dire", []

        if prose:
            morceau = ("\n\n" if emis else "") + prose
            emis.append(morceau)
            yield {"type": "delta", "text": morceau}
        if question:
            morceau = ("\n\n" if emis else "") + question
            emis.append(morceau)
            yield {"type": "delta", "text": morceau}
        response = "".join(emis)

        quick_replies = _chips_propres(chips, garder_option=False)
        # Une question restee dans la prose compte aussi: sinon des puces
        # differees viennent contredire la question (banc du round 3, s05-2).
        question_posee = bool(gagnant) or bool(question) or contient_question(prose)
        lecture_reussie = any(a.succes and a.outil in LECTURES_RENDUES
                              for a in registre.actions)
        lecture_sans_liste = compo.lecture_sans_liste or (lecture_reussie and not faits)
        try:
            marqueurs = list(marqueurs_bruts(response))
        except Exception:  # noqa: BLE001 - une mesure ne casse pas un tour
            marqueurs = []
        rejetees = compo.rejetees

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
            " agir=%.1fs/%dep/%d->%dj/r%d/c%d dire=%.1fs/%dep/%d->%dj/r%d"
            " asked=%d form=%d choices=%d read_without_list=%d raw_marker_count=%d"
            " motif=%s choix_code=%d chemin=%s tour=%.2fs",
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
            # MESURE DES QUESTIONS (lot 3g): a-t-on demande, par quel canal,
            # combien de boutons, une lecture sans liste, du texte machine.
            1 if question_posee else 0,
            1 if formulaire else 0,
            len(quick_replies),
            1 if lecture_sans_liste else 0,
            len(marqueurs),
            motif or "-",
            choix_code,
            # Latence du chemin rapide (D6) contre la boucle complete.
            "code" if par_le_code else "agir",
            time.perf_counter() - depart_tour,
        )

        question_affichee = "" if motif == "formulaire" else question
        metadonnees = {
            "agent": "v2",
            "en_reponse_a": self._exclu,
            "quick_replies": quick_replies,
            "interactive_inputs": formulaire or [],
            "question_posee": question_posee,
            "question": question_affichee,
            "question_motif": motif or "",
            # SEULEMENT les demandes rendues dans la question gagnante: une
            # demande que l'utilisateur n'a pas vue ne doit jamais pouvoir
            # etre autorisee par sa reponse.
            "demandes": [
                {**d, "chips": _chips_propres(d.get("chips"), garder_option=True)}
                for d in compo.demandes
            ] if par_demande else [],
            "faits_rendus": faits,
            "raw_markers": marqueurs,
            "lecture_sans_liste": lecture_sans_liste,
            "actions": [
                {"id": a.id, "outil": a.outil, "succes": bool(a.succes),
                 "par_le_code": bool((a.donnees or {}).get("par_le_code"))}
                for a in registre.actions
            ],
        }
        ConversationMessage.objects.create(
            user=user, role="assistant", content=response,
            metadata=_json_sur(metadonnees))

        evenement = {
            "type": "done",
            "response": response,
            "quick_replies": quick_replies,
            "blocks_created": self._crees(registre, "create_block", "created"),
            "tasks_created": self._crees(registre, "create_task", "task"),
            "raisonnement": raisonnement,
            "question_posee": question_posee,
            "question": question_affichee,
            "question_motif": motif or "",
        }
        if formulaire:
            evenement["interactive_inputs"] = formulaire
            # Mesure: le denominateur du taux de remplissage. Une ligne par
            # formulaire AFFICHE, donc par tour, meme si le modele en a
            # presente plusieurs (seul le dernier atteint l'utilisateur).
            logger.info(
                "formulaire presente user=%s champs=%d types=%s",
                user.id, len(formulaire),
                ",".join(str(champ.get("type", "")) for champ in formulaire),
            )
        yield evenement

    @staticmethod
    def _evenement_de_file(element) -> dict:
        # La file transporte deux formes: un fragment de raisonnement (texte
        # nu) et un appel d'outil (dictionnaire deja pret).
        genre, charge = element
        return ({"type": genre, "text": charge} if isinstance(charge, str)
                else {"type": genre, **charge})

    def _vider_file(self):
        """Emet ce que la file contient deja, sans attendre."""
        if self._file_pensees is None:
            return
        while True:
            try:
                element = self._file_pensees.get_nowait()
            except queue.Empty:
                return
            if element is not None:
                yield self._evenement_de_file(element)

    def _agir_en_fond(self, user: User, message_enrichi: str, registre: Registre):
        """AGIR dans le pool, ses pensees streamees. Rend (raisonnement, panne).

        AGIR tourne dans un THREAD pour qu'on puisse emettre pendant qu'il
        travaille. Mesure du 2026-08-28: sur une demande multi-etapes il
        occupe 15 s des 25 s du tour, et l'utilisateur n'avait rien a lire
        pendant ce temps. Le raisonnement etait bien capte, mais emis apres
        coup: il decrivait une reflexion deja terminee.
        """
        raisonnement, panne = "", None
        file_agir = self._file_pensees

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
                file_agir.put(None)  # sentinelle de fin

        futur = _POOL_AGIR.submit(travailler)
        fragments = 0
        while True:
            try:
                element = file_agir.get(timeout=ATTENTE_PENSEE)
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
            yield self._evenement_de_file(element)
        futur.result()  # remonte une panne du pool lui-meme, pas d'AGIR
        self._file_pensees = None

        # Repli pour les fournisseurs qui ne streament pas leurs deltas: sans
        # lui, leur raisonnement n'atteindrait le client que dans la charge
        # utile finale, et le volet resterait vide tout le tour. On perd le
        # gain de latence, jamais l'information.
        if raisonnement and not fragments:
            yield {"type": "thinking", "text": raisonnement}
        return raisonnement, panne

    @staticmethod
    def _tour_decide(registre: Registre, message: str) -> bool:
        """Le code a-t-il tout decide ce tour (D6) ? Faux au moindre doute."""
        try:
            decide = _charger_tour_decide()
            return bool(decide(registre, message)) if callable(decide) else False
        except Exception:  # noqa: BLE001 - dans le doute, AGIR tourne
            logger.error("Decision du code illisible", exc_info=True)
            return False

    def _choisir_question(self, user: User, message: str, attachment,
                          registre: Registre, attachment_traite_ce_tour: bool,
                          reemises=(), sans_forcee: bool = False):
        """La question UNIQUE du tour, selon PRIORITE, ou None.

        Rend {"source", "motif", "question", "chips", "demandes",
        "cles_posees", "interactive_inputs"}. Les demandes viennent du
        registre (gardes du code, present_choices), dans l'ordre des actions,
        puis des demandes reemises apres une reponse floue (une cle deja au
        registre n'est pas doublee). Une demande abandonnee par le code (D2)
        n'est jamais reposee: elle a sa ligne dans les faits. `sans_forcee`:
        sur le chemin rapide, aucune question forcee ne s'ajoute a la
        decision du code.
        """
        demandes = [a.donnees["demande"] for a in registre.actions
                    if isinstance((a.donnees or {}).get("demande"), dict)
                    and not _abandonnee(a)]
        deja = {d.get("cle") for d in demandes}
        demandes += [d for d in reemises or [] if d.get("cle") not in deja]
        formulaire = self._dernier_formulaire(registre)
        rang_formulaire = PRIORITE.index("formulaire")

        if demandes:
            haut = min(_rang(d.get("motif")) for d in demandes)
            if not formulaire or haut < rang_formulaire:
                code = self._question_des_demandes(demandes)
                if code:
                    return code

        if formulaire:
            return {"source": "formulaire", "motif": "formulaire", "question": "",
                    "chips": [], "demandes": [], "cles_posees": [],
                    "interactive_inputs": formulaire}

        if sans_forcee:
            return None
        try:
            forcee = _charger_question_forcee()(
                user, message, attachment, registre, attachment_traite_ce_tour)
        except Exception:  # noqa: BLE001 - un bouton ne fait jamais tomber un tour
            logger.error("Question forcee indisponible", exc_info=True)
            forcee = None
        if isinstance(forcee, dict):
            chips = _chips_propres(forcee.get("chips"), garder_option=False)
            texte = (forcee.get("question") or "").strip()
            if texte or chips:
                return {"source": "forcee", "motif": forcee.get("motif") or "creneaux",
                        "question": texte, "chips": chips, "demandes": [],
                        "cles_posees": [], "interactive_inputs": None}
        return None

    @staticmethod
    def _question_des_demandes(demandes: list[dict]):
        texte, chips, cles = question_code(demandes)
        texte = (texte or "").strip()
        chips = _chips_propres(chips, garder_option=True)
        if not texte and not chips:
            return None
        cles = [c for c in (cles or []) if c]
        rendues = set(cles)
        posees: list[dict] = []
        vues: set = set()
        for d in demandes:
            cle = d.get("cle")
            if cle in rendues and cle not in vues:
                vues.add(cle)
                posees.append({**d, "chips": chips})
        if posees:
            motif = posees[0].get("motif") or ""
        else:
            motif = min(demandes, key=lambda d: _rang(d.get("motif"))).get("motif") or ""
        return {"source": "demande", "motif": motif, "question": texte,
                "chips": chips, "demandes": posees, "cles_posees": cles,
                "interactive_inputs": None}

    def _deux_derniers_echanges(self, user: User) -> str:
        """Les quatre derniers messages avant celui-ci, courts, pour DIRE."""
        try:
            lignes = list(ConversationMessage.objects
                          .filter(user=user).exclude(pk=self._exclu)
                          .order_by("-created_at", "-pk")[:4])
        except Exception:  # noqa: BLE001 - un contexte absent ne casse pas un tour
            logger.debug("Historique court illisible", exc_info=True)
            return ""
        sortie = []
        for ligne in reversed(lignes):
            qui = "Utilisateur" if ligne.role == "user" else "Assistant"
            contenu = " ".join((ligne.content or "").split())
            if len(contenu) > ECHANGE_MAX:
                contenu = f"{contenu[:ECHANGE_MAX]}..."
            sortie.append(f"{qui}: {contenu}")
        return "\n".join(sortie)

    @staticmethod
    def _journaliser_reponse_formulaire(user: User, message: str) -> None:
        """Mesure: numerateur (repondu) et abandon explicite (passe) du
        formulaire. Les deux phrases sont celles que le frontend envoie
        (InteractiveInputs.tsx, ChatContainer.tsx)."""
        propre = (message or "").strip()
        if propre.startswith("Voici mes réponses"):
            logger.info("formulaire repondu user=%s", user.id)
        elif propre == "On verra ça plus tard, continuons sans formulaire.":
            logger.info("formulaire passe user=%s", user.id)

    @staticmethod
    def _dernier_formulaire(registre: Registre):
        """Le dernier formulaire presente avec succes, releve du registre.

        Le prompt d'AGIR recommande present_form pour demander plusieurs
        infos d'un coup, et la vue ne relaie que la cle interactive_inputs
        du resultat. Sans ce releve, v2 presenterait des formulaires que
        l'utilisateur ne verrait jamais."""
        for a in reversed(registre.actions):
            if a.outil == "present_form" and a.succes:
                return (a.donnees or {}).get("interactive_inputs")
        return None

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
                continue
            texte = ligne.content
            # Les boutons ne sont plus colles au texte: sans cette ligne, le
            # modele ne saurait pas que « 15 h 50 à 17 h 20 » repond a un choix
            # qu'il a lui-meme propose au tour precedent.
            proposes = (ligne.metadata or {}).get("quick_replies") or []
            libelles = [str(c.get("label")) for c in proposes
                        if isinstance(c, dict) and c.get("label")]
            if libelles:
                texte = f"{texte}\n[Choix proposés : {' | '.join(libelles)}]"
            messages.append(ModelResponse(parts=[TextPart(content=texte)]))
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
            return f"{faits}\n\n{REPLI_PROSE}"
        return REPLI_QUESTION

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
