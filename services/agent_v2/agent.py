"""
La boucle unique (2026-09-29): COMPRENDRE+OUTILLER, RECONCILIER, RENDRE.

La difference de fond avec v1 tient en une phrase: le recit d'action n'est
plus produit par le modele. La boucle outille et alimente un registre ecrit
par le runtime; le code rend un compte rendu factuel depuis ce registre; la
prose du modele ne survit que contre des refs verifiees (verifier_prose), et
toute affirmation d'action qui n'existe pas est ecartee avant l'assemblage.

Depuis le 2026-09-14 (lots 1 a 3), un tour s'affiche en trois sections
streamees dans leur ordre final: FAITS (code), PROSE (boucle verifiee),
QUESTION (une seule, choisie par PRIORITE). done.response est exactement la
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
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Optional
from types import SimpleNamespace

from django.conf import settings
from django.contrib.auth.models import User
from django.db import close_old_connections
from django.db.models import F
from django.utils import timezone
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (ModelRequest, ModelResponse, PartDeltaEvent,
                                  PartStartEvent, TextPart,
                                  ThinkingPart, ThinkingPartDelta, UserPromptPart)
from pydantic_ai.usage import UsageLimits

from core.models import BudgetJetonsJournalier, ConversationMessage, UploadedDocument
from services.agent_v2 import lecture, regles
from services.agent_v2 import memoire as memoire_prefs
from services.agent_v2 import jugement as _jugement
from services.agent_v2.mesure import verifier_prose
from services.agent_v2.modeles import (REGLAGES_BOUCLE_SANS_RAISONNEMENT,
                                       modele_agir)
from services.agent_v2.outils import outils_pour
from services.agent_v2.prompts import prompt_agir
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


# ── Garde-fous de disponibilite et de cout ────────────────────────────────
#
# Deux risques documentes mais non bornes: un tour sans deadline murale
# (397 s mesures en prod le 2026-08-29) et une facture LLM sans plafond.
# La philosophie reste la meme que budget_epuise: le tour est tronque,
# jamais rate. Le registre survit, les faits deja vrais sont rendus.


def _delai_tour() -> float:
    """Duree murale maximale d'un tour (AGIR + DIRE), en secondes."""
    try:
        return max(1.0, float(getattr(settings, "AGENT_V2_DELAI_TOUR", 180.0)))
    except (TypeError, ValueError):
        return 180.0


def _budget_jetons_agir() -> int:
    try:
        return max(0, int(getattr(settings, "AGENT_V2_BUDGET_JETONS_AGIR", 300000)))
    except (TypeError, ValueError):
        return 300000


def _budget_jetons_dire() -> int:
    try:
        return max(0, int(getattr(settings, "AGENT_V2_BUDGET_JETONS_DIRE", 100000)))
    except (TypeError, ValueError):
        return 100000


def _budget_jetons_jour(user=None) -> int:
    """Plafond journalier de jetons pour l'utilisateur.

    Les comptes anonymes (mode sans inscription) ont un plafond resserre
    (anti-abus) : quand il est epuise, le tour suggere la creation d'un
    compte pour continuer. Un profil manquant vaut compte normal (direction
    prudente : on ne resserre jamais par defaut).
    """
    from core.anonyme import est_anonyme
    if user is not None and est_anonyme(user):
        try:
            return max(0, int(getattr(settings, "AGENT_V2_BUDGET_JETONS_JOUR_ANON", 400000)))
        except (TypeError, ValueError):
            return 400000
    try:
        return max(0, int(getattr(settings, "AGENT_V2_BUDGET_JETONS_JOUR", 2000000)))
    except (TypeError, ValueError):
        return 2000000


def _limites_jetons(budget: int) -> dict:
    """Parametres UsageLimits pour un budget jetons (total = entree + sortie).
    Un budget nul ou negatif desactive la garde plutot qu'imposer un plafond
    de zero qui tronquerait chaque tour."""
    if budget > 0:
        return {"total_tokens_limit": budget}
    return {}


def _jetons_phase(cout: dict) -> int:
    """Jetons consommes par une phase, depuis son dict _cout."""
    if not isinstance(cout, dict):
        return 0
    return sum(int(cout.get(k, 0) or 0) for k in ("entree", "sortie", "raisonnement"))


def _budget_jour_epuise(user) -> bool:
    """Le compteur journalier de l'utilisateur a-t-il atteint le plafond ?

    Un compteur illisible (table absente, DB en vrac) ne bloque jamais un
    tour: la garde est un coupe-circuit, pas un verrou.
    """
    if _budget_jetons_jour(user) <= 0:
        return False
    try:
        ligne = BudgetJetonsJournalier.objects.filter(
            user=user, jour=timezone.localdate()).first()
    except Exception:  # noqa: BLE001
        return False
    return ligne is not None and (ligne.jetons or 0) >= _budget_jetons_jour(user)


def _enregistrer_jetons(user, total: int) -> None:
    """Ajoute les jetons du tour au compteur journalier (increment atomique)."""
    if total <= 0:
        return
    ligne, _ = BudgetJetonsJournalier.objects.get_or_create(
        user=user, jour=timezone.localdate(), defaults={"jetons": 0})
    BudgetJetonsJournalier.objects.filter(pk=ligne.pk).update(
        jetons=F("jetons") + int(total))

# Une phrase TERMINEE dans un flux de texte: ponctuation finale, guillemets
# fermants eventuels, puis une espace. Tant que l'espace n'est pas arrivee,
# la phrase peut encore s'allonger (« 18 h. » contre « 18 h.30 »).
_FIN_DE_PHRASE_STREAMEE = re.compile(r"[.!?…][\"'»)\]]*\s")

# Une question par tour, la premiere presente dans cet ordre. Les gardes du
# code passent avant tout: une suppression retenue sans sa question serait un
# refus muet. Le formulaire passe avant le choix du modele, et DIRE ferme la
# marche.
PRIORITE = [
    "portee_jour", "destructif", "heure_refusee", "creation_en_masse",
    "optimisation", "formulaire", "choix_modele", "question_libre", "chevauchement",
    "fin_recurrence", "creneaux", "dire",
]
CHIPS_MAX = 4

REPLI_PROSE = "Voici ce qui a changé. Dis-moi si tu veux autre chose."
REPLI_PROSE_LECTURE = "Dis-moi si tu veux autre chose."
REPLI_QUESTION = "Je n'ai pas compris. Tu veux ajouter, déplacer ou voir quelque chose ?"
PROSE_FORMULAIRE = "Il me manque quelques précisions."
PROSE_REPRISE = "Il me faut juste cette précision avant de toucher à ton horaire."
PROSE_ANNULEE = "D'accord, je ne change rien."
PROSE_DELAI = ("J'ai mis trop de temps à te répondre, je m'arrête ici. "
               "Ce qui est déjà fait est affiché plus haut.")
PROSE_BUDGET_JOUR = ("J'ai atteint ma limite d'IA pour aujourd'hui, "
                     "je ne peux pas traiter ça maintenant. Réessaie demain.")
# Compte anonyme au plafond resserre : on suggere la sortie, pas l'attente.
PROSE_BUDGET_JOUR_ANON = ("J'ai atteint ma limite d'IA pour aujourd'hui. "
                          "Crée un compte gratuit pour continuer sans limite.")

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


def _charger_creneaux_types():
    """boutons.creneaux_types (regle creneaux), charge par une fonction pour les tests."""
    from services.agent_v2.boutons import creneaux_types
    return creneaux_types


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


def _chips_reponse(chips, demandes) -> list[dict]:
    """Les quick replies envoyees au front, avec le postback des demandes.

    Une puce qui appartient a une demande RENDUE porte `demande` (la cle) et
    `option` (l'id): au tap, le front les renvoie tels quels et l'egalite
    d'identifiants remplace la comparaison de texte. Champs additifs: v1,
    le MCP et un front ancien les ignorent sans rien casser.
    """
    propres = _chips_propres(chips, garder_option=False)
    par_texte: dict[tuple, tuple] = {}
    for d in demandes or []:
        cle = (d or {}).get("cle") if isinstance(d, dict) else None
        if not cle:
            continue
        for chip in d.get("chips") or []:
            if not isinstance(chip, dict) or chip.get("option") is None:
                continue
            ref = (sans_tiret_long(str(chip.get("label") or "")),
                   sans_tiret_long(str(chip.get("value") or "")))
            par_texte.setdefault(ref, (cle, chip["option"]))
    for propre in propres:
        info = par_texte.get((propre["label"], propre["value"]))
        if info:
            propre["demande"], propre["option"] = info
    return propres


def _rang(motif) -> int:
    # Un motif inconnu est traite comme informatif, au rang du chevauchement.
    return PRIORITE.index(motif) if motif in PRIORITE else PRIORITE.index("chevauchement")


class PlannerAgentV2:
    """Le nom est fixe par benchmarks/harness.py, qui l'importe tel quel."""

    # La vue chat ne passe `tap` (postback structure d'une puce) qu'aux
    # agents qui l'annoncent: v1 ne le connait pas et ne doit pas le recevoir.
    accepte_tap = True

    def __init__(self, user: Optional[User] = None):
        self.user = user
        # Poses ici et pas seulement dans le flux: _boucle et _historique sont
        # appelables directement (banc, tests), et une instance a demi
        # initialisee leverait un AttributeError loin de sa cause.
        self._tache: str = ""
        self._exclu: Optional[int] = None
        self._file_pensees: Optional[queue.Queue] = None
        self._message_brut: Optional[str] = None
        self._tap: Optional[dict] = None
        self._registre_courant: Optional[Registre] = None

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
        """Capte le RAISONNEMENT au fil de sa production, et le TEXTE final.

        run_sync execute le graphe en entier, donc tous les outils tournent;
        run_stream_sync s'arreterait a la premiere sortie « finale » et
        sauterait les appels suivants. Le handler est le seul moyen d'avoir
        les deux.

        Deux defauts corriges le 2026-09-14: la version precedente lisait
        `content_delta` sur tout evenement, donc les brouillons en francais
        de la boucle (TextPartDelta) se melaient au raisonnement anglais du
        volet; et elle ignorait PartStartEvent, qui porte le PREMIER fragment
        de chaque partie (d'ou « user wants » sans « The »).

        Boucle unique (2026-09-29): seul le raisonnement est streame en
        direct. Le texte final est une sortie structuree (JSON): il n'est
        jamais pousse tel quel, la reponse verifiee part en deltas ordonnes
        apres composition.
        """
        async for evenement in evenements:
            if isinstance(evenement, PartStartEvent):
                if isinstance(evenement.part, ThinkingPart):
                    self.pousser_pensee(evenement.part.content)
            elif isinstance(evenement, PartDeltaEvent):
                if isinstance(evenement.delta, ThinkingPartDelta):
                    self.pousser_pensee(evenement.delta.content_delta)

    def _boucle(self, user: User, message: str, registre: Registre) -> ReponseDire | None:
        """Boucle unique (2026-09-29): UN SEUL appel modele par tour.

        Le modele comprend, outille et rend une reponse STRUCTUREE
        (ReponseDire): prose + refs vers le registre + lecture typee pour les
        regles du code. Plus d'etapes LIRE ni DIRE separees.

        Le registre est alimente par l'adaptateur d'outils a chaque execution:
        cette methode ne l'ecrit pas elle-meme, et c'est voulu. Une action ne
        peut entrer dans le registre qu'en ayant reellement ete executee.
        """
        # Les regles de la garde (confirmation, heure dite, portee d'un jour)
        # lisent le message TAPE, jamais sa version enrichie du document.
        brut = self._message_brut if self._message_brut is not None else message
        outils = outils_pour(user, registre, message_du_tour=message,
                             tache=self._tache, signaler=self.signaler_outil,
                             message_brut=brut, tap=self._tap)
        # instructions= et non system_prompt=: pydantic-ai n'ajoute les system
        # prompts QUE si l'historique est vide. Sur tout tour de suivi, la
        # boucle tournait sans date, sans table de decision ni semaine type,
        # et a place une revision en 2025 (banc du 2026-09-14). Les
        # instructions partent a chaque requete.
        agent = Agent(
            modele_agir(),
            instructions=prompt_agir(user),
            tools=outils,
            output_type=ReponseDire,
            # DeepSeek refuse tool_choice=required (la sortie structuree)
            # en mode thinking: mesure du 2026-08-24, gardee pour la boucle
            # qui rend elle aussi une sortie structuree.
            model_settings=REGLAGES_BOUCLE_SANS_RAISONNEMENT,
        )

        depart = time.perf_counter()
        try:
            resultat = self._executer_boucle(
                agent, message,
                message_history=self._historique(user),
                usage_limits=UsageLimits(request_limit=BUDGET_ETAPES,
                                         **_limites_jetons(_budget_jetons_agir())),
                event_stream_handler=self._sur_evenements,
            )
            self._cout_boucle = _cout(resultat, time.perf_counter() - depart)
        except UsageLimitExceeded:
            # Le tour est tronque, pas rate: les outils deja executes ont
            # ecrit. Le bloc factuel le dira, c'est tout l'interet du registre.
            # Declenche par le budget d'etapes OU le budget jetons.
            registre.budget_epuise = True
            self._cout_boucle = {"etapes": BUDGET_ETAPES, "entree": 0, "sortie": 0,
                                 "duree": time.perf_counter() - depart}
            return None
        sortie = getattr(resultat, "output", None)
        return sortie if isinstance(sortie, ReponseDire) else None

    @staticmethod
    def _executer_boucle(agent, message, **kwargs):
        """La boucle, avec UNE reprise bornee sur stream vide.

        Defaut observe en prod le 2026-09-29: un fournisseur rend un stream
        vide (« Streamed response ended without content or tool calls ») et
        le tour meurt sans action. Un seul reessai, pas d'acharnement: la
        resilience reste assuree par la chaine de repli (FallbackModel).
        """
        try:
            return agent.run_sync(message, **kwargs)
        except Exception as e:  # noqa: BLE001
            if "without content or tool calls" not in str(e):
                raise
            logger.warning("Boucle: stream vide, une reprise bornee")
            return agent.run_sync(message, **kwargs)

    def process_message_stream(
        self,
        user: User,
        message: str,
        attachment: Optional[UploadedDocument] = None,
        *,
        use_streaming: bool = True,
        generate_quick_replies: bool = False,
        tap: Optional[dict] = None,
    ):
        """Contrat SSE additif: status, thinking, tool, delta, done. Les deltas
        arrivent dans l'ordre final (faits, prose, question) et done.response
        en est exactement la concatenation.

        `tap` est le postback structure d'une puce touchee:
        {"demande": cle, "option": id}. Il ne vaut que contre la demande en
        attente qui porte cette cle (memes gardes, meme fenetre); tout le
        reste du tour lit le message comme avant."""
        self.user = user
        depart_tour = time.perf_counter()
        self._depart_tour = depart_tour  # lu par _delai_restant (deadline du tour)
        self._cout_boucle = {}
        # Le message TAPE, avant tout enrichissement: c'est lui que lisent la
        # garde et les boutons forces.
        self._message_brut = message
        self._tap = tap if isinstance(tap, dict) and tap.get("demande") and tap.get("option") else None
        self._journaliser_reponse_formulaire(user, message)

        # Persiste d'abord, puis exclut CETTE ligne de l'historique par son id.
        # v1 devait s'en remettre a un filet (B9: message sauve, relu, puis
        # rajoute, donc duplique a chaque requete); ici la duplication est
        # structurellement impossible.
        courant = ConversationMessage.objects.create(
            user=user, role="user", content=message,
            metadata={"tap": self._tap} if self._tap else {})
        self._exclu = courant.pk
        # Identifie CE tour pour les cles d'idempotence: deux tours
        # distincts peuvent legitimement refaire la meme action, un meme
        # tour rejoue ne doit l'executer qu'une fois.
        self._tache = f"{user.pk}:{courant.pk}"

        registre = Registre()
        self._registre_courant = registre

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

        # Commande memoire deterministe (memoire.py): « souviens-toi que... »,
        # « oublie... », « que sais-tu de moi ? ». Executee par le code AVANT
        # AGIR, comme l'import: memoriser, oublier ou lister ne demande aucun
        # appel au modele. Le resultat entre au registre pour que DIRE
        # l'annonce; AGIR ne recoit que le reste eventuel du message.
        # Un tap structure (chip d'une question en attente) n'est jamais une
        # commande memoire.
        try:
            commande_memoire = (memoire_prefs.interpreter(message)
                                if self._tap is None else None)
        except Exception:  # noqa: BLE001 - une commande illisible ne casse pas le tour
            commande_memoire = None
            logger.error("Interpretation memoire impossible", exc_info=True)
        if commande_memoire is not None:
            try:
                from services.agent.tools.base import ToolResult
                if commande_memoire.genre == "memoriser":
                    phrase = memoire_prefs.memoriser(user, commande_memoire.enonce)
                elif commande_memoire.genre == "oublier":
                    phrase = memoire_prefs.oublier(user, commande_memoire.enonce)
                else:
                    phrase = memoire_prefs.lister(user)
                registre.ajouter("memoire", {"commande": commande_memoire.genre},
                                 ToolResult(success=True, message=phrase))
                if commande_memoire.pure and message_enrichi == message:
                    # Rien d'autre a faire ce tour: AGIR ne doit pas relire
                    # une commande memoire comme une demande de planification.
                    message_enrichi = ("(Commande memoire deja executee par le "
                                       "code: aucune action de planification.)")
                elif commande_memoire.reste:
                    message_enrichi = commande_memoire.reste
            except Exception:  # noqa: BLE001 - le tour continue sans la memoire
                logger.error("Commande memoire non executee", exc_info=True)

        # La reponse a une question du tour precedent s'execute par le CODE,
        # avant AGIR: un tap sur « Tous les jeudis » supprime la serie sans
        # qu'un modele ait a le refaire (ni a pouvoir le rater). AGIR recoit
        # le bilan pour ne pas le rejouer; le message brut reste intact.
        self._file_pensees = queue.Queue()
        choix: list = []
        try:
            # `tap` seulement s'il existe: les simulations de tests et tout
            # remplacant sans ce parametre restent valides.
            extra_tap = {"tap": self._tap} if self._tap else {}
            choix = list(_charger_appliquer_choix()(
                user, registre, message, tache=self._tache,
                signaler=self.signaler_outil, **extra_tap) or [])
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
        # Compteur journalier epuise: on saute les phases LLM (LIRE, AGIR,
        # DIRE). Le tour continue avec le registre tel quel: les choix du
        # code et les faits restent rendus, sans couter un jeton de plus.
        budget_jour_epuise = not par_le_code and _budget_jour_epuise(user)
        if budget_jour_epuise:
            logger.warning("agent_v2 budget jetons journalier epuise user=%s", user.pk)
        # Voie rapide sociale: une salutation ou un remerciement ne merite
        # pas la boucle lourde (15k tokens de contexte). Le juge semantique
        # tranche, jamais une liste de mots. La reponse breve entre dans un
        # ReponseDire pour suivre le chemin de composition normal.
        voie_rapide = (
            not par_le_code and not budget_jour_epuise
            and self._voie_rapide_sociale(user, message, attachment)
        )
        if voie_rapide:
            texte_rapide = self._reponse_rapide(user, message)
            if texte_rapide:
                reponse_boucle = ReponseDire(ouverture=texte_rapide)
                logger.info("agent_v2 voie rapide sociale user=%s", user.pk)
            else:
                voie_rapide = False
        # Boucle unique: un seul appel modele par tour (plus de LIRE ni de
        # DIRE separes). Le chemin rapide et le budget epuise sautent la
        # boucle; le registre (choix du code, faits) est rendu tel quel.
        reponse_boucle: ReponseDire | None = reponse_boucle if voie_rapide else None
        panne = None
        raisonnement = ""
        if par_le_code or budget_jour_epuise or voie_rapide:
            self._file_pensees = None
        else:
            yield {"type": "status", "text": "Réflexion..."}
            reponse_boucle, panne, raisonnement = yield from self._boucle_en_fond(
                user, message_enrichi, registre)
        if panne is not None:
            # Une panne de la boucle ne doit pas effacer ce que les outils
            # ont deja ecrit: le registre survit et le tour continue.
            logger.error("Boucle a echoue: %s", panne, exc_info=panne)

        etat: dict = {}
        if registre.mutations():
            yield {"type": "status", "text": "Je relis ton planning..."}
            etat = reconcilier(user, registre)
            detecter_ecarts(registre)

        # La lecture typee vient de la boucle elle-meme (champ `lecture` de
        # sa reponse structuree): plus d'appel LIRE separe. Elle alimente les
        # deux regles qui en ont besoin (formulaire_cours, creneaux).
        configurees = frozenset() if par_le_code else lecture.regles_actives()
        lecture_typee = (self._lecture_de_boucle(user, message, reponse_boucle)
                         if configurees else None)
        actives = configurees if lecture_typee is not None else frozenset()
        regle, prose_regle = "-", ""
        if regles.FORMULAIRE_COURS in actives:
            try:
                prose_regle = regles.appliquer_formulaire_cours(
                    lecture_typee, registre, attachment=attachment, reemises=reemises)
            except Exception as e:  # noqa: BLE001 - une regle ne casse pas un tour
                logger.warning("agent_v2 regle=%s illisible erreur=%s",
                               regles.FORMULAIRE_COURS, type(e).__name__)
                prose_regle = ""
            if prose_regle:
                regle = regles.FORMULAIRE_COURS

        # La question du tour est choisie AVANT les faits: les actions
        # retenues que cette question couvre n'ont pas a etre redites, les
        # autres recoivent leur ligne « pas encore ».
        gagnant = self._choisir_question(
            user, message, attachment, registre, attachment_traite_ce_tour,
            reemises=reemises, sans_forcee=par_le_code,
            lecture_typee=lecture_typee if regles.CRENEAUX in actives else None)
        if gagnant and gagnant.get("regle"):
            regle = gagnant["regle"]
        par_demande = bool(gagnant) and gagnant.get("source") == "demande"
        cles_posees = set(gagnant.get("cles_posees") or []) if par_demande else set()

        # Une lecture qui n'a servi qu'a preparer un formulaire ou un choix ne
        # se deverse pas au-dessus de la question (banc du round 4, s02-1).
        sans_lecture = bool(gagnant) and (
            gagnant.get("source") == "formulaire" or gagnant.get("motif") in ("choix_modele", "question_libre")
        ) and not any(a.succes and a.est_mutation for a in registre.actions)
        titres_vises = [((d or {}).get("cible") or {}).get("titre")
                        for d in (gagnant.get("demandes") or [])] if par_demande else []
        faits = bloc_factuel(registre, cles_posees=cles_posees, sans_lecture=sans_lecture,
                             titres_vises=[t for t in titres_vises if t])
        # La section RESTE: demande contre place, une soustraction rendue par
        # du code. Elle rejoint les faits AVANT la redaction et le flux: le
        # manque se nomme au meme instant que le succes qu'il tempere.
        reste = bloc_reste(message, registre)
        if reste:
            faits = f"{faits}\n{reste}" if faits else reste
        faits = sans_tiret_long(faits or "")

        # Les faits partent AVANT la redaction: ils sont deja vrais, et
        # l'utilisateur n'a pas a attendre l'enrobage pour les voir.
        emis: list[str] = []
        if faits:
            emis.append(faits)
            yield {"type": "delta", "text": faits}

        supprimees = 0
        fuites: list[str] = []
        # Regle formulaire_cours: le code parle seul (pas de prose generee).
        # Chemin rapide et budget journalier epuise: meme traitement.
        t0_verif = time.perf_counter()
        if par_le_code or prose_regle or budget_jour_epuise:
            compo = composer(None, registre, faits, gagnant)
        elif reponse_boucle is not None:
            # Verifier-puis-rendre: la prose de la boucle ne survit que contre
            # des refs verifiees (ids d'actions reelles du registre). Sur un
            # tour de reprise, le code parle seul (PROSE_REPRISE plus bas).
            brut = (reponse_boucle.model_copy(update={"ouverture": "", "suite": ""})
                    if reemises else reponse_boucle)
            brut, supprimees, fuites = verifier_prose(brut, registre)
            compo = composer(brut, registre, faits, gagnant)
        else:
            # La boucle n'a rien rendu (panne, budget d'etapes): les faits
            # parlent, et le repli aussi (plus bas).
            compo = composer(None, registre, faits, gagnant)
        duree_verif = time.perf_counter() - t0_verif

        # Zero tiret long dans ce que lit l'utilisateur, quelle que soit la
        # source (banc du round 3, s06-1).
        prose, question = sans_tiret_long(compo.prose), sans_tiret_long(compo.question)
        motif, chips = compo.motif, compo.chips
        if prose_regle:
            prose = sans_tiret_long(prose_regle)
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
        if panne is not None and reponse_boucle is None and faits and not prose:
            # La boucle est tombee. Se taire laisserait croire que rien n'a
            # eu lieu, alors que le planning a peut-etre change.
            if registre.delai_depasse:
                prose = PROSE_DELAI
            else:
                prose = REPLI_PROSE if mutation_reussie else REPLI_PROSE_LECTURE
        if formulaire and not faits and not prose:
            prose = PROSE_FORMULAIRE
        if budget_jour_epuise and not faits and not prose and not question and not formulaire:
            # Le compteur journalier est epuise et le tour n'a rien produit:
            # on le dit plutot que de laisser un silence. Un compte anonyme
            # au plafond resserre se voit proposer la sortie (creer un
            # compte), pas l'attente.
            from core.anonyme import est_anonyme
            if est_anonyme(user):
                prose, motif = PROSE_BUDGET_JOUR_ANON, "budget_jour"
            else:
                prose, motif = PROSE_BUDGET_JOUR, "budget_jour"
        elif not faits and not prose and not question and not formulaire:
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

        quick_replies = _chips_reponse(chips, compo.demandes if par_demande else [])
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

        # Recueillie avant la question quand une regle est configuree; sinon
        # ici seulement, une fois la reponse figee. La ligne du tour nomme la
        # regle qui a decide.
        metadonnees_lire = {"lecture": "boucle"}

        # Une seule ligne par tour, mais pas toujours au meme niveau: une
        # reference rejetee est un mensonge que la garantie structurelle vient
        # d'attraper, et une fuite est une affirmation d'action sans recu.
        # Ce sont LES deux signaux du projet; en INFO ils se noieraient dans
        # le bruit et personne ne les verrait passer.
        cout = getattr(self, "_cout_boucle", None) or {}
        anormal = bool(rejetees or fuites)
        logger.log(
            logging.WARNING if anormal else logging.INFO,
            "agent_v2 tour actions=%d rejetees=%d fuites=%d supprimees=%d ecarts=%d%s"
            " boucle=%.1fs/%dep/%d->%dj/r%d/c%d verif=%.2fs"
            " asked=%d form=%d choices=%d read_without_list=%d redites=%d"
            " raw_marker_count=%d"
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
            cout.get("duree", 0.0), cout.get("etapes", 0),
            cout.get("entree", 0), cout.get("sortie", 0),
            cout.get("raisonnement", 0), cout.get("cache", 0),
            duree_verif,
            # MESURE DES QUESTIONS (lot 3g): a-t-on demande, par quel canal,
            # combien de boutons, une lecture sans liste, du texte machine.
            1 if question_posee else 0,
            1 if formulaire else 0,
            len(quick_replies),
            1 if lecture_sans_liste else 0,
            compo.redites,
            len(marqueurs),
            motif or "-",
            choix_code,
            # Latence du chemin rapide (D6) contre la boucle complete.
            "code" if par_le_code else "boucle",
            time.perf_counter() - depart_tour,
        )

        question_affichee = "" if motif == "formulaire" else question

        # Compteur journalier (garde-fou facture): son echec ne doit jamais
        # casser un tour qui vient de reussir.
        try:
            total_jetons = _jetons_phase(cout)
            if total_jetons > 0:
                _enregistrer_jetons(user, total_jetons)
        except Exception:  # noqa: BLE001
            logger.error("Compteur de jetons non enregistre", exc_info=True)

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
            "redites": compo.redites,
            # Le nom que l'utilisateur a donne au cours du formulaire du code:
            # au tour de la reponse, create_block le garde (outils.py).
            "formulaire_nom": next((str((a.donnees or {}).get("nom") or "")
                                    for a in registre.actions
                                    if a.outil == "present_form" and a.succes
                                    and (a.donnees or {}).get("par_le_code")), ""),
            "actions": [
                {"id": a.id, "outil": a.outil, "succes": bool(a.succes),
                 "par_le_code": bool((a.donnees or {}).get("par_le_code"))}
                for a in registre.actions
            ],
        }
        metadonnees.update(metadonnees_lire)
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

    def _delai_restant(self) -> float:
        """Secondes restantes avant la deadline du tour.

        Plancher a 1 s: un appel borne doit toujours avoir le temps de
        tenter quelque chose, meme en fin de tour. Hors tour (tests qui
        appellent _dire directement), c'est la deadline entiere.
        """
        depart = getattr(self, "_depart_tour", None)
        if depart is None:
            return _delai_tour()
        return max(1.0, _delai_tour() - (time.perf_counter() - depart))

    def _boucle_en_fond(self, user: User, message_enrichi: str, registre: Registre):
        """La boucle dans le pool, ses pensees streamees.

        Rend (ReponseDire|None, panne, raisonnement): le raisonnement est le
        texte pense collecte pendant le drainage, pour l'evenement done.

        La boucle tourne dans un THREAD pour qu'on puisse emettre pendant
        qu'elle travaille. Mesure du 2026-08-28: sur une demande multi-etapes
        la phase LLM occupe 15 s des 25 s du tour, et l'utilisateur n'avait
        rien a lire pendant ce temps. Le raisonnement etait bien capte, mais
        emis apres coup: il decrivait une reflexion deja terminee.
        """
        reponse, panne = None, None
        pensees: list[str] = []
        file_agir = self._file_pensees
        # Deadline murale du tour: le drainage s'interrompt quand elle est
        # depassee au lieu de tenir la connexion SSE ouverte indefiniment
        # (397 s en prod le 2026-08-29). Le thread orphelin finit seul en
        # tache de fond: ses ecritures hors registre sont sans effet
        # (signaler_outil est neutre sans file, et l'instance est jettee a
        # la fin de la requete).
        echeance = time.monotonic() + self._delai_restant()

        def travailler():
            nonlocal reponse, panne
            # Ce thread vit hors du cycle de requete Django, qui ferme les
            # connexions: on s'en charge des deux cotes.
            close_old_connections()
            try:
                reponse = self._boucle(user, message_enrichi, registre)
            except Exception as e:  # noqa: BLE001
                panne = e
            finally:
                close_old_connections()
                file_agir.put(None)  # sentinelle de fin

        futur = _POOL_AGIR.submit(travailler)
        fragments = 0
        while True:
            restant = echeance - time.monotonic()
            if restant <= 0:
                registre.delai_depasse = True
                panne = panne or TimeoutError("delai du tour depasse pendant la boucle")
                logger.warning("agent_v2 delai depasse pendant la boucle user=%s", user.pk)
                break
            try:
                element = file_agir.get(timeout=min(ATTENTE_PENSEE, restant))
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
            evt = self._evenement_de_file(element)
            if evt["type"] == "thinking":
                pensees.append(evt["text"])
            yield evt
        if not registre.delai_depasse:
            try:
                futur.result(timeout=max(0.1, echeance - time.monotonic()))
            except FuturesTimeoutError:
                # Paranoia: le drainage a vu la sentinelle mais le thread ne
                # rend pas la main. Meme traitement que ci-dessus.
                registre.delai_depasse = True
                panne = panne or TimeoutError(
                    "delai du tour depasse: la boucle ne rend pas la main")
                logger.warning("agent_v2 la boucle ne rend pas la main user=%s", user.pk)
        self._file_pensees = None

        return reponse, panne, "".join(pensees)

    @staticmethod
    def _lecture_de_boucle(user: User, message: str,
                           reponse_boucle: ReponseDire | None):
        """La lecture typee pour les regles, produite par la boucle unique.

        Remplace l'appel LIRE separe: le champ `lecture` de la reponse
        structuree traverse les memes regles (lecture_du_tour), sur une
        preparation faite par le code avec le message TAPE.
        """
        lecture_boucle = getattr(reponse_boucle, "lecture", None)
        if lecture_boucle is None:
            return None
        try:
            preparation = lecture.preparer(user, message)
        except Exception:  # noqa: BLE001
            return None
        suivi = SimpleNamespace(preparation=preparation)
        resultat = SimpleNamespace(lecture=lecture_boucle, statut=lecture.OK)
        try:
            return regles.lecture_du_tour(suivi, resultat)
        except Exception:  # noqa: BLE001 - une lecture ne casse pas un tour
            logger.error("Lecture de boucle illisible", exc_info=True)
            return None

    @staticmethod
    def _tour_decide(registre: Registre, message: str) -> bool:
        """Le code a-t-il tout decide ce tour (D6) ? Faux au moindre doute."""
        try:
            decide = _charger_tour_decide()
            return bool(decide(registre, message)) if callable(decide) else False
        except Exception:  # noqa: BLE001 - dans le doute, AGIR tourne
            logger.error("Decision du code illisible", exc_info=True)
            return False

    def _voie_rapide_sociale(self, user: User, message: str, attachment) -> bool:
        """Le message merite-t-il la voie rapide (simple interaction sociale) ?

        La decision est SEMANTIQUE (juge Jev, question typee), jamais une
        liste de mots ni une regex: la doctrine l'interdit. Les garde-fous
        sont STRUCTURELS (pas d'intention): pas de piece jointe, pas de tap,
        pas de demande en attente, message court. Seuil de confiance eleve
        (0.9): rater une optimisation vaut mieux que rater une vraie demande.
        """
        try:
            # Garde-fous structurels: jamais de voie rapide si le tour a
            # autre chose a traiter qu'un simple message texte court.
            if attachment is not None:
                return False
            if self._tap is not None:
                return False
            if len(message.split()) > 8:
                return False
            from services.agent_v2 import demandes as _demandes
            if _demandes.demandes_en_attente(user):
                return False
            # Le JUGE tranche, pas une liste de mots.
            resultats = _jugement.juger(
                message, {"sociale": _jugement.q_interaction_sociale()})
            rep = (resultats or {}).get("sociale") or {}
            return (
                rep.get("statut") == _jugement.STATUT_DECISION
                and rep.get("valeur") == "oui"
                and float(rep.get("confiance") or 0) >= 0.9
            )
        except Exception:  # noqa: BLE001 - dans le doute, la boucle tourne
            logger.warning("Voie rapide sociale illisible", exc_info=True)
            return False

    def _reponse_rapide(self, user: User, message: str) -> str:
        """Repond a une interaction sociale sans la boucle lourde.

        Prompt minimal, aucun outil, aucun contexte planning: le message
        n'attend qu'une reponse sociale breve. En cas d'echec, chaine vide
        (l'appelant bascule sur la boucle normale).
        """
        try:
            from services.agent_v2.modeles import modele_agir
            agent = Agent(
                modele_agir(),
                instructions=(
                    "Tu es l'assistant Planner, chaleureux et direct, "
                    "tutoiement, francais quebecois. Reponds en UNE phrase "
                    "breve a ce simple message social. Ne parle jamais de "
                    "planning, d'horaire ou de taches: l'utilisateur n'a "
                    "rien demande."
                ),
            )
            resultat = agent.run_sync(message)
            texte = (getattr(resultat, "output", "") or "").strip()
            return texte if isinstance(texte, str) else ""
        except Exception:  # noqa: BLE001 - la boucle normale prend le relais
            logger.warning("Reponse rapide impossible", exc_info=True)
            return ""

    def _choisir_question(self, user: User, message: str, attachment,
                          registre: Registre, attachment_traite_ce_tour: bool,
                          reemises=(), sans_forcee: bool = False, lecture_typee=None):
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
        if lecture_typee is not None:
            # Regle creneaux: la jambe typee passe d'abord; si elle ne rend
            # rien, question_forcee tourne exactement comme sur main.
            try:
                typee = _charger_creneaux_types()(user, message, attachment, registre, lecture_typee)
            except Exception as e:  # noqa: BLE001 - un bouton ne fait jamais tomber un tour
                logger.warning("agent_v2 regle=creneaux illisible erreur=%s", type(e).__name__)
                typee = None
            if isinstance(typee, dict):
                chips = _chips_propres(typee.get("chips"), garder_option=False)
                texte = (typee.get("question") or "").strip()
                if texte or chips:
                    return {"source": "forcee", "motif": typee.get("motif") or "creneaux",
                            "question": texte, "chips": chips, "demandes": [],
                            "cles_posees": [], "interactive_inputs": None,
                            "regle": "creneaux"}
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
        tap: Optional[dict] = None,
    ) -> dict:
        """Enveloppe non streamee: draine le flux, seule source de verite."""
        done: dict = {}
        for event in self.process_message_stream(
            user, message, attachment,
            use_streaming=False,
            generate_quick_replies=generate_quick_replies,
            tap=tap,
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
        rend bien: on delegue plutot que de dupliquer.
        La chip « Mémoriser ? » (memoire inferee) passe devant: un tap
        rejoue la voie explicite, confirmation incluse.
        """
        chips: list[dict] = []
        try:
            chip = memoire_prefs.chip_inference(user, user_message or "")
            if chip:
                chips.append(chip)
        except Exception:  # noqa: BLE001 - une suggestion ne remonte jamais d'erreur
            logger.debug("Chip memoire indisponible", exc_info=True)
        try:
            from services.agent.agent import PlannerAgent
            chips.extend(PlannerAgent().quick_replies_for(
                user, user_message, assistant_response) or [])
        except Exception:  # noqa: BLE001 - une suggestion ne remonte jamais d'erreur
            logger.debug("Suggestions indisponibles", exc_info=True)
        return chips

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
                        if attachment.processing_error:
                            # L'analyse a ECHOUE: inutile d'attendre la borne
                            # de 45 s pour annoncer un "retard" qui n'en est
                            # pas un. Le contexte ci-dessous le dit franchement.
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
