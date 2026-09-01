"""
Reveil silencieux iOS via APNs.

Quand le planning change cote serveur (Planning, agent, MCP, admin), on envoie
un push "background" (content-available: 1, priorite 5) a chaque appareil iOS
de l'utilisateur. L'app se reveille quelques secondes, relit le planning et
pose ses alarmes locales: une alarme iPhone ne peut etre posee que par l'app
sur l'appareil, le serveur ne fait que la reveiller.

Degrade en silence tant que APNS_TEAM_ID / APNS_KEY_ID / APNS_AUTH_KEY ne sont
pas poses. Aucune valeur de secret (cle .p8, jeton) n'est jamais journalisee.
"""
import logging
import threading
import time
from datetime import timedelta

import httpx
import jwt
from django.conf import settings
from django.db import connection
from django.utils import timezone

logger = logging.getLogger(__name__)

HOTE_PRODUCTION = "https://api.push.apple.com"
HOTE_SANDBOX = "https://api.sandbox.push.apple.com"
TIMEOUT_SECONDES = 10

# Apple refuse un jeton de plus de 60 min et n'en veut pas un nouveau plus
# d'une fois par 20 min: garder le meme jeton 50 minutes respecte les deux.
DUREE_JETON_SECONDES = 50 * 60

# Lissage par utilisateur: deux changements a moins de 10 s = un envoi
# immediat + un rattrapage 12 s plus tard, jamais une rafale.
# iOS rationne les pushes silencieux (vecu 2026-09-01: apres six reveils en
# 35 minutes, plus aucun n'a atteint l'app). On regroupe donc plus large:
# un reveil immediat, puis au plus un rattrapage par demi-minute.
LISSAGE_SECONDES = 30
DELAI_RATTRAPAGE_SECONDES = 35
# Reveil confirme ou relance: le telephone renvoie un bilan apres chaque pose
# (POST /telemetrie/alarmes/), qui vaut accuse de reception. Sans accuse dans
# les deux minutes, le filet de send_reminders relance, au plus trois fois par
# demande: un push saute par iOS est rattrape au tick suivant, sans que
# l'utilisateur ouvre l'app.
DELAI_RELANCE_SECONDES = 120
MAX_RELANCES = 3
# Filet de securite (tick send_reminders): une demande vieille de plus de 5 s
# et jamais envoyee (worker redemarre avant son Timer) part a ce moment-la.
DELAI_FILET_SECONDES = 5

_jeton_cache = {"valeur": None, "emis_a": 0.0}
_jeton_verrou = threading.Lock()
_minuteries: dict = {}
_minuteries_verrou = threading.Lock()


def apns_configured() -> bool:
    return bool(settings.APNS_TEAM_ID and settings.APNS_KEY_ID and settings.APNS_AUTH_KEY)


def _horloge() -> float:
    return time.time()


def _cle_privee() -> str:
    # Une variable Railway collee sur une ligne porte des "\n" litteraux.
    return settings.APNS_AUTH_KEY.replace("\\n", "\n").strip() + "\n"


def jeton_apns() -> str:
    """JWT ES256 de l'en-tete authorization, garde en memoire 50 minutes."""
    maintenant = _horloge()
    with _jeton_verrou:
        if _jeton_cache["valeur"] and maintenant - _jeton_cache["emis_a"] < DUREE_JETON_SECONDES:
            return _jeton_cache["valeur"]
        valeur = jwt.encode(
            {"iss": settings.APNS_TEAM_ID, "iat": int(maintenant)},
            _cle_privee(),
            algorithm="ES256",
            headers={"kid": settings.APNS_KEY_ID},
        )
        _jeton_cache["valeur"] = valeur
        _jeton_cache["emis_a"] = maintenant
        return valeur


def invalider_jeton() -> None:
    with _jeton_verrou:
        _jeton_cache["valeur"] = None
        _jeton_cache["emis_a"] = 0.0


# --- Envoi a un appareil ----------------------------------------------------

def _motif(reponse) -> str:
    try:
        return str((reponse.json() or {}).get("reason", ""))
    except ValueError:
        return ""


def _noter_erreur(appareil, texte: str) -> None:
    from core.models import AppareilPush

    AppareilPush.objects.filter(pk=appareil.pk).update(derniere_erreur=texte[:200])


def _poster(appareil, raison: str):
    hote = HOTE_SANDBOX if appareil.environnement == "sandbox" else HOTE_PRODUCTION
    en_tetes = {
        "authorization": f"bearer {jeton_apns()}",
        "apns-topic": settings.APNS_BUNDLE_ID,
        "apns-push-type": "background",
        "apns-priority": "5",
        "apns-expiration": "0",
    }
    corps = {"aps": {"content-available": 1}, "raison": raison}
    with httpx.Client(http2=True, timeout=TIMEOUT_SECONDES) as client:
        return client.post(f"{hote}/3/device/{appareil.token}", headers=en_tetes, json=corps)


def envoyer_reveil(appareil, raison: str = "planning") -> bool:
    """Un push background a un appareil. True si Apple l'a accepte. Ne leve jamais.

    Un BadDeviceToken en production bascule l'appareil en sandbox et reessaie
    une fois (build Xcode = sandbox, TestFlight et App Store = production, on
    ne le sait pas d'avance); en sandbox aussi, l'appareil est supprime.
    """
    if not apns_configured():
        return False
    for tentative in (1, 2):
        try:
            reponse = _poster(appareil, raison)
        except httpx.HTTPError as e:
            logger.warning("APNs injoignable pour l'appareil %s: %s", appareil.pk, e)
            _noter_erreur(appareil, f"reseau: {type(e).__name__}")
            return False
        except Exception as e:  # noqa: BLE001
            logger.error("APNs erreur pour l'appareil %s: %s", appareil.pk, e)
            _noter_erreur(appareil, f"erreur: {type(e).__name__}")
            return False

        statut = reponse.status_code
        if statut == 200:
            if appareil.derniere_erreur:
                _noter_erreur(appareil, "")
            # Une ligne par envoi accepte: c'est la seule trace qu'un reveil
            # est parti (le guide de mise en service la cherche).
            logger.info("APNs reveil accepte: appareil %s, raison %s", appareil.pk, raison)
            return True

        motif = _motif(reponse)
        if statut == 400 and motif == "BadDeviceToken":
            if appareil.environnement != "sandbox" and tentative == 1:
                # update() plutot que save(update_fields): la ligne a pu
                # disparaitre entre-temps (retrait concurrent), et save leverait.
                from core.models import AppareilPush
                AppareilPush.objects.filter(pk=appareil.pk).update(environnement="sandbox")
                appareil.environnement = "sandbox"
                continue
            logger.info("APNs BadDeviceToken en sandbox aussi: appareil %s supprime", appareil.pk)
            appareil.delete()
            return False
        if statut == 410:
            logger.info("APNs Unregistered: appareil %s supprime", appareil.pk)
            appareil.delete()
            return False
        if statut == 403:
            invalider_jeton()
            logger.error(
                "APNs refuse le jeton fournisseur (%s): verifier APNS_TEAM_ID, APNS_KEY_ID et APNS_AUTH_KEY",
                motif,
            )
            _noter_erreur(appareil, f"403 {motif}")
            return False
        logger.warning("APNs %s (%s) pour l'appareil %s", statut, motif, appareil.pk)
        _noter_erreur(appareil, f"{statut} {motif}")
        return False
    return False


# --- Lissage par utilisateur ------------------------------------------------

def _envoyer_a_tous(appareils, raison: str) -> int:
    return sum(1 for appareil in appareils if envoyer_reveil(appareil, raison))


def reveiller_utilisateur(user, raison: str = "planning") -> int:
    """Reveille les appareils d'un utilisateur, au plus un envoi par 10 s.

    Retourne le nombre d'envois acceptes. Un changement qui tombe dans les
    10 s d'un envoi est note (demande_a) et rattrape 12 s plus tard.
    """
    if not apns_configured():
        return 0
    from core.models import AppareilPush, ReveilPlanning

    user_id = getattr(user, "pk", user)
    appareils = list(AppareilPush.objects.filter(user_id=user_id))
    if not appareils:
        return 0

    maintenant = timezone.now()
    reveil, _ = ReveilPlanning.objects.get_or_create(user_id=user_id)
    recent = (
        reveil.envoye_a is not None
        and (maintenant - reveil.envoye_a).total_seconds() < LISSAGE_SECONDES
    )
    if not recent:
        # Reclamer le creneau avant d'envoyer: deux workers qui voient le
        # meme changement n'envoient qu'une fois.
        reclame = ReveilPlanning.objects.filter(pk=reveil.pk, envoye_a=reveil.envoye_a).update(
            demande_a=maintenant, envoye_a=maintenant
        )
        if reclame:
            return _envoyer_a_tous(appareils, raison)
    ReveilPlanning.objects.filter(pk=reveil.pk).update(demande_a=maintenant)
    _armer_rattrapage(user_id)
    return 0


def rattraper_reveil(user_id: int, raison: str = "planning") -> int:
    """Envoie le reveil en attente d'un utilisateur (demande_a > envoye_a), sinon rien.

    Idempotent entre workers: le creneau est reclame par une mise a jour
    conditionnelle avant l'envoi.
    """
    if not apns_configured():
        return 0
    from core.models import AppareilPush, ReveilPlanning

    reveil = ReveilPlanning.objects.filter(user_id=user_id).first()
    if reveil is None or reveil.demande_a is None:
        return 0
    if reveil.envoye_a is not None and reveil.demande_a <= reveil.envoye_a:
        return 0
    reclame = ReveilPlanning.objects.filter(pk=reveil.pk, envoye_a=reveil.envoye_a).update(
        envoye_a=timezone.now()
    )
    if not reclame:
        return 0
    return _envoyer_a_tous(list(AppareilPush.objects.filter(user_id=user_id)), raison)


def confirmer_reveil(user_id: int) -> None:
    """Le telephone a repose ses alarmes: la derniere demande est servie."""
    from core.models import ReveilPlanning

    ReveilPlanning.objects.filter(user_id=user_id).update(confirme_a=timezone.now(), relances=0)


def relancer_reveils_non_confirmes() -> int:
    """Renvoie un reveil aux comptes dont le dernier envoi n'a pas ete confirme.

    Un envoi accepte par Apple peut ne jamais atteindre l'app (budget iOS des
    pushes silencieux). Sans bilan du telephone dans les DELAI_RELANCE_SECONDES
    qui suivent, on relance, au plus MAX_RELANCES fois par demande.
    """
    if not apns_configured():
        return 0
    from django.db.models import F, Q

    from core.models import AppareilPush, ReveilPlanning

    maintenant = timezone.now()
    limite = maintenant - timedelta(seconds=DELAI_RELANCE_SECONDES)
    candidats = ReveilPlanning.objects.filter(
        Q(confirme_a__isnull=True) | Q(confirme_a__lt=F("envoye_a")),
        envoye_a__isnull=False,
        envoye_a__lte=limite,
        relances__lt=MAX_RELANCES,
    )
    envois = 0
    for reveil in list(candidats):
        appareils = list(AppareilPush.objects.filter(user_id=reveil.user_id))
        if not appareils:
            continue
        # Reclamer la relance (idempotent entre workers) avant d'envoyer.
        reclame = ReveilPlanning.objects.filter(pk=reveil.pk, relances=reveil.relances).update(
            envoye_a=maintenant, relances=reveil.relances + 1
        )
        if not reclame:
            continue
        logger.info("APNs relance %d pour user=%s (aucun bilan depuis l'envoi)", reveil.relances + 1, reveil.user_id)
        envois += _envoyer_a_tous(appareils, "planning")
    return envois


def rattraper_tous_les_reveils() -> int:
    """Filet de securite: reveils demandes et pas envoyes, puis envoyes et pas confirmes."""
    if not apns_configured():
        return 0
    from django.db.models import F, Q

    from core.models import ReveilPlanning

    limite = timezone.now() - timedelta(seconds=DELAI_FILET_SECONDES)
    en_attente = ReveilPlanning.objects.filter(
        Q(envoye_a__isnull=True) | Q(demande_a__gt=F("envoye_a")),
        demande_a__lte=limite,
    ).values_list("user_id", flat=True)
    envois = sum(rattraper_reveil(user_id) for user_id in list(en_attente))
    return envois + relancer_reveils_non_confirmes()


def _armer_rattrapage(user_id: int) -> None:
    """Un seul Timer par utilisateur et par processus."""
    with _minuteries_verrou:
        if user_id in _minuteries:
            return
        minuterie = threading.Timer(DELAI_RATTRAPAGE_SECONDES, _rattrapage_en_thread, args=(user_id,))
        minuterie.daemon = True
        _minuteries[user_id] = minuterie
        minuterie.start()


def _rattrapage_en_thread(user_id: int) -> None:
    # Liberer la place avant d'envoyer: une demande qui arrive pendant
    # l'envoi arme un nouveau Timer au lieu d'attendre le filet.
    with _minuteries_verrou:
        _minuteries.pop(user_id, None)
    try:
        rattraper_reveil(user_id)
    except Exception:  # noqa: BLE001
        logger.exception("Rattrapage du reveil impossible pour l'utilisateur %s", user_id)
    finally:
        connection.close()
