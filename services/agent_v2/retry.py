"""Politique de retry facon OpenCode pour la boucle AGIR.

Reference : packages/core/src/session/runner/retry.ts (branche dev).
- Backoff exponentiel 2s, 4s, 8s puis plafonne a 10s par intervalle.
- Max 10 tentatives (~84s d'attente cumulee).
- `retry-after` honore mais plafonne a 15 min (un retry-after hostile ou
  bugge ne doit pas bloquer un tour des heures).
- Timeouts : max 3 essais (une tentative qui a timeout a deja attendu).

Seules les erreurs reessayables (transport, rate limit, 5xx provider)
sont retentees. Les 4xx (hors 429) et les erreurs de validation ne le
sont jamais.
"""

import logging
import re

logger = logging.getLogger(__name__)

TENTATIVES_MAX = 10
TENTATIVES_MAX_TIMEOUT = 3
DELAI_MAX_INTERVALLE = 10.0
RETRY_AFTER_MAX = 15 * 60.0


def delai_backoff(tentative: int) -> float:
    """Delai avant la tentative n (1-indexee) : 2, 4, 8 puis 10s."""
    return min(2.0 ** tentative, DELAI_MAX_INTERVALLE)


def _retry_after(erreur: Exception) -> float | None:
    """Extrait un retry-after (secondes) de l'erreur, si le provider en donne."""
    texte = str(erreur)
    m = re.search(r"retry-?after[\"']?\s*[:=]\s*(\d+(?:\.\d+)?)", texte, re.IGNORECASE)
    if m:
        try:
            return min(float(m.group(1)), RETRY_AFTER_MAX)
        except ValueError:
            return None
    # Certains SDK exposent l'en-tete sur l'exception.
    for attr in ("retry_after", "retryAfterMs", "retry_after_ms"):
        val = getattr(erreur, attr, None)
        if val is not None:
            try:
                secondes = float(val) / 1000.0 if "ms" in attr.lower() or attr == "retryAfterMs" else float(val)
                return min(secondes, RETRY_AFTER_MAX)
            except (TypeError, ValueError):
                continue
    reponse = getattr(erreur, "response", None)
    headers = getattr(reponse, "headers", None)
    if headers:
        for cle in ("retry-after", "Retry-After"):
            try:
                val = headers.get(cle)
            except Exception:  # noqa: BLE001
                val = None
            if val:
                try:
                    return min(float(val), RETRY_AFTER_MAX)
                except (TypeError, ValueError):
                    continue
    return None


def est_reessayable(erreur: Exception) -> bool:
    """Transport, rate limit et 5xx : oui. 4xx (hors 429) et validation : non."""
    texte = str(erreur).lower()
    statut = getattr(erreur, "status_code", None) or getattr(erreur, "status", None)
    try:
        statut = int(statut) if statut is not None else None
    except (TypeError, ValueError):
        statut = None
    if statut is not None:
        if statut == 429:
            return True
        if 500 <= statut < 600:
            return True
        if 400 <= statut < 500:
            return False
    marqueurs_reessayables = (
        "rate limit", "ratelimit", "too many requests", "429",
        "timeout", "timed out", "connection", "temporarily unavailable",
        "502", "503", "500", "internal error", "overloaded",
        "stream",  # stream interrompu : OpenCode retente apres output commence
    )
    if any(m in texte for m in marqueurs_reessayables):
        return True
    marqueurs_definitifs = (
        "invalid", "validation", "bad request", "unauthorized",
        "forbidden", "not found", "authentication",
    )
    if any(m in texte for m in marqueurs_definitifs):
        return False
    # Par defaut : les erreurs reseau/transport inconnues sont reessayables,
    # les erreurs metier ne devraient pas arriver ici (typées en amont).
    return True


def delai_avant_reessai(tentative: int, erreur: Exception) -> float:
    """Delai OpenCode : max(backoff, retry-after plafonne)."""
    backoff = delai_backoff(tentative)
    ra = _retry_after(erreur)
    if ra is not None:
        return max(backoff, ra)
    return backoff


def est_timeout(erreur: Exception) -> bool:
    texte = str(erreur).lower()
    return "timeout" in texte or "timed out" in texte
