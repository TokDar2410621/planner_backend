"""Géocodage adresse -> coordonnées (rappels basés sur la localisation, Phase 0).

Utilise Nominatim (OpenStreetMap), gratuit, pour convertir une adresse saisie
en (latitude, longitude). Best-effort: en cas d'échec (réseau, rate limit,
adresse introuvable) on renvoie None et le lieu reste sans coordonnées, sans
casser la création du lieu.

Politique Nominatim respectée: User-Agent identifiant l'app, un seul résultat,
timeout court. Pour un usage intensif il faudra migrer vers Mapbox Permanent ou
un Nominatim auto-hébergé (voir plan géoloc).
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

_ENDPOINT = "https://nominatim.openstreetmap.org/search"
_USER_AGENT = "PlannerAI/1.0 (+https://day-wise-bot.vercel.app)"
_TIMEOUT_SECONDS = 4


def geocode_address(address: str):
    """(latitude, longitude) floats pour `address`, ou None si indisponible."""
    address = (address or "").strip()
    if not address:
        return None

    query = urllib.parse.urlencode({
        "q": address,
        "format": "json",
        "limit": 1,
        "addressdetails": 0,
    })
    url = f"{_ENDPOINT}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # réseau, timeout, JSON, HTTP...
        logger.warning("Geocoding failed for %r: %s", address, exc)
        return None

    if not payload:
        return None

    try:
        lat = float(payload[0]["lat"])
        lon = float(payload[0]["lon"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None

    return lat, lon
