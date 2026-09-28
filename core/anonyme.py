"""Comptes anonymes (mode sans inscription, modele Firebase Anonymous Auth).

Un compte anonyme est un User Django ordinaire dont le profil porte
``is_anonymous=True`` et un ``device_id`` unique. A la connexion
(Google/Apple), le compte est CONVERTI (email attache, ``is_anonymous=False``,
``device_id`` libere) au lieu d'en creer un nouveau : les donnees (planning,
conversation, preferences) suivent sans migration.

Ce module ne depend de rien d'autre que de duck-typing : il est importe par
``core/views.py`` (throttle, conversion) et ``services/agent_v2/agent.py``
(budget), sans risque d'import circulaire.
"""

import re

# device_id : 8 a 64 caracteres, lettres/chiffres/tirets/underscores.
# (UUID d'appareil, identifiant securise iOS/Android, etc.)
DEVICE_ID_RE = re.compile(r'^[A-Za-z0-9_-]{8,64}$')


def est_anonyme(user) -> bool:
    """Vrai si ``user`` est un compte anonyme.

    Defensif : un profil manquant (donnees anciennes) ou un utilisateur non
    authentifie vaut NON-anonyme. On ne veut jamais elargir un quota ou
    ouvrir une conversion par erreur.
    """
    if user is None or not getattr(user, 'is_authenticated', False):
        return False
    profil = getattr(user, 'profile', None)
    return bool(getattr(profil, 'is_anonymous', False))


def device_id_valide(device_id) -> bool:
    """Valide le format d'un device_id fourni par le client."""
    return isinstance(device_id, str) and bool(DEVICE_ID_RE.match(device_id))
