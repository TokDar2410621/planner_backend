"""
Révocation Sign in with Apple — exigence Apple depuis juin 2022.

Une app qui propose Sign in with Apple ET la suppression de compte doit
révoquer les jetons Apple de l'utilisateur au moment de la suppression
(prolongement de la guideline 5.1.1(v)). Concrètement:

  1. À la CONNEXION, le client envoie aussi le code d'autorisation; on
     l'échange contre un refresh token (POST /auth/token) qu'on garde sur le
     profil.
  2. À la SUPPRESSION, on révoque ce refresh token (POST /auth/revoke) avant
     d'effacer le compte.

Les deux appels exigent un `client_secret`: un JWT ES256 signé avec la clé
« Sign in with Apple » du portail développeur (APPLE_SIGNIN_PRIVATE_KEY,
APPLE_SIGNIN_KEY_ID, APPLE_TEAM_ID — posées sur Railway, jamais dans le code).

Le `client_id` varie selon le chemin de connexion: Services ID pour le web
(com.tokamdarius.planner.web), bundle id pour l'app native
(com.tokamdarius.planner). On mémorise donc l'audience du jeton vérifié et on
réutilise la même pour l'échange comme pour la révocation.
"""
import logging
import time

import jwt
import requests
from django.conf import settings

logger = logging.getLogger(__name__)

APPLE_TOKEN_URL = 'https://appleid.apple.com/auth/token'
APPLE_REVOKE_URL = 'https://appleid.apple.com/auth/revoke'


def revocation_configured() -> bool:
    return bool(
        getattr(settings, 'APPLE_SIGNIN_PRIVATE_KEY', '')
        and getattr(settings, 'APPLE_SIGNIN_KEY_ID', '')
        and getattr(settings, 'APPLE_TEAM_ID', '')
    )


def make_client_secret(client_id: str) -> str:
    """JWT ES256 exigé par Apple comme client_secret (validité max 6 mois; on
    prend 20 minutes, il est régénéré à chaque appel)."""
    now = int(time.time())
    return jwt.encode(
        {
            'iss': settings.APPLE_TEAM_ID,
            'iat': now,
            'exp': now + 20 * 60,
            'aud': 'https://appleid.apple.com',
            'sub': client_id,
        },
        settings.APPLE_SIGNIN_PRIVATE_KEY,
        algorithm='ES256',
        headers={'kid': settings.APPLE_SIGNIN_KEY_ID},
    )


def exchange_authorization_code(code: str, client_id: str) -> str:
    """Échange le code d'autorisation contre un refresh token. Rend '' si Apple
    refuse (code déjà consommé, expiré — il ne vit que 5 minutes)."""
    try:
        resp = requests.post(
            APPLE_TOKEN_URL,
            data={
                'client_id': client_id,
                'client_secret': make_client_secret(client_id),
                'code': code,
                'grant_type': 'authorization_code',
            },
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning('Apple token exchange refuse (%s): %s', resp.status_code, resp.text[:200])
            return ''
        return resp.json().get('refresh_token', '') or ''
    except Exception as e:  # noqa: BLE001 - l'echange est best-effort
        logger.warning('Apple token exchange en erreur: %s', e)
        return ''


def revoke_refresh_token(refresh_token: str, client_id: str) -> bool:
    """Révoque le refresh token chez Apple. Best-effort: un échec est loggué,
    jamais bloquant — un endpoint Apple en panne ne doit pas rendre la
    suppression de compte impossible."""
    try:
        resp = requests.post(
            APPLE_REVOKE_URL,
            data={
                'client_id': client_id,
                'client_secret': make_client_secret(client_id),
                'token': refresh_token,
                'token_type_hint': 'refresh_token',
            },
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=10,
        )
        if resp.status_code == 200:
            return True
        logger.warning('Apple revoke refuse (%s): %s', resp.status_code, resp.text[:200])
        return False
    except Exception as e:  # noqa: BLE001
        logger.warning('Apple revoke en erreur: %s', e)
        return False
