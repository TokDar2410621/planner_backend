"""
Sign in with Apple — verify the identity token server-side.

The client (web 'Sign in with Apple JS' or a native app) sends Apple's identity
token (a JWT, RS256). We verify its signature against Apple's published public
keys (JWKS), and its issuer/audience/expiry, then trust its email claim.

`APPLE_CLIENT_ID` is the audience Apple stamped the token with:
- web: the Services ID (e.g. "com.planner.web");
- native iOS: the app bundle ID.
Support a comma-separated list so web + native tokens both pass.
"""
import jwt
from django.conf import settings

APPLE_ISSUER = 'https://appleid.apple.com'
APPLE_JWKS_URL = 'https://appleid.apple.com/auth/keys'

# Module-level JWKS client caches Apple's keys across requests.
_jwk_client = None


def _get_jwk_client():
    global _jwk_client
    if _jwk_client is None:
        _jwk_client = jwt.PyJWKClient(APPLE_JWKS_URL)
    return _jwk_client


def allowed_audiences():
    raw = getattr(settings, 'APPLE_CLIENT_ID', '') or ''
    return [a.strip() for a in raw.split(',') if a.strip()]


def apple_configured() -> bool:
    return bool(allowed_audiences())


def verify_apple_identity_token(id_token: str) -> dict:
    """Verify an Apple identity token and return its claims. Raises on invalid."""
    audiences = allowed_audiences()
    if not audiences:
        raise ValueError('APPLE_CLIENT_ID non configuré.')
    signing_key = _get_jwk_client().get_signing_key_from_jwt(id_token)
    return jwt.decode(
        id_token,
        signing_key.key,
        algorithms=['RS256'],
        audience=audiences,
        issuer=APPLE_ISSUER,
        options={'require': ['exp', 'iat', 'sub']},
    )
