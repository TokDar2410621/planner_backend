"""Lire un flux SSE dans un test, qu'il soit synchrone ou asynchrone.

La vue de streaming rend desormais un iterateur ASYNCHRONE: sous ASGI, Django
ne sait pas servir un generateur synchrone au fil de l'eau, il le draine et
n'envoie qu'a la fin. Mesure au navigateur le 2026-08-29: le client recevait
status, thinking, tool, delta et done TOUS ENSEMBLE a 12,84 s, alors que le
serveur les emettait de 0,07 s a 21 s.

Le client de test reste synchrone; ce petit pont lui permet de consommer les
deux formes sans que chaque test ait a le savoir.
"""
from asgiref.sync import async_to_sync


def corps_du_flux(reponse) -> str:
    """Le corps complet d'une reponse en flux, decode."""
    contenu = reponse.streaming_content
    if hasattr(contenu, "__aiter__"):
        async def _lire():
            return b"".join([morceau async for morceau in contenu])
        return async_to_sync(_lire)().decode("utf-8")
    return b"".join(contenu).decode("utf-8")
