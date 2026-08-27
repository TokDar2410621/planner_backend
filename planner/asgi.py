"""
ASGI config for Planner AI backend.

Pourquoi l'ASGI plutot que WSGI + gevent, mesure le 2026-08-27:

L'agent v2 tourne sur PydanticAI, donc sur asyncio. Sous gunicorn -k gevent,
tous les greenlets partagent UN thread systeme, et asyncio suit sa boucle par
thread. Consequence mesuree sur 8 conversations concurrentes: 1 passe, 7
echouent en RuntimeError: asyncio.run() cannot be called from a running event
loop. Avant meme cela, l'ORM levait SynchronousOnlyOperation depuis la boucle.
C'est un conflit connu (gunicorn #3000 et #3053), pas une bizarrerie locale.

Ici, PAS de patch psycogreen: il n'existe que pour rendre psycopg2 cooperatif
avec le hub gevent, absent sous uvicorn. Le laisser serait au mieux inutile,
au pire trompeur pour la prochaine personne qui lit ce fichier.

Les taches de fond (traitement de document) passent par
utils.helpers.run_in_background, qui utilise threading.Thread et non des
greenlets: elles fonctionnent a l'identique ici, en vrais threads systeme.
"""
import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'planner.settings')

application = get_asgi_application()
