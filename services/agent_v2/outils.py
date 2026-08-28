"""
Adaptateur: les 30 outils de v1 exposes a PydanticAI, INCHANGES.

Leur description et leur schema portent des regles produit; on les transmet
tels quels, jamais reecrits. Chaque execution alimente le registre, seule
source de verite du tour.

Trois pieges, tous verifies plutot que supposes:

1. Tool(fonction, name=..., description=...) derive le schema des annotations
   Python. Sur une fonction **kwargs, cela produit {"properties": {}}, donc 30
   outils sans le moindre parametre et un agent incapable de rien creer. Seul
   Tool.from_schema transmet le vrai schema (sonde du 2026-08-24).

2. requires_confirmation est applique par agent.py (lignes 921 a 944) et PAS
   par execute_tool. Un adaptateur qui appelle les outils en direct contourne
   la garde, et clear_all_blocks efface un planning sans confirmation.

3. L'ORM doit passer par sync_to_async, mais avec thread_sensitive=FALSE.
   Sous ASGI, Django execute la vue synchrone dans un thread de son pool, et
   on y demarre une boucle asyncio pour PydanticAI. Avec thread_sensitive=True
   asgiref veut rejouer l'ORM dans ce meme thread, deja bloque a attendre la
   boucle: interblocage, observe en production le 2026-08-27. Les connexions
   sont fermees des deux cotes, ce thread vivant hors du cycle de requete.
"""
from __future__ import annotations

import logging
import re
import unicodedata

from asgiref.sync import sync_to_async
from django.db import close_old_connections
from django.contrib.auth.models import User
from pydantic_ai.tools import Tool

from services.agent.tools import ALL_TOOLS
from services.agent.tools.base import ToolResult
from services.agent_v2.registre import (OUTILS_DE_MUTATION, Registre,
                                        _empreinte, boucle_detectee)

logger = logging.getLogger(__name__)

# Repris a l'identique de la garde v1 (agent.py:160-172). C'est le SEUL endroit
# du paquet ou une expression reguliere subsiste, et c'est une garde de
# securite sur l'intention de l'utilisateur, pas une verification de verite.
_CONFIRME = re.compile(
    r"\b(efface|supprime|vide|enleve|retire|reset|recommence|reinitialise|"
    r"oui|confirme|d'accord|ok|vas-y|fais-le|je confirme|c'est bon)\b",
    re.IGNORECASE,
)


def autorise_destructif(message) -> bool:
    """L'utilisateur a-t-il exprime une intention destructrice explicite ?"""
    if not message or not isinstance(message, str):
        return False
    plat = unicodedata.normalize("NFKD", message).encode("ascii", "ignore").decode("ascii")
    return bool(_CONFIRME.search(plat))


def _fabriquer(outil, user: User, registre: Registre, message_du_tour: str,
               tache: str, cache: dict):
    """Rend la coroutine que PydanticAI appellera avec les arguments du modele."""

    def _appel_ferme(*args, **kwargs):
        """L'ORM tourne dans un thread du pool d'asgiref, hors du cycle de
        requete qui ferme les connexions. On les ferme donc nous-memes des
        deux cotes, comme le fait database_sync_to_async de channels."""
        close_old_connections()
        try:
            return outil.execute(*args, **kwargs)
        finally:
            close_old_connections()

    # thread_sensitive=FALSE, et c'est mesure, pas theorique. Sous ASGI, Django
    # execute la vue synchrone dans un thread de son pool; on y demarre une
    # boucle asyncio pour PydanticAI. Avec thread_sensitive=True, asgiref veut
    # rejouer l'ORM dans CE thread, lequel est bloque a attendre la boucle:
    # interblocage, observe en production le 2026-08-27 (l'outil create_block
    # est appele, puis plus rien, et le client reessaie trois fois).
    executer_sync = sync_to_async(_appel_ferme, thread_sensitive=False)

    async def executer(**kwargs) -> str:
        if getattr(outil, "requires_confirmation", False) and not autorise_destructif(message_du_tour):
            refus = ToolResult(
                success=False,
                data={"needs_confirmation": True},
                message=(
                    "Action destructrice: il faut une confirmation explicite de "
                    "l'utilisateur avant de la faire. Demande-la, n'invente pas."
                ),
            )
            registre.ajouter(outil.name, kwargs, refus)
            return refus.to_string()

        # Meme marqueur que v1: le banc capte les appels d'outils par le
        # logger parent « services », et cette ligne est ce qu'il cherche.
        logger.info(f"Executing tool: {outil.name}({kwargs})")

        # IDEMPOTENCE, sur les ECRITURES seulement. Observe en production le
        # 2026-08-27: un tour bloque a fait reessayer le client trois fois, et
        # create_block est parti trois fois avec les memes arguments. Ce
        # jour-la l'outil echouait; le meme scenario avec un outil qui reussit
        # lentement creerait trois blocs.
        #
        # La cle vient de l'identite METIER de l'action (tache, outil,
        # arguments normalises) et JAMAIS d'un identifiant tire a chaque
        # tentative: un jeton neuf par essai ferait voir une action neuve a
        # chaque fois et le motif s'effondrerait en silence.
        #
        # Les LECTURES en sont exclues: elles doivent refleter l'etat courant,
        # sinon l'agent devient aveugle a ses propres ecritures du meme tour.
        cle = None
        if outil.name in OUTILS_DE_MUTATION:
            cle = f"{tache}:{_empreinte(outil.name, kwargs)}"
            if cle in cache:
                deja = cache[cle]
                logger.info("Idempotence: %s deja execute ce tour, resultat rejoue",
                            outil.name)
                registre.ajouter(outil.name, kwargs, deja)
                return deja.to_string()

        try:
            resultat = await executer_sync(user, **kwargs)
        except Exception as e:  # noqa: BLE001
            # v1 degrade une exception d'outil en ToolResult d'echec. Sans
            # cela, l'exception avorterait le run ET le registre ne garderait
            # aucune trace de la mutation tentee: le bloc factuel se tairait
            # sur un echec, ce qui est l'inverse du but.
            logger.error("Tool %s a leve: %s", outil.name, e, exc_info=True)
            resultat = ToolResult(
                success=False, data={}, message=f"Erreur de l'outil: {e}"
            )
        # On ne met en cache que les SUCCES: un echec peut etre transitoire
        # (429, timeout), et rejouer un echec empecherait toute reprise.
        if cle is not None and resultat.success:
            cache[cle] = resultat
        registre.ajouter(outil.name, kwargs, resultat)

        # Garde de terminaison, ici parce que c'est le seul point par lequel
        # TOUS les appels passent. Le budget d'etapes seul ne suffit pas: il
        # finit par arreter le tour, mais apres avoir brule dix allers-retours
        # a rejouer la meme action. On rend au modele une consigne explicite
        # plutot que de lever: il peut encore rediger une reponse utile avec
        # ce qu'il a, et le bloc factuel dira que le tour a ete coupe.
        if boucle_detectee(registre):
            registre.boucle_interrompue = True
            logger.warning(
                "Boucle detectee: %s rejoue a l'identique, tour interrompu", outil.name)
            return (
                "ARRET: tu viens de rejouer trois fois la meme action avec les "
                "memes arguments sans progresser. N'appelle plus d'outil. "
                "Reponds a l'utilisateur avec ce que tu as deja."
            )
        return resultat.to_string()

    executer.__name__ = outil.name
    return executer


def outils_pour(user: User, registre: Registre, message_du_tour: str = "",
                tache: str = "") -> list[Tool]:
    """Les 30 outils de v1, prets pour PydanticAI, branches sur ce registre.

    `tache` identifie le tour: il entre dans la cle d'idempotence pour que
    deux tours distincts puissent legitimement refaire la meme action, alors
    qu'un meme tour rejoue ne l'execute qu'une fois.
    """
    cache: dict = {}
    return [
        Tool.from_schema(
            _fabriquer(outil, user, registre, message_du_tour, tache, cache),
            outil.name,
            outil.description,
            outil.parameters,
        )
        for outil in ALL_TOOLS
    ]


def schema_expose(tool: Tool) -> dict:
    """Le schema JSON tel que le modele le verra, pour le test de parite."""
    return tool.function_schema.json_schema
