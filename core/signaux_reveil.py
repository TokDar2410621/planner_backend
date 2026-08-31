"""
Reveil silencieux: un bloc change cote serveur, le telephone se reveille.

post_save et post_delete sur les modeles qui font le planning (RecurringBlock,
ScheduledBlock, RecurringBlockException), connectes dans CoreConfig.ready().
Chaque changement programme, apres commit, un reveil APNs de l'utilisateur
(services/apns.py), quelle que soit la source: Planning, agent, MCP, admin.
Zero cout tant que APNs n'est pas configure: le handler sort a la premiere
ligne.
"""
import logging

from django.core.exceptions import ObjectDoesNotExist
from django.db import transaction
from django.db.models.signals import post_delete, post_save

from services.apns import apns_configured, reveiller_utilisateur

logger = logging.getLogger(__name__)

MODELES_PLANNING = ("RecurringBlock", "ScheduledBlock", "RecurringBlockException")


def _utilisateur_de(instance):
    """L'utilisateur touche par ce changement, None s'il n'existe plus."""
    try:
        if getattr(instance, "user_id", None):
            return instance.user
        bloc = getattr(instance, "recurring_block", None)
        if bloc is not None:
            return bloc.user
        tache = getattr(instance, "task", None)
        if tache is not None:
            return tache.user
    except ObjectDoesNotExist:
        # Suppression en cascade d'un compte: l'utilisateur est deja parti.
        return None
    return None


def _reveiller(user):
    try:
        reveiller_utilisateur(user, "planning")
    except Exception:  # noqa: BLE001
        logger.exception("Reveil planning impossible pour l'utilisateur %s", getattr(user, "pk", None))


def programmer_reveil(sender, instance, **kwargs):
    if not apns_configured():
        return
    user = _utilisateur_de(instance)
    if user is None:
        return
    transaction.on_commit(lambda: _reveiller(user))


def connecter():
    from django.apps import apps

    for nom in MODELES_PLANNING:
        modele = apps.get_model("core", nom)
        post_save.connect(programmer_reveil, sender=modele, dispatch_uid=f"reveil_planning_save_{nom}")
        post_delete.connect(programmer_reveil, sender=modele, dispatch_uid=f"reveil_planning_delete_{nom}")
