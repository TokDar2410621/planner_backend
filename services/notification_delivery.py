"""Livraison visible commune aux navigateurs et a l'application iOS."""
from core.models import AppareilPush, PushSubscription
from services import apns, push


def has_device(user) -> bool:
    """Un appareil est pret uniquement si son canal est effectivement utilisable."""
    return (
        (push.push_configured() and PushSubscription.objects.filter(user=user).exists())
        or (apns.apns_configured() and AppareilPush.objects.filter(
            user=user, notifications_autorisees=True).exists())
    )


def send_visible(user, title: str, body: str, url: str) -> int:
    """Renvoie le nombre d'appareils ayant accepte une notification visible."""
    count = push.send_to_user(user, title, body, url=url) if push.push_configured() else 0
    if apns.apns_configured():
        for device in AppareilPush.objects.filter(
                user=user, notifications_autorisees=True):
            count += int(apns.envoyer_alerte(device, title, body, url))
    return count
