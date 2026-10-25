"""Valide et consomme la reponse structuree a une question visible du chat."""
import re
import math
from datetime import date, timedelta

from django.db import transaction
from django.utils import timezone

from core.models import ConversationMessage


_HEURE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


class ReponseQuestionRefusee(ValueError):
    def __init__(self, message, question=None):
        super().__init__(message)
        self.question = question


def _valider_champ(champ, valeur):
    kind = champ.get("type")
    if kind in ("text", "radio", "select"):
        if not isinstance(valeur, str) or not 0 < len(valeur.strip()) <= 500:
            return False
        if kind in ("radio", "select"):
            known = {o.get("value") for o in champ.get("options", [])}
            if valeur not in known and not (kind == "radio" and champ.get("allowOther")):
                return False
    elif kind == "checkbox":
        known = {o.get("value") for o in champ.get("options", [])}
        if (not isinstance(valeur, list) or len(valeur) > len(known)
                or any(not isinstance(v, str) or v not in known for v in valeur)):
            return False
    elif kind in ("time", "time_range"):
        if kind == "time":
            return isinstance(valeur, str) and bool(_HEURE.fullmatch(valeur))
        return (isinstance(valeur, dict) and set(valeur) == {"start", "end"}
                and all(isinstance(v, str) and _HEURE.fullmatch(v) for v in valeur.values()))
    elif kind == "date":
        try:
            date.fromisoformat(valeur)
        except (TypeError, ValueError):
            return False
    elif kind in ("number", "duration"):
        if isinstance(valeur, bool) or not isinstance(valeur, (int, float)):
            return False
        if not math.isfinite(valeur):
            return False
        if kind == "duration" and (valeur <= 0 or int(valeur) != valeur):
            return False
        if champ.get("min") is not None and valeur < champ["min"]:
            return False
        if champ.get("max") is not None and valeur > champ["max"]:
            return False
    else:
        return False
    return True


def consommer_reponse(user, conversation, courant, payload):
    """Retourne les valeurs verifiees, sans jamais autoriser une mutation."""
    if not isinstance(payload, dict) or not isinstance(payload.get("id"), str):
        raise ReponseQuestionRefusee("La réponse au formulaire est incomplète.")
    valeurs = payload.get("values")
    if not isinstance(valeurs, dict) or len(valeurs) > 6:
        raise ReponseQuestionRefusee("La réponse au formulaire est invalide.")
    with transaction.atomic():
        precedent = (ConversationMessage.objects.select_for_update()
                     .filter(user=user, conversation=conversation, pk__lt=courant.pk)
                     .order_by("-pk").first())
        if precedent is None or precedent.role != "assistant":
            raise ReponseQuestionRefusee("Cette question n'est plus la dernière de la conversation.")
        meta = precedent.metadata if isinstance(precedent.metadata, dict) else {}
        question = meta.get("question_custom") or {}
        if (question.get("id") != payload["id"] or question.get("kind") not in ("text", "form")
                or meta.get("question_answered_by")):
            raise ReponseQuestionRefusee("Cette question n'est plus active dans cette conversation.")
        if timezone.now() - precedent.created_at >= timedelta(minutes=30):
            raise ReponseQuestionRefusee("Cette question a expiré. Redemande-la pour continuer.")
        champs = question.get("fields") or []
        if not champs or set(valeurs) != {c.get("id") for c in champs}:
            raise ReponseQuestionRefusee("Les champs envoyés ne correspondent pas à la question affichée.", question)
        if not all(_valider_champ(c, valeurs[c["id"]]) for c in champs):
            raise ReponseQuestionRefusee("Une valeur du formulaire est invalide. Corrige-la et renvoie-la.", question)
        precedent.metadata = {**meta, "question_answered_by": courant.pk}
        precedent.save(update_fields=["metadata"])
    courant.metadata = {**(courant.metadata or {}), "question_answer": {
        "id": payload["id"], "values": valeurs,
    }}
    courant.save(update_fields=["metadata"])
    return valeurs
