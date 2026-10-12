"""Échéance et usages partagés par les appels fournisseurs d'un tour."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
import time

from django.conf import settings


def _limite_jugements():
    try:
        return max(0, int(getattr(settings, "AGENT_MAX_JUGEMENTS", 6)))
    except (TypeError, ValueError):
        return 6


@dataclass
class TraceTour:
    echeance: float
    usages: list[dict] = field(default_factory=list)
    total_jetons: int = 0
    limite_jugements: int = field(default_factory=_limite_jugements)
    jugements: int = 0
    reservation: object | None = field(default=None, repr=False)
    _verrou: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def restant(self) -> float:
        return max(0.0, self.echeance - time.monotonic())

    def admettre_jugement(self) -> bool:
        with self._verrou:
            if self.restant() <= 0 or self.jugements >= self.limite_jugements:
                return False
            self.jugements += 1
            return True

    @staticmethod
    def _mesure(usage):
        """Garde les mesures, jamais les messages ou les réponses du modèle."""
        champs = {"phase", "fournisseur", "etapes", "entree", "sortie", "total",
                  "raisonnement", "cache", "duree", "usage_connu", "succes", "cache_decision",
                  "jetons_provisionnes", "jetons_factures"}
        mesure = {k: v for k, v in usage.items() if k in champs}
        total = mesure.get('jetons_factures', mesure.get("total"))
        if mesure.get('usage_connu') is False:
            total = mesure.get('jetons_factures', mesure.get('jetons_provisionnes'))
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            total = sum(v for k in ("entree", "sortie")
                        if isinstance((v := mesure.get(k)), int) and not isinstance(v, bool) and v > 0)
        return mesure, total

    def ajouter(self, usage: dict) -> int:
        mesure, total = self._mesure(usage)
        with self._verrou:
            index = len(self.usages)
            self.usages.append(mesure)
            self.total_jetons += total
            return index

    def remplacer(self, index: int, usage: dict) -> None:
        mesure, total = self._mesure(usage)
        with self._verrou:
            _, ancien_total = self._mesure(self.usages[index])
            self.usages[index] = mesure
            self.total_jetons += total - ancien_total


trace_courante: ContextVar[TraceTour | None] = ContextVar("trace_tour", default=None)
