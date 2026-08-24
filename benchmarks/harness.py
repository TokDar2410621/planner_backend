"""
Harnais du banc: monde de test, exécution d'un tour, notation.

Le banc reconstruit pour CHAQUE épreuve un utilisateur neuf avec un planning
connu, envoie des messages à l'agent sous test, puis compare l'état RÉEL de la
base à la vérité attendue. Aucune notation ne repose sur ce que l'agent
raconte: seuls comptent la base et, pour le ton, un juge LLM sur grille.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from typing import Callable, Optional

from django.contrib.auth.models import User
from django.utils import timezone

from core.models import RecurringBlock, ScheduledBlock, Task, UploadedDocument


# ---------------------------------------------------------------- monde de test

@dataclass
class Monde:
    """L'utilisateur du banc et son planning de départ."""

    user: User
    lundi: date  # lundi de la semaine de référence

    def bloc(self, titre: str, dow: int, debut: str, fin: str,
             btype: str = "course") -> RecurringBlock:
        return RecurringBlock.objects.create(
            user=self.user, title=titre, block_type=btype, day_of_week=dow,
            start_time=dtime.fromisoformat(debut), end_time=dtime.fromisoformat(fin),
            active=True,
        )

    def tache(self, titre: str, **kw) -> Task:
        return Task.objects.create(user=self.user, title=titre, **kw)

    # --- lectures pour la notation (l'état RÉEL, jamais le récit de l'agent)

    def blocs(self, titre_contient: str = "") -> list[RecurringBlock]:
        qs = RecurringBlock.objects.filter(user=self.user, active=True)
        if titre_contient:
            qs = qs.filter(title__icontains=titre_contient)
        return list(qs.order_by("day_of_week", "start_time"))

    def places(self, jour: Optional[date] = None) -> list[ScheduledBlock]:
        qs = ScheduledBlock.objects.filter(user=self.user)
        if jour:
            qs = qs.filter(date=jour)
        return list(qs.select_related("task").order_by("date", "start_time"))

    def taches(self) -> list[Task]:
        return list(Task.objects.filter(user=self.user).order_by("id"))


def monde_neuf(prefixe: str) -> Monde:
    """Utilisateur jetable, consentement IA accordé, semaine de référence."""
    u = User.objects.create_user(
        username=f"bench-{prefixe}-{uuid.uuid4().hex[:8]}", password="x"
    )
    u.profile.ai_consent_at = timezone.now()
    u.profile.save(update_fields=["ai_consent_at"])
    today = timezone.localtime().date()
    lundi = today - timedelta(days=today.weekday())
    return Monde(user=u, lundi=lundi)


# ---------------------------------------------------------------- exécution

@dataclass
class Tour:
    """Un échange: ce qu'on a demandé, ce que l'agent a répondu et fait."""

    message: str
    reponse: str
    outils: list[str] = field(default_factory=list)
    secondes: float = 0.0
    erreur: str = ""


class _CaptureOutils(logging.Handler):
    """Écoute le logger de l'agent et retient les noms d'outils exécutés."""

    def __init__(self, cible: list):
        super().__init__(level=logging.INFO)
        self.cible = cible
        self._logger = logging.getLogger("services.agent.agent")

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        marqueur = "Executing tool: "
        if marqueur in msg:
            reste = msg.split(marqueur, 1)[1]
            self.cible.append(reste.split("(", 1)[0].strip())

    def start(self) -> None:
        self._logger.addHandler(self)

    def stop(self) -> None:
        self._logger.removeHandler(self)


class Pilote:
    """Adaptateur d'agent: v1 aujourd'hui, v2 quand il existera."""

    def __init__(self, nom: str, fabrique: Callable[[], object]):
        self.nom = nom
        self._fabrique = fabrique

    def envoyer(self, user: User, message: str,
                attachment: Optional[UploadedDocument] = None) -> Tour:
        debut = time.monotonic()
        # Le payload « done » n'expose pas les appels d'outils: on les capte
        # depuis le logger de l'agent, qui trace « Executing tool: <nom>(... ».
        # Comparer ce qui a VRAIMENT tourné au texte produit est le coeur de
        # l'épreuve « vérité d'action », donc cette capture fait partie du banc.
        outils: list[str] = []
        piege = _CaptureOutils(outils)
        try:
            agent = self._fabrique()
            piege.start()
            try:
                res = agent.process_message(
                    user, message, attachment, generate_quick_replies=False
                )
            finally:
                piege.stop()
            return Tour(
                message=message,
                reponse=(res or {}).get("response", "") or "",
                outils=outils,
                secondes=round(time.monotonic() - debut, 1),
            )
        except Exception as e:  # noqa: BLE001 - une épreuve ratée n'arrête pas le banc
            return Tour(message=message, reponse="", secondes=round(time.monotonic() - debut, 1),
                        erreur=f"{type(e).__name__}: {e}"[:300])


def pilote_v1() -> Pilote:
    from services.agent import PlannerAgent
    return Pilote("v1", PlannerAgent)


def pilote_v2() -> Pilote:
    from services.agent_v2 import PlannerAgentV2  # existera après l'incrément 1
    return Pilote("v2", PlannerAgentV2)


# ---------------------------------------------------------------- notation

@dataclass
class Note:
    """Résultat d'une épreuve: points obtenus sur points possibles."""

    epreuve: str
    obtenus: float
    possibles: float
    details: list[str] = field(default_factory=list)
    tours: list[Tour] = field(default_factory=list)

    def point(self, gagne: bool, libelle: str, poids: float = 1.0) -> None:
        self.possibles += poids
        if gagne:
            self.obtenus += poids
        self.details.append(f"{'OK ' if gagne else 'RATE'} ({poids:g}) {libelle}")

    @property
    def pourcent(self) -> float:
        return 100.0 * self.obtenus / self.possibles if self.possibles else 0.0


def hhmm(t) -> str:
    return t.strftime("%H:%M") if t else ""
