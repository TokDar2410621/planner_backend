"""Cerveau quotidien proactif — tourne dans le worker Railway toutes les 15 min.

Auto-gardé : chaque utilisateur n'est traité qu'UNE fois par jour (ligne
DailyBrainReport), et seulement à partir de --after heures (heure locale
America/Toronto, le fuseau produit). Le push ne part que
si le brief contient quelque chose d'actionnable.

    python manage.py daily_brain                # tous les users actifs
    python manage.py daily_brain --user 8       # un seul
    python manage.py daily_brain --force        # ignore l'heure (tests/debug)
"""
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from django.utils import timezone

from core.models import DailyBrainReport, RecurringBlock, Task
from services.daily_brain import build_daily_brief, push_daily_brief


class Command(BaseCommand):
    help = "Brief quotidien proactif: rollover + conflits + échéances, puis push."

    def add_arguments(self, parser):
        parser.add_argument("--after", type=int, default=7,
                            help="Heure locale minimale (TIME_ZONE produit, défaut 7h).")
        parser.add_argument("--user", type=int, default=None)
        parser.add_argument("--force", action="store_true",
                            help="Ignore la fenêtre horaire.")

    def handle(self, *args, **opts):
        now = timezone.localtime()
        if now.hour < opts["after"] and not opts["force"]:
            self.stdout.write(
                f"DailyBrain: avant {opts['after']}h serveur, rien à faire.")
            return

        today = timezone.localdate()
        if opts["user"]:
            users = User.objects.filter(id=opts["user"])
        else:
            # Users « actifs » = au moins un bloc récurrent actif ou une tâche
            # ouverte, pas encore traités aujourd'hui.
            active_ids = (
                set(RecurringBlock.objects.filter(active=True)
                    .values_list("user_id", flat=True))
                | set(Task.objects.filter(completed=False)
                      .values_list("user_id", flat=True))
            )
            done_ids = set(DailyBrainReport.objects.filter(date=today)
                           .values_list("user_id", flat=True))
            users = User.objects.filter(id__in=(active_ids - done_ids))

        processed = pushed = 0
        for user in users:
            try:
                payload = build_daily_brief(user)
                processed += 1
                if payload.get("items"):
                    if push_daily_brief(user, payload):
                        pushed += 1
                        DailyBrainReport.objects.filter(
                            user=user, date=today).update(pushed=True)
            except Exception as e:  # noqa: BLE001 - un user cassé n'arrête pas le batch
                self.stderr.write(f"DailyBrain user {user.id}: {e}")

        self.stdout.write(
            f"DailyBrain: {processed} user(s) traité(s), {pushed} push(s) envoyé(s).")
