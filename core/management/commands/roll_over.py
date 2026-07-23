"""
Roll missed blocks forward for users (structural forgiveness).

    python manage.py roll_over            # all users with a push subscription
    python manage.py roll_over --all      # every user
    python manage.py roll_over --user 8

Best run once a day at day-start. Primary trigger is the per-user endpoint
POST /schedule/rollover/ (on app open); this cron is a safety net.
"""
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand

from core.models import PushSubscription
from services.rollover import roll_over_missed


class Command(BaseCommand):
    help = "Reprogramme les blocs passés non complétés vers aujourd'hui+."

    def add_arguments(self, parser):
        parser.add_argument("--user", type=int, default=None)
        parser.add_argument("--all", action="store_true", help="Tous les users.")

    def handle(self, *args, **opts):
        if opts["user"]:
            users = User.objects.filter(id=opts["user"])
        elif opts["all"]:
            users = User.objects.all()
        else:
            uids = PushSubscription.objects.values_list("user_id", flat=True).distinct()
            users = User.objects.filter(id__in=list(uids))

        total = 0
        for u in users:
            report = roll_over_missed(u)
            if report["rolled"]:
                total += report["rolled"]
                self.stdout.write(f"user {u.id} {u.username}: {report['rolled']} bloc(s) reprogrammé(s)")
        self.stdout.write(f"Roll-over: {total} bloc(s) reprogrammé(s).")
