"""
Remove exact-duplicate recurring blocks.

    python manage.py dedupe_blocks            # all users
    python manage.py dedupe_blocks --user 8   # one user
"""
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand

from services.blocks_maintenance import dedupe_recurring_blocks


class Command(BaseCommand):
    help = "Supprime les blocs récurrents en double (même jour/heure/type)."

    def add_arguments(self, parser):
        parser.add_argument("--user", type=int, default=None, help="Limiter à un user id.")

    def handle(self, *args, **opts):
        users = (
            User.objects.filter(id=opts["user"]) if opts["user"] else User.objects.all()
        )
        total = 0
        for u in users:
            report = dedupe_recurring_blocks(u)
            if report["removed"]:
                total += report["removed"]
                self.stdout.write(
                    f"user {u.id} {u.username}: {report['removed']} doublon(s) supprimé(s)"
                )
        self.stdout.write(f"Terminé: {total} bloc(s) en double supprimé(s).")
