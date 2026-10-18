"""Cree les prochaines occurrences des taches et rappels recurrents."""
from django.core.management.base import BaseCommand

from services.reminder_series import materialize_active


class Command(BaseCommand):
    help = 'Maintient une fenetre de 35 jours pour les series actives.'

    def handle(self, *args, **options):
        count = materialize_active()
        self.stdout.write(f'{count} occurrence(s) creee(s).')
