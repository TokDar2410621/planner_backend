"""Clear all recurring blocks from the database."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'planner.settings')

import django
django.setup()

from core.models import RecurringBlock, UploadedDocument

# Count before
count = RecurringBlock.objects.count()
print(f'Blocs avant: {count}')

# Delete all blocks
RecurringBlock.objects.all().delete()

# Also reset documents processed status
UploadedDocument.objects.all().delete()

print(f'Tous les blocs et documents supprimés!')
print(f'Blocs restants: {RecurringBlock.objects.count()}')
print(f'Documents restants: {UploadedDocument.objects.count()}')
