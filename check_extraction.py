"""Check extracted data and blocks."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'planner.settings')

import django
django.setup()

from core.models import UploadedDocument, RecurringBlock

# Get latest document
doc = UploadedDocument.objects.filter(processed=True).order_by('-id').first()
if doc:
    print(f'=== Document {doc.id}: {doc.file.name} ===')
    print(f'Method: {doc.extracted_data.get("extraction_method", "?")}')
    print()

    print('=== JSON COURSES ===')
    for c in doc.extracted_data.get('courses', []):
        print(f'{c.get("day")} {c.get("start_time")}-{c.get("end_time")}: {c.get("name")} @ {c.get("location")}')

    print()
    print('=== BLOCS CREES EN DB ===')
    blocks = RecurringBlock.objects.filter(source_document=doc).order_by('day_of_week', 'start_time')
    days = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
    for b in blocks:
        print(f'{days[b.day_of_week]} {b.start_time.strftime("%H:%M")}-{b.end_time.strftime("%H:%M")}: {b.title} @ {b.location} (type={b.block_type})')

    print()
    print(f'Total blocs: {blocks.count()}')
else:
    print('Aucun document traité')

print()
print('=== TOUS LES BLOCS ===')
all_blocks = RecurringBlock.objects.all().order_by('day_of_week', 'start_time')
days = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
for b in all_blocks:
    print(f'{days[b.day_of_week]} {b.start_time.strftime("%H:%M")}-{b.end_time.strftime("%H:%M")}: {b.title} @ {b.location}')
print(f'Total: {all_blocks.count()}')
