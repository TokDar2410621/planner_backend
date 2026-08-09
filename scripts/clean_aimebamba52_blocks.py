"""Vide les blocs recurrents poubelle du compte aimebamba52 (43 blocs issus
des imports rates d'avant le fix). Ses taches/objectifs ne sont pas touches.

  export DATABASE_URL=$(railway variables -s Postgres --kv | grep '^DATABASE_PUBLIC_URL=' | cut -d= -f2-)
  source venv/Scripts/activate && python scripts/clean_aimebamba52_blocks.py --run
"""
import os, sys, io, django

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'planner.settings')
django.setup()

from django.contrib.auth.models import User
from core.models import RecurringBlock

u = User.objects.get(username='aimebamba52')
qs = RecurringBlock.all_objects.filter(user=u)
print('blocs recurrents de aimebamba52:', qs.count())
for b in qs.order_by('day_of_week', 'start_time')[:50]:
    print(f' - dow{b.day_of_week} {b.start_time} {b.title[:40]}')
if '--run' in sys.argv:
    deleted = qs.delete()
    print('supprimes:', deleted[0])
else:
    print('(dry-run: relance avec --run pour supprimer)')
