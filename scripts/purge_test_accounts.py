"""Purge des comptes de test prod (domaines synthetiques uniquement).

Garanties: ne touche AUCUN @gmail.com (assert), epargne les 2 comptes de la
suite de regression. Lancer depuis planner_backend avec la DB publique:

  export DATABASE_URL=$(railway variables -s Postgres --kv | grep '^DATABASE_PUBLIC_URL=' | cut -d= -f2-)
  source venv/Scripts/activate && python scripts/purge_test_accounts.py --run

Sans --run: dry-run (liste seulement).
"""
import os, sys, io, django

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'planner.settings')
django.setup()

from django.contrib.auth.models import User
from django.db.models import Q

KEEP = {'testall_ux_74737', 'formtest2_85312'}
synth = (Q(email__iendswith='@example.com') | Q(email__iendswith='@ex.com')
         | Q(email__iendswith='@t.local') | Q(email__iendswith='@test.com')
         | Q(email__istartswith='tokamdarius+'))
targets = User.objects.filter(synth).exclude(username__in=KEEP)
assert targets.filter(email__iendswith='@gmail.com').count() == 0, 'SECURITE: gmail dans la cible'
print(f'cibles: {targets.count()} comptes')
for u in targets.order_by('date_joined'):
    print(' -', u.username, '|', u.email)
if '--run' in sys.argv:
    deleted = targets.delete()
    print('supprimes:', deleted[0], 'objets en cascade | restants:', User.objects.count())
else:
    print('(dry-run: relance avec --run pour supprimer)')
