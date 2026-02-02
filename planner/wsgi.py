"""
WSGI config for Planner AI backend.
"""
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'planner.settings')

# Patch psycopg2 for gevent compatibility BEFORE loading Django
# This is required for background greenlets to work with database queries
try:
    from psycogreen.gevent import patch_psycopg
    patch_psycopg()
    print("[WSGI] Patched psycopg2 for gevent compatibility", flush=True)
except ImportError:
    print("[WSGI] psycogreen not available, skipping patch", flush=True)
except Exception as e:
    print(f"[WSGI] Error patching psycopg2: {e}", flush=True)

application = get_wsgi_application()
