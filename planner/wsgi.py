"""
WSGI config for Planner AI backend.
"""
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'planner.settings')

application = get_wsgi_application()
