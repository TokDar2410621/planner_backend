"""
Gunicorn configuration for Planner AI backend.
"""
import multiprocessing

# Bind to all interfaces
bind = "0.0.0.0:8000"


def post_fork(server, worker):
    """
    Called after a worker has been forked.
    Patches psycopg2 to be compatible with gevent greenlets.
    Without this patch, database queries in background greenlets will block.
    """
    from psycogreen.gevent import patch_psycopg
    patch_psycopg()
    server.log.info("Patched psycopg2 for gevent compatibility")

# Worker configuration
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "gevent"
worker_connections = 1000  # Max concurrent connections per worker

# Timeout configuration (for long PDF processing)
timeout = 120  # 2 minutes (default is 30s)
graceful_timeout = 30
keepalive = 5

# Request handling
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Development vs Production
reload = False  # Set to True for development
