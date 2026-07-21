"""
Regression tests for GROUP A1 (config / deploy / hygiene).

Covers:
  - token_blacklist installed (JWT rotation blacklist)
  - DRF throttling configured (UserRateThrottle + AnonRateThrottle + chat/upload scopes)
  - STORAGES configured (Django 5.1+ replacement for DEFAULT_FILE_STORAGE) with
    Cloudinary wired for the "default" backend and WhiteNoise for staticfiles
  - S1: SECRET_KEY has no insecure hardcoded fallback active, and the settings
    module fails fast at boot when DEBUG=False and SECRET_KEY is missing, while
    still allowing an insecure dev default when DEBUG=True
  - S5: production hardening flags are present and gated on DEBUG
  - D6: services/core loggers are not forced to DEBUG in production
"""
import os
import subprocess
import sys

from django.conf import settings
from django.test import SimpleTestCase

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The exact insecure fallback that used to ship hardcoded in settings.py.
LEGACY_INSECURE_KEY = 'django-insecure-change-me-in-production'


class TokenBlacklistTests(SimpleTestCase):
    def test_token_blacklist_app_installed(self):
        self.assertIn(
            'rest_framework_simplejwt.token_blacklist',
            settings.INSTALLED_APPS,
        )


class ThrottlingTests(SimpleTestCase):
    def test_default_throttle_classes_configured(self):
        classes = settings.REST_FRAMEWORK.get('DEFAULT_THROTTLE_CLASSES', ())
        self.assertIn(
            'rest_framework.throttling.UserRateThrottle', classes
        )
        self.assertIn(
            'rest_framework.throttling.AnonRateThrottle', classes
        )
        self.assertIn(
            'rest_framework.throttling.ScopedRateThrottle', classes
        )

    def test_throttle_rates_include_chat_and_upload_scopes(self):
        rates = settings.REST_FRAMEWORK.get('DEFAULT_THROTTLE_RATES', {})
        for scope in ('user', 'anon', 'chat', 'upload'):
            self.assertIn(scope, rates)
            self.assertTrue(rates[scope], f'{scope} rate must be non-empty')


class StoragesTests(SimpleTestCase):
    def test_storages_configured(self):
        self.assertTrue(hasattr(settings, 'STORAGES'))
        self.assertIn('default', settings.STORAGES)
        self.assertIn('staticfiles', settings.STORAGES)

    def test_no_legacy_default_file_storage_only(self):
        # Django 5.1 removed DEFAULT_FILE_STORAGE; STORAGES must drive it now.
        self.assertIn('BACKEND', settings.STORAGES['default'])

    def test_staticfiles_uses_whitenoise(self):
        self.assertEqual(
            settings.STORAGES['staticfiles']['BACKEND'],
            'whitenoise.storage.CompressedManifestStaticFilesStorage',
        )

    def test_whitenoise_middleware_wired(self):
        self.assertIn(
            'whitenoise.middleware.WhiteNoiseMiddleware', settings.MIDDLEWARE
        )


class SecretKeyTests(SimpleTestCase):
    def test_secret_key_not_legacy_insecure_fallback(self):
        # The insecure hardcoded fallback must never be the active key.
        self.assertNotEqual(settings.SECRET_KEY, LEGACY_INSECURE_KEY)

    def test_secret_key_is_set(self):
        self.assertTrue(settings.SECRET_KEY)

    def _boot_settings(self, secret_key, debug):
        """Import planner.settings in a subprocess with a controlled env.

        SECRET_KEY is passed explicitly (empty string simulates 'unset' because
        python-dotenv does not override an already-present env var), so .env
        cannot leak a real key into the subprocess.
        """
        env = dict(os.environ)
        env['SECRET_KEY'] = secret_key
        env['DEBUG'] = debug
        env['DJANGO_SETTINGS_MODULE'] = 'planner.settings'
        return subprocess.run(
            [sys.executable, '-c',
             'from django.conf import settings; settings.SECRET_KEY'],
            cwd=BACKEND_DIR,
            env=env,
            capture_output=True,
            text=True,
        )

    def test_fail_fast_when_debug_false_and_no_secret_key(self):
        result = self._boot_settings(secret_key='', debug='False')
        self.assertNotEqual(
            result.returncode, 0,
            msg='settings must fail fast without SECRET_KEY when DEBUG=False',
        )
        self.assertIn('SECRET_KEY', result.stderr)

    def test_dev_default_allowed_when_debug_true(self):
        result = self._boot_settings(secret_key='', debug='True')
        self.assertEqual(
            result.returncode, 0,
            msg=f'DEBUG=True should allow a dev default. stderr:\n{result.stderr}',
        )


class ProdHardeningTests(SimpleTestCase):
    def test_csrf_trusted_origins_present(self):
        self.assertTrue(hasattr(settings, 'CSRF_TRUSTED_ORIGINS'))
        self.assertTrue(len(settings.CSRF_TRUSTED_ORIGINS) >= 1)

    def _boot_and_dump(self, debug):
        """Boot settings with DEBUG toggled and dump the hardening flags."""
        env = dict(os.environ)
        env['DEBUG'] = debug
        env['SECRET_KEY'] = 'x' * 60  # long enough to boot cleanly
        env['DJANGO_SETTINGS_MODULE'] = 'planner.settings'
        code = (
            'import json;'
            'from django.conf import settings;'
            'print(json.dumps({'
            '"ssl": getattr(settings, "SECURE_SSL_REDIRECT", False),'
            '"session": getattr(settings, "SESSION_COOKIE_SECURE", False),'
            '"csrf": getattr(settings, "CSRF_COOKIE_SECURE", False),'
            '"hsts": getattr(settings, "SECURE_HSTS_SECONDS", 0),'
            '"nosniff": getattr(settings, "SECURE_CONTENT_TYPE_NOSNIFF", False),'
            '"proxy": getattr(settings, "SECURE_PROXY_SSL_HEADER", None) is not None,'
            '"svc_log": settings.LOGGING["loggers"]["services"]["level"],'
            '}))'
        )
        result = subprocess.run(
            [sys.executable, '-c', code],
            cwd=BACKEND_DIR, env=env, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        import json
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_hardening_enabled_when_debug_false(self):
        flags = self._boot_and_dump(debug='False')
        self.assertTrue(flags['ssl'])
        self.assertTrue(flags['session'])
        self.assertTrue(flags['csrf'])
        self.assertGreater(flags['hsts'], 0)
        self.assertTrue(flags['nosniff'])
        self.assertTrue(flags['proxy'])
        # D6: loggers must not be DEBUG in production.
        self.assertEqual(flags['svc_log'], 'INFO')

    def test_hardening_relaxed_when_debug_true(self):
        flags = self._boot_and_dump(debug='True')
        self.assertFalse(flags['ssl'])
        self.assertFalse(flags['session'])
        self.assertFalse(flags['csrf'])
        self.assertEqual(flags['svc_log'], 'DEBUG')
