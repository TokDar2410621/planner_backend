"""Mot de passe oublié: request (anti-énumération) + confirm (token single-use).

En tests, Django route send_mail vers locmem (django.core.mail.outbox): on
vérifie le VRAI contenu du mail (lien uid+token) et on rejoue le lien contre
/confirm/ — le flux complet, pas juste les statuts HTTP.
"""
import re

from django.contrib.auth.models import User
from django.core import mail
from django.test import TestCase, override_settings
from rest_framework.test import APIClient


@override_settings(EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend')
class PasswordResetTests(TestCase):
    def setUp(self):
        # Le throttle password_reset (10/h) compte dans le cache partagé du
        # process de test: purge pour que chaque test parte de zéro.
        from django.core.cache import cache
        cache.clear()
        self.client_api = APIClient()
        self.user = User.objects.create_user(
            "resetme", email="reset@example.com", password="Ancien!pass1")

    def _request(self, email):
        return self.client_api.post(
            "/api/auth/password-reset/", {"email": email}, format="json")

    def _extract_link(self):
        body = mail.outbox[-1].body
        m = re.search(r"uid=([^&\s]+)&token=([^\s]+)", body)
        assert m, body
        return m.group(1), m.group(2)

    def _confirm(self, uid, token, password):
        return self.client_api.post(
            "/api/auth/password-reset/confirm/",
            {"uid": uid, "token": token, "new_password": password}, format="json")

    def test_request_sends_email_with_reset_link(self):
        resp = self._request("reset@example.com")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("reset-password?uid=", mail.outbox[0].body)
        self.assertEqual(mail.outbox[0].to, ["reset@example.com"])

    def test_unknown_email_same_response_no_mail(self):
        # Anti-énumération: même 200 + même message, zéro mail.
        resp = self._request("inconnu@example.com")
        self.assertEqual(resp.status_code, 200)
        known = self._request("reset@example.com")
        self.assertEqual(resp.data, known.data)
        self.assertEqual(len(mail.outbox), 1)  # seul le connu a reçu un mail

    def test_full_flow_changes_password(self):
        self._request("reset@example.com")
        uid, token = self._extract_link()
        resp = self._confirm(uid, token, "Nouveau!pass2")
        self.assertEqual(resp.status_code, 200, resp.data)
        # l'ancien mot de passe ne marche plus, le nouveau oui (login par email)
        login = self.client_api.post("/api/auth/login/", {
            "username": "reset@example.com", "password": "Nouveau!pass2"},
            format="json")
        self.assertEqual(login.status_code, 200)
        old = self.client_api.post("/api/auth/login/", {
            "username": "reset@example.com", "password": "Ancien!pass1"},
            format="json")
        self.assertEqual(old.status_code, 401)

    def test_token_is_single_use(self):
        self._request("reset@example.com")
        uid, token = self._extract_link()
        self.assertEqual(self._confirm(uid, token, "Nouveau!pass2").status_code, 200)
        # rejouer le même lien: refusé (le hash du mdp a changé -> token mort)
        self.assertEqual(self._confirm(uid, token, "Encore!pass3").status_code, 400)

    def test_bad_token_rejected(self):
        self._request("reset@example.com")
        uid, _ = self._extract_link()
        self.assertEqual(self._confirm(uid, "token-bidon", "Nouveau!pass2").status_code, 400)

    def test_garbage_uid_rejected(self):
        self.assertEqual(self._confirm("zzz", "tok", "Nouveau!pass2").status_code, 400)

    def test_weak_password_rejected(self):
        self._request("reset@example.com")
        uid, token = self._extract_link()
        resp = self._confirm(uid, token, "123")
        self.assertEqual(resp.status_code, 400)
        # et le mot de passe n'a PAS changé
        login = self.client_api.post("/api/auth/login/", {
            "username": "resetme", "password": "Ancien!pass1"}, format="json")
        self.assertEqual(login.status_code, 200)

    def test_inactive_account_gets_no_mail(self):
        self.user.is_active = False
        self.user.save(update_fields=["is_active"])
        self._request("reset@example.com")
        self.assertEqual(len(mail.outbox), 0)

    def test_duplicate_emails_each_get_their_link(self):
        User.objects.create_user(
            "doublon", email="reset@example.com", password="Autre!pass9")
        self._request("reset@example.com")
        self.assertEqual(len(mail.outbox), 2)

    def test_missing_email_400(self):
        self.assertEqual(self._request("").status_code, 400)
