"""Connexion par EMAIL au formulaire (bug attrapé par le parcours Playwright réel).

Le front envoie {username: <email>, password}; authenticate() n'authentifie que
par username -> tout compte pseudo+email était enfermé dehors. LoginView résout
maintenant l'email vers le(s) username(s); le mot de passe départage les
doublons d'email historiques.
"""
from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework.test import APIClient


class EmailLoginTests(TestCase):
    def setUp(self):
        self.client_api = APIClient()
        self.user = User.objects.create_user(
            "dariuspseudo", email="darius@example.com", password="Str0ng!pass")

    def _login(self, identifier, password):
        return self.client_api.post(
            "/api/auth/login/", {"username": identifier, "password": password},
            format="json")

    def test_login_with_email_succeeds(self):
        resp = self._login("darius@example.com", "Str0ng!pass")
        self.assertEqual(resp.status_code, 200, resp.data)
        self.assertIn("tokens", resp.data)
        self.assertEqual(resp.data["user"]["username"], "dariuspseudo")

    def test_login_with_email_case_insensitive(self):
        resp = self._login("DARIUS@Example.COM", "Str0ng!pass")
        self.assertEqual(resp.status_code, 200, resp.data)

    def test_login_with_username_still_works(self):
        resp = self._login("dariuspseudo", "Str0ng!pass")
        self.assertEqual(resp.status_code, 200, resp.data)

    def test_wrong_password_still_rejected(self):
        resp = self._login("darius@example.com", "mauvais-mdp")
        self.assertEqual(resp.status_code, 401)

    def test_unknown_email_rejected(self):
        resp = self._login("inconnu@example.com", "Str0ng!pass")
        self.assertEqual(resp.status_code, 401)

    def test_duplicate_email_password_disambiguates(self):
        # Doublon d'email historique: chaque compte garde son mot de passe;
        # celui qui matche gagne.
        User.objects.create_user(
            "autrecompte", email="darius@example.com", password="Autre!pass9")
        r1 = self._login("darius@example.com", "Str0ng!pass")
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r1.data["user"]["username"], "dariuspseudo")
        r2 = self._login("darius@example.com", "Autre!pass9")
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.data["user"]["username"], "autrecompte")

    def test_inactive_user_rejected_even_via_email(self):
        self.user.is_active = False
        self.user.save(update_fields=["is_active"])
        resp = self._login("darius@example.com", "Str0ng!pass")
        self.assertEqual(resp.status_code, 401)
