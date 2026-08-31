"""Reveil silencieux iOS (APNs): jeton, envoi, lissage, signaux, endpoints.

Aucune connexion reseau: httpx.Client est remplace par un double qui rejoue
des reponses, et la cle ES256 est generee ici meme.
"""
import time
from datetime import time as dt_time, timedelta
from unittest.mock import patch

import httpx
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APITestCase

from core.models import (
    AppareilPush,
    RecurringBlock,
    RecurringBlockException,
    ReveilPlanning,
    ScheduledBlock,
    Task,
)
from services import apns

_CLE = ec.generate_private_key(ec.SECP256R1())
CLE_PRIVEE_PEM = _CLE.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption(),
).decode()
CLE_PUBLIQUE_PEM = _CLE.public_key().public_bytes(
    serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
).decode()

CONFIG = dict(
    APNS_TEAM_ID="TEAM0000AB",
    APNS_KEY_ID="KEY0000XYZ",
    APNS_AUTH_KEY=CLE_PRIVEE_PEM,
    APNS_BUNDLE_ID="com.tokamdarius.planner",
)
TOKEN = "ab" * 32


class _ClientFactice:
    """Remplace httpx.Client: enregistre les appels, rejoue les reponses."""

    def __init__(self, reponses):
        self.reponses = list(reponses)
        self.appels = []
        self.constructions = []

    def __call__(self, **kwargs):
        self.constructions.append(kwargs)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, **kwargs):
        self.appels.append((url, kwargs))
        reponse = self.reponses.pop(0)
        if isinstance(reponse, Exception):
            raise reponse
        return reponse


def _reponse(statut, motif=None):
    if motif is None:
        return httpx.Response(statut)
    return httpx.Response(statut, json={"reason": motif})


def _client(*reponses):
    """(patcheur, client): `with patcheur:` remplace httpx.Client par le double."""
    client = _ClientFactice(reponses)
    return patch.object(apns.httpx, "Client", client), client


@override_settings(**CONFIG)
class JetonApnsTest(SimpleTestCase):
    def setUp(self):
        apns.invalider_jeton()

    def tearDown(self):
        apns.invalider_jeton()

    def test_configure_seulement_avec_les_trois_cles(self):
        self.assertTrue(apns.apns_configured())
        for vide in ("APNS_TEAM_ID", "APNS_KEY_ID", "APNS_AUTH_KEY"):
            with override_settings(**{vide: ""}):
                self.assertFalse(apns.apns_configured())

    def test_jeton_es256_valide(self):
        jeton = apns.jeton_apns()
        entete = jwt.get_unverified_header(jeton)
        self.assertEqual(entete["alg"], "ES256")
        self.assertEqual(entete["kid"], "KEY0000XYZ")
        claims = jwt.decode(jeton, CLE_PUBLIQUE_PEM, algorithms=["ES256"])
        self.assertEqual(claims["iss"], "TEAM0000AB")
        self.assertAlmostEqual(claims["iat"], time.time(), delta=5)

    def test_cle_avec_sauts_de_ligne_litteraux(self):
        with override_settings(APNS_AUTH_KEY=CLE_PRIVEE_PEM.replace("\n", "\\n")):
            jeton = apns.jeton_apns()
        jwt.decode(jeton, CLE_PUBLIQUE_PEM, algorithms=["ES256"])

    def test_jeton_en_cache_puis_regenere_apres_50_minutes(self):
        debut = time.time() - 2 * 3600
        with patch.object(apns, "_horloge", return_value=debut):
            premier = apns.jeton_apns()
            self.assertEqual(apns.jeton_apns(), premier)
        with patch.object(apns, "_horloge", return_value=debut + 49 * 60):
            self.assertEqual(apns.jeton_apns(), premier)
        with patch.object(apns, "_horloge", return_value=debut + 50 * 60):
            nouveau = apns.jeton_apns()
        self.assertNotEqual(nouveau, premier)
        claims = jwt.decode(nouveau, CLE_PUBLIQUE_PEM, algorithms=["ES256"])
        self.assertEqual(claims["iat"], int(debut + 50 * 60))

    def test_invalider_force_un_nouveau_jeton(self):
        with patch.object(apns, "_horloge", return_value=time.time() - 60):
            premier = apns.jeton_apns()
        apns.invalider_jeton()
        self.assertNotEqual(apns.jeton_apns(), premier)


@override_settings(**CONFIG)
class EnvoyerReveilTest(TestCase):
    def setUp(self):
        apns.invalider_jeton()
        self.user = User.objects.create_user("ios", password="pw-ios-12345")
        self.appareil = AppareilPush.objects.create(user=self.user, token=TOKEN)

    def tearDown(self):
        apns.invalider_jeton()

    def test_en_tetes_et_corps_exacts(self):
        patcheur, client = _client(_reponse(200))
        with patcheur:
            self.assertTrue(apns.envoyer_reveil(self.appareil, raison="planning"))
        self.assertEqual(client.constructions, [{"http2": True, "timeout": 10}])
        url, kwargs = client.appels[0]
        self.assertEqual(url, f"https://api.push.apple.com/3/device/{TOKEN}")
        self.assertEqual(kwargs["json"], {"aps": {"content-available": 1}, "raison": "planning"})
        en_tetes = kwargs["headers"]
        self.assertEqual(
            set(en_tetes),
            {"authorization", "apns-topic", "apns-push-type", "apns-priority", "apns-expiration"},
        )
        self.assertEqual(en_tetes["authorization"], f"bearer {apns.jeton_apns()}")
        self.assertEqual(en_tetes["apns-topic"], "com.tokamdarius.planner")
        self.assertEqual(en_tetes["apns-push-type"], "background")
        self.assertEqual(en_tetes["apns-priority"], "5")
        self.assertEqual(en_tetes["apns-expiration"], "0")

    def test_200_efface_la_derniere_erreur(self):
        AppareilPush.objects.filter(pk=self.appareil.pk).update(derniere_erreur="429 TooManyRequests")
        self.appareil.refresh_from_db()
        patcheur, _ = _client(_reponse(200))
        with patcheur:
            self.assertTrue(apns.envoyer_reveil(self.appareil))
        self.appareil.refresh_from_db()
        self.assertEqual(self.appareil.derniere_erreur, "")

    def test_bad_device_token_production_bascule_en_sandbox_et_reussit(self):
        patcheur, client = _client(_reponse(400, "BadDeviceToken"), _reponse(200))
        with patcheur:
            self.assertTrue(apns.envoyer_reveil(self.appareil))
        self.assertEqual(len(client.appels), 2)
        self.assertTrue(client.appels[0][0].startswith("https://api.push.apple.com/3/device/"))
        self.assertTrue(client.appels[1][0].startswith("https://api.sandbox.push.apple.com/3/device/"))
        self.appareil.refresh_from_db()
        self.assertEqual(self.appareil.environnement, "sandbox")

    def test_bad_device_token_en_sandbox_supprime_l_appareil(self):
        AppareilPush.objects.filter(pk=self.appareil.pk).update(environnement="sandbox")
        self.appareil.refresh_from_db()
        patcheur, client = _client(_reponse(400, "BadDeviceToken"))
        with patcheur:
            self.assertFalse(apns.envoyer_reveil(self.appareil))
        self.assertEqual(len(client.appels), 1)
        self.assertEqual(AppareilPush.objects.count(), 0)

    def test_410_unregistered_supprime_l_appareil(self):
        patcheur, _ = _client(_reponse(410, "Unregistered"))
        with patcheur:
            self.assertFalse(apns.envoyer_reveil(self.appareil))
        self.assertEqual(AppareilPush.objects.count(), 0)

    def test_403_invalide_le_jeton_et_retourne_false(self):
        apns.jeton_apns()
        self.assertIsNotNone(apns._jeton_cache["valeur"])
        patcheur, _ = _client(_reponse(403, "ExpiredProviderToken"))
        with patcheur, self.assertLogs("services.apns", level="ERROR"):
            self.assertFalse(apns.envoyer_reveil(self.appareil))
        self.assertIsNone(apns._jeton_cache["valeur"])
        self.appareil.refresh_from_db()
        self.assertEqual(self.appareil.derniere_erreur, "403 ExpiredProviderToken")

    def test_erreur_reseau_retourne_false_sans_lever(self):
        patcheur, _ = _client(httpx.ConnectError("injoignable"))
        with patcheur, self.assertLogs("services.apns", level="WARNING"):
            self.assertFalse(apns.envoyer_reveil(self.appareil))
        self.appareil.refresh_from_db()
        self.assertEqual(self.appareil.derniere_erreur, "reseau: ConnectError")
        self.assertEqual(AppareilPush.objects.count(), 1)

    def test_429_et_5xx_retournent_false_et_gardent_l_appareil(self):
        for statut, motif in ((429, "TooManyRequests"), (503, "ServiceUnavailable")):
            patcheur, client = _client(_reponse(statut, motif))
            with patcheur, self.assertLogs("services.apns", level="WARNING"):
                self.assertFalse(apns.envoyer_reveil(self.appareil))
            self.assertEqual(len(client.appels), 1)
            self.appareil.refresh_from_db()
            self.assertEqual(self.appareil.derniere_erreur, f"{statut} {motif}")
        self.assertEqual(AppareilPush.objects.count(), 1)

    def test_non_configure_n_envoie_rien(self):
        patcheur, client = _client()
        with override_settings(APNS_AUTH_KEY=""), patcheur:
            self.assertFalse(apns.envoyer_reveil(self.appareil))
        self.assertEqual(client.appels, [])


@override_settings(**CONFIG)
class LissageReveilTest(TestCase):
    def setUp(self):
        apns.invalider_jeton()
        apns._minuteries.clear()
        self.user = User.objects.create_user("lisse", password="pw-lisse-12345")
        self.appareil = AppareilPush.objects.create(user=self.user, token="cd" * 32)

    def tearDown(self):
        apns.invalider_jeton()
        apns._minuteries.clear()

    def test_deux_ecritures_rapprochees_un_envoi_puis_un_rattrapage(self):
        patcheur, client = _client(_reponse(200), _reponse(200))
        with patcheur, patch.object(apns, "_armer_rattrapage") as armer:
            self.assertEqual(apns.reveiller_utilisateur(self.user), 1)
            self.assertEqual(apns.reveiller_utilisateur(self.user), 0)
            self.assertEqual(len(client.appels), 1)
            armer.assert_called_once_with(self.user.pk)
            reveil = ReveilPlanning.objects.get(user=self.user)
            self.assertGreater(reveil.demande_a, reveil.envoye_a)

            self.assertEqual(apns.rattraper_reveil(self.user.pk), 1)
            self.assertEqual(len(client.appels), 2)
            # Idempotent: un second rattrapage n'envoie rien.
            self.assertEqual(apns.rattraper_reveil(self.user.pk), 0)
            self.assertEqual(len(client.appels), 2)
        reveil.refresh_from_db()
        self.assertLessEqual(reveil.demande_a, reveil.envoye_a)

    def test_envoi_immediat_quand_le_dernier_date_de_plus_de_10_s(self):
        ReveilPlanning.objects.create(user=self.user, envoye_a=timezone.now() - timedelta(seconds=11))
        patcheur, client = _client(_reponse(200))
        with patcheur, patch.object(apns, "_armer_rattrapage") as armer:
            self.assertEqual(apns.reveiller_utilisateur(self.user), 1)
        self.assertEqual(len(client.appels), 1)
        armer.assert_not_called()

    def test_tous_les_appareils_de_l_utilisateur(self):
        AppareilPush.objects.create(user=self.user, token="ef" * 32)
        patcheur, client = _client(_reponse(200), _reponse(200))
        with patcheur:
            self.assertEqual(apns.reveiller_utilisateur(self.user), 2)
        self.assertEqual(len(client.appels), 2)

    def test_sans_appareil_aucune_ligne_creee(self):
        sans = User.objects.create_user("vide", password="pw-vide-12345")
        patcheur, client = _client()
        with patcheur:
            self.assertEqual(apns.reveiller_utilisateur(sans), 0)
        self.assertEqual(client.appels, [])
        self.assertFalse(ReveilPlanning.objects.filter(user=sans).exists())

    def test_non_configure_zero_requete(self):
        with override_settings(APNS_KEY_ID=""), self.assertNumQueries(0):
            self.assertEqual(apns.reveiller_utilisateur(self.user), 0)

    def test_rattraper_tous_les_reveils(self):
        autre = User.objects.create_user("autre", password="pw-autre-12345")
        AppareilPush.objects.create(user=autre, token="01" * 32)
        frais = User.objects.create_user("frais", password="pw-frais-12345")
        AppareilPush.objects.create(user=frais, token="02" * 32)
        deja = User.objects.create_user("deja", password="pw-deja-12345")
        AppareilPush.objects.create(user=deja, token="03" * 32)
        maintenant = timezone.now()
        # Jamais envoye et assez vieux: a rattraper.
        ReveilPlanning.objects.create(user=self.user, demande_a=maintenant - timedelta(seconds=6))
        # Demande posterieure au dernier envoi: a rattraper.
        ReveilPlanning.objects.create(
            user=autre,
            demande_a=maintenant - timedelta(seconds=30),
            envoye_a=maintenant - timedelta(seconds=60),
        )
        # Trop frais: le Timer du processus s'en charge.
        ReveilPlanning.objects.create(user=frais, demande_a=maintenant - timedelta(seconds=1))
        # Deja envoye.
        ReveilPlanning.objects.create(
            user=deja,
            demande_a=maintenant - timedelta(seconds=60),
            envoye_a=maintenant - timedelta(seconds=30),
        )
        patcheur, client = _client(_reponse(200), _reponse(200))
        with patcheur:
            self.assertEqual(apns.rattraper_tous_les_reveils(), 2)
        self.assertEqual(len(client.appels), 2)
        tokens = sorted(url.rsplit("/", 1)[1] for url, _ in client.appels)
        self.assertEqual(tokens, sorted(["01" * 32, "cd" * 32]))
        with patcheur:
            self.assertEqual(apns.rattraper_tous_les_reveils(), 0)
        self.assertEqual(len(client.appels), 2)

    def test_rattraper_tous_non_configure(self):
        ReveilPlanning.objects.create(user=self.user, demande_a=timezone.now() - timedelta(seconds=60))
        with override_settings(APNS_TEAM_ID=""), self.assertNumQueries(0):
            self.assertEqual(apns.rattraper_tous_les_reveils(), 0)

    def test_armer_rattrapage_dedoublonne_par_utilisateur(self):
        crees = []

        class MinuterieFactice:
            def __init__(self, delai, fonction, args=()):
                self.delai, self.fonction, self.args = delai, fonction, args
                self.daemon = False
                self.demarree = False
                crees.append(self)

            def start(self):
                self.demarree = True

        with patch.object(apns.threading, "Timer", MinuterieFactice):
            apns._armer_rattrapage(7)
            apns._armer_rattrapage(7)
            apns._armer_rattrapage(8)
        self.assertEqual([m.args for m in crees], [(7,), (8,)])
        self.assertEqual(crees[0].delai, 12)
        self.assertIs(crees[0].fonction, apns._rattrapage_en_thread)
        self.assertTrue(all(m.daemon and m.demarree for m in crees))

    def test_rattrapage_en_thread_libere_la_minuterie_et_la_connexion(self):
        apns._minuteries[9] = object()
        with patch.object(apns, "rattraper_reveil") as rattraper, patch.object(apns, "connection") as conn:
            apns._rattrapage_en_thread(9)
        rattraper.assert_called_once_with(9)
        conn.close.assert_called_once()
        self.assertNotIn(9, apns._minuteries)

    def test_rattrapage_en_thread_ne_leve_pas(self):
        with patch.object(apns, "rattraper_reveil", side_effect=RuntimeError("boum")), patch.object(
            apns, "connection"
        ), self.assertLogs("services.apns", level="ERROR"):
            apns._rattrapage_en_thread(9)


@override_settings(**CONFIG)
class SignauxReveilTest(TestCase):
    def setUp(self):
        apns.invalider_jeton()
        self.user = User.objects.create_user("signal", password="pw-signal-12345")

    def tearDown(self):
        apns.invalider_jeton()

    def _bloc(self):
        return RecurringBlock.objects.create(
            user=self.user,
            title="Cours",
            block_type="course",
            day_of_week=0,
            start_time=dt_time(9, 0),
            end_time=dt_time(10, 0),
        )

    def test_creation_reveille_apres_commit(self):
        with patch("core.signaux_reveil.reveiller_utilisateur") as reveiller:
            with self.captureOnCommitCallbacks(execute=True) as rappels:
                self._bloc()
                reveiller.assert_not_called()
            reveiller.assert_called_once_with(self.user, "planning")
        self.assertEqual(len(rappels), 1)

    def test_modification_et_suppression_reveillent(self):
        bloc = self._bloc()
        with patch("core.signaux_reveil.reveiller_utilisateur") as reveiller:
            with self.captureOnCommitCallbacks(execute=True):
                bloc.title = "Cours avance"
                bloc.save()
            with self.captureOnCommitCallbacks(execute=True):
                bloc.delete()
        self.assertEqual(reveiller.call_count, 2)

    def test_bloc_planifie_et_exception_reveillent(self):
        bloc = self._bloc()
        tache = Task.objects.create(user=self.user, title="Devoir")
        with patch("core.signaux_reveil.reveiller_utilisateur") as reveiller:
            with self.captureOnCommitCallbacks(execute=True):
                ScheduledBlock.objects.create(
                    user=self.user,
                    task=tache,
                    date=timezone.localdate(),
                    start_time=dt_time(14, 0),
                    end_time=dt_time(15, 0),
                )
            with self.captureOnCommitCallbacks(execute=True):
                RecurringBlockException.objects.create(
                    user=self.user, recurring_block=bloc, date=timezone.localdate()
                )
        self.assertEqual(reveiller.call_count, 2)
        for appel in reveiller.call_args_list:
            self.assertEqual(appel.args, (self.user, "planning"))

    def test_rien_quand_non_configure(self):
        with override_settings(APNS_AUTH_KEY=""), patch(
            "core.signaux_reveil.reveiller_utilisateur"
        ) as reveiller:
            with self.captureOnCommitCallbacks(execute=True) as rappels:
                bloc = self._bloc()
                bloc.delete()
        reveiller.assert_not_called()
        self.assertEqual(rappels, [])

    def test_une_erreur_du_reveil_ne_remonte_pas(self):
        with patch("core.signaux_reveil.reveiller_utilisateur", side_effect=RuntimeError("boum")):
            with self.assertLogs("core.signaux_reveil", level="ERROR"):
                with self.captureOnCommitCallbacks(execute=True):
                    self._bloc()

    def test_suppression_du_compte_ne_plante_pas(self):
        AppareilPush.objects.create(user=self.user, token="99" * 32)
        self._bloc()
        patcheur, client = _client()
        with patcheur, self.captureOnCommitCallbacks(execute=True):
            self.user.delete()
        self.assertEqual(client.appels, [])
        self.assertEqual(ReveilPlanning.objects.count(), 0)
        self.assertEqual(AppareilPush.objects.count(), 0)

    def test_reveil_reel_de_bout_en_bout(self):
        AppareilPush.objects.create(user=self.user, token="77" * 32)
        patcheur, client = _client(_reponse(200))
        with patcheur, self.captureOnCommitCallbacks(execute=True):
            self._bloc()
        self.assertEqual(len(client.appels), 1)
        self.assertTrue(client.appels[0][0].endswith("/3/device/" + "77" * 32))
        reveil = ReveilPlanning.objects.get(user=self.user)
        self.assertIsNotNone(reveil.envoye_a)


class AppareilPushEndpointTest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user("app", password="pw-app-12345")
        self.autre = User.objects.create_user("app2", password="pw-app2-12345")
        self.url = reverse("push-appareil")
        self.url_retrait = reverse("push-appareil-retirer")

    def test_sans_auth_401(self):
        r = self.client.post(self.url, {"token": TOKEN}, format="json")
        self.assertEqual(r.status_code, status.HTTP_401_UNAUTHORIZED)
        r = self.client.post(self.url_retrait, {"token": TOKEN}, format="json")
        self.assertEqual(r.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_sans_token_400(self):
        self.client.force_authenticate(self.user)
        for corps in ({}, {"token": ""}, {"token": "   "}, {"token": 42}):
            r = self.client.post(self.url, corps, format="json")
            self.assertEqual(r.status_code, status.HTTP_400_BAD_REQUEST, corps)
        self.assertEqual(AppareilPush.objects.count(), 0)

    def test_enregistre_puis_rafraichit(self):
        self.client.force_authenticate(self.user)
        r = self.client.post(
            self.url,
            {"token": TOKEN, "app_version": "1.2.0"},
            format="json",
            HTTP_USER_AGENT="Planner iOS",
        )
        self.assertEqual(r.status_code, status.HTTP_201_CREATED)
        self.assertTrue(r.data["created"])
        appareil = AppareilPush.objects.get(token=TOKEN)
        self.assertEqual(appareil.user, self.user)
        self.assertEqual(appareil.plateforme, "ios")
        self.assertEqual(appareil.environnement, "production")
        self.assertEqual(appareil.app_version, "1.2.0")
        self.assertEqual(appareil.user_agent, "Planner iOS")
        vu = appareil.last_seen_at

        r2 = self.client.post(
            self.url, {"token": TOKEN, "plateforme": "ios", "app_version": "1.3.0"}, format="json"
        )
        self.assertEqual(r2.status_code, status.HTTP_200_OK)
        self.assertFalse(r2.data["created"])
        self.assertEqual(AppareilPush.objects.count(), 1)
        appareil.refresh_from_db()
        self.assertEqual(appareil.app_version, "1.3.0")
        self.assertGreaterEqual(appareil.last_seen_at, vu)

    def test_un_token_qui_change_de_compte_suit_le_compte(self):
        self.client.force_authenticate(self.user)
        self.client.post(self.url, {"token": TOKEN}, format="json")
        self.client.force_authenticate(self.autre)
        r = self.client.post(self.url, {"token": TOKEN}, format="json")
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertEqual(AppareilPush.objects.count(), 1)
        self.assertEqual(AppareilPush.objects.get(token=TOKEN).user, self.autre)

    def test_retirer(self):
        AppareilPush.objects.create(user=self.user, token=TOKEN)
        AppareilPush.objects.create(user=self.autre, token="ff" * 32)
        self.client.force_authenticate(self.user)

        # Le token d'un autre compte n'est pas touche.
        r = self.client.post(self.url_retrait, {"token": "ff" * 32}, format="json")
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertEqual(r.data["deleted"], 0)
        self.assertEqual(AppareilPush.objects.count(), 2)

        r = self.client.post(self.url_retrait, {"token": TOKEN}, format="json")
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertEqual(r.data["deleted"], 1)
        self.assertFalse(AppareilPush.objects.filter(token=TOKEN).exists())

        # Sans token: 200 quand meme, rien de supprime.
        r = self.client.post(self.url_retrait, {}, format="json")
        self.assertEqual(r.status_code, status.HTTP_200_OK)
        self.assertEqual(r.data["deleted"], 0)


class ChevauxDeTroieUpdateTests(TestCase):
    """Les deux chemins qui ecrivent par QuerySet.update(), donc sans signal:
    ils doivent demander le reveil a la main."""

    def setUp(self):
        self.user = User.objects.create_user(username="maj", password="x")

    def test_confirmer_tous_les_blocs_reveille(self):
        RecurringBlock.objects.create(
            user=self.user, title="Cours", block_type="course", day_of_week=0,
            start_time=dt_time(9, 0), end_time=dt_time(10, 0),
            status=RecurringBlock.STATUS_PENDING,
        )
        from rest_framework.test import APIClient
        client = APIClient()
        client.force_authenticate(self.user)
        with patch("services.apns.reveiller_utilisateur") as reveiller:
            with self.captureOnCommitCallbacks(execute=True):
                reponse = client.post("/api/recurring-blocks/confirm_all/")
        self.assertEqual(reponse.status_code, 200)
        self.assertEqual(reponse.data["confirmed"], 1)
        reveiller.assert_called_once_with(self.user, "planning")

    def test_vider_le_planning_par_l_agent_reveille(self):
        RecurringBlock.objects.create(
            user=self.user, title="Cours", block_type="course", day_of_week=0,
            start_time=dt_time(9, 0), end_time=dt_time(10, 0),
        )
        from services.agent.tools.blocks import ClearAllBlocksTool
        with patch("services.apns.reveiller_utilisateur") as reveiller:
            with self.captureOnCommitCallbacks(execute=True):
                resultat = ClearAllBlocksTool().execute(self.user, confirm=True)
        self.assertTrue(resultat.success)
        reveiller.assert_called_once_with(self.user, "planning")


class JournauxSansJetonTests(SimpleTestCase):
    def test_httpx_ne_journalise_pas_les_urls(self):
        # httpx ecrit a INFO l'URL complete de chaque requete, et celle d'APNs
        # contient le jeton d'appareil: le logger doit rester a WARNING.
        import logging
        self.assertGreaterEqual(logging.getLogger("httpx").getEffectiveLevel(), logging.WARNING)
        self.assertGreaterEqual(logging.getLogger("httpcore").getEffectiveLevel(), logging.WARNING)
