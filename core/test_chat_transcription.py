"""
La dictee du bouton micro: POST /chat/transcrire/ rend {texte} et RIEN
d'autre. Aucun tour d'agent, aucune ecriture: la voix n'est qu'une porte
d'entree vers le pipeline texte, l'utilisateur relit avant d'envoyer.

Le fournisseur est simule partout: aucun test ne paie un appel Gemini.
"""
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from core.models import ConversationMessage
from services.transcription import TranscriptionIndisponible


def _audio(mime="audio/webm", octets=b"x" * 128, nom="dictee.webm"):
    return SimpleUploadedFile(nom, octets, content_type=mime)


class TranscriptionViewTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(username="dictee", password="x")
        profil = self.user.profile
        profil.ai_consent_at = timezone.now()
        profil.save(update_fields=["ai_consent_at"])
        self.client_api = APIClient()
        self.client_api.force_authenticate(self.user)

    def _poster(self, **kwargs):
        return self.client_api.post("/api/chat/transcrire/",
                                    {"audio": _audio(**kwargs)}, format="multipart")

    def test_transcrit_et_rend_le_texte(self):
        with patch("services.transcription.transcrire_audio",
                   return_value="Ajoute mon gym jeudi à 18 h") as transcrire:
            reponse = self._poster()
        self.assertEqual(reponse.status_code, 200)
        self.assertEqual(reponse.data["texte"], "Ajoute mon gym jeudi à 18 h")
        donnees, mime = transcrire.call_args.args
        self.assertEqual(mime, "audio/webm")
        self.assertEqual(donnees, b"x" * 128)

    def test_le_mime_de_safari_passe_meme_avec_codecs(self):
        with patch("services.transcription.transcrire_audio", return_value="ok"):
            reponse = self._poster(mime="audio/mp4;codecs=mp4a.40.2", nom="d.m4a")
        self.assertEqual(reponse.status_code, 200, reponse.data)

    def test_aucun_tour_d_agent_ni_message_persiste(self):
        with patch("services.transcription.transcrire_audio", return_value="bonjour"):
            self._poster()
        self.assertEqual(ConversationMessage.objects.filter(user=self.user).count(), 0)

    def test_sans_consentement_ia_403(self):
        profil = self.user.profile
        profil.ai_consent_at = None
        profil.save(update_fields=["ai_consent_at"])
        with patch("services.transcription.transcrire_audio", return_value="x") as transcrire:
            reponse = self._poster()
        self.assertEqual(reponse.status_code, 403)
        transcrire.assert_not_called()

    def test_fichier_absent_400(self):
        reponse = self.client_api.post("/api/chat/transcrire/", {}, format="multipart")
        self.assertEqual(reponse.status_code, 400)

    def test_format_inconnu_400(self):
        with patch("services.transcription.transcrire_audio", return_value="x") as transcrire:
            reponse = self._poster(mime="video/mp4", nom="film.mp4")
        self.assertEqual(reponse.status_code, 400)
        transcrire.assert_not_called()

    def test_trop_gros_400(self):
        with patch("services.transcription.transcrire_audio", return_value="x") as transcrire:
            reponse = self._poster(octets=b"x" * (5 * 1024 * 1024 + 1))
        self.assertEqual(reponse.status_code, 400)
        transcrire.assert_not_called()

    def test_fournisseur_en_panne_503_generique(self):
        with patch("services.transcription.transcrire_audio",
                   side_effect=TranscriptionIndisponible("secret: https://cle@api")):
            reponse = self._poster()
        self.assertEqual(reponse.status_code, 503)
        # Jamais le detail du fournisseur (une HTTPError porte l'URL entiere).
        self.assertNotIn("cle", str(reponse.data))
        self.assertIn("indisponible", reponse.data["error"])

    def test_anonyme_refuse(self):
        anonyme = APIClient()
        reponse = anonyme.post("/api/chat/transcrire/",
                               {"audio": _audio()}, format="multipart")
        self.assertIn(reponse.status_code, (401, 403))
