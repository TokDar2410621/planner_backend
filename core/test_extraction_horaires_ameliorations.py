"""Ameliorations de l'extraction d'horaires (5 points, 2026-09-27).

1. HEIC/HEIF accepte a l'upload (photos iPhone) au lieu du 400.
2. "M" seul (mardi/mercredi) resolu par heuristique, jamais jete en silence.
3. parse_time_string: minutes EXACTES (fini l'arrondi au quart d'heure),
   accepte aussi "8h30"/"8h".
4. Canal d'avertissements dans extracted_data['_warnings'], downscale des
   photos avant vision, EXTRACTION_VERSION=3.
5. Attente agent: sortie anticipee si l'analyse a echoue + message franc.
"""
import os
import tempfile
from datetime import time as dt_time
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from rest_framework.exceptions import ValidationError

import services.document_processor as dp
from core.models import RecurringBlock, UploadedDocument
from core.validators import (
    is_heic_header,
    sniff_kind,
    validate_upload_file,
)
from services.agent.agent import PlannerAgent
from services.document_processor import (
    MAX_IMAGE_SIDE_PX,
    DocumentProcessor,
    compute_block_confidence,
    parse_time_string,
)

_HEIC = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00" + b"\x00" * 32
_HEIC_MIF1 = b"\x00\x00\x00\x18ftypmif1\x00\x00\x00\x00" + b"\x00" * 32
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _agent_sans_llm():
    """_build_attachment_context n'utilise que la piece jointe."""
    return PlannerAgent.__new__(PlannerAgent)


class HeicUploadTests(TestCase):
    def test_heic_accepte(self):
        f = SimpleUploadedFile("photo.heic", _HEIC, content_type="image/heic")
        self.assertEqual(validate_upload_file(f), "heic")

    def test_heif_accepte(self):
        f = SimpleUploadedFile("photo.heif", _HEIC, content_type="image/heif")
        self.assertEqual(validate_upload_file(f), "heic")

    def test_heic_marque_mif1_acceptee(self):
        f = SimpleUploadedFile("photo.heic", _HEIC_MIF1, content_type="image/heic")
        self.assertEqual(validate_upload_file(f), "heic")

    def test_heic_contenu_incoherent_rejete(self):
        f = SimpleUploadedFile("photo.heic", _PNG, content_type="image/heic")
        with self.assertRaises(ValidationError):
            validate_upload_file(f)

    def test_sniff_kind_heic_vers_image(self):
        self.assertEqual(sniff_kind(_HEIC[:16]), "image")

    def test_is_heic_header(self):
        self.assertTrue(is_heic_header(_HEIC[:16]))
        self.assertFalse(is_heic_header(_PNG[:16]))


class ParseTimeStringTests(TestCase):
    def test_minutes_exactes_pas_d_arrondi(self):
        # Avant: 08:07 -> 08:00 (arrondi silencieux au quart d'heure).
        self.assertEqual(parse_time_string("08:07"), dt_time(8, 7))
        self.assertEqual(parse_time_string("14:52"), dt_time(14, 52))

    def test_formats_francais(self):
        self.assertEqual(parse_time_string("8h30"), dt_time(8, 30))
        self.assertEqual(parse_time_string("8h"), dt_time(8, 0))
        self.assertEqual(parse_time_string("14h"), dt_time(14, 0))

    def test_invalides(self):
        self.assertIsNone(parse_time_string(""))
        self.assertIsNone(parse_time_string("n'importe quoi"))
        self.assertIsNone(parse_time_string("25:00"))
        self.assertIsNone(parse_time_string("10:75"))


class ResolveDayTests(TestCase):
    def test_jour_normal(self):
        self.assertEqual(DocumentProcessor._resolve_day("lundi", []), (0, False))
        self.assertEqual(DocumentProcessor._resolve_day(" Mer. ", []), (2, False))

    def test_m_avec_frere_mercredi_explicite(self):
        # "M" + "mercredi" dans le meme doc -> "M" = mardi.
        self.assertEqual(
            DocumentProcessor._resolve_day("M", ["mercredi", "lundi"]), (1, True)
        )

    def test_m_avec_frere_mardi_explicite(self):
        self.assertEqual(
            DocumentProcessor._resolve_day("m", ["mar", "jeu"]), (2, True)
        )

    def test_m_ambigu_non_resolu(self):
        # Grille L M M J V: deux "M", aucun explicite -> non resolu, mais
        # signale (ambigu=True) au lieu d'etre jete en silence.
        self.assertEqual(DocumentProcessor._resolve_day("M", ["M", "L"]), (None, True))
        self.assertEqual(DocumentProcessor._resolve_day("m", []), (None, True))

    def test_inconnu_non_ambigu(self):
        self.assertEqual(DocumentProcessor._resolve_day("xyz", []), (None, False))
        self.assertEqual(DocumentProcessor._resolve_day("", []), (None, False))

    def test_jour_ambigu_force_pending(self):
        conf = compute_block_confidence(
            start_defaulted=False,
            end_defaulted=False,
            title_generic=False,
            day_ambiguous=True,
        )
        self.assertLess(conf, 0.6)


class LoadImageTests(TestCase):
    def test_downscale_grande_photo(self):
        from PIL import Image

        proc = DocumentProcessor()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            Image.new("RGB", (3000, 100), "white").save(tmp.name, "PNG")
            try:
                img = proc._load_image(tmp.name)
                self.assertLessEqual(max(img.size), MAX_IMAGE_SIDE_PX)
            finally:
                os.unlink(tmp.name)

    def test_heic_sans_librairie_erreur_explicite(self):
        proc = DocumentProcessor()
        with tempfile.NamedTemporaryFile(suffix=".heic", delete=False) as tmp:
            tmp.write(_HEIC)
            tmp.flush()
            try:
                with patch.object(dp, "HEIF_AVAILABLE", False):
                    with self.assertRaisesRegex(RuntimeError, "pillow-heif"):
                        proc._load_image(tmp.name)
            finally:
                os.unlink(tmp.name)

    def test_version_extraction_bump(self):
        self.assertEqual(DocumentProcessor.EXTRACTION_VERSION, 3)


class ExtractionWarningsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="warn_user", password="x")
        self.proc = DocumentProcessor()

    def _doc(self):
        return UploadedDocument.objects.create(
            user=self.user,
            file_name="horaire.png",
            file=SimpleUploadedFile("horaire.png", _PNG, content_type="image/png"),
        )

    def test_jour_inconnu_ignore_avec_avertissement(self):
        doc = self._doc()
        data = {"courses": [
            {"name": "Maths", "day": "M", "start_time": "08:00", "end_time": "10:00"},
        ]}
        blocks = self.proc._create_recurring_blocks(doc, data)
        self.assertEqual(blocks, [])
        doc.refresh_from_db()
        warnings = doc.extracted_data.get("_warnings", [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("M", warnings[0])
        self.assertIn("Maths", warnings[0])

    def test_jour_ambigu_resolu_heuristique_vers_pending(self):
        doc = self._doc()
        data = {"courses": [
            {"name": "Physique", "day": "m", "start_time": "08:00", "end_time": "10:00"},
            {"name": "Chimie", "day": "mercredi", "start_time": "10:00", "end_time": "12:00"},
        ]}
        blocks = self.proc._create_recurring_blocks(doc, data)
        physique = RecurringBlock.all_objects.get(
            source_document=doc, title="Physique"
        )
        # "m" + "mercredi" explicite -> mardi (1), mais en pending.
        self.assertEqual(physique.day_of_week, 1)
        self.assertEqual(physique.status, RecurringBlock.STATUS_PENDING)
        doc.refresh_from_db()
        warnings = doc.extracted_data.get("_warnings", [])
        self.assertTrue(any("ambigu" in w for w in warnings))

    def test_heure_defaut_avec_avertissement(self):
        doc = self._doc()
        data = {"courses": [
            {"name": "Bio", "day": "lundi", "start_time": "n/a", "end_time": "10:00"},
        ]}
        self.proc._create_recurring_blocks(doc, data)
        bloc = RecurringBlock.all_objects.get(source_document=doc, title="Bio")
        self.assertEqual(bloc.start_time, dt_time(9, 0))
        doc.refresh_from_db()
        warnings = doc.extracted_data.get("_warnings", [])
        self.assertTrue(any("09:00 par défaut" in w for w in warnings))


class AttachmentFailureContextTests(TestCase):
    def test_echec_franc_pas_de_faux_retard(self):
        agent = _agent_sans_llm()
        doc = UploadedDocument(
            file_name="scan.pdf", processed=False,
            processing_error="Le traitement du document a échoué.",
        )
        ctx = agent._build_attachment_context(doc)
        self.assertIn("ÉCHOUÉ", ctx)
        self.assertNotIn("plus de temps que prévu", ctx)

    def test_encore_en_analyse_sans_erreur(self):
        agent = _agent_sans_llm()
        doc = UploadedDocument(
            file_name="scan.pdf", processed=False, processing_error=None,
        )
        ctx = agent._build_attachment_context(doc)
        self.assertIn("ENCORE EN ANALYSE", ctx)
