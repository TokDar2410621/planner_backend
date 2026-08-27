"""
La piece jointe et l'ancrage temporel.

Deux defauts trouves en faisant tourner v2 de bout en bout contre un vrai
modele le 2026-08-26, aucun des deux visible en test unitaire:

1. process_message_stream acceptait un `attachment` et ne s'en servait JAMAIS.
   A « voici mon horaire, gere ca » avec un PDF joint contenant Physique et
   Anglais, l'agent repondait « J'ai jete un oeil a ton horaire » puis
   decrivait le planning DEJA en base. Le document etait perdu en silence, et
   la reponse laissait croire le contraire. L'envoi d'horaire est le premier
   chemin d'entree du produit.

2. A « priorise mes revisions », le modele a place 7 seances dont 4 dans le
   PASSE (deux la veille, deux le jour meme a des heures ecoulees) et les a
   toutes annoncees comme calees. La reconciliation comparait la date demandee
   a la date obtenue, jamais la date obtenue a MAINTENANT.
"""
from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone

from core.models import UploadedDocument
from services.agent.tools.base import ToolResult
from services.agent_v2.redaction import ReponseDire
from services.agent_v2.registre import Registre


class PieceJointeTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='doc', password='x')
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def _message_vu_par_agir(self, attachment=None, texte="voici mon horaire, gere ca"):
        vus = {}

        def _agir(self_agent, user, message, registre):
            vus['message'] = message
            return None

        with patch.object(self.Agent, '_agir', _agir), \
             patch.object(self.Agent, '_dire', return_value=ReponseDire(ouverture="Ok.")):
            self.Agent().process_message(self.user, texte, attachment)
        return vus['message']

    def test_le_contenu_du_document_atteint_le_modele(self):
        doc = UploadedDocument.objects.create(
            user=self.user, file_name='horaire.pdf',
            document_type='course_schedule', processed=True,
            extracted_data={'courses': [{'name': 'Physique', 'day': 'mercredi'}]})
        message = self._message_vu_par_agir(doc)
        self.assertIn('Physique', message)
        self.assertIn('horaire.pdf', message)

    def test_un_document_encore_en_analyse_est_annonce_comme_tel(self):
        """Ne jamais laisser croire que le contenu est lu quand il ne l'est pas."""
        doc = UploadedDocument.objects.create(
            user=self.user, file_name='lent.pdf', processed=False)
        with self.settings(ATTACHMENT_WAIT_SECONDS=0):
            message = self._message_vu_par_agir(doc)
        self.assertIn('ANALYSE', message.upper())

    def test_sans_piece_jointe_un_import_recent_est_quand_meme_signale(self):
        """« gere ca » au tour suivant ne doit pas produire un refus du type
        « je ne peux pas traiter de document »."""
        from core.models import RecurringBlock
        doc = UploadedDocument.objects.create(
            user=self.user, file_name='recent.pdf',
            document_type='course_schedule', processed=True,
            extracted_data={'courses': [{'name': 'Anglais'}]})
        # Le contexte d'import se calcule depuis les blocs REELLEMENT issus du
        # document, pas depuis le document seul: un import qui n'a rien produit
        # ne doit rien annoncer.
        RecurringBlock.objects.create(
            user=self.user, title='Anglais', block_type='course', day_of_week=3,
            start_time='14:00', end_time='16:00', source_document=doc)
        message = self._message_vu_par_agir(None, texte="c'est bon ?")
        self.assertIn('IMPORT', message.upper())
        self.assertIn('Anglais', message)

    def test_sans_document_le_message_reste_intact(self):
        """Contre-epreuve: on n'injecte rien quand il n'y a rien a injecter."""
        self.assertEqual(self._message_vu_par_agir(None, texte="bonjour"), "bonjour")


class AncrageTemporelTests(TestCase):
    """Une seance placee dans le passe est inutilisable, et l'annoncer comme
    calee est un mensonge que la validation des references ne voit pas: la
    mutation a REELLEMENT eu lieu, elle est juste inutile."""

    def _registre_avec_date(self, date, heure_fin="23:59"):
        registre = Registre()
        registre.ajouter('schedule_task_at', {'date': str(date)}, ToolResult(
            success=True,
            data={'scheduled_block': {'date': str(date), 'end_time': heure_fin}},
            message=f"Revision calee le {date}"))
        return registre

    def test_une_seance_placee_hier_produit_un_ecart(self):
        from services.agent_v2.reconciliation import detecter_ecarts
        registre = self._registre_avec_date(timezone.localdate() - timedelta(days=1))
        detecter_ecarts(registre)
        self.assertEqual(len(registre.ecarts), 1)
        self.assertIn('passe', registre.ecarts[0].description.lower())
        self.assertIn('PAS ete annule', registre.ecarts[0].description)

    def test_une_seance_a_venir_ne_produit_aucun_ecart(self):
        from services.agent_v2.reconciliation import detecter_ecarts
        registre = self._registre_avec_date(timezone.localdate() + timedelta(days=2))
        detecter_ecarts(registre)
        self.assertEqual(registre.ecarts, [])

    def test_une_seance_du_jour_deja_terminee_produit_un_ecart(self):
        """Le cas reel: il etait 23h30 et le modele a cale 09:00-12:00 le jour
        meme. La date seule ne suffit donc pas, il faut l'heure de fin."""
        from services.agent_v2.reconciliation import detecter_ecarts
        registre = self._registre_avec_date(timezone.localdate(), heure_fin="00:01")
        detecter_ecarts(registre)
        self.assertEqual(len(registre.ecarts), 1)
