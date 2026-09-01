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

    def _ecarts_a(self, maintenant_txt, date, heure_fin):
        """Fige l'heure courante.

        Une premiere version de ces tests lisait l'horloge reelle et a casse au
        passage de minuit: a 00h00, aucun creneau « du jour deja termine » ne
        peut exister. Un test d'ancrage temporel qui depend de l'heure a
        laquelle on le lance ne prouve rien.
        """
        from datetime import datetime

        from services.agent_v2 import reconciliation

        faux = timezone.make_aware(
            datetime.strptime(maintenant_txt, "%Y-%m-%d %H:%M"),
            timezone.get_current_timezone())
        registre = self._registre_avec_date(date, heure_fin)
        with patch.object(reconciliation.timezone, 'localtime', return_value=faux):
            reconciliation.detecter_ecarts(registre)
        return registre.ecarts

    def test_une_seance_placee_hier_produit_un_ecart(self):
        ecarts = self._ecarts_a("2026-08-26 23:30", "2026-08-25", "12:00")
        self.assertEqual(len(ecarts), 1)
        self.assertIn('passe', ecarts[0].description.lower())
        self.assertIn('PAS ete annule', ecarts[0].description)

    def test_une_seance_a_venir_ne_produit_aucun_ecart(self):
        self.assertEqual(self._ecarts_a("2026-08-26 23:30", "2026-08-28", "11:00"), [])

    def test_une_seance_du_jour_deja_terminee_produit_un_ecart(self):
        """Le cas reel: il etait 23h30 et le modele a cale 09:00-12:00 le jour
        meme. La date seule ne suffit donc pas, il faut l'heure de fin."""
        self.assertEqual(len(self._ecarts_a("2026-08-26 23:30", "2026-08-26", "12:00")), 1)

    def test_une_seance_du_jour_ENCORE_a_venir_ne_produit_aucun_ecart(self):
        """Contre-epreuve du meme jour: sans elle, un detecteur qui signale
        tout ce qui porte la date du jour passerait les trois autres tests."""
        self.assertEqual(self._ecarts_a("2026-08-26 08:00", "2026-08-26", "12:00"), [])


class ImportAuRegistreTests(TestCase):
    """L'import d'un document est une entree du registre, pas un secret d'AGIR.

    Deux tours reels du 2026-09-01 (compte vitrine, horaire en image): les
    blocs etaient crees par le processeur de documents, AGIR le savait par le
    message enrichi, mais DIRE recevait « REGISTRE DU TOUR: VIDE » et repondait
    « Je n'ai pas encore recu ton horaire de session. Envoie-moi ton horaire ».
    """

    def setUp(self):
        self.user = User.objects.create_user(username='imp', password='x')
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def _tour(self, attachment=None, texte="voici mon horaire, gere ca"):
        vus = {}

        def _agir(self_agent, user, message, registre):
            vus['registre'] = registre
            return None

        with patch.object(self.Agent, '_agir', _agir), \
             patch.object(self.Agent, '_dire', return_value=ReponseDire(ouverture="Ok.")):
            resultat = self.Agent().process_message(self.user, texte, attachment)
        return resultat, vus['registre']

    def _document(self, nom='horaire.png', cours=None):
        return UploadedDocument.objects.create(
            user=self.user, file_name=nom, document_type='course_schedule',
            processed=True, extracted_data={'courses': cours or []})

    def test_le_document_du_tour_est_une_mutation_et_son_recap_est_affiche(self):
        from core.models import RecurringBlock
        doc = self._document(cours=[{'name': 'Physique'}, {'name': 'Anglais'}])
        RecurringBlock.objects.create(
            user=self.user, title='Physique', block_type='course', day_of_week=2,
            start_time='13:30', end_time='15:20', source_document=doc)
        RecurringBlock.objects.create(
            user=self.user, title='Anglais', block_type='course', day_of_week=4,
            start_time='09:00', end_time='11:00', source_document=doc)

        resultat, registre = self._tour(doc)

        imports = [a for a in registre.actions if a.outil == 'import_document']
        self.assertEqual(len(imports), 1)
        self.assertTrue(imports[0].est_mutation)
        self.assertTrue(imports[0].succes)
        # Le recap est rendu par du code, donc present meme quand DIRE ne dit
        # que « Ok. »: l'utilisateur lit ses cours, pas une prose.
        reponse = resultat['response']
        self.assertIn('2 entrées ajoutées', reponse)
        self.assertIn('Physique : mercredi 13:30-15:20', reponse)
        self.assertIn('Anglais : vendredi 09:00-11:00', reponse)
        self.assertIn('horaire.png', reponse)

    def test_le_recap_nomme_les_cours_ecartes_par_le_processeur(self):
        """5 cours lus, 4 crees: le manquant doit etre nomme, avec sa raison."""
        from core.models import RecurringBlock
        doc = self._document(cours=[
            {'name': 'Physique', 'day': 'mercredi', 'start_time': '13:30', 'end_time': '15:20'},
            {'name': 'Programmation Web', 'day': 'mardi', 'start_time': '15:00', 'end_time': '17:00'},
        ])
        RecurringBlock.objects.create(
            user=self.user, title='Physique', block_type='course', day_of_week=2,
            start_time='13:30', end_time='15:20', source_document=doc)

        resultat, _ = self._tour(doc)

        self.assertIn('1 entrée ajoutée', resultat['response'])
        self.assertIn('Non ajouté : Programmation Web (mardi 15:00-17:00)', resultat['response'])

    def test_le_brief_de_dire_ne_dit_plus_que_le_registre_est_vide(self):
        from core.models import RecurringBlock
        doc = self._document(cours=[{'name': 'Physique'}])
        RecurringBlock.objects.create(
            user=self.user, title='Physique', block_type='course', day_of_week=2,
            start_time='13:30', end_time='15:20', source_document=doc)
        _, registre = self._tour(doc)
        brief = self.Agent._brief_dire("voici mon horaire", registre, {}, "")
        self.assertNotIn('VIDE', brief)
        self.assertIn('import_document', brief)

    def test_un_import_recent_sans_piece_jointe_est_une_lecture(self):
        """« c'est bon ? » au tour suivant: DIRE connait les blocs, mais le
        bloc factuel ne rejoue pas l'import."""
        from core.models import RecurringBlock
        doc = self._document(nom='recent.pdf', cours=[{'name': 'Anglais'}])
        RecurringBlock.objects.create(
            user=self.user, title='Anglais', block_type='course', day_of_week=3,
            start_time='14:00', end_time='16:00', source_document=doc)

        resultat, registre = self._tour(None, texte="c'est bon ?")

        lectures = [a for a in registre.actions if a.outil == 'import_recent']
        self.assertEqual(len(lectures), 1)
        self.assertFalse(lectures[0].est_mutation)
        self.assertNotIn('Horaire importé', resultat['response'])
        brief = self.Agent._brief_dire("c'est bon ?", registre, {}, "")
        self.assertNotIn('VIDE', brief)
        self.assertIn('Anglais', brief)

    def test_un_document_sans_entree_exploitable_est_dit_sans_annoncer_un_import(self):
        doc = self._document(nom='vide.pdf', cours=[])
        resultat, registre = self._tour(doc)
        self.assertEqual([a.outil for a in registre.actions], ['import_recent'])
        self.assertIn('aucun cours', registre.actions[0].message)
        self.assertNotIn('Horaire importé', resultat['response'])

    def test_sans_import_le_registre_reste_vide(self):
        _, registre = self._tour(None, texte="bonjour")
        self.assertTrue(registre.vide())

    def test_un_import_vieux_de_plus_de_vingt_minutes_ne_compte_plus(self):
        from datetime import timedelta
        from core.models import RecurringBlock
        doc = self._document(nom='vieux.pdf', cours=[{'name': 'Anglais'}])
        UploadedDocument.objects.filter(pk=doc.pk).update(
            uploaded_at=timezone.now() - timedelta(minutes=45))
        RecurringBlock.objects.create(
            user=self.user, title='Anglais', block_type='course', day_of_week=3,
            start_time='14:00', end_time='16:00', source_document=doc)
        _, registre = self._tour(None, texte="c'est bon ?")
        self.assertTrue(registre.vide())
