"""
Fin de serie et depart repousse sur update_block (etape 1 du plan LIRE).

Enquete du 2026-09-14: « arrete mon quart a partir de decembre » terminait la
serie sans question, parce que la garde cherchait un verbe de suppression dans
le message. La garde se decide maintenant sur les dates demandees et l'etat du
bloc, jamais sur les mots. Decision de Darius: une fin lointaine ecrite
librement (« mon quart finit le 15 octobre ») demande aussi un tap. Seule
exception: la reponse immediate a la question de fin de recurrence posee apres
le dernier import, pour une fin au-dela de 7 jours sur un bloc de ce document
encore sans fin.

Relecture Codex du meme jour: la cle de la demande porte les dates (une puce
pour le 15 octobre n'autorise pas le 1er decembre), la cible relue compare les
bornes de la serie, et une question sur deux bornes nomme les deux.

Horloge du harnais: lundi 14 septembre 2026, 8 h (jour 0 = lundi).
"""
from datetime import date

from django.test import TransactionTestCase

from core.models import ConversationMessage, RecurringBlock, UploadedDocument
from core.test_agent_v2_gardes import AUJOURDHUI, HarnaisGardes, puces
from services.agent_v2 import outils as outils_v2
from services.agent_v2.redaction import question_code
from services.agent_v2.registre import Registre

OUI = 'Oui, je confirme.'


class FinDeSerieSansLireLeMessageTests(HarnaisGardes, TransactionTestCase):

    def assertRetenue(self, action, **dates):
        self.assertFalse(action.succes)
        demande = action.donnees['demande']
        self.assertEqual(demande['motif'], 'destructif')
        self.assertEqual(demande['outil'], 'update_block')
        self.assertEqual(demande['cle'], outils_v2._cle_destructive(
            'update_block', {'block_id': self.q.id, **dates}))
        return demande

    def taper(self, demande, tache='u:2'):
        self.attendre([puces(demande)], OUI)
        registre = Registre()
        return outils_v2.appliquer_choix_en_attente(self.user, registre, OUI, tache)

    def test_une_fin_ecrite_librement_demande_confirmation_quels_que_soient_les_mots(self):
        cas = (('mon quart finit le 15 octobre', '2026-10-15'),
               ('arrete mon quart a partir de decembre', '2026-12-01'),
               ('mon quart finit le 15 octobr', '2026-10-15'),
               ('ok', '2027-01-10'))
        for i, (brut, fin) in enumerate(cas):
            with self.subTest(brut=brut):
                self.assertRetenue(self.premier_tour(brut, 'update_block', tache=f'f:{i}',
                                                     block_id=self.q.id, end_date=fin),
                                   end_date=fin)
                self.q.refresh_from_db()
                self.assertIsNone(self.q.end_date)

    def test_le_tap_exact_execute_la_fin_demandee(self):
        demande = self.assertRetenue(self.premier_tour(
            'mon quart finit le 15 octobre', 'update_block',
            block_id=self.q.id, end_date='2026-10-15'), end_date='2026-10-15')
        self.taper(demande)
        self.q.refresh_from_db()
        self.assertEqual(self.q.end_date, date(2026, 10, 15))

    def test_la_puce_d_une_date_n_autorise_pas_une_autre_date(self):
        demande = self.assertRetenue(self.premier_tour(
            'mon quart finit le 15 octobre', 'update_block',
            block_id=self.q.id, end_date='2026-10-15'), end_date='2026-10-15')
        self.attendre([puces(demande)], OUI)
        registre, tools = self.outils(OUI, tache='u:2')
        self.appeler(tools, 'update_block', block_id=self.q.id, end_date='2026-12-01')
        self.assertFalse(registre.actions[-1].succes)
        self.q.refresh_from_db()
        self.assertIsNone(self.q.end_date)

    def test_une_borne_changee_entre_la_question_et_le_tap_n_execute_rien(self):
        demande = self.assertRetenue(self.premier_tour(
            'mon quart finit le 15 octobre', 'update_block',
            block_id=self.q.id, end_date='2026-10-15'), end_date='2026-10-15')
        RecurringBlock.objects.filter(id=self.q.id).update(end_date=date(2026, 9, 30))
        self.taper(demande)
        self.q.refresh_from_db()
        self.assertEqual(self.q.end_date, date(2026, 9, 30))

    def test_raccourcir_une_fin_existante_demande_confirmation(self):
        self.q.end_date = date(2026, 12, 1)
        self.q.save()
        self.assertRetenue(self.premier_tour('mon quart finit plus tot', 'update_block',
                                             block_id=self.q.id, end_date='2026-11-01'),
                           end_date='2026-11-01')
        self.q.refresh_from_db()
        self.assertEqual(self.q.end_date, date(2026, 12, 1))

    def test_prolonger_une_serie_ne_demande_rien(self):
        self.q.end_date = date(2026, 10, 1)
        self.q.save()
        action = self.premier_tour('je garde mon quart jusqu en decembre', 'update_block',
                                   block_id=self.q.id, end_date='2026-12-01')
        self.assertTrue(action.succes)
        self.q.refresh_from_db()
        self.assertEqual(self.q.end_date, date(2026, 12, 1))

    def test_un_depart_repousse_demande_confirmation(self):
        demande = self.assertRetenue(self.premier_tour(
            'la session commence le 24 octobre', 'update_block',
            block_id=self.q.id, start_date='2026-10-24'), start_date='2026-10-24')
        self.assertEqual(demande['cible']['date'], '2026-10-24')
        self.q.refresh_from_db()
        self.assertIsNone(self.q.start_date)

    def test_avancer_le_depart_ou_le_poser_aujourd_hui_ne_demande_rien(self):
        action = self.premier_tour('mon quart commence aujourd hui', 'update_block',
                                   tache='d:1', block_id=self.q.id, start_date='2026-09-14')
        self.assertTrue(action.succes)
        self.q.start_date = date(2026, 10, 1)
        self.q.save()
        action = self.premier_tour('mon quart commence plus tot', 'update_block',
                                   tache='d:2', block_id=self.q.id, start_date='2026-09-20')
        self.assertTrue(action.succes)
        self.q.refresh_from_db()
        self.assertEqual(self.q.start_date, date(2026, 9, 20))

    def test_la_question_nomme_la_ou_les_dates(self):
        fin = self.assertRetenue(self.premier_tour(
            'mon quart finit le 15 octobre', 'update_block', tache='q:1',
            block_id=self.q.id, end_date='2026-10-15'), end_date='2026-10-15')
        texte, _, _ = question_code([fin], AUJOURDHUI)
        self.assertIn('terminer Quart au dépanneur le ', texte)
        self.assertIn('15 oct', texte)

        depart = self.assertRetenue(self.premier_tour(
            'la session commence demain', 'update_block', tache='q:2',
            block_id=self.q.id, start_date='2026-09-15'), start_date='2026-09-15')
        texte, _, _ = question_code([depart], AUJOURDHUI)
        self.assertIn('reporter le début de Quart au dépanneur à demain', texte)

        deux = self.assertRetenue(self.premier_tour(
            'de fin octobre a mi-decembre', 'update_block', tache='q:3',
            block_id=self.q.id, start_date='2026-10-24', end_date='2026-12-15'),
            start_date='2026-10-24', end_date='2026-12-15')
        texte, _, _ = question_code([deux], AUJOURDHUI)
        self.assertIn('changer les dates de Quart au dépanneur (début le ', texte)
        self.assertIn('24 oct', texte)
        self.assertIn('fin le ', texte)
        self.assertNotIn('\u2014', texte)


class ReponseALaFinDeRecurrenceTests(HarnaisGardes, TransactionTestCase):

    def setUp(self):
        super().setUp()
        self.doc = self.document()
        self.importe = self.bloc('Chimie générale', 1, '13:00', '15:50', source_document=self.doc)

    def document(self):
        return UploadedDocument.objects.create(
            user=self.user, file='documents/horaire.pdf', file_name='horaire.pdf',
            document_type='course_schedule', processed=True)

    def question_de_fin_posee(self, motif='fin_recurrence'):
        ConversationMessage.objects.create(
            user=self.user, role='assistant', content='Jusqu’à quand ?',
            metadata={'question_motif': motif})

    def premier_tour(self, brut, nom, tache=None, **kwargs):
        """Comme le harnais, avec la tache de production « user:message »."""
        courant = self.message_courant(brut)
        registre, tools = self.outils(brut, tache=tache or f'{self.user.pk}:{courant.pk}')
        self.appeler(tools, nom, **kwargs)
        return registre.actions[-1]

    def test_un_tour_plus_ancien_ne_profite_pas_de_la_reponse_d_un_autre(self):
        ancien = self.message_courant('mets fin a ma chimie')
        self.question_de_fin_posee()
        self.message_courant('jusqu au 18 decembre')
        registre, tools = self.outils('mets fin a ma chimie', tache=f'{self.user.pk}:{ancien.pk}')
        self.appeler(tools, 'update_block', block_id=self.importe.id, end_date='2026-12-18')
        self.assertRetenue(registre.actions[-1], self.importe)

    def test_sans_identifiant_du_message_courant_demande(self):
        self.question_de_fin_posee()
        self.assertRetenue(self.premier_tour('jusqu au 18 decembre', 'update_block', tache='u:1',
                                             block_id=self.importe.id, end_date='2026-12-18'),
                           self.importe)

    def test_une_fin_lointaine_pour_un_bloc_du_dernier_import_passe(self):
        self.question_de_fin_posee()
        action = self.premier_tour('jusqu au 18 decembre', 'update_block',
                                   block_id=self.importe.id, end_date='2026-12-18')
        self.assertTrue(action.succes)
        self.importe.refresh_from_db()
        self.assertEqual(self.importe.end_date, date(2026, 12, 18))

    def assertRetenue(self, action, bloc):
        self.assertFalse(action.succes)
        bloc.refresh_from_db()
        self.assertIsNone(bloc.end_date)

    def test_une_fin_proche_demande_quand_meme(self):
        self.question_de_fin_posee()
        self.assertRetenue(self.premier_tour('jusqu a vendredi', 'update_block',
                                             block_id=self.importe.id, end_date='2026-09-18'),
                           self.importe)

    def test_un_bloc_non_importe_demande_quand_meme(self):
        self.question_de_fin_posee()
        self.assertRetenue(self.premier_tour('jusqu au 18 decembre', 'update_block',
                                             block_id=self.q.id, end_date='2026-12-18'),
                           self.q)

    def test_un_bloc_d_un_import_plus_ancien_demande_quand_meme(self):
        self.document()
        self.question_de_fin_posee()
        self.assertRetenue(self.premier_tour('jusqu au 18 decembre', 'update_block',
                                             block_id=self.importe.id, end_date='2026-12-18'),
                           self.importe)

    def test_sans_la_question_juste_avant_demande(self):
        self.question_de_fin_posee(motif='dire')
        self.assertRetenue(self.premier_tour('jusqu au 18 decembre', 'update_block',
                                             block_id=self.importe.id, end_date='2026-12-18'),
                           self.importe)

    def test_un_autre_message_entre_la_question_et_la_reponse_demande(self):
        self.question_de_fin_posee()
        self.message_courant('autre chose')
        self.assertRetenue(self.premier_tour('jusqu au 18 decembre', 'update_block',
                                             block_id=self.importe.id, end_date='2026-12-18'),
                           self.importe)
