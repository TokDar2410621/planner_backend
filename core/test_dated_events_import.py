"""
Faille « horaire de matchs » (vécue par l'ami de Darius, 2026-08-05) : un
événement extrait AVEC une date calendaire doit devenir une entrée PONCTUELLE
(Task + ScheduledBlock verrouillé), jamais un bloc récurrent hebdomadaire
anonyme répété à vie.
"""
from datetime import time

from django.contrib.auth.models import User
from django.test import TestCase

from core.models import RecurringBlock, ScheduledBlock, Task, UploadedDocument
from services.document_processor import DocumentProcessor


class DatedEventsImportTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('matchuser', password='pw-match-1')
        self.doc = UploadedDocument.objects.create(
            user=self.user, file_name='matchs.png', document_type='other',
            extracted_data={}, processed=False,
        )
        self.proc = DocumentProcessor.__new__(DocumentProcessor)

    def _run(self, events):
        return DocumentProcessor._create_recurring_blocks(
            self.proc, self.doc, {'events': events}
        )

    def test_dated_event_becomes_scheduled_block_not_recurring(self):
        self._run([{
            'name': 'Match: Jonquière vs Saint-Félicien',
            'date': '2026-08-29', 'day': 'samedi',
            'start_time': '18:00', 'end_time': '19:30',
            'location': 'Cégep de St-Félicien',
        }])
        self.assertEqual(RecurringBlock.objects.filter(user=self.user).count(), 0)
        sb = ScheduledBlock.objects.get(user=self.user)
        self.assertEqual(str(sb.date), '2026-08-29')
        self.assertEqual(sb.start_time, time(18, 0))
        self.assertEqual(sb.end_time, time(19, 30))
        self.assertTrue(sb.locked)
        self.assertEqual(sb.task.title, 'Match: Jonquière vs Saint-Félicien')
        self.assertIn('Cégep de St-Félicien', sb.task.description)

    def test_three_saturday_matches_stay_three_dated_entries(self):
        # L'ancien code en faisait 3 blocs récurrents TOUS les samedis.
        self._run([
            {'name': 'M104', 'date': '2026-08-29', 'start_time': '18:00', 'end_time': '19:30'},
            {'name': 'M112', 'date': '2026-09-12', 'start_time': '15:00', 'end_time': '16:30'},
            {'name': 'M136', 'date': '2026-10-10', 'start_time': '13:00', 'end_time': '14:30'},
        ])
        self.assertEqual(RecurringBlock.objects.filter(user=self.user).count(), 0)
        dates = set(ScheduledBlock.objects.filter(user=self.user).values_list('date', flat=True))
        self.assertEqual({str(d) for d in dates}, {'2026-08-29', '2026-09-12', '2026-10-10'})

    def test_rerun_is_idempotent(self):
        ev = [{'name': 'M104', 'date': '2026-08-29', 'start_time': '18:00', 'end_time': '19:30'}]
        self._run(ev)
        self._run(ev)
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 1)

    def test_missing_end_time_defaults_to_two_hours(self):
        self._run([{'name': 'Tournoi', 'date': '2026-09-01', 'start_time': '10:00'}])
        sb = ScheduledBlock.objects.get(user=self.user)
        self.assertEqual(sb.end_time, time(12, 0))

    def test_undated_event_still_recurring_but_named(self):
        # Sans date: comportement hebdomadaire conservé, mais le NOM n'est plus
        # perdu (le code lisait 'title', Gemini renvoie 'name').
        self._run([{'name': 'Réunion équipe', 'day': 'mercredi',
                    'start_time': '18:00', 'end_time': '19:00'}])
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 0)
        rb = RecurringBlock.objects.get(user=self.user)
        self.assertEqual(rb.title, 'Réunion équipe')
        self.assertEqual(rb.day_of_week, 2)

    def test_dated_course_routed_to_dated_entry(self):
        # Vecu ami: la vision classe les matchs en 'courses' (course_code M104).
        # La date doit primer sur la recurrence.
        DocumentProcessor._create_recurring_blocks(self.proc, self.doc, {'courses': [{
            'name': 'Match: Jonquière vs Saint-Félicien', 'course_code': 'M104',
            'day': 'samedi', 'date': '2026-08-29',
            'start_time': '18:00', 'end_time': '19:30',
            'location': 'Cégep de St-Félicien',
        }]})
        self.assertEqual(RecurringBlock.objects.filter(user=self.user).count(), 0)
        sb = ScheduledBlock.objects.get(user=self.user)
        self.assertEqual(str(sb.date), '2026-08-29')
        self.assertTrue(sb.locked)

    def test_reupload_other_document_no_recurring_duplicates(self):
        # Vecu ami: 19 -> 28 -> 43 blocs en re-important (les blocs d'import
        # naissent 'flexible', invisibles pour find_recurring_conflicts).
        payload = {'courses': [{'name': 'Anglais', 'day': 'lundi',
                                'start_time': '08:00', 'end_time': '10:00'}]}
        DocumentProcessor._create_recurring_blocks(self.proc, self.doc, payload)
        doc2 = UploadedDocument.objects.create(
            user=self.user, file_name='matchs2.png', document_type='other',
            extracted_data={}, processed=False,
        )
        DocumentProcessor._create_recurring_blocks(self.proc, doc2, payload)
        self.assertEqual(RecurringBlock.objects.filter(user=self.user).count(), 1)

    def test_cache_rejects_prior_extraction_version(self):
        # Vecu ami: sa photo re-importee resservait l'extraction SANS dates
        # d'avant le fix (from_cache=True). Une version anterieure = cache MISS.
        h = 'a' * 64
        UploadedDocument.objects.create(
            user=self.user, file_name='old.png', document_type='other',
            content_hash=h, processed=True,
            extracted_data={'courses': [{'name': 'x'}]},
        )
        self.assertIsNone(DocumentProcessor._check_cache(self.proc, self.user, h))
        UploadedDocument.objects.create(
            user=self.user, file_name='new.png', document_type='other',
            content_hash='b' * 64, processed=True,
            extracted_data={'courses': [{'name': 'y'}],
                            'extraction_version': DocumentProcessor.EXTRACTION_VERSION},
        )
        self.assertIsNotNone(DocumentProcessor._check_cache(self.proc, self.user, 'b' * 64))

    def test_past_locked_event_not_in_unscheduled(self):
        # Un evenement date importe et deja passe (Gatineau 25 juillet vs
        # aujourd'hui) ne doit pas trainer dans « À planifier ».
        from rest_framework.test import APIClient
        from rest_framework.authtoken.models import Token
        self._run([{'name': 'Gatineau', 'date': '2020-01-04',
                    'start_time': '09:00', 'end_time': '11:00'}])
        client = APIClient()
        token, _ = Token.objects.get_or_create(user=self.user)
        client.credentials(HTTP_AUTHORIZATION=f'Token {token.key}')
        resp = client.get('/api/schedule/')
        titles = [t['title'] for t in resp.json()['unscheduled_tasks']]
        self.assertNotIn('Gatineau', titles)
