"""
Bornes de validité des blocs récurrents (start_date/end_date).

Demande Darius 2026-08-05: « Entraînements Gaillards (24 août début) » importé
sans borne s'affichait dès le 10 août et pour toujours; l'agent ne pouvait pas
enregistrer une fin de session. Ici: fenêtre respectée par la matérialisation,
l'import, les outils agent et l'iCal.
"""
from datetime import date, time

from django.contrib.auth.models import User
from django.test import TestCase

from core.models import RecurringBlock, UploadedDocument
from services.document_processor import DocumentProcessor
from services.scheduling.day_view import effective_day_blocks
from services.ical import build_calendar
from services.agent.tools import create_block, find_free_slots, get_week_schedule, schedule_task_at, update_block, harnais


def _mk(user, **kw):
    base = dict(user=user, title='Entraînements', block_type='sport',
                day_of_week=0, start_time=time(18, 0), end_time=time(19, 30))
    base.update(kw)
    return RecurringBlock.objects.create(**base)


class BoundsModelTest(TestCase):
    def setUp(self):
        self.u = User.objects.create_user('bounds', password='pw-bounds-1')

    def test_active_on_window(self):
        b = _mk(self.u, start_date=date(2026, 8, 24), end_date=date(2026, 10, 15))
        self.assertFalse(b.active_on(date(2026, 8, 17)))
        self.assertTrue(b.active_on(date(2026, 8, 24)))
        self.assertTrue(b.active_on(date(2026, 10, 15)))
        self.assertFalse(b.active_on(date(2026, 10, 19)))

    def test_day_view_respects_bounds(self):
        _mk(self.u, start_date=date(2026, 8, 24))
        # lundi 17 août = avant le début -> absent; lundi 24 août -> présent
        self.assertEqual(effective_day_blocks(self.u, date(2026, 8, 17)), [])
        titles = [e['title'] for e in effective_day_blocks(self.u, date(2026, 8, 24))]
        self.assertIn('Entraînements', titles)

    def test_import_captures_bounds_on_courses(self):
        doc = UploadedDocument.objects.create(
            user=self.u, file_name='h.png', document_type='other',
            extracted_data={}, processed=False)
        proc = DocumentProcessor.__new__(DocumentProcessor)
        DocumentProcessor._create_recurring_blocks(proc, doc, {'courses': [{
            'name': 'Entraînements - Gaillards', 'day': 'lundi',
            'start_time': '18:00', 'end_time': '19:30',
            'start_date': '2026-08-24',
        }]})
        b = RecurringBlock.objects.get(user=self.u)
        self.assertEqual(b.start_date, date(2026, 8, 24))
        self.assertIsNone(b.end_date)

    def test_agent_tools_set_bounds(self):
        res = harnais.appeler(create_block, self.u, dict(title='Natation', block_type='sport', days=[2],
            start_time='07:00', end_time='08:00', start_date='2026-09-01'))
        self.assertTrue(res.success, res.message)
        b = RecurringBlock.objects.get(user=self.u, title='Natation')
        self.assertEqual(b.start_date, date(2026, 9, 1))
        res = harnais.appeler(update_block, self.u, dict(block_id=b.id, end_date='2026-10-15'))
        self.assertTrue(res.success, res.message)
        b.refresh_from_db()
        self.assertEqual(b.end_date, date(2026, 10, 15))
        # retrait de la borne par chaine vide
        res = harnais.appeler(update_block, self.u, dict(block_id=b.id, end_date=''))
        self.assertTrue(res.success, res.message)
        b.refresh_from_db()
        self.assertIsNone(b.end_date)

    def test_update_rejects_inverted_bounds(self):
        b = _mk(self.u, start_date=date(2026, 9, 1))
        res = harnais.appeler(update_block, self.u, dict(block_id=b.id, end_date='2026-08-01'))
        self.assertFalse(res.success)
        # 2026-10-14: le message doit donner les valeurs et l'action,
        # sinon le modele ne peut pas se corriger (fin mai 2026 prise
        # pour une fin dans le passe au lieu de mai 2027).
        self.assertIn('2026-08-01', res.message)
        self.assertIn('2026-09-01', res.message)
        self.assertIn('année', res.message)

    def test_create_rejects_inverted_bounds_with_values(self):
        res = harnais.appeler(create_block, self.u, dict(title='Natation', block_type='sport', days=[2],
            start_time='07:00', end_time='08:00',
            start_date='2026-10-14', end_date='2026-05-31'))
        self.assertFalse(res.success)
        self.assertIn('2026-05-31', res.message)
        self.assertIn('2026-10-14', res.message)
        self.assertIn('année', res.message)


class ErreursActionnablesTest(TestCase):
    """Audit 2026-10-14 (option 2) : tout message d'echec d'outil donne
    la valeur recue + l'action corrective, sinon le modele ne peut pas
    se corriger et abandonne apres une tentative."""

    def setUp(self):
        self.u = User.objects.create_user('erreurs', password='<redacted>')

    def test_semaine_date_invalide_donne_la_valeur(self):
        res = harnais.appeler(get_week_schedule, self.u, dict(start_date='31 mai'))
        self.assertFalse(res.success)
        self.assertIn('31 mai', res.message)
        self.assertIn('YYYY-MM-DD', res.message)

    def test_creneaux_date_invalide_donne_la_valeur(self):
        res = harnais.appeler(find_free_slots, self.u, dict(date='demain'))
        self.assertFalse(res.success)
        self.assertIn('demain', res.message)
        self.assertIn('YYYY-MM-DD', res.message)

    def test_planification_heure_invalide_donne_les_valeurs(self):
        res = harnais.appeler(schedule_task_at, self.u, dict(title='X', date='2026-10-15',
            start_time='25:00', end_time='26:00'))
        self.assertFalse(res.success)
        self.assertIn('25:00', res.message)
        self.assertIn('26:00', res.message)

    def test_update_jour_invalide_donne_la_valeur(self):
        b = _mk(self.u, start_date=date(2026, 9, 1))
        res = harnais.appeler(update_block, self.u, dict(block_id=b.id, day_of_week='nimporte'))
        self.assertFalse(res.success)
        self.assertIn('nimporte', res.message)

    def test_ical_until_and_ended_blocks(self):
        _mk(self.u, title='Fini', end_date=date(2020, 1, 6))
        _mk(self.u, title='Session', day_of_week=1, end_date=date(2099, 10, 15))
        ics = build_calendar(self.u)
        self.assertNotIn('Fini', ics)
        self.assertIn('UNTIL=20991015T235959', ics)
