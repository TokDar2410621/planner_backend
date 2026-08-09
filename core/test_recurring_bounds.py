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
from services.agent.tools.blocks import CreateBlockTool, UpdateBlockTool
from services.ical import build_calendar


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
        res = CreateBlockTool().execute(
            self.u, title='Natation', block_type='sport', days=[2],
            start_time='07:00', end_time='08:00', start_date='2026-09-01')
        self.assertTrue(res.success, res.message)
        b = RecurringBlock.objects.get(user=self.u, title='Natation')
        self.assertEqual(b.start_date, date(2026, 9, 1))
        res = UpdateBlockTool().execute(self.u, block_id=b.id, end_date='2026-10-15')
        self.assertTrue(res.success, res.message)
        b.refresh_from_db()
        self.assertEqual(b.end_date, date(2026, 10, 15))
        # retrait de la borne par chaine vide
        res = UpdateBlockTool().execute(self.u, block_id=b.id, end_date='')
        self.assertTrue(res.success, res.message)
        b.refresh_from_db()
        self.assertIsNone(b.end_date)

    def test_update_rejects_inverted_bounds(self):
        b = _mk(self.u, start_date=date(2026, 9, 1))
        res = UpdateBlockTool().execute(self.u, block_id=b.id, end_date='2026-08-01')
        self.assertFalse(res.success)

    def test_ical_until_and_ended_blocks(self):
        _mk(self.u, title='Fini', end_date=date(2020, 1, 6))
        _mk(self.u, title='Session', day_of_week=1, end_date=date(2099, 10, 15))
        ics = build_calendar(self.u)
        self.assertNotIn('Fini', ics)
        self.assertIn('UNTIL=20991015T235959', ics)
