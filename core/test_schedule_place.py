"""
POST /api/schedule/place/ : équivalent REST de l'outil agent schedule_task_at.
Vécu Darius 2026-08-07 : le MCP n'avait AUCUN moyen de poser une tâche à une
date — le placement daté n'existait que dans le chat.
"""
from datetime import time

from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient

from core.models import ScheduledBlock


class SchedulePlaceTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('placer', password='pw-place-1')
        token, _ = Token.objects.get_or_create(user=self.user)
        self.client = APIClient()
        # auth Token = exactement ce qu'envoie le MCP
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {token.key}')

    def test_place_creates_locked_dated_entry(self):
        resp = self.client.post('/api/schedule/place/', {
            'title': 'Rdv dentiste', 'date': '2026-09-03',
            'start_time': '14:00', 'end_time': '15:00',
        }, format='json')
        self.assertEqual(resp.status_code, 200, resp.content)
        sb = ScheduledBlock.objects.get(user=self.user)
        self.assertEqual(str(sb.date), '2026-09-03')
        self.assertEqual(sb.start_time, time(14, 0))
        self.assertTrue(sb.locked)
        self.assertEqual(sb.task.title, 'Rdv dentiste')

    def test_same_title_and_date_upserts(self):
        for end in ('15:00', '14:45'):
            self.client.post('/api/schedule/place/', {
                'title': 'Rdv dentiste', 'date': '2026-09-03',
                'start_time': '14:00', 'end_time': end,
            }, format='json')
        blocks = ScheduledBlock.objects.filter(user=self.user)
        self.assertEqual(blocks.count(), 1)
        self.assertEqual(blocks.first().end_time, time(14, 45))

    def test_invalid_date_is_400(self):
        resp = self.client.post('/api/schedule/place/', {
            'title': 'X', 'date': '03/09/2026',
            'start_time': '14:00', 'end_time': '15:00',
        }, format='json')
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(ScheduledBlock.objects.filter(user=self.user).exists())

    def test_conflict_names_the_actual_item(self):
        # Vécu 2026-08-09: le refus annonçait l'union fusionnée de la journée
        # (« 08:00-21:30 ») pour un trou de 15 min. Le message doit nommer
        # l'occupation réelle et sa vraie plage.
        self.client.post('/api/schedule/place/', {
            'title': 'Reunion equipe', 'date': '2026-09-03',
            'start_time': '11:00', 'end_time': '13:00',
        }, format='json')
        resp = self.client.post('/api/schedule/place/', {
            'title': 'Pause cafe', 'date': '2026-09-03',
            'start_time': '12:15', 'end_time': '12:30',
        }, format='json')
        self.assertEqual(resp.status_code, 400)
        msg = resp.json()['message']
        self.assertIn('Reunion equipe', msg)
        self.assertIn('11:00-13:00', msg)
