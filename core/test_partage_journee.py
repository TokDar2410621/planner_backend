"""Le partage « journée »: un lien VIVANT qui montre la journée du moment.

Demande de Darius du 2026-09-03. Le lien de portée `today` ne fige pas la
date du partage: ouvert demain, il montre demain. La vue publique filtre donc
au moment de la lecture: jour de semaine courant (heure murale
America/Toronto), fenêtres start/end_date respectées, occurrences sautées
exclues, tâches du jour seulement.
"""
from datetime import time, timedelta

from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from core.models import (
    RecurringBlock,
    RecurringBlockException,
    ScheduledBlock,
    SharedSchedule,
    Task,
)


def _bloc(user, titre, dow, debut=time(9, 0), fin=time(11, 0), **extra):
    return RecurringBlock.objects.create(
        user=user, title=titre, block_type='course', day_of_week=dow,
        start_time=debut, end_time=fin, **extra)


class CreationDuPartageTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='ana', password='x')
        self.api = APIClient()
        self.api.force_authenticate(self.user)

    def test_par_defaut_le_lien_couvre_la_semaine(self):
        r = self.api.post('/api/schedule/share/', {}, format='json')
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.data['scope'], 'week')
        self.assertEqual(r.data['title'], 'Mon planning')

    def test_un_lien_journee_se_cree_et_se_nomme(self):
        r = self.api.post('/api/schedule/share/', {'scope': 'today'}, format='json')
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.data['scope'], 'today')
        self.assertEqual(r.data['title'], 'Ma journée')

    def test_un_titre_fourni_garde_la_main(self):
        r = self.api.post('/api/schedule/share/',
                          {'scope': 'today', 'title': 'Ma journée de mercredi'},
                          format='json')
        self.assertEqual(r.data['title'], 'Ma journée de mercredi')

    def test_une_portee_inconnue_est_refusee(self):
        r = self.api.post('/api/schedule/share/', {'scope': 'month'}, format='json')
        self.assertEqual(r.status_code, 400)

    def test_la_liste_expose_la_portee(self):
        self.api.post('/api/schedule/share/', {'scope': 'today'}, format='json')
        r = self.api.get('/api/schedule/share/')
        self.assertEqual(r.data[0]['scope'], 'today')


class PageJourneeTests(TestCase):
    """La vue publique d'un lien « journée », lue avec un client anonyme."""

    def setUp(self):
        self.user = User.objects.create_user(
            username='ana', password='x', first_name='Ana')
        self.aujourd_hui = timezone.localdate()
        self.dow = self.aujourd_hui.weekday()
        self.autre_dow = (self.dow + 3) % 7
        self.partage = SharedSchedule.objects.create(
            user=self.user, title='Ma journée', scope=SharedSchedule.SCOPE_TODAY)
        self.anonyme = APIClient()

    def _page(self):
        r = self.anonyme.get(f'/api/shared/{self.partage.share_token}/')
        self.assertEqual(r.status_code, 200)
        return r.data

    def test_seuls_les_blocs_du_jour_sont_rendus(self):
        _bloc(self.user, 'Chimie', self.dow)
        _bloc(self.user, 'Philo', self.autre_dow)
        page = self._page()
        self.assertEqual(page['scope'], 'today')
        self.assertEqual(page['date'], self.aujourd_hui.isoformat())
        self.assertEqual([b['title'] for b in page['recurring_blocks']], ['Chimie'])

    def test_une_occurrence_sautee_disparait_de_la_journee(self):
        garde = _bloc(self.user, 'Chimie', self.dow)
        saute = _bloc(self.user, 'Labo', self.dow, debut=time(13, 0), fin=time(15, 0))
        RecurringBlockException.objects.create(
            user=self.user, recurring_block=saute, date=self.aujourd_hui)
        titres = [b['title'] for b in self._page()['recurring_blocks']]
        self.assertEqual(titres, [garde.title])

    def test_un_bloc_hors_fenetre_de_dates_disparait(self):
        _bloc(self.user, 'Session finie', self.dow,
              end_date=self.aujourd_hui - timedelta(days=7))
        _bloc(self.user, 'Session a venir', self.dow,
              start_date=self.aujourd_hui + timedelta(days=7))
        _bloc(self.user, 'Session courante', self.dow,
              start_date=self.aujourd_hui - timedelta(days=30),
              end_date=self.aujourd_hui + timedelta(days=30))
        titres = [b['title'] for b in self._page()['recurring_blocks']]
        self.assertEqual(titres, ['Session courante'])

    def test_les_taches_incluses_se_limitent_au_jour(self):
        self.partage.include_tasks = True
        self.partage.save()
        tache = Task.objects.create(user=self.user, title='Rapport')
        ScheduledBlock.objects.create(
            user=self.user, task=tache, date=self.aujourd_hui,
            start_time=time(14, 0), end_time=time(15, 0))
        ScheduledBlock.objects.create(
            user=self.user, task=tache, date=self.aujourd_hui + timedelta(days=1),
            start_time=time(14, 0), end_time=time(15, 0))
        page = self._page()
        self.assertEqual(len(page['scheduled_tasks']), 1)
        self.assertEqual(page['scheduled_tasks'][0]['date'], self.aujourd_hui.isoformat())

    def test_le_lien_semaine_garde_tout_et_se_declare(self):
        self.partage.scope = SharedSchedule.SCOPE_WEEK
        self.partage.save()
        _bloc(self.user, 'Chimie', self.dow)
        _bloc(self.user, 'Philo', self.autre_dow)
        page = self._page()
        self.assertEqual(page['scope'], 'week')
        self.assertNotIn('date', page)
        self.assertEqual(len(page['recurring_blocks']), 2)
