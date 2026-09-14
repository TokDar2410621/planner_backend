"""
Reconciliation: le runtime relit ce qu'il vient d'ecrire et compare.

Les formes de ToolResult.data utilisees ici sont celles du code REEL, verifiees
outil par outil. Le plan precedent testait une forme inventee (`data['date']` a
la racine) que schedule_task_at ne produit jamais: le detecteur ne se serait
jamais declenche en production, avec un test vert par-dessus.
"""
from datetime import date, timedelta

from django.contrib.auth.models import User
from django.test import TestCase

from services.agent.tools.base import ToolResult
from services.agent_v2.reconciliation import LECTURES, detecter_ecarts, reconcilier
from services.agent_v2.registre import OUTILS_DE_MUTATION, Registre


class TableDeLectureTests(TestCase):
    def test_chaque_outil_de_mutation_a_une_lecture(self):
        manquants = sorted(OUTILS_DE_MUTATION - set(LECTURES))
        self.assertEqual(manquants, [], f'sans lecture: {manquants}')

    def test_chaque_lecture_est_un_outil_reel(self):
        from services.agent.tools import TOOL_MAP
        for mute, lecture in LECTURES.items():
            with self.subTest(outil=mute):
                self.assertIn(lecture, TOOL_MAP, f'{lecture} n existe pas')


class EcartDeDateTests(TestCase):
    def test_schedule_task_at_date_imbriquee(self):
        """La vraie forme: data['scheduled_block']['date']."""
        r = Registre()
        demande = (date.today() + timedelta(days=1)).isoformat()
        obtenue = (date.today() + timedelta(days=40)).isoformat()
        r.ajouter('schedule_task_at', {'date': demande, 'title': 'Revision'},
                  ToolResult(success=True, message='planifiee',
                             data={'scheduled_block': {'date': obtenue, 'title': 'Revision'}}))
        detecter_ecarts(r)
        self.assertEqual(len(r.ecarts), 1)
        self.assertIn(obtenue, r.ecarts[0].description)

    def test_aucun_ecart_quand_la_date_correspond(self):
        r = Registre()
        d = (date.today() + timedelta(days=1)).isoformat()
        r.ajouter('schedule_task_at', {'date': d},
                  ToolResult(success=True, data={'scheduled_block': {'date': d}}))
        detecter_ecarts(r)
        self.assertEqual(r.ecarts, [])

    def test_skip_block_occurrence_date_a_la_racine(self):
        # Dates relatives: les dates fixes 2026-09-01 / 2026-09-08 sont passees
        # depuis le 2026-09-08, et l'ecart « passe » s'ajoutait au compte.
        r = Registre()
        demande = (date.today() + timedelta(days=10)).isoformat()
        obtenue = (date.today() + timedelta(days=17)).isoformat()
        r.ajouter('skip_block_occurrence', {'date': demande},
                  ToolResult(success=True, data={'date': obtenue, 'title': 'Maths'}))
        detecter_ecarts(r)
        self.assertEqual(len(r.ecarts), 1)

    def test_un_echec_ne_produit_pas_d_ecart(self):
        """Un refus est deja dit par le bloc factuel: pas de doublon."""
        r = Registre()
        r.ajouter('schedule_task_at', {'date': '2026-09-01'},
                  ToolResult(success=False, message='chevauchement'))
        detecter_ecarts(r)
        self.assertEqual(r.ecarts, [])


class SuccesSansMutationTests(TestCase):
    """Quatre cas verifies ou success=True sans rien changer."""

    def test_create_task_dedoublonnee_est_signalee(self):
        r = Registre()
        r.ajouter('create_task', {'title': 'Reviser'},
                  ToolResult(success=True, message='Tache deja presente (non dupliquee)',
                             data={'task': {'id': 1, 'title': 'Reviser'}}))
        detecter_ecarts(r)
        self.assertEqual(len(r.ecarts), 1)
        self.assertIn('deja', r.ecarts[0].description.lower())

    def test_optimize_week_non_applique_est_signale(self):
        r = Registre()
        r.ajouter('optimize_week', {'apply': False},
                  ToolResult(success=True, message='plan propose',
                             data={'applied': False, 'moved_count': 0}))
        detecter_ecarts(r)
        self.assertEqual(len(r.ecarts), 1)
        self.assertIn('propos', r.ecarts[0].description.lower())


class GenresDEcartTests(TestCase):
    """Chaque cas de detecter_ecarts porte un genre et des donnees pour rendu.py,
    et sa description (pour le modele) reste celle d'avant."""

    def test_passe(self):
        r = Registre()
        hier = (date.today() - timedelta(days=1)).isoformat()
        r.ajouter('schedule_task_at', {'date': hier, 'title': 'Révision'},
                  ToolResult(success=True, data={'scheduled_block': {
                      'date': hier, 'title': 'Révision',
                      'start_time': '14:00', 'end_time': '15:00'}}))
        detecter_ecarts(r)
        self.assertEqual([e.genre for e in r.ecarts], ['passe'])
        e = r.ecarts[0]
        self.assertTrue(e.description.startswith('CREE mais dans le passe'))
        self.assertEqual(e.donnees, {'date': hier, 'debut': '14:00', 'fin': '15:00',
                                     'titre': 'Révision'})

    def test_date_differente(self):
        r = Registre()
        demande = (date.today() + timedelta(days=1)).isoformat()
        obtenue = (date.today() + timedelta(days=3)).isoformat()
        r.ajouter('schedule_task_at', {'date': demande, 'title': 'Lecture'},
                  ToolResult(success=True, data={'scheduled_block': {
                      'date': obtenue, 'title': 'Lecture', 'end_time': '23:00'}}))
        detecter_ecarts(r)
        self.assertEqual([e.genre for e in r.ecarts], ['date_differente'])
        self.assertEqual(r.ecarts[0].description,
                         f'date demandee {demande}, date obtenue {obtenue}')
        self.assertEqual(r.ecarts[0].donnees,
                         {'demandee': demande, 'obtenue': obtenue, 'titre': 'Lecture'})

    def test_tache_existante(self):
        r = Registre()
        r.ajouter('create_task', {'title': 'Reviser'},
                  ToolResult(success=True, message='Tache deja presente (non dupliquee)',
                             data={'task': {'id': 1, 'title': 'Reviser'}}))
        detecter_ecarts(r)
        self.assertEqual([e.genre for e in r.ecarts], ['tache_existante'])
        self.assertEqual(r.ecarts[0].description, "tache deja presente, rien n'a ete cree")
        self.assertEqual(r.ecarts[0].donnees, {'titre': 'Reviser'})

    def test_plan_propose(self):
        r = Registre()
        r.ajouter('organize_day', {'date': '2099-01-01'},
                  ToolResult(success=True, data={'applied': False, 'date': '2099-01-01'}))
        detecter_ecarts(r)
        self.assertEqual([e.genre for e in r.ecarts], ['plan_propose'])
        self.assertEqual(r.ecarts[0].description, "plan seulement propose, rien n'a ete applique")

    def test_preferences_inchangees(self):
        r = Registre()
        r.ajouter('update_preferences', {}, ToolResult(success=True, data={'updated_fields': []}))
        detecter_ecarts(r)
        self.assertEqual([e.genre for e in r.ecarts], ['preferences_inchangees'])
        self.assertEqual(r.ecarts[0].description, "aucune preference n'a change")

    def test_rien_a_restaurer(self):
        r = Registre()
        futur = (date.today() + timedelta(days=2)).isoformat()
        r.ajouter('restore_block_occurrence', {'date': futur},
                  ToolResult(success=True, data={'date': futur, 'title': 'Gym', 'restored': False}))
        detecter_ecarts(r)
        self.assertEqual([e.genre for e in r.ecarts], ['rien_a_restaurer'])
        self.assertEqual(r.ecarts[0].description, 'aucune occurrence sautee a restaurer')
        self.assertEqual(r.ecarts[0].donnees, {'date': futur, 'titre': 'Gym'})


class ReconciliationTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='recon', password='x')

    def test_un_tour_sans_mutation_ne_relit_rien(self):
        r = Registre()
        r.ajouter('list_blocks', {}, ToolResult(success=True))
        self.assertEqual(reconcilier(self.user, r), {})

    def test_un_tour_avec_mutation_relit_le_bon_outil(self):
        r = Registre()
        r.ajouter('create_block', {'title': 'Maths'}, ToolResult(success=True))
        etat = reconcilier(self.user, r)
        self.assertIn('list_blocks', etat)
