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
        r = Registre()
        r.ajouter('skip_block_occurrence', {'date': '2026-09-01'},
                  ToolResult(success=True, data={'date': '2026-09-08', 'title': 'Maths'}))
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
