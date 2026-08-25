"""
Le registre est la seule source de verite d'un tour.

Ecrit par le RUNTIME et jamais par le modele, il porte un identifiant par
action. La phase DIRE ne peut citer qu'une action dont l'identifiant existe
ici.
"""
from django.test import SimpleTestCase

from services.agent.tools.base import ToolResult
from services.agent_v2.registre import OUTILS_DE_MUTATION, Registre


class RegistreTests(SimpleTestCase):
    def test_un_registre_neuf_est_vide(self):
        r = Registre()
        self.assertTrue(r.vide())
        self.assertEqual(r.actions, [])
        self.assertFalse(r.budget_epuise)

    def test_chaque_action_recoit_un_identifiant_unique(self):
        r = Registre()
        a1 = r.ajouter('create_block', {'title': 'Maths'}, ToolResult(success=True))
        a2 = r.ajouter('create_block', {'title': 'Physique'}, ToolResult(success=True))
        self.assertNotEqual(a1.id, a2.id)
        self.assertIs(r.par_id(a1.id), a1)
        self.assertFalse(r.vide())

    def test_un_echec_est_consigne_comme_les_reussites(self):
        r = Registre()
        a = r.ajouter('create_block', {'title': 'Sport'},
                      ToolResult(success=False, message="Chevauchement avec 'Travail'"))
        self.assertFalse(a.succes)
        self.assertIn('Chevauchement', a.message)

    def test_les_mutations_sont_distinguees_des_lectures(self):
        r = Registre()
        r.ajouter('list_blocks', {}, ToolResult(success=True))
        mut = r.ajouter('delete_block', {'block_id': 7}, ToolResult(success=True))
        self.assertEqual([a.id for a in r.mutations()], [mut.id])

    def test_les_trois_outils_oublies_par_v1_sont_des_mutations(self):
        """v1 ignore organize_day, optimize_week et cancel_scheduled_block
        dans MUTATION_TOOLS alors qu'ils ecrivent en base."""
        for nom in ('organize_day', 'optimize_week', 'cancel_scheduled_block'):
            self.assertIn(nom, OUTILS_DE_MUTATION)

    def test_la_liste_couvre_celle_de_v1(self):
        from services.agent.agent import MUTATION_TOOLS as v1
        self.assertTrue(v1 <= OUTILS_DE_MUTATION,
                        f'absents de v2: {sorted(v1 - OUTILS_DE_MUTATION)}')

    def test_un_ecart_reference_son_action(self):
        r = Registre()
        a = r.ajouter('schedule_task_at', {'date': '2026-09-01'}, ToolResult(success=True))
        e = r.ajouter_ecart(a.id, 'date obtenue 2026-10-01')
        self.assertEqual(e.action_id, a.id)
        self.assertIs(r.par_id(e.id), e)

    def test_un_identifiant_inconnu_ne_resout_rien(self):
        r = Registre()
        r.ajouter('create_block', {}, ToolResult(success=True))
        for mauvais in ('a99', '', None, 42):
            self.assertIsNone(r.par_id(mauvais))
