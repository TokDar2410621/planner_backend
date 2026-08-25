"""
La garantie structurelle, testee sans le moindre appel LLM.

Le modele DIRE ne redige pas les faits: il rend un JSON ou chaque action citee
porte la reference d'une entree de registre. Le runtime assemble et supprime
toute action dont la reference n'existe pas.
"""
from django.test import SimpleTestCase

from services.agent.tools.base import ToolResult
from services.agent_v2.redaction import (
    ActionCitee, ReponseDire, assembler, bloc_factuel,
)
from services.agent_v2.registre import Registre


def _deux_creations() -> Registre:
    r = Registre()
    r.ajouter('create_block', {'title': 'Maths'},
              ToolResult(success=True, message="Bloc 'Maths' cree (09:00-12:00) les Lundi"))
    r.ajouter('create_block', {'title': 'Sport'},
              ToolResult(success=False, message="Chevauchement avec 'Travail' (09:00-17:00)"))
    return r


class BlocFactuelTests(SimpleTestCase):
    def test_un_registre_vide_ne_produit_aucun_bloc(self):
        self.assertEqual(bloc_factuel(Registre()), '')

    def test_les_reussites_et_les_echecs_apparaissent(self):
        texte = bloc_factuel(_deux_creations())
        self.assertIn('Maths', texte)
        self.assertIn('Chevauchement', texte)

    def test_un_echec_n_est_jamais_noye_dans_un_total(self):
        r = Registre()
        for i in range(8):
            r.ajouter('create_block', {'title': f'Cours {i}'},
                      ToolResult(success=True, message=f"Bloc 'Cours {i}' cree"))
        r.ajouter('create_block', {'title': 'Rate'},
                  ToolResult(success=False, message="Chevauchement avec 'Sommeil'"))
        texte = bloc_factuel(r)
        self.assertIn('8', texte)
        self.assertIn('Sommeil', texte)

    def test_les_ecarts_apparaissent(self):
        r = _deux_creations()
        r.ajouter_ecart('a1', 'date demandee 2026-09-01, date obtenue 2026-10-01')
        self.assertIn('2026-10-01', bloc_factuel(r))

    def test_un_budget_epuise_est_dit(self):
        r = _deux_creations()
        r.budget_epuise = True
        self.assertIn('interrompu', bloc_factuel(r).lower())

    def test_une_lecture_seule_ne_produit_aucun_fait(self):
        """Une consultation n'est pas une action a raconter."""
        r = Registre()
        r.ajouter('list_blocks', {}, ToolResult(success=True, message='3 blocs trouves'))
        self.assertEqual(bloc_factuel(r), '')


class AssemblageTests(SimpleTestCase):
    def test_une_action_referencee_survit(self):
        r = _deux_creations()
        brut = ReponseDire(ouverture="Voila.",
                           actions=[ActionCitee(ref='a1', phrase="Maths est cale le lundi.")],
                           suite="")
        texte, rejetees = assembler(brut, r)
        self.assertEqual(rejetees, 0)
        self.assertIn('Maths est cale', texte)

    def test_une_action_inventee_est_supprimee_et_comptee(self):
        """LE test central du projet."""
        r = _deux_creations()
        brut = ReponseDire(
            ouverture="C'est note.",
            actions=[
                ActionCitee(ref='a1', phrase="Maths est cale le lundi."),
                ActionCitee(ref='a42', phrase="J'ai aussi reorganise toute ta semaine."),
            ],
            suite="")
        texte, rejetees = assembler(brut, r)
        self.assertEqual(rejetees, 1)
        self.assertIn('Maths est cale', texte)
        self.assertNotIn('reorganise toute ta semaine', texte)

    def test_un_registre_vide_ne_laisse_passer_aucune_action(self):
        """Le cas du 18 aout: tour sans le moindre outil."""
        brut = ReponseDire(
            ouverture="C'est note,",
            actions=[ActionCitee(ref='a1', phrase="j'ai supprime les blocs qui chevauchent.")],
            suite="")
        texte, rejetees = assembler(brut, Registre())
        self.assertEqual(rejetees, 1)
        self.assertNotIn('supprime', texte)

    def test_une_reference_vide_est_rejetee(self):
        r = _deux_creations()
        brut = ReponseDire(ouverture="", actions=[ActionCitee(ref='', phrase="J'ai tout fait.")], suite="")
        texte, rejetees = assembler(brut, r)
        self.assertEqual(rejetees, 1)
        self.assertNotIn('tout fait', texte)

    def test_le_bloc_factuel_est_dans_le_texte_final(self):
        r = _deux_creations()
        texte, _ = assembler(ReponseDire(ouverture="Salut."), r)
        self.assertIn('Maths', texte)
