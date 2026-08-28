"""
Accepter les NOMS de jours, pas seulement les numeros.

DEFAUT REPRODUIT LE 2026-08-28. L'audit de production du 19 aout decrivait un
decalage de +1 jour a l'import: des matchs annonces samedi stockes dimanche,
annonces dimanche stockes lundi. L'utilisateur avait fini par vider ses 28
blocs. Le defaut est toujours vivant, rejoue le 28 aout sur l'agent v2:

    Chicoutimi demande SAMEDI   -> stocke DIMANCHE
    Thetford   demande DIMANCHE -> stocke LUNDI

L'outil n'est PAS en cause: un appel direct avec days=[5] stocke bien samedi.
Le schema n'est pas ambigu non plus, il dit « 0=Lundi, 1=Mardi, ..., 6=Dimanche ».
C'est le MODELE qui se trompe, et seulement sur les jours de fin de semaine,
ou il retombe sur la convention americaine dimanche=0.

Une convention bien documentee ne suffit donc pas. On enleve l'occasion de se
tromper: le modele peut ecrire « samedi », et le code convertit. Les numeros
restent acceptes, v1 sert la production avec.
"""
from django.contrib.auth.models import User
from django.test import TestCase

from core.models import RecurringBlock
from services.agent.tools import execute_tool
from services.agent.tools.blocks import normaliser_jours

JOURS = ('lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche')


class NormalisationTests(TestCase):
    def test_les_numeros_restent_acceptes(self):
        """Retrocompatibilite: v1 sert la production avec des numeros."""
        self.assertEqual(normaliser_jours([0, 3, 6]), [0, 3, 6])

    def test_les_noms_sont_convertis(self):
        for i, nom in enumerate(JOURS):
            with self.subTest(nom=nom):
                self.assertEqual(normaliser_jours([nom]), [i])

    def test_les_accents_et_la_casse_ne_genent_pas(self):
        self.assertEqual(normaliser_jours(['SAMEDI', 'Dimanche', 'MeRcReDi']), [5, 6, 2])

    def test_les_numeros_en_texte_passent(self):
        """Un modele qui rend « 5 » plutot que 5 ne doit pas casser."""
        self.assertEqual(normaliser_jours(['5', '6']), [5, 6])

    def test_on_peut_melanger_noms_et_numeros(self):
        self.assertEqual(normaliser_jours([0, 'samedi', '6']), [0, 5, 6])

    def test_les_doublons_disparaissent_et_l_ordre_tient(self):
        self.assertEqual(normaliser_jours(['samedi', 5, 'Samedi']), [5])

    def test_un_jour_inconnu_est_ignore_et_non_devine(self):
        """Deviner rendrait le defaut d'origine possible par un autre chemin."""
        self.assertEqual(normaliser_jours(['samdi', 'lundi', 42, None, '']), [0])

    def test_une_valeur_seule_est_acceptee_comme_une_liste(self):
        self.assertEqual(normaliser_jours('samedi'), [5])
        self.assertEqual(normaliser_jours(3), [3])


class CreationParNomTests(TestCase):
    """Le cas exact de l'audit, bout en bout."""

    def setUp(self):
        self.user = User.objects.create_user(username='jours', password='x')

    def _creer(self, titre, jours):
        return execute_tool('create_block', self.user, {
            'title': titre, 'block_type': 'course', 'days': jours,
            'start_time': '13:00', 'end_time': '14:30'})

    def test_samedi_nomme_est_stocke_samedi(self):
        self._creer('Match Chicoutimi', ['samedi'])
        b = RecurringBlock.objects.get(user=self.user, title='Match Chicoutimi')
        self.assertEqual(b.day_of_week, 5, f"stocke {JOURS[b.day_of_week]}")

    def test_dimanche_nomme_est_stocke_dimanche(self):
        """Le jour ou le modele se trompait le plus: dimanche=0 en convention
        americaine, 6 chez nous."""
        self._creer('Match Thetford', ['dimanche'])
        b = RecurringBlock.objects.get(user=self.user, title='Match Thetford')
        self.assertEqual(b.day_of_week, 6, f"stocke {JOURS[b.day_of_week]}")

    def test_les_numeros_creent_toujours_au_bon_jour(self):
        """Contre-epreuve de non-regression sur le chemin existant."""
        self._creer('Cours', [5])
        b = RecurringBlock.objects.get(user=self.user, title='Cours')
        self.assertEqual(b.day_of_week, 5)

    def test_le_schema_annonce_les_deux_formes(self):
        """Si le schema ne le dit pas, le modele ne s'en servira jamais: la
        description EST l'interface."""
        from services.agent.tools import TOOL_MAP
        desc = TOOL_MAP['create_block'].parameters['properties']['days']['description']
        self.assertIn('samedi', desc.lower())
        self.assertIn('0=Lundi', desc)

    def test_le_schema_ACCEPTE_les_noms_dans_son_type(self):
        """La description ne suffit pas: un modele qui suit le type declare
        avant la prose enverrait encore des numeros, donc encore des erreurs
        de convention. Le type doit autoriser ce que la description conseille."""
        from services.agent.tools import TOOL_MAP
        items = TOOL_MAP['create_block'].parameters['properties']['days']['items']
        self.assertEqual(items.get('type'), 'string')


class RecapitulatifsDeV1Tests(TestCase):
    """v1 rend deux recapitulatifs DETERMINISTES a partir des arguments
    d'appel. Ils lisaient `days` en supposant des entiers.

    Sans normalisation, « samedi » partait tel quel dans un calcul de date
    pour le premier, et se faisait ecarter par un int() pour le second, ce qui
    faisait disparaitre le recap en silence. Un recap muet est pire qu'un
    plantage: personne ne le remarque."""

    def _appel(self, jours):
        return [{
            "tool": "create_block",
            "args": {"title": "Match", "block_type": "sport", "days": jours,
                     "start_time": "13:00", "end_time": "14:30"},
            "result": {"success": True, "data": {"created": [{"id": 1}]}},
        }]

    def test_le_recap_de_creation_nomme_les_jours_donnes_par_leur_nom(self):
        """Le recap ne se declenche qu'a partir de DEUX jours: c'est voulu,
        il decrit une habitude multi-jours et non un bloc isole."""
        from services.agent.agent import _creation_recap_footer
        rendu = _creation_recap_footer(self._appel(["samedi", "dimanche"]))
        self.assertTrue(rendu, "le recap a disparu pour des jours nommes")
        self.assertIn("sam", rendu)
        self.assertIn("dim", rendu)

    def test_le_recap_est_identique_en_noms_et_en_numeros(self):
        """Contre-epreuve de non-regression sur le chemin d'origine."""
        from services.agent.agent import _creation_recap_footer
        self.assertEqual(_creation_recap_footer(self._appel([5, 6])),
                         _creation_recap_footer(self._appel(["samedi", "dimanche"])))
