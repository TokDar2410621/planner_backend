"""
Le flux doit porter les APPELS D'OUTILS, pas seulement le raisonnement.

L'interface sait afficher une trace complete (raisonnement, puis un jalon par
outil avec son resultat depliable). Le backend n'envoyait que le raisonnement:
l'utilisateur voyait l'agent penser, puis un compte rendu surgir, sans rien
entre les deux.

Le registre contient deja tout ce qu'il faut. On le diffuse AU FIL de
l'execution, depuis l'adaptateur, seul point par lequel tous les appels
passent.

CONTRAINTE: ajout PUREMENT ADDITIF au contrat SSE. Un binaire iOS deja publie
ignore un type d'evenement inconnu, il ne doit pas s'en trouver plus mal.
"""
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase

from services.agent.tools.base import ToolResult
from services.agent_v2.redaction import ReponseDire
from services.agent_v2.registre import Registre


class TraceDesOutilsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='trace', password='x')
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    @staticmethod
    def _agir_avec_deux_outils(self_agent, user, message, registre):
        """Simule AGIR: deux outils, une lecture puis une ecriture."""
        for outil, params, res in (
            ('list_blocks', {'day_of_week': 'jeudi'},
             ToolResult(success=True, data={'count': 2}, message='2 blocs trouves')),
            ('create_block', {'title': 'Maths'},
             ToolResult(success=True, message="Bloc 'Maths' cree")),
        ):
            action = registre.ajouter(outil, params, res)
            self_agent.signaler_outil(action)
        return "Je regarde le planning."

    def _evenements(self, agir=None):
        with patch.object(self.Agent, '_agir', agir or self._agir_avec_deux_outils), \
             patch.object(self.Agent, '_dire',
                          return_value=ReponseDire(ouverture="C'est fait.")):
            return list(self.Agent().process_message_stream(self.user, "ajoute maths"))

    def test_chaque_outil_produit_un_evenement(self):
        outils = [e for e in self._evenements() if e['type'] == 'tool']
        self.assertEqual(len(outils), 2)
        self.assertEqual([o['name'] for o in outils], ['list_blocks', 'create_block'])

    def test_l_evenement_porte_de_quoi_l_afficher(self):
        """L'interface a besoin du nom, du succes et du message rendu par
        l'outil. Sans le message, le jalon serait vide au depliage."""
        premier = next(e for e in self._evenements() if e['type'] == 'tool')
        for cle in ('name', 'ok', 'message', 'id'):
            self.assertIn(cle, premier)
        self.assertTrue(premier['ok'])
        self.assertEqual(premier['message'], '2 blocs trouves')

    def test_les_outils_arrivent_AVANT_le_compte_rendu(self):
        """Tout l'interet est de meubler l'attente: apres le compte rendu, un
        jalon ne sert plus a rien."""
        types = [e['type'] for e in self._evenements()]
        self.assertLess(types.index('tool'), types.index('delta'))
        self.assertEqual(types[-1], 'done')

    def test_un_echec_est_signale_comme_tel(self):
        """Un refus doit se voir dans la trace, pas seulement dans le texte
        final: c'est la que l'utilisateur comprend ce qui s'est passe."""
        def agir_qui_echoue(self_agent, user, message, registre):
            action = registre.ajouter(
                'create_block', {'title': 'X'},
                ToolResult(success=False, message='Chevauchement detecte'))
            self_agent.signaler_outil(action)
            return ""
        outils = [e for e in self._evenements(agir_qui_echoue) if e['type'] == 'tool']
        self.assertEqual(len(outils), 1)
        self.assertFalse(outils[0]['ok'])
        self.assertIn('Chevauchement', outils[0]['message'])

    def test_aucun_argument_brut_ne_fuit_dans_le_flux(self):
        """Les parametres peuvent contenir du contenu utilisateur. On envoie
        de quoi AFFICHER, jamais le dictionnaire brut."""
        premier = next(e for e in self._evenements() if e['type'] == 'tool')
        self.assertNotIn('args', premier)
        self.assertNotIn('parametres', premier)

    def test_un_agir_sans_outil_ne_produit_aucun_evenement_outil(self):
        """Contre-epreuve: on n'invente pas de jalon."""
        def muet(self_agent, user, message, registre):
            return ""
        types = [e['type'] for e in self._evenements(muet)]
        self.assertNotIn('tool', types)


class SignalerOutilTests(TestCase):
    """La primitive, testee a part: sans file, elle doit etre sans effet, sinon
    tout appel direct a _agir (banc, tests) casserait."""

    def setUp(self):
        from services.agent_v2 import PlannerAgentV2
        self.agent = PlannerAgentV2()

    def test_sans_file_l_appel_ne_leve_pas(self):
        registre = Registre()
        action = registre.ajouter('list_blocks', {}, ToolResult(success=True))
        self.agent.signaler_outil(action)

    def test_avec_file_l_evenement_arrive(self):
        import queue
        self.agent._file_pensees = queue.Queue()
        registre = Registre()
        action = registre.ajouter('list_blocks', {}, ToolResult(
            success=True, message='ok'))
        self.agent.signaler_outil(action)
        genre, charge = self.agent._file_pensees.get_nowait()
        self.assertEqual(genre, 'tool')
        self.assertEqual(charge['name'], 'list_blocks')
