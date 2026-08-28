"""
Le raisonnement doit arriver PENDANT qu'il se produit.

Mesure du 2026-08-28: sur une demande multi-etapes, un tour prend 25 s dont
15 s dans AGIR. Pendant ces 15 s l'utilisateur voit « Reflexion... » puis plus
rien. L'evenement `thinking` existait deja, mais il etait emis APRES le retour
d'AGIR, donc il n'apportait rien a l'attente: il decrivait une reflexion
terminee.

CE QU'ON NE STREAME PAS, ET POURQUOI. La sortie de DIRE passe par la
validation des references: une phrase citant une action qui n'existe pas dans
le registre est SUPPRIMEE a l'assemblage. La streamer mot a mot afficherait
donc le mensonge pendant une seconde avant de l'effacer. On perdrait la
garantie centrale de l'agent pour gagner deux secondes et demie. DIRE reste
non streamee, et c'est un choix, pas une limite technique.
"""
import queue
import time
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase

from services.agent.tools.base import ToolResult
from services.agent_v2.redaction import ReponseDire


class OrdreDesEvenementsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='flux', password='x')
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    @staticmethod
    def _agir_qui_pense(self_agent, user, message, registre):
        """Simule un AGIR lent qui produit du raisonnement en cours de route."""
        for morceau in ("Je regarde ", "le planning ", "du jeudi."):
            self_agent.pousser_pensee(morceau)
            time.sleep(0.05)
        registre.ajouter('create_block', {'title': 'Maths'},
                         ToolResult(success=True, message="Bloc 'Maths' cree"))
        return "Je regarde le planning du jeudi."

    def _evenements(self, agir=None):
        with patch.object(self.Agent, '_agir', agir or self._agir_qui_pense), \
             patch.object(self.Agent, '_dire',
                          return_value=ReponseDire(ouverture="C'est fait.")):
            return list(self.Agent().process_message_stream(self.user, "ajoute maths"))

    def test_le_raisonnement_arrive_avant_le_compte_rendu(self):
        """C'est TOUTE la raison d'etre du streaming: si les pensees arrivent
        apres les faits, elles ne masquent aucune attente."""
        types = [e['type'] for e in self._evenements()]
        self.assertIn('thinking', types)
        self.assertIn('delta', types)
        self.assertLess(types.index('thinking'), types.index('delta'))

    def test_le_raisonnement_arrive_en_plusieurs_morceaux(self):
        """Un seul bloc emis d'un coup serait le comportement d'AVANT,
        simplement deplace: il faut plusieurs evenements."""
        pensees = [e for e in self._evenements() if e['type'] == 'thinking']
        self.assertGreaterEqual(len(pensees), 2)

    def test_done_reste_le_dernier_evenement(self):
        """Le contrat SSE est additif: le client remplace toujours la bulle
        par done.response, et done doit rester terminal."""
        evts = self._evenements()
        self.assertEqual(evts[-1]['type'], 'done')
        self.assertIn('response', evts[-1])

    def test_un_agir_muet_ne_produit_aucun_thinking(self):
        """Contre-epreuve: on n'invente pas d'evenement quand le modele ne
        raisonne pas. Un thinking vide ferait clignoter le volet pour rien."""
        def muet(self_agent, user, message, registre):
            return ""
        types = [e['type'] for e in self._evenements(muet)]
        self.assertNotIn('thinking', types)

    def test_une_panne_d_agir_ne_bloque_pas_le_flux(self):
        """AGIR tourne maintenant dans un thread: une exception qui y reste
        coincee ferait attendre le client jusqu'au timeout."""
        def casse(self_agent, user, message, registre):
            raise RuntimeError("modele indisponible")
        evts = self._evenements(casse)
        self.assertEqual(evts[-1]['type'], 'done')

    def test_le_raisonnement_complet_reste_dans_le_done(self):
        """Les clients qui ignorent `thinking` (binaire iOS deja publie) le
        recuperent quand meme dans la charge utile finale."""
        evts = self._evenements()
        self.assertIn('planning', evts[-1].get('raisonnement', ''))


class PousseePenseeTests(TestCase):
    """La primitive elle-meme, testee a part: elle doit etre sans effet quand
    personne n'ecoute, sinon tout appel direct a _agir (banc, tests) casserait."""

    def setUp(self):
        from services.agent_v2 import PlannerAgentV2
        self.agent = PlannerAgentV2()

    def test_sans_file_la_poussee_ne_leve_pas(self):
        self.agent.pousser_pensee("perdu dans le vide")

    def test_avec_file_la_poussee_arrive(self):
        self.agent._file_pensees = queue.Queue()
        self.agent.pousser_pensee("une pensee")
        self.assertEqual(self.agent._file_pensees.get_nowait(), ("thinking", "une pensee"))

    def test_une_pensee_vide_n_est_pas_poussee(self):
        self.agent._file_pensees = queue.Queue()
        self.agent.pousser_pensee("")
        self.agent.pousser_pensee(None)
        self.assertTrue(self.agent._file_pensees.empty())


class RepliSansStreamingTests(TestCase):
    """Tous les fournisseurs ne streament pas leurs deltas de raisonnement.

    Sans repli, leur raisonnement n'atteindrait le client que dans la charge
    utile finale et le volet resterait vide tout le tour. On perd alors le
    gain de latence, jamais l'information."""

    def setUp(self):
        self.user = User.objects.create_user(username='repli', password='x')
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def test_un_agir_qui_ne_pousse_rien_emet_quand_meme_son_raisonnement(self):
        def sans_deltas(self_agent, user, message, registre):
            return "Raisonnement rendu d'un bloc, sans fragments."

        with patch.object(self.Agent, '_agir', sans_deltas), \
             patch.object(self.Agent, '_dire',
                          return_value=ReponseDire(ouverture="Ok.")):
            evts = list(self.Agent().process_message_stream(self.user, "bonjour"))

        pensees = [e for e in evts if e['type'] == 'thinking']
        self.assertEqual(len(pensees), 1)
        self.assertIn("d'un bloc", pensees[0]['text'])

    def test_un_agir_qui_pousse_ne_reemet_PAS_le_bloc_entier(self):
        """Contre-epreuve: sans elle, un fournisseur qui streame afficherait
        son raisonnement deux fois, en fragments puis en entier."""
        def avec_deltas(self_agent, user, message, registre):
            self_agent.pousser_pensee("un fragment")
            return "un fragment"

        with patch.object(self.Agent, '_agir', avec_deltas), \
             patch.object(self.Agent, '_dire',
                          return_value=ReponseDire(ouverture="Ok.")):
            evts = list(self.Agent().process_message_stream(self.user, "bonjour"))

        self.assertEqual(len([e for e in evts if e['type'] == 'thinking']), 1)
