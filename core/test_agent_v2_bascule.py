"""
L'aiguillage v1 / v2, par compte.

La bascule ET le retour arriere doivent etre un UPDATE en base, sans
deploiement: c'est la seule facon de ramener un utilisateur sur v1 en quelques
secondes si v2 derape en production.

Les TROIS vues sont couvertes. Une seule oubliee laisserait un compte bascule
avec un agent sur /chat/ et l'autre sur /chat/stream/, donc un historique
ecrit par deux boucles differentes.
"""
import json

from core.lecture_flux import corps_du_flux
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from core.models import ConversationMessage


class AiguillageTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='bascule', password='x')
        # Le consentement IA (Apple 5.1.2(i)) garde les trois vues.
        self.user.profile.ai_consent_at = timezone.now()
        self.user.profile.save()
        self.client.force_authenticate(user=self.user)

    def _basculer(self, valeur):
        self.user.profile.agent_v2 = valeur
        self.user.profile.save()

    def test_le_drapeau_est_faux_par_defaut(self):
        """Un deploiement ne doit basculer personne."""
        self.assertFalse(User.objects.create_user(
            username='neuf', password='x').profile.agent_v2)

    def test_sans_drapeau_chat_utilise_v1(self):
        with patch('core.views.PlannerAgent') as v1:
            v1.return_value.process_message.return_value = {'response': 'ok'}
            self.client.post('/api/chat/', {'message': 'salut'})
        self.assertTrue(v1.called)

    def test_avec_drapeau_chat_utilise_v2(self):
        self._basculer(True)
        with patch('services.agent_v2.PlannerAgentV2') as v2, \
             patch('core.views.PlannerAgent') as v1:
            v2.return_value.process_message.return_value = {'response': 'ok'}
            self.client.post('/api/chat/', {'message': 'salut'})
        self.assertTrue(v2.called)
        self.assertFalse(v1.called)

    def test_avec_drapeau_quick_replies_utilise_v2(self):
        self._basculer(True)
        with patch('services.agent_v2.PlannerAgentV2') as v2, \
             patch('core.views.PlannerAgent') as v1:
            v2.return_value.quick_replies_for.return_value = []
            self.client.post('/api/chat/quick-replies/',
                             {'message': 'a', 'response': 'b'})
        self.assertTrue(v2.called)
        self.assertFalse(v1.called)

    def test_avec_drapeau_le_flux_utilise_v2(self):
        self._basculer(True)
        with patch('services.agent_v2.PlannerAgentV2') as v2, \
             patch('core.views.PlannerAgent') as v1:
            v2.return_value.process_message_stream.return_value = iter(
                [{'type': 'done', 'response': 'ok'}])
            reponse = self.client.post('/api/chat/stream/', {'message': 'salut'})
            corps_du_flux(reponse).encode('utf-8')
        self.assertTrue(v2.called)
        self.assertFalse(v1.called)

    def test_sans_drapeau_le_flux_utilise_v1(self):
        with patch('core.views.PlannerAgent') as v1:
            v1.return_value.process_message_stream.return_value = iter(
                [{'type': 'done', 'response': 'ok'}])
            reponse = self.client.post('/api/chat/stream/', {'message': 'salut'})
            corps_du_flux(reponse).encode('utf-8')
        self.assertTrue(v1.called)

    def test_un_profil_absent_retombe_sur_v1(self):
        """Un utilisateur sans profil ne doit pas rendre un 500: la bascule
        est un confort, pas une dependance."""
        from core.views import _agent_pour
        from services.agent import PlannerAgent
        orphelin = User.objects.create_user(username='orphelin', password='x')
        orphelin.profile.delete()
        orphelin.refresh_from_db()
        self.assertIsInstance(_agent_pour(orphelin), PlannerAgent)


class FormatCommunTests(TestCase):
    """N'inspecte PAS le modele Django (ce serait vrai quoi qu'il arrive):
    fait ecrire un tour par CHAQUE agent et compare les messages produits.

    Un compte bascule sur v2 puis ramene sur v1 doit retrouver un historique
    lisible, et reciproquement."""

    def setUp(self):
        self.u1 = User.objects.create_user(username='fmt1', password='x')
        self.u2 = User.objects.create_user(username='fmt2', password='x')

    def _messages(self, user):
        return list(ConversationMessage.objects.filter(user=user)
                    .order_by('created_at').values_list('role', flat=True))

    def test_les_deux_agents_ecrivent_le_meme_format(self):
        from services.agent.agent import PlannerAgent
        from services.agent_v2 import PlannerAgentV2
        from services.agent_v2.redaction import ReponseDire
        from services.llm.base import LLMResponse

        faux = type('FauxLLM', (), {
            'supports_streaming': False,
            'is_available': lambda self: True,
            'generate_with_history': lambda self, **kw: LLMResponse(text="Salut."),
            'generate': lambda self, *a, **kw: LLMResponse(text="Salut."),
        })()
        with patch.object(PlannerAgent, '_build_provider', return_value=faux):
            PlannerAgent().process_message(self.u1, "bonjour")

        with patch.object(PlannerAgentV2, '_agir', lambda s, u, m, r: None), \
             patch.object(PlannerAgentV2, '_dire',
                          return_value=ReponseDire(ouverture="Salut.")):
            PlannerAgentV2().process_message(self.u2, "bonjour")

        self.assertEqual(self._messages(self.u1), self._messages(self.u2))
        self.assertEqual(self._messages(self.u2), ['user', 'assistant'])

    def test_v2_ne_persiste_jamais_le_raisonnement(self):
        """Le raisonnement est ephemere a l'affichage (spec 3.4). S'il entrait
        dans l'historique, il reviendrait dans le contexte de chaque tour
        suivant, y compris apres un retour sur v1."""
        from services.agent_v2 import PlannerAgentV2
        from services.agent_v2.redaction import ReponseDire

        with patch.object(PlannerAgentV2, '_agir',
                          lambda s, u, m, r: "PENSEE INTERNE SECRETE"), \
             patch.object(PlannerAgentV2, '_dire',
                          return_value=ReponseDire(ouverture="Salut.")):
            res = PlannerAgentV2().process_message(self.u2, "bonjour")

        self.assertEqual(res['raisonnement'], "PENSEE INTERNE SECRETE")
        for message in ConversationMessage.objects.filter(user=self.u2):
            self.assertNotIn("SECRETE", message.content)
