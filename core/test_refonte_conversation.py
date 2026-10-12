"""Contrat du tour execute avec le SDK et des modeles locaux scripts."""
import asyncio
import json
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TransactionTestCase, override_settings
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from core.models import BudgetJetonsJournalier, ConversationMessage, PreferenceUtilisateur
from services.agent_v2.conversation import AgentConversation, DecisionTour


def modele_decision(payload, appels):
    def repondre(messages, info):
        appels.append((messages, info))
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)],
                             usage=RequestUsage(input_tokens=100, output_tokens=20))
    return FunctionModel(repondre)


class ContratConversationTests(TransactionTestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username='contrat-tour')

    def executer(self, message, payload):
        appels = []
        with patch('services.agent_v2.conversation.modele_agir',
                   return_value=modele_decision(payload, appels)), \
             patch('services.agent_v2.agent.PlannerAgentV2.process_message_stream') as planning:
            resultat = AgentConversation().process_message(self.user, message)
        return resultat, appels, planning

    def test_bonjour_ne_charge_ni_calendrier_ni_outils(self):
        with patch('services.agent_v2.prompts.build_context', side_effect=AssertionError('calendrier')):
            resultat, appels, planning = self.executer(
                'Bonjour', {'mode': 'conversation', 'reponse': 'Salut ! Comment tu vas ?'})
        self.assertEqual(resultat['response'], 'Salut ! Comment tu vas ?')
        self.assertEqual(len(appels), 1)
        self.assertEqual(appels[0][1].function_tools, [])
        planning.assert_not_called()
        self.assertEqual(BudgetJetonsJournalier.objects.get(user=self.user).jetons, 120)
        self.assertEqual(ConversationMessage.objects.filter(user=self.user).count(), 2)

    def test_message_long_sans_demande_ne_devient_pas_du_planning(self):
        _, appels, planning = self.executer(
            'Je passe simplement te souhaiter une excellente journee et te dire merci pour hier',
            {'mode': 'conversation', 'reponse': 'Merci, bonne journée à toi aussi !'})
        self.assertEqual(len(appels), 1)
        planning.assert_not_called()

    def test_doute_pose_question_sans_outil(self):
        resultat, _, planning = self.executer(
            'Et pour la suite ?', {'mode': 'clarification', 'reponse': 'Tu veux reprendre quel point ?'})
        planning.assert_not_called()
        self.assertTrue(resultat['question_posee'])

    def test_planning_autorise_delegue_une_seule_fois(self):
        payload = {'mode': 'planning', 'objectif': 'Consulter la semaine', 'consultation': 'semaine'}
        appels = []
        with patch('services.agent_v2.conversation.modele_agir',
                   return_value=modele_decision(payload, appels)), \
             patch('services.agent_v2.agent.PlannerAgentV2.process_message_stream',
                   return_value=iter([{'type': 'done', 'response': 'Semaine lue'}])) as planning:
            agent = AgentConversation()
            resultat = agent.process_message(self.user, 'Montre ma semaine')
        self.assertEqual(resultat['response'], 'Semaine lue')
        planning.assert_called_once()
        self.assertEqual(agent._decision_tour.mode, 'planning')
        self.assertEqual(len(appels), 1)

    def test_memoire_semantique_ne_relance_pas_la_boucle(self):
        resultat, _, planning = self.executer(
            'Garde en tete mon envie de courir le mardi',
            {'mode': 'memoire', 'memoire_action': 'memoriser', 'enonce': 'Je cours le mardi'})
        planning.assert_not_called()
        self.assertTrue(PreferenceUtilisateur.objects.filter(user=self.user, actif=True,
                                                             enonce='Je cours le mardi').exists())
        self.assertIn('mardi', resultat['response'])

    def test_panne_ne_donne_jamais_acces_au_planning(self):
        with patch('services.agent_v2.conversation.modele_agir', side_effect=RuntimeError('indisponible')), \
             patch('services.agent_v2.agent.PlannerAgentV2.process_message_stream') as planning:
            resultat = AgentConversation().process_message(self.user, 'Bonjour')
        planning.assert_not_called()
        self.assertTrue(resultat['response'])
        metadata = ConversationMessage.objects.filter(role='assistant').latest('pk').metadata
        self.assertFalse(metadata['usage'][0]['usage_connu'])

    @override_settings(AGENT_DECISION_DELAI=0.2)
    def test_timeout_annule_le_modele_et_ne_lance_aucun_outil(self):
        annule = []
        async def lent(messages, info):
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                annule.append(True)
                raise
            return ModelResponse(parts=[])
        with patch('services.agent_v2.conversation.modele_agir', return_value=FunctionModel(lent)), \
             patch('services.agent_v2.agent.PlannerAgentV2.process_message_stream') as planning:
            AgentConversation().process_message(self.user, 'Bonjour')
        self.assertEqual(annule, [True])
        planning.assert_not_called()

    def test_historique_decision_borne_et_sans_donnees_autre_utilisateur(self):
        autre = get_user_model().objects.create_user(username='autre-contrat')
        ConversationMessage.objects.create(user=autre, role='user', content='SECRET-AUTRE')
        for i in range(20):
            ConversationMessage.objects.create(user=self.user, role='user', content=str(i) + 'x'*5000)
        _, appels, _ = self.executer('Bonjour', {'mode': 'conversation', 'reponse': 'Salut !'})
        texte = str(appels[0][0])
        self.assertNotIn('SECRET-AUTRE', texte)
        self.assertLess(len(texte), 18000)

    def test_pas_de_suggestions_payantes_pour_une_conversation(self):
        self.executer('Bonjour', {'mode': 'conversation', 'reponse': 'Salut !'})
        with patch('services.agent.agent.PlannerAgent.quick_replies_for') as suggestions:
            self.assertEqual(AgentConversation().quick_replies_for(self.user, 'Bonjour', 'Salut !'), [])
        suggestions.assert_not_called()

    def test_schema_interdit_memoire_vide_et_conversation_avec_consultation(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            DecisionTour(mode='memoire', memoire_action='memoriser')
        with self.assertRaises(ValidationError):
            DecisionTour(mode='conversation', reponse='Salut', consultation='semaine')

    def test_tap_invente_n_ouvre_pas_le_planning(self):
        appels = []
        with patch('services.agent_v2.conversation.modele_agir', return_value=modele_decision(
                {'mode': 'conversation', 'reponse': 'Salut !'}, appels)), \
             patch('services.agent_v2.agent.PlannerAgentV2.process_message_stream') as planning:
            AgentConversation().process_message(self.user, 'Salut', tap={'demande': 'inventee', 'option': 'confirmer'})
        self.assertEqual(len(appels), 1)
        planning.assert_not_called()

    def test_recit_mutation_sans_outil_est_epure(self):
        resultat, _, planning = self.executer('Bonjour', {
            'mode': 'conversation', 'reponse': "J'ai supprimé tous tes cours."})
        planning.assert_not_called()
        self.assertNotIn('supprimé', resultat['response'])

    def test_consultation_traverse_vraie_boucle_sdk_et_orm(self):
        from services.agent_v2.redaction import ReponseDire
        requetes = []
        def planifier(messages, info):
            requetes.append(info)
            if len(requetes) == 1:
                return ModelResponse(parts=[ToolCallPart('get_week_schedule', {})],
                                     usage=RequestUsage(input_tokens=100, output_tokens=20))
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, ReponseDire().model_dump())],
                                 usage=RequestUsage(input_tokens=100, output_tokens=20))
        with patch('services.agent_v2.conversation.modele_agir', return_value=modele_decision({
                'mode': 'planning', 'objectif': 'Voir la semaine', 'consultation': 'semaine'}, [])), \
             patch('services.agent_v2.agent.modele_agir', return_value=FunctionModel(planifier)), \
             patch.object(AgentConversation, '_sur_evenements', None), \
             patch('services.agent_v2.agent._jugement.juger', side_effect=AssertionError('jugement superflu')), \
             self.assertLogs('services.agent_v2.agent', level='DEBUG') as logs:
            resultat = AgentConversation().process_message(self.user, 'Montre ma semaine')
        self.assertEqual(len(requetes), 2, logs.output)
        self.assertNotIn('create_block', [t.name for t in requetes[0].function_tools])
        self.assertTrue(resultat['response'])
        meta = ConversationMessage.objects.filter(user=self.user, role='assistant').latest('pk').metadata
        self.assertEqual(meta['contrat_tour'], 'planning')
        self.assertEqual(meta['actions'][0]['outil'], 'get_week_schedule')
        self.assertTrue(meta['actions'][0]['succes'])
        self.assertEqual(BudgetJetonsJournalier.objects.get(user=self.user).jetons, 360)

    def test_trace_jugement_traverse_thread_et_compte_une_fois(self):
        from services.agent_v2.trace_tour import trace_courante
        from services.agent_v2.redaction import ReponseDire
        def boucle(agent, user, message, registre):
            trace = trace_courante.get()
            self.assertIsNotNone(trace)
            from services.agent_v2.budget_tour import appel_fournisseur
            with appel_fournisseur(phase='jugement', fournisseur='local',
                                  jetons_entree=200, max_sortie=10) as appel:
                appel.terminer({'entree': 20, 'sortie': 1}, True)
            return ReponseDire()
        with patch('services.agent_v2.conversation.modele_agir', return_value=modele_decision({
                'mode': 'planning', 'objectif': 'Voir des taches', 'consultation': 'autre'}, [])), \
             patch('services.agent_v2.agent.PlannerAgentV2._boucle', boucle):
            AgentConversation().process_message(self.user, 'Vois les taches')
        self.assertIsNone(trace_courante.get())
        self.assertEqual(BudgetJetonsJournalier.objects.get(user=self.user).jetons, 141)
        meta = ConversationMessage.objects.filter(user=self.user, role='assistant').latest('pk').metadata
        self.assertEqual([u['entree'] for u in meta['usage'] if u['phase'] == 'jugement'], [20])

    def test_panne_compteur_ne_transforme_pas_reponse_en_echec(self):
        with patch('services.agent_v2.budget_tour.ReservationTour.rapprocher', side_effect=RuntimeError('compteur')):
            resultat, _, planning = self.executer('Bonjour', {'mode': 'conversation', 'reponse': 'Salut !'})
        self.assertEqual(resultat['response'], 'Salut !')
        planning.assert_not_called()
        self.assertGreater(BudgetJetonsJournalier.objects.get(user=self.user).jetons, 120)

    def test_document_joint_conserve_chemin_recu_meme_avec_salutation(self):
        from types import SimpleNamespace
        with patch('services.agent_v2.conversation.modele_agir', return_value=modele_decision({
                'mode': 'conversation', 'reponse': 'Salut !'}, [])), \
             patch('services.agent_v2.agent.PlannerAgentV2.process_message_stream',
                   return_value=iter([{'type': 'done', 'response': 'Document pris en compte'}])) as planning:
            agent = AgentConversation()
            agent.process_message(self.user, 'Bonjour', attachment=SimpleNamespace(processed=True))
        planning.assert_called_once()
        self.assertEqual(agent._decision_tour.mode, 'planning')
        self.assertFalse(agent._decision_tour.autoriser_modifications)


class ReponseAUneQuestionDuCodeTests(TransactionTestCase):
    """Le routeur ne doit jamais avaler la reponse a une question du code.

    Le code retient une action destructive et pose sa question. Le message
    suivant est peut-etre le « oui ». Si le routeur y repond lui-meme, l'action
    retenue meurt en silence et personne ne sait que la suppression demandee
    n'a pas eu lieu. Symetrique du filet du document joint juste au-dessus.
    """

    def setUp(self):
        self.user = get_user_model().objects.create_user(username='reponse-garde')

    def _question_posee(self, minutes=0):
        from datetime import timedelta
        from django.utils import timezone
        precedent = ConversationMessage.objects.create(
            user=self.user, role='user', content='supprime mon cours de chimie')
        emise = timezone.now() - timedelta(minutes=minutes)
        ConversationMessage.objects.create(
            user=self.user, role='assistant',
            content='Je supprime toutes les occurrences ou juste celle-la ?',
            metadata={'en_reponse_a': precedent.pk, 'question_posee': True,
                      'demandes': [{'cle': 'portee:1', 'motif': 'portee_jour',
                                    'emise_le': emise.isoformat(),
                                    'options': [{'id': 'serie', 'libelle': 'Toutes'},
                                                {'id': 'jour', 'libelle': 'Juste celle-la'}]}]})

    def _tour(self, message, payload):
        with patch('services.agent_v2.conversation.modele_agir',
                   return_value=modele_decision(payload, [])), \
             patch('services.agent_v2.agent.PlannerAgentV2.process_message_stream',
                   return_value=iter([{'type': 'done', 'response': 'Traite'}])) as planning:
            agent = AgentConversation()
            agent.process_message(self.user, message)
        return agent, planning

    def test_un_oui_rejoint_la_garde_meme_classe_en_conversation(self):
        self._question_posee()
        agent, planning = self._tour('oui', {'mode': 'conversation', 'reponse': 'D accord !'})
        planning.assert_called_once()
        self.assertEqual(agent._decision_tour.mode, 'planning')
        # La garde en aval doit pouvoir executer l'action retenue.
        self.assertTrue(agent._decision_tour.autoriser_modifications)

    def test_sans_question_en_attente_la_conversation_reste_au_routeur(self):
        agent, planning = self._tour('merci !', {'mode': 'conversation', 'reponse': 'Avec plaisir !'})
        planning.assert_not_called()
        self.assertEqual(agent._decision_tour.mode, 'conversation')

    def test_une_question_perimee_ne_rouvre_pas_la_boucle(self):
        self._question_posee(minutes=45)
        agent, planning = self._tour('oui', {'mode': 'conversation', 'reponse': 'D accord !'})
        planning.assert_not_called()
        self.assertEqual(agent._decision_tour.mode, 'conversation')

    def test_une_question_d_un_autre_echange_ne_compte_pas(self):
        """La demande doit repondre au message precedent, pas flotter."""
        from django.utils import timezone
        ConversationMessage.objects.create(user=self.user, role='user', content='autre chose')
        ConversationMessage.objects.create(
            user=self.user, role='assistant', content='Je supprime quoi ?',
            metadata={'en_reponse_a': 999999,
                      'demandes': [{'cle': 'portee:1', 'motif': 'portee_jour',
                                    'emise_le': timezone.now().isoformat(),
                                    'options': [{'id': 'serie', 'libelle': 'Toutes'}]}]})
        agent, planning = self._tour('oui', {'mode': 'conversation', 'reponse': 'D accord !'})
        planning.assert_not_called()
        self.assertEqual(agent._decision_tour.mode, 'conversation')
