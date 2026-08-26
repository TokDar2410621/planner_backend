"""
La boucle complete, avec des modeles SIMULES (aucun appel reseau).

On patche la CLASSE, jamais une instance: la boucle en fabrique plusieurs.

Le scenario du 18 aout est rejoue avec un modele qui TENTE de mentir, et une
contre-epreuve verifie qu'un recit VRAI survit: sans elle, un agent qui
supprime tout passerait le premier test.
"""
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase, override_settings

from core.models import ConversationMessage
from services.agent.tools.base import ToolResult
from services.agent_v2.redaction import ActionCitee, ReponseDire


def _agir_muet(self_agent, user, message, registre):
    """AGIR qui n'appelle aucun outil: le registre reste vide."""
    return None


def _agir_qui_cree(self_agent, user, message, registre):
    registre.ajouter('create_block', {'title': 'Maths'},
                     ToolResult(success=True, message="Bloc 'Maths' cree"))
    return None


class BoucleTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='boucle', password='x')
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def test_un_tour_sans_outil_ne_livre_aucune_affirmation(self):
        """Le 18 aout: « je vais supprimer puis ajouter », zero outil."""
        menteur = ReponseDire(
            ouverture="Absolument.",
            actions=[ActionCitee(ref='a1',
                                 phrase="J'ai supprime les blocs qui chevauchent.")],
            suite="")
        with patch.object(self.Agent, '_agir', _agir_muet), \
             patch.object(self.Agent, '_dire', return_value=menteur):
            res = self.Agent().process_message(self.user, "mes cours sont prioritaires")
        self.assertNotIn('supprime les blocs', res['response'])
        self.assertIn('Absolument', res['response'])

    def test_un_recit_vrai_survit(self):
        """Contre-epreuve: sans elle, un agent qui supprime tout passerait."""
        vrai = ReponseDire(
            ouverture="C'est fait.",
            actions=[ActionCitee(ref='a1', phrase="Maths est cale le lundi.")],
            suite="")
        with patch.object(self.Agent, '_agir', _agir_qui_cree), \
             patch.object(self.Agent, '_dire', return_value=vrai):
            res = self.Agent().process_message(self.user, "ajoute maths")
        self.assertIn('Maths est cale', res['response'])
        self.assertIn('Bloc', res['response'])

    def test_les_quatre_cles_du_contrat_sont_presentes(self):
        """views.py:861 lit result['response'] par indexation DIRECTE: une cle
        manquante rend un 500 a l'utilisateur."""
        with patch.object(self.Agent, '_agir', _agir_muet), \
             patch.object(self.Agent, '_dire', return_value=ReponseDire(ouverture="Salut.")):
            res = self.Agent().process_message(self.user, "bonjour")
        for cle in ('response', 'quick_replies', 'blocks_created', 'tasks_created'):
            self.assertIn(cle, res)

    def test_le_flux_emet_done_en_dernier(self):
        with patch.object(self.Agent, '_agir', _agir_muet), \
             patch.object(self.Agent, '_dire', return_value=ReponseDire(ouverture="Salut.")):
            evts = list(self.Agent().process_message_stream(self.user, "bonjour"))
        self.assertEqual(evts[-1]['type'], 'done')
        self.assertIn('response', evts[-1])

    def test_quick_replies_for_existe_et_ne_leve_jamais(self):
        """Une vue l'appelle et avale les exceptions: sans cette methode, les
        chips disparaitraient en silence pour tout compte bascule."""
        res = self.Agent().quick_replies_for(self.user, "", "")
        self.assertEqual(res, [])


class PersistanceTests(TestCase):
    """L'historique doit rester lisible par v1: meme modele, memes roles. Un
    compte bascule sur v2 puis ramene sur v1 ne doit rien perdre."""

    def setUp(self):
        self.user = User.objects.create_user(username='persist', password='x')
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def _tour(self, message="bonjour", reponse=None):
        reponse = reponse or ReponseDire(ouverture="Salut.")
        with patch.object(self.Agent, '_agir', _agir_muet), \
             patch.object(self.Agent, '_dire', return_value=reponse):
            return self.Agent().process_message(self.user, message)

    def test_les_deux_messages_du_tour_sont_persistes(self):
        self._tour("bonjour")
        roles = list(ConversationMessage.objects.filter(user=self.user)
                     .order_by('created_at').values_list('role', flat=True))
        self.assertEqual(roles, ['user', 'assistant'])

    def test_le_message_utilisateur_n_est_pas_duplique(self):
        """Filet B9 de v1: le message etait sauve, relu, puis rajoute."""
        self._tour("bonjour")
        self.assertEqual(
            ConversationMessage.objects.filter(user=self.user, role='user').count(), 1)

    def test_l_historique_envoye_au_modele_exclut_le_message_courant(self):
        """Compter les lignes en base ne prouve rien: c'est l'historique PASSE
        au modele qui dupliquerait le tour courant. On l'inspecte donc.

        v1 sauvait, relisait, puis rajoutait le message: il partait deux fois a
        chaque requete. Ici l'exclusion se fait par cle primaire, donc la
        duplication est structurellement impossible."""
        self._tour("premier message")
        vus = {}

        def _agir_qui_regarde(self_agent, user, message, registre):
            vus['historique'] = [
                p.content for m in self_agent._historique(user)
                for p in m.parts if hasattr(p, 'content')
            ]
            return None

        with patch.object(self.Agent, '_agir', _agir_qui_regarde), \
             patch.object(self.Agent, '_dire', return_value=ReponseDire(ouverture="Ok.")):
            self.Agent().process_message(self.user, "deuxieme message")

        self.assertIn('premier message', vus['historique'])
        self.assertNotIn('deuxieme message', vus['historique'])

    def test_la_reponse_persistee_est_celle_qui_est_rendue(self):
        res = self._tour("ajoute maths")
        dernier = ConversationMessage.objects.filter(
            user=self.user, role='assistant').latest('created_at')
        self.assertEqual(dernier.content, res['response'])


@override_settings(DEEPSEEK_API_KEY='factice')
class ReglagesTests(TestCase):
    """Le test de construction verifiait que la constante vaut ce qu'elle vaut.
    Ici on verifie qu'elle atteint VRAIMENT l'agent DIRE: sans ce reglage,
    DeepSeek refuse tool_choice=required en mode thinking et DIRE echoue dix
    fois sur dix (mesure du 2026-08-24)."""

    def setUp(self):
        self.user = User.objects.create_user(username='reglages', password='x')
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def test_dire_construit_son_agent_avec_le_reglage_qui_coupe_le_raisonnement(self):
        from services.agent_v2 import agent as module_agent
        from services.agent_v2.modeles import REGLAGES_DIRE

        vus = {}

        class AgentEspion:
            def __init__(self, *a, **kw):
                vus.update(kw)

            def run_sync(self, *a, **kw):
                class R:
                    output = ReponseDire(ouverture="Salut.")
                return R()

        with patch.object(module_agent, 'Agent', AgentEspion), \
             patch.object(self.Agent, '_agir', _agir_muet):
            self.Agent().process_message(self.user, "bonjour")

        self.assertEqual(vus.get('model_settings'), REGLAGES_DIRE)


class MatierePremiereTests(TestCase):
    """DIRE doit RECEVOIR le contenu des lectures.

    Defaut observe le 2026-08-25 sur un tour reel: a « tu peux me rappeler ce
    que j'ai mardi ? », l'agent appelait bien list_blocks puis repondait « J'ai
    trouve votre bloc de mardi. Veux-tu que je t'en donne le detail ? ». Il
    avait la donnee et ne la livrait pas.

    Cause: bloc_factuel ne rend que les MUTATIONS, et le brief ne passait que
    le message de chaque action. Sur un tour de pure lecture, DIRE savait
    qu'un outil avait tourne sans savoir ce qu'il avait renvoye.
    """

    def setUp(self):
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def _brief_avec(self, outil, donnees, message=""):
        from services.agent_v2.registre import Registre
        registre = Registre()
        registre.ajouter(outil, {}, ToolResult(
            success=True, data=donnees, message=message))
        return self.Agent._brief_dire("et mardi ?", registre, {}, ''), registre

    def test_le_contenu_d_une_lecture_arrive_dans_le_brief(self):
        brief, _ = self._brief_avec(
            'list_blocks',
            {'blocks': [{'title': 'Cours de chimie', 'day_name': 'Mardi',
                         'start_time': '09:00', 'end_time': '11:00'}]},
            message="1 bloc trouve")
        self.assertIn('Cours de chimie', brief)
        self.assertIn('09:00', brief)

    def test_une_mutation_ne_deverse_PAS_ses_donnees_dans_le_brief(self):
        """Contre-epreuve: le recit des mutations reste tenu par le bloc
        factuel et la validation des references. Deverser leurs donnees
        brutes rouvrirait le canal que la garantie structurelle ferme."""
        brief, _ = self._brief_avec(
            'create_block',
            {'created': [{'title': 'Bloc secret', 'id': 42}]},
            message="1 bloc cree")
        self.assertNotIn('Bloc secret', brief)
        self.assertIn('1 bloc cree', brief)


class ResilienceTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='resilience', password='x')
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def test_une_panne_de_dire_apres_mutation_annonce_QUAND_MEME_les_faits(self):
        """Le pire cas du produit: on a ecrit dans le planning et le redacteur
        tombe. Se taire laisserait l'utilisateur croire que rien n'a eu lieu,
        alors que son planning a change."""
        with patch.object(self.Agent, '_agir', _agir_qui_cree), \
             patch.object(self.Agent, '_dire', side_effect=RuntimeError('502')):
            res = self.Agent().process_message(self.user, "ajoute maths")
        self.assertIn('Maths', res['response'])

    def test_un_budget_epuise_est_dit_a_l_utilisateur(self):
        def _agir_sature(self_agent, user, message, registre):
            registre.budget_epuise = True
            return None

        with patch.object(self.Agent, '_agir', _agir_sature), \
             patch.object(self.Agent, '_dire', return_value=ReponseDire(ouverture="Bon.")):
            res = self.Agent().process_message(self.user, "fais tout")
        self.assertIn('interrompu', res['response'].lower())
