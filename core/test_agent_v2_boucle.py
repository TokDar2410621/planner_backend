"""
La boucle complete, avec une boucle SIMULEE (aucun appel reseau).

On patche la CLASSE, jamais une instance: la boucle en fabrique plusieurs.

Le scenario du 18 aout est rejoue avec un modele qui TENTE de mentir, et une
contre-epreuve verifie qu'un recit VRAI survit: sans elle, un agent qui
supprime tout passerait le premier test.
"""
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase, TransactionTestCase, override_settings

from core.models import ConversationMessage
from services.agent.tools.base import ToolResult
from services.agent_v2.redaction import ActionCitee, ReponseDire


def _effet_muet(self_agent, user, message, registre):
    """Boucle simulee qui n'appelle aucun outil: le registre reste vide."""


def _effet_qui_cree(self_agent, user, message, registre):
    # Les donnees ont la forme que rend create_block: le rendu des faits lit
    # les donnees, jamais le message ecrit pour le modele.
    registre.ajouter('create_block', {'title': 'Maths'},
                     ToolResult(success=True, message="Bloc 'Maths' cree",
                                data={'created': [{
                                    'title': 'Maths', 'day_of_week': 0,
                                    'day_name': 'Lundi', 'start_time': '09:00',
                                    'end_time': '12:00'}]}))


def _boucle_simulee(effet, reponse):
    """Une _boucle patchee: l'effet remplit le registre, puis la reponse
    structuree simulee est rendue."""
    def _boucle(self_agent, user, message, registre):
        effet(self_agent, user, message, registre)
        return reponse
    return _boucle


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
        with patch.object(self.Agent, '_boucle',
                          _boucle_simulee(_effet_muet, menteur)):
            res = self.Agent().process_message(self.user, "mes cours sont prioritaires")
        self.assertNotIn('supprime les blocs', res['response'])
        # Boucle unique: verifier_prose coupe phrase par phrase, pas en
        # guillotine. L'affirmation d'action sans recu est coupee; la
        # garantie de securite tient (aucune action affirmee sans recu).
        self.assertNotIn("J'ai supprime", res['response'])
        self.assertTrue(res['response'].strip())

    def test_un_recit_vrai_survit(self):
        """Contre-epreuve: sans elle, un agent qui supprime tout passerait.

        Un seul narrateur depuis le 2026-09-14: l'action vraie est racontee
        par les faits rendus par le code, et la phrase citee par le modele
        n'est plus recopiee a cote (elle doublait chaque ligne)."""
        vrai = ReponseDire(
            ouverture="C'est fait.",
            actions=[ActionCitee(ref='a1', phrase="Maths est cale le lundi.")],
            suite="")
        with patch.object(self.Agent, '_boucle',
                          _boucle_simulee(_effet_qui_cree, vrai)):
            res = self.Agent().process_message(self.user, "ajoute maths")
        self.assertIn('Maths', res['response'])
        self.assertNotIn('Maths est cale', res['response'])

    def test_les_quatre_cles_du_contrat_sont_presentes(self):
        """views.py:861 lit result['response'] par indexation DIRECTE: une cle
        manquante rend un 500 a l'utilisateur."""
        with patch.object(self.Agent, '_boucle',
                          _boucle_simulee(_effet_muet, ReponseDire(ouverture="Salut."))):
            res = self.Agent().process_message(self.user, "bonjour")
        for cle in ('response', 'quick_replies', 'blocks_created', 'tasks_created'):
            self.assertIn(cle, res)

    def test_le_flux_emet_done_en_dernier(self):
        with patch.object(self.Agent, '_boucle',
                          _boucle_simulee(_effet_muet, ReponseDire(ouverture="Salut."))):
            evts = list(self.Agent().process_message_stream(self.user, "bonjour"))
        self.assertEqual(evts[-1]['type'], 'done')
        self.assertIn('response', evts[-1])

    def test_quick_replies_for_existe_et_ne_leve_jamais(self):
        """Une vue l'appelle et avale les exceptions: sans cette methode, les
        chips disparaitraient en silence pour tout compte bascule."""
        res = self.Agent().quick_replies_for(self.user, "", "")
        self.assertEqual(res, [])


class PersistanceTests(TransactionTestCase):
    """L'historique doit rester lisible par v1: meme modele, memes roles. Un
    compte bascule sur v2 puis ramene sur v1 ne doit rien perdre.

    TransactionTestCase depuis le 2026-08-28: AGIR tourne desormais dans un
    thread du pool pour que le raisonnement puisse etre streame pendant qu'il
    travaille. Ce thread ne voit pas la transaction non validee d'un TestCase,
    donc l'historique y arrivait vide. La production n'a pas ce probleme,
    ATOMIC_REQUESTS valant False: chaque ecriture est validee immediatement et
    visible de tout thread."""

    def setUp(self):
        self.user = User.objects.create_user(username='persist', password='x')
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def _tour(self, message="bonjour", reponse=None):
        reponse = reponse or ReponseDire(ouverture="Salut.")
        with patch.object(self.Agent, '_boucle',
                          _boucle_simulee(_effet_muet, reponse)):
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

        def _boucle_qui_regarde(self_agent, user, message, registre):
            vus['historique'] = [
                p.content for m in self_agent._historique(user)
                for p in m.parts if hasattr(p, 'content')
            ]
            return ReponseDire(ouverture="Ok.")

        with patch.object(self.Agent, '_boucle', _boucle_qui_regarde):
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
    Ici on verifie qu'elle atteint VRAIMENT l'agent de la boucle: sans ce
    reglage, DeepSeek refuse tool_choice=required en mode thinking et la
    boucle echoue dix fois sur dix (mesure du 2026-08-24, transposee de DIRE
    a la boucle unique qui rend aussi une sortie structuree)."""

    def setUp(self):
        self.user = User.objects.create_user(username='reglages', password='x')
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def test_la_boucle_construit_son_agent_avec_le_reglage_qui_coupe_le_raisonnement(self):
        from services.agent_v2 import agent as module_agent
        from services.agent_v2.modeles import REGLAGES_BOUCLE_SANS_RAISONNEMENT

        vus = {}

        class AgentEspion:
            def __init__(self, *a, **kw):
                vus.update(kw)

        class Resultat:
            output = ReponseDire(ouverture="Salut.")

        with patch.object(module_agent, 'Agent', AgentEspion), \
             patch.object(module_agent, 'modele_agir', return_value=object()), \
             patch.object(self.Agent, '_executer_boucle', return_value=Resultat()):
            self.Agent().process_message(self.user, "bonjour")

        self.assertEqual(vus.get('model_settings'), REGLAGES_BOUCLE_SANS_RAISONNEMENT)

class ResilienceTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='resilience', password='x')
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def test_une_panne_de_boucle_apres_mutation_annonce_QUAND_MEME_les_faits(self):
        """Le pire cas du produit: on a ecrit dans le planning et la boucle
        tombe. Se taire laisserait l'utilisateur croire que rien n'a eu lieu,
        alors que son planning a change."""
        def _boucle(self_agent, user, message, registre):
            _effet_qui_cree(self_agent, user, message, registre)
            raise RuntimeError('502')

        with patch.object(self.Agent, '_boucle', _boucle):
            res = self.Agent().process_message(self.user, "ajoute maths")
        self.assertIn('Maths', res['response'])

    def test_un_budget_epuise_est_dit_a_l_utilisateur(self):
        def _boucle_sature(self_agent, user, message, registre):
            registre.budget_epuise = True
            return ReponseDire(ouverture="Bon.")

        with patch.object(self.Agent, '_boucle', _boucle_sature):
            res = self.Agent().process_message(self.user, "fais tout")
        texte = res['response'].lower()
        self.assertTrue('interrompu' in texte or 'arrêté' in texte, texte)
