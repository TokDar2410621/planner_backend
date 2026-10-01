"""La consultation aboutit toujours a une liste ou a une vraie question.

Defaut A, mesure en production le 2026-09-29 sur des comptes jetables: sur
« Mon planning d'aujourd'hui stp » en PREMIER message, 2 tours sur 3 rendaient
« Je n'ai pas compris. Tu veux ajouter, deplacer ou voir quelque chose ? ».
Chaine reproduite localement: la boucle n'appelait aucun outil (le prompt lui
interdisait de relire le jour courant), ecrivait le planning en prose en
l'annoncant par « Voici ... », la regle (e) de composer retirait cette phrase
faute de liste affichee, et le code servait son repli.

La suite d'evals FR ne pouvait pas voir ce defaut: ses tests appellent
eux-memes get_today_schedule. Ici la boucle est SIMULEE et n'appelle AUCUN
outil, comme en prod. Aucun appel reseau: le juge est scripte.

TransactionTestCase, comme les scenarios FR: le filet execute la lecture par
un asyncio.run, et une transaction de test non committee verrouille SQLite des
qu'un autre contexte lit la table.
"""
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TransactionTestCase
from django.utils import timezone

from core.models import RecurringBlock
from core.test_agent_v2_jugement import juger_script
from services.agent_v2.redaction import ReponseDire

PROSE_QUI_ANNONCE = ("Voici ton planning d'aujourd'hui : Conception "
                     "d'applications de 8 h à 11 h, puis L'entreprise et ses "
                     "systèmes de 13 h à 16 h.")


def _boucle_muette(reponse):
    """Une boucle qui repond sans appeler le moindre outil (le cas de prod)."""
    def _boucle(self_agent, user, message, registre):
        return reponse
    return _boucle


class ConsultationTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='consultation', password='x')
        aujourdhui = timezone.localdate().weekday()
        RecurringBlock.objects.create(
            user=self.user, title="Conception d’applications", block_type='course',
            day_of_week=aujourdhui, start_time='08:00', end_time='11:00')
        RecurringBlock.objects.create(
            user=self.user, title="L’entreprise et ses systèmes", block_type='course',
            day_of_week=aujourdhui, start_time='13:00', end_time='16:00')
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def _tour(self, message, juge):
        with patch('services.agent_v2.jugement.juger', juge), \
                patch.object(self.Agent, '_boucle',
                             _boucle_muette(ReponseDire(ouverture=PROSE_QUI_ANNONCE))):
            return self.Agent().process_message(self.user, message)

    def test_le_code_lit_la_journee_quand_la_boucle_ne_l_a_pas_fait(self):
        message = "Mon planning d'aujourd'hui stp"
        res = self._tour(message, juger_script(
            {message: {"consultation": ("journee", 0.95)},
             # Le tour passe par les autres jugements du chemin normal; les
             # laisser indisponibles est le comportement prudent attendu.
             }))
        self.assertIn('Conception', res['response'])
        self.assertIn("L’entreprise", res['response'])
        self.assertNotIn("Je n'ai pas compris", res['response'])
        # La prose du modele ne raconte pas la liste que le code affiche.
        self.assertNotIn("Voici ton planning", res['response'])

    def test_sans_decision_du_juge_on_demande_quoi_voir(self):
        """Juge indisponible: pas de liste devinee, mais pas « je n'ai pas
        compris » non plus, puisque la boucle annoncait bien une liste."""
        res = self._tour("Mon planning d'aujourd'hui stp", juger_script({}))
        self.assertIn('ta journée', res['response'])
        self.assertNotIn("Je n'ai pas compris", res['response'])
        self.assertEqual([p['label'] for p in res['quick_replies']],
                         ['Ma journée', 'Ma semaine', 'Mes tâches'])

    def test_le_juge_qui_dit_aucune_ne_declenche_aucune_lecture(self):
        """« aucune »: la personne ne demandait pas a voir son horaire. Le
        code n'affiche rien qu'elle n'a pas demande."""
        message = "Merci beaucoup"
        res = self._tour(message, juger_script(
            {message: {"consultation": ("aucune", 0.95)}}))
        self.assertNotIn('Conception', res['response'])

    def test_une_confiance_faible_ne_declenche_aucune_lecture(self):
        message = "Mon planning d'aujourd'hui stp"
        res = self._tour(message, juger_script(
            {message: {"consultation": ("journee", 0.7)}}))
        self.assertNotIn('Conception', res['response'])


class SalutationSansLectureTests(TransactionTestCase):
    """Rien ne s'affiche sans demande, et la regle n'enumere aucun mot.

    Deux tentatives ratees avant celle-ci, toutes deux par enumeration: des
    exemples entre guillemets (« hey », « salut »), puis une liste de formules
    (salutation, remerciement, acquiescement). « Je suis la » n'entrait dans
    aucune, et la journee s'affichait. Une liste de mots ne converge jamais:
    la regle pose le PRINCIPE, et le juge porte sur ce que la personne attend.
    """

    def test_la_description_n_appelle_la_lecture_que_sur_une_demande(self):
        from services.agent_v2 import outils as outils_v2
        d = outils_v2.DESCRIPTIONS_V2['get_today_schedule']
        self.assertIn('QUE si la personne DEMANDE a voir', d)
        self.assertIn('quels que soient ses mots', d)

    def test_le_prompt_dit_rien_sans_demande(self):
        from services.agent_v2.prompts import REGLES_AGIR
        self.assertIn("Rien ne s'affiche sans demande", REGLES_AGIR)
        self.assertIn('aucune lecture', REGLES_AGIR.lower())
        self.assertIn('quels que soient ses mots', REGLES_AGIR)

    def test_aucune_regle_n_enumere_des_formules_de_politesse(self):
        """La garde contre la faute commise deux fois."""
        from services.agent_v2 import jugement, outils as outils_v2
        from services.agent_v2.prompts import PROSE_BOUCLE, REGLES_AGIR
        textes = {
            'description': outils_v2.DESCRIPTIONS_V2['get_today_schedule'],
            'regles': REGLES_AGIR,
            'prose': PROSE_BOUCLE,
            'juge': jugement.q_interaction_sociale()['instructions'],
        }
        for ou, texte in textes.items():
            with self.subTest(ou=ou):
                plat = texte.lower()
                # Une enumeration se reconnait a ses exemples accoles.
                self.assertNotIn('saluer, remercier', plat)
                self.assertNotIn('salutation, un remerciement', plat)
                self.assertNotIn('« hey »', plat)
                self.assertNotIn('« salut »', plat)


class VoieRapideSocialeTests(TransactionTestCase):
    """La voie rapide repond a une salutation SANS la boucle.

    Question de Darius le 2026-09-30: « pourquoi il reflechit encore sur
    hey ? ». Mesure directe du juge en production: valeur=True, confiance 0.94
    a 0.96 sur « Hey », « Salut », « Merci ». Le juge disait oui, mais le code
    comparait a la CHAINE « oui » alors qu'une reponse noul est un BOOLEEN
    (jugement.py l.507): la voie etait morte, et chaque salutation payait un
    tour complet de la boucle.

    Les tests ne pouvaient pas le voir: le faux juge rendait la chaine
    scriptee. Il respecte desormais le contrat du vrai.
    """

    def setUp(self):
        self.user = User.objects.create_user(username='voie-rapide', password='x')
        aujourdhui = timezone.localdate().weekday()
        RecurringBlock.objects.create(
            user=self.user, title="Conception d’applications", block_type='course',
            day_of_week=aujourdhui, start_time='08:00', end_time='11:00')
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def test_une_salutation_prend_la_voie_rapide(self):
        """La boucle ne tourne pas: on la remplace par une bombe."""
        def _boucle_interdite(self_agent, user, message, registre):
            raise AssertionError('la boucle a tourne sur une salutation')

        with patch('services.agent_v2.jugement.juger',
                   juger_script({'Hey': {'sociale': ('oui', 0.94)}})), \
                patch.object(self.Agent, '_boucle', _boucle_interdite), \
                patch.object(self.Agent, '_reponse_rapide',
                             lambda s, u, m: 'Salut !'):
            res = self.Agent().process_message(self.user, 'Hey')
        self.assertEqual(res['response'], 'Salut !')
        self.assertNotIn('Conception', res['response'])

    def test_une_vraie_demande_ne_prend_pas_la_voie_rapide(self):
        message = "Montre-moi ma journée"
        with patch('services.agent_v2.jugement.juger',
                   juger_script({message: {'sociale': ('non', 0.94)}})), \
                patch.object(self.Agent, '_boucle',
                             _boucle_muette(ReponseDire(ouverture='Voilà.'))):
            res = self.Agent().process_message(self.user, message)
        self.assertTrue(res['response'])

    def test_le_faux_juge_rend_un_booleen_comme_le_vrai(self):
        """Le piege qui a laisse passer le bug: un double trop permissif."""
        from services.agent_v2 import jugement as j
        faux = juger_script({'Hey': {'sociale': ('oui', 0.94)}})
        rep = faux('Hey', {'sociale': j.q_interaction_sociale()})['sociale']
        self.assertIs(rep['valeur'], True)
        choix = juger_script({'m': {'portee': ('serie', 0.95)}})
        rep2 = choix('m', {'portee': j.q_portee_changement()})['portee']
        self.assertEqual(rep2['valeur'], 'serie')


class PlanningDuJourEffectifTests(TransactionTestCase):
    """Le PLANNING AUJOURD'HUI du prompt dit la verite.

    Mesure le 2026-10-01 sur le compte de Darius: il avait saute l'occurrence
    de « Conception d'applications » du jour, et le prompt l'affichait quand
    meme. Or le prompt affirme juste au-dessus que ce bloc EST l'etat effectif:
    le modele raisonnait donc sur un creneau occupe qui etait libre.
    """

    def setUp(self):
        self.user = User.objects.create_user(username='effectif', password='x')
        self.jour = timezone.localdate()
        self.cours = RecurringBlock.objects.create(
            user=self.user, title='Conception', block_type='course',
            day_of_week=self.jour.weekday(), start_time='08:00', end_time='11:00')
        RecurringBlock.objects.create(
            user=self.user, title='Entreprise', block_type='course',
            day_of_week=self.jour.weekday(), start_time='13:00', end_time='16:00')

    def _lignes(self):
        from services.agent.context_builder import build_context
        return build_context(self.user)['today']['blocks']

    def test_une_occurrence_sautee_sort_du_planning_du_jour(self):
        from core.models import RecurringBlockException
        self.assertTrue(any('Conception' in l for l in self._lignes()))
        RecurringBlockException.objects.create(
            user=self.user, recurring_block=self.cours, date=self.jour)
        lignes = self._lignes()
        self.assertFalse(any('Conception' in l for l in lignes))
        # Le reste de la journee ne bouge pas.
        self.assertTrue(any('Entreprise' in l for l in lignes))

    def test_une_occurrence_sautee_un_autre_jour_ne_change_rien(self):
        from datetime import timedelta

        from core.models import RecurringBlockException
        RecurringBlockException.objects.create(
            user=self.user, recurring_block=self.cours,
            date=self.jour + timedelta(days=7))
        self.assertTrue(any('Conception' in l for l in self._lignes()))

    def test_le_prompt_ne_montre_pas_un_cours_annule(self):
        from core.models import RecurringBlockException
        from services.agent_v2.prompts import prompt_agir
        RecurringBlockException.objects.create(
            user=self.user, recurring_block=self.cours, date=self.jour)
        prompt = prompt_agir(self.user)
        debut = prompt.index("PLANNING AUJOURD'HUI")
        section = prompt[debut:prompt.index('SEMAINE TYPE', debut)]
        self.assertNotIn('Conception', section)
        self.assertIn('Entreprise', section)


class ContexteDuJugeTests(TransactionTestCase):
    """Le juge decide sur le message ET ce qui le precede.

    Darius, le 2026-10-01: « le probleme n'est pas que "je suis la" ne
    declenche plus rien, mais que "je suis la" ne declenche plus rien SANS
    CONTEXTE. Il peut poser une question, puis apres on dit "je suis la", et
    la ca doit declencher. »

    `juger(etat, questions)` accepte un dict de contexte depuis le debut, mais
    tous les appels lui passaient la chaine du message. Mesure en production
    apres correction: « Je suis la » seul est juge social; apres « Tu veux voir
    ta journee ? » il ne l'est plus.
    """

    def setUp(self):
        self.user = User.objects.create_user(username='contexte', password='x')
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def test_le_juge_recoit_ce_que_l_agent_vient_de_dire(self):
        from core.models import ConversationMessage
        ConversationMessage.objects.create(
            user=self.user, role='assistant', content='Tu veux voir ta journée ?')
        vus = {}

        def _faux(etat, questions):
            vus['etat'] = etat
            return {q: {'valeur': None, 'confiance': 0.0, 'probabilites': None,
                        'statut': 'indisponible'} for q in questions}

        with patch('services.agent_v2.jugement.juger', _faux):
            self.Agent()._voie_rapide_sociale(self.user, 'Je suis la', None)
        self.assertIsInstance(vus.get('etat'), dict)
        self.assertEqual(vus['etat']['message'], 'Je suis la')
        self.assertIn('Tu veux voir ta journée ?', vus['etat']['agent_a_dit'])

    def test_sans_historique_le_contexte_est_vide_et_ne_casse_rien(self):
        vus = {}

        def _faux(etat, questions):
            vus['etat'] = etat
            return {q: {'valeur': None, 'confiance': 0.0, 'probabilites': None,
                        'statut': 'indisponible'} for q in questions}

        with patch('services.agent_v2.jugement.juger', _faux):
            rapide = self.Agent()._voie_rapide_sociale(self.user, 'Bonjour', None)
        self.assertFalse(rapide)  # juge indisponible: comportement prudent
        self.assertEqual(vus['etat']['agent_a_dit'], '')

    def test_le_contexte_est_borne(self):
        """Le juge a besoin du sens, pas de la liste affichee."""
        from core.models import ConversationMessage
        ConversationMessage.objects.create(
            user=self.user, role='assistant', content='x' * 2000)
        self.assertEqual(len(self.Agent()._dernier_mot_de_l_agent(self.user)), 400)

    def test_la_question_du_juge_nomme_le_champ_de_contexte(self):
        from services.agent_v2 import jugement as j
        instructions = j.q_interaction_sociale()['instructions']
        self.assertIn('agent_a_dit', instructions)
        self.assertIn('repond a ce que l\'assistant vient de dire', instructions)
