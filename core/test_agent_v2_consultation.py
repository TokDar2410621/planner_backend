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
    """Une salutation ne declenche aucune lecture, meme avec un historique.

    Mesure en production le 2026-09-30, apres le lot 2: sur un compte neuf,
    « Hey » est propre 6 fois sur 6; apres un tour qui a affiche le planning,
    « Hey » le redeballe 3 fois sur 3. Le contrat de prose l interdisait deja,
    mais trop tard: la liste est rendue par le code des qu une lecture reussit,
    donc l interdiction doit vivre la ou le modele DECIDE d appeler l outil.

    Ces tests verrouillent le contrat (les descriptions et le prompt), seul
    levier sur un choix d outil: aucun test ne peut forcer la decision d un
    modele. La mesure vit au banc de production.
    """

    def test_la_description_de_la_lecture_exclut_les_salutations(self):
        from services.agent_v2 import outils as outils_v2
        d = outils_v2.DESCRIPTIONS_V2['get_today_schedule']
        self.assertIn("salutation", d)
        self.assertIn("ne DEMANDE rien a voir", d)
        self.assertIn("AUCUNE lecture", d)
        # La notion suffit: pas d'exemples de mots, qui feraient decider le
        # modele sur un vocabulaire au lieu du sens.
        for mot in ('« hey »', '« salut »', '« merci »'):
            self.assertNotIn(mot, d)

    def test_le_prompt_dit_la_meme_regle(self):
        from services.agent_v2.prompts import REGLES_AGIR
        self.assertIn("ne DEMANDE rien a voir", REGLES_AGIR)
        self.assertIn("aucune lecture", REGLES_AGIR)

    def test_la_regle_vit_aussi_la_ou_le_modele_choisit_l_outil(self):
        """Le contrat de prose seul ne suffit pas: il arrive apres l appel."""
        from services.agent_v2 import outils as outils_v2
        from services.agent_v2.prompts import PROSE_BOUCLE
        self.assertIn('salut', PROSE_BOUCLE.lower())
        self.assertIn('salutation', outils_v2.DESCRIPTIONS_V2['get_today_schedule'])


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
