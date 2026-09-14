"""
Gardes du code, round 4 (revues gardes, regressions et banc du 2026-09-14).

G1  « garde tous les jeudis » repondu a la question de portee supprimait la
    serie: le lecteur voyait « tous les » avant l'intention de garder.
G2  la garde d'heure dite sautait les titres courts ou vides de sens (Gym,
    Bac, cours) et les titres renommes par le modele.
R1  apres un refus d'heure dite, la garde armee retenait toute creation du
    meme jour, meme un autre element, sous la cle de la premiere question:
    la seconde demande disparaissait sans un mot.
B1  apres un oui vague a une question de portee, la demande se perdait; le
    tour suivant « Tous les jeudis » reposait la meme question.

Horloge du harnais: lundi 14 septembre 2026, 8 h (jour 0 = lundi).
"""
from datetime import timedelta

from django.test import SimpleTestCase, TransactionTestCase

from core.models import ConversationMessage, RecurringBlock, ScheduledBlock
from core.test_agent_v2_gardes import AUJOURDHUI, HarnaisGardes, puces
from services.agent_v2 import demandes as dem
from services.agent_v2 import outils as outils_v2
from services.agent_v2.registre import Registre

PORTEE = {'motif': 'portee_jour', 'cle': 'p',
          'options': [{'id': 'occurrence'}, {'id': 'serie'}, {'id': 'annuler'}]}

GARDER = (
    'garde tous les jeudis', 'laisse-le toujours', 'je veux le garder chaque semaine',
    'garde-le définitivement', 'Garde tous les jeudis', 'conserve la série',
    'ne touche pas à tous les jeudis', 'ne change rien, tous les jeudis restent',
    'non, garde chaque jeudi', 'finalement, garde tous les jeudis',
    'annule, laisse tous les jeudis', 'laisse ce jeudi', 'garde celui-là',
)


class G1LectureDePorteeTests(SimpleTestCase):

    def test_garder_ou_annuler_ne_donne_jamais_la_serie(self):
        for brut in GARDER:
            with self.subTest(brut=brut):
                self.assertIn(dem.option_choisie(brut, PORTEE), (None, 'annuler'))

    def test_la_serie_ne_se_lit_que_sur_la_puce(self):
        # Round 6 (D1): remplace « la serie reste lisible ». La lecture libre
        # de la portee est retiree; la puce exacte seule donne la serie.
        cas = {'tous les jeudis': None, 'Tous les jeudis': None,
               'la série': None, 'oui, tous les jeudis': None,
               'supprime tous les jeudis': None, 'efface-le définitivement': None,
               'enlève-le chaque semaine': None,
               'Tous les jeudis (supprimer la série).': None,
               'toujours': None, 'définitivement': None,
               "j'aimerais que tous les jeudis soient libres": None,
               'Non, ne change rien.': 'annuler', 'non': 'annuler',
               'juste celui-là': None}
        for brut, attendu in cas.items():
            with self.subTest(brut=brut):
                self.assertEqual(dem.option_choisie(brut, PORTEE), attendu)
        avec_puces = puces(dict(PORTEE, cible={'jour': 3, 'date': '2026-09-17'}))
        for brut in ('tous les jeudis', 'Tous les jeudis', 'Tous les jeudis (supprimer la série).'):
            with self.subTest(brut=brut, puces=True):
                self.assertEqual(dem.option_choisie(brut, avec_puces), 'serie')


class G1GarderNeSupprimeRienTests(HarnaisGardes, TransactionTestCase):

    def test_garder_tous_les_jeudis_ne_supprime_pas_la_serie(self):
        demande = puces(self.demande_portee())
        for i, brut in enumerate(GARDER):
            with self.subTest(brut=brut):
                self.attendre([demande], brut)
                registre = Registre()
                outils_v2.appliquer_choix_en_attente(self.user, registre, brut, f'g:{i}')
                # Round 6: la decision « annulee » est consignee (succes, pas
                # une mutation); aucune mutation ne doit reussir.
                self.assertFalse(any(a.succes and a.est_mutation for a in registre.actions))
                self.assertActif(self.q)
                # Le modele qui tenterait la suppression est retenu aussi.
                _, tools = self.outils(brut, registre=registre, tache=f'g:{i}')
                self.appeler(tools, 'delete_block', block_id=self.q.id)
                self.assertFalse(any(a.succes and a.est_mutation for a in registre.actions))
                self.assertActif(self.q)
        # Temoin: la reponse nue passe, la lecture de l'attente fonctionne.
        self.attendre([demande], 'Tous les jeudis')
        outils_v2.appliquer_choix_en_attente(self.user, Registre(), 'Tous les jeudis', 'g:t')
        self.assertActif(self.q, False)


class G2HeureDiteTitreCourtTests(HarnaisGardes, TransactionTestCase):

    def _refuse(self, registre):
        action = registre.actions[-1]
        self.assertFalse(action.succes, action.message)
        d = action.donnees
        self.assertTrue(d.get('heure_dite') == '15:00'
                        or (d.get('demande') or {}).get('motif') == 'heure_refusee', d)

    def test_titre_court_schedule_task_at(self):
        for i, (brut, titre) in enumerate((('ajoute gym jeudi à 15 h', 'Gym'),
                                           ('mets Bac jeudi à 15 h', 'Bac'),
                                           ('ajoute mon cours jeudi à 15 h', 'Cours'),
                                           # Round 6 (D3): « Entraînement » pour « gym »
                                           # est l'ecart accepte, verrouille dans
                                           # test_agent_v2_gardes_r6.D3HeureDiteTests.
                                           ('mon rendez-vous jeudi à 15 h', 'Rendez-vous'))):
            with self.subTest(brut=brut, titre=titre):
                self.message_courant(brut)
                registre, tools = self.outils(brut, tache=f'c:{i}')
                self.appeler(tools, 'schedule_task_at', title=titre, date='2026-09-17',
                             start_time='13:00', end_time='14:00')
                self._refuse(registre)
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 0)

    def test_titre_court_create_block(self):
        brut = 'ajoute gym le jeudi à 15 h'
        self.message_courant(brut)
        registre, tools = self.outils(brut)
        self.appeler(tools, 'create_block', title='Gym', block_type='sport',
                     days=['jeudi'], start_time='13:00', end_time='14:00')
        self._refuse(registre)
        self.assertFalse(RecurringBlock.objects.filter(user=self.user, title='Gym').exists())

    def test_l_heure_dite_passe_et_une_duree_ne_compte_pas(self):
        brut = 'ajoute gym jeudi à 15 h'
        self.message_courant(brut)
        registre, tools = self.outils(brut)
        self.appeler(tools, 'schedule_task_at', title='Gym', date='2026-09-17',
                     start_time='15:00', end_time='16:00')
        self.assertTrue(registre.actions[-1].succes, registre.actions[-1].message)
        for i, brut in enumerate(('ajoute 2 h de gym jeudi', 'ajoute gym jeudi pendant 1 h',
                                  'place gym jeudi avant 15 h')):
            with self.subTest(brut=brut):
                self.message_courant(brut)
                registre, tools = self.outils(brut, tache=f'd:{i}')
                self.appeler(tools, 'schedule_task_at', title='Gym', date='2026-09-17',
                             start_time=f'{8 + i}:00', end_time=f'{9 + i}:00')
                self.assertTrue(registre.actions[-1].succes, registre.actions[-1].message)


class R1GardeArmeeCibleeTests(HarnaisGardes, TransactionTestCase):

    def test_un_autre_element_du_meme_jour_passe(self):
        demain = AUJOURDHUI + timedelta(days=1)
        self.bloc('Travail', demain.weekday(), '13:00', '17:00', block_type='work',
                  flexibility='fixed')
        brut = 'Rdv dentiste demain 14h a 15h, et place aussi une heure de lecture demain'
        self.message_courant(brut)
        registre, tools = self.outils(brut)
        self.appeler(tools, 'schedule_task_at', title='Dentiste', date=demain.isoformat(),
                     start_time='14:00', end_time='15:00')
        refus = registre.actions[-1]
        self.assertFalse(refus.succes)
        self.assertEqual(refus.donnees['demande']['motif'], 'heure_refusee')

        self.appeler(tools, 'schedule_task_at', title='Lecture', date=demain.isoformat(),
                     start_time='19:00', end_time='20:00')
        lecture = registre.actions[-1]
        self.assertTrue(lecture.succes, lecture.message)
        self.assertTrue(ScheduledBlock.objects.filter(user=self.user, task__title='Lecture').exists())

        # Le meme element deplace a une autre heure reste retenu.
        self.appeler(tools, 'schedule_task_at', title='RDV dentiste', date=demain.isoformat(),
                     start_time='18:00', end_time='19:00')
        self.assertFalse(registre.actions[-1].succes)
        self.assertEqual(registre.actions[-1].donnees['demande']['cle'],
                         refus.donnees['demande']['cle'])


class B1QuestionReposeeParLeCodeTests(HarnaisGardes, TransactionTestCase):

    def test_oui_vague_puis_tous_les_jeudis_supprime_la_serie(self):
        from services.agent_v2.agent import PlannerAgentV2

        demande = puces(self.demande_portee())
        vague = 'Oui, supprime ces trois blocs.'
        u2 = self.attendre([demande], vague)

        # Tour 2: aucune reponse claire. Le CODE repose la meme demande.
        registre = Registre()
        sorties = outils_v2.appliquer_choix_en_attente(self.user, registre, vague, 'u:2')
        self.assertIn('SANS REPONSE CLAIRE', sorties[0]['resume'])
        self.assertEqual(len(registre.actions), 1)
        reposee = registre.actions[0]
        self.assertFalse(reposee.succes)
        self.assertEqual(reposee.message, outils_v2.MESSAGE_RETENUE)
        self.assertEqual(reposee.donnees['demande']['cle'], demande['cle'])
        self.assertEqual(reposee.donnees['demande']['motif'], 'portee_jour')
        self.assertNotIn('chips', reposee.donnees['demande'])
        self.assertActif(self.q)

        # Le modele qui retente la suppression est retenu sur la meme cle.
        _, tools = self.outils(vague, registre=registre, tache='u:2')
        self.appeler(tools, 'delete_block', block_id=self.q.id)
        self.assertFalse(registre.actions[-1].succes)
        self.assertEqual(registre.actions[-1].donnees['demande']['cle'], demande['cle'])
        self.assertActif(self.q)

        # La question du tour, telle que l'agent la choisit et la persiste.
        demandes = [a.donnees['demande'] for a in registre.actions if a.donnees.get('demande')]
        question = PlannerAgentV2._question_des_demandes(demandes)
        self.assertEqual([d['cle'] for d in question['demandes']], [demande['cle']])
        self.assertTrue(question['chips'])
        ConversationMessage.objects.create(
            user=self.user, role='assistant', content=question['question'],
            metadata={'en_reponse_a': u2.pk, 'demandes': question['demandes']})

        # Tour 3: une reponse explicite tranche.
        self.message_courant('Tous les jeudis')
        registre = Registre()
        sorties = outils_v2.appliquer_choix_en_attente(self.user, registre, 'Tous les jeudis', 'u:3')
        self.assertEqual(sorties[0]['option'], 'serie')
        self.assertTrue(registre.actions[-1].succes)
        self.assertActif(self.q, False)

    def test_une_demande_alteree_n_est_pas_reposee(self):
        autre = self.bloc('Chimie générale', 3, '13:00', '15:00')
        demande = puces(self.demande_portee())
        demande['options'][1] = {**demande['options'][1],
                                 'effet': {'outil': 'delete_block',
                                           'parametres': {'block_id': autre.id}}}
        self.attendre([demande], 'Tous les jeudis (supprimer la série).')
        registre = Registre()
        outils_v2.appliquer_choix_en_attente(self.user, registre,
                                             'Tous les jeudis (supprimer la série).', 'u:2')
        self.assertEqual(registre.actions, [])
        self.assertActif(autre)
        self.assertActif(self.q)
