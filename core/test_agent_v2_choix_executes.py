"""
Choix de l'utilisateur executes par le code (lot 1d-effets).

Quand l'utilisateur touche une puce (« Tous les jeudis »), le code execute
l'effet de l'option AVANT que le modele ne tourne, par le meme chemin que ses
appels: registre, signal, idempotence. Le modele ne peut ni s'attribuer ces
actions (elles sont au registre, rendues par le code) ni les refaire (« Deja
fait par le code ce tour. »).

TransactionTestCase pour la meme raison que core/test_agent_v2_gardes.py.
"""
from datetime import date

from django.test import TransactionTestCase
from django.utils import timezone

from core.models import RecurringBlock, RecurringBlockException, Task
from core.test_agent_v2_gardes import HarnaisGardes, puces
from services.agent_v2 import outils as outils_v2
from services.agent_v2.registre import Registre


class ChoixExecutesTests(HarnaisGardes, TransactionTestCase):

    def appliquer(self, brut, registre=None, tache='u:2', signaler=None):
        registre = registre if registre is not None else Registre()
        sorties = outils_v2.appliquer_choix_en_attente(
            self.user, registre, brut, tache, signaler=signaler)
        return registre, sorties

    def test_sans_demande_en_attente_rien(self):
        self.message_courant('Tous les jeudis (supprimer la série).')
        registre, sorties = self.appliquer('Tous les jeudis (supprimer la série).')
        self.assertEqual(sorties, [])
        self.assertEqual(registre.actions, [])
        self.assertActif(self.q)

    def test_occurrence_executee_par_le_code(self):
        demande = puces(self.demande_portee())
        brut = "Seulement ce jeudi 17 sept. (sauter l'occurrence)."
        self.attendre([demande], brut)
        signaux = []
        registre, sorties = self.appliquer(brut, signaler=signaux.append)

        self.assertEqual(len(sorties), 1)
        sortie = sorties[0]
        self.assertEqual(sortie['option'], 'occurrence')
        self.assertEqual(sortie['cle'], demande['cle'])
        self.assertEqual(sortie['motif'], 'portee_jour')
        self.assertIsNotNone(sortie['action_id'])
        self.assertIn('FAIT PAR LE CODE', sortie['resume'])
        self.assertTrue(RecurringBlockException.objects.filter(
            recurring_block=self.q, date=date(2026, 9, 17)).exists())
        action = registre.par_id(sortie['action_id'])
        self.assertEqual(action.outil, 'skip_block_occurrence')
        self.assertTrue(action.succes)
        self.assertTrue(action.donnees['par_le_code'])
        self.assertEqual(action.donnees['cle_demande'], demande['cle'])
        self.assertEqual(signaux, [action])
        self.assertActif(self.q)

        # Le modele refait la meme chose: rien n'est execute ni consigne.
        _, tools = self.outils(brut, registre=registre, tache='u:2')
        retour = self.appeler(tools, 'skip_block_occurrence', date='2026-09-17',
                              title='Quart au dépanneur', block_type='work')
        self.assertEqual(retour, 'Deja fait par le code ce tour.')
        self.assertEqual(len(registre.actions), 1)

        # Et il ne peut pas supprimer la serie a la place: l'utilisateur a
        # choisi l'occurrence.
        retour = self.appeler(tools, 'delete_block', block_id=self.q.id)
        self.assertEqual(retour, outils_v2.MESSAGE_DEJA_TRANCHE)
        self.assertEqual(len(registre.actions), 1)
        self.assertActif(self.q)

    def test_serie_executee_par_le_code(self):
        demande = puces(self.demande_portee())
        brut = 'Tous les jeudis (supprimer la série).'
        self.attendre([demande], brut)
        registre, sorties = self.appliquer(brut)
        self.assertEqual(sorties[0]['option'], 'serie')
        self.assertActif(self.q, False)
        action = registre.par_id(sorties[0]['action_id'])
        self.assertEqual(action.outil, 'delete_block')
        self.assertTrue(action.succes)
        self.assertTrue(action.donnees['par_le_code'])

        _, tools = self.outils(brut, registre=registre, tache='u:2')
        retour = self.appeler(tools, 'delete_block', block_id=self.q.id)
        self.assertEqual(retour, 'Deja fait par le code ce tour.')
        self.assertEqual(len(registre.actions), 1)

    def test_annuler_ne_fait_rien(self):
        demande = puces(self.demande_portee())
        self.attendre([demande], 'Non, ne change rien.')
        registre, sorties = self.appliquer('Non, ne change rien.')
        self.assertEqual(len(sorties), 1)
        self.assertEqual(sorties[0]['option'], 'annuler')
        self.assertIsNone(sorties[0]['action_id'])
        self.assertIn('REFUSE', sorties[0]['resume'])
        self.assertEqual(registre.actions, [])
        self.assertActif(self.q)

    def test_sans_reponse_claire_n_agit_pas(self):
        demande = puces(self.demande_portee())
        self.attendre([demande], 'oui')
        registre, sorties = self.appliquer('oui')
        self.assertIsNone(sorties[0]['option'])
        self.assertIn('SANS REPONSE CLAIRE', sorties[0]['resume'])
        self.assertEqual(registre.actions, [])
        self.assertActif(self.q)

    def test_confirmer_destructif_execute_la_tache_retenue(self):
        tache = Task.objects.create(user=self.user, title='Rapport de labo')
        demande = puces(self.premier_tour('supprime la tâche rapport de labo', 'delete_task',
                                          task_id=tache.id, confirm=False).donnees['demande'])
        self.attendre([demande], 'Oui, je confirme.')
        registre, sorties = self.appliquer('Oui, je confirme.')
        self.assertEqual(sorties[0]['option'], 'confirmer')
        self.assertFalse(Task.objects.filter(id=tache.id).exists())
        action = registre.par_id(sorties[0]['action_id'])
        self.assertEqual(action.outil, 'delete_task')
        self.assertTrue(action.succes)
        self.assertTrue(action.donnees['par_le_code'])

    def test_plan_inchange_applique(self):
        self.bloc('Sport', 1, '06:00', '07:00', block_type='sport', flexibility='flexible')
        demande = puces(self.premier_tour('optimise ma semaine', 'optimize_week',
                                          apply=True).donnees['demande'])
        self.attendre([demande], 'Oui, applique le plan.')
        registre, sorties = self.appliquer('Oui, applique le plan.')
        action = registre.par_id(sorties[0]['action_id'])
        self.assertTrue(action.succes)
        self.assertTrue(action.donnees['applied'])
        self.assertTrue(action.donnees['par_le_code'])

    def test_plan_change_redemande(self):
        self.bloc('Sport', 1, '06:00', '07:00', block_type='sport', flexibility='flexible')
        demande = puces(self.premier_tour('optimise ma semaine', 'optimize_week',
                                          apply=True).donnees['demande'])
        demande['parametres'] = {**demande['parametres'], 'plan_hash': 'x'}
        self.attendre([demande], 'Oui, applique le plan.')
        registre, sorties = self.appliquer('Oui, applique le plan.')

        plans = [a for a in registre.actions if a.outil == 'optimize_week']
        self.assertEqual(len(plans), 1)
        self.assertFalse(plans[0].succes)
        nouvelle = plans[0].donnees['demande']
        self.assertEqual(nouvelle['motif'], 'optimisation')
        self.assertNotEqual(nouvelle['parametres']['plan_hash'], 'x')
        self.assertIn('PLAN CHANGE', sorties[0]['resume'])

    def test_effet_altere_n_execute_rien(self):
        """Une demande dont l'effet ne vise plus la cible de sa cle est
        rejetee: la cle lie la question a ce qu'elle nomme."""
        autre = self.bloc('Chimie générale', 3, '13:00', '15:00')
        demande = puces(self.demande_portee())
        demande['options'][1] = {**demande['options'][1],
                                 'effet': {'outil': 'delete_block',
                                           'parametres': {'block_id': autre.id}}}
        self.attendre([demande], 'Tous les jeudis (supprimer la série).')
        registre, sorties = self.appliquer('Tous les jeudis (supprimer la série).')
        self.assertIsNone(sorties[0]['action_id'])
        self.assertEqual(registre.actions, [])
        self.assertActif(autre)
        self.assertActif(self.q)

    def test_choix_de_creneau_resume_pour_le_modele(self):
        demande = {
            'type': 'choix', 'motif': 'heure_refusee', 'cle': 'h1', 'outil': 'schedule_task_at',
            'parametres': {}, 'cible': {'titre': 'Dentiste', 'date': '2026-09-17'},
            'options': [
                {'id': 'creneau_1', 'effet': None,
                 'cible': {'titre': 'Dentiste', 'date': '2026-09-17', 'debut': '11:50', 'fin': '12:50'}},
                {'id': 'autre_jour', 'effet': None, 'cible': {'titre': 'Dentiste'}},
            ],
            'chips': [
                {'label': '11 h 50 à 12 h 50', 'value': 'Va pour 11 h 50 à 12 h 50 jeu. 17 sept.',
                 'option': 'creneau_1'},
                {'label': 'Un autre jour', 'value': 'Je préfère un autre jour.', 'option': 'autre_jour'},
            ],
            'emise_le': timezone.now().isoformat(),
        }
        self.attendre([demande], 'Va pour 11 h 50 à 12 h 50 jeu. 17 sept.')
        registre, sorties = self.appliquer('Va pour 11 h 50 à 12 h 50 jeu. 17 sept.')
        self.assertEqual(sorties[0]['option'], 'creneau_1')
        self.assertIsNone(sorties[0]['action_id'])
        self.assertIn('11:50', sorties[0]['resume'])
        self.assertEqual(registre.actions, [])

    def test_resume_sans_accents_pour_le_modele(self):
        demande = puces(self.demande_portee())
        self.attendre([demande], 'Tous les jeudis (supprimer la série).')
        _, sorties = self.appliquer('Tous les jeudis (supprimer la série).')
        self.assertIn('Quart au depanneur', sorties[0]['resume'])
        self.assertEqual(RecurringBlock.objects.filter(user=self.user, active=True).count(), 0)
