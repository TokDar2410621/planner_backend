"""
Round 9, fixeur f9-gardes. Chaque classe a ete ecrite AVANT son correctif et
vue en echec.

K1  un effet destructif stocke relit sa cible avant de s'executer. Si le bloc,
    la tache ou l'evenement a change entre la question et la puce (autre
    appareil, MCP, application web), rien ne s'execute, la demande tombe et
    une ligne du code le dit.
K3  « deplace mon gym a 7 h » sur un bloc de 19 h: seule la lecture egale a
    l'heure actuelle nomme le bloc, 7 h reste une heure dite.
K4  un mot de moment de la journee dans la proposition (« jeudi soir a 6 h »,
    « a 6 h le soir », « tous les matins a 7 h ») choisit la lecture stricte;
    « 2 h de l'apres-midi » n'est pas une duree.

Horloge du harnais: lundi 14 septembre 2026, 8 h (jour 0 = lundi).
"""
from datetime import date, time

from django.test import SimpleTestCase, TransactionTestCase

from core.models import RecurringBlock, RecurringBlockException, ScheduledBlock, Task
from core.test_agent_v2_gardes import HarnaisGardes, puces
from services.agent_v2 import demandes as dem
from services.agent_v2 import outils as outils_v2
from services.agent_v2.registre import Registre

JEUDI = '2026-09-17'
OUI = 'Oui, je confirme.'
SERIE = 'Tous les jeudis (supprimer la série).'
OCCURRENCE = "Seulement ce jeudi 17 sept. (sauter l'occurrence)."


# ── K1: la cible relue avant tout effet destructif ──────────────────────────


class K1CibleChangeeTests(HarnaisGardes, TransactionTestCase):

    def appliquer(self, brut, tache='k1:2'):
        registre = Registre()
        sorties = outils_v2.appliquer_choix_en_attente(self.user, registre, brut, tache)
        return registre, sorties

    def assertRienExecute(self, registre, sorties, brut, tache='k1:2'):
        self.assertFalse(any(a.succes and a.est_mutation for a in registre.actions),
                         [(a.outil, a.message) for a in registre.actions])
        changees = [a for a in registre.actions if (a.donnees or {}).get('cible_changee')]
        self.assertEqual(len(changees), 1, [a.donnees for a in registre.actions])
        action = changees[0]
        self.assertTrue(action.donnees['abandonnee_par_le_code'])
        self.assertEqual(action.donnees['decision_code'], 'abandonnee')
        ligne = action.donnees['ligne_cible_changee']
        self.assertIn('a changé depuis ma question', ligne)
        self.assertTrue(ligne.endswith('Redis-le si tu veux toujours.'), ligne)
        self.assertNotIn('—', ligne)
        self.assertEqual(sorties[0]['decision_code'], 'abandonnee')
        self.assertIsNone(sorties[0]['action_id'])
        self.assertIn('CIBLE CHANGEE', sorties[0]['resume'])
        self.assertTrue(outils_v2.tour_entierement_decide_par_le_code(registre, brut))
        return action

    def test_serie_bloc_deplace_entre_la_question_et_la_puce(self):
        demande = puces(self.demande_portee(tache='k1:1'))
        self.attendre([demande], SERIE)
        RecurringBlock.objects.filter(id=self.q.id).update(start_time=time(20, 0))
        registre, sorties = self.appliquer(SERIE)
        action = self.assertRienExecute(registre, sorties, SERIE)
        self.assertIn('je n\'ai rien supprimé', action.donnees['ligne_cible_changee'])
        self.assertActif(self.q)

        # Le modele qui tente la suppression le meme tour n'y arrive pas et
        # ne repose pas de question.
        avant = len(registre.actions)
        _, tools = self.outils(SERIE, registre=registre, tache='k1:2')
        retour = self.appeler(tools, 'delete_block', block_id=self.q.id)
        self.assertEqual(len(registre.actions), avant)
        self.assertIn('change', retour)
        self.assertActif(self.q)

    def test_serie_bloc_remplace_par_un_autre(self):
        demande = puces(self.demande_portee(tache='k1:1'))
        self.attendre([demande], SERIE)
        RecurringBlock.objects.filter(id=self.q.id).update(active=False)
        nouveau = self.bloc('Quart au dépanneur', 3, '19:00', '02:00', block_type='work',
                            flexibility='fixed', is_night_shift=True)
        registre, sorties = self.appliquer(SERIE)
        self.assertRienExecute(registre, sorties, SERIE)
        self.assertActif(nouveau)

    def test_occurrence_titre_change(self):
        demande = puces(self.demande_portee(tache='k1:1'))
        self.attendre([demande], OCCURRENCE)
        RecurringBlock.objects.filter(id=self.q.id).update(title='Quart au Couche-Tard')
        registre, sorties = self.appliquer(OCCURRENCE)
        self.assertRienExecute(registre, sorties, OCCURRENCE)
        self.assertFalse(RecurringBlockException.objects.filter(recurring_block=self.q).exists())

    def test_bloc_inchange_s_execute_toujours(self):
        demande = puces(self.demande_portee(tache='k1:1'))
        self.attendre([demande], SERIE)
        registre, sorties = self.appliquer(SERIE)
        self.assertEqual(sorties[0]['decision_code'], 'execute')
        self.assertActif(self.q, False)

    def test_tache_renommee(self):
        tache = Task.objects.create(user=self.user, title='Rapport de labo')
        demande = puces(self.premier_tour('supprime la tâche rapport de labo', 'delete_task',
                                          tache='k1:1', task_id=tache.id,
                                          confirm=False).donnees['demande'])
        self.attendre([demande], OUI)
        Task.objects.filter(id=tache.id).update(title='Rapport de chimie')
        registre, sorties = self.appliquer(OUI)
        self.assertRienExecute(registre, sorties, OUI)
        self.assertTrue(Task.objects.filter(id=tache.id).exists())

    def test_tache_completee_ailleurs(self):
        tache = Task.objects.create(user=self.user, title='Rapport de labo')
        demande = puces(self.premier_tour('supprime la tâche rapport de labo', 'delete_task',
                                          tache='k1:1', task_id=tache.id,
                                          confirm=False).donnees['demande'])
        self.attendre([demande], OUI)
        Task.objects.filter(id=tache.id).update(completed=True)
        registre, sorties = self.appliquer(OUI)
        self.assertRienExecute(registre, sorties, OUI)
        self.assertTrue(Task.objects.filter(id=tache.id).exists())

    def _evenement(self, titre='Dentiste', debut='10:00', fin='11:00', tache=None):
        tache = tache or Task.objects.create(user=self.user, title=titre)
        return ScheduledBlock.objects.create(
            user=self.user, task=tache, date=date(2026, 9, 17),
            start_time=time.fromisoformat(debut), end_time=time.fromisoformat(fin))

    def _demande_annulation(self):
        return puces(self.premier_tour('annule mon dentiste jeudi', 'cancel_scheduled_block',
                                       tache='k1:1', date=JEUDI,
                                       title='Dentiste').donnees['demande'])

    def test_evenement_deplace(self):
        ev = self._evenement()
        demande = self._demande_annulation()
        self.attendre([demande], OUI)
        ScheduledBlock.objects.filter(id=ev.id).update(start_time=time(13, 0),
                                                       end_time=time(14, 0))
        registre, sorties = self.appliquer(OUI)
        action = self.assertRienExecute(registre, sorties, OUI)
        self.assertIn("je n'ai rien annulé", action.donnees['ligne_cible_changee'])
        self.assertTrue(ScheduledBlock.objects.filter(id=ev.id).exists())

    def test_evenement_ajoute_au_lot(self):
        ev = self._evenement()
        demande = self._demande_annulation()
        self.attendre([demande], OUI)
        autre = self._evenement(debut='15:00', fin='16:00', tache=ev.task)
        registre, sorties = self.appliquer(OUI)
        self.assertRienExecute(registre, sorties, OUI)
        self.assertTrue(ScheduledBlock.objects.filter(id__in=[ev.id, autre.id]).count() == 2)

    def test_evenement_inchange_s_annule(self):
        ev = self._evenement()
        demande = self._demande_annulation()
        self.attendre([demande], OUI)
        registre, sorties = self.appliquer(OUI)
        self.assertEqual(sorties[0]['decision_code'], 'execute')
        self.assertFalse(ScheduledBlock.objects.filter(id=ev.id).exists())

    def test_vider_le_planning_apres_un_ajout(self):
        demande = puces(self.premier_tour('vide tout mon planning', 'clear_all_blocks',
                                          tache='k1:1', confirm=True).donnees['demande'])
        self.attendre([demande], OUI)
        nouveau = self.bloc('Gym', 1, '17:00', '18:00', block_type='sport')
        registre, sorties = self.appliquer(OUI)
        self.assertRienExecute(registre, sorties, OUI)
        self.assertActif(self.q)
        self.assertActif(nouveau)

    def test_vider_le_planning_inchange(self):
        demande = puces(self.premier_tour('vide tout mon planning', 'clear_all_blocks',
                                          tache='k1:1', confirm=True).donnees['demande'])
        self.attendre([demande], OUI)
        registre, sorties = self.appliquer(OUI)
        self.assertEqual(sorties[0]['decision_code'], 'execute')
        self.assertActif(self.q, False)

    def test_arret_d_un_bloc_deplace_de_jour(self):
        gym = self.bloc('Gym', 1, '17:00', '18:00', block_type='sport')
        demande = puces(self.premier_tour('supprime mon gym à partir de demain', 'update_block',
                                          tache='k1:1', block_id=gym.id,
                                          end_date='2026-09-15').donnees['demande'])
        self.assertEqual(demande['outil'], 'update_block')
        self.attendre([demande], OUI)
        RecurringBlock.objects.filter(id=gym.id).update(day_of_week=2)
        registre, sorties = self.appliquer(OUI)
        self.assertRienExecute(registre, sorties, OUI)
        gym.refresh_from_db()
        self.assertIsNone(gym.end_date)

    def test_le_modele_autorise_par_la_puce_relit_aussi_la_cible(self):
        """Sans passer par appliquer_choix_en_attente: l'appel du modele que la
        puce autorise ne supprime pas une cible changee."""
        demande = puces(self.demande_portee(tache='k1:1'))
        self.attendre([demande], SERIE)
        RecurringBlock.objects.filter(id=self.q.id).update(end_time=time(3, 0))
        registre, tools = self.outils(SERIE, tache='k1:3')
        retour = self.appeler(tools, 'delete_block', block_id=self.q.id)
        self.assertActif(self.q)
        self.assertFalse(any(a.succes and a.est_mutation for a in registre.actions))
        self.assertTrue(any((a.donnees or {}).get('cible_changee') for a in registre.actions))
        self.assertIn('change', retour)


# ── K3: seule la lecture egale a l'heure actuelle nomme le bloc ─────────────


class K3HeureActuelleTests(HarnaisGardes, TransactionTestCase):

    def setUp(self):
        super().setUp()
        self.gym = self.bloc('Gym', 1, '19:00', '20:00', block_type='sport')

    def _deplacer(self, brut, debut, tache):
        self.message_courant(brut)
        registre, tools = self.outils(brut, tache=tache)
        fin = f'{(int(debut[:2]) + 1) % 24:02d}{debut[2:]}'
        self.appeler(tools, 'update_block', block_id=self.gym.id, start_time=debut, end_time=fin)
        return registre.actions[-1]

    def test_une_autre_heure_est_refusee(self):
        for i, debut in enumerate(('15:00', '09:00', '13:30')):
            with self.subTest(debut=debut):
                action = self._deplacer('déplace mon gym à 7 h', debut, f'k3:r{i}')
                self.assertFalse(action.succes, action.message)
                self.assertIn('07:00', action.message)
                self.gym.refresh_from_db()
                self.assertEqual(self.gym.start_time, time(19, 0))

    def test_l_heure_dite_passe(self):
        action = self._deplacer('déplace mon gym à 7 h', '07:00', 'k3:a')
        self.assertTrue(action.succes, action.message)

    def test_l_heure_qui_nomme_le_bloc_ne_s_impose_pas(self):
        # « de 7 h » nomme le bloc de 19 h quand une autre heure dit ou il va.
        for i, (brut, debut) in enumerate((('mon gym de 7 h, déplace-le à 9 h', '21:00'),
                                           ('déplace mon gym de 7 h à 9 h', '09:00'),
                                           ('déplace mon gym de 7 h à 9 h', '21:00'))):
            with self.subTest(brut=brut, debut=debut):
                RecurringBlock.objects.filter(id=self.gym.id).update(
                    start_time=time(19, 0), end_time=time(20, 0))
                action = self._deplacer(brut, debut, f'k3:n{i}')
                self.assertTrue(action.succes, action.message)
        RecurringBlock.objects.filter(id=self.gym.id).update(
            start_time=time(19, 0), end_time=time(20, 0))
        action = self._deplacer('déplace mon gym de 7 h à 9 h', '11:00', 'k3:m')
        self.assertFalse(action.succes)
        self.assertIn('09:00', action.message)
        self.assertNotIn('07:00', action.message)


# ── K4: le moment de la journee choisit la lecture ──────────────────────────


def _lectures(texte):
    plat = dem.sans_accents(texte)
    return [outils_v2._lectures_d_heure(plat, ou, v) for v, ou in dem.heures_dites_positions(texte)]


class K4LecturesTests(SimpleTestCase):

    def test_moment_de_la_journee_dans_la_proposition(self):
        cas = (
            ('souper jeudi soir à 6 h', [['18:00']]),
            ('demain matin à 7 h', [['07:00']]),
            ('gym à 6 h le soir', [['18:00']]),
            ('ce soir à 8 h', [['20:00']]),
            ('cet après-midi à 3 h', [['15:00']]),
            ('cet après midi à 3 h', [['15:00']]),
            ("cet aprem à 2 h 30", [['14:30']]),
            ('tous les matins à 7 h', [['07:00']]),
            ("jeudi avant-midi à 10 h", [['10:00']]),
            ('souper à 6 h', [['18:00']]),
            ('gym à 7 h pm', [['19:00']]),
            ('mardi en soirée à 9 h', [['21:00']]),
            ("cours à 2 h de l'après-midi", [['14:00']]),
            ('gym jeudi à 7 h', [['07:00', '19:00']]),
            # Le marqueur va a l'heure la plus proche seulement.
            ('de 9 h à 5 h le soir', [['09:00', '21:00'], ['17:00']]),
            ('de 7 h du matin à 9 h', [['07:00'], ['09:00', '21:00']]),
            # Une autre proposition ne compte pas.
            ('gym ce matin, souper à 6 h', [['18:00']]),
            ('gym à 7 h, souper ce soir', [['07:00', '19:00']]),
        )
        for texte, attendu in cas:
            with self.subTest(texte=texte):
                self.assertEqual(_lectures(texte), attendu)

    def test_apres_midi_n_est_pas_une_duree(self):
        self.assertEqual(dem.heures_dites("cours à 2 h de l'après-midi"), ['02:00'])
        self.assertEqual(dem.heures_dites("cours à 2 h de l’aprem"), ['02:00'])
        self.assertEqual(dem.heures_dites('cours à 2 h de la nuit'), ['02:00'])
        # Les durees restent des durees.
        self.assertEqual(dem.heures_dites('2 h de lecture jeudi'), [])
        self.assertEqual(dem.heures_dites("2 h d'étude par jour"), [])


class K4GardeTests(HarnaisGardes, TransactionTestCase):

    def _appel(self, brut, titre, debut, tache):
        self.message_courant(brut)
        registre, tools = self.outils(brut, tache=tache)
        fin = f'{(int(debut[:2]) + 1) % 24:02d}{debut[2:]}'
        self.appeler(tools, 'schedule_task_at', title=titre, date=JEUDI,
                     start_time=debut, end_time=fin)
        return registre.actions[-1]

    def _creer(self, brut, titre, debut, tache, jour=1):
        self.message_courant(brut)
        registre, tools = self.outils(brut, tache=tache)
        fin = f'{(int(debut[:2]) + 1) % 24:02d}{debut[2:]}'
        self.appeler(tools, 'create_block', title=titre, days=[jour], start_time=debut,
                     end_time=fin, block_type='sport')
        return registre.actions[-1]

    def test_la_lecture_contraire_est_refusee(self):
        refus = (('souper jeudi soir à 6 h', 'Souper', '06:00'),
                 ('gym jeudi à 6 h le soir', 'Gym', '06:00'),
                 ('gym jeudi matin à 5 h', 'Gym', '17:00'),
                 ('souper jeudi à 6 h', 'Souper', '06:00'),
                 ("cours jeudi à 2 h de l'après-midi", 'Cours', '02:00'),
                 ('cours jeudi après-midi à 3 h', 'Cours', '03:00'))
        for i, (brut, titre, debut) in enumerate(refus):
            with self.subTest(brut=brut, debut=debut):
                ScheduledBlock.objects.filter(user=self.user).delete()
                self.assertFalse(self._appel(brut, titre, debut, f'k4:r{i}').succes)

    def test_la_bonne_lecture_passe(self):
        acceptes = (('souper jeudi soir à 6 h', 'Souper', '18:00'),
                    ('gym jeudi à 6 h le soir', 'Gym', '18:00'),
                    ('gym jeudi matin à 5 h', 'Gym', '05:00'),
                    ("cours jeudi à 2 h de l'après-midi", 'Cours', '14:00'),
                    ('cours jeudi après-midi à 3 h', 'Cours', '15:00'))
        for i, (brut, titre, debut) in enumerate(acceptes):
            with self.subTest(brut=brut, debut=debut):
                ScheduledBlock.objects.filter(user=self.user).delete()
                action = self._appel(brut, titre, debut, f'k4:a{i}')
                self.assertTrue(action.succes, action.message)

    def test_bloc_recurrent_tous_les_matins(self):
        self.assertFalse(self._creer('gym tous les matins à 6 h', 'Gym', '18:00', 'k4:m').succes)
        self.assertFalse(self._creer('gym les mardis soir à 6 h', 'Gym', '06:00', 'k4:s').succes)
        RecurringBlock.objects.filter(user=self.user, title='Gym').delete()
        self.assertTrue(self._creer('gym tous les matins à 6 h', 'Gym', '06:00', 'k4:ma').succes)
