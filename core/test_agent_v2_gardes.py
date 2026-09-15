"""
Gardes du code sur les outils v2 (lots 1d, 2d, 2c-outils, 3a-donnees).

Le defaut d'origine (enquete du 2026-09-14): la garde destructive cherchait
« supprime|efface|ok|oui » n'importe ou dans le message, donc la demande de
suppression etait sa propre confirmation, et delete_block ou
cancel_scheduled_block n'etaient pas gardes du tout. Ces tests verrouillent le
contrat qui la remplace:

- une action destructrice ne passe qu'avec une reponse donnee AU TOUR SUIVANT
  a une question posee, liee a l'identite de la cible et lue par demande;
- une suppression qui nomme un jour demande la portee (occurrence ou serie);
- une heure dite par l'utilisateur n'est jamais changee en silence;
- plus de cinq creations, ou un plan de semaine applique, demandent d'abord;
- « cette semaine » ne cree pas d'habitude sans fin.

TransactionTestCase: l'adaptateur execute l'ORM dans un thread d'executeur,
qui ne voit pas la transaction non validee d'un TestCase.

Horloge: lundi 14 septembre 2026, 8 h, America/Toronto (jour 0 = lundi).
"""
import asyncio
from datetime import date, datetime, time, timedelta
from unittest import mock

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TransactionTestCase
from django.utils import timezone

from core.models import (ConversationMessage, RecurringBlock,
                         RecurringBlockException, ScheduledBlock, Task)
from services.agent.tools import TOOL_MAP, execute_tool
from services.agent_v2 import demandes as dem
from services.agent_v2 import outils as outils_v2
from services.agent_v2.outils import outils_pour
from services.agent_v2.registre import Registre

AUJOURDHUI = date(2026, 9, 14)  # un lundi
JOURS = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']
MOIS = {9: 'sept.', 10: 'oct.'}

_COPIE = {
    'portee_jour': {
        'occurrence': ('Seulement ce {jour}', "Seulement ce {jour} {date} (sauter l'occurrence)."),
        'serie': ('Tous les {jour}s', 'Tous les {jour}s (supprimer la série).'),
        'annuler': ('Non, garde tout', 'Non, ne change rien.'),
    },
    'destructif': {
        'confirmer': ('Oui, confirme', 'Oui, je confirme.'),
        'annuler': ('Non, garde tout', 'Non, ne change rien.'),
    },
    'creation_en_masse': {
        'confirmer': ('Oui, continue', 'Oui, continue les ajouts.'),
        'annuler': ('Non, arrête là', 'Non, arrête là.'),
    },
    'optimisation': {
        'confirmer': ('Applique le plan', 'Oui, applique le plan.'),
        'annuler': ("Montre d'abord", "Montre-moi d'abord la proposition."),
    },
}


def puces(demande):
    """La demande telle que b6 la persiste: avec ses puces (copie du contrat)."""
    cible = demande.get('cible') or {}
    jour = JOURS[cible['jour']] if isinstance(cible.get('jour'), int) else ''
    date_courte = ''
    if cible.get('date'):
        d = date.fromisoformat(cible['date'])
        date_courte = f'{d.day} {MOIS.get(d.month, "")}'
    chips = []
    for option in demande['options']:
        copie = _COPIE.get(demande['motif'], {}).get(option['id'])
        if copie:
            chips.append({
                'label': copie[0].format(jour=jour, date=date_courte),
                'value': copie[1].format(jour=jour, date=date_courte),
                'option': option['id'],
            })
    return {**demande, 'chips': chips}


def _minutes(hhmm):
    h, m = hhmm.split(':')
    return int(h) * 60 + int(m)


class HarnaisGardes:
    """Melange sans TestCase: importe par core/test_agent_v2_choix_executes.py
    sans que ses tests soient decouverts deux fois."""

    def setUp(self):
        super().setUp()
        self.user = User.objects.create_user(username='gardes', password='x')
        vrai_localtime = timezone.localtime
        vrai_localdate = timezone.localdate
        maintenant = timezone.make_aware(datetime(2026, 9, 14, 8, 0))

        def faux_localtime(value=None, timezone=None):
            return maintenant if value is None else vrai_localtime(value, timezone)

        def faux_localdate(value=None, timezone=None):
            return AUJOURDHUI if value is None else vrai_localdate(value, timezone)

        for cible, remplacant in (('django.utils.timezone.localtime', faux_localtime),
                                  ('django.utils.timezone.localdate', faux_localdate)):
            patcheur = mock.patch(cible, side_effect=remplacant)
            patcheur.start()
            self.addCleanup(patcheur.stop)
        # Jeudi 19 h a 2 h: le quart de nuit de l'enquete.
        self.q = self.bloc('Quart au dépanneur', 3, '19:00', '02:00', block_type='work',
                           flexibility='fixed', is_night_shift=True)

    # -------------------------------------------------------------- fixtures

    def bloc(self, titre, jour, debut, fin, block_type='course', flexibility=None, **extra):
        return RecurringBlock.objects.create(
            user=self.user, title=titre, block_type=block_type, day_of_week=jour,
            start_time=time.fromisoformat(debut), end_time=time.fromisoformat(fin),
            flexibility=flexibility, **extra)

    def message_courant(self, texte):
        return ConversationMessage.objects.create(user=self.user, role='user', content=texte)

    def attendre(self, demandes, texte):
        """[U1, A1 qui repond a U1 avec ses demandes, U2 courant]."""
        u1 = ConversationMessage.objects.create(user=self.user, role='user', content='demande')
        ConversationMessage.objects.create(
            user=self.user, role='assistant', content='question',
            metadata={'en_reponse_a': u1.pk, 'demandes': demandes})
        return self.message_courant(texte)

    def outils(self, brut, enrichi=None, registre=None, tache='u:1'):
        registre = registre if registre is not None else Registre()
        tools = {t.name: t for t in outils_pour(
            self.user, registre, brut if enrichi is None else enrichi,
            tache=tache, message_brut=brut)}
        return registre, tools

    def appeler(self, tools, nom, **kwargs):
        return asyncio.run(tools[nom].function_schema.function(**kwargs))

    def premier_tour(self, brut, nom, tache='u:1', **kwargs):
        """Le tour de la REQUETE: message courant, un appel, l'action consignee."""
        self.message_courant(brut)
        registre, tools = self.outils(brut, tache=tache)
        self.appeler(tools, nom, **kwargs)
        return registre.actions[-1]

    def demande_portee(self, tache='u:1'):
        action = self.premier_tour('efface tout jeudi', 'delete_block', tache=tache,
                                   block_id=self.q.id)
        return action.donnees['demande']

    def assertActif(self, bloc, actif=True):
        bloc.refresh_from_db()
        self.assertEqual(bloc.active, actif)


class GardesDestructivesTests(HarnaisGardes, TransactionTestCase):

    def test_suppression_d_un_jour_demande_la_portee(self):
        action = self.premier_tour('efface tout jeudi', 'delete_block', block_id=self.q.id)
        self.assertFalse(action.succes)
        self.assertEqual(action.message, outils_v2.MESSAGE_RETENUE)
        self.assertTrue(action.donnees['needs_confirmation'])
        demande = action.donnees['demande']
        self.assertEqual(demande['motif'], 'portee_jour')
        self.assertEqual([o['id'] for o in demande['options']], ['occurrence', 'serie', 'annuler'])
        # Quart de nuit: l'occurrence est le soir ou il COMMENCE, le jeudi.
        self.assertEqual(demande['cible']['date'], '2026-09-17')
        self.assertEqual(demande['cible']['jour'], 3)
        self.assertEqual(demande['cle'], dem.cle_demande(
            'portee_jour', {'block_id': self.q.id, 'date': '2026-09-17'}))
        self.assertEqual(demande['options'][0]['effet'], {
            'outil': 'skip_block_occurrence',
            'parametres': {'date': '2026-09-17', 'title': 'Quart au dépanneur',
                           'block_type': 'work'}})
        self.assertEqual(demande['options'][1]['effet'],
                         {'outil': 'delete_block', 'parametres': {'block_id': self.q.id}})
        self.assertIsNone(demande['options'][2]['effet'])
        self.assertTrue(demande['emise_le'])
        self.assertActif(self.q)

    def test_la_serie_confirmee_au_tour_suivant_passe(self):
        demande = puces(self.demande_portee())
        brut = 'Tous les jeudis (supprimer la série).'
        self.attendre([demande], brut)
        registre, tools = self.outils(brut, tache='u:2')
        self.appeler(tools, 'delete_block', block_id=self.q.id)
        self.assertTrue(registre.actions[-1].succes)
        self.assertActif(self.q, False)

    def test_oui_seul_ne_tranche_pas_la_portee(self):
        demande = puces(self.demande_portee())
        self.attendre([demande], 'oui')
        registre, tools = self.outils('oui', tache='u:2')
        self.appeler(tools, 'delete_block', block_id=self.q.id)
        action = registre.actions[-1]
        self.assertFalse(action.succes)
        # C'est la question de PORTEE qu'on repose, pas une confirmation.
        self.assertEqual(action.donnees['demande']['motif'], 'portee_jour')
        self.assertEqual(action.donnees['demande']['cle'], demande['cle'])
        self.assertNotIn('chips', action.donnees['demande'])
        self.assertActif(self.q)

    def test_portee_ambigue_redemandee(self):
        demande = puces(self.demande_portee())
        for brut in ('pas tous les jeudis, juste celui-là',
                     'non, seulement ce jeudi, pas chaque semaine'):
            with self.subTest(brut=brut):
                self.assertIsNone(dem.option_choisie(brut, demande))
        brut = 'pas tous les jeudis, juste celui-là'
        self.attendre([demande], brut)
        registre, tools = self.outils(brut, tache='u:2')
        self.appeler(tools, 'delete_block', block_id=self.q.id)
        self.assertFalse(registre.actions[-1].succes)
        self.assertActif(self.q)
        self.assertFalse(RecurringBlockException.objects.exists())

    def test_une_reponse_contraire_ne_repose_pas_la_question(self):
        """« Non, ne change rien » puis un appel du modele: rien ne s'execute
        et rien n'est consigne, sinon la meme question reviendrait."""
        demande = puces(self.demande_portee())
        brut = 'Non, ne change rien.'
        self.attendre([demande], brut)
        registre, tools = self.outils(brut, tache='u:2')
        retour = self.appeler(tools, 'delete_block', block_id=self.q.id)
        self.assertEqual(retour, outils_v2.MESSAGE_DEJA_TRANCHE)
        self.assertEqual(registre.actions, [])
        self.assertActif(self.q)

    def test_confirmation_generique_au_tour_suivant(self):
        chimie = self.bloc('Chimie générale', 1, '13:00', '15:00')
        action = self.premier_tour('supprime mon cours de chimie', 'delete_block',
                                   block_id=chimie.id)
        demande = action.donnees['demande']
        self.assertEqual(demande['motif'], 'destructif')
        self.assertEqual([o['id'] for o in demande['options']], ['confirmer', 'annuler'])
        self.assertEqual(demande['cible']['titre'], 'Chimie générale')
        self.assertActif(chimie)

        # Round 6 (D1): un oui libre n'autorise plus rien; seule la puce le fait.
        self.attendre([puces(demande)], 'Oui, vas-y')
        registre, tools = self.outils('Oui, vas-y', tache='u:2')
        self.appeler(tools, 'delete_block', block_id=chimie.id)
        self.assertFalse(registre.actions[-1].succes)
        self.assertActif(chimie)

        self.attendre([puces(demande)], 'Oui, je confirme.')
        registre, tools = self.outils('Oui, je confirme.', tache='u:3')
        self.appeler(tools, 'delete_block', block_id=chimie.id)
        self.assertTrue(registre.actions[-1].succes)
        self.assertActif(chimie, False)

    def test_ok_mais_autre_chose_ne_confirme_pas(self):
        chimie = self.bloc('Chimie générale', 1, '13:00', '15:00')
        demande = puces(self.premier_tour('supprime mon cours de chimie', 'delete_block',
                                          block_id=chimie.id).donnees['demande'])
        brut = 'ok mais enlève aussi la chimie'
        self.assertIsNone(dem.option_choisie(brut, demande))
        self.attendre([demande], brut)
        registre, tools = self.outils(brut, tache='u:2')
        self.appeler(tools, 'delete_block', block_id=chimie.id)
        self.assertFalse(registre.actions[-1].succes)
        self.assertActif(chimie)

    def test_la_confirmation_vise_la_meme_cible(self):
        anglais = self.bloc('Anglais', 0, '08:00', '09:00')
        biologie = self.bloc('Biologie', 2, '08:00', '09:00')
        demande = puces(self.premier_tour("supprime mon cours d'anglais", 'delete_block',
                                          block_id=anglais.id).donnees['demande'])
        self.attendre([demande], 'oui')
        registre, tools = self.outils('oui', tache='u:2')
        self.appeler(tools, 'delete_block', block_id=biologie.id)
        self.assertFalse(registre.actions[-1].succes)
        self.assertActif(biologie)
        self.assertActif(anglais)

    def test_option_evaluee_par_demande(self):
        gym = self.bloc('Gym', 1, '18:00', '19:00', block_type='sport')
        serie = 'Tous les jeudis (supprimer la série).'

        # 1. Seule la question de portee, comme b6 persiste le gagnant.
        self.attendre([puces(self.demande_portee())], serie)
        registre, tools = self.outils(serie, tache='u:2')
        self.appeler(tools, 'delete_block', block_id=self.q.id)
        self.assertTrue(registre.actions[-1].succes)

        # 2. Portee d'un autre bloc ET vidage du planning en attente ensemble:
        # la puce « serie » n'est pas une confirmation du vidage.
        chimie = self.bloc('Chimie générale', 3, '13:00', '15:00')
        portee = puces(self.premier_tour('efface tout jeudi', 'delete_block', tache='u:3',
                                         block_id=chimie.id).donnees['demande'])
        vider = puces(self.premier_tour('vide mon planning', 'clear_all_blocks', tache='u:4',
                                        confirm=True).donnees['demande'])
        self.attendre([portee, vider], serie)
        registre, tools = self.outils(serie, tache='u:5')
        self.appeler(tools, 'delete_block', block_id=chimie.id)
        self.assertTrue(registre.actions[-1].succes)
        self.appeler(tools, 'clear_all_blocks', confirm=True)
        self.assertFalse(registre.actions[-1].succes)
        self.assertActif(gym)

        # 3. La confirmation d'ajouts en serie n'autorise pas un vidage.
        self.attendre([vider], 'Oui, continue les ajouts.')
        registre, tools = self.outils('Oui, continue les ajouts.', tache='u:6')
        self.appeler(tools, 'clear_all_blocks', confirm=True)
        self.assertFalse(registre.actions[-1].succes)
        self.assertActif(gym)

    def test_suppression_de_tache_confirmee_au_tour_suivant(self):
        tache = Task.objects.create(user=self.user, title='Rapport de labo')
        action = self.premier_tour('supprime la tâche rapport de labo', 'delete_task',
                                   task_id=tache.id, confirm=False)
        self.assertFalse(action.succes)
        demande = action.donnees['demande']
        self.assertEqual(demande['cle'], dem.cle_demande('delete_task', {'task_id': tache.id}))
        self.assertEqual(demande['options'][0]['effet']['parametres'],
                         {'task_id': tache.id, 'confirm': True})

        self.attendre([puces(demande)], 'Oui, je confirme.')
        registre, tools = self.outils('Oui, je confirme.', tache='u:2')
        self.appeler(tools, 'delete_task', task_id=tache.id, confirm=True)
        self.assertTrue(registre.actions[-1].succes)
        self.assertFalse(Task.objects.filter(id=tache.id).exists())

    def test_la_confirmation_force_confirm(self):
        """Le modele oublie confirm=true au tour de confirmation: le code le
        force, la cible etant deja confirmee par l'utilisateur."""
        tache = Task.objects.create(user=self.user, title='Rapport de labo')
        demande = puces(self.premier_tour('supprime la tâche rapport de labo', 'delete_task',
                                          task_id=tache.id, confirm=False).donnees['demande'])
        self.attendre([demande], 'Oui, je confirme.')
        registre, tools = self.outils('Oui, je confirme.', tache='u:2')
        self.appeler(tools, 'delete_task', task_id=tache.id, confirm=False)
        self.assertTrue(registre.actions[-1].succes)
        self.assertTrue(registre.actions[-1].parametres['confirm'])

    def test_un_tour_intercale_annule_l_attente(self):
        demande = puces(self.demande_portee())
        u1 = self.message_courant('efface tout jeudi')
        ConversationMessage.objects.create(
            user=self.user, role='assistant', content='question',
            metadata={'en_reponse_a': u1.pk, 'demandes': [demande]})
        self.message_courant('au fait, et demain ?')
        brut = 'Tous les jeudis (supprimer la série).'
        self.message_courant(brut)
        self.assertEqual(dem.demandes_en_attente(self.user), [])
        registre, tools = self.outils(brut, tache='u:2')
        self.appeler(tools, 'delete_block', block_id=self.q.id)
        self.assertFalse(registre.actions[-1].succes)
        self.assertActif(self.q)

    def test_tours_chevauches_ignores(self):
        demande = puces(self.demande_portee())
        u1 = self.message_courant('efface tout jeudi')
        self.message_courant('un deuxieme message envoye trop vite')
        # La reponse au premier message arrive APRES le deuxieme.
        ConversationMessage.objects.create(
            user=self.user, role='assistant', content='question',
            metadata={'en_reponse_a': u1.pk, 'demandes': [demande]})
        brut = 'Tous les jeudis (supprimer la série).'
        self.message_courant(brut)
        self.assertEqual(dem.demandes_en_attente(self.user), [])
        registre, tools = self.outils(brut, tache='u:2')
        self.appeler(tools, 'delete_block', block_id=self.q.id)
        self.assertFalse(registre.actions[-1].succes)
        self.assertActif(self.q)

    def test_demande_expiree(self):
        demande = puces(self.demande_portee())
        demande['emise_le'] = (timezone.now() - timedelta(minutes=31)).isoformat()
        self.attendre([demande], 'Tous les jeudis (supprimer la série).')
        self.assertEqual(dem.demandes_en_attente(self.user), [])

        fraiche = dict(demande, emise_le=(timezone.now() - timedelta(minutes=29)).isoformat())
        self.attendre([fraiche], 'Tous les jeudis (supprimer la série).')
        self.assertEqual(len(dem.demandes_en_attente(self.user)), 1)

    def test_la_demande_ne_se_confirme_pas_elle_meme(self):
        self.bloc('Gym', 1, '18:00', '19:00', block_type='sport')
        action = self.premier_tour('ok efface tout mon planning', 'clear_all_blocks', confirm=True)
        self.assertFalse(action.succes)
        self.assertEqual(action.donnees['demande']['motif'], 'destructif')
        self.assertEqual(RecurringBlock.objects.filter(user=self.user, active=True).count(), 2)

    def test_message_brut_et_non_enrichi(self):
        chimie = self.bloc('Chimie générale', 1, '13:00', '15:00')
        demande = puces(self.premier_tour('supprime mon cours de chimie', 'delete_block',
                                          block_id=chimie.id).donnees['demande'])
        brut = 'Oui, je confirme.'
        enrichi = brut + '\n\nIMPORT RECENT: Calcul différentiel lundi 10:30, cette semaine, jeudi'
        self.attendre([demande], brut)

        # Contre-epreuve: lue sur le message enrichi, la regle refuserait.
        registre_e = Registre()
        tools_e = {t.name: t for t in outils_pour(self.user, registre_e, enrichi, tache='u:x')}
        self.appeler(tools_e, 'delete_block', block_id=chimie.id)
        self.assertFalse(registre_e.actions[-1].succes)
        self.assertActif(chimie)

        registre, tools = self.outils(brut, enrichi=enrichi, tache='u:2')
        self.appeler(tools, 'delete_block', block_id=chimie.id)
        self.assertTrue(registre.actions[-1].succes)
        self.assertActif(chimie, False)

        # Tour neuf: les heures et « cette semaine » de l'import ne comptent pas.
        self.bloc('Calcul différentiel', 3, '10:00', '11:50')
        brut = 'ajoute ma lecture'
        self.message_courant(brut)
        registre, tools = self.outils(brut, enrichi=brut + enrichi[len('Oui, je confirme.'):],
                                      tache='u:3')
        self.appeler(tools, 'schedule_task_at', title='Lecture', date='2026-09-17',
                     start_time='10:30', end_time='11:30')
        conflit = registre.actions[-1]
        self.assertFalse(conflit.succes)
        self.assertIn('conflict', conflit.donnees)
        self.assertNotIn('demande', conflit.donnees)
        self.appeler(tools, 'create_block', title='Étude', block_type='revision',
                     days=['lundi'], start_time='16:00', end_time='18:00')
        cree = registre.actions[-1]
        self.assertTrue(cree.succes)
        self.assertNotIn('borne_auto', cree.donnees)
        self.assertIsNone(RecurringBlock.objects.get(user=self.user, title='Étude').end_date)

    def test_saut_en_masse_contourne_refuse(self):
        action = self.premier_tour('efface tout jeudi', 'skip_block_occurrence',
                                   date='2026-09-17', title='Quart au dépanneur')
        self.assertFalse(action.succes)
        self.assertEqual(action.donnees['demande']['motif'], 'portee_jour')
        self.assertFalse(RecurringBlockException.objects.exists())

        gym = self.bloc('Gym', 1, '18:00', '19:00', block_type='sport')
        action = self.premier_tour('saute mon gym demain', 'skip_block_occurrence', tache='u:2',
                                   date='2026-09-15', title='Gym')
        self.assertTrue(action.succes)
        self.assertTrue(RecurringBlockException.objects.filter(
            recurring_block=gym, date=date(2026, 9, 15)).exists())

    def test_fin_de_bloc_demande_confirmation(self):
        action = self.premier_tour("mon quart s'arrête aujourd'hui", 'update_block',
                                   block_id=self.q.id, end_date='2026-09-14')
        self.assertFalse(action.succes)
        demande = action.donnees['demande']
        self.assertEqual(demande['motif'], 'destructif')
        self.assertEqual(demande['cle'], outils_v2._cle_destructive(
            'update_block', {'block_id': self.q.id, 'end_date': '2026-09-14'}))
        self.q.refresh_from_db()
        self.assertIsNone(self.q.end_date)

    def test_une_fin_future_demande_aussi_confirmation(self):
        # Decision de Darius (2026-09-14): une fin lointaine ecrite librement
        # demande un tap. Detail dans core/test_agent_v2_fin_de_serie.py.
        action = self.premier_tour('mon quart finit le 15 octobre', 'update_block',
                                   block_id=self.q.id, end_date='2026-10-15')
        self.assertFalse(action.succes)
        self.assertEqual(action.donnees['demande']['motif'], 'destructif')
        self.q.refresh_from_db()
        self.assertIsNone(self.q.end_date)

    def test_annulation_d_evenement_demande_confirmation(self):
        tache = Task.objects.create(user=self.user, title='Dentiste')
        ScheduledBlock.objects.create(user=self.user, task=tache, date=date(2026, 9, 16),
                                      start_time=time(14, 0), end_time=time(15, 0), locked=True)
        action = self.premier_tour('annule mon dentiste', 'cancel_scheduled_block',
                                   date='2026-09-16', title='Dentiste')
        self.assertFalse(action.succes)
        demande = action.donnees['demande']
        self.assertEqual(demande['motif'], 'destructif')
        self.assertEqual(demande['cle'], dem.cle_demande(
            'cancel_scheduled_block', {'date': '2026-09-16', 'title': 'dentiste'}))
        self.assertEqual(demande['cible']['debut'], '14:00')
        self.assertTrue(ScheduledBlock.objects.filter(user=self.user).exists())


class HeuresDitesTests(HarnaisGardes, TransactionTestCase):

    def test_heure_dite_jamais_changee_en_silence(self):
        from services.scheduling.placement import open_intervals

        self.bloc('Calcul différentiel', 3, '10:00', '11:50')
        brut = 'mets mon rendez-vous chez le dentiste jeudi à 10 h 30'
        self.message_courant(brut)
        registre, tools = self.outils(brut)
        self.appeler(tools, 'schedule_task_at', title='Dentiste', date='2026-09-17',
                     start_time='10:30', end_time='11:30')
        refus = registre.actions[-1]
        self.assertFalse(refus.succes)
        self.assertIn('conflict', refus.donnees)
        demande = refus.donnees['demande']
        self.assertEqual(demande['motif'], 'heure_refusee')
        self.assertEqual(demande['cible']['avec']['titre'], 'Calcul différentiel')
        creneaux = [o for o in demande['options'] if o['id'].startswith('creneau_')]
        self.assertIn(len(creneaux), (2, 3))
        self.assertEqual(demande['options'][-1]['id'], 'autre_jour')
        libres = open_intervals(self.user, date(2026, 9, 17), 0, 1440)
        for option in creneaux:
            s, e = _minutes(option['cible']['debut']), _minutes(option['cible']['fin'])
            self.assertEqual(e - s, 60)
            self.assertTrue(any(ls <= s and e <= le for ls, le in libres), option)
            self.assertEqual(option['cible']['date'], '2026-09-17')

        # Meme jour, autre heure, AUTRE titre: toujours refuse.
        self.appeler(tools, 'schedule_task_at', title='RDV dentiste', date='2026-09-17',
                     start_time='12:00', end_time='13:00')
        second = registre.actions[-1]
        self.assertFalse(second.succes)
        self.assertEqual(second.donnees['demande']['cle'], demande['cle'])
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 0)

    def test_heure_dite_create_block_jamais_changee(self):
        self.bloc('Calcul différentiel', 0, '10:00', '11:50')
        brut = 'ajoute Statistiques le lundi à 10 h'
        self.message_courant(brut)
        registre, tools = self.outils(brut)
        self.appeler(tools, 'create_block', title='Statistiques', block_type='course',
                     days=['lundi'], start_time='10:00', end_time='12:00')
        premier = registre.actions[-1]
        self.assertEqual(premier.donnees['created'], [])
        self.assertEqual(premier.donnees['demande']['motif'], 'heure_refusee')
        self.assertEqual(premier.donnees['demande']['cible']['jour'], 0)

        self.appeler(tools, 'create_block', title='Stats', block_type='course',
                     days=['lundi'], start_time='12:00', end_time='14:00')
        self.assertFalse(registre.actions[-1].succes)
        self.assertEqual(RecurringBlock.objects.filter(user=self.user, title='Stats').count(), 0)

    def test_sans_heure_dite_le_repli_reste_permis(self):
        self.bloc('Calcul différentiel', 3, '10:00', '11:50')
        brut = 'planifie ma lecture jeudi'
        self.message_courant(brut)
        registre, tools = self.outils(brut)
        self.appeler(tools, 'schedule_task_at', title='Lecture', date='2026-09-17',
                     start_time='10:30', end_time='11:30')
        self.assertFalse(registre.actions[-1].succes)
        self.assertNotIn('demande', registre.actions[-1].donnees)
        self.appeler(tools, 'schedule_task_at', title='Lecture', date='2026-09-17',
                     start_time='13:00', end_time='14:00')
        self.assertTrue(registre.actions[-1].succes)

    def test_chevauchement_sans_heure_dite_informe(self):
        self.bloc('Calcul différentiel', 0, '10:00', '11:50')
        brut = 'ajoute Statistiques le lundi'
        self.message_courant(brut)
        registre, tools = self.outils(brut)
        self.appeler(tools, 'create_block', title='Statistiques', block_type='course',
                     days=['lundi'], start_time='10:00', end_time='12:00')
        demande = registre.actions[-1].donnees['demande']
        self.assertEqual(demande['motif'], 'chevauchement')
        self.assertEqual([o['id'] for o in demande['options']], ['autre_heure', 'annuler'])
        # Informatif: un autre essai passe.
        self.appeler(tools, 'create_block', title='Statistiques', block_type='course',
                     days=['lundi'], start_time='13:00', end_time='15:00')
        self.assertTrue(registre.actions[-1].succes)


class OptimisationEtCreationsTests(HarnaisGardes, TransactionTestCase):

    def test_optimisation_appliquee_exige_un_tour_de_confirmation(self):
        self.bloc('Sport', 1, '06:00', '07:00', block_type='sport', flexibility='flexible')
        action = self.premier_tour('optimise ma semaine', 'optimize_week', apply=True)
        self.assertFalse(action.succes)
        demande = action.donnees['demande']
        self.assertEqual(demande['motif'], 'optimisation')
        self.assertEqual(demande['cle'], 'optimize_week:apply')
        self.assertEqual(len(demande['parametres']['plan_hash']), 12)

        registre, tools = self.outils('optimise ma semaine', tache='u:1b')
        self.appeler(tools, 'optimize_week', apply=False)
        self.assertTrue(registre.actions[-1].succes)

        self.attendre([puces(demande)], 'Oui, applique le plan.')
        registre, tools = self.outils('Oui, applique le plan.', tache='u:2')
        self.appeler(tools, 'optimize_week', apply=True)
        self.assertTrue(registre.actions[-1].succes)
        self.assertTrue(registre.actions[-1].donnees['applied'])

    def test_plus_de_cinq_creations_demandent_confirmation(self):
        semaine = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi']
        brut = 'ajoute mes cours du matin'
        self.message_courant(brut)
        registre, tools = self.outils(brut)
        self.appeler(tools, 'create_block', title='Cours du matin', block_type='course',
                     days=semaine, start_time='08:00', end_time='09:00')
        self.assertEqual(len(registre.actions[-1].donnees['created']), 5)
        self.appeler(tools, 'schedule_task_at', title='Lecture', date='2026-09-19',
                     start_time='14:00', end_time='15:00')
        refus = registre.actions[-1]
        self.assertFalse(refus.succes)
        demande = refus.donnees['demande']
        self.assertEqual(demande['motif'], 'creation_en_masse')
        self.assertEqual(demande['cible']['nombre'], 6)
        self.assertEqual(demande['cible']['deja'], 5)
        self.assertEqual(demande['cible']['titre'], 'Lecture')

        # Tour neuf: six jours d'un coup, rien n'est ecrit.
        six = semaine + ['samedi']
        brut = 'ajoute du yoga tous les jours sauf le dimanche'
        self.message_courant(brut)
        registre, tools = self.outils(brut, tache='u:2')
        self.appeler(tools, 'create_block', title='Yoga', block_type='sport', days=six,
                     start_time='18:00', end_time='19:00')
        refus = registre.actions[-1]
        self.assertFalse(refus.succes)
        self.assertEqual(RecurringBlock.objects.filter(user=self.user, title='Yoga').count(), 0)

        # Tour de confirmation: tout passe dans CE tour.
        self.attendre([puces(refus.donnees['demande'])], 'Oui, continue les ajouts.')
        registre, tools = self.outils('Oui, continue les ajouts.', tache='u:3')
        self.appeler(tools, 'create_block', title='Yoga', block_type='sport', days=six,
                     start_time='18:00', end_time='19:00')
        self.appeler(tools, 'schedule_task_at', title='Lecture', date='2026-09-19',
                     start_time='14:00', end_time='15:00')
        self.assertTrue(all(a.succes for a in registre.actions))
        self.assertEqual(RecurringBlock.objects.filter(user=self.user, title='Yoga').count(), 6)
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 1)

    def test_rejouer_une_creation_ne_compte_pas_double(self):
        """L'idempotence consigne deux fois la meme creation: le compte des
        creations du tour se fait par identifiant, pas par ligne."""
        brut = 'ajoute mes cours du matin'
        self.message_courant(brut)
        registre, tools = self.outils(brut)
        args = dict(title='Cours du matin', block_type='course',
                    days=['lundi', 'mardi', 'mercredi'], start_time='08:00', end_time='09:00')
        self.appeler(tools, 'create_block', **args)
        self.appeler(tools, 'create_block', **args)
        self.appeler(tools, 'schedule_task_at', title='Lecture', date='2026-09-19',
                     start_time='14:00', end_time='15:00')
        self.assertTrue(registre.actions[-1].succes)

    def test_cette_semaine_borne_le_recurrent(self):
        cas = (
            ('je veux étudier plus cette semaine', '2026-09-20'),
            ('à partir de cette semaine je fais du sport', None),
            ('pour cette semaine et les suivantes', None),
        )
        for i, (brut, fin) in enumerate(cas):
            with self.subTest(brut=brut):
                RecurringBlock.objects.filter(user=self.user, title='Étude').delete()
                self.message_courant(brut)
                registre, tools = self.outils(brut, tache=f'u:{i}')
                self.appeler(tools, 'create_block', title='Étude', block_type='revision',
                             days=['lundi'], start_time='16:00', end_time='18:00')
                action = registre.actions[-1]
                self.assertTrue(action.succes)
                bloc = RecurringBlock.objects.get(user=self.user, title='Étude', active=True)
                if fin:
                    self.assertEqual(bloc.end_date, date.fromisoformat(fin))
                    self.assertEqual(bloc.start_date, AUJOURDHUI)
                    self.assertEqual(action.donnees['borne_auto'], {'end_date': fin})
                else:
                    self.assertIsNone(bloc.end_date)
                    self.assertNotIn('borne_auto', action.donnees)


class DonneesDesOutilsTests(HarnaisGardes, TransactionTestCase):

    def test_donnees_structurees_des_outils(self):
        calcul = self.bloc('Calcul différentiel', 0, '10:00', '11:50')

        r = execute_tool('create_block', self.user, {
            'title': 'Statistiques', 'block_type': 'course', 'days': [0],
            'start_time': '10:00', 'end_time': '12:00'})
        # Message capture sur le code d'avant le lot: octet pour octet.
        self.assertEqual(r.message, "0 bloc(s) créé(s): Statistiques (10:00-12:00). 1 sauté(s): "
                                    "Chevauchement avec 'Calcul différentiel' (10:00-11:50)")
        saut = r.data['skipped'][0]
        self.assertEqual(saut['motif'], 'chevauchement')
        self.assertEqual(saut['avec'], {'titre': 'Calcul différentiel', 'debut': '10:00', 'fin': '11:50'})
        self.assertEqual((saut['titre'], saut['debut'], saut['fin']), ('Statistiques', '10:00', '12:00'))
        self.assertEqual(saut['reason'], "Chevauchement avec 'Calcul différentiel' (10:00-11:50)")

        r = execute_tool('create_block', self.user, {
            'title': 'Calcul différentiel', 'block_type': 'course', 'days': [0],
            'start_time': '10:00', 'end_time': '11:50'})
        self.assertEqual(r.data['skipped'][0]['motif'], 'doublon')
        self.assertEqual(r.message, "0 bloc(s) créé(s): Calcul différentiel (10:00-11:50). 1 sauté(s): "
                                    "'Calcul différentiel' existe déjà le Lundi à 10:00 (aucun doublon créé)")

        r = execute_tool('create_block', self.user, {
            'title': 'Maths', 'block_type': 'course', 'days': [2],
            'start_time': '09:00', 'end_time': '12:00'})
        self.assertEqual(r.message, '1 bloc(s) créé(s): Maths (09:00-12:00) les Mercredi')
        self.assertEqual(r.data['skipped'], [])
        maths_id = r.data['created'][0]['id']

        r = execute_tool('update_block', self.user, {'block_id': calcul.id, 'end_time': '11:40'})
        self.assertTrue(r.success)
        self.assertEqual(r.message, "Bloc 'Calcul différentiel' mis à jour.")
        self.assertEqual(r.data['avant'], {'title': 'Calcul différentiel', 'day_of_week': 0,
                                           'start_time': '10:00', 'end_time': '11:50',
                                           'flexibility': 'fixed'})

        r = execute_tool('update_block', self.user, {
            'block_id': maths_id, 'day_of_week': 'lundi', 'start_time': '10:30', 'end_time': '11:30'})
        self.assertFalse(r.success)
        self.assertEqual(r.message, "Modification annulée: chevauchement avec 'Calcul différentiel' (10:00-11:40).")
        self.assertEqual(r.data['conflit'], {'titre': 'Calcul différentiel', 'debut': '10:00',
                                             'fin': '11:40', 'jour': 0})

        r = execute_tool('delete_block', self.user, {'block_id': maths_id})
        self.assertEqual(r.message, "Bloc 'Maths' supprimé.")
        self.assertEqual(r.data['deleted_id'], maths_id)
        self.assertEqual((r.data['block']['title'], r.data['block']['day_of_week']), ('Maths', 2))

        premiere = execute_tool('create_task', self.user, {'title': 'Lire'})
        seconde = execute_tool('create_task', self.user, {'title': 'Lire'})
        self.assertFalse(premiere.data['deja_presente'])
        self.assertTrue(seconde.data['deja_presente'])
        self.assertEqual(seconde.message, "Tâche 'Lire' déjà présente (non dupliquée).")

        r = execute_tool('delete_task', self.user, {'task_id': premiere.data['task']['id'], 'confirm': True})
        self.assertEqual(r.data['title'], 'Lire')

        r = execute_tool('schedule_task_at', self.user, {
            'title': 'Café', 'date': '2026-09-14', 'start_time': '10:30', 'end_time': '11:00'})
        self.assertEqual(r.data['conflict']['titre'], 'Calcul différentiel')
        self.assertFalse(r.data['conflict']['sommeil'])

        self.bloc('Sommeil', 2, '23:00', '07:00', block_type='sleep', flexibility='flexible')
        r = execute_tool('schedule_task_at', self.user, {
            'title': 'Série télé', 'date': '2026-09-16', 'start_time': '23:15', 'end_time': '23:45'})
        self.assertFalse(r.success)
        self.assertTrue(r.data['conflict']['sommeil'])
        self.assertIsNone(r.data['conflict']['titre'])

        r = execute_tool('get_week_schedule', self.user, {})
        lundi = r.data['days'][0]
        self.assertEqual(lundi['blocks'], ['Calcul différentiel (10:00-11:40)'])
        self.assertEqual(lundi['detail'], [{'title': 'Calcul différentiel', 'start_time': '10:00',
                                            'end_time': '11:40', 'block_type': 'course',
                                            'is_flexible': False}])

    def test_le_sommeil_reporte_protege_le_matin(self):
        """Couture avec b2: le sommeil reporte apres un quart de nuit refuse
        la fenetre qu'il occupe, comme le sommeil place."""
        with mock.patch('services.agent.tools.schedule.intervalles_sommeil_reporte',
                        return_value=[(120, 600)]):
            r = execute_tool('schedule_task_at', self.user, {
                'title': 'Café', 'date': '2026-09-18', 'start_time': '09:00', 'end_time': '09:30'})
        self.assertFalse(r.success)
        self.assertEqual(r.data['conflict'], {'start_time': '02:00', 'end_time': '10:00',
                                              'titre': None, 'sommeil': True})


class DescriptionsTests(SimpleTestCase):

    def test_descriptions_ne_poussent_plus_a_deviner(self):
        planifier = TOOL_MAP['schedule_task_at'].description
        creer = TOOL_MAP['create_block'].description
        libres = TOOL_MAP['find_free_slots'].description
        self.assertNotIn('au lieu de lui demander', planifier)
        self.assertNotIn('DÈS que', creer)
        self.assertNotIn("ne demande pas l'heure", libres)
        for description in (planifier, creer, libres):
            self.assertIn('rendez-vous', description)
            self.assertNotIn('\u2014', description)


class LectureDesReponsesTests(SimpleTestCase):
    """Les regles pures de demandes.py."""

    PORTEE = {'motif': 'portee_jour', 'cle': 'p',
              'options': [{'id': 'occurrence'}, {'id': 'serie'}, {'id': 'annuler'}]}
    DESTRUCTIF = {'motif': 'destructif', 'cle': 'd',
                  'options': [{'id': 'confirmer'}, {'id': 'annuler'}]}

    def test_cle_d_identite(self):
        self.assertEqual(dem.cle_demande('x', {'a': 1, 'b': 2}), dem.cle_demande('x', {'b': 2, 'a': 1}))
        self.assertNotEqual(dem.cle_demande('x', {'a': 1}), dem.cle_demande('y', {'a': 1}))
        self.assertEqual(len(dem.cle_demande('x', {})), 12)

    def test_oui_non(self):
        # Round 6 (D1): sans puce, aucun oui libre ne confirme; « non » ferme.
        cas = {'Oui, vas-y': None, 'oui': None, "d'accord": None,
               'c’est bon': None, 'Oui, je confirme.': None,
               'non merci': 'annuler', 'ok mais enlève aussi la chimie': None,
               'ok efface tout mon planning': None,
               'oui je pense que ce serait bien de le faire un jour': None}
        for brut, attendu in cas.items():
            with self.subTest(brut=brut):
                self.assertEqual(dem.option_choisie(brut, self.DESTRUCTIF), attendu)
        avec_puces = puces(dict(self.DESTRUCTIF, cible={}))
        for brut, attendu in {'Oui, je confirme.': 'confirmer', 'Oui, confirme': 'confirmer',
                              'Oui, vas-y': None, 'Non, ne change rien.': 'annuler'}.items():
            with self.subTest(brut=brut, puces=True):
                self.assertEqual(dem.option_choisie(brut, avec_puces), attendu)

    def test_portee(self):
        # Round 6 (D1): la portee ne se lit plus en texte libre; la puce seule.
        cas = {'Tous les jeudis (supprimer la série).': None, 'la série': None,
               "Seulement ce jeudi 17 sept. (sauter l'occurrence).": None,
               'juste celui-là': None, 'non': 'annuler', 'oui': None,
               'je ne veux pas tous les jeudis': None,
               'pas tous les jeudis, juste celui-là': None}
        for brut, attendu in cas.items():
            with self.subTest(brut=brut):
                self.assertEqual(dem.option_choisie(brut, self.PORTEE), attendu)
        avec_puces = puces(dict(self.PORTEE, cible={'jour': 3, 'date': '2026-09-17'}))
        for brut, attendu in {'Tous les jeudis (supprimer la série).': 'serie',
                              "Seulement ce jeudi 17 sept. (sauter l'occurrence).": 'occurrence',
                              'la série': None, 'juste celui-là': None}.items():
            with self.subTest(brut=brut, puces=True):
                self.assertEqual(dem.option_choisie(brut, avec_puces), attendu)

    def test_une_puce_ne_repond_qu_a_sa_demande(self):
        etrangere = dict(self.DESTRUCTIF, chips=[
            {'label': 'Tous les jeudis', 'value': 'Tous les jeudis (supprimer la série).',
             'option': 'serie'}])
        self.assertIsNone(dem.option_choisie('Tous les jeudis (supprimer la série).', etrangere))

    def test_heures_dites(self):
        cas = {'mets mon rendez-vous chez le dentiste jeudi à 10 h 30': ['10:30'],
               'ajoute Statistiques le lundi à 10 h': ['10:00'],
               '10h30 puis 14h': ['10:30', '14:00'], 'à 10:30': ['10:30'],
               'Va pour 11 h 50 à 12 h 50 jeu. 17 sept.': ['11:50', '12:50'],
               'planifie 2 h de lecture jeudi': [], 'pendant 1h30': [],
               'rien à cette heure-là': [],
               # Reponses de formulaire: une duree n'est pas une heure (banc s02-2).
               "Voici mes réponses :\nTemps d'étude total: 4 h\nJours: Mar, Mer, Sam, Dim": [],
               "Temps d'étude en plus: 4 h": [], 'Durée: 1 h': [], 'Durée : 1 h 30': [],
               'Voici mes réponses :\nJours de gym: Lundi, Mercredi, Vendredi\nDurée: 1 h': [],
               'Combien d\'heures: 3 h': [], '2 h par jour': [],
               'Voici mes réponses :\nDate du rendez-vous: 2026-09-17\n'
               'Heure du rendez-vous: 10:30 - 11:30': ['10:30', '11:30'],
               'Voici mes réponses :\nJours: Mardi, Jeudi\nHoraire: 16:00 - 17:50':
                   ['16:00', '17:50']}
        for brut, attendu in cas.items():
            with self.subTest(brut=brut):
                self.assertEqual(dem.heures_dites(brut), attendu)

    def test_jour_vise(self):
        for brut in ('efface tout jeudi', 'demain', 'le 17 septembre', 'tous les lundis',
                     "aujourd'hui"):
            with self.subTest(brut=brut):
                self.assertTrue(dem.jour_vise(brut))
        for brut in ('supprime mon cours de chimie', 'Oui, je confirme.'):
            with self.subTest(brut=brut):
                self.assertFalse(dem.jour_vise(brut))

    def test_date_visee(self):
        self.assertEqual(dem.date_visee('efface tout jeudi', 3, AUJOURDHUI), date(2026, 9, 17))
        self.assertEqual(dem.date_visee('le jeudi 24 septembre', 3, AUJOURDHUI), date(2026, 9, 24))
        # Une date qui ne tombe pas le jour du bloc est ignoree.
        self.assertEqual(dem.date_visee('demain', 3, AUJOURDHUI), date(2026, 9, 17))
        # Lundi = jour 0: aujourd'hui compte.
        self.assertEqual(dem.date_visee('ce lundi', 0, AUJOURDHUI), AUJOURDHUI)


class EcheanceSansJourTests(HarnaisGardes, TransactionTestCase):
    """Banc du 2026-09-14, round 2 (s09-1): « place ma revision de chimie 2h
    avant vendredi » a ete place aujourd'hui sans demander le jour."""

    def test_echeance_sans_jour_propose_les_jours_libres(self):
        brut = 'place ma revision de chimie 2h avant vendredi'
        self.message_courant(brut)
        registre, tools = self.outils(brut)
        self.appeler(tools, 'schedule_task_at', title='Révision de chimie', date='2026-09-14',
                     start_time='16:00', end_time='18:00')
        refus = registre.actions[-1]
        self.assertFalse(refus.succes)
        demande = refus.donnees['demande']
        self.assertEqual((demande['motif'], demande['source']), ('choix_modele', 'jours'))
        self.assertEqual(demande['question'], 'Quel jour veux-tu placer Révision de chimie ?')
        self.assertEqual([o['libelle'] for o in demande['options']],
                         ["Aujourd'hui", 'Demain', 'Mercredi 16 septembre', 'Jeudi 17 septembre'])
        self.assertEqual([o['cible']['date'] for o in demande['options']],
                         ['2026-09-14', '2026-09-15', '2026-09-16', '2026-09-17'])
        self.assertTrue(all(o['effet'] is None for o in demande['options']))
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 0)

        # Le tap sur une puce nomme le jour: l'ajout passe.
        valeur = demande['options'][2]['valeur']
        self.assertEqual(valeur, 'Place Révision de chimie mercredi 16 septembre.')
        self.message_courant(valeur)
        registre, tools = self.outils(valeur, tache='u:2')
        self.appeler(tools, 'schedule_task_at', title='Révision de chimie', date='2026-09-16',
                     start_time='16:00', end_time='18:00')
        self.assertTrue(registre.actions[-1].succes, registre.actions[-1].donnees)

    def test_un_jour_nomme_ou_sans_echeance_passe(self):
        for i, brut in enumerate(('place ma révision mercredi avant vendredi',
                                  "d'ici jeudi, mets ma lecture demain",
                                  'place ma révision mercredi')):
            with self.subTest(brut=brut):
                self.message_courant(brut)
                registre, tools = self.outils(brut, tache=f'u:{i}')
                self.appeler(tools, 'schedule_task_at', title=f'Révision {i}', date='2026-09-16',
                             start_time=f'{9 + 2 * i}:00', end_time=f'{10 + 2 * i}:00')
                self.assertTrue(registre.actions[-1].succes, registre.actions[-1].donnees)


class ContournementsDeLaRevueTests(HarnaisGardes, TransactionTestCase):
    """Les contournements prouves par la revue « gardes » du 2026-09-14."""

    def test_oui_avec_un_jour_ne_confirme_pas_la_serie(self):
        chimie = self.bloc('Chimie générale', 3, '13:00', '15:00')
        demande = puces(self.premier_tour('supprime mon cours de chimie', 'delete_block',
                                          block_id=chimie.id).donnees['demande'])
        self.assertEqual(demande['motif'], 'destructif')
        for brut in ('oui pour jeudi seulement', 'oui juste cette fois', 'oui le 17 sept.',
                     'ok demain'):
            with self.subTest(brut=brut):
                self.assertIsNone(dem.option_choisie(brut, demande))
        # Round 6 (D1): un oui poli ne confirme plus; la puce exacte, oui.
        for brut in ('oui merci', 'Oui, vas-y', "ok c'est bon"):
            with self.subTest(brut=brut):
                self.assertIsNone(dem.option_choisie(brut, demande))
        for brut in ('oui je confirme', 'Oui, je confirme.', 'Oui, confirme'):
            with self.subTest(brut=brut):
                self.assertEqual(dem.option_choisie(brut, demande), 'confirmer')
        brut = 'oui pour jeudi seulement'
        self.attendre([demande], brut)
        registre = Registre()
        outils_v2.appliquer_choix_en_attente(self.user, registre, brut, 'u:2')
        self.assertFalse(any(a.succes for a in registre.actions))
        self.assertActif(chimie)

    def test_une_fin_proche_ou_demandee_comme_suppression_est_retenue(self):
        chimie = self.bloc('Chimie générale', 1, '13:00', '15:00')
        cas = (('supprime mon cours de chimie', '2026-09-15'),
               ('enlève la chimie', '2026-12-20'),
               ('mon cours de chimie finit mercredi', '2026-09-16'))
        for i, (brut, fin) in enumerate(cas):
            with self.subTest(brut=brut):
                action = self.premier_tour(brut, 'update_block', tache=f'u:{i}',
                                           block_id=chimie.id, end_date=fin)
                self.assertFalse(action.succes)
                self.assertEqual(action.donnees['demande']['motif'], 'destructif')
                chimie.refresh_from_db()
                self.assertIsNone(chimie.end_date)
        action = self.premier_tour('efface la chimie', 'update_block', tache='u:9',
                                   block_id=chimie.id, start_date='2027-01-10')
        self.assertFalse(action.succes)
        chimie.refresh_from_db()
        self.assertIsNone(chimie.start_date)

    def test_heure_dite_sans_essai_jamais_changee(self):
        self.bloc('Calcul différentiel', 3, '10:00', '11:50')
        brut = 'mets mon rendez-vous chez le dentiste jeudi à 10 h 30'
        self.message_courant(brut)
        registre, tools = self.outils(brut)
        self.appeler(tools, 'find_free_slots', date='2026-09-17')
        self.appeler(tools, 'schedule_task_at', title='Dentiste', date='2026-09-17',
                     start_time='13:00', end_time='14:00')
        refus = registre.actions[-1]
        self.assertFalse(refus.succes)
        demande = refus.donnees['demande']
        self.assertEqual(demande['motif'], 'heure_refusee')
        self.assertEqual(demande['cible']['debut'], '10:30')
        self.assertTrue(any(o['id'].startswith('creneau_') for o in demande['options']))
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 0)

    def test_duree_de_formulaire_ne_retient_pas_les_ajouts(self):
        """Banc du 2026-09-14 (s02-2): « Temps d'étude total: 4 h » etait lu
        comme 04:00 et les quatre evenements d'etude etaient retenus."""
        cas = ("Voici mes réponses :\nTemps d'étude total: 4 h\nJours: Mar, Mer, Sam, Dim",
               'Voici mes réponses :\nDurée: 1 h\nÉtude: mardi')
        for i, brut in enumerate(cas):
            with self.subTest(brut=brut):
                self.message_courant(brut)
                registre, tools = self.outils(brut, tache=f'u:{i}')
                self.appeler(tools, 'schedule_task_at', title=f'Étude {i + 1}', date='2026-09-15',
                             start_time=f'{16 + i}:00', end_time=f'{17 + i}:00')
                self.assertTrue(registre.actions[-1].succes, registre.actions[-1].donnees)
                self.assertNotIn('heure_dite', registre.actions[-1].donnees)
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 2)

    def test_heure_dite_libre_mais_ignoree_est_reessayee(self):
        brut = 'ajoute le dentiste jeudi à 15 h'
        self.message_courant(brut)
        registre, tools = self.outils(brut)
        self.appeler(tools, 'schedule_task_at', title='Dentiste', date='2026-09-17',
                     start_time='13:00', end_time='14:00')
        self.assertFalse(registre.actions[-1].succes)
        self.assertEqual(registre.actions[-1].donnees['heure_dite'], '15:00')
        self.appeler(tools, 'schedule_task_at', title='Dentiste', date='2026-09-17',
                     start_time='15:00', end_time='16:00')
        self.assertTrue(registre.actions[-1].succes)

    def test_une_borne_ou_l_heure_d_un_autre_element_ne_bloque_pas(self):
        cas = (('place ma révision jeudi avant 10 h', 'Révision', '08:00', '09:30'),
               ("j'ai un quart jeudi à 19 h, ajoute mon étude jeudi", 'Étude', '13:00', '14:00'))
        for i, (brut, titre, debut, fin) in enumerate(cas):
            with self.subTest(brut=brut):
                self.message_courant(brut)
                registre, tools = self.outils(brut, tache=f'u:{i}')
                self.appeler(tools, 'schedule_task_at', title=titre, date='2026-09-17',
                             start_time=debut, end_time=fin)
                self.assertTrue(registre.actions[-1].succes, registre.actions[-1].message)

    def test_une_date_passee_est_refusee(self):
        brut = 'place ma révision jeudi'
        self.message_courant(brut)
        registre, tools = self.outils(brut)
        self.appeler(tools, 'schedule_task_at', title='Révision', date='2025-09-18',
                     start_time='14:00', end_time='15:00')
        refus = registre.actions[-1]
        self.assertFalse(refus.succes)
        self.assertEqual(refus.donnees['date_passee'], '2025-09-18')
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 0)

    def test_la_question_des_ajouts_ne_nomme_que_le_retenu(self):
        brut = 'ajoute mes cours du matin'
        self.message_courant(brut)
        registre, tools = self.outils(brut)
        self.appeler(tools, 'create_block', title='Cours du matin', block_type='course',
                     days=['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi'],
                     start_time='08:00', end_time='09:00')
        self.appeler(tools, 'schedule_task_at', title='Lecture', date='2026-09-19',
                     start_time='14:00', end_time='15:00')
        cible = registre.actions[-1].donnees['demande']['cible']
        self.assertEqual(cible['titres'], ['Lecture'])
        self.assertEqual(cible['crees'], ['Cours du matin'])

    def test_heure_refusee_d_un_bloc_recurrent_est_marquee(self):
        self.bloc('Calcul différentiel', 2, '10:00', '11:50')
        brut = 'ajoute Statistiques le mercredi à 10 h'
        self.message_courant(brut)
        registre, tools = self.outils(brut)
        self.appeler(tools, 'create_block', title='Statistiques', block_type='course',
                     days=['mercredi'], start_time='10:00', end_time='12:00')
        demande = registre.actions[-1].donnees['demande']
        self.assertEqual(demande['motif'], 'heure_refusee')
        self.assertTrue(demande['cible']['recurrent'])
        self.assertTrue(all(o['cible'].get('recurrent') for o in demande['options']))
