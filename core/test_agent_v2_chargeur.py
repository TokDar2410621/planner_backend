"""Le chargeur d'outils: chercher un outil rare, puis l'appeler.

Mesure du 2026-10-01 sur 600 tours de production: 75 % des tours n'appellent
AUCUN outil, et les 33 outils pesaient 10 400 jetons a chaque tour. Dix-sept
d'entre eux passent derriere deux outils, neuf n'ayant jamais servi sur
l'echantillon.

Le test qui compte est le dernier: un appel INDIRECT doit rencontrer les memes
gardes qu'un appel direct. Un chemin parallele supprimerait un bloc sans
confirmation, et c'est le seul vrai risque de ce mecanisme.
"""
import asyncio
import json

from django.contrib.auth.models import User
from django.test import TransactionTestCase
from django.utils import timezone

from core.models import RecurringBlock, Task
from core.test_agent_v2_jugement import juger_script
from services.agent.tools import ALL_TOOLS
from services.agent_v2 import chargeur
from services.agent_v2.outils import (NOM_APPEL, NOM_CHERCHEUR, OUTILS_EXPOSES,
                                      OUTILS_QUESTION_ANCIENS,
                                      description_v2, outils_pour_le_modele,
                                      schema_expose)
from services.agent_v2.redaction import question_code
from services.agent_v2.registre import Registre


class ExpositionTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='chargeur', password='x')

    def _outils(self, message=''):
        registre = Registre()
        return registre, {t.name: t for t in outils_pour_le_modele(
            self.user, registre, message_du_tour=message, tache='test:chargeur',
            message_brut=message)}

    def test_seuls_les_outils_du_quotidien_et_les_deux_meta_sont_exposes(self):
        _, outils = self._outils()
        attendus = set(OUTILS_EXPOSES) | {NOM_CHERCHEUR, NOM_APPEL}
        self.assertEqual(set(outils), attendus)

    def test_le_contexte_envoye_au_modele_a_bien_maigri(self):
        _, outils = self._outils()
        apres = sum(len(t.description)
                    + len(json.dumps(schema_expose(t), ensure_ascii=False))
                    for t in outils.values())
        avant = sum(len(description_v2(t))
                    + len(json.dumps(t.PARAMETERS, ensure_ascii=False))
                    for t in ALL_TOOLS)
        self.assertLess(apres, avant)
        # Mesure du 2026-10-01: 31 105 car -> 22 597, soit 27 %. On verrouille
        # un gain d'au moins 20 %, pour laisser respirer les descriptions.
        self.assertGreater((avant - apres) / avant, 0.20)

    def test_deux_suppressions_ciblees_sont_directes_et_groupables(self):
        """Une demande courante ne doit pas dependre du chargeur rare.

        Cas de production du 2026-10-13: « supprime toutes les recontres avec
        Camille » a bien lu les taches, puis a epuise la boucle sans jamais
        atteindre delete_task. Deux cibles exactes doivent produire une seule
        confirmation groupee, sans rien supprimer au premier tour.
        """
        message = "Parfait supprime toutes les recontres avec Camille"
        registre, outils = self._outils(message)
        self.assertIn('delete_task', outils)
        self.assertIn('delete_block', outils)

        for index in range(60):
            Task.objects.create(user=self.user, title=f'Tache historique {index}')
        disponibilite = Task.objects.create(
            user=self.user, title='Rencontre (disponibilite de Camille)')
        rendez_vous = Task.objects.create(
            user=self.user, title='Rendez-vous avec Camille')
        asyncio.run(outils['list_tasks'].function_schema.function(query='Camille'))
        trouves = {t['id'] for t in registre.actions[-1].donnees['tasks']}
        self.assertEqual(trouves, {disponibilite.pk, rendez_vous.pk})

        for tache in (disponibilite, rendez_vous):
            asyncio.run(outils['delete_task'].function_schema.function(
                task_id=tache.pk, confirm=True))

        demandes = [(a.donnees or {}).get('demande') for a in registre.actions]
        demandes = [d for d in demandes if isinstance(d, dict)]
        question, chips, cles = question_code(demandes)
        self.assertIn(disponibilite.title, question)
        self.assertIn(rendez_vous.title, question)
        self.assertEqual([c['option'] for c in chips], ['confirmer', 'annuler'])
        self.assertEqual(len(cles), 2)
        self.assertTrue(Task.objects.filter(pk=disponibilite.pk).exists())
        self.assertTrue(Task.objects.filter(pk=rendez_vous.pk).exists())

    def test_chaque_outil_est_soit_expose_soit_joignable(self):
        """Aucun outil ne doit devenir inatteignable."""
        _, outils = self._outils()
        caches = [o for o in ALL_TOOLS if o.NAME not in OUTILS_EXPOSES
                  and o.NAME not in OUTILS_QUESTION_ANCIENS]
        self.assertTrue(caches)
        for outil in caches:
            with self.subTest(outil=outil.NAME):
                self.assertNotIn(outil.NAME, outils)
                # Joignable: son nom ressort d'une recherche sur son nom.
                trouve = chargeur.chercher(outil.NAME.replace('_', ' '),
                                           caches, description_v2)
                self.assertIn(outil.NAME, trouve)


class RechercheTests(TransactionTestCase):
    def setUp(self):
        self.caches = [o for o in ALL_TOOLS if o.NAME not in OUTILS_EXPOSES
                       and o.NAME not in OUTILS_QUESTION_ANCIENS]

    def _noms(self, besoin):
        rendu = chargeur.chercher(besoin, self.caches, description_v2)
        return [l.strip() for l in rendu.splitlines()
                if l.strip() and ' ' not in l.strip()]

    def test_le_bon_outil_arrive_en_premier(self):
        for besoin, attendu in (
                ('envoyer une notification', 'send_notification'),
                ('reorganiser ma journee', 'organize_day'),
                ('vider tout mon planning', 'clear_all_blocks'),
                ('mes objectifs', 'list_goals'),
        ):
            with self.subTest(besoin=besoin):
                self.assertEqual(self._noms(besoin)[:1], [attendu])

    def test_un_besoin_sans_reponse_rend_la_liste_plutot_que_rien(self):
        """Le modele ne doit jamais rester bloque."""
        rendu = chargeur.chercher('xyzzy', self.caches, description_v2)
        self.assertIn('Aucun outil', rendu)
        self.assertIn('clear_all_blocks', rendu)

    def test_un_besoin_vide_rend_la_liste(self):
        rendu = chargeur.chercher('', self.caches, description_v2)
        self.assertIn('clear_all_blocks', rendu)

    def test_le_schema_rendu_nomme_les_parametres_requis(self):
        rendu = chargeur.chercher('modifier une tache', self.caches, description_v2)
        self.assertIn('task_id', rendu)
        self.assertIn('requis', rendu)


class AppelIndirectTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='appel', password='x')
        self.jour = timezone.localdate()
        self.cours = RecurringBlock.objects.create(
            user=self.user, title='Gym', block_type='sport',
            day_of_week=self.jour.weekday(), start_time='18:00', end_time='19:00')

    def _appeler(self, nom, parametres, message='', juge=None):
        from unittest.mock import patch
        registre = Registre()
        with patch('services.agent_v2.jugement.juger', juge or juger_script({})):
            outils = {t.name: t for t in outils_pour_le_modele(
                self.user, registre, message_du_tour=message, tache='test:appel',
                message_brut=message)}
            sortie = asyncio.run(outils[NOM_APPEL].function_schema.function(
                nom=nom, parametres=parametres))
        return registre, sortie

    def test_un_outil_cache_s_execute(self):
        tache = Task.objects.create(user=self.user, title='Lire')
        registre, _ = self._appeler('complete_task', {'task_id': tache.id}) \
            if 'complete_task' not in OUTILS_EXPOSES else (None, None)
        if registre is None:  # complete_task est expose: on prend update_task
            tache2 = Task.objects.create(user=self.user, title='Autre')
            registre, _ = self._appeler('update_task',
                                        {'task_id': tache2.id, 'title': 'Renommee'})
            tache2.refresh_from_db()
            self.assertEqual(tache2.title, 'Renommee')
        self.assertTrue(registre.actions)
        self.assertTrue(registre.actions[-1].succes, registre.actions[-1].message)

    def test_l_action_entre_au_registre_comme_un_appel_direct(self):
        tache = Task.objects.create(user=self.user, title='A renommer')
        registre, _ = self._appeler('update_task',
                                    {'task_id': tache.id, 'title': 'Renommee'})
        self.assertEqual(registre.actions[-1].outil, 'update_task')
        self.assertTrue(registre.actions[-1].est_mutation)
        self.assertEqual(registre.recus_confirmes(), [])
        from services.agent_v2.reconciliation import reconcilier
        reconcilier(self.user, registre)
        self.assertEqual(len(registre.recus_confirmes()), 1)

    def test_un_outil_deja_expose_est_refuse(self):
        _, sortie = self._appeler('get_today_schedule', {})
        self.assertIn('directement', sortie)

    def test_un_outil_inconnu_est_refuse_avec_la_liste(self):
        _, sortie = self._appeler('outil_imaginaire', {})
        self.assertIn('inconnu', sortie)
        self.assertIn('clear_all_blocks', sortie)

    def test_un_parametre_requis_manquant_est_refuse_avant_execution(self):
        registre, sortie = self._appeler('update_task', {})
        self.assertIn('task_id', sortie)
        self.assertFalse(registre.actions)

    def test_des_parametres_illisibles_sont_refuses(self):
        registre, sortie = self._appeler('update_task', 'pas du json')
        self.assertIn('illisibles', sortie)
        self.assertFalse(registre.actions)

    def test_des_parametres_en_chaine_json_sont_acceptes(self):
        """Le modele envoie parfois la chaine au lieu de l'objet."""
        tache = Task.objects.create(user=self.user, title='Chaine')
        registre, _ = self._appeler(
            'update_task', json.dumps({'task_id': tache.id, 'title': 'Via chaine'}))
        tache.refresh_from_db()
        self.assertEqual(tache.title, 'Via chaine')
        self.assertTrue(registre.actions[-1].succes)

    # ── le test qui compte ────────────────────────────────────────────────

    def test_la_garde_destructive_tient_a_travers_le_chargeur(self):
        """Un appel INDIRECT rencontre la meme garde qu'un appel direct.

        Sans cela, le chargeur serait un chemin parallele qui supprime un bloc
        recurrent sans confirmation. C'est le seul vrai risque du mecanisme.
        """
        registre, sortie = self._appeler('clear_all_blocks', {'confirm': True},
                                         message='vide tout mon planning')
        # Rien n'est supprime: le code retient et pose sa question.
        self.cours.refresh_from_db()
        self.assertTrue(self.cours.active)
        action = registre.actions[-1]
        self.assertFalse(action.succes)
        demande = (action.donnees or {}).get('demande') or {}
        self.assertIn(demande.get('motif'), ('destructif', 'portee_jour'))
        self.assertTrue(demande.get('options'))

    def test_vider_le_planning_passe_aussi_par_sa_garde(self):
        registre, _ = self._appeler('clear_all_blocks', {'confirm': True},
                                    message='efface tout')
        self.cours.refresh_from_db()
        self.assertTrue(self.cours.active)
        demande = (registre.actions[-1].donnees or {}).get('demande') or {}
        self.assertEqual(demande.get('motif'), 'destructif')
