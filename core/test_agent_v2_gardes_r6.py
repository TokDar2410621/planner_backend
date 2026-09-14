"""
Gardes du code, round 6: moins de surface de lecture.

Diagnostic des rounds 1 a 5: chaque revue trouvait une nouvelle tournure
francaise mal lue par les lecteurs de reponse libre (oui en tete, lecteur de
portee, reponse nue) ou par le repli « une seule heure ferme dans le message ».
Rafistoler les regex ouvrait de nouveaux trous. Ce round retire la surface.

D1  Une suppression ne s'execute et ne s'autorise que sur la puce exacte de la
    demande en attente (egalite normalisee avec sa valeur ou son libelle).
    Aucune reponse libre ne produit « serie », « occurrence » ou « confirmer ».
    Une reponse libre qui ne dit que garder ou annuler ferme la demande.
D2  Sans option, la demande gardee est reposee UNE fois, avec sa date
    d'origine, si le message ne porte pas de nouvelle requete. Ensuite, ou sur
    une nouvelle requete, elle est abandonnee et consignee comme telle.
D3  Plus de repli « une seule heure ferme »: la garde d'heure dite ne vaut que
    si le titre de l'appel est dans une proposition qui porte sa propre heure.
D6  tour_entierement_decide_par_le_code dit quand AGIR peut etre saute.

Horloge du harnais: lundi 14 septembre 2026, 8 h (jour 0 = lundi).
"""
from datetime import timedelta

from django.test import SimpleTestCase, TransactionTestCase
from django.utils import timezone

from core.models import (ConversationMessage, RecurringBlock,
                         RecurringBlockException, ScheduledBlock, Task)
from core.test_agent_v2_gardes import AUJOURDHUI, HarnaisGardes, puces
from services.agent_v2 import demandes as dem
from services.agent_v2 import outils as outils_v2
from services.agent_v2.registre import Registre

# Demandes pures, avec les puces exactes que rendu.py produit.
PORTEE = puces({'motif': 'portee_jour', 'cle': 'p', 'outil': 'delete_block',
                'cible': {'titre': 'Quart au dépanneur', 'jour': 3, 'date': '2026-09-17'},
                'options': [{'id': 'occurrence'}, {'id': 'serie'}, {'id': 'annuler'}]})
DESTR = puces({'motif': 'destructif', 'cle': 'd', 'outil': 'delete_block',
               'cible': {'titre': 'Quart au dépanneur', 'jour': 3},
               'options': [{'id': 'confirmer'}, {'id': 'annuler'}]})
ANNULER_EVENEMENT = puces({'motif': 'destructif', 'cle': 'c', 'outil': 'cancel_scheduled_block',
                           'cible': {'titre': 'Dentiste', 'date': '2026-09-16'},
                           'options': [{'id': 'confirmer'}, {'id': 'annuler'}]})
MASSE = puces({'motif': 'creation_en_masse', 'cle': 'creation_en_masse', 'outil': 'create_block',
               'cible': {'titre': 'Yoga'},
               'options': [{'id': 'confirmer'}, {'id': 'annuler'}]})
PLAN = puces({'motif': 'optimisation', 'cle': 'optimize_week:apply', 'outil': 'optimize_week',
              'cible': {}, 'options': [{'id': 'confirmer'}, {'id': 'annuler'}]})


# ── D1: la puce exacte, et rien d'autre, tranche une suppression ────────────


class D1PuceExacteTests(SimpleTestCase):

    def test_la_puce_exacte_tranche_modulo_casse_accents_espaces_ponctuation(self):
        cas = {
            'Tous les jeudis': 'serie', 'tous les jeudis': 'serie', 'TOUS LES JEUDIS': 'serie',
            '  Tous   les jeudis  ': 'serie', 'Tous les jeudis.': 'serie',
            'Tous les jeudis (supprimer la série).': 'serie',
            'tous les jeudis (supprimer la serie)': 'serie',
            'Seulement ce jeudi': 'occurrence',
            "Seulement ce jeudi 17 sept. (sauter l'occurrence).": 'occurrence',
            'Non, garde tout': 'annuler', 'Non, ne change rien.': 'annuler',
        }
        for brut, attendu in cas.items():
            with self.subTest(brut=brut):
                self.assertEqual(dem.option_choisie(brut, PORTEE), attendu)
        self.assertEqual(dem.option_choisie('Oui, confirme', DESTR), 'confirmer')
        self.assertEqual(dem.option_choisie('oui je confirme', DESTR), 'confirmer')
        self.assertEqual(dem.option_choisie('Oui, continue les ajouts.', MASSE), 'confirmer')
        self.assertEqual(dem.option_choisie('Applique le plan', PLAN), 'confirmer')

    def test_aucune_reponse_libre_ne_produit_une_suppression(self):
        """Les tournures que les rounds 1 a 5 lisaient comme serie, occurrence
        ou confirmation: plus aucune ne tranche."""
        libres = (
            'oui', 'Oui, vas-y', "d'accord", 'ok', "c'est bon", 'go', 'vas-y', 'oui merci',
            'la série', 'oui, tous les jeudis', 'supprime tous les jeudis',
            'enlève le quart tous les jeudis', 'efface-le définitivement',
            'enlève-le chaque semaine', 'juste celui-là', 'oui mais seulement ce jeudi',
            'seulement ce jeudi svp', 'tous les jeudis stp', 'Tous les jeudis ?',
            'Oui, supprime ces trois blocs.', 'oui pour jeudi seulement', 'supprime-le',
            'oui, supprime la série', 'tous', 'chaque jeudi', 'jeudi seulement',
        )
        for demande in (PORTEE, DESTR, MASSE, PLAN):
            for brut in libres:
                with self.subTest(motif=demande['motif'], brut=brut):
                    self.assertNotIn(dem.option_choisie(brut, demande),
                                     ('serie', 'occurrence', 'confirmer'))

    def test_une_demande_sans_puces_ne_tranche_rien_de_destructif(self):
        nue = {k: v for k, v in PORTEE.items() if k != 'chips'}
        for brut in ('Tous les jeudis', 'Tous les jeudis (supprimer la série).', 'oui'):
            with self.subTest(brut=brut):
                self.assertIsNone(dem.option_choisie(brut, nue))

    def test_une_puce_ne_repond_qu_a_sa_demande(self):
        self.assertIsNone(dem.option_choisie('Tous les jeudis (supprimer la série).', DESTR))
        self.assertIsNone(dem.option_choisie('Oui, continue les ajouts.', DESTR))

    def test_garder_ou_annuler_en_texte_libre_ferme_la_demande(self):
        garder = (
            'non', 'Non.', 'non merci', 'garde-le', 'Non, garde tous les jeudis.',
            'garde tous les jeudis', 'laisse-le', 'laisse tomber', 'annule', 'finalement non',
            'finalement, garde tous les jeudis', 'conserve la série', 'laisse-le toujours',
            'Je veux la garder chaque semaine', 'non non garde tous les jeudis je te dis',
            'ne touche pas à tous les jeudis', 'ne change rien', "n'efface rien",
            "n'efface rien, tous les jeudis restent", 'garde le quart', "j'ai changé d'avis, garde-le",
            'annule, laisse tous les jeudis',
        )
        for demande in (PORTEE, DESTR):
            for brut in garder:
                with self.subTest(motif=demande['motif'], brut=brut):
                    self.assertEqual(dem.option_choisie(brut, demande), 'annuler')
        self.assertEqual(dem.option_choisie('non', MASSE), 'annuler')
        self.assertEqual(dem.option_choisie('non, arrête', MASSE), 'annuler')
        self.assertEqual(dem.option_choisie('non', PLAN), 'annuler')

    def test_garder_ambigu_ne_tranche_rien(self):
        ambigus = (
            'non pas tous les jeudis', 'non, seulement ce jeudi', 'laisse ce jeudi',
            'garde celui-là', 'non, plutôt vendredi', 'non, mets-le vendredi',
            'laisse tomber ce cours, tous les jeudis', 'laisse tomber le quart',
            'oui, annule', 'non ?', 'garde-le ?', 'non mais supprime la chimie',
            'garde le quart mais efface le gym', 'change rien mais enlève le gym',
            'je sais pas', 'pas tous les jeudis',
        )
        for brut in ambigus:
            with self.subTest(brut=brut):
                self.assertIsNone(dem.option_choisie(brut, PORTEE))

    def test_annule_ne_garde_pas_un_evenement_qu_on_propose_d_annuler(self):
        """« J'annule le dentiste ? » « annule »: c'est un oui, pas un refus. Ni
        l'un ni l'autre n'est lu en texte libre."""
        for brut in ('annule', 'annule-le', 'oui annule'):
            with self.subTest(brut=brut):
                self.assertIsNone(dem.option_choisie(brut, ANNULER_EVENEMENT))
        self.assertEqual(dem.option_choisie('non, garde-le', ANNULER_EVENEMENT), 'annuler')
        self.assertEqual(dem.option_choisie('Oui, je confirme.', ANNULER_EVENEMENT), 'confirmer')


class D1BoutEnBoutTests(HarnaisGardes, TransactionTestCase):

    def test_une_reponse_libre_n_execute_ni_n_autorise_rien(self):
        gym = self.bloc('Gym', 3, '17:00', '18:00', block_type='sport')
        for i, brut in enumerate(('supprime tous les jeudis', 'oui, tous les jeudis', 'la série',
                                  'juste celui-là', 'enlève le quart tous les jeudis')):
            with self.subTest(brut=brut):
                demande = puces(self.demande_portee(tache=f'r:{i}'))
                self.attendre([demande], brut)
                registre = Registre()
                outils_v2.appliquer_choix_en_attente(self.user, registre, brut, f'e:{i}')
                self.assertFalse(any(a.succes for a in registre.actions))
                # Le modele qui tente la suppression ou le saut est retenu.
                _, tools = self.outils(brut, registre=registre, tache=f'e:{i}')
                self.appeler(tools, 'delete_block', block_id=self.q.id)
                self.appeler(tools, 'skip_block_occurrence', date='2026-09-17',
                             title='Quart au dépanneur', block_type='work')
                self.assertFalse(any(a.succes for a in registre.actions))
                self.assertActif(self.q)
                self.assertActif(gym)
                self.assertFalse(RecurringBlockException.objects.exists())

    def test_oui_libre_ne_confirme_plus_une_suppression_de_tache(self):
        tache = Task.objects.create(user=self.user, title='Rapport de labo')
        demande = puces(self.premier_tour('supprime la tâche rapport de labo', 'delete_task',
                                          task_id=tache.id, confirm=False).donnees['demande'])
        for i, brut in enumerate(('Oui, vas-y', 'oui', "d'accord")):
            with self.subTest(brut=brut):
                self.attendre([demande], brut)
                registre, tools = self.outils(brut, tache=f'o:{i}')
                outils_v2.appliquer_choix_en_attente(self.user, registre, brut, f'o:{i}')
                self.appeler(tools, 'delete_task', task_id=tache.id, confirm=True)
                self.assertFalse(any(a.succes for a in registre.actions))
                self.assertTrue(Task.objects.filter(id=tache.id).exists())
        # La puce exacte, elle, execute.
        self.attendre([demande], 'Oui, je confirme.')
        registre = Registre()
        outils_v2.appliquer_choix_en_attente(self.user, registre, 'Oui, je confirme.', 'o:t')
        self.assertFalse(Task.objects.filter(id=tache.id).exists())

    def test_garder_en_texte_libre_ferme_et_retient_le_modele(self):
        demande = puces(self.demande_portee())
        brut = 'Non, garde tous les jeudis.'
        self.attendre([demande], brut)
        registre = Registre()
        sorties = outils_v2.appliquer_choix_en_attente(self.user, registre, brut, 'g:1')
        self.assertEqual(sorties[0]['option'], 'annuler')
        _, tools = self.outils(brut, registre=registre, tache='g:1')
        retour = self.appeler(tools, 'delete_block', block_id=self.q.id)
        self.assertEqual(retour, outils_v2.MESSAGE_DEJA_TRANCHE)
        # Seule la decision « annulee » est au registre, et ce n'est pas une mutation.
        self.assertEqual([(a.outil, a.donnees.get('decision_code')) for a in registre.actions],
                         [(outils_v2.OUTIL_DECISION, 'annulee')])
        self.assertFalse(any(a.succes and a.est_mutation for a in registre.actions))
        self.assertActif(self.q)


# ── D2: reposee une fois, puis abandonnee ───────────────────────────────────


class D2ReposeeUneFoisTests(HarnaisGardes, TransactionTestCase):

    def _decisions(self, registre):
        return [a for a in registre.actions if (a.donnees or {}).get('decision_code')]

    def _persister(self, u_precedent, demandes):
        from services.agent_v2.agent import PlannerAgentV2

        question = PlannerAgentV2._question_des_demandes(demandes)
        ConversationMessage.objects.create(
            user=self.user, role='assistant', content=question['question'],
            metadata={'en_reponse_a': u_precedent.pk, 'demandes': question['demandes']})

    def test_vague_reposee_puis_vague_abandonnee(self):
        demande = puces(self.demande_portee())
        origine = (timezone.now() - timedelta(minutes=10)).isoformat()
        demande['emise_le'] = origine
        u2 = self.attendre([demande], 'oui')

        registre = Registre()
        outils_v2.appliquer_choix_en_attente(self.user, registre, 'oui', 'v:1')
        [reposee] = self._decisions(registre)
        self.assertFalse(reposee.succes)
        self.assertEqual(reposee.donnees['decision_code'], 'reposee')
        self.assertTrue(reposee.donnees['reposee_par_le_code'])
        self.assertEqual(reposee.donnees['demande']['cle'], demande['cle'])
        self.assertEqual(reposee.donnees['demande']['emise_le'], origine)
        self.assertEqual(reposee.donnees['demande']['reemissions'], 1)
        self.assertNotIn('chips', reposee.donnees['demande'])
        self.assertActif(self.q)

        self._persister(u2, [reposee.donnees['demande']])
        self.message_courant('oui oui')
        registre = Registre()
        outils_v2.appliquer_choix_en_attente(self.user, registre, 'oui oui', 'v:2')
        [abandon] = self._decisions(registre)
        self.assertFalse(abandon.succes)
        self.assertFalse(abandon.est_mutation)
        self.assertEqual(abandon.donnees['decision_code'], 'abandonnee')
        self.assertTrue(abandon.donnees['abandonnee_par_le_code'])
        self.assertEqual(abandon.donnees['demande']['cle'], demande['cle'])
        self.assertEqual(abandon.donnees['demande']['cible']['titre'], 'Quart au dépanneur')
        self.assertNotIn('reposee_par_le_code', abandon.donnees)
        self.assertFalse(any(a.succes for a in registre.actions))
        self.assertActif(self.q)

    def test_la_puce_exacte_apres_la_question_reposee_execute(self):
        demande = puces(self.demande_portee())
        u2 = self.attendre([demande], 'Oui, supprime-le.')
        registre = Registre()
        outils_v2.appliquer_choix_en_attente(self.user, registre, 'Oui, supprime-le.', 'x:1')
        [reposee] = self._decisions(registre)
        self._persister(u2, [reposee.donnees['demande']])
        self.message_courant('Tous les jeudis')
        registre = Registre()
        outils_v2.appliquer_choix_en_attente(self.user, registre, 'Tous les jeudis', 'x:2')
        [execute] = self._decisions(registre)
        self.assertTrue(execute.succes)
        self.assertEqual(execute.outil, 'delete_block')
        self.assertEqual(execute.donnees['decision_code'], 'execute')
        self.assertActif(self.q, False)

    def test_une_nouvelle_requete_abandonne_sans_reposer(self):
        for i, brut in enumerate(("c'est quoi mon horaire demain ?", 'ajoute gym demain à 18 h',
                                  'merci, bonne nuit', 'supprime mon gym tous les jeudis')):
            with self.subTest(brut=brut):
                demande = puces(self.demande_portee(tache=f'n:{i}'))
                self.attendre([demande], brut)
                registre = Registre()
                outils_v2.appliquer_choix_en_attente(self.user, registre, brut, f'n:{i}')
                decisions = self._decisions(registre)
                self.assertEqual([a.donnees['decision_code'] for a in decisions], ['abandonnee'])
                self.assertFalse(any((a.donnees or {}).get('reposee_par_le_code')
                                     for a in registre.actions))
                self.assertActif(self.q)

    def test_annulee_consignee_sans_question(self):
        demande = puces(self.demande_portee())
        for i, brut in enumerate(('Non, ne change rien.', 'laisse tomber')):
            with self.subTest(brut=brut):
                self.attendre([demande], brut)
                registre = Registre()
                outils_v2.appliquer_choix_en_attente(self.user, registre, brut, f'a:{i}')
                [annulee] = self._decisions(registre)
                self.assertEqual(annulee.donnees['decision_code'], 'annulee')
                self.assertNotIn('demande', annulee.donnees)
                self.assertFalse(annulee.est_mutation)
                self.assertActif(self.q)

    def test_apres_abandon_le_modele_pose_une_demande_neuve(self):
        demande = puces(self.demande_portee())
        demande['emise_le'] = (timezone.now() - timedelta(minutes=20)).isoformat()
        demande['reemissions'] = 1
        brut = 'enlève le quart jeudi'
        self.attendre([demande], brut)
        registre = Registre()
        outils_v2.appliquer_choix_en_attente(self.user, registre, brut, 'm:1')
        self.assertEqual([a.donnees['decision_code'] for a in self._decisions(registre)],
                         ['abandonnee'])
        _, tools = self.outils(brut, registre=registre, tache='m:1')
        self.appeler(tools, 'delete_block', block_id=self.q.id)
        neuve = registre.actions[-1].donnees['demande']
        self.assertFalse(registre.actions[-1].succes)
        self.assertNotEqual(neuve['emise_le'], demande['emise_le'])
        self.assertFalse(neuve.get('reemissions'))
        self.assertActif(self.q)


# ── D6: le tour entierement decide par le code ──────────────────────────────


class D6TourDecideParLeCodeTests(HarnaisGardes, TransactionTestCase):

    def _decide(self, demandes, brut, tache):
        self.attendre(demandes, brut)
        registre = Registre()
        outils_v2.appliquer_choix_en_attente(self.user, registre, brut, tache)
        return outils_v2.tour_entierement_decide_par_le_code(registre, brut)

    def test_decide_par_le_code(self):
        cas = ('Tous les jeudis (supprimer la série).', 'Non, ne change rien.', 'non merci',
               'oui')
        for i, brut in enumerate(cas):
            with self.subTest(brut=brut):
                RecurringBlock.all_objects.filter(pk=self.q.pk).update(active=True)
                demande = puces(self.demande_portee(tache=f'd:{i}'))
                self.assertTrue(self._decide([demande], brut, f'd:{i}'))

    def test_seconde_reponse_vague_decidee(self):
        demande = puces(self.demande_portee())
        demande['reemissions'] = 1
        self.assertTrue(self._decide([demande], 'oui oui', 's:1'))

    def test_pas_decide(self):
        demande = puces(self.demande_portee())
        # Une nouvelle requete: AGIR doit la servir.
        self.assertFalse(self._decide([demande], "c'est quoi mon horaire demain ?", 'p:1'))
        self.assertFalse(self._decide([demande], 'ajoute gym demain à 18 h', 'p:2'))
        # Aucune demande en attente.
        self.message_courant('Tous les jeudis')
        registre = Registre()
        outils_v2.appliquer_choix_en_attente(self.user, registre, 'Tous les jeudis', 'p:3')
        self.assertFalse(outils_v2.tour_entierement_decide_par_le_code(registre, 'Tous les jeudis'))
        self.assertFalse(outils_v2.tour_entierement_decide_par_le_code(Registre(), 'oui'))

    def test_une_puce_qui_demande_une_suite_au_modele_n_est_pas_decidee(self):
        masse = puces({'type': 'confirmation', 'motif': 'creation_en_masse',
                       'cle': 'creation_en_masse', 'outil': 'create_block', 'parametres': {},
                       'cible': {'titre': 'Yoga'},
                       'options': [{'id': 'confirmer', 'effet': None},
                                   {'id': 'annuler', 'effet': None}],
                       'emise_le': timezone.now().isoformat()})
        self.assertFalse(self._decide([masse], 'Oui, continue les ajouts.', 'c:1'))
        creneau = {
            'type': 'choix', 'motif': 'heure_refusee', 'cle': 'h1', 'outil': 'schedule_task_at',
            'parametres': {}, 'cible': {'titre': 'Dentiste', 'date': '2026-09-17'},
            'options': [{'id': 'creneau_1', 'effet': None,
                         'cible': {'titre': 'Dentiste', 'date': '2026-09-17',
                                   'debut': '11:50', 'fin': '12:50'}}],
            'chips': [{'label': '11 h 50 à 12 h 50',
                       'value': 'Va pour 11 h 50 à 12 h 50 jeu. 17 sept.', 'option': 'creneau_1'}],
            'emise_le': timezone.now().isoformat()}
        self.assertFalse(self._decide([creneau], 'Va pour 11 h 50 à 12 h 50 jeu. 17 sept.', 'c:2'))

    def test_un_autre_message_que_celui_du_tour_n_est_pas_decide(self):
        demande = puces(self.demande_portee())
        self.attendre([demande], 'Non, ne change rien.')
        registre = Registre()
        outils_v2.appliquer_choix_en_attente(self.user, registre, 'Non, ne change rien.', 'z:1')
        self.assertFalse(outils_v2.tour_entierement_decide_par_le_code(
            registre, 'ajoute gym demain à 18 h'))


# ── D3: plus de repli « une seule heure ferme » ─────────────────────────────


class D3HeureDiteTests(HarnaisGardes, TransactionTestCase):

    def _appel(self, brut, nom, tache, **kwargs):
        self.message_courant(brut)
        registre, tools = self.outils(brut, tache=tache)
        self.appeler(tools, nom, **kwargs)
        return registre.actions[-1]

    def _refusee(self, action, dite='15:00'):
        self.assertFalse(action.succes, action.message)
        d = action.donnees
        self.assertTrue(d.get('heure_dite') == dite
                        or (d.get('demande') or {}).get('motif') == 'heure_refusee', d)

    def test_le_titre_dans_une_proposition_avec_son_heure_reste_garde(self):
        cas = (('ajoute gym jeudi à 15 h', 'Gym'), ('mets Bac jeudi à 15 h', 'Bac'),
               ('ajoute mon cours jeudi à 15 h', 'Cours'),
               ('mon rendez-vous jeudi à 15 h', 'Rendez-vous'),
               ('ajoute gym jeudi, à 15 h', 'Gym'), ('Gym jeudi. À 15 h.', 'Gym'))
        for i, (brut, titre) in enumerate(cas):
            with self.subTest(brut=brut, titre=titre):
                self._refusee(self._appel(brut, 'schedule_task_at', f't:{i}', title=titre,
                                          date='2026-09-17', start_time='13:00',
                                          end_time='14:00'))
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 0)

    def test_une_heure_donnee_a_autre_chose_ne_s_impose_plus(self):
        """Le repli imposait l'heure unique du message a tout ajout du jour.
        Ces heures appartiennent a un autre element, ou a rien qu'on sache lier
        a l'appel: le code ne les impose plus."""
        demain = (AUJOURDHUI + timedelta(days=1)).isoformat()
        cas = (('mon cours finit à 15 h demain, place ma lecture', 'Lecture', '17:00'),
               ('demain le souper est à 18 h, ajoute du gym', 'Gym', '07:00'),
               ("j'ai un cours à 14 h demain, ajoute une séance de muscu après",
                'Séance de muscu', '18:00'),
               ('demain je finis à 17 h, trouve-moi du temps pour étudier', 'Étude', '19:00'),
               ('demain à 9 h je suis chez ma mère, place mon ménage', 'Ménage', '13:00'))
        for i, (brut, titre, debut) in enumerate(cas):
            with self.subTest(brut=brut):
                fin = f'{int(debut[:2]) + 1:02d}:00'
                action = self._appel(brut, 'schedule_task_at', f'a:{i}', title=titre,
                                     date=demain, start_time=debut, end_time=fin)
                self.assertTrue(action.succes, action.message)

    def test_ecart_accepte_titre_renomme_ou_heure_par_pronom(self):
        """Ecart documente dans outils.py: un titre renomme par le modele, ou
        une heure donnee par pronom dans une autre proposition, n'est pas garde
        par le code (la regle du prompt le couvre). Production main n'a aucune
        garde d'heure dite."""
        cas = (('ajoute gym jeudi à 15 h', 'Entraînement'),
               ('ajoute gym jeudi et mets-le à 15 h', 'Gym'))
        for i, (brut, titre) in enumerate(cas):
            with self.subTest(brut=brut, titre=titre):
                action = self._appel(brut, 'schedule_task_at', f'e:{i}', title=titre,
                                     date='2026-09-17', start_time=f'{13 + i}:00',
                                     end_time=f'{14 + i}:00')
                self.assertTrue(action.succes, action.message)

    def test_la_garde_armee_vise_toujours_le_meme_element(self):
        demain = AUJOURDHUI + timedelta(days=1)
        self.bloc('Travail', demain.weekday(), '13:00', '17:00', block_type='work',
                  flexibility='fixed')
        brut = 'Rdv dentiste demain 14h a 15h, et place aussi une heure de lecture demain'
        self.message_courant(brut)
        registre, tools = self.outils(brut)
        self.appeler(tools, 'schedule_task_at', title='Dentiste', date=demain.isoformat(),
                     start_time='14:00', end_time='15:00')
        refus = registre.actions[-1]
        self.assertEqual(refus.donnees['demande']['motif'], 'heure_refusee')
        self.appeler(tools, 'schedule_task_at', title='RDV dentiste', date=demain.isoformat(),
                     start_time='18:00', end_time='19:00')
        self.assertFalse(registre.actions[-1].succes)
        self.appeler(tools, 'schedule_task_at', title='Lecture', date=demain.isoformat(),
                     start_time='19:00', end_time='20:00')
        self.assertTrue(registre.actions[-1].succes, registre.actions[-1].message)
