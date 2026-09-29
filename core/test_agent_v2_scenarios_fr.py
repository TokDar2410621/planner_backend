"""Suite d'evals FR : 30 scenarios tires d'echecs reels de production.

Contexte fige : mardi 29 septembre 2026, 8 h 00, America/Toronto.
« demain » = mercredi 30 septembre 2026, « aujourd'hui » = 2026-09-29.

Chaque scenario = message(s) FR reel(s) + setup DB minimal, avec assertions
sur l'ETAT FINAL en base (lignes RecurringBlock/Task/ScheduledBlock creees,
supprimees, restaurees) ET sur la verite de la reponse (aucune affirmation
d'action sans recu dans le registre, aucun element invente non nomme par
l'utilisateur).

Contrat teste (stable pendant le rebuild boucle-unique, voir
docs/boucle-unique-2026-09-29.md) :
- outils types + gardes via outils_pour, juge Jev scripte (zero reseau) ;
- registre d'actions (les recus) ;
- demandes.py (questions en attente, puces, gardes destructives) ;
- rendu.py (rendu deterministe) et mesure.py (detecteur anti-mensonge).

Jamais d'assertions sur les internals de l'agent (agent.py, lecture LLM,
DIRE) : ils sont en cours de refactor par un autre lot. Certains scenarios
documentent le comportement CIBLE et peuvent echouer sur l'archi actuelle :
c'est attendu, l'explication est dans le scenario.
"""
import asyncio
from datetime import date, datetime, time
from unittest import expectedFailure, mock

from django.contrib.auth.models import User
from django.test import TransactionTestCase
from django.utils import timezone

from core.models import (ConversationMessage, RecurringBlock,
                         RecurringBlockException, ScheduledBlock, Task)
from core.test_agent_v2_gardes import puces
from core.test_agent_v2_jugement import juger_script
from services.agent_v2 import demandes as dem
from services.agent_v2 import rendu
from services.agent_v2.mesure import (epurer_reponse, fuite_lexicale,
                                       fuites_reponse)
from services.agent_v2.outils import outils_pour
from services.agent_v2.redaction import ReponseDire
from services.agent_v2.registre import Registre

AUJOURDUI = date(2026, 9, 29)  # un mardi
DEMAIN = date(2026, 9, 30)     # un mercredi


class HarnaisScenarios:
    """Melange sans TestCase : horloge figee au mardi 29 septembre 2026."""

    def setUp(self):
        super().setUp()
        self.user = User.objects.create_user(username='scenarios_fr', password='x')
        vrai_localtime = timezone.localtime
        vrai_localdate = timezone.localdate
        maintenant = timezone.make_aware(datetime(2026, 9, 29, 8, 0))

        def faux_localtime(value=None, timezone=None):
            return maintenant if value is None else vrai_localtime(value, timezone)

        def faux_localdate(value=None, timezone=None):
            return AUJOURDUI if value is None else vrai_localdate(value, timezone)

        for cible, remplacant in (('django.utils.timezone.localtime', faux_localtime),
                                  ('django.utils.timezone.localdate', faux_localdate)):
            patcheur = mock.patch(cible, side_effect=remplacant)
            patcheur.start()
            self.addCleanup(patcheur.stop)

    # -------------------------------------------------------------- fixtures

    def bloc(self, titre, jour, debut, fin, block_type='course',
             flexibility=None, **extra):
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

    def outils(self, brut, tache='u:1', registre=None):
        registre = registre if registre is not None else Registre()
        tools = {t.name: t for t in outils_pour(
            self.user, registre, brut, tache=tache, message_brut=brut)}
        return registre, tools

    def appeler(self, registre, tools, nom, **kwargs):
        """Appelle l'outil et rend l'Action consignee (donnees, succes, message).

        Le wrapper d'outils ne rend qu'une chaine ; le vrai resultat vit
        dans le registre.
        """
        asyncio.run(tools[nom].function_schema.function(**kwargs))
        return registre.actions[-1]

    def premier_tour(self, brut, nom, tache='u:1', **kwargs):
        """Le tour de la REQUETE : message courant, un appel, l'action consignee."""
        self.message_courant(brut)
        registre, tools = self.outils(brut, tache=tache)
        self.appeler(registre, tools, nom, **kwargs)
        return registre.actions[-1]

    def juge_saut_ponctuel(self, brut):
        """Juge scripte : « à la place du cours » = saut ponctuel.

        Sans juge, saut_suspect rend Vrai par prudence et la garde de portee
        retient le skip. Le comportement cible (prod 2026-09-29 : aucun
        « seulement ce mardi ou tous les mardis ? » pose) exige un juge qui
        tranche saut_ponctuel.
        """
        return mock.patch(
            "services.agent_v2.jugement.juger",
            juger_script({brut: {"saut": ("saut_ponctuel", 0.95)}}))

    def assertActif(self, bloc, actif=True):
        bloc.refresh_from_db()
        self.assertEqual(bloc.active, actif)

    # ------------------------------------------------------- aides de verite

    def assert_recu(self, registre, outil):
        """La prose peut affirmer : l'action a un recu de succes."""
        self.assertTrue(
            any(a.outil == outil and a.succes for a in registre.actions),
            f"aucun recu de succes pour {outil} : "
            f"{[(a.outil, a.succes) for a in registre.actions]}")

    def assert_sans_recu(self, registre, outil):
        self.assertFalse(
            any(a.outil == outil and a.succes for a in registre.actions),
            f"recu inattendu pour {outil}")

    def assert_prose_adossee(self, prose: str, registre: Registre):
        """Chaque affirmation d'action de la prose a un recu dans le registre.

        C'est la moitie « code » du contrat verifier-puis-rendre : le
        detecteur lexical (mesure) signale les affirmations, le registre
        prouve qu'elles ont eu lieu. verification.py croisera les deux.
        """
        revendications = fuite_lexicale(prose)
        if not revendications:
            return
        self.assertTrue(
            any(a.succes and a.est_mutation for a in registre.actions),
            f"affirmations sans recu : {revendications!r} / prose : {prose!r}")

    def assert_prose_refusee(self, prose: str, registre: Registre):
        """La prose affirme une action ; le registre ne contient AUCUN recu
        de reussite : l'affirmation est une invention pure.

        C'est le pendant negatif d'assert_prose_adossee : ici on verrouille
        qu'une prose affirmant un « Planifié » refuse serait detectee.
        """
        revendications = fuite_lexicale(prose)
        self.assertTrue(revendications,
                        f"aucune affirmation detectee : {prose!r}")
        self.assertFalse(
            any(a.succes and a.est_mutation for a in registre.actions),
            f"recu inattendu pour : {prose!r}")

    def assert_rendu_sans_invention(self, registre: Registre, intrus: str):
        """Le rendu deterministe ne raconte que le registre : jamais d'intrus."""
        texte = rendu.rendre_faits(registre, aujourdhui=AUJOURDUI)
        self.assertNotIn(intrus, texte)


# ============================================================================
# Bloc A : remplacer un cours par un examen (echecs reels du 2026-09-29)
# ============================================================================

class RemplacerCoursParExamenTests(HarnaisScenarios, TransactionTestCase):

    def _cours_entreprise(self):
        # Le vrai cours du mardi : « L'entreprise et ses systèmes », 13 h-16 h.
        return self.bloc("L'entreprise et ses systèmes", 1, '13:00', '16:00')

    def test_01_examen_remplace_cours_entreprise_heures_reprises(self):
        """« J'ai examen à la place du cours d'entreprise » (aujourd'hui).

        Comportement cible : l'agent retrouve le cours (13 h-16 h), saute son
        occurrence et planifie l'examen sur les MEMES heures, sans demander.
        En prod le 2026-09-29, l'agent a demande les heures trois fois.
        """
        cours = self._cours_entreprise()
        brut = "J'ai examen à la place du cours d'entreprise"
        self.message_courant(brut)
        registre, tools = self.outils(brut)

        # Lecture : le cours existe bien aujourd'hui, 13 h a 16 h.
        lecture = self.appeler(registre, tools, 'get_today_schedule', date='2026-09-29')
        titres = [b['title'] for b in lecture.donnees['blocks']]
        self.assertIn("L'entreprise et ses systèmes", titres)
        cours_lu = next(b for b in lecture.donnees['blocks']
                        if b['title'] == "L'entreprise et ses systèmes")
        self.assertEqual((cours_lu['start_time'], cours_lu['end_time']),
                         ('13:00', '16:00'))

        # Remplacement : occurrence sautee + examen sur les heures du cours.
        # En prod, le juge tranche « saut_ponctuel » sur ce message (une seule
        # occurrence visee) : on scripte ce jugement, le repli prudent
        # (juge indisponible -> question de portee) ne s'applique pas ici.
        with self.juge_saut_ponctuel(brut):
            self.appeler(registre, tools, 'skip_block_occurrence', date='2026-09-29',
                         title="L'entreprise et ses systèmes", block_type='course')
        self.appeler(registre, tools, 'schedule_task_at', title='Examen', date='2026-09-29',
                     start_time='13:00', end_time='16:00')

        # ETAT FINAL : l'occurrence est ignoree, l'examen est planifie 13 h-16 h,
        # le bloc recurrent reste actif (les autres mardis sont intacts).
        self.assertTrue(RecurringBlockException.objects.filter(
            user=self.user, recurring_block=cours, date=AUJOURDUI).exists())
        examen = ScheduledBlock.objects.get(user=self.user, date=AUJOURDUI,
                                            task__title='Examen')
        self.assertEqual(examen.start_time, time(13, 0))
        self.assertEqual(examen.end_time, time(16, 0))
        self.assertActif(cours, True)
        self.assert_recu(registre, 'skip_block_occurrence')
        self.assert_recu(registre, 'schedule_task_at')

        # VERITE : la prose affirme, les recus prouvent.
        prose = "Planifié : Examen, aujourd'hui de 13 h à 16 h."
        self.assert_prose_adossee(prose, registre)

    def test_02_examen_demain_matin_remplacement_cible(self):
        """« J'ai examen à la place du cours de demain matin ».

        Echec reel du 2026-09-29 : l'agent a repondu le repli code en dur
        « Je n'ai pas compris. Tu veux ajouter, déplacer ou voir quelque
        chose ? ». Comportement cible : le message est routable (ni
        suppression, ni incomprehension), l'agent lit les cours de demain
        matin, trouve l'unique candidat et le remplace sans detour.
        """
        cours = self.bloc("Gestion d'infrastructures info", 2, '08:00', '11:00')
        brut = "J'ai examen à la place du cours de demain matin"
        with mock.patch("services.agent_v2.jugement.juger",
                        juger_script({brut: {"suppression": (False, 0.95),
                                             "saut": ("saut_ponctuel", 0.95)}})):
            # Routable : ni suppression, ni repli incomprehension.
            self.assertFalse(dem.suppression_demandee(brut))
            registre, tools = self.outils(brut)
            # Lecture : demain matin, un seul candidat plausible.
            lecture = self.appeler(registre, tools, 'get_today_schedule',
                                   date='2026-09-30')
            self.assertTrue(lecture.succes)
            candidats = [b for b in lecture.donnees['blocks']
                         if b['start_time'] < '12:00']
            self.assertEqual(len(candidats), 1)
            self.assertEqual(candidats[0]['title'],
                             "Gestion d'infrastructures info")
            # Remplacement direct sur l'unique candidat, sans detour.
            self.appeler(registre, tools, 'skip_block_occurrence', date='2026-09-30',
                         title="Gestion d'infrastructures info", block_type='course')
            self.appeler(registre, tools, 'schedule_task_at', title='Examen',
                         date='2026-09-30', start_time='08:00', end_time='11:00')
        # ETAT FINAL : occurrence ignoree demain, examen sur ses heures.
        self.assertTrue(RecurringBlockException.objects.filter(
            user=self.user, recurring_block=cours, date=DEMAIN).exists())
        examen = ScheduledBlock.objects.get(user=self.user, date=DEMAIN)
        self.assertEqual((examen.start_time, examen.end_time),
                         (time(8, 0), time(11, 0)))

    def test_03_begaiement_cours_entreprise_reste_routable(self):
        """« Demain j'ai examen à la place du cours de à la place du cours
        d'entreprise » (phrase reelle, begayee, du 2026-09-29).

        Le begaiement ne doit ni declencher une suppression ni rendre le
        message incomprehensible : le juge tranche sur le sens.
        """
        brut = ("Demain j'ai examen à la place du cours de à la place "
                "du cours d'entreprise")
        with mock.patch("services.agent_v2.jugement.juger",
                        juger_script({brut: {"suppression": (False, 0.9)}})):
            self.assertFalse(dem.suppression_demandee(brut))

    def test_04_heures_reprises_du_cours_jamais_redemandees(self):
        """Le coeur du rate : les heures viennent du cours, pas de l'utilisateur.

        L'examen planifie porte exactement les heures du cours remplace.
        Aucune question d'heure n'est emise par ce flux (pas de present_choices,
        pas de poser_question, pas de present_form dans le registre).
        """
        cours = self._cours_entreprise()
        brut = "J'ai examen à la place du cours d'entreprise"
        registre, tools = self.outils(brut)
        with self.juge_saut_ponctuel(brut):
            self.appeler(registre, tools, 'skip_block_occurrence', date='2026-09-29',
                         title="L'entreprise et ses systèmes", block_type='course')
        # Les heures sont lues sur le bloc, jamais demandees.
        self.appeler(registre, tools, 'schedule_task_at', title='Examen', date='2026-09-29',
                     start_time=cours.start_time.strftime('%H:%M'),
                     end_time=cours.end_time.strftime('%H:%M'))
        examen = ScheduledBlock.objects.get(user=self.user, date=AUJOURDUI,
                                            task__title='Examen')
        self.assertEqual((examen.start_time, examen.end_time),
                         (cours.start_time, cours.end_time))
        self.assertFalse(any(a.outil in ('present_choices', 'poser_question',
                                         'present_form')
                             for a in registre.actions))

    def test_05_supprime_examen_8h11_remets_cours_normal(self):
        """« Tu peux supprimer l'examen de 8 à 11 et remettre le cours
        qu'il y a normalement là-bas » (echange reel du 2026-09-29).

        L'examen disparait, l'occurrence du cours est retablie, sans rien
        inventer : le cours restaure est celui qui existe vraiment ce jour-la.
        cancel_scheduled_block est destructif : la garde le retient au tour 1,
        la puce « Oui, je confirme. » liee a CET examen l'autorise au tour 2.
        """
        cours = self.bloc("Conception d'applications", 1, '08:00', '11:00')
        RecurringBlockException.objects.create(
            user=self.user, recurring_block=cours, date=AUJOURDUI)
        tache = Task.objects.create(user=self.user, title='Examen')
        ScheduledBlock.objects.create(
            user=self.user, task=tache, date=AUJOURDUI,
            start_time=time(8, 0), end_time=time(11, 0))

        brut = ("Tu peux supprimer l'examen de 8 à 11 et remettre le cours "
                "qu'il y a normalement là-bas")
        retenue = self.premier_tour(
            brut, 'cancel_scheduled_block', date='2026-09-29', title='Examen')
        self.assertFalse(retenue.succes)
        demande = retenue.donnees['demande']
        self.assertEqual(demande['motif'], 'destructif')

        # Tour 2 : l'utilisateur confirme sur la puce liee a cet examen.
        self.attendre([puces(demande)], 'Oui, je confirme.')
        registre, tools = self.outils('Oui, je confirme.', tache='u:2')
        annulation = self.appeler(registre, tools, 'cancel_scheduled_block',
                                  date='2026-09-29', title='Examen')
        self.assertTrue(annulation.succes)
        restauration = self.appeler(registre, tools, 'restore_block_occurrence',
                                    date='2026-09-29',
                                    title="Conception d'applications",
                                    block_type='course')

        # ETAT FINAL : plus d'examen 8 h-11 h, plus d'exception, cours actif.
        self.assertFalse(ScheduledBlock.objects.filter(
            user=self.user, date=AUJOURDUI, task__title='Examen').exists())
        self.assertFalse(RecurringBlockException.objects.filter(
            user=self.user, recurring_block=cours, date=AUJOURDUI).exists())
        self.assertTrue(restauration.succes)
        self.assertActif(cours, True)
        self.assert_recu(registre, 'cancel_scheduled_block')
        self.assert_recu(registre, 'restore_block_occurrence')

        # VERITE : « est de retour » s'adosse aux recus ; le rendu ne cite
        # que ce qui a vraiment change.
        self.assert_prose_adossee(
            "Conception d'applications est de retour aujourd'hui.", registre)
        self.assert_rendu_sans_invention(registre, 'Gridar')

    def test_06_restauration_sans_annulation_preexistante(self):
        """Restaurer une occurrence jamais ignoree : succes doux, sans invention.

        En prod, l'agent a affirme « il n'y avait déjà plus d'examen à 8 h
        (il avait été remplacé par celui de 13 h, car ils portaient le même
        nom) » : une causalite inventee. Le contrat : succes=True, message
        honnete (« aucune annulation à retirer »), et la prose ne doit pas
        inventer de mecanisme.
        """
        self.bloc('Conception d\u2019applications', 1, '08:00', '11:00')
        registre, tools = self.outils('remets le cours de 8 h')
        retour = self.appeler(registre, tools, 'restore_block_occurrence',
                              date='2026-09-29',
                              title='Conception d\u2019applications',
                              block_type='course')
        self.assertTrue(retour.succes)
        self.assertIn('aucune annulation', retour.message)
        # Le detecteur ne couvre pas les causalites inventees : c'est le
        # croisement prose/registre (verification.py) qui devra les attraper.
        # Ici on verrouille au moins le message honnete de l'outil.
        self.assertFalse(RecurringBlockException.objects.exists())

    def test_07_meme_nom_deplace_sans_dupliquer(self):
        """Meme titre planifie deux fois : upsert, pas de doublon silencieux.

        Echec reel : l'agent a soutenu que l'examen de 8 h « avait été
        remplacé par celui de 13 h, car ils portaient le même nom », puis a
        parle de « deux examens ». Le contrat de schedule_task_at (upsert
        N06) : le second appel DEPLACE le bloc (memes heures finales),
        ne cree aucun doublon. La prose doit donc dire « deplace », jamais
        « deux examens ».
        """
        registre, tools = self.outils("j'ai deux examens aujourd'hui")
        self.appeler(registre, tools, 'schedule_task_at', title='Examen', date='2026-09-29',
                     start_time='08:00', end_time='11:00')
        self.appeler(registre, tools, 'schedule_task_at', title='Examen', date='2026-09-29',
                     start_time='13:00', end_time='16:00')
        examens = list(ScheduledBlock.objects.filter(
            user=self.user, date=AUJOURDUI, task__title='Examen'
        ).order_by('start_time'))
        # UN SEUL bloc, deplace sur les dernieres heures : pas de doublon.
        self.assertEqual(len(examens), 1)
        self.assertEqual((examens[0].start_time, examens[0].end_time),
                         (time(13, 0), time(16, 0)))

    def test_07b_deux_examens_noms_distincts_restent_distincts(self):
        """Deux examens aux noms distincts restent deux blocs distincts."""
        registre, tools = self.outils("j'ai deux examens aujourd'hui")
        self.appeler(registre, tools, 'schedule_task_at', title='Examen de maths',
                     date='2026-09-29', start_time='08:00', end_time='11:00')
        self.appeler(registre, tools, 'schedule_task_at', title='Examen de physique',
                     date='2026-09-29', start_time='13:00', end_time='16:00')
        examens = list(ScheduledBlock.objects.filter(
            user=self.user, date=AUJOURDUI
        ).order_by('start_time'))
        self.assertEqual(len(examens), 2)
        self.assertEqual([(e.start_time, e.end_time) for e in examens],
                         [(time(8, 0), time(11, 0)), (time(13, 0), time(16, 0))])

    def test_08_remplacement_ne_touche_pas_les_autres_elements(self):
        """Regle d'or : ne jamais toucher un element non nomme par l'utilisateur.

        Le remplacement du cours d'entreprise ne doit ni desactiver ni
        ignorer les autres blocs du mardi.
        """
        cours = self._cours_entreprise()
        gridar = self.bloc('Gridar J20 | Mots-clés localisés', 1, '18:00', '21:00')
        publiar = self.bloc('Publiar J05 | Facebook Page', 1, '11:20', '12:20')
        brut = "J'ai examen à la place du cours d'entreprise"
        registre, tools = self.outils(brut)
        with self.juge_saut_ponctuel(brut):
            self.appeler(registre, tools, 'skip_block_occurrence', date='2026-09-29',
                         title="L'entreprise et ses systèmes", block_type='course')
        self.appeler(registre, tools, 'schedule_task_at', title='Examen', date='2026-09-29',
                     start_time='13:00', end_time='16:00')
        self.assertActif(gridar, True)
        self.assertActif(publiar, True)
        self.assertFalse(RecurringBlockException.objects.filter(
            recurring_block__in=[gridar, publiar]).exists())
        # Une seule exception, sur le seul bloc nomme.
        self.assertEqual(RecurringBlockException.objects.filter(
            user=self.user, date=AUJOURDUI).count(), 1)
        self.assertTrue(RecurringBlockException.objects.filter(
            recurring_block=cours).exists())


# ============================================================================
# Bloc B : lecture d'horaire (echecs reels du 2026-09-29)
# ============================================================================

class LectureHoraireTests(HarnaisScenarios, TransactionTestCase):

    def _mardi_charge(self):
        self.bloc('Conception d\u2019applications', 1, '08:00', '11:00')
        self.bloc("L'entreprise et ses systèmes", 1, '13:00', '16:00')
        self.bloc('Gridar J20 | Mots-clés localisés', 1, '18:00', '21:00')

    def test_09_mon_emploi_du_jour_liste_lhoraire(self):
        """« Mon emploi du jour » : rate du 2026-09-29 (formulation voisine
        appelait get_today_schedule, celle-ci n'a pas ete comprise).

        Comportement cible : la lecture liste les vrais cours du jour, avec
        leurs vraies heures, sans inventer, sans supprimer.
        """
        self._mardi_charge()
        with mock.patch("services.agent_v2.jugement.juger",
                        juger_script({"Mon emploi du jour":
                                      {"suppression": (False, 0.95)}})):
            self.assertFalse(dem.suppression_demandee("Mon emploi du jour"))

        registre, tools = self.outils("Mon emploi du jour")
        # Sans date explicite, l'outil lit « aujourd'hui » (horloge figee).
        lecture = self.appeler(registre, tools, 'get_today_schedule')
        self.assertTrue(lecture.succes)
        self.assertEqual(lecture.donnees['date'], '2026-09-29')
        par_titre = {b['title']: b for b in lecture.donnees['blocks']}
        self.assertEqual(par_titre['Conception d\u2019applications']['start_time'], '08:00')
        self.assertEqual(par_titre["L'entreprise et ses systèmes"]['end_time'], '16:00')
        self.assertIn('Gridar J20 | Mots-clés localisés', par_titre)
        # Lecture pure : aucune mutation en base.
        self.assertFalse(any(a.est_mutation and a.succes
                             for a in registre.actions))
        self.assertFalse(RecurringBlockException.objects.exists())

    def test_10_mon_horaire_de_demain(self):
        """« Mon horaire de demain » : les cours du mercredi 30 septembre."""
        self.bloc('Gestion d\u2019infrastructures info', 2, '08:00', '11:00')
        self.bloc('Développement d\u2019applications', 2, '14:00', '17:00')
        registre, tools = self.outils("Mon horaire de demain")
        lecture = self.appeler(registre, tools, 'get_today_schedule', date='2026-09-30')
        titres = [b['title'] for b in lecture.donnees['blocks']]
        self.assertIn('Gestion d\u2019infrastructures info', titres)
        self.assertIn('Développement d\u2019applications', titres)
        bloc_matin = next(b for b in lecture.donnees['blocks']
                          if b['title'] == 'Gestion d\u2019infrastructures info')
        self.assertEqual((bloc_matin['start_time'], bloc_matin['end_time']),
                         ('08:00', '11:00'))

    def test_11_jour_vide_aucun_cours_invente(self):
        """Un jour sans cours : la lecture rend vide, sans inventer de cours."""
        registre, tools = self.outils("Mon emploi du jour")
        lecture = self.appeler(registre, tools, 'get_today_schedule', date='2026-10-04')
        self.assertTrue(lecture.succes)
        self.assertEqual(lecture.donnees['blocks'], [])
        # Le rendu deterministe ne raconte que le registre : aucun titre
        # sorti de nulle part.
        self.assert_rendu_sans_invention(registre, 'Examen')
        self.assert_rendu_sans_invention(registre, 'Conception')

    def test_12_occurrence_ignoree_absente_de_la_lecture(self):
        """Apres le remplacement, le cours ignore n'apparait plus aujourd'hui."""
        cours = self.bloc("L'entreprise et ses systèmes", 1, '13:00', '16:00')
        brut = "J'ai examen à la place du cours d'entreprise"
        registre, tools = self.outils(brut)
        with self.juge_saut_ponctuel(brut):
            self.appeler(registre, tools, 'skip_block_occurrence', date='2026-09-29',
                         title="L'entreprise et ses systèmes", block_type='course')
        self.appeler(registre, tools, 'schedule_task_at', title='Examen', date='2026-09-29',
                     start_time='13:00', end_time='16:00')
        lecture = self.appeler(registre, tools, 'get_today_schedule', date='2026-09-29')
        titres = [b['title'] for b in lecture.donnees['blocks']]
        self.assertNotIn("L'entreprise et ses systèmes", titres)
        self.assertIn('Examen', titres)
        # ... mais le bloc recurrent existe toujours (les autres mardis).
        self.assertActif(cours, True)


# ============================================================================
# Bloc C : suppressions destructives (gardes)
# ============================================================================

class SuppressionsDestructivesTests(HarnaisScenarios, TransactionTestCase):

    def test_13_supprime_cours_demande_confirmation_exacte(self):
        """« supprime mon cours de chimie » : rien ne s'execute, la question
        exacte est posee (motif destructif, cible nommee)."""
        chimie = self.bloc('Chimie générale', 1, '13:00', '15:00')
        script = {"supprime mon cours de chimie": {"jour_vise": (False, 0.9)}}
        with mock.patch("services.agent_v2.jugement.juger", juger_script(script)):
            action = self.premier_tour('supprime mon cours de chimie',
                                       'delete_block', block_id=chimie.id)
        self.assertFalse(action.succes)
        demande = action.donnees['demande']
        self.assertEqual(demande['motif'], 'destructif')
        self.assertEqual(demande['cible']['titre'], 'Chimie générale')
        self.assertEqual([o['id'] for o in demande['options']],
                         ['confirmer', 'annuler'])
        self.assertActif(chimie, True)

    def test_14_puce_confirmer_execute(self):
        """La puce exacte « Oui, je confirme. » au tour suivant execute."""
        chimie = self.bloc('Chimie générale', 1, '13:00', '15:00')
        script = {"supprime mon cours de chimie": {"jour_vise": (False, 0.9)},
                  "Oui, je confirme.": {"jour_vise": (False, 0.9)}}
        with mock.patch("services.agent_v2.jugement.juger", juger_script(script)):
            action = self.premier_tour('supprime mon cours de chimie',
                                       'delete_block', block_id=chimie.id)
            demande = action.donnees['demande']
            self.attendre([puces(demande)], 'Oui, je confirme.')
            registre, tools = self.outils('Oui, je confirme.', tache='u:2')
            self.appeler(registre, tools, 'delete_block', block_id=chimie.id)
        self.assertTrue(registre.actions[-1].succes)
        self.assertActif(chimie, False)
        self.assert_recu(registre, 'delete_block')

    def test_15_oui_libre_ne_confirme_pas(self):
        """« Oui, vas-y » en texte libre : la garde refuse (D1, round 6)."""
        chimie = self.bloc('Chimie générale', 1, '13:00', '15:00')
        script = {"supprime mon cours de chimie": {"jour_vise": (False, 0.9)},
                  "Oui, vas-y": {"jour_vise": (False, 0.9),
                                 "intention": ("accepte", 0.9)}}
        with mock.patch("services.agent_v2.jugement.juger", juger_script(script)):
            action = self.premier_tour('supprime mon cours de chimie',
                                       'delete_block', block_id=chimie.id)
            demande = action.donnees['demande']
            self.attendre([puces(demande)], 'Oui, vas-y')
            registre, tools = self.outils('Oui, vas-y', tache='u:2')
            self.appeler(registre, tools, 'delete_block', block_id=chimie.id)
        self.assertFalse(registre.actions[-1].succes)
        self.assertActif(chimie, True)

    def test_16_suppression_ne_touche_pas_un_autre_bloc(self):
        """Apres suppression confirmee de la chimie, le cours de maths est intact."""
        chimie = self.bloc('Chimie générale', 1, '13:00', '15:00')
        maths = self.bloc('Maths discrètes', 1, '10:00', '12:00')
        script = {"supprime mon cours de chimie": {"jour_vise": (False, 0.9)},
                  "Oui, je confirme.": {"jour_vise": (False, 0.9)}}
        with mock.patch("services.agent_v2.jugement.juger", juger_script(script)):
            action = self.premier_tour('supprime mon cours de chimie',
                                       'delete_block', block_id=chimie.id)
            demande = action.donnees['demande']
            self.attendre([puces(demande)], 'Oui, je confirme.')
            registre, tools = self.outils('Oui, je confirme.', tache='u:2')
            self.appeler(registre, tools, 'delete_block', block_id=chimie.id)
        self.assertActif(chimie, False)
        self.assertActif(maths, True)

    def test_17_efface_tout_jeudi_demande_la_portee(self):
        """« efface tout jeudi » nomme un jour : question de portee d'abord."""
        quart = self.bloc('Quart au dépanneur', 3, '19:00', '02:00',
                          block_type='work', flexibility='fixed',
                          is_night_shift=True)
        action = self.premier_tour('efface tout jeudi', 'delete_block',
                                   block_id=quart.id)
        self.assertFalse(action.succes)
        demande = action.donnees['demande']
        self.assertEqual(demande['motif'], 'portee_jour')
        self.assertEqual(demande['cible']['date'], '2026-10-01')  # jeudi
        self.assertActif(quart, True)
        self.assertFalse(RecurringBlockException.objects.exists())

    def test_18_confirmation_liee_a_la_cible_exacte(self):
        """Une confirmation ne vaut que pour la cible de sa question.

        Bout en bout : la demande de suppression porte sur l'examen de
        chimie. Si l'appel suivant vise l'examen de maths, la garde le
        retient : la puce « Oui, je confirme. » liee a la chimie ne
        supprime jamais les maths.
        """
        for titre in ('Examen de chimie', 'Examen de maths'):
            tache = Task.objects.create(user=self.user, title=titre)
            ScheduledBlock.objects.create(
                user=self.user, task=tache, date=AUJOURDUI,
                start_time=time(13, 0), end_time=time(16, 0))
        retenue = self.premier_tour(
            "supprime l'examen de chimie", 'cancel_scheduled_block',
            date='2026-09-29', title='Examen de chimie')
        self.assertFalse(retenue.succes)
        demande = retenue.donnees['demande']
        self.assertEqual(demande['motif'], 'destructif')

        # L'utilisateur confirme... mais l'appel vise les maths.
        self.attendre([puces(demande)], 'Oui, je confirme.')
        registre, tools = self.outils('Oui, je confirme.', tache='u:2')
        tentative = self.appeler(registre, tools, 'cancel_scheduled_block',
                                 date='2026-09-29', title='Examen de maths')
        self.assertFalse(tentative.succes)
        # Les deux examens sont toujours la : rien n'a ete supprime.
        self.assertEqual(ScheduledBlock.objects.filter(
            user=self.user, date=AUJOURDUI).count(), 2)


# ============================================================================
# Bloc D : verite des reponses (anti-mensonge)
# ============================================================================

class VeriteDesReponsesTests(HarnaisScenarios, TransactionTestCase):

    def test_19_prose_sans_recu_est_signalee(self):
        """« Planifié : Examen, aujourd'hui de 13 h à 16 h. » sans action en
        base : le detecteur signale l'affirmation, la guillotine la retire.

        C'est le garde-fou contre le « c'est fait » alors que rien n'a eu lieu.
        """
        prose = "Planifié : Examen, aujourd'hui de 13 h à 16 h."
        self.assertTrue(fuite_lexicale(prose),
                        "le detecteur doit signaler l'affirmation d'action")
        reponse = ReponseDire(ouverture=prose)
        self.assertTrue(fuites_reponse(reponse))
        epuree, supprimees = epurer_reponse(reponse)
        self.assertEqual(supprimees, 1)
        self.assertEqual(epuree.ouverture, "")
        # Et en base : rien.
        self.assertFalse(ScheduledBlock.objects.exists())

    def test_20_prose_adossee_aux_recus_passe(self):
        """La meme prose, avec les recus en base et au registre : rien a couper."""
        cours = self.bloc("L'entreprise et ses systèmes", 1, '13:00', '16:00')
        brut = "J'ai examen à la place du cours d'entreprise"
        registre, tools = self.outils(brut)
        with self.juge_saut_ponctuel(brut):
            self.appeler(registre, tools, 'skip_block_occurrence', date='2026-09-29',
                         title="L'entreprise et ses systèmes", block_type='course')
        self.appeler(registre, tools, 'schedule_task_at', title='Examen', date='2026-09-29',
                     start_time='13:00', end_time='16:00')
        prose = "Planifié : Examen, aujourd'hui de 13 h à 16 h."
        # L'affirmation est bien une affirmation (detecteur), mais elle est
        # vraie : chaque action affirmee a son recu.
        self.assertTrue(fuite_lexicale(prose))
        self.assert_prose_adossee(prose, registre)
        examen = ScheduledBlock.objects.get(user=self.user, date=AUJOURDUI,
                                            task__title='Examen')
        self.assertEqual((examen.start_time, examen.end_time),
                         (time(13, 0), time(16, 0)))
        self.assertActif(cours, True)

    def test_21_question_reposee_nest_pas_une_affirmation(self):
        """« Demain, quel cours remplacer par l'examen ? » : une question,
        zero affirmation d'action, zero mutation."""
        with mock.patch("services.agent_v2.jugement.juger",
                        juger_script({"Demain, quel cours remplacer par l'examen ?":
                                      {"suppression": (False, 0.9)}})):
            self.assertFalse(dem.suppression_demandee(
                "Demain, quel cours remplacer par l'examen ?"))
        question = "Demain, quel cours remplacer par l'examen ?"
        self.assertEqual(fuite_lexicale(question), [])
        self.assertEqual(fuites_reponse(ReponseDire(question=question)), [])
        self.assertFalse(ScheduledBlock.objects.exists())
        self.assertFalse(RecurringBlockException.objects.exists())

    def test_22_rendu_ne_raconte_que_le_registre(self):
        """Le rendu deterministe ne peut pas inventer « Gridar J19 » :
        l'element n'est ni nomme par l'utilisateur ni au registre."""
        cours = self.bloc("L'entreprise et ses systèmes", 1, '13:00', '16:00')
        brut = "J'ai examen à la place du cours d'entreprise"
        registre, tools = self.outils(brut)
        with self.juge_saut_ponctuel(brut):
            self.appeler(registre, tools, 'skip_block_occurrence', date='2026-09-29',
                         title="L'entreprise et ses systèmes", block_type='course')
        self.appeler(registre, tools, 'schedule_task_at', title='Examen', date='2026-09-29',
                     start_time='13:00', end_time='16:00')
        texte = rendu.rendre_faits(registre, aujourdhui=AUJOURDUI)
        self.assertNotIn('Gridar', texte)
        # ... mais il raconte bien ce qui a vraiment change.
        self.assertIn('Examen', texte)
        self.assertActif(cours, True)

    @expectedFailure
    def test_23_justification_inventee_sans_recu(self):
        """Echec reel : « il avait été remplacé par celui de 13 h, car ils
        portaient le même nom » : causalite inventee, aucun recu.

        COMPORTEMENT CIBLE (xrange en attendant verification.py) : toute
        explication causale (« car ... ») doit s'adosser a un recu du
        registre. Le detecteur lexical actuel ne voit pas les causalites.
        """
        registre = Registre()  # aucun recu : rien ne s'est passe
        self.assertFalse(ScheduledBlock.objects.exists())
        prose = ("il avait été remplacé par celui de 13 h, "
                 "car ils portaient le même nom")
        # Cible : le croisement prose/registre refuse la causalite sans recu.
        self.assertTrue(
            any(a.succes and a.est_mutation for a in registre.actions),
            f"causalite affirmee sans recu, non refusee : {prose!r}")


# ============================================================================
# Bloc E : robustesse et cas limites
# ============================================================================

class RobustesseTests(HarnaisScenarios, TransactionTestCase):

    @expectedFailure
    def test_24_faute_frappe_retrouve_le_cours(self):
        """« cours d'entriprise » (faute reelle) : COMPORTEMENT CIBLE (xrange),
        la resolution retrouve semantiquement le cours malgre la faute.

        L'architecture actuelle echoue proprement sans rien toucher (le
        contrat minimum : jamais de skip sur un mauvais bloc), mais ne
        retrouve pas le cours. La tolerance aux fautes appartient a la
        boucle (le modele lit l'horaire et rapproche) : pas encore evaluee
        au niveau des outils scriptes.
        """
        cours = self.bloc("L'entreprise et ses systèmes", 1, '13:00', '16:00')
        brut = "J'ai examen à la place du cours d'entriprise"
        registre, tools = self.outils(brut)
        with self.juge_saut_ponctuel(brut):
            retour = self.appeler(registre, tools, 'skip_block_occurrence',
                                  date='2026-09-29', title="cours d'entriprise",
                                  block_type='course')
        # CIBLE : le cours est retrouve malgre la faute, l'occurrence saute.
        self.assertTrue(
            retour.succes,
            "le cours n'a pas ete retrouve malgre la faute de frappe")
        self.assertTrue(RecurringBlockException.objects.filter(
            user=self.user, recurring_block=cours, date=AUJOURDUI).exists())

    def test_25_message_bruit_aucune_mutation(self):
        """« euh » : aucune mutation, aucune exception, aucune demande piegee."""
        registre, tools = self.outils("euh")
        self.appeler(registre, tools, 'get_today_schedule', date='2026-09-29')
        self.assertFalse(any(a.est_mutation and a.succes
                             for a in registre.actions))
        self.assertFalse(RecurringBlockException.objects.exists())
        self.assertFalse(ScheduledBlock.objects.exists())

    def test_26_refus_apres_question_annule_tout(self):
        """« non, laisse tomber » apres une question de suppression : la
        demande est abandonnee, rien ne s'execute."""
        chimie = self.bloc('Chimie générale', 1, '13:00', '15:00')
        script = {"supprime mon cours de chimie": {"jour_vise": (False, 0.9)},
                  "non, laisse tomber": {"intention": ("refuse", 0.92)}}
        with mock.patch("services.agent_v2.jugement.juger", juger_script(script)):
            action = self.premier_tour('supprime mon cours de chimie',
                                       'delete_block', block_id=chimie.id)
            demande = action.donnees['demande']
            self.assertEqual(dem.option_choisie("non, laisse tomber",
                                                puces(demande)), "annuler")
        self.assertActif(chimie, True)

    def test_27_deux_evenements_meme_jour_heures_distinctes(self):
        """Examen le matin + examen l'apres-midi : deux creneaux distincts,
        listes tous les deux a la lecture."""
        registre, tools = self.outils("j'ai deux examens aujourd'hui")
        self.appeler(registre, tools, 'schedule_task_at', title='Examen de maths',
                     date='2026-09-29', start_time='08:00', end_time='11:00')
        self.appeler(registre, tools, 'schedule_task_at', title='Examen de physique',
                     date='2026-09-29', start_time='13:00', end_time='16:00')
        lecture = self.appeler(registre, tools, 'get_today_schedule', date='2026-09-29')
        examens = [b for b in lecture.donnees['blocks'] if b['title'].startswith('Examen de')]
        self.assertEqual(len(examens), 2)
        self.assertEqual([(e['start_time'], e['end_time']) for e in examens],
                         [('08:00', '11:00'), ('13:00', '16:00')])

    def test_28_chevauchement_refuse_sans_ecraser(self):
        """Un examen qui chevauche un bloc existant est REFUSE, pas ecrase.

        Contrat de schedule_task_at : en cas de conflit avec un bloc
        recurrent, l'outil echoue proprement, ne cree rien, et le bloc
        existant reste actif. La prose ne doit donc jamais annoncer
        « Planifié » sans recu.
        """
        self.bloc('Réunion labo', 1, '13:00', '14:00')
        registre, tools = self.outils("J'ai examen de 13 h à 16 h")
        refus = self.appeler(registre, tools, 'schedule_task_at', title='Examen',
                             date='2026-09-29', start_time='13:00', end_time='16:00')
        self.assertFalse(refus.succes)
        self.assertFalse(ScheduledBlock.objects.filter(
            user=self.user, date=AUJOURDUI, task__title='Examen').exists())
        # Le bloc recurrent chevauche n'a pas ete desactive ni ignore.
        labo = RecurringBlock.objects.get(user=self.user, title='Réunion labo')
        self.assertActif(labo, True)
        self.assertFalse(RecurringBlockException.objects.filter(
            recurring_block=labo).exists())
        # Sans recu, « Planifié : Examen » serait une invention.
        self.assert_prose_refusee("Planifié : Examen, de 13 h à 16 h.", registre)

    def test_29_idempotence_meme_tour_pas_de_doublon(self):
        """Le meme appel rejoue dans le meme tour ne cree qu'une seule ligne."""
        registre, tools = self.outils("J'ai examen de 13 h à 16 h", tache='u:1')
        kwargs = dict(title='Examen', date='2026-09-29',
                      start_time='13:00', end_time='16:00')
        self.appeler(registre, tools, 'schedule_task_at', **kwargs)
        self.appeler(registre, tools, 'schedule_task_at', **kwargs)  # rejoue
        # Le tour a consigne deux actions...
        self.assertEqual(
            sum(1 for a in registre.actions if a.outil == 'schedule_task_at'), 2)
        # ... mais une seule ligne en base (tester-puis-poser).
        self.assertEqual(ScheduledBlock.objects.filter(
            user=self.user, date=AUJOURDUI, task__title='Examen').count(), 1)

    def test_30_question_outil_sans_effet_de_bord(self):
        """present_choices : une question ne mute jamais la base."""
        registre, tools = self.outils("Demain, quel cours remplacer par l'examen ?")
        retour = self.appeler(
            registre, tools, 'present_choices',
            question="Quel cours remplacer par l'examen ?",
            options=[{"label": "Gestion d'infrastructures info",
                      "value": "Gestion d'infrastructures info"},
                     {"label": "Développement d'applications",
                      "value": "Développement d'applications"}])
        self.assertFalse(ScheduledBlock.objects.exists())
        self.assertFalse(RecurringBlockException.objects.exists())
        self.assertFalse(RecurringBlock.objects.filter(active=False).exists())
