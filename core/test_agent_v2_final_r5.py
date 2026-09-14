"""
Round 5, correcteur final (revues gardes, verite, regressions, lisibilite et
banc du round 4). Chaque classe a ete ecrite AVANT son correctif et vue en
echec.

G1  lecture de portee: une question en echo (« Tous les jeudis ? »), un autre
    jour (« tous les lundis »), une occurrence niee (« pas seulement ce
    jeudi ») ou une negation elidee (« n'efface rien ») ne suppriment rien;
    une reponse de garde qui nomme la portee (« Non, garde tous les jeudis. »)
    tranche au lieu de reposer la question sans fin.
R2  reemission collante: une demande n'est reposee que sur une reponse
    plausible, une seule fois, et « supprime mon gym tous les jeudis » ne
    supprime jamais le quart en attente.
G2  heure dite dans une autre proposition que le titre (« ajoute gym jeudi et
    mets-le a 15 h »); l'heure d'un AUTRE element (« j'ai un cours a 14 h »)
    ou une fin de journee ne s'impose pas a un ajout renomme.
B5  une duree de formulaire (« Heures d'etude: 4 h ») n'est pas une heure.
V1  le brouillon d'AGIR ne fait entrer au brief que des questions et offres
    structurelles, et rien quand une action a echoue ce tour.
L1  une affirmation d'absence contredite par la liste affichee tombe; une
    lecture de soutien ne se deverse pas sous un formulaire.

Horloge du harnais: lundi 14 septembre 2026, 8 h (jour 0 = lundi).
"""
from datetime import timedelta
from unittest.mock import patch

from django.test import SimpleTestCase, TransactionTestCase

from core import test_agent_v2_voix_r4 as voix_r4
from core.models import (ConversationMessage, RecurringBlock,
                         RecurringBlockException, ScheduledBlock)
from core.test_agent_v2_gardes import AUJOURDHUI, HarnaisGardes, puces
from core.test_agent_v2_narrateur import NarrateurBase, faux_rendu, formulaire, ok
from services.agent.tools.base import ToolResult
from services.agent_v2 import demandes as dem
from services.agent_v2 import outils as outils_v2
from services.agent_v2 import redaction
from services.agent_v2.agent import PlannerAgentV2
from services.agent_v2.mesure import fuite_question, questions_et_offres
from services.agent_v2.redaction import ReponseDire, composer
from services.agent_v2.registre import Registre

PORTEE_JEUDI = {'motif': 'portee_jour', 'cle': 'p', 'cible': {'titre': 'Quart au dépanneur',
                                                              'jour': 3, 'date': '2026-09-17'},
                'options': [{'id': 'occurrence'}, {'id': 'serie'}, {'id': 'annuler'}]}
DESTR = {'motif': 'destructif', 'cle': 'd',
         'options': [{'id': 'confirmer'}, {'id': 'annuler'}]}


# ── G1: lecture de la reponse de portee ─────────────────────────────────────


class G1LecteurTests(SimpleTestCase):

    def test_rien_ne_supprime_ni_ne_saute(self):
        for brut in ('Tous les jeudis ?', "c'est tous les jeudis ?", 'tous les lundis',
                     'pas juste ce jeudi, tous', 'pas seulement ce jeudi',
                     "n'efface rien, tous les jeudis restent",
                     "N'enlève rien, tous les jeudis restent",
                     'je veux pu rien effacer, tous les jeudis restent',
                     'efface rien, tous les jeudis restent',
                     'supprime mon gym tous les jeudis', 'enlève la chimie tous les jeudis'):
            with self.subTest(brut=brut):
                self.assertIn(dem.option_choisie(brut, PORTEE_JEUDI), (None, 'annuler'))

    def test_oui_interrogatif_ne_confirme_pas(self):
        for brut in ('oui ?', 'ok ?', 'Oui?'):
            with self.subTest(brut=brut):
                self.assertIsNone(dem.option_choisie(brut, DESTR))
        self.assertEqual(dem.option_choisie('oui', DESTR), 'confirmer')

    def test_garder_en_nommant_la_portee_tranche(self):
        for brut in ('Non, garde tous les jeudis.', 'Non garde tous les jeudis',
                     'Laisse tous les jeudis', 'Garde-la tous les jeudis',
                     'Je veux la garder chaque semaine', 'non non garde tous les jeudis je te dis',
                     'garde tous les jeudis', "n'efface rien, tous les jeudis restent"):
            with self.subTest(brut=brut):
                self.assertEqual(dem.option_choisie(brut, PORTEE_JEUDI), 'annuler')

    def test_les_reponses_claires_restent_lisibles(self):
        cas = {'tous les jeudis': 'serie', 'Tous les jeudis': 'serie',
               'supprime tous les jeudis': 'serie', 'enlève le quart tous les jeudis': 'serie',
               'oui mais seulement ce jeudi': 'occurrence', 'juste celui-là': 'occurrence',
               'Tous les jeudis (supprimer la série).': 'serie',
               'non pas tous les jeudis': None, 'laisse tomber ce cours, tous les jeudis': None}
        for brut, attendu in cas.items():
            with self.subTest(brut=brut):
                self.assertEqual(dem.option_choisie(brut, PORTEE_JEUDI), attendu)


class G1BoutEnBoutTests(HarnaisGardes, TransactionTestCase):

    def test_aucune_de_ces_reponses_n_agit(self):
        gym = self.bloc('Gym', 3, '17:00', '18:00', block_type='sport')
        for i, brut in enumerate(('Tous les jeudis ?', "c'est tous les jeudis ?", 'tous les lundis',
                                  'pas juste ce jeudi, tous', 'pas seulement ce jeudi',
                                  "n'efface rien, tous les jeudis restent",
                                  'supprime mon gym tous les jeudis')):
            with self.subTest(brut=brut):
                demande = puces(self.demande_portee(tache=f'r:{i}'))
                self.attendre([demande], brut)
                registre = Registre()
                outils_v2.appliquer_choix_en_attente(self.user, registre, brut, f'e:{i}')
                self.assertFalse(any(a.succes for a in registre.actions))
                self.assertActif(self.q)
                self.assertActif(gym)
                self.assertEqual(RecurringBlockException.objects.filter(
                    recurring_block=self.q).count(), 0)


# ── R2: reemission seulement sur une reponse plausible, une fois ────────────


class R2ReemissionTests(HarnaisGardes, TransactionTestCase):

    def _reemises(self, registre):
        return [a.donnees['demande'] for a in registre.actions
                if isinstance((a.donnees or {}).get('demande'), dict)]

    def test_un_message_sans_rapport_ne_repose_pas(self):
        for i, brut in enumerate(("c'est quoi mon horaire demain ?", 'ajoute gym demain à 18 h',
                                  'merci, bonne nuit')):
            with self.subTest(brut=brut):
                demande = puces(self.demande_portee(tache=f's:{i}'))
                self.attendre([demande], brut)
                registre = Registre()
                outils_v2.appliquer_choix_en_attente(self.user, registre, brut, f'n:{i}')
                self.assertEqual(self._reemises(registre), [])
                self.assertActif(self.q)

    def test_reposee_une_seule_fois_avec_sa_date_d_origine(self):
        demande = puces(self.demande_portee())
        u2 = self.attendre([demande], 'oui')
        registre = Registre()
        outils_v2.appliquer_choix_en_attente(self.user, registre, 'oui', 'o:1')
        reposees = self._reemises(registre)
        self.assertEqual([d['cle'] for d in reposees], [demande['cle']])
        self.assertEqual(reposees[0]['emise_le'], demande['emise_le'])

        question = PlannerAgentV2._question_des_demandes(reposees)
        ConversationMessage.objects.create(
            user=self.user, role='assistant', content=question['question'],
            metadata={'en_reponse_a': u2.pk, 'demandes': question['demandes']})
        self.message_courant('oui oui')
        registre = Registre()
        outils_v2.appliquer_choix_en_attente(self.user, registre, 'oui oui', 'o:2')
        self.assertEqual(self._reemises(registre), [])
        self.assertActif(self.q)

    def test_la_suppression_d_un_autre_element_ne_touche_pas_le_quart(self):
        gym = self.bloc('Gym', 3, '17:00', '18:00', block_type='sport')
        demande = puces(self.demande_portee())
        u2 = self.attendre([demande], 'oui')
        registre = Registre()
        outils_v2.appliquer_choix_en_attente(self.user, registre, 'oui', 'w:1')
        question = PlannerAgentV2._question_des_demandes(self._reemises(registre))
        ConversationMessage.objects.create(
            user=self.user, role='assistant', content=question['question'],
            metadata={'en_reponse_a': u2.pk, 'demandes': question['demandes']})
        brut = 'supprime mon gym tous les jeudis'
        self.message_courant(brut)
        registre = Registre()
        outils_v2.appliquer_choix_en_attente(self.user, registre, brut, 'w:2')
        self.assertFalse(any(a.succes for a in registre.actions))
        self.assertEqual(self._reemises(registre), [])
        self.assertActif(self.q)
        self.assertActif(gym)


class L2BoucleDeGardeTests(voix_r4.DemandeReemiseTests):
    """Tours complets: la garde qui nomme la portee ferme la boucle, et un
    tour qui repose la question ne dit pas « je garde »."""

    test_oui_flou_puis_tous_les_jeudis = None
    test_une_reponse_claire_n_est_pas_reemise = None

    def test_non_garde_tous_les_jeudis_ferme_la_question(self):
        premier = self._tour('enlève le quart de jeudi', agir=self._supprimer)
        self.assertEqual(premier['question_motif'], 'portee_jour')
        done = self._tour('Non, garde tous les jeudis.')
        self.assertNotEqual(done['question_motif'], 'portee_jour')
        self.assertFalse(done['question_posee'])
        self.assertEqual(self._meta()['demandes'], [])
        self.assertTrue(RecurringBlock.all_objects.get(pk=self.quart.pk).active)

    def test_un_sujet_sans_rapport_ne_repose_pas_la_question(self):
        self._tour('enlève le quart de jeudi', agir=self._supprimer)
        done = self._tour("c'est quoi mon horaire demain ?")
        self.assertNotEqual(done['question_motif'], 'portee_jour')
        self.assertEqual(self._meta()['demandes'], [])

    def test_la_question_reposee_ne_porte_pas_d_accuse_de_decision(self):
        self._tour('enlève le quart de jeudi', agir=self._supprimer)
        done = self._tour('Oui, supprime-le.',
                          dire=ReponseDire(ouverture="D'accord, je garde ton quart."))
        self.assertEqual(done['question_motif'], 'portee_jour')
        self.assertNotIn('je garde', done['response'])


# ── G2: heure dite, propositions et elements distincts ──────────────────────


class G2HeureDiteTests(HarnaisGardes, TransactionTestCase):

    def _appel(self, brut, nom, tache, **kwargs):
        self.message_courant(brut)
        registre, tools = self.outils(brut, tache=tache)
        self.appeler(tools, nom, **kwargs)
        return registre.actions[-1]

    def _refusee(self, action):
        self.assertFalse(action.succes, action.message)
        d = action.donnees
        self.assertTrue(d.get('heure_dite') == '15:00'
                        or (d.get('demande') or {}).get('motif') == 'heure_refusee', d)

    def test_heure_dans_une_autre_proposition(self):
        for i, brut in enumerate(('ajoute gym jeudi et mets-le à 15 h', 'ajoute gym jeudi, à 15 h',
                                  'Gym jeudi. À 15 h.')):
            with self.subTest(brut=brut, outil='schedule_task_at'):
                self._refusee(self._appel(brut, 'schedule_task_at', f'p:{i}', title='Gym',
                                          date='2026-09-17', start_time='13:00',
                                          end_time='14:00'))
            with self.subTest(brut=brut, outil='create_block'):
                self._refusee(self._appel(brut, 'create_block', f'b:{i}', title='Gym',
                                          block_type='sport', days=['jeudi'],
                                          start_time='13:00', end_time='14:00'))
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 0)
        self.assertFalse(RecurringBlock.objects.filter(user=self.user, title='Gym').exists())

    def test_l_heure_d_un_autre_element_ne_s_impose_pas(self):
        demain = (AUJOURDHUI + timedelta(days=1)).isoformat()
        cas = (("j'ai un cours à 14 h demain, ajoute une séance de muscu après", 'Gym', '16:00'),
               ("j'ai un cours à 14 h demain, ajoute une séance de muscu après",
                'Séance de muscu', '18:00'),
               ('demain je finis à 17 h, trouve-moi du temps pour étudier', 'Étude', '19:00'))
        for i, (brut, titre, debut) in enumerate(cas):
            with self.subTest(brut=brut, titre=titre):
                fin = f'{int(debut[:2]) + 1}:00'
                action = self._appel(brut, 'schedule_task_at', f'a:{i}', title=titre,
                                     date=demain, start_time=debut, end_time=fin)
                self.assertTrue(action.succes, action.message)


# ── B5: une duree de formulaire n'est pas une heure ─────────────────────────


class B5DureeDeFormulaireTests(HarnaisGardes, TransactionTestCase):

    def test_le_formulaire_d_etude_se_place(self):
        brut = ("Voici mes réponses :\nHeures d'étude: 4 h\n"
                "Jours d'étude: Mardi, Mercredi, Samedi, Dimanche")
        self.assertEqual(dem.heures_dites(brut), [])
        self.message_courant(brut)
        registre, tools = self.outils(brut)
        for jour in ('2026-09-15', '2026-09-16', '2026-09-19', '2026-09-20'):
            self.appeler(tools, 'schedule_task_at', title='Étude', date=jour,
                         start_time='10:00', end_time='11:00')
            self.assertTrue(registre.actions[-1].succes, registre.actions[-1].message)
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 4)

    def test_une_heure_de_rendez_vous_etiquetee_reste_une_heure(self):
        self.assertEqual(dem.heures_dites('Voici mes réponses :\nHeure du rendez-vous: 14 h'),
                         ['14:00'])


# ── V1: le brouillon d'AGIR ─────────────────────────────────────────────────

AFFIRMATIONS = (
    "Je l'ai mis jeudi à 9 h, veux-tu que je change ?",
    "C'est noté pour jeudi à 14 h, autre chose ?",
    'Ta tâche est inscrite jeudi à 14 h, tu veux un rappel ?',
    'Tu as maintenant ta tâche jeudi à 14 h, veux-tu un rappel ?',
    'Ton gym est maintenant jeudi à 9 h, tu veux autre chose ?',
    "Le gym est désormais à 9 h, veux-tu que j'ajoute un rappel ?",
    'Ton cours de stats est rendu à 14 h, ça marche ?',
    'Ton rendez-vous est confirmé pour jeudi 15 h, autre chose ?',
    'Ton gym de jeudi est parti, on en remet un vendredi ?',
    "Ton gym, je l'ai mis jeudi à 9 h, ça te va ?",
    'Plus de gym jeudi, tu veux autre chose ?',
    'Ton cours ne figure plus à ton horaire, ça te va ?',
)
LEGITIMES = (
    'Veux-tu que je le déplace à 14 h ?',
    "Dis-moi l'heure et je le place.",
    'Tu veux enlever ton cours seulement ce jeudi ou tous les jeudis ?',
    'Quel jour te va le mieux ?',
    'À quelle heure commence ton quart ?',
)


class V1BrouillonTests(SimpleTestCase):

    def test_les_affirmations_ne_passent_pas(self):
        for brouillon in AFFIRMATIONS:
            with self.subTest(brouillon=brouillon):
                self.assertEqual(questions_et_offres(brouillon), '')

    def test_les_questions_et_offres_passent(self):
        for brouillon in LEGITIMES:
            with self.subTest(brouillon=brouillon):
                self.assertEqual(questions_et_offres(brouillon), brouillon)

    def test_fuite_question_voit_la_premiere_personne_a_clitique(self):
        for texte in ("Je l'ai mis jeudi à 9 h, veux-tu que je change ?",
                      "C'est noté pour jeudi à 14 h, autre chose ?",
                      'Je te les ai inscrits jeudi, ça te va ?'):
            with self.subTest(texte=texte):
                self.assertTrue(fuite_question(texte))
        self.assertEqual(fuite_question("C'est bon pour toi ?"), [])

    def test_une_action_en_echec_efface_le_brouillon(self):
        echecs = (ToolResult(success=False, data={'date_passee': '2026-09-10'},
                             message='Refuse par le code: 2026-09-10 est deja passe.'),
                  ToolResult(success=False, data={}, message="Erreur de l'outil: conflit"))
        for resultat in echecs:
            registre = Registre()
            registre.ajouter('schedule_task_at', {'title': 'Rapport', 'date': '2026-09-10',
                                                  'start_time': '14:00'}, resultat)
            for brouillon in AFFIRMATIONS + ('Veux-tu que je le place vendredi ?',):
                with self.subTest(resultat=resultat.message, brouillon=brouillon):
                    brief = PlannerAgentV2._brief_dire('mets mon rapport jeudi à 14 h', registre,
                                                       {}, '', brouillon=brouillon)
                    self.assertNotIn("BROUILLON D'AGIR", brief)

    def test_registre_vide_rien_d_affirme(self):
        for brouillon in AFFIRMATIONS:
            with self.subTest(brouillon=brouillon):
                brief = PlannerAgentV2._brief_dire('mets mon rapport jeudi à 14 h', Registre(),
                                                   {}, '', brouillon=brouillon)
                self.assertNotIn("BROUILLON D'AGIR", brief)


# ── L1: absence contredite, lecture deversee sous un formulaire ─────────────


class L1AbsenceEtLectureTests(NarrateurBase):

    def test_une_absence_contredite_par_la_liste_tombe(self):
        registre = Registre()
        registre.ajouter('list_blocks', {}, ToolResult(success=True, message='ok',
                                                      data={'blocks': []}))
        faits = 'Lundi\n- Calcul différentiel, 9 h à 12 h'
        compo = composer(ReponseDire(ouverture="Il n'y a pas de cours de maths dans ton horaire.",
                                     suite='Tu peux ajouter une révision.'),
                         registre, faits, None)
        self.assertNotIn("n'y a pas", compo.prose)
        self.assertIn('révision', compo.prose)

    def test_la_lecture_de_soutien_ne_se_deverse_pas_sous_un_formulaire(self):
        rendu = faux_rendu(self.journal)
        rendu.rendre_lecture = lambda registre, aujourdhui=None: 'LISTE DE LA SEMAINE'
        with patch.object(redaction, '_charger_rendu', return_value=rendu):
            _, done = self.tour(actions=[ok('list_blocks'), formulaire()],
                                dire=ReponseDire(ouverture='Parfait.'))
        self.assertNotIn('LISTE DE LA SEMAINE', done['response'])

    def test_une_lecture_seule_reste_affichee(self):
        rendu = faux_rendu(self.journal)
        rendu.rendre_lecture = lambda registre, aujourdhui=None: 'LISTE DE LA SEMAINE'
        with patch.object(redaction, '_charger_rendu', return_value=rendu):
            _, done = self.tour(actions=[ok('list_blocks')], dire=ReponseDire(ouverture='Voilà.'))
        self.assertIn('LISTE DE LA SEMAINE', done['response'])
