"""
Round 8, correctifs du fixeur. Chaque classe a ete ecrite AVANT son correctif
et vue en echec.

F1  une heure nue de 1 a 11 (« souper jeudi a 6 h ») se lit H ou H+12: aucun
    appel a l'une ou l'autre n'est refuse, et un refus nomme les deux. Un
    marqueur (du matin, am, du soir, pm, de l'apres-midi) garde la lecture
    stricte.
F2  un verbe de garde nie (« ne le garde pas ») ne ferme pas la demande.
F3  la mecanique d'interface tombe, un conseil ordinaire reste.
F4  un oui clair confirme une creation en masse; jamais une suppression.
F5  une piece jointe au tour interdit le chemin rapide.
F6  une nouvelle demande destructive (« efface tout ») est une nouvelle
    requete: la demande en attente est abandonnee, rien ne s'execute.
"""
from unittest.mock import patch

from django.test import SimpleTestCase, TransactionTestCase
from django.utils import timezone

from core.models import ScheduledBlock, UploadedDocument
from core.test_agent_v2_gardes import HarnaisGardes, puces
from core.test_agent_v2_gardes_r6 import ANNULER_EVENEMENT, DESTR, MASSE, PLAN, PORTEE
from core.test_agent_v2_voix_r6 import _TourDecideBase
from services.agent_v2 import demandes as dem
from services.agent_v2 import outils as outils_v2
from services.agent_v2.agent import PlannerAgentV2
from services.agent_v2.redaction import ReponseDire, composer
from services.agent_v2.registre import Registre

CRENEAU = {
    'type': 'choix', 'motif': 'heure_refusee', 'cle': 'h1', 'outil': 'schedule_task_at',
    'parametres': {}, 'cible': {'titre': 'Dentiste', 'date': '2026-09-17'},
    'options': [{'id': 'creneau_1', 'effet': None,
                 'cible': {'titre': 'Dentiste', 'date': '2026-09-17',
                           'debut': '11:50', 'fin': '12:50'}},
                {'id': 'autre_jour', 'effet': None, 'cible': {'titre': 'Dentiste'}}],
    'chips': [{'label': '11 h 50 à 12 h 50',
               'value': 'Va pour 11 h 50 à 12 h 50 jeu. 17 sept.', 'option': 'creneau_1'}],
}


# ── F1: heure nue, deux lectures ────────────────────────────────────────────


class F1HeureNueTests(HarnaisGardes, TransactionTestCase):

    def _appel(self, brut, titre, debut, tache):
        self.message_courant(brut)
        registre, tools = self.outils(brut, tache=tache)
        fin = f'{(int(debut[:2]) + 1) % 24:02d}{debut[2:]}'
        self.appeler(tools, 'schedule_task_at', title=titre, date='2026-09-17',
                     start_time=debut, end_time=fin)
        return registre.actions[-1]

    def test_une_heure_nue_accepte_les_deux_lectures(self):
        cas = (('souper jeudi à 6 h', 'Souper', '18:00'),
               # Round 9 (K4): « souper » est un soir; « souper jeudi a 6 h »
               # ne vaut plus 06:00 (core/test_agent_v2_gardes_r9.py).
               ('lecture jeudi à 6 h', 'Lecture', '06:00'),
               # Le harnais a un quart le jeudi de 19 h a 24 h: les heures
               # restent avant, pour que seul le code de l'heure dite parle.
               ('gym jeudi à 5 h', 'Gym', '17:00'),
               ('gym jeudi à 4 h 30', 'Gym', '16:30'),
               ('ajoute lecture jeudi à 3:15', 'Lecture', '15:15'))
        for i, (brut, titre, debut) in enumerate(cas):
            with self.subTest(brut=brut, debut=debut):
                ScheduledBlock.objects.filter(user=self.user).delete()
                action = self._appel(brut, titre, debut, f'n:{i}')
                self.assertTrue(action.succes, action.message)

    def test_un_marqueur_garde_la_lecture_stricte(self):
        refus = (('gym jeudi à 7 h du matin', 'Gym', '19:00'),
                 ('souper jeudi à 6 h du soir', 'Souper', '06:00'),
                 ('souper jeudi à 6 h pm', 'Souper', '06:00'),
                 ('gym jeudi à 7 h am', 'Gym', '19:00'))
        for i, (brut, titre, debut) in enumerate(refus):
            with self.subTest(brut=brut, debut=debut):
                self.assertFalse(self._appel(brut, titre, debut, f'r:{i}').succes)
        acceptes = (('souper jeudi à 6 h du soir', 'Souper', '18:00'),
                    ("cours jeudi à 2 h de l'après-midi", 'Cours', '14:00'),
                    ('gym jeudi à 7 h du matin', 'Gym', '07:00'))
        for i, (brut, titre, debut) in enumerate(acceptes):
            with self.subTest(brut=brut, debut=debut):
                ScheduledBlock.objects.filter(user=self.user).delete()
                action = self._appel(brut, titre, debut, f'a:{i}')
                self.assertTrue(action.succes, action.message)

    def test_a_une_autre_heure_le_refus_nomme_les_deux_lectures(self):
        action = self._appel('gym jeudi à 7 h', 'Gym', '13:00', 'x:1')
        self.assertFalse(action.succes)
        self.assertIn('07:00', action.message)
        self.assertIn('19:00', action.message)


# ── F2: garder nie ──────────────────────────────────────────────────────────


class F2GarderNieTests(SimpleTestCase):

    def test_un_verbe_de_garde_nie_ne_ferme_pas_la_demande(self):
        nies = ('ne le garde pas', 'je veux pas le garder', 'pas besoin de le garder',
                'garde rien', 'laisse pas', 'je le garde pas', 'non, le garde pas',
                'je veux plus le garder', 'ne le conserve pas')
        for demande in (PORTEE, DESTR):
            for brut in nies:
                with self.subTest(motif=demande['motif'], brut=brut):
                    self.assertFalse(dem.annulation_libre(brut, demande))
                    self.assertIsNone(dem.option_choisie(brut, demande))
        self.assertIsNone(dem.option_choisie("n'arrête pas", MASSE))

    def test_garder_affirme_ferme_toujours(self):
        for brut in ('non, garde-le', 'Non, garde tout', 'laisse-le', 'laisse tomber',
                     'Je veux la garder chaque semaine', "n'efface rien, garde-le"):
            with self.subTest(brut=brut):
                self.assertEqual(dem.option_choisie(brut, PORTEE), 'annuler')


# ── F3: mecanique d'interface seulement ─────────────────────────────────────


CONSEILS = (
    "Choisis une matière à la fois.",
    "Choisis tes trois jours et l'heure.",
    "Remplis ta gourde avant le gym.",
    "Réponds par courriel à ton prof avant jeudi.",
    "Ajuste ton heure de coucher si besoin.",
    "Mets le lait dans la liste d'épicerie.",
    "Relis l'étendue de la matière avant l'examen.",
    "Sélectionne tes trois priorités du jour.",
    "Dans tes réponses d'examen, justifie chaque étape.",
    "L'horaire est affiché au babillard.",
    "Compare les heures proposées par ton prof.",
)
INTERFACE = (
    "Touche un des boutons ci-dessous.",
    "Appuie sur la puce qui te convient.",
    "Prends la puce du haut.",
    "Remplis le formulaire.",
    "Coche les jours voulus.",
    "Clique sur Confirmer.",
    "Remplis ces champs.",
    "Réponds « Tous les jeudis ».",
    "Le tout est pré-rempli à 2 h.",
)


class F3MecaniqueTests(SimpleTestCase):

    def _prose(self, phrase):
        return composer(ReponseDire(ouverture=phrase), Registre(), "", None).prose

    def test_un_conseil_ordinaire_reste(self):
        for phrase in CONSEILS:
            with self.subTest(phrase=phrase):
                self.assertEqual(self._prose(phrase), phrase)

    def test_la_mecanique_d_interface_tombe(self):
        for phrase in INTERFACE:
            with self.subTest(phrase=phrase):
                self.assertEqual(self._prose(phrase), "")


# ── F4: un oui clair confirme ce qui ne detruit rien ────────────────────────


class F4OuiClairTests(SimpleTestCase):

    OUI = ('oui', 'ok', 'ouais', 'vas-y', 'continue', 'go', "d'accord", 'Oui, vas-y',
           'oui merci', 'ok stp', 'Oui!', 'OK, continue', "oui s'il te plaît")

    def test_un_oui_clair_confirme_la_creation_en_masse(self):
        for brut in self.OUI:
            with self.subTest(brut=brut):
                self.assertEqual(dem.option_choisie(brut, MASSE), 'confirmer')

    def test_un_oui_qui_porte_autre_chose_ne_confirme_pas(self):
        for brut in ('oui, tous les jeudis', 'Oui, supprime ces trois blocs.',
                     'oui mais seulement ce jeudi', 'oui ?', 'ok pour yoga mais pas gym',
                     'oui, ajoute aussi du gym'):
            with self.subTest(brut=brut):
                self.assertNotEqual(dem.option_choisie(brut, MASSE), 'confirmer')

    def test_un_oui_ne_tranche_jamais_une_suppression_ni_un_creneau(self):
        for demande in (PORTEE, DESTR, PLAN, ANNULER_EVENEMENT, CRENEAU):
            for brut in self.OUI:
                with self.subTest(motif=demande['motif'], brut=brut):
                    self.assertIsNone(dem.option_choisie(brut, demande))


# ── F5: piece jointe, AGIR tourne ───────────────────────────────────────────


class F5PieceJointeTests(_TourDecideBase):

    def test_une_piece_jointe_interdit_le_chemin_rapide(self):
        self.decide = True
        doc = UploadedDocument.objects.create(user=self.user, file_name='horaire.pdf',
                                              processed=True)

        def _agir(self_agent, user, msg, registre):
            self.ordre.append('agir')
            return ''

        with patch.object(PlannerAgentV2, '_contexte_document', return_value=iter(())), \
             patch.object(PlannerAgentV2, '_agir', _agir), \
             patch.object(PlannerAgentV2, '_dire', return_value=ReponseDire(ouverture='Ok.')):
            list(PlannerAgentV2().process_message_stream(self.user, 'oui', doc))
        self.assertIn('agir', self.ordre)

    def test_sans_piece_jointe_le_chemin_rapide_reste(self):
        self.decide = True

        def _agir(*a, **k):
            raise AssertionError('AGIR ne doit pas tourner')

        with patch.object(PlannerAgentV2, '_agir', _agir), \
             patch.object(PlannerAgentV2, '_dire', return_value=ReponseDire(ouverture='Ok.')):
            list(PlannerAgentV2().process_message_stream(self.user, 'oui'))
        self.assertNotIn('agir', self.ordre)


# ── F6: une nouvelle demande destructive est une nouvelle requete ───────────


class F6NouvelleDestructionTests(HarnaisGardes, TransactionTestCase):

    NOUVELLES = ('efface tout', 'vide tout', 'supprime tous mes blocs', 'supprime tout ce jeudi')

    def test_ce_n_est_pas_une_reponse(self):
        for brut in self.NOUVELLES:
            with self.subTest(brut=brut):
                self.assertFalse(dem.reponse_plausible(brut, PORTEE))
        # Une portee dite en toutes lettres reste une reponse floue a reposer.
        self.assertTrue(dem.reponse_plausible('supprime tous les jeudis', PORTEE))

    def test_la_demande_est_abandonnee_et_rien_ne_s_execute(self):
        for i, brut in enumerate(self.NOUVELLES):
            with self.subTest(brut=brut):
                demande = puces(self.demande_portee(tache=f'f6:{i}'))
                self.attendre([demande], brut)
                registre = Registre()
                outils_v2.appliquer_choix_en_attente(self.user, registre, brut, f'f6:{i}')
                codes = [a.donnees.get('decision_code') for a in registre.actions
                         if (a.donnees or {}).get('decision_code')]
                self.assertEqual(codes, ['abandonnee'])
                self.assertFalse(any(a.succes for a in registre.actions))
                self.assertFalse(outils_v2.tour_entierement_decide_par_le_code(registre, brut))
                self.assertActif(self.q)
