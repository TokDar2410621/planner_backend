"""
Round 9, dernier fixeur. Ecrits AVANT les correctifs et vus en echec.

K1  cancel_scheduled_block sur plusieurs rangees: la cible stockait la
    premiere rangee et les ids seulement. Une deuxieme rangee deplacee le meme
    jour (meme ids, premiere rangee intacte) etait annulee a la puce.
K3  « autre heure » etait calculee sur tout le message: une heure d'un autre
    element (« et mon yoga a 9 h », « j'ai un rendez-vous a 10 h ») retirait
    l'heure dite du gym, et le modele deplacait le gym a n'importe quelle heure.
K4  un mot de moment de la journee complement d'un autre nom (« avant le
    souper », « mon quart du soir », « ma soiree ») ne choisit plus la lecture:
    seul un marqueur positif (« jeudi soir », « ce soir », « a 6 h le soir »)
    le fait. Sinon, les deux lectures valent.

Horloge du harnais: lundi 14 septembre 2026, 8 h (jour 0 = lundi).
"""
from datetime import date, time

from django.test import SimpleTestCase, TransactionTestCase

from core.models import RecurringBlock, ScheduledBlock, Task
from core.test_agent_v2_gardes import HarnaisGardes, puces
from services.agent_v2 import demandes as dem
from services.agent_v2 import outils as outils_v2
from services.agent_v2.registre import Registre

MERCREDI = '2026-09-16'
JEUDI = '2026-09-17'
OUI = 'Oui, je confirme.'


# ── K1: une photo par evenement ─────────────────────────────────────────────


class K1LotDEvenementsTests(HarnaisGardes, TransactionTestCase):

    def _ev(self, tache, debut, fin):
        return ScheduledBlock.objects.create(
            user=self.user, task=tache, date=date(2026, 9, 17),
            start_time=time.fromisoformat(debut), end_time=time.fromisoformat(fin))

    def _question(self):
        tache = Task.objects.create(user=self.user, title='Étude')
        a = self._ev(tache, '09:00', '10:00')
        b = self._ev(tache, '14:00', '15:00')
        demande = puces(self.premier_tour('annule mon étude jeudi', 'cancel_scheduled_block',
                                          tache='k1l:1', date=JEUDI,
                                          title='Étude').donnees['demande'])
        self.attendre([demande], OUI)
        return a, b

    def test_deuxieme_rangee_deplacee_le_meme_jour(self):
        a, b = self._question()
        ScheduledBlock.objects.filter(id=b.id).update(start_time=time(16, 0),
                                                      end_time=time(17, 0))
        registre = Registre()
        sorties = outils_v2.appliquer_choix_en_attente(self.user, registre, OUI, 'k1l:2')
        self.assertEqual(ScheduledBlock.objects.filter(id__in=[a.id, b.id]).count(), 2)
        self.assertFalse(any(x.succes and x.est_mutation for x in registre.actions))
        self.assertEqual(sorties[0]['decision_code'], 'abandonnee')
        changees = [x for x in registre.actions if (x.donnees or {}).get('cible_changee')]
        self.assertEqual(len(changees), 1)
        self.assertIn('a changé depuis ma question', changees[0].donnees['ligne_cible_changee'])
        self.assertIn("je n'ai rien annulé", changees[0].donnees['ligne_cible_changee'])

    def test_lot_inchange_s_annule(self):
        a, b = self._question()
        registre = Registre()
        sorties = outils_v2.appliquer_choix_en_attente(self.user, registre, OUI, 'k1l:2')
        self.assertEqual(sorties[0]['decision_code'], 'execute')
        self.assertFalse(ScheduledBlock.objects.filter(id__in=[a.id, b.id]).exists())


# ── K3: l'autre heure doit nommer le meme bloc ──────────────────────────────


class K3AutreHeureTests(HarnaisGardes, TransactionTestCase):

    def setUp(self):
        super().setUp()
        self.gym = self.bloc('Gym', 1, '19:00', '20:00', block_type='sport')
        self.bloc('Yoga', 3, '18:00', '19:00', block_type='sport')

    def _deplacer(self, brut, debut, tache):
        RecurringBlock.objects.filter(id=self.gym.id).update(start_time=time(19, 0),
                                                             end_time=time(20, 0))
        self.message_courant(brut)
        registre, tools = self.outils(brut, tache=tache)
        fin = f'{(int(debut[:2]) + 1) % 24:02d}{debut[2:]}'
        self.appeler(tools, 'update_block', block_id=self.gym.id, start_time=debut, end_time=fin)
        return registre.actions[-1]

    MESSAGES = ('déplace mon gym à 7 h et mon yoga à 9 h',
                "déplace mon gym à 7 h, j'ai un rendez-vous à 10 h",
                'déplace mon gym à 7 h. Mon cours finit à 9 h.')

    def test_une_heure_d_un_autre_element_ne_libere_pas_le_gym(self):
        for i, brut in enumerate(self.MESSAGES):
            with self.subTest(brut=brut):
                action = self._deplacer(brut, '15:00', f'k3a:r{i}')
                self.assertFalse(action.succes, action.message)
                self.assertIn('07:00', action.message)
                self.gym.refresh_from_db()
                self.assertEqual(self.gym.start_time, time(19, 0))

    def test_l_heure_dite_passe_toujours(self):
        for i, brut in enumerate(self.MESSAGES):
            with self.subTest(brut=brut):
                action = self._deplacer(brut, '07:00', f'k3a:a{i}')
                self.assertTrue(action.succes, action.message)

    def test_le_pronom_garde_l_heure_qui_nomme_le_bloc(self):
        action = self._deplacer('mon gym de 7 h, déplace-le à 9 h', '21:00', 'k3a:p')
        self.assertTrue(action.succes, action.message)


# ── K4: seul un marqueur positif choisit la lecture ─────────────────────────


def _lectures(texte):
    plat = dem.sans_accents(texte)
    return [outils_v2._lectures_d_heure(plat, ou, v) for v, ou in dem.heures_dites_positions(texte)]


class K4ComplementDeNomTests(SimpleTestCase):

    def test_complement_d_un_autre_nom_garde_les_deux_lectures(self):
        cas = (
            ('mets mon déjeuner mercredi à 8 h avant le souper', [['08:00', '20:00']]),
            ('mets mon gym mercredi à 7 h avant mon quart du soir', [['07:00', '19:00']]),
            ('rendez-vous mercredi à 9 h pour préparer ma soirée', [['09:00', '21:00']]),
            ('mon quart du soir à 7 h', [['07:00', '19:00']]),
            ('lecture à 9 h pas le soir', [['09:00', '21:00']]),
        )
        for texte, attendu in cas:
            with self.subTest(texte=texte):
                self.assertEqual(_lectures(texte), attendu)

    def test_marqueurs_positifs_toujours_stricts(self):
        cas = (
            ('souper jeudi soir à 6 h', [['18:00']]),
            ('mets mon souper jeudi à 6 h', [['18:00']]),
            ('gym jeudi matin à 5 h', [['05:00']]),
            ('gym à 6 h le soir', [['18:00']]),
            ('ce soir à 8 h', [['20:00']]),
            ('tous les matins à 7 h', [['07:00']]),
            ('mardi en soirée à 9 h', [['21:00']]),
        )
        for texte, attendu in cas:
            with self.subTest(texte=texte):
                self.assertEqual(_lectures(texte), attendu)


class K4ComplementGardeTests(HarnaisGardes, TransactionTestCase):

    CAS = (("mets mon déjeuner {j} à 8 h avant le souper", 'Déjeuner', '08:00'),
           ("mets mon gym {j} à 7 h avant mon quart du soir", 'Gym', '07:00'),
           ("rendez-vous {j} à 9 h pour préparer ma soirée", 'Rendez-vous', '09:00'))

    def _appel(self, brut, titre, jour, debut, tache):
        ScheduledBlock.objects.filter(user=self.user).delete()
        self.message_courant(brut)
        registre, tools = self.outils(brut, tache=tache)
        fin = f'{(int(debut[:2]) + 1) % 24:02d}{debut[2:]}'
        self.appeler(tools, 'schedule_task_at', title=titre, date=jour,
                     start_time=debut, end_time=fin)
        return registre.actions[-1]

    def test_l_heure_dite_est_creee(self):
        for nom_jour, jour in (('mercredi', MERCREDI), ('jeudi', JEUDI)):
            for i, (modele, titre, debut) in enumerate(self.CAS):
                brut = modele.format(j=nom_jour)
                with self.subTest(brut=brut):
                    action = self._appel(brut, titre, jour, debut, f'k4c:{nom_jour}{i}')
                    self.assertTrue(action.succes, action.message)
