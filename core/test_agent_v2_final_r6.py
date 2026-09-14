"""
Round 6, dernier correcteur. Chaque classe a ete ecrite AVANT son correctif et
vue en echec.

I3/I4 heure dite: une heure que l'utilisateur a DITE ailleurs dans le message
    (correction, destination d'un deplacement) n'est jamais refusee; l'heure
    actuelle d'un bloc deplace le nomme, elle ne dit pas ou il va; une heure
    d'un autre element (« apres mon cours a 15 h ») n'est pas celle du titre.
I4 chemin rapide: un oui ou un non en tete ne suffit plus a faire d'un message
    une reponse; « non, supprime plutot mon gym » est une nouvelle requete.
I2 brouillon: « que je viens d'ajouter » affirme une action, meme en question.
I4 lecture: une ligne d'abandon ne remplace plus la lecture demandee.

Horloge du harnais: lundi 14 septembre 2026, 8 h (jour 0 = lundi).
"""
from unittest.mock import patch

from django.test import SimpleTestCase, TransactionTestCase

from core.models import RecurringBlock, ScheduledBlock
from core.test_agent_v2_gardes import HarnaisGardes, puces
from core.test_agent_v2_narrateur import NarrateurBase, demande
from core.test_agent_v2_voix_r6 import LIGNE_ABANDON, abandonnee
from services.agent.tools.base import ToolResult
from services.agent_v2 import outils as outils_v2
from services.agent_v2 import redaction, rendu
from services.agent_v2.mesure import fuite_question, questions_et_offres
from services.agent_v2.redaction import ReponseDire
from services.agent_v2.registre import Registre


# ── I3/I4: l'heure dite ailleurs dans le message ────────────────────────────


class HeureDiteAilleursTests(HarnaisGardes, TransactionTestCase):

    def _appel(self, brut, nom, tache, **kwargs):
        self.message_courant(brut)
        registre, tools = self.outils(brut, tache=tache)
        self.appeler(tools, nom, **kwargs)
        return registre.actions[-1]

    def test_une_correction_dans_le_message_passe(self):
        cas = (("Ajoute gym jeudi à 15 h, en fait non, à 17 h", 'Gym', '17:00', '18:00'),
               ("Gym jeudi à 15 h. Non, change pour 17 h.", 'Gym', '17:00', '18:00'),
               ("mets le dentiste jeudi à 15 h, finalement plutôt 16 h", 'Dentiste',
                '16:00', '17:00'))
        for i, (brut, titre, debut, fin) in enumerate(cas):
            with self.subTest(brut=brut):
                # Chaque cas sur un jeudi vide: deux Gym a 17 h se chevaucheraient.
                ScheduledBlock.objects.filter(user=self.user).delete()
                action = self._appel(brut, 'schedule_task_at', f'c:{i}', title=titre,
                                     date='2026-09-17', start_time=debut, end_time=fin)
                self.assertTrue(action.succes, action.message)
                self.assertNotIn('heure_dite', action.donnees)
                self.assertEqual(ScheduledBlock.objects.filter(
                    user=self.user, start_time=debut).count(), 1)

    def test_deplacer_un_bloc_nomme_par_son_heure_actuelle(self):
        gym = self.bloc('Gym', 2, '15:00', '16:00', block_type='sport')
        for i, brut in enumerate(("Le gym de 15 h, tu peux le mettre à 17 h ?",
                                  "Mon gym de 15 h, déplace-le à 17 h.",
                                  "Mon cours de gym de 15 h, mets-le à 17 h stp")):
            with self.subTest(brut=brut):
                RecurringBlock.objects.filter(pk=gym.pk).update(start_time='15:00',
                                                                end_time='16:00')
                action = self._appel(brut, 'update_block', f'u:{i}', block_id=gym.id,
                                     start_time='17:00', end_time='18:00')
                self.assertTrue(action.succes, action.message)
                gym.refresh_from_db()
                self.assertEqual(gym.start_time.strftime('%H:%M'), '17:00')

    def test_l_heure_d_un_autre_element_n_est_pas_celle_du_titre(self):
        cas = (('ajoute gym jeudi une heure après mon cours à 15 h', 'Gym', '16:00'),
               ('mets gym jeudi avant mon cours de 15 h', 'Gym', '13:00'),
               # Le quart du harnais occupe jeudi 19 h a minuit.
               ('place ma lecture jeudi après le souper à 17 h', 'Lecture', '18:00'))
        for i, (brut, titre, debut) in enumerate(cas):
            with self.subTest(brut=brut):
                fin = f'{int(debut[:2]) + 1:02d}:00'
                action = self._appel(brut, 'schedule_task_at', f'b:{i}', title=titre,
                                     date='2026-09-17', start_time=debut, end_time=fin)
                self.assertTrue(action.succes, action.message)

    def test_une_heure_jamais_dite_reste_refusee(self):
        """Le correctif ne rouvre pas le trou: l'appel qui pose une heure que
        l'utilisateur n'a dite nulle part est toujours retenu."""
        cas = (("Ajoute gym jeudi à 15 h, en fait non, à 17 h", 'Gym', '13:00'),
               ('ajoute gym jeudi à 15 h', 'Gym', '13:00'),
               ('mets le dentiste jeudi à 15 h', 'Dentiste', '16:00'))
        for i, (brut, titre, debut) in enumerate(cas):
            with self.subTest(brut=brut):
                fin = f'{int(debut[:2]) + 1:02d}:00'
                action = self._appel(brut, 'schedule_task_at', f'r:{i}', title=titre,
                                     date='2026-09-17', start_time=debut, end_time=fin)
                self.assertFalse(action.succes, action.message)
        self.assertEqual(ScheduledBlock.objects.filter(user=self.user).count(), 0)


# ── I4: le chemin rapide n'avale plus une correction ────────────────────────


class CheminRapideCorrectionTests(HarnaisGardes, TransactionTestCase):

    CORRECTIONS = ("non, supprime plutôt mon gym",
                   "non c'est le gym que je voulais enlever",
                   "oui et enlève aussi le gym",
                   "ok, le gym aussi",
                   "non pas lui, le gym")

    def _decide(self, demandes, brut, tache):
        self.attendre(demandes, brut)
        registre = Registre()
        outils_v2.appliquer_choix_en_attente(self.user, registre, brut, tache)
        codes = [(a.donnees or {}).get('decision_code') for a in registre.actions]
        return outils_v2.tour_entierement_decide_par_le_code(registre, brut), codes

    def test_une_correction_qui_nomme_un_autre_element_laisse_agir_tourner(self):
        self.bloc('Gym', 3, '17:00', '18:00', block_type='sport')
        for reemissions in (0, 1):
            for i, brut in enumerate(self.CORRECTIONS):
                with self.subTest(brut=brut, reemissions=reemissions):
                    RecurringBlock.all_objects.filter(pk=self.q.pk).update(active=True)
                    dem = puces(self.demande_portee(tache=f'z:{reemissions}:{i}'))
                    dem['reemissions'] = reemissions
                    decide, codes = self._decide([dem], brut, f'z:{reemissions}:{i}')
                    self.assertFalse(decide)
                    self.assertNotIn('reposee', codes)
                    self.assertActif(self.q)

    def test_un_oui_vague_reste_repose(self):
        dem = puces(self.demande_portee())
        for i, brut in enumerate(('oui', 'ok', 'non ?', 'oui oui')):
            with self.subTest(brut=brut):
                decide, codes = self._decide([dem], brut, f'o:{i}')
                self.assertTrue(decide)
                self.assertEqual(codes, ['reposee'])


# ── I2: « je viens d'ajouter » sous une tete de question ────────────────────


class BrouillonJeViensDeTests(SimpleTestCase):

    PHRASES = ("Tu veux un rappel pour le gym que je viens d'ajouter ?",
               "Quelle heure veux-tu pour le gym que je viens d'ajouter ?",
               "Veux-tu garder le cours que je viens d'effacer ?",
               "Tu veux un rappel pour le gym que je viens tout juste de placer ?",
               "Veux-tu que je déplace aussi le cours que je viens d'enlever ?")

    def test_le_brouillon_tombe(self):
        for phrase in self.PHRASES:
            with self.subTest(phrase=phrase):
                self.assertEqual(questions_et_offres(phrase), "")

    def test_fuite_question_la_voit(self):
        for phrase in self.PHRASES:
            with self.subTest(phrase=phrase):
                self.assertTrue(fuite_question(phrase))

    def test_une_offre_ordinaire_passe_toujours(self):
        for phrase in ("Veux-tu que je le déplace, ou tu préfères vendredi ?",
                       "Tu veux que je vienne te le rappeler jeudi ?"):
            with self.subTest(phrase=phrase):
                self.assertEqual(questions_et_offres(phrase), phrase)


# ── I4: la ligne d'abandon ne remplace plus la lecture ──────────────────────


def _lecture_mardi(registre):
    registre.ajouter("list_blocks", {"day_of_week": 1}, ToolResult(
        success=True, message="ok",
        data={"blocks": [{"title": "Programmation", "start_time": "10:00",
                          "end_time": "12:50", "day_of_week": 1},
                         {"title": "Anglais", "start_time": "14:00",
                          "end_time": "15:50", "day_of_week": 1}]}))


class AbandonEtLectureTests(SimpleTestCase):

    def test_la_lecture_suit_la_ligne_d_abandon(self):
        registre = Registre()
        registre.ajouter("delete_block", {}, abandonnee(demande("portee_jour", "p1")))
        _lecture_mardi(registre)
        texte = redaction.bloc_factuel(registre)
        self.assertIn(LIGNE_ABANDON, texte)
        self.assertIn("Programmation", texte)
        self.assertIn("Anglais", texte)
        self.assertLess(texte.index(LIGNE_ABANDON), texte.index("Programmation"))

    def test_sans_lecture_la_ligne_seule(self):
        registre = Registre()
        registre.ajouter("delete_block", {}, abandonnee(demande("portee_jour", "p1")))
        self.assertEqual(redaction.bloc_factuel(registre), LIGNE_ABANDON)

    def test_une_mutation_reussie_garde_les_faits_seuls(self):
        registre = Registre()
        registre.ajouter("delete_block", {}, abandonnee(demande("portee_jour", "p1")))
        _lecture_mardi(registre)
        registre.ajouter("create_block", {"title": "Gym"}, ToolResult(
            success=True, message="ok",
            data={"block": {"id": 3, "title": "Gym", "day_of_week": 1,
                            "start_time": "18:00", "end_time": "19:00"}}))
        texte = redaction.bloc_factuel(registre)
        self.assertIn(LIGNE_ABANDON, texte)
        self.assertNotIn("Programmation", texte)

    def test_plusieurs_abandons_une_seule_ligne(self):
        registre = Registre()
        registre.ajouter("delete_block", {}, abandonnee(
            demande("portee_jour", "p1", cible={"titre": "Chimie générale (labo)", "jour": 1})))
        registre.ajouter("delete_block", {}, abandonnee(demande("portee_jour", "p2")))
        faits = rendu.rendre_faits(registre)
        self.assertEqual(faits, "Je laisse tomber la suppression de Chimie générale (labo) et "
                                "de Quart au dépanneur. Redis-le si tu veux toujours.")
        self.assertEqual(faits.count("laisse tomber"), 1)


class AbandonEtLectureAuTourTests(NarrateurBase):

    def setUp(self):
        super().setUp()
        patcheur = patch.object(redaction, "_charger_rendu", return_value=rendu)
        patcheur.start()
        self.addCleanup(patcheur.stop)

    def test_horaire_demande_pendant_une_question_en_attente(self):
        registre_vu = Registre()
        registre_vu.ajouter("delete_block", {}, abandonnee(demande("portee_jour", "p1")))
        _lecture_mardi(registre_vu)
        actions = [(a.outil, a.parametres, ToolResult(success=a.succes, message=a.message,
                                                       data=a.donnees))
                   for a in registre_vu.actions]
        _, done = self.tour(actions=actions, message="c'est quoi mon horaire demain ?",
                            dire=ReponseDire(ouverture="Voilà ta journée."))
        self.assertIn(LIGNE_ABANDON, done["response"])
        self.assertIn("Programmation", done["response"])
        self.assertEqual(self.metadonnees()["demandes"], [])
