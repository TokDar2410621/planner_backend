"""
Les suggestions proactives se lisent en HEURE MURALE, et jamais en UTC.

Trois defauts mesures le 2026-08-29, tous invisibles en journee et tous
absurdes pour l'utilisateur:

1. `task.deadline.date()` rend la date UTC. Une echeance a 23:59 a Montreal
   vaut 03:59 UTC le lendemain: une tache due AUJOURD'HUI etait annoncee
   « est due demain », echeance dans 17 heures.
2. `timezone.now().date()` bascule sur demain des 20 h heure murale. Tout le
   jeu de suggestions du soir portait donc sur le lendemain, sans le dire.
3. La fenetre des creneaux etait figee a 8 h - 22 h, sans notion d'heure
   courante, et le trou etait rendu ENTIER. Sur une journee vide cela donnait
   « Tu as 840 minutes de libre de 08:00 a 22:00 », et un clic envoyait a
   l'agent « Planifie X de 08:00 a 22:00 »: un bloc de quatorze heures.

Le temps est FIGE dans chaque cas: sans cela ces tests passeraient le matin et
echoueraient le soir, ce qui est exactement le bogue qu'ils surveillent.
"""
from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone

from core.models import Task
from services.ai_insights import AIInsightsService

MONTREAL = ZoneInfo("America/Toronto")


def a(heure: int, minute: int = 0, jour: int = 29):
    """Un instant a l'heure MURALE de Montreal, en aout (donc EDT, UTC-4)."""
    return datetime(2026, 8, jour, heure, minute, tzinfo=MONTREAL)


class HeureMuraleTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='murale', password='x')

    def _suggestions(self, maintenant):
        """Les suggestions telles que l'utilisateur les verrait a cet instant."""
        with patch.object(timezone, 'now', return_value=maintenant):
            return AIInsightsService().get_proactive_suggestions(self.user, limit=5)

    def _tache(self, titre, echeance, minutes=60, type_='deep_work'):
        return Task.objects.create(
            user=self.user, title=titre, completed=False, deadline=echeance,
            estimated_duration_minutes=minutes, priority=8, task_type=type_,
        )

    # ── 1. L'echeance ────────────────────────────────────────────────────
    def test_une_tache_due_ce_soir_n_est_pas_dite_due_demain(self):
        """Le defaut d'origine: 23:59 heure murale tombe demain en UTC."""
        self._tache("Rendre le TP de stats", a(23, 59))
        messages = [s.message for s in self._suggestions(a(6, 12))
                    if s.type == 'reminder']
        self.assertTrue(messages, "aucun rappel produit")
        self.assertNotIn('demain', ' '.join(messages))
        self.assertIn("aujourd'hui", ' '.join(messages))

    def test_une_tache_due_ce_soir_n_est_pas_dite_en_retard_le_matin(self):
        """« En retard » se juge sur l'INSTANT, pas sur la date."""
        self._tache("Rendre le TP de stats", a(23, 59))
        messages = [s.message for s in self._suggestions(a(8, 0))
                    if s.type == 'reminder']
        self.assertNotIn('en retard', ' '.join(messages))

    def test_une_echeance_reellement_passee_est_dite_en_retard(self):
        """Contre-epreuve: on n'a pas simplement supprime le cas."""
        self._tache("Rendre le TP de stats", a(9, 0))
        messages = [s.message for s in self._suggestions(a(14, 0))
                    if s.type == 'reminder']
        self.assertIn('en retard', ' '.join(messages))

    def test_une_tache_due_demain_est_bien_dite_demain(self):
        self._tache("Presentation d'histoire", a(10, 0, jour=30))
        messages = [s.message for s in self._suggestions(a(14, 0))
                    if s.type == 'reminder']
        self.assertIn('demain', ' '.join(messages))

    # ── 2. La bascule du soir ────────────────────────────────────────────
    def test_a_vingt_heures_la_journee_visee_reste_aujourd_hui(self):
        """A 20 h a Montreal il est minuit passe a Greenwich. La date UTC
        aurait fait sauter tout le jeu de suggestions au lendemain."""
        self._tache("Rendre le TP de stats", a(23, 59))
        messages = [s.message for s in self._suggestions(a(20, 30))
                    if s.type == 'reminder']
        self.assertTrue(messages, "aucun rappel produit a 20 h 30")
        self.assertNotIn('demain', ' '.join(messages))

    # ── 3. Les creneaux ──────────────────────────────────────────────────
    def test_un_creneau_deja_passe_n_est_pas_propose(self):
        """A 16 h, le moteur proposait encore « de 08:00 a 22:00 »."""
        self._tache("Reviser la geologie", a(10, 0, jour=31))
        creneaux = [s for s in self._suggestions(a(16, 0)) if s.type == 'free_time']
        self.assertTrue(creneaux, "aucun creneau propose a 16 h")
        for s in creneaux:
            debut = s.metadata['gap_start']
            self.assertGreaterEqual(debut, '16:00', f"creneau dans le passe: {s.message}")

    def test_le_creneau_propose_fait_la_taille_de_la_tache(self):
        """Sur une journee vide le trou fait 14 h. Proposer le trou entier
        demandait a l'agent un bloc de quatorze heures."""
        self._tache("Reviser la geologie", a(10, 0, jour=31), minutes=90)
        creneaux = [s for s in self._suggestions(a(9, 0)) if s.type == 'free_time']
        self.assertTrue(creneaux, "aucun creneau propose")
        s = creneaux[0]
        debut = datetime.strptime(s.metadata['gap_start'][:5], '%H:%M')
        fin = datetime.strptime(s.metadata['gap_end'][:5], '%H:%M')
        self.assertEqual(fin - debut, timedelta(minutes=90))

    def test_aucun_creneau_quand_la_journee_est_finie(self):
        """A 22 h il ne reste rien a proposer avant la fin de la fenetre."""
        self._tache("Reviser la geologie", a(10, 0, jour=31))
        creneaux = [s for s in self._suggestions(a(22, 0)) if s.type == 'free_time']
        self.assertEqual(creneaux, [])

    def test_le_message_ne_promet_plus_des_centaines_de_minutes(self):
        """Garde-fou de formulation: « 840 minutes de libre » etait le
        symptome visible du trou rendu entier."""
        self._tache("Reviser la geologie", a(10, 0, jour=31), minutes=60)
        for s in self._suggestions(a(9, 0)):
            if s.type == 'free_time':
                self.assertNotIn('minutes de libre', s.message)
