"""Remplacer une occurrence : un seul geste, ou rien.

Defaut B, mesure en production le 2026-09-29 sur « J'ai examen a la place du
cours d'entreprise » (cours recurrent le mardi de 13 h a 16 h, un mardi) :
5 essais, 4 comportements distincts.

  1 fois correct : occurrence sautee, examen place de 13 h a 16 h.
  2 fois l'occurrence sautee SANS examen place (« Bonne chance pour
    l'examen ! ») : la personne perd son cours et n'a rien a la place.
  1 fois le bloc recurrent RENOMME « Examen - L'entreprise... » : permanent,
    tous les mardis.
  1 fois deux examens, aujourd'hui et demain.

Le bon geste demandait au modele d'enchainer deux outils dans le bon ordre,
sans raisonnement (r0). replace_block_occurrence en fait UN seul geste,
atomique, et libere le creneau avant de placer.

Aucun appel reseau : les outils sont appeles directement, comme dans les
scenarios FR, et le juge est scripte.
"""
import asyncio
from datetime import date, time, timedelta

from django.contrib.auth.models import User
from django.test import TransactionTestCase
from django.utils import timezone

from core.models import (RecurringBlock, RecurringBlockException, ScheduledBlock,
                         Task)
from services.agent_v2.outils import outils_pour
from services.agent_v2.registre import Registre
from services.agent_v2.rendu import rendre_faits


class RemplacementTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='remplacement', password='x')
        self.jour = timezone.localdate()
        self.cours = RecurringBlock.objects.create(
            user=self.user, title="L’entreprise et ses systèmes", block_type='course',
            day_of_week=self.jour.weekday(), start_time='13:00', end_time='16:00')

    def _outils(self, message=""):
        registre = Registre()
        outils = {t.name: t for t in outils_pour(
            self.user, registre, message_du_tour=message, tache='test:1',
            message_brut=message)}
        return registre, outils

    def _appeler(self, registre, outils, nom, **kwargs):
        asyncio.run(outils[nom].function_schema.function(**kwargs))
        return registre.actions[-1]

    def test_un_seul_geste_libere_et_place(self):
        registre, outils = self._outils("J'ai examen à la place du cours d'entreprise")
        action = self._appeler(registre, outils, 'replace_block_occurrence',
                               date=self.jour.isoformat(), replacement_title='Examen',
                               title='entreprise', block_type='course')
        self.assertTrue(action.succes, action.message)

        # ETAT FINAL : l'occurrence de ce jour est ignoree, l'examen occupe le
        # creneau libere, et la serie reste intacte pour les autres semaines.
        self.assertTrue(RecurringBlockException.objects.filter(
            user=self.user, recurring_block=self.cours, date=self.jour).exists())
        place = ScheduledBlock.objects.get(user=self.user, date=self.jour,
                                           task__title='Examen')
        self.assertEqual((place.start_time, place.end_time), (time(13, 0), time(16, 0)))
        self.cours.refresh_from_db()
        self.assertTrue(self.cours.active)
        self.assertEqual(self.cours.title, "L’entreprise et ses systèmes")
        # Un seul reçu, pour un seul geste.
        self.assertEqual(len(registre.recus_confirmes()), 1)

    def test_les_heures_du_bloc_remplace_sont_reprises_par_defaut(self):
        registre, outils = self._outils()
        self._appeler(registre, outils, 'replace_block_occurrence',
                      date=self.jour.isoformat(), replacement_title='Examen',
                      title='entreprise')
        place = ScheduledBlock.objects.get(user=self.user, task__title='Examen')
        self.assertEqual((place.start_time, place.end_time), (time(13, 0), time(16, 0)))

    def test_le_fait_rendu_dit_ce_qui_prend_la_place_et_ce_qui_reste(self):
        registre, outils = self._outils()
        self._appeler(registre, outils, 'replace_block_occurrence',
                      date=self.jour.isoformat(), replacement_title='Examen',
                      title='entreprise')
        faits = rendre_faits(registre)
        self.assertIn('Examen prend la place de', faits)
        self.assertIn("L’entreprise et ses systèmes", faits)
        self.assertIn('13 h', faits)
        self.assertIn('restent', faits)
        self.assertNotIn('—', faits)

    def test_sans_bloc_ce_jour_rien_n_est_touche(self):
        libre = self.jour + timedelta(days=1)
        if libre.weekday() == self.jour.weekday():  # garde-fou theorique
            libre += timedelta(days=1)
        registre, outils = self._outils()
        action = self._appeler(registre, outils, 'replace_block_occurrence',
                               date=libre.isoformat(), replacement_title='Examen')
        self.assertFalse(action.succes)
        self.assertFalse(RecurringBlockException.objects.exists())
        self.assertFalse(ScheduledBlock.objects.exists())

    def test_plusieurs_blocs_le_meme_jour_demandent_lequel(self):
        RecurringBlock.objects.create(
            user=self.user, title="Conception d’applications", block_type='course',
            day_of_week=self.jour.weekday(), start_time='08:00', end_time='11:00')
        registre, outils = self._outils()
        action = self._appeler(registre, outils, 'replace_block_occurrence',
                               date=self.jour.isoformat(), replacement_title='Examen',
                               block_type='course')
        self.assertFalse(action.succes)
        self.assertIn('candidates', action.donnees)
        self.assertFalse(RecurringBlockException.objects.exists())
        self.assertFalse(ScheduledBlock.objects.exists())

    def test_un_placement_impossible_laisse_l_occurrence_en_place(self):
        """Tout ou rien: sans remplacant place, la personne garde son cours.

        Un evenement ponctuel deja verrouille sur le creneau fait echouer le
        placement; l'occurrence ne doit pas rester sautee pour autant.
        """
        autre = Task.objects.create(user=self.user, title='Rendez-vous')
        ScheduledBlock.objects.create(user=self.user, task=autre, date=self.jour,
                                      start_time='13:00', end_time='16:00', locked=True)
        registre, outils = self._outils()
        action = self._appeler(registre, outils, 'replace_block_occurrence',
                               date=self.jour.isoformat(), replacement_title='Examen',
                               title='entreprise')
        self.assertFalse(action.succes)
        self.assertFalse(RecurringBlockException.objects.filter(
            user=self.user, recurring_block=self.cours, date=self.jour).exists())
        self.assertFalse(ScheduledBlock.objects.filter(task__title='Examen').exists())

    def test_l_outil_reste_invisible_a_la_v1(self):
        """v1 ne sait pas rendre ce fait: il ne doit pas voir l'outil."""
        from services.agent.tools import V2_SEULEMENT, get_tools_for_claude
        self.assertIn('replace_block_occurrence', V2_SEULEMENT)
        noms = {o['name'] for o in get_tools_for_claude()}
        self.assertNotIn('replace_block_occurrence', noms)


class SautSansRemplacantTests(TransactionTestCase):
    """Un saut d'occurrence sans rien a la place ne finit pas par « bonne chance ».

    Prod du 2026-09-29, 2 essais sur 5: l'occurrence sautee, aucun examen
    place, et la reponse souhaitait bonne chance. La boucle est SIMULEE et ne
    fait que le demi-geste, comme le vrai modele ce jour-la.
    """

    def setUp(self):
        from core.test_agent_v2_jugement import juger_script  # noqa: F401
        self.user = User.objects.create_user(username='demi-geste', password='x')
        self.jour = timezone.localdate()
        self.cours = RecurringBlock.objects.create(
            user=self.user, title="L’entreprise et ses systèmes", block_type='course',
            day_of_week=self.jour.weekday(), start_time='13:00', end_time='16:00')
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def _boucle_qui_saute_seulement(self, prose):
        """La boucle saute l'occurrence, puis souhaite bonne chance."""
        from services.agent.tools import TOOL_MAP
        from services.agent_v2.redaction import ReponseDire

        jour = self.jour.isoformat()

        def _boucle(self_agent, user, message, registre):
            resultat = TOOL_MAP['skip_block_occurrence'].execute(
                user, date=jour, title='entreprise', block_type='course')
            registre.ajouter('skip_block_occurrence',
                             {'date': jour, 'title': 'entreprise'}, resultat)
            return ReponseDire(ouverture=prose, refs=['a1'])
        return _boucle

    def _tour(self, message, juge):
        from unittest.mock import patch
        with patch('services.agent_v2.jugement.juger', juge), \
                patch.object(self.Agent, '_boucle',
                             self._boucle_qui_saute_seulement("Bonne chance !")):
            return self.Agent().process_message(self.user, message)

    def test_le_code_demande_ce_qui_prend_la_place(self):
        from core.test_agent_v2_jugement import juger_script
        message = "J'ai examen à la place du cours d'entreprise"
        res = self._tour(message, juger_script(
            {message: {"remplacant": ("oui", 0.95)}}))
        self.assertIn('à la place', res['response'])
        self.assertIn('13 h', res['response'])

    def test_un_simple_retrait_ne_declenche_aucune_question(self):
        """« annule mon cours de mardi »: rien ne prend la place, on se taît."""
        from core.test_agent_v2_jugement import juger_script
        message = "Annule mon cours d'entreprise aujourd'hui"
        res = self._tour(message, juger_script(
            {message: {"remplacant": ("non", 0.95)}}))
        self.assertNotIn('à la place', res['response'])

    def test_sans_decision_du_juge_on_ne_demande_rien(self):
        from core.test_agent_v2_jugement import juger_script
        res = self._tour("J'ai examen à la place du cours", juger_script({}))
        self.assertNotIn('à la place, de', res['response'])


class RenommageDeLaSerieTests(TransactionTestCase):
    """Un remplacement ponctuel ne renomme jamais toute la serie.

    Prod du 2026-09-29, un essai sur cinq: le bloc recurrent renomme
    « Examen - L'entreprise et ses systemes », pour tous les mardis.

    Un premier correctif refusait l'appel en nommant le bon outil, mais il
    DEPENDAIT du juge: juge muet, renommage silencieux. La garde de portee
    (core/test_agent_v2_portee_changement.py) ne depend de lui que pour EVITER
    la question. Ici on verifie l'essentiel: la serie ne change pas sans que
    la portee soit tranchee.
    """

    def setUp(self):
        self.user = User.objects.create_user(username='renommage', password='x')
        self.jour = timezone.localdate()
        self.cours = RecurringBlock.objects.create(
            user=self.user, title="L’entreprise et ses systèmes", block_type='course',
            day_of_week=self.jour.weekday(), start_time='13:00', end_time='16:00')

    def _appeler_update(self, message, juge, titre='Examen'):
        from unittest.mock import patch
        registre = Registre()
        with patch('services.agent_v2.jugement.juger', juge):
            outils = {t.name: t for t in outils_pour(
                self.user, registre, message_du_tour=message, tache='test:renom',
                message_brut=message)}
            asyncio.run(outils['update_block'].function_schema.function(
                block_id=self.cours.id, title=titre))
        return registre.actions[-1]

    def test_sans_juge_la_serie_n_est_pas_renommee_en_silence(self):
        from core.test_agent_v2_jugement import juger_script
        message = "J'ai examen à la place du cours d'entreprise"
        action = self._appeler_update(message, juger_script({}))
        self.assertFalse(action.succes)
        self.assertEqual((action.donnees or {}).get('demande', {}).get('motif'),
                         'portee_changement')
        self.cours.refresh_from_db()
        self.assertEqual(self.cours.title, "L’entreprise et ses systèmes")

    def test_un_vrai_renommage_passe_quand_le_juge_voit_la_serie(self):
        from core.test_agent_v2_jugement import juger_script
        message = "Renomme mon cours d'entreprise en Systèmes d'entreprise"
        action = self._appeler_update(message, juger_script(
            {message: {"portee": ("serie", 0.95)}}),
            titre="Systèmes d'entreprise")
        self.assertTrue(action.succes, action.message)
        self.cours.refresh_from_db()
        self.assertEqual(self.cours.title, "Systèmes d'entreprise")


class CandidatsDuRemplacementTests(RemplacementTests):
    """Deux cours le meme jour: l'utilisateur doit LIRE qu'on lui demande lequel.

    Sans l'outil dans le tuple des candidats (rendu.py), la personne lisait
    « Je n'ai pas pu remplacer ce creneau pour une fois, rien n'a change. »,
    sans la question. Le tour etait perdu alors que l'outil avait rendu les
    candidats.
    """

    def test_le_refus_nomme_les_candidats_et_demande_lequel(self):
        RecurringBlock.objects.create(
            user=self.user, title="Conception d’applications", block_type='course',
            day_of_week=self.jour.weekday(), start_time='08:00', end_time='11:00')
        registre, outils = self._outils()
        self._appeler(registre, outils, 'replace_block_occurrence',
                      date=self.jour.isoformat(), replacement_title='Examen',
                      block_type='course')
        faits = rendre_faits(registre)
        self.assertIn('dis-moi lequel', faits)
        self.assertIn("Conception d’applications", faits)
        self.assertIn("L’entreprise et ses systèmes", faits)


class GroupementDesFaitsTests(TransactionTestCase):
    """Au-dela de cinq faits, les faits d'une meme famille se regroupent.

    Trouve par relecture croisee et reproduit: la famille du remplacement
    n'etait pas declaree, et l'acces au tableau des familles n'etait pas
    protege. Six faits dont deux remplacements levaient KeyError, et la
    reponse ENTIERE tombait (rendre_faits est appelee sans filet).
    """

    def _registre(self, famille_inconnue=False):
        from services.agent.tools.base import ToolResult
        r = Registre()
        for i in range(2):
            r.ajouter('replace_block_occurrence',
                      {'date': '2026-09-29', 'replacement_title': f'Examen {i}'},
                      ToolResult(success=True, message='ok', data={
                          'date': '2026-09-29',
                          'remplace': {'title': f'Cours {i}', 'start_time': '13:00',
                                       'end_time': '16:00'},
                          'scheduled_block': {'title': f'Examen {i}',
                                              'date': '2026-09-29',
                                              'start_time': '13:00',
                                              'end_time': '16:00'}}))
        for i in range(4):
            r.ajouter('create_task', {'title': f'Tâche {i}'},
                      ToolResult(success=True, message='ok',
                                 data={'task': {'title': f'Tâche {i}'}}))
        return r

    def test_deux_remplacements_parmi_six_faits_se_regroupent(self):
        faits = rendre_faits(self._registre())
        self.assertIn('séances remplacées', faits)
        self.assertIn('Examen 0', faits)
        self.assertIn('Examen 1', faits)

    def test_une_famille_inconnue_ne_fait_pas_tomber_la_reponse(self):
        """Le garde ne bloque jamais en silence: les faits restent detailles."""
        from unittest.mock import patch

        from services.agent_v2 import rendu
        familles = dict(rendu._FAMILLES)
        familles.pop('occurrence_remplacee')
        with patch.object(rendu, '_FAMILLES', familles):
            faits = rendre_faits(self._registre())
        self.assertIn('Examen 0', faits)
        self.assertIn('Examen 1', faits)
        self.assertIn('tâches ajoutées', faits)


class ConflitDuRemplacementTests(RemplacementTests):
    """Un placement refuse dit POURQUOI, il ne tombe pas dans l'echec generique."""

    def test_le_refus_pour_conflit_nomme_ce_qui_bloque(self):
        autre = Task.objects.create(user=self.user, title='Rendez-vous')
        ScheduledBlock.objects.create(user=self.user, task=autre, date=self.jour,
                                      start_time='13:00', end_time='16:00', locked=True)
        registre, outils = self._outils()
        self._appeler(registre, outils, 'replace_block_occurrence',
                      date=self.jour.isoformat(), replacement_title='Examen',
                      title='entreprise')
        faits = rendre_faits(registre)
        self.assertIn('Examen', faits)
        self.assertIn('Rendez-vous', faits)
