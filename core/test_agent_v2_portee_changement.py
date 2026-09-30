"""Changer le titre ou les heures d'une serie demande sa portee.

update_block porte sur TOUTE la serie. Mesure en production le 2026-09-29 :
« J'ai examen a la place du cours d'entreprise » a renomme le bloc recurrent
« Examen - L'entreprise et ses systemes », pour tous les mardis.

Le lot 3 a pose un refus, mais il DEPEND du juge : juge indisponible et le
renommage silencieux revient. Cette garde est l'inverse : active par defaut,
le code demande la portee, et le juge ne sert qu'a eviter une question quand
il voit clairement que la serie entiere est visee.

Aucun appel reseau : le juge est scripte.
"""
import asyncio

from django.contrib.auth.models import User
from django.test import TransactionTestCase
from django.utils import timezone

from core.models import RecurringBlock
from core.test_agent_v2_jugement import juger_script
from services.agent_v2.outils import outils_pour
from services.agent_v2.registre import Registre
from services.agent_v2.rendu import rendre_demandes


class PorteeDuChangementTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='portee', password='x')
        self.jour = timezone.localdate()
        self.cours = RecurringBlock.objects.create(
            user=self.user, title="L’entreprise et ses systèmes", block_type='course',
            day_of_week=self.jour.weekday(), start_time='13:00', end_time='16:00')

    def _appeler(self, message, juge, **kwargs):
        from unittest.mock import patch
        registre = Registre()
        with patch('services.agent_v2.jugement.juger', juge):
            outils = {t.name: t for t in outils_pour(
                self.user, registre, message_du_tour=message, tache='test:portee',
                message_brut=message)}
            asyncio.run(outils['update_block'].function_schema.function(
                block_id=self.cours.id, **kwargs))
        return registre

    def _demandes(self, registre):
        return [d for a in registre.actions
                for d in [(a.donnees or {}).get('demande')] if d]

    def test_un_renommage_est_retenu_et_la_portee_est_demandee(self):
        registre = self._appeler("Change le cours d'entreprise en Examen",
                                 juger_script({}), title='Examen')
        action = registre.actions[-1]
        self.assertFalse(action.succes)
        self.cours.refresh_from_db()
        self.assertEqual(self.cours.title, "L’entreprise et ses systèmes")
        demandes = self._demandes(registre)
        self.assertTrue(demandes)
        self.assertEqual(demandes[0]['motif'], 'portee_changement')

    def test_un_changement_d_heures_seul_garde_sa_propre_garde(self):
        """Les heures sont hors perimetre, et c'est voulu.

        Elles ont deja la garde « heure dite » (round r9), qui verifie que le
        modele respecte l'heure demandee. Doubler d'une question de portee
        masquerait ce refus plus precis, et un tour ne pose qu'une question.
        La portee d'un changement d'heures reste un manque connu.
        """
        registre = self._appeler("Décale mon cours d'entreprise à midi",
                                 juger_script({}), start_time='12:00', end_time='15:00')
        demandes = self._demandes(registre)
        self.assertFalse(any(d['motif'] == 'portee_changement' for d in demandes))

    def test_la_question_parle_de_changer_et_non_d_enlever(self):
        registre = self._appeler("Change le cours d'entreprise en Examen",
                                 juger_script({}), title='Examen')
        question, chips, _ = rendre_demandes(self._demandes(registre), self.jour)
        self.assertIn('changer', question)
        self.assertNotIn('enlever', question)
        self.assertEqual([c['option'] for c in chips],
                         ['occurrence', 'serie', 'annuler'])
        self.assertIn('Seulement ce', chips[0]['label'])

    def test_le_juge_qui_voit_la_serie_evite_la_question(self):
        """« Renomme mon cours pour de bon »: pas de question, la serie change."""
        message = "Renomme mon cours d'entreprise en Systèmes, pour de bon"
        registre = self._appeler(message, juger_script(
            {message: {"portee": ("serie", 0.95)}}), title='Systèmes')
        self.assertTrue(registre.actions[-1].succes, registre.actions[-1].message)
        self.cours.refresh_from_db()
        self.assertEqual(self.cours.title, 'Systèmes')

    def test_un_juge_incertain_fait_demander(self):
        message = "Change le cours d'entreprise en Examen"
        registre = self._appeler(message, juger_script(
            {message: {"portee": ("incertain", 0.95)}}), title='Examen')
        self.assertFalse(registre.actions[-1].succes)
        self.cours.refresh_from_db()
        self.assertEqual(self.cours.title, "L’entreprise et ses systèmes")

    def test_sans_changement_reel_aucune_question(self):
        """Le meme titre et les memes heures ne changent rien: pas de garde."""
        registre = self._appeler("Mets a jour mon cours", juger_script({}),
                                 title="L’entreprise et ses systèmes")
        self.assertTrue(registre.actions[-1].succes, registre.actions[-1].message)

    def test_les_bornes_de_serie_gardent_leur_propre_garde(self):
        """Une fin de serie avancee reste une garde destructive, pas une portee."""
        registre = self._appeler("Arrête mon cours à la fin octobre",
                                 juger_script({}), end_date='2026-10-31')
        demandes = self._demandes(registre)
        if demandes:
            self.assertEqual(demandes[0]['motif'], 'destructif')

    def test_l_effet_de_la_puce_occurrence_est_accepte(self):
        """Sans cette validation, la puce ne ferait RIEN: _effet_valide exige
        que l'effet vise exactement la cible de la cle."""
        from services.agent_v2 import outils as o
        registre = self._appeler("Change le cours d'entreprise en Examen",
                                 juger_script({}), title='Examen')
        demande = self._demandes(registre)[0]
        effets = {opt['id']: opt.get('effet') for opt in demande['options']}
        self.assertTrue(o._effet_valide(demande, 'occurrence', effets['occurrence']))
        self.assertTrue(o._effet_valide(demande, 'serie', effets['serie']))
        # L'effet de l'occurrence remplace, il ne renomme pas.
        self.assertEqual(effets['occurrence']['outil'], 'replace_block_occurrence')
        self.assertEqual(effets['occurrence']['parametres']['replacement_title'], 'Examen')
        self.assertEqual(effets['serie']['outil'], 'update_block')

    def test_un_effet_altere_est_rejete(self):
        """Une demande alteree ne peut pas faire executer autre chose."""
        from services.agent_v2 import outils as o
        registre = self._appeler("Change le cours d'entreprise en Examen",
                                 juger_script({}), title='Examen')
        demande = self._demandes(registre)[0]
        effets = {opt['id']: opt.get('effet') for opt in demande['options']}
        altere = dict(effets['occurrence'])
        altere['parametres'] = dict(altere['parametres'], date='2099-01-01')
        self.assertFalse(o._effet_valide(demande, 'occurrence', altere))

    def test_les_heures_donnees_suivent_les_deux_options(self):
        """« Examen de 13 h a 16 h a la place du cours »: sans cela, l'examen
        etait place aux heures du COURS et les heures de la personne perdues."""
        from services.agent_v2 import outils as o
        registre = self._appeler("Mercredi examen de 13 h à 16 h à la place du cours",
                                 juger_script({}), title='Examen',
                                 start_time='13:00', end_time='16:00')
        demande = self._demandes(registre)[0]
        effets = {opt['id']: opt.get('effet') for opt in demande['options']}
        for option in ('occurrence', 'serie'):
            with self.subTest(option=option):
                params = effets[option]['parametres']
                self.assertEqual(params.get('start_time'), '13:00')
                self.assertEqual(params.get('end_time'), '16:00')
                self.assertTrue(o._effet_valide(demande, option, effets[option]))

    def test_la_cle_change_avec_les_heures_proposees(self):
        """Une puce qui accepte 13 h a 16 h n'autorise pas un autre creneau."""
        from services.agent_v2 import outils as o
        a = o._cle_changement(self.cours.id, {'title': 'Examen',
                                              'start_time': '13:00', 'end_time': '16:00'})
        b = o._cle_changement(self.cours.id, {'title': 'Examen',
                                              'start_time': '18:00', 'end_time': '21:00'})
        self.assertNotEqual(a, b)

    def test_les_deux_listes_de_priorite_connaissent_le_motif(self):
        """agent.py et rendu.py ont chacune leur liste: un motif absent de
        l'une passe derriere un formulaire."""
        from services.agent_v2 import agent as a
        from services.agent_v2 import rendu as r
        for liste in (a.PRIORITE, r.PRIORITE):
            self.assertIn('portee_changement', liste)
            self.assertLess(liste.index('portee_changement'), liste.index('formulaire'))

    def test_la_cible_changee_couvre_ce_motif(self):
        """Garde-fou K1: la puce ne s'execute pas sur un etat perime."""
        from services.agent_v2 import outils as o
        registre = self._appeler("Change le cours d'entreprise en Examen",
                                 juger_script({}), title='Examen')
        demande = self._demandes(registre)[0]
        self.assertFalse(o._cible_changee(self.user, demande))
        RecurringBlock.objects.filter(id=self.cours.id).update(start_time='09:00')
        self.assertTrue(o._cible_changee(self.user, demande))

    def test_laisse_tomber_ferme_la_question(self):
        from services.agent_v2 import demandes as d
        self.assertIn('portee_changement', d.MOTIFS_LECTURE_LIBRE)

    def test_un_appel_composite_est_refuse(self):
        """Identite ET bornes dans le meme appel: deux gestes, deux questions.

        Prouve le 2026-09-30: une date NON destructive desarmait toute garde
        et la serie etait renommee en silence; une date destructive faisait
        gagner la garde des bornes, dont la question ne parle que de la date,
        et confirmer la date renommait la serie au passage.
        """
        for bornes in ({'end_date': '2026-12-31'}, {'start_date': '2026-10-06'}):
            with self.subTest(bornes=bornes):
                registre = self._appeler("Renomme et borne mon cours",
                                         juger_script({}), title='Examen', **bornes)
                action = registre.actions[-1]
                self.assertFalse(action.succes)
                self.assertIn('deux appels', action.message)
                self.cours.refresh_from_db()
                self.assertEqual(self.cours.title, "L’entreprise et ses systèmes")
                self.assertIsNone(self.cours.end_date)

    def test_deplacer_la_serie_de_jour_demande_aussi_sa_portee(self):
        """day_of_week deplace TOUTE la serie, comme le titre la renomme."""
        autre = (self.jour.weekday() + 1) % 7
        registre = self._appeler("Mets mon cours d'entreprise le jeudi",
                                 juger_script({}), day_of_week=str(autre))
        self.assertFalse(registre.actions[-1].succes)
        self.assertEqual(self._demandes(registre)[0]['motif'], 'portee_changement')
        self.cours.refresh_from_db()
        self.assertEqual(self.cours.day_of_week, self.jour.weekday())

    def test_une_apostrophe_redressee_n_est_pas_un_changement(self):
        """Le modele ecrit l'apostrophe droite, la base porte la courbe."""
        registre = self._appeler("Mets a jour mon cours", juger_script({}),
                                 title="L'entreprise et ses systèmes")
        self.assertTrue(registre.actions[-1].succes, registre.actions[-1].message)

    def test_l_effet_serie_n_accepte_aucun_parametre_clandestin(self):
        from services.agent_v2 import outils as o
        registre = self._appeler("Change le cours d'entreprise en Examen",
                                 juger_script({}), title='Examen')
        demande = self._demandes(registre)[0]
        effets = {opt['id']: opt.get('effet') for opt in demande['options']}
        trafique = dict(effets['serie'])
        trafique['parametres'] = dict(trafique['parametres'], day_of_week='0')
        self.assertFalse(o._effet_valide(demande, 'serie', trafique))
        trafique2 = dict(effets['serie'])
        trafique2['parametres'] = dict(trafique2['parametres'], title='Poubelle')
        self.assertFalse(o._effet_valide(demande, 'serie', trafique2))

    def test_les_champs_secondaires_suivent_l_option_serie(self):
        """Lieu et type tombaient de la puce: le code annonçait un geste
        partiel comme fait."""
        registre = self._appeler("Renomme mon cours et mets-le en salle B-210",
                                 juger_script({}), title='Examen',
                                 location='B-210')
        effets = {o['id']: o.get('effet') for o in self._demandes(registre)[0]['options']}
        self.assertEqual(effets['serie']['parametres'].get('location'), 'B-210')


class ReglagesDeTestTests(TransactionTestCase):
    """Les reglages qui rendraient la suite rouge ou aveugle.

    Mesure en CI le 2026-09-30: sans DEBUG dans l'environnement, DEBUG vaut
    False, SECURE_SSL_REDIRECT s'active et le client de test recoit un 301 sur
    chaque requete. 203 assertions en echec sur un arbre propre.
    """

    def test_la_redirection_https_est_neutralisee_sous_les_tests(self):
        from django.conf import settings
        self.assertTrue(getattr(settings, '_EN_TEST', False))
        self.assertFalse(getattr(settings, 'SECURE_SSL_REDIRECT', False))

    def test_une_requete_du_client_de_test_n_est_pas_redirigee(self):
        """La preuve par l'usage, quel que soit DEBUG."""
        reponse = self.client.get('/api/auth/me/')
        self.assertNotEqual(reponse.status_code, 301)

    def test_le_juge_est_coupe_du_reseau_sous_les_tests(self):
        from django.conf import settings
        self.assertEqual(settings.JEV_API_KEY, '')
        self.assertEqual(settings.TYPESAFE_API_KEY, '')
        self.assertEqual(settings.JUGEMENT_REPLI_LLM, '0')
