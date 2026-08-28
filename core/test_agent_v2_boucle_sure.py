"""
Les deux garde-fous que la recherche d'aout 2026 designe comme critiques et
qui manquaient a la boucle.

1. TERMINAISON. La taxonomie MAST, construite sur plus de 1600 traces reelles
   a travers sept frameworks, place la terminaison en tete des defaillances:
   le mode « inconscient des conditions de terminaison » pese 12,4 % a lui
   seul, et les boucles d'actions identiques jusqu'a epuisement du budget
   absorbent plus d'un quart des echecs sur certains bancs. Notre boucle avait
   un budget d'etapes, donc elle finissait toujours par s'arreter, mais elle
   pouvait bruler dix tours a rappeler le meme outil avec les memes arguments.

2. IDEMPOTENCE. Observe en production le 2026-08-27: pendant l'incident ASGI,
   un tour bloque a fait reessayer le client TROIS fois, et les logs montrent
   create_block appele trois fois avec des arguments identiques. Ce jour-la
   l'outil echouait, donc rien n'a ete duplique. Le meme scenario avec un
   outil qui REUSSIT et met trop de temps a repondre creerait trois blocs.

La regle contre-intuitive de la litterature: la cle se derive de l'identite
METIER de l'action, jamais d'un UUID tire a chaque tentative. Un UUID neuf par
essai fait voir au serveur une requete neuve a chaque fois, et le motif
s'effondre en silence.
"""
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase, TransactionTestCase

from services.agent.tools.base import ToolResult
from services.agent_v2.redaction import ReponseDire
from services.agent_v2.registre import Registre


class RepetitionTests(TestCase):
    def setUp(self):
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def _registre_avec(self, appels):
        registre = Registre()
        for outil, params in appels:
            registre.ajouter(outil, params, ToolResult(success=True, message="ok"))
        return registre

    def test_trois_appels_identiques_declenchent_la_detection(self):
        from services.agent_v2.registre import boucle_detectee
        r = self._registre_avec([
            ("list_blocks", {"day_of_week": 1}),
            ("list_blocks", {"day_of_week": 1}),
            ("list_blocks", {"day_of_week": 1}),
        ])
        self.assertTrue(boucle_detectee(r))

    def test_deux_appels_identiques_ne_declenchent_pas(self):
        """Relire apres une ecriture est legitime; deux fois n'est pas une
        boucle. Un detecteur trop sensible couperait des tours valides."""
        from services.agent_v2.registre import boucle_detectee
        r = self._registre_avec([
            ("list_blocks", {"day_of_week": 1}),
            ("create_block", {"title": "Maths"}),
            ("list_blocks", {"day_of_week": 1}),
        ])
        self.assertFalse(boucle_detectee(r))

    def test_les_memes_arguments_dans_un_ordre_different_comptent_pareil(self):
        """Un dict n'a pas d'ordre stable: sans normalisation, le detecteur
        raterait la repetition la plus banale."""
        from services.agent_v2.registre import boucle_detectee
        r = self._registre_avec([
            ("get_usage", {"a": 1, "b": 2}),
            ("get_usage", {"b": 2, "a": 1}),
            ("get_usage", {"a": 1, "b": 2}),
        ])
        self.assertTrue(boucle_detectee(r))

    def test_trois_appels_au_meme_outil_avec_des_arguments_DIFFERENTS_passent(self):
        """Contre-epreuve indispensable: creer trois blocs d'affilee est le
        comportement NORMAL d'un import d'horaire."""
        from services.agent_v2.registre import boucle_detectee
        r = self._registre_avec([
            ("create_block", {"title": "Maths", "days": [0]}),
            ("create_block", {"title": "Physique", "days": [1]}),
            ("create_block", {"title": "Chimie", "days": [2]}),
        ])
        self.assertFalse(boucle_detectee(r))

    def test_une_boucle_detectee_est_dite_a_l_utilisateur(self):
        """Elle doit apparaitre dans le compte rendu, pas mourir dans un log:
        un tour tronque sans explication ressemble a une panne."""
        from services.agent_v2.redaction import bloc_factuel
        r = self._registre_avec([
            ("list_blocks", {}), ("list_blocks", {}), ("list_blocks", {}),
        ])
        r.boucle_interrompue = True
        rendu = bloc_factuel(r).lower()
        # On verifie le SENS, pas le vocabulaire d'implementation: l'utilisateur
        # doit comprendre que le tour a ete coupe et pourquoi, sans lire le mot
        # « boucle » qui ne veut rien dire pour lui.
        self.assertIn("interrompu", rendu)
        self.assertIn("sans progresser", rendu)


class IdempotenceTests(TransactionTestCase):
    """TransactionTestCase: l'adaptateur passe par sync_to_async, donc l'ORM
    tourne dans un thread qui ne voit pas la transaction non validee d'un
    TestCase."""

    def setUp(self):
        self.user = User.objects.create_user(username='idem', password='x')

    def _appeler_deux_fois(self, nom, **kwargs):
        import asyncio

        from services.agent_v2.outils import outils_pour
        registre = Registre()
        outils = {t.name: t for t in outils_pour(self.user, registre, "", tache="t-42")}
        fonction = outils[nom].function_schema.function
        asyncio.run(fonction(**kwargs))
        asyncio.run(fonction(**kwargs))
        return registre

    def test_le_second_appel_identique_n_atteint_JAMAIS_l_outil(self):
        """Compte les executions REELLES, pas les lignes en base.

        Une premiere version de ce test verifiait qu'un seul bloc existait
        apres deux appels. Il passait meme avec une cle d'idempotence tiree au
        hasard a chaque tentative, c'est-a-dire avec la couche entierement
        neutralisee: create_block a sa PROPRE deduplication, et c'est elle qui
        rattrapait le doublon. Le test etait donc vide de sens. On observe
        maintenant ce qu'on pretend proteger."""
        import asyncio

        from services.agent.tools import TOOL_MAP
        from services.agent_v2.outils import outils_pour

        executions = []
        vrai = TOOL_MAP['create_block'].execute

        def compter(*a, **kw):
            executions.append(kw)
            return vrai(*a, **kw)

        registre = Registre()
        with patch.object(TOOL_MAP['create_block'], 'execute', compter):
            outils = {t.name: t for t in outils_pour(
                self.user, registre, "", tache="t-idem")}
            f = outils['create_block'].function_schema.function
            args = dict(title='Maths', block_type='course', days=[0],
                        start_time='09:00', end_time='12:00')
            asyncio.run(f(**args))
            asyncio.run(f(**args))

        self.assertEqual(len(executions), 1,
                         "le second appel identique a atteint l'outil")

    def test_le_second_appel_est_quand_meme_consigne(self):
        """Le registre doit garder trace des DEUX appels: le bloc factuel
        raconte ce qui s'est passe, pas ce qu'on aurait voulu."""
        registre = self._appeler_deux_fois(
            'create_block', title='Physique', block_type='course',
            days=[1], start_time='09:00', end_time='11:00')
        self.assertEqual(len(registre.actions), 2)

    def test_deux_appels_DIFFERENTS_ecrivent_bien_deux_fois(self):
        """Contre-epreuve: sans elle, un cache trop large bloquerait la
        creation de blocs legitimement voisins."""
        import asyncio

        from core.models import RecurringBlock
        from services.agent_v2.outils import outils_pour
        registre = Registre()
        outils = {t.name: t for t in outils_pour(self.user, registre, "", tache="t-43")}
        f = outils['create_block'].function_schema.function
        base = dict(block_type='course', start_time='09:00', end_time='12:00')
        asyncio.run(f(title='Maths', days=[0], **base))
        asyncio.run(f(title='Chimie', days=[1], **base))
        self.assertEqual(RecurringBlock.objects.filter(user=self.user).count(), 2)

    def test_une_lecture_n_est_jamais_mise_en_cache(self):
        """Une lecture doit refleter l'etat COURANT. La mettre en cache
        rendrait l'agent aveugle a ses propres ecritures dans le meme tour."""
        import asyncio

        from services.agent_v2.outils import outils_pour
        registre = Registre()
        outils = {t.name: t for t in outils_pour(self.user, registre, "", tache="t-44")}
        lire = outils['list_blocks'].function_schema.function
        creer = outils['create_block'].function_schema.function
        asyncio.run(lire())
        asyncio.run(creer(title='Sport', block_type='sport', days=[2],
                          start_time='18:00', end_time='19:00'))
        seconde = asyncio.run(lire())
        self.assertIn('Sport', seconde)
