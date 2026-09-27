"""
Parallelisme des outils de lecture (agent v2).

Verrouille le chantier perf du 2026-09-27: les lectures batchées par le
modèle dans une même étape s'exécutent en parallèle (pydantic-ai dispatche
les appels via asyncio.create_task sauf outil marqué sequential); les
MUTATIONS restent sérialisées par le verrou du tour (garde
tester-puis-poser, cache d'idempotence).

Aucun appel réseau ici: faux outils synchrones, threads réels, SQLite de test.
"""
import asyncio
import threading
import time

from django.contrib.auth.models import User
from django.test import TestCase

from services.agent.tools import ALL_TOOLS
from services.agent.tools.base import ToolResult
from services.agent_v2.outils import _fabriquer, outils_pour
from services.agent_v2.prompts import REGLES_AGIR
from services.agent_v2.registre import (OUTILS_DE_MUTATION, OUTILS_PARALLELES,
                                        Registre)


class _Piste:
    """Mesure le chevauchement réel des exécutions, tous outils confondus."""

    def __init__(self):
        self._verrou = threading.Lock()
        self.en_cours = 0
        self.max_en_cours = 0

    def entree(self):
        with self._verrou:
            self.en_cours += 1
            self.max_en_cours = max(self.max_en_cours, self.en_cours)

    def sortie(self):
        with self._verrou:
            self.en_cours -= 1


class _FauxOutil:
    """Outil synchrone qui dort un peu pour rendre le chevauchement visible.

    Les noms « lecture_test_* » ne correspondent à aucune branche de garde:
    l'appel traverse jusqu'à execute sans toucher la base.
    """

    def __init__(self, nom, piste, delai=0.25):
        self.name = nom
        self._piste = piste
        self._delai = delai
        self.appels = 0

    def execute(self, user, **kwargs):
        self._piste.entree()
        try:
            time.sleep(self._delai)
        finally:
            self._piste.sortie()
            self.appels += 1
        return ToolResult(success=True, message=f"{self.name} ok", data={})


def _lancer(outils_fabriques):
    async def _tout():
        await asyncio.gather(*[fabrique() for fabrique in outils_fabriques])

    asyncio.run(_tout())


class LecturesParallelesTests(TestCase):
    def test_trois_lectures_se_chevauchent(self):
        user = User.objects.create_user(username="par1")
        registre = Registre()
        piste = _Piste()
        fabriques = [
            _fabriquer(_FauxOutil(f"lecture_test_{c}", piste), user, registre,
                       "montre mon planning", f"tache-par-{c}")
            for c in "abc"
        ]
        _lancer(fabriques)
        # Les trois lectures ont tourné en même temps, pas l'une après l'autre.
        self.assertEqual(piste.max_en_cours, 3)
        self.assertEqual(len(registre.actions), 3)
        self.assertEqual(
            {a.outil for a in registre.actions},
            {"lecture_test_a", "lecture_test_b", "lecture_test_c"},
        )

    def test_ids_de_registre_uniques_sous_concurrence(self):
        registre = Registre()
        erreurs = []

        def _ajouter(i):
            try:
                registre.ajouter("lecture_test_a", {"i": i},
                                 ToolResult(success=True, message="ok", data={}))
            except Exception as e:  # noqa: BLE001
                erreurs.append(e)

        threads = [threading.Thread(target=_ajouter, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(erreurs, [])
        ids = [a.id for a in registre.actions]
        self.assertEqual(len(ids), 20)
        self.assertEqual(len(set(ids)), 20)
        # L'index couvre tout le monde: par_id retrouve chaque action.
        for a in registre.actions:
            self.assertIs(registre.par_id(a.id), a)


class MutationsSerialiseesTests(TestCase):
    def test_deux_mutations_ne_se_chevauchent_jamais(self):
        # create_task et update_task sont des mutations sans branche de garde
        # qui touche la base: l'appel traverse jusqu'à execute.
        user = User.objects.create_user(username="par2")
        registre = Registre()
        piste = _Piste()
        fabriques = [
            _fabriquer(_FauxOutil("create_task", piste, delai=0.3), user,
                       registre, "cree une tache", "tache-par-m1"),
            _fabriquer(_FauxOutil("update_task", piste, delai=0.3), user,
                       registre, "modifie une tache", "tache-par-m2"),
        ]
        _lancer(fabriques)
        # Le verrou du tour a sérialisé: jamais deux mutations à la fois.
        self.assertEqual(piste.max_en_cours, 1)
        self.assertEqual(len(registre.actions), 2)

    def test_flags_sequential_allowlist(self):
        user = User.objects.create_user(username="par3")
        registre = Registre()
        for outil in outils_pour(user, registre):
            with self.subTest(outil=outil.name):
                self.assertEqual(
                    outil.tool_def.sequential,
                    outil.name not in OUTILS_PARALLELES,
                    "seules les lectures auditees partent en parallele; "
                    "tout nouvel outil est sequentiel par defaut",
                )

    def test_chaque_outil_du_modele_est_classe(self):
        noms = {t.name for t in ALL_TOOLS}
        self.assertEqual(
            noms - OUTILS_PARALLELES, noms & OUTILS_DE_MUTATION,
            "chaque outil du modele est soit une mutation (sequentiel), "
            "soit une lecture auditee (parallele): aucun ne doit rester "
            "non classe",
        )


class PromptBatchingTests(TestCase):
    def test_instruction_de_batching_presente(self):
        self.assertIn("LECTURES GROUPEES", REGLES_AGIR)
        self.assertIn("en un seul bloc d'appels", REGLES_AGIR)
        self.assertIn("Ne groupe jamais une ecriture", REGLES_AGIR)


class BatchReelTests(TestCase):
    """Un vrai lot multi-appels, dispatché par pydantic-ai lui-même.

    test_trois_lectures_se_chevauchent prouvait que les wrappers se
    chevauchent via asyncio.gather; ici c'est le MODELE (faux) qui émet
    trois appels dans UNE SEULE réponse, et c'est le dispatch réel de
    pydantic-ai qui décide parallèle ou séquentiel selon les drapeaux
    sequential. Sans réseau: FunctionModel ne fait aucun appel HTTP.
    """

    def test_batch_modele_trois_lectures_en_parallele(self):
        from pydantic_ai import Agent
        from pydantic_ai.messages import (ModelResponse, TextPart,
                                          ToolCallPart)
        from pydantic_ai.models.function import AgentInfo, FunctionModel
        from pydantic_ai.tools import Tool

        user = User.objects.create_user(username="par-batch")
        registre = Registre()
        piste = _Piste()
        fabriques = {
            f"lecture_test_{c}": _fabriquer(
                _FauxOutil(f"lecture_test_{c}", piste), user,
                registre, "montre mon planning", f"tache-batch-{c}")
            for c in "abc"
        }

        def _outil(nom):
            async def _appel() -> str:
                return await fabriques[nom]()

            _appel.__name__ = nom
            return Tool(_appel, name=nom, description="lecture de test",
                        sequential=False)

        def _fonction(messages, info: AgentInfo) -> ModelResponse:
            if any(isinstance(m, ModelResponse) for m in messages):
                return ModelResponse(parts=[TextPart(content="voilà")])
            return ModelResponse(parts=[
                ToolCallPart(tool_name=f"lecture_test_{c}", args={},
                             tool_call_id=f"appel-{c}")
                for c in "abc"
            ])

        agent = Agent(FunctionModel(_fonction),
                      tools=[_outil(f"lecture_test_{c}") for c in "abc"])
        resultat = asyncio.run(agent.run("montre mon planning"))
        # Le dispatch a chevauché les trois lectures, pas l'une après l'autre.
        self.assertEqual(piste.max_en_cours, 3)
        self.assertEqual(len(registre.actions), 3)
        self.assertIn("voilà", resultat.output)

    def test_batch_mixte_lecture_mutation_reste_sequentiel(self):
        """Contre-épreuve: un seul appel marqué sequential dans le lot force
        TOUT le lot en séquentiel (sémantique pydantic-ai). C'est ce qui
        protège les mutations quand le modèle les batche avec des lectures.
        """
        from pydantic_ai import Agent
        from pydantic_ai.messages import (ModelResponse, TextPart,
                                          ToolCallPart)
        from pydantic_ai.models.function import AgentInfo, FunctionModel
        from pydantic_ai.tools import Tool

        user = User.objects.create_user(username="par-batch-mixte")
        registre = Registre()
        piste = _Piste()
        fabriques = {
            "lecture_test_a": _fabriquer(
                _FauxOutil("lecture_test_a", piste), user, registre,
                "montre mon planning", "tache-mixte-a"),
            # Nom de vraie mutation: _fabriquer lui fait tenir le verrou
            # du tour, comme en production.
            "create_task": _fabriquer(
                _FauxOutil("create_task", piste), user, registre,
                "cree une tache", "tache-mixte-m"),
        }

        def _outil(nom, sequential):
            async def _appel() -> str:
                return await fabriques[nom]()

            _appel.__name__ = nom
            return Tool(_appel, name=nom, description="outil de test",
                        sequential=sequential)

        def _fonction(messages, info: AgentInfo) -> ModelResponse:
            if any(isinstance(m, ModelResponse) for m in messages):
                return ModelResponse(parts=[TextPart(content="voilà")])
            return ModelResponse(parts=[
                ToolCallPart(tool_name="lecture_test_a", args={},
                             tool_call_id="appel-a"),
                ToolCallPart(tool_name="create_task", args={},
                             tool_call_id="appel-m"),
            ])

        agent = Agent(FunctionModel(_fonction), tools=[
            _outil("lecture_test_a", sequential=False),
            _outil("create_task", sequential=True),
        ])
        resultat = asyncio.run(agent.run("montre mon planning"))
        # Le lot mixte est resté séquentiel: jamais deux outils à la fois.
        self.assertEqual(piste.max_en_cours, 1)
        self.assertEqual(len(registre.actions), 2)
        self.assertIn("voilà", resultat.output)
