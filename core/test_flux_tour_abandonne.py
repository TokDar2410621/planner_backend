"""Un tour de chat se termine cote serveur, meme si le client part.

Lot 1a. Avant le drain detache, la vue pompait le generateur de l'agent
depuis la boucle asynchrone: un client qui coupait arretait le tour la ou il
en etait, outils deja executes mais ni message de l'assistant ni ligne de
tour sauves (12 messages sur 138 sans reponse en production).

TransactionTestCase partout: le drain tourne dans un thread du pool et ecrit
en base, il doit voir les lignes commitees par le test et inversement.
"""
import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from django.contrib.auth.models import User
from django.db import close_old_connections
from django.test import TransactionTestCase
from django.utils import timezone
from rest_framework.test import APIClient

from core.lecture_flux import corps_du_flux
from core.models import ConversationMessage
from core.views import STATUT_ATTENTE, TRAME_ERREUR, lancer_flux
from services.agent_v2 import PlannerAgentV2


def _trame(evenement):
    return "data: " + json.dumps(evenement, ensure_ascii=False) + "\n\n"


async def _tout_lire(flux):
    return [morceau async for morceau in flux]


def _attendre(condition, delai):
    fin = time.monotonic() + delai
    while time.monotonic() < fin:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


class FluxTourAbandonneTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user("abandon", password="pw-123456")

    def _assistant(self):
        return ConversationMessage.objects.filter(
            user=self.user, role="assistant").count()

    # ---------------------------------------------------------- deconnexion

    def test_deconnexion_le_tour_se_termine_et_sauve(self):
        user = self.user
        liberer = threading.Event()
        outil_vu = threading.Event()

        def generateur():
            yield {"type": "status", "text": "Réflexion..."}
            yield {"type": "tool", "name": "create_block", "ok": True,
                   "id": "a1", "message": ""}
            outil_vu.set()
            liberer.wait(10)
            ConversationMessage.objects.create(
                user=user, role="assistant", content="ok")
            yield {"type": "done", "response": "ok"}

        with self.assertLogs("core.views", "INFO") as journal:
            async def scenario():
                flux, futur = lancer_flux(generateur(), user.id)
                premier = await flux.__anext__()
                # Le drain a deja recu l'outil quand le client part.
                self.assertTrue(outil_vu.wait(5))
                await flux.aclose()
                return premier, futur

            premier, futur = asyncio.run(scenario())
            self.assertEqual(premier, _trame(
                {"type": "status", "text": "Réflexion..."}))
            # Le client est parti AVANT la fin: rien n'est encore sauve.
            self.assertEqual(self._assistant(), 0)
            liberer.set()
            futur.result(timeout=5)

        self.assertEqual(self._assistant(), 1)
        avertissements = [l for l in journal.output if l.startswith("WARNING:")]
        self.assertEqual(len(avertissements), 1)
        self.assertIn("tour abandonne", avertissements[0])
        self.assertIn("create_block", avertissements[0])
        self.assertIn("outils=1 ok=1", avertissements[0])
        fins = [l for l in journal.output if "tour abandonne termine" in l]
        self.assertEqual(len(fins), 1)
        self.assertIn("evenements=3 done=True", fins[0])

    def test_annulation_pendant_l_attente(self):
        user = self.user
        # Echauffement: le thread du pool existe deja avant la mesure.
        asyncio.run(_tout_lire(lancer_flux(
            iter([{"type": "done", "response": "x"}]), user.id)[0]))
        base = threading.active_count()

        liberer = threading.Event()
        bloque = threading.Event()

        def generateur():
            yield {"type": "status", "text": "Réflexion..."}
            bloque.set()
            liberer.wait(10)
            ConversationMessage.objects.create(
                user=user, role="assistant", content="ok")
            yield {"type": "done", "response": "ok"}

        with self.assertLogs("core.views", "WARNING") as journal:
            async def scenario():
                flux, futur = lancer_flux(generateur(), user.id)
                await flux.__anext__()
                self.assertTrue(bloque.wait(5))
                tache = asyncio.ensure_future(flux.__anext__())
                await asyncio.sleep(0.2)  # la lecture attend dans le get()
                self.assertFalse(tache.done())
                tache.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await tache
                return futur

            futur = asyncio.run(scenario())
            liberer.set()
            futur.result(timeout=5)

        self.assertEqual(self._assistant(), 1)
        self.assertTrue(any("tour abandonne" in l for l in journal.output))
        self.assertTrue(_attendre(lambda: threading.active_count() <= base, 2),
                        f"threads: {threading.active_count()} > {base}")

    def test_aucune_lecture_le_tour_aboutit_quand_meme(self):
        user = self.user
        sauve = threading.Event()

        def generateur():
            yield {"type": "status", "text": "Réflexion..."}
            ConversationMessage.objects.create(
                user=user, role="assistant", content="ok")
            sauve.set()
            yield {"type": "done", "response": "ok"}

        flux, futur = lancer_flux(generateur(), user.id)
        futur.result(timeout=5)
        self.assertTrue(sauve.is_set())
        self.assertEqual(self._assistant(), 1)
        # Le lecteur n'a jamais demarre: le fermer ne leve rien.
        asyncio.run(flux.aclose())

    def test_deconnexion_apres_la_fin_du_drain_une_seule_ligne_de_fin(self):
        user = self.user
        evenements = [{"type": "status", "text": "a"},
                      {"type": "tool", "name": "delete_block", "ok": False,
                       "id": "a1", "message": ""},
                      {"type": "done", "response": "ok"}]
        with self.assertLogs("core.views", "INFO") as journal:
            async def scenario():
                flux, futur = lancer_flux(iter(evenements), user.id)
                premier = await flux.__anext__()
                await asyncio.wrap_future(futur)  # drain fini, client encore la
                await flux.aclose()
                return premier

            asyncio.run(scenario())
        self.assertTrue(any("tour abandonne user=" in l and "ok=0" in l
                            and "delete_block" in l for l in journal.output))
        fins = [l for l in journal.output if "tour abandonne termine" in l]
        self.assertEqual(len(fins), 1)
        self.assertIn("done=True", fins[0])

    # ------------------------------------------------------ deux tours, pool

    def test_deux_tours_consecutifs_meme_processus(self):
        """Regression du blocage au deuxieme tour (agent.py, _POOL_AGIR)."""
        from services.agent_v2.agent import _POOL_AGIR

        user = self.user

        def generateur(n):
            ConversationMessage.objects.create(
                user=user, role="user", content=f"tour {n}")
            yield {"type": "status", "text": "Réflexion..."}

            def agir():
                # Meme forme qu'AGIR: ORM hors coroutine, puis une boucle
                # asyncio dans le thread du pool (ce que fait run_sync).
                close_old_connections()
                try:
                    compte = ConversationMessage.objects.filter(user=user).count()

                    async def modele():
                        await asyncio.sleep(0.01)
                        return f"pensée {n}"

                    return asyncio.run(modele()), compte
                finally:
                    close_old_connections()

            pensee, compte = _POOL_AGIR.submit(agir).result(timeout=5)
            yield {"type": "thinking", "text": pensee}
            ConversationMessage.objects.create(
                user=user, role="assistant", content=f"réponse {n} ({compte})")
            yield {"type": "done", "response": f"réponse {n}"}

        self.user.profile.ai_consent_at = timezone.now()
        self.user.profile.save(update_fields=["ai_consent_at"])
        client = APIClient()
        client.force_authenticate(self.user)

        depart = time.monotonic()
        corps = []
        for n in (1, 2):
            with patch.object(PlannerAgentV2, "process_message_stream",
                              side_effect=lambda u, m, a, n=n: generateur(n)):
                reponse = client.post("/api/chat/stream/", {"message": f"tour {n}"})
                corps.append(corps_du_flux(reponse))
        self.assertLess(time.monotonic() - depart, 10)

        for n, texte in zip((1, 2), corps):
            self.assertEqual(texte, "".join([
                _trame({"type": "status", "text": "Réflexion..."}),
                _trame({"type": "thinking", "text": f"pensée {n}"}),
                _trame({"type": "done", "response": f"réponse {n}"}),
            ]))
        self.assertEqual(self._assistant(), 2)

    def test_pool_sature_emet_un_statut(self):
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flux-test")
        occupe = threading.Event()
        liberer = threading.Event()

        def travail_long():
            occupe.set()
            liberer.wait(10)

        pool.submit(travail_long)
        self.assertTrue(occupe.wait(5))
        evenements = [{"type": "status", "text": "Réflexion..."},
                      {"type": "done", "response": "ok"}]
        try:
            with patch("core.views._POOL_FLUX", pool):
                async def scenario():
                    flux, futur = lancer_flux(iter(evenements), self.user.id)
                    depart = time.monotonic()
                    premier = await flux.__anext__()
                    delai = time.monotonic() - depart
                    liberer.set()
                    reste = [m async for m in flux]
                    return premier, delai, reste, futur

                premier, delai, reste, futur = asyncio.run(scenario())
            futur.result(timeout=5)
        finally:
            liberer.set()
            pool.shutdown(wait=True)

        self.assertEqual(premier, _trame({
            "type": "status",
            "text": "Un instant, je termine une autre demande...",
        }))
        self.assertEqual(STATUT_ATTENTE["text"],
                         "Un instant, je termine une autre demande...")
        self.assertLess(delai, 2)
        self.assertEqual(reste, [_trame(e) for e in evenements])

    def test_generateur_lent_mais_demarre_pas_de_statut(self):
        """Le statut d'attente dit « pool plein », pas « agent lent »."""
        def generateur():
            time.sleep(1.3)
            yield {"type": "done", "response": "ok"}

        async def scenario():
            flux, futur = lancer_flux(generateur(), self.user.id)
            return await _tout_lire(flux)

        self.assertEqual(asyncio.run(scenario()),
                         [_trame({"type": "done", "response": "ok"})])

    # --------------------------------------------------- flux sans incident

    def test_flux_complet_sans_abandon(self):
        evenements = [
            {"type": "status", "text": "Je consulte ton planning…"},
            {"type": "thinking", "text": "L'utilisateur veut sa journée."},
            {"type": "tool", "id": "a1", "name": "get_today_schedule",
             "ok": True, "message": "Journée lue"},
            {"type": "delta", "text": "**Jeudi**\n9 h à 10 h · Réunion"},
            {"type": "done", "response": "**Jeudi**\n9 h à 10 h · Réunion",
             "quick_replies": [], "question_posee": False},
        ]
        with self.assertNoLogs("core.views", "WARNING"):
            async def scenario():
                flux, futur = lancer_flux(iter(evenements), self.user.id)
                return await _tout_lire(flux), futur

            trames, futur = asyncio.run(scenario())
            futur.result(timeout=5)
        self.assertEqual(
            trames,
            ["data: " + json.dumps(e, ensure_ascii=False) + "\n\n"
             for e in evenements])

    def test_panne_d_agent_en_plein_flux(self):
        def generateur():
            yield {"type": "status", "text": "Réflexion..."}
            raise RuntimeError("boom")

        with self.assertLogs("core.views", "ERROR") as journal:
            async def scenario():
                flux, futur = lancer_flux(generateur(), self.user.id)
                return await _tout_lire(flux), futur

            trames, futur = asyncio.run(scenario())
            futur.result(timeout=5)
        self.assertTrue(futur.done())
        self.assertEqual(trames, [
            _trame({"type": "status", "text": "Réflexion..."}),
            TRAME_ERREUR,
        ])
        self.assertIn("Erreur interne lors du traitement du message.", TRAME_ERREUR)
        self.assertNotIn("boom", "".join(trames))
        self.assertTrue(any("PlannerAgent stream error" in l for l in journal.output))
        self.assertFalse(any("tour abandonne" in l for l in journal.output))

    def test_trame_d_erreur_identique_a_l_ancienne(self):
        self.assertEqual(TRAME_ERREUR, "data: " + json.dumps({
            "type": "error",
            "error": "Erreur interne lors du traitement du message.",
        }) + "\n\n")

    def test_evenement_non_serialisable_ne_coupe_pas_le_tour(self):
        user = self.user

        def generateur():
            yield {"type": "status", "text": "x", "quand": timezone.now()}
            ConversationMessage.objects.create(
                user=user, role="assistant", content="ok")
            yield {"type": "done", "response": "ok"}

        async def scenario():
            flux, futur = lancer_flux(generateur(), user.id)
            return await _tout_lire(flux), futur

        trames, futur = asyncio.run(scenario())
        futur.result(timeout=5)
        self.assertEqual(len(trames), 2)
        self.assertEqual(trames[1], _trame({"type": "done", "response": "ok"}))
        self.assertEqual(self._assistant(), 1)

    # ---------------------------------------------------------------- vue

    def test_la_vue_poursuit_le_tour_apres_deconnexion(self):
        user = self.user
        user.profile.ai_consent_at = timezone.now()
        user.profile.save(update_fields=["ai_consent_at"])
        client = APIClient()
        client.force_authenticate(user)
        liberer = threading.Event()
        sauve = threading.Event()

        def generateur():
            yield {"type": "status", "text": "Réflexion..."}
            liberer.wait(10)
            ConversationMessage.objects.create(
                user=user, role="assistant", content="ok")
            sauve.set()
            yield {"type": "done", "response": "ok"}

        with patch.object(PlannerAgentV2, "process_message_stream",
                          side_effect=lambda u, m, a: generateur()):
            reponse = client.post("/api/chat/stream/", {"message": "salut"})
            self.assertEqual(reponse["Content-Type"], "text/event-stream")
            self.assertEqual(reponse["Cache-Control"], "no-cache")
            self.assertEqual(reponse["X-Accel-Buffering"], "no")

            async def partir():
                contenu = reponse.streaming_content
                premier = await contenu.__anext__()
                await contenu.aclose()
                return premier

            with self.assertLogs("core.views", "WARNING"):
                premier = asyncio.run(partir())
            liberer.set()
            self.assertTrue(sauve.wait(5))

        self.assertIn(b'"type": "status"', premier)
        self.assertEqual(self._assistant(), 1)
