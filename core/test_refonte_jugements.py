"""Mesure et borne des jugements, HTTP simulé et vrai runtime SDK local."""
import asyncio
from contextlib import contextmanager
import time
from unittest.mock import MagicMock, patch

from django.test import TransactionTestCase, override_settings
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from services.agent_v2 import jugement
from services.agent_v2.trace_tour import TraceTour, trace_courante


@contextmanager
def trace_pour(delai=30, **kwargs):
    from django.contrib.auth import get_user_model
    from services.agent_v2.budget_tour import ouvrir_trace, fermer_trace
    user, _ = get_user_model().objects.get_or_create(username='jugement-reservation')
    trace = ouvrir_trace(user, delai=delai)
    for cle, valeur in kwargs.items():
        setattr(trace, cle, valeur)
    token = trace_courante.set(trace)
    try:
        yield trace
    finally:
        fermer_trace(trace)
        trace_courante.reset(token)


def reponse_jev(usage=None):
    reponse = MagicMock()
    reponse.json.return_value = {"answers": {"q": {"noul": 0.95}}}
    if usage is not None:
        reponse.json.return_value["usage"] = usage
    return reponse


def modele_local(invalide=False):
    def repondre(messages, info):
        arguments = {} if invalide else {"reponses": {"q": {"valeur": "oui", "confiance": 0.95}}}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, arguments)],
                             usage=RequestUsage(input_tokens=30, output_tokens=6))
    return FunctionModel(repondre)


class JugementsBornesTests(TransactionTestCase):
    def setUp(self):
        jugement.vider_cache()
        self.questions = {"q": jugement.q_suppression()}

    @override_settings(JEV_API_KEY="", TYPESAFE_API_KEY="", JUGEMENT_REPLI_LLM="1")
    def test_repli_structure_transmet_le_reglage_sans_raisonnement(self):
        reglages = []
        def repondre(messages, info):
            reglages.append(info.model_settings)
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                {"reponses": {"q": {"valeur": "oui", "confiance": 0.95}}})],
                usage=RequestUsage(input_tokens=30, output_tokens=6))
        with trace_pour(), patch('services.agent_v2.modeles.modele_agir',
                                return_value=FunctionModel(repondre)):
            resultat = jugement.juger('supprime', self.questions)
        self.assertEqual(resultat['q']['statut'], 'decision')
        self.assertEqual(reglages[0].get('extra_body'), {'thinking': {'type': 'disabled'}})

    @override_settings(JEV_API_KEY="cle-factice", JUGEMENT_REPLI_LLM="0")
    def test_usage_jev_compte_entree_sortie_sans_doubler_raisonnement(self):
        with trace_pour() as trace, patch("requests.post", return_value=reponse_jev(
                {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "reasoning_tokens": 8})):
            jugement.juger("supprime", self.questions)
        self.assertEqual(trace.total_jetons, 120)
        self.assertTrue(trace.usages[0]["usage_connu"])

    @override_settings(JEV_API_KEY="cle-factice", JUGEMENT_REPLI_LLM="0")
    def test_usage_absent_reste_inconnu(self):
        with trace_pour() as trace, patch("requests.post", return_value=reponse_jev()):
            jugement.juger("supprime", self.questions)
        self.assertEqual(len(trace.usages), 1)
        self.assertFalse(trace.usages[0]["usage_connu"])
        self.assertIsNone(trace.usages[0]["entree"])

    @override_settings(JEV_API_KEY="cle-factice", JUGEMENT_REPLI_LLM="0")
    def test_timeout_jev_signale_un_appel_usage_inconnu(self):
        with trace_pour() as trace, patch("requests.post", side_effect=TimeoutError):
            reponse = jugement.juger("supprime", self.questions)
        self.assertEqual(reponse["q"]["statut"], "indisponible")
        self.assertEqual(len(trace.usages), 1)
        self.assertFalse(trace.usages[0]["usage_connu"])
        self.assertFalse(trace.usages[0]["succes"])

    @override_settings(JEV_API_KEY="cle-factice", JEV_TIMEOUT=30, JUGEMENT_REPLI_LLM="0")
    def test_timeout_http_borne_par_restant(self):
        with trace_pour(0.25), patch("requests.post", return_value=reponse_jev()) as post:
            jugement.juger("supprime", self.questions)
        self.assertGreater(post.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(post.call_args.kwargs["timeout"], 0.25)

    @override_settings(JEV_API_KEY="cle-factice", JUGEMENT_REPLI_LLM="1")
    def test_aucun_fournisseur_apres_echeance(self):
        with trace_pour(-1) as trace, patch("requests.post") as post, patch.object(jugement, "_juger_llm") as llm:
            reponse = jugement.juger("supprime", self.questions)
        post.assert_not_called()
        llm.assert_not_called()
        self.assertEqual(reponse["q"]["statut"], "indisponible")
        self.assertEqual(trace.total_jetons, 0)

    @override_settings(JEV_API_KEY="cle-factice", JUGEMENT_REPLI_LLM="1")
    def test_pas_de_repli_quand_jev_epuise_echeance(self):
        with trace_pour() as trace:
            def echeance_epuisee(*args, **kwargs):
                trace.echeance = time.monotonic() - 1
                raise TimeoutError()
            with patch("requests.post", side_effect=echeance_epuisee), patch.object(jugement, "_juger_llm") as llm:
                jugement.juger("supprime", self.questions)
        llm.assert_not_called()

    @override_settings(JEV_API_KEY="cle-factice", JUGEMENT_REPLI_LLM="0")
    def test_septieme_jugement_bloque(self):
        with trace_pour() as trace, patch("requests.post", return_value=reponse_jev()) as post:
            for numero in range(7):
                reponse = jugement.juger(f"message {numero}", self.questions)
        self.assertEqual(post.call_count, 6)
        self.assertEqual(trace.jugements, 6)
        self.assertEqual(reponse["q"]["statut"], "indisponible")

    @override_settings(JEV_API_KEY="cle-factice", JUGEMENT_REPLI_LLM="0", AGENT_MAX_JUGEMENTS=1)
    def test_limite_configurable_et_cache_sans_double_facture(self):
        with trace_pour() as trace, patch("requests.post", return_value=reponse_jev(
                {"input_tokens": 10, "output_tokens": 2})) as post:
            jugement.juger("supprime", self.questions)
            jugement.juger("supprime", self.questions)
            reponse = jugement.juger("autre", self.questions)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(trace.total_jetons, 12)
        self.assertTrue(trace.usages[-1]["cache_decision"])
        self.assertEqual(reponse["q"]["statut"], "indisponible")

    @override_settings(JEV_API_KEY="", TYPESAFE_API_KEY="", JUGEMENT_REPLI_LLM="1")
    def test_repli_sdk_reel_avec_usage(self):
        with trace_pour() as trace, patch("services.agent_v2.modeles.modele_agir", return_value=modele_local()):
            reponse = jugement.juger("supprime", self.questions)
        self.assertEqual(reponse["q"]["statut"], "decision")
        self.assertEqual(trace.total_jetons, 36)
        self.assertEqual(trace.usages[0]["etapes"], 1)

    @override_settings(JEV_API_KEY="", TYPESAFE_API_KEY="", JUGEMENT_REPLI_LLM="1")
    def test_repli_sdk_invalide_conserve_usage_deja_consomme(self):
        with trace_pour() as trace, patch("services.agent_v2.modeles.modele_agir", return_value=modele_local(invalide=True)):
            reponse = jugement.juger("supprime", self.questions)
        self.assertEqual(reponse["q"]["statut"], "indisponible")
        self.assertEqual(trace.total_jetons, 36)
        self.assertFalse(trace.usages[0]["succes"])

    @override_settings(JEV_API_KEY="", TYPESAFE_API_KEY="", JUGEMENT_REPLI_LLM="1")
    def test_repli_sdk_annule_a_echeance(self):
        entrees, annulations, terminaisons = [], [], []
        async def lent(messages, info):
            entrees.append(time.monotonic())
            try:
                await asyncio.sleep(3)
            except asyncio.CancelledError:
                annulations.append(time.monotonic())
                raise
            terminaisons.append(True)
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                {"reponses": {"q": {"valeur": "oui", "confiance": 0.95}}})])
        # L'admission PostgreSQL doit finir avant d'éprouver l'annulation SDK.
        with trace_pour(1) as trace, patch("services.agent_v2.modeles.modele_agir", return_value=FunctionModel(lent)):
            depart = time.monotonic()
            reponse = jugement.juger("supprime", self.questions)
            duree = time.monotonic() - depart
        self.assertEqual(len(entrees), 1)
        self.assertLess(entrees[0], trace.echeance)
        self.assertEqual(len(annulations), 1)
        self.assertGreaterEqual(annulations[0], trace.echeance - 0.05)
        self.assertEqual(terminaisons, [])
        self.assertLess(duree, 2)
        self.assertEqual(reponse["q"]["statut"], "indisponible")
        self.assertEqual(len(trace.usages), 1)
        self.assertFalse(trace.usages[0]["usage_connu"])
        self.assertEqual(trace.usages[0]["etapes"], 1)
        self.assertGreater(trace.total_jetons, 0)
        self.assertTrue(trace.reservation.fermee)

    @override_settings(JEV_API_KEY="", TYPESAFE_API_KEY="", JUGEMENT_REPLI_LLM="1")
    def test_reponse_sdk_valide_json_mais_decision_invalide_reste_echec_mesure(self):
        def mauvaise_reponse(messages, info):
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                {"reponses": {"q": {"valeur": "SECRET-DU-MESSAGE", "confiance": 0.95}}})],
                usage=RequestUsage(input_tokens=30, output_tokens=6))
        with trace_pour() as trace, patch("services.agent_v2.modeles.modele_agir", return_value=FunctionModel(mauvaise_reponse)):
            reponse = jugement.juger("supprime", self.questions)
        self.assertEqual(reponse["q"]["statut"], "indisponible")
        self.assertEqual(trace.total_jetons, 36)
        self.assertFalse(trace.usages[0]["succes"])

    def test_trace_ne_garde_pas_texte_ni_secret(self):
        trace = TraceTour(time.monotonic() + 10)
        trace.ajouter({"message": "message prive", "secret": "cle", "entree": 10,
                       "sortie": 3, "raisonnement": 2, "usage_connu": True})
        self.assertEqual(trace.total_jetons, 13)
        self.assertNotIn("message", trace.usages[0])
        self.assertNotIn("secret", trace.usages[0])

    @override_settings(JEV_API_KEY="cle-factice", JUGEMENT_REPLI_LLM="0")
    def test_reponse_jev_invalide_ne_journalise_pas_le_contenu(self):
        questions = {"q": jugement.q_portee()}
        reponse = MagicMock()
        reponse.json.return_value = {"answers": {"q": {"choice": "SECRET-DU-MESSAGE", "confidence": 1}}}
        with trace_pour(), patch("requests.post", return_value=reponse), self.assertLogs("services.agent_v2.jugement", level="WARNING") as logs:
            jugement.juger("message prive", questions)
        self.assertNotIn("SECRET-DU-MESSAGE", " ".join(logs.output))
