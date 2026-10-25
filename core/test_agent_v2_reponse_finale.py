"""Le message du chat conserve la réponse du modèle après une lecture d'outil."""
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TransactionTestCase

from core.models import Goal
from core.test_agent_v2_jugement import juger_script
from services.agent.tools import execute_tool
from services.agent.tools.base import ToolResult
from services.agent_v2 import agent as module_agent
from services.agent_v2.agent import PlannerAgentV2
from services.agent_v2.mesure import verifier_prose
from services.agent_v2.redaction import ReponseDire, bloc_factuel, composer
from services.agent_v2.registre import Registre
from services.agent_v2.validation_sortie import retour_validation


class FiltreLectureTests(SimpleTestCase):
    def test_rejet_de_sortie_devient_un_retour_exploitable(self):
        registre = Registre()
        registre.ajouter("create_task", {"title": "Rapport"}, ToolResult(
            success=False, issue="echec", cause="base_indisponible"))

        retour = retour_validation("J'ai créé la tâche Rapport.", registre)

        self.assertIn("VALIDATION DE LA RÉPONSE", retour)
        self.assertIn("autre outil", retour)
        self.assertIsNone(retour_validation("La base est indisponible.", registre))

    def test_rejet_apres_mutation_ne_certifie_pas_la_paraphrase(self):
        registre = Registre()
        registre.ajouter("create_task", {"title": "Sport"}, ToolResult(
            success=True, data={"task": {"title": "Sport"}}))

        retour = retour_validation(
            "Ton calendrier ne comporte dorénavant plus aucun cours.", registre)

        self.assertIn("tour de modification", retour)
        self.assertIn("a_verifier", retour)

    def test_offre_sans_affirmation_survit_apres_mutation(self):
        registre = Registre()
        registre.ajouter("create_task", {"title": "Sport"}, ToolResult(
            success=True, data={"task": {"title": "Sport"}}))
        brut = ReponseDire(ouverture="Je peux aussi lui fixer une échéance.")

        propre, _, _ = verifier_prose(brut, registre)

        self.assertEqual(propre.ouverture, brut.ouverture)
        self.assertIsNone(retour_validation(brut.ouverture, registre))

    def test_lecture_non_demandee_revient_au_modele(self):
        registre = Registre()
        registre.ajouter("get_today_schedule", {}, ToolResult(
            success=True,
            data={"blocks": [{"title": "Conception d'applications",
                              "start_time": "08:00", "end_time": "11:00"}]},
            public="- 8 h à 11 h · Conception d'applications",
        ))
        faits = bloc_factuel(registre)

        retour = retour_validation(
            "Voici ton planning : Conception d'applications de 8 h à 11 h.",
            registre, faits_controle=faits)

        self.assertIn("lecture non demandée", retour)
        self.assertIsNone(retour_validation("Je t'écoute.", registre,
                                           faits_controle=faits))

    def test_validation_reprise_peut_appeler_un_autre_outil(self):
        from pydantic_ai import Agent, ModelRetry
        from pydantic_ai.messages import (ModelResponse, RetryPromptPart,
                                          TextPart, ToolCallPart)
        from pydantic_ai.models.function import FunctionModel

        registre = Registre()
        appels = []

        def modele(messages, _info):
            appels.append(messages)
            if len(appels) == 1:
                return ModelResponse(parts=[TextPart(content="J'ai supprimé tes cours.")])
            if len(appels) == 2:
                return ModelResponse(parts=[ToolCallPart(tool_name="verifier_etat", args={})])
            return ModelResponse(parts=[TextPart(content="La lecture a échoué : la base est indisponible.")])

        def verifier_etat() -> str:
            registre.ajouter("list_tasks", {}, ToolResult(
                success=False, issue="echec", cause="base_indisponible"))
            return "La base est indisponible."

        agent = Agent(FunctionModel(modele), tools=[verifier_etat], output_retries=2)

        @agent.output_validator
        def valider(_contexte, sortie):
            retour = retour_validation(sortie, registre)
            if retour:
                raise ModelRetry(retour)
            return sortie

        resultat = agent.run_sync("Supprime mes cours")

        self.assertEqual(resultat.output, "La lecture a échoué : la base est indisponible.")
        self.assertEqual(len(appels), 3)
        self.assertTrue(any(
            isinstance(part, RetryPromptPart)
            for message in appels[1] for part in message.parts
        ))
        self.assertEqual([a.outil for a in registre.actions], ["list_tasks"])

    def test_liste_redigee_par_le_modele_reste_visible(self):
        registre = Registre()
        registre.ajouter("list_goals", {}, ToolResult(
            success=True,
            data={"goals": [{"title": "Trouver 10 utilisateurs de proveIt"}]},
            public="Tes objectifs :\n- Trouver 10 utilisateurs de proveIt",
        ))
        brut = ReponseDire(
            ouverture="Voici tes objectifs :\n- Trouver 10 utilisateurs de proveIt",
        )

        propre, _, _ = verifier_prose(brut, registre)
        texte = composer(propre, registre, "", None).texte

        self.assertIn("Trouver 10 utilisateurs de proveIt", texte)

    def test_date_absente_lue_et_expliquee_reste_visible(self):
        registre = Registre()
        registre.ajouter("list_goals", {}, ToolResult(
            success=True,
            data={"goals": [{"title": "Trouver 10 utilisateurs de proveIt",
                             "deadline": None}], "count": 1},
            public="Tes objectifs :\n- Trouver 10 utilisateurs de proveIt",
        ))
        faits = bloc_factuel(registre)
        brut = ReponseDire(ouverture="Cet objectif n'a pas de date limite fixée.")
        self.assertIn("n'a pas de date limite fixée", composer(brut, registre, faits, None).texte)

        propre, _, _ = verifier_prose(brut, registre, sans_faits=not faits)
        texte = composer(propre, registre, faits, None).texte

        self.assertIn("n'a pas de date limite fixée", texte)

    def test_echec_outil_explique_par_le_modele_reste_visible(self):
        registre = Registre()
        registre.ajouter("list_goals", {}, ToolResult(
            success=False,
            issue="echec",
            cause="service_indisponible",
            public="Je n'ai pas pu lire tes objectifs.",
        ))
        faits = bloc_factuel(registre)
        brut = ReponseDire(
            ouverture="Je n'ai pas pu consulter tes objectifs : le service est indisponible.",
            suite="Réessaie plus tard.",
        )

        propre, _, _ = verifier_prose(brut, registre, sans_faits=not faits)
        texte = composer(propre, registre, faits, None).texte

        self.assertIn("le service est indisponible", texte)
        self.assertIn("Réessaie plus tard", texte)

    def test_succes_partiel_reste_explicable(self):
        registre = Registre()
        registre.ajouter("list_goals", {}, ToolResult(
            success=True,
            issue="partiel",
            cause="une_source_indisponible",
            data={"goals": [{"title": "Objectif A"}]},
            public="Tes objectifs :\n- Objectif A",
        ))
        brut = ReponseDire(
            ouverture="J'ai trouvé l'objectif A. Une autre source est indisponible."
        )

        propre, _, _ = verifier_prose(brut, registre)
        texte = composer(propre, registre, "", None).texte

        self.assertIn("Une autre source est indisponible", texte)


class ReponseApresOutilTests(TransactionTestCase):
    def test_mutation_relu_puis_reprise_sans_nouvelle_ecriture(self):
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
        from pydantic_ai.models.function import DeltaToolCall, FunctionModel

        user = User.objects.create_user(username="reprise-mutation")
        registre = Registre()
        appels_outils = []
        etapes = []

        def etape(messages):
            parties = [p for m in messages for p in getattr(m, "parts", [])]
            retours = [p.tool_name for p in parties if isinstance(p, ToolReturnPart)]
            if "list_tasks" in retours:
                return "apres_lecture"
            if "create_task" in retours:
                return "apres_creation"
            return "creation"

        def modele(messages, _info):
            phase = etape(messages)
            etapes.append(phase)
            if phase == "creation":
                return ModelResponse(parts=[ToolCallPart(tool_name="create_task", args={"title": "Sport"})])
            if phase == "apres_creation":
                if etapes.count(phase) == 1:
                    return ModelResponse(parts=[TextPart(content="J'ai supprimé tous tes cours.")])
                return ModelResponse(parts=[ToolCallPart(tool_name="list_tasks", args={})])
            return ModelResponse(parts=[TextPart(content="La tâche Sport est présente.")])

        async def en_flux(messages, _info):
            phase = etape(messages)
            etapes.append(phase)
            if phase == "creation":
                yield {0: DeltaToolCall(name="create_task", json_args='{"title":"Sport"}')}
            elif phase == "apres_creation":
                if etapes.count(phase) == 1:
                    yield "J'ai supprimé tous tes cours."
                else:
                    yield {0: DeltaToolCall(name="list_tasks", json_args="{}")}
            else:
                yield "La tâche Sport est présente."

        def create_task(title: str) -> str:
            appels_outils.append("create_task")
            resultat = execute_tool("create_task", user, {"title": title})
            registre.ajouter("create_task", {"title": title}, resultat)
            return resultat.to_string()

        def list_tasks() -> str:
            appels_outils.append("list_tasks")
            resultat = execute_tool("list_tasks", user, {})
            registre.ajouter("list_tasks", {}, resultat)
            return resultat.to_string()

        with patch.object(module_agent, "modele_agir",
                          return_value=FunctionModel(modele, stream_function=en_flux)), \
             patch("services.agent_v2.budget_tour.modele_borne", side_effect=lambda m, _tag: m), \
             patch.object(module_agent, "outils_pour_le_modele", return_value=[create_task, list_tasks]), \
             patch.object(module_agent, "prompt_agir", return_value="Réponds à la demande."), \
             patch.object(PlannerAgentV2, "_historique", return_value=[]):
            reponse = PlannerAgentV2()._boucle(user, "Crée la tâche Sport", registre)

        self.assertEqual(appels_outils, ["create_task", "list_tasks"])
        self.assertEqual(len(registre.recus_confirmes()), 1)
        self.assertNotIn("supprimé", reponse.ouverture)

    def test_boucle_planner_renvoie_le_rejet_puis_permet_une_lecture(self):
        from pydantic_ai.messages import (ModelResponse, RetryPromptPart,
                                          TextPart, ToolCallPart, ToolReturnPart)
        from pydantic_ai.models.function import DeltaToolCall, FunctionModel

        user = User.objects.create_user(username="reprise-validation")
        registre = Registre()
        appels = []

        def etape(messages):
            parties = [p for m in messages for p in getattr(m, "parts", [])]
            if any(isinstance(p, ToolReturnPart) for p in parties):
                return "fin"
            if any(isinstance(p, RetryPromptPart) for p in parties):
                return "lecture"
            return "rejet"

        def modele(messages, _info):
            etat = etape(messages)
            appels.append(etat)
            if etat == "rejet":
                return ModelResponse(parts=[TextPart(content="J'ai supprimé tes cours.")])
            if etat == "lecture":
                return ModelResponse(parts=[ToolCallPart(tool_name="verifier_etat", args={})])
            return ModelResponse(parts=[TextPart(
                content="Je n'ai pas pu vérifier tes tâches : la base est indisponible.")])

        async def en_flux(messages, _info):
            etat = etape(messages)
            appels.append(etat)
            if etat == "rejet":
                yield "J'ai supprimé tes cours."
            elif etat == "lecture":
                yield {0: DeltaToolCall(name="verifier_etat", json_args="{}")}
            else:
                yield "Je n'ai pas pu vérifier tes tâches : la base est indisponible."

        def verifier_etat() -> str:
            registre.ajouter("list_tasks", {}, ToolResult(
                success=False, issue="echec", cause="base_indisponible"))
            return "La base est indisponible."

        with patch.object(module_agent, "modele_agir",
                          return_value=FunctionModel(modele, stream_function=en_flux)), \
             patch("services.agent_v2.budget_tour.modele_borne", side_effect=lambda m, _tag: m), \
             patch.object(module_agent, "outils_pour_le_modele", return_value=[verifier_etat]), \
             patch.object(module_agent, "prompt_agir", return_value="Réponds à la demande."), \
             patch.object(PlannerAgentV2, "_historique", return_value=[]):
            reponse = PlannerAgentV2()._boucle(user, "Supprime mes cours", registre)

        self.assertEqual(appels, ["rejet", "lecture", "fin"])
        self.assertIn("base est indisponible", reponse.ouverture)
        self.assertEqual([a.outil for a in registre.actions], ["list_tasks"])

    def test_echec_de_mutation_est_explique_sans_succes_invente(self):
        user = User.objects.create_user(username="creation-en-echec")

        def boucle(agent, utilisateur, message, registre):
            agent.signaler_outil(registre.ajouter("create_task", {"title": "Rapport"}, ToolResult(
                success=False,
                issue="echec",
                cause="base_indisponible",
                public="La tâche n'a pas été créée.",
            )))
            return ReponseDire(
                ouverture="Je n'ai pas pu créer la tâche Rapport : la base est indisponible."
            )

        with patch("services.agent_v2.jugement.juger", juger_script({})), \
             patch.object(module_agent, "_charger_appliquer_choix",
                          return_value=lambda *args, **kwargs: []), \
             patch.object(PlannerAgentV2, "_voie_rapide_sociale", return_value=False), \
             patch.object(PlannerAgentV2, "_boucle", new=boucle):
            reponse = PlannerAgentV2().process_message(
                user, "Crée la tâche Rapport", generate_quick_replies=False)

        self.assertIn("la base est indisponible", reponse["response"])
        self.assertIn("La tâche n'a pas été créée.", reponse["response"])

    def test_question_sur_echeance_absente_recoit_la_reponse_du_modele(self):
        user = User.objects.create_user(username="objectif-sans-echeance")
        Goal.objects.create(user=user, title="Trouver 10 utilisateurs de proveIt")

        def boucle(agent, utilisateur, message, registre):
            resultat = execute_tool("list_goals", utilisateur, {})
            self.assertTrue(resultat.success)
            self.assertIsNone(resultat.data["goals"][0]["deadline"])
            agent.signaler_outil(registre.ajouter("list_goals", {}, resultat))
            return ReponseDire(
                ouverture="Cet objectif n'a pas de date limite fixée.",
                suite="Je peux t'aider à en ajouter une.",
            )

        with patch("services.agent_v2.jugement.juger", juger_script({})), \
             patch.object(module_agent, "_charger_appliquer_choix",
                          return_value=lambda *args, **kwargs: []), \
             patch.object(PlannerAgentV2, "_voie_rapide_sociale", return_value=False), \
             patch.object(PlannerAgentV2, "_boucle", new=boucle):
            reponse = PlannerAgentV2().process_message(
                user, "Quelle est la date limite de mon objectif ?",
                generate_quick_replies=False)

        self.assertIn("n'a pas de date limite fixée", reponse["response"])
        self.assertIn("Je peux t'aider à en ajouter une", reponse["response"])
        self.assertLess(reponse["response"].index("Tes objectifs :"),
                        reponse["response"].index("n'a pas de date limite fixée"))

    def test_echec_d_outil_est_explique_par_le_modele(self):
        user = User.objects.create_user(username="objectifs-en-echec")

        def boucle(agent, utilisateur, message, registre):
            agent.signaler_outil(registre.ajouter("list_goals", {}, ToolResult(
                success=False,
                issue="echec",
                cause="service_indisponible",
                public="Je n'ai pas pu lire tes objectifs.",
            )))
            return ReponseDire(
                ouverture="La lecture de tes objectifs a échoué : le service est indisponible.",
                suite="Réessaie plus tard.",
            )

        with patch("services.agent_v2.jugement.juger", juger_script({})), \
             patch.object(module_agent, "_charger_appliquer_choix",
                          return_value=lambda *args, **kwargs: []), \
             patch.object(PlannerAgentV2, "_voie_rapide_sociale", return_value=False), \
             patch.object(PlannerAgentV2, "_boucle", new=boucle):
            reponse = PlannerAgentV2().process_message(
                user, "Quels sont mes objectifs actuellement ?",
                generate_quick_replies=False)

        self.assertIn("le service est indisponible", reponse["response"])
        self.assertIn("Réessaie plus tard", reponse["response"])
        self.assertNotEqual(reponse["response"], "Je n'ai pas pu lire tes objectifs.")
