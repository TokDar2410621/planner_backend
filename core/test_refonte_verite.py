"""Contre-exemples du contrat de preuve, avec les vrais outils et SQLite."""
from datetime import date, timedelta
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase

from core.models import RecurringBlock, RecurringBlockOverride, ScheduledBlock, Task
from services.agent.tools import execute_tool
from services.agent.tools.base import ToolResult
from services.agent_v2.mesure import verifier_prose
from services.agent_v2.reconciliation import reconcilier
from services.agent_v2.redaction import ReponseDire, bloc_factuel, composer
from services.agent_v2.registre import Registre


class PreuvesMutationTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="preuve")
        self.autre = User.objects.create_user(username="autre-preuve")
        self.registre = Registre()

    def agir(self, outil, **parametres):
        resultat = execute_tool(outil, self.user, parametres)
        self.assertTrue(resultat.success, resultat.message)
        return self.registre.ajouter(outil, parametres, resultat)

    def test_lecture_reussie_ne_prouve_pas_une_suppression(self):
        action = self.registre.ajouter("list_tasks", {}, ToolResult(success=True))
        reponse = ReponseDire(ouverture="J'ai supprimé tous tes cours.", refs=[action.id])
        propre, _, _ = verifier_prose(reponse, self.registre)
        self.assertNotIn("supprimé", composer(propre, self.registre, "", None).texte)

    def test_mutation_ne_valide_pas_un_autre_fait_ni_une_paraphrase(self):
        action = self.agir("create_task", title="Sport")
        reconcilier(self.user, self.registre)
        for texte in ("J'ai supprimé tes cours.", "Ton calendrier ne comporte dorénavant plus aucun cours."):
            with self.subTest(texte=texte):
                propre, _, _ = verifier_prose(ReponseDire(ouverture=texte, refs=[action.id]), self.registre)
                self.assertEqual(propre.ouverture, "")

    def test_succes_brut_attend_relecture_et_ne_saffiche_pas_comme_confirme(self):
        action = self.agir("create_task", title="Sport")
        self.assertEqual(self.registre.recus_confirmes(), [])
        self.assertNotIn("ajoutée", bloc_factuel(self.registre))
        self.assertNotEqual(action.recu.get("statut"), "confirme")

    def test_creation_reelle_confirmee_apres_relecture(self):
        self.agir("create_task", title="Sport")
        reconcilier(self.user, self.registre)
        self.assertEqual(len(self.registre.recus_confirmes()), 1)
        self.assertIn("Sport", bloc_factuel(self.registre))

    def test_date_cible_hors_semaine_courante_verifiee(self):
        self.agir("schedule_task_at", title="Voyage", date="2099-02-11", start_time="14:00", end_time="15:00")
        reconcilier(self.user, self.registre)
        self.assertEqual(len(self.registre.recus_confirmes()), 1)

    def test_valeur_contradictoire_empeche_confirmation(self):
        action = self.agir("create_task", title="Sport")
        Task.objects.filter(pk=action.donnees["task"]["id"]).update(title="Autre titre")
        reconcilier(self.user, self.registre)
        self.assertEqual(self.registre.recus_confirmes(), [])
        self.assertNotIn("Sport", bloc_factuel(self.registre))

    def test_date_differente_empeche_confirmation(self):
        action = self.agir("schedule_task_at", title="Voyage", date="2099-02-11", start_time="14:00", end_time="15:00")
        ScheduledBlock.objects.filter(pk=action.donnees["scheduled_block"]["id"]).update(date=date(2099, 2, 12))
        reconcilier(self.user, self.registre)
        self.assertEqual(self.registre.recus_confirmes(), [])

    def test_relecture_inaccessible_ne_confirme_pas(self):
        self.agir("create_task", title="Sport")
        with patch.object(Task.objects, "filter", side_effect=RuntimeError("lecture indisponible")):
            reconcilier(self.user, self.registre)
        self.assertEqual(self.registre.recus_confirmes(), [])

    def test_un_objet_autre_utilisateur_ne_prouve_rien(self):
        resultat = execute_tool("create_task", self.autre, {"title": "Privé"})
        self.registre.ajouter("create_task", {"title": "Privé"}, resultat)
        reconcilier(self.user, self.registre)
        self.assertEqual(self.registre.recus_confirmes(), [])

    def test_annulation_ciblee_retablie_avant_relecture_non_confirmee(self):
        self.agir("schedule_task_at", title="Voyage", date="2099-02-11", start_time="14:00", end_time="15:00")
        self.registre = Registre()
        self.agir("cancel_scheduled_block", title="Voyage", date="2099-02-11")
        execute_tool("schedule_task_at", self.user, {"title": "Voyage", "date": "2099-02-11", "start_time": "14:00", "end_time": "15:00"})
        reconcilier(self.user, self.registre)
        self.assertEqual(self.registre.recus_confirmes(), [])

    def test_occurrence_sautee_verifiee_a_sa_date_exacte(self):
        jour = date(2099, 2, 11)
        bloc = RecurringBlock.objects.create(user=self.user, title="Sport", block_type="sport", day_of_week=jour.weekday(), start_time="14:00", end_time="15:00")
        self.agir("skip_block_occurrence", title=bloc.title, date=jour.isoformat())
        reconcilier(self.user, self.registre)
        self.assertEqual(len(self.registre.recus_confirmes()), 1)
        bloc.exceptions.all().delete()
        reconcilier(self.user, self.registre)
        self.assertEqual(self.registre.recus_confirmes(), [])

    def test_nuit_verifie_les_deux_fragments(self):
        action = self.agir("schedule_task_at", title="Nuit", date="2099-02-11", start_time="22:00", end_time="02:00")
        reconcilier(self.user, self.registre)
        self.assertEqual(len(self.registre.recus_confirmes()), 1)
        ScheduledBlock.objects.filter(user=self.user, date="2099-02-12").delete()
        reconcilier(self.user, self.registre)
        self.assertEqual(self.registre.recus_confirmes(), [])

    def test_crud_blocs_taches_objectifs_et_preferences(self):
        scenarios = [
            ("create_block", {"title": "Sport", "block_type": "sport", "days": [2], "start_time": "14:00", "end_time": "15:00"}),
            ("create_task", {"title": "Travail", "deadline": "2099-02-11", "description": ""}),
            ("create_goal", {"title": "Apprendre", "goal_type": "short_term"}),
            ("update_preferences", {"min_sleep_hours": 8}),
        ]
        for outil, arguments in scenarios:
            with self.subTest(outil=outil):
                self.registre = Registre()
                self.agir(outil, **arguments)
                reconcilier(self.user, self.registre)
                self.assertEqual(len(self.registre.recus_confirmes()), 1, self.registre.actions[0].recu)

    def test_bloc_mis_a_jour_puis_supprime_est_reverifie(self):
        self.agir("create_block", title="Sport", block_type="sport", days=[2], start_time="14:00", end_time="15:00")
        bloc = RecurringBlock.objects.get(user=self.user)
        self.registre = Registre()
        self.agir("update_block", block_id=bloc.pk, start_time="15:00", end_time="16:00")
        reconcilier(self.user, self.registre)
        self.assertEqual(len(self.registre.recus_confirmes()), 1)
        self.registre = Registre()
        self.agir("delete_block", block_id=bloc.pk)
        reconcilier(self.user, self.registre)
        self.assertEqual(len(self.registre.recus_confirmes()), 1)

    def test_tache_completee_et_supprimee(self):
        self.agir("create_task", title="Sport")
        tache = Task.objects.get(user=self.user)
        self.registre = Registre()
        self.agir("complete_task", task_id=tache.pk)
        reconcilier(self.user, self.registre)
        self.assertEqual(len(self.registre.recus_confirmes()), 1)
        self.registre = Registre()
        self.agir("delete_task", task_id=tache.pk, confirm=True)
        reconcilier(self.user, self.registre)
        self.assertEqual(len(self.registre.recus_confirmes()), 1)

    def test_dedoublonnage_signale_existence_sans_creation(self):
        self.agir("create_task", title="Sport")
        self.registre = Registre()
        self.agir("create_task", title="Sport", priority=8)
        reconcilier(self.user, self.registre)
        self.assertEqual(self.registre.recus_confirmes(), [])
        self.assertTrue(self.registre.actions[0].etat_verifie)
        self.assertIn("déjà", bloc_factuel(self.registre))

    def test_proposition_sans_ecriture_reste_affichable(self):
        self.registre.ajouter("organize_day", {"date": "2099-02-11", "apply": False},
                              ToolResult(success=True, data={"date": "2099-02-11", "applied": False,
                                  "placed": [{"title": "Sport", "start_time": "14:00", "end_time": "15:00"}]}))
        reconcilier(self.user, self.registre)
        self.assertEqual(self.registre.recus_confirmes(), [])
        self.assertNotIn("relecture", bloc_factuel(self.registre))

    def test_restauration_deja_effective_ne_devient_pas_mutation(self):
        jour = date(2099, 2, 11)
        RecurringBlock.objects.create(user=self.user, title="Sport", block_type="sport", day_of_week=jour.weekday(), start_time="14:00", end_time="15:00")
        self.agir("restore_block_occurrence", title="Sport", date=jour.isoformat())
        reconcilier(self.user, self.registre)
        self.assertEqual(self.registre.recus_confirmes(), [])
        self.assertTrue(self.registre.actions[0].etat_verifie)

    def test_resultat_faux_meme_si_demande_correspond_base_ne_saffiche_pas(self):
        action = self.agir("schedule_task_at", title="Voyage", date="2099-02-11", start_time="14:00", end_time="15:00")
        action.donnees["scheduled_block"]["date"] = "2099-02-12"
        reconcilier(self.user, self.registre)
        self.assertEqual(self.registre.recus_confirmes(), [])
        self.assertNotIn("Voyage", bloc_factuel(self.registre))

    def test_conversation_explicitement_autorisee_garde_son_texte(self):
        propre, _, _ = verifier_prose(ReponseDire(ouverture="Bonjour !"), self.registre, autoriser_conversation=True)
        self.assertEqual(propre.ouverture, "Bonjour !")

    def test_sans_faits_rendus_la_prose_innocente_vit(self):
        """Le code n'a rien rendu: l'effacer repondrait « je n'ai pas compris »."""
        self.registre.ajouter("list_tasks", {}, ToolResult(success=True))
        propre, supprimees, _ = verifier_prose(
            ReponseDire(ouverture="Ok."), self.registre, sans_faits=True)
        self.assertEqual(propre.ouverture, "Ok.")
        self.assertEqual(supprimees, 0)

    def test_sans_faits_rendus_une_action_affirmee_tombe_quand_meme(self):
        """La garde large part, la garde phrase par phrase reste.

        C'est ici que le correctif pourrait avoir ouvert une porte: le modele
        s'attribue une suppression sur un tour ou le code n'a rien rendu.
        """
        action = self.registre.ajouter("list_tasks", {}, ToolResult(success=True))
        for texte in ("J'ai supprimé tous tes cours.",
                      "C'est fait, ton gym est déplacé à 20 h.",
                      "Parfait ! J'ai supprimé tous tes cours."):
            with self.subTest(texte=texte):
                propre, supprimees, fuites = verifier_prose(
                    ReponseDire(ouverture=texte, refs=[action.id]),
                    self.registre, sans_faits=True)
                self.assertNotIn("supprimé", propre.ouverture)
                self.assertNotIn("déplacé", propre.ouverture)
                self.assertGreaterEqual(supprimees, 1)
                self.assertTrue(fuites)

    def test_une_mutation_reussie_garde_la_garde_entiere(self):
        """Le code doit raconter ce qu'il a fait, jamais le modele."""
        self.agir("create_task", title="Sport")
        reconcilier(self.user, self.registre)
        propre, supprimees, _ = verifier_prose(
            ReponseDire(ouverture="Voilà !"), self.registre, sans_faits=True)
        self.assertEqual(propre.ouverture, "")
        self.assertEqual(supprimees, 1)

    def test_import_relecture_document_exact_et_bloc_archive_refuse(self):
        from core.models import UploadedDocument
        from services.agent_v2.importation import inscrire_import
        doc = UploadedDocument.objects.create(user=self.user, file_name="horaire.pdf", processed=True)
        bloc = RecurringBlock.objects.create(user=self.user, source_document=doc, title="Cours",
                                             block_type="course", day_of_week=2, start_time="14:00", end_time="15:00")
        action = inscrire_import(self.registre, self.user, doc)
        self.assertEqual(action.parametres["document_id"], doc.pk)
        reconcilier(self.user, self.registre)
        self.assertEqual(len(self.registre.recus_confirmes()), 1)
        RecurringBlock.objects.filter(pk=bloc.pk).update(active=False)
        reconcilier(self.user, self.registre)
        self.assertEqual(self.registre.recus_confirmes(), [])

    def test_plan_relit_identite_et_valeurs_a_date_exacte(self):
        jour = date(2099, 2, 11)
        bloc = RecurringBlock.objects.create(user=self.user, title="Sport", block_type="sport",
                                             day_of_week=jour.weekday(), start_time="14:00", end_time="15:00")
        place = {"block_id": bloc.pk, "title": "Sport", "start_time": "14:00", "end_time": "15:00"}
        jours = [{"date": (jour + timedelta(days=i)).isoformat(),
                  "placed": [place] if i == 0 else [], "moved": [place] if i == 0 else []}
                 for i in range(7)]
        self.registre.ajouter("optimize_week", {"start_date": jour.isoformat(), "apply": True},
                             ToolResult(success=True, data={"applied": True, "start_date": jour.isoformat(),
                                 "moved_count": 1, "days": jours}))
        # Un rendu aux bonnes heures sans écriture ne prouve aucune application.
        reconcilier(self.user, self.registre)
        self.assertEqual(self.registre.recus_confirmes(), [])
        override = RecurringBlockOverride.objects.create(
            user=self.user, recurring_block=bloc, date=jour,
            start_time="14:00", end_time="15:00")
        reconcilier(self.user, self.registre)
        self.assertEqual(len(self.registre.recus_confirmes()), 1)
        RecurringBlockOverride.objects.filter(pk=override.pk).update(start_time="16:00", end_time="17:00")
        reconcilier(self.user, self.registre)
        self.assertEqual(self.registre.recus_confirmes(), [])

    def test_override_autre_date_ou_autre_utilisateur_ne_prouve_pas_le_plan(self):
        jour = date(2099, 2, 11)
        bloc = RecurringBlock.objects.create(user=self.user, title="Sport", block_type="sport",
                                             day_of_week=jour.weekday(), start_time="14:00", end_time="15:00")
        place = {"block_id": bloc.pk, "title": "Sport", "start_time": "14:00", "end_time": "15:00"}
        self.registre.ajouter("organize_day", {"date": jour.isoformat(), "apply": True},
                             ToolResult(success=True, data={"applied": True, "date": jour.isoformat(),
                                 "moved_count": 1, "placed": [place], "moved": [place]}))
        override = RecurringBlockOverride.objects.create(user=self.user, recurring_block=bloc,
                                                         date=jour + timedelta(days=7),
                                                         start_time="14:00", end_time="15:00")
        reconcilier(self.user, self.registre)
        self.assertEqual(self.registre.recus_confirmes(), [])
        # Corruption injectée par UPDATE, qui contourne volontairement clean().
        RecurringBlockOverride.objects.filter(pk=override.pk).update(user=self.autre, date=jour)
        reconcilier(self.user, self.registre)
        self.assertEqual(self.registre.recus_confirmes(), [])

    def test_bloc_archive_entre_mutation_et_relecture_ne_confirme_pas(self):
        self.agir("create_block", title="Sport", block_type="sport", days=[2], start_time="14:00", end_time="15:00")
        RecurringBlock.objects.filter(user=self.user).update(active=False)
        reconcilier(self.user, self.registre)
        self.assertEqual(self.registre.recus_confirmes(), [])

