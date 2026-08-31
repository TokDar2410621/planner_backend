"""
Les boutons garantis par le code dans v2, et la mesure des formulaires.

v2 emettait toujours done.quick_replies = [] et le frontend n'affiche des
chips que si la liste est non vide: les deux garanties de v1 (question de fin
de recurrence apres un import, creneaux libres sur une planification bloquee)
n'existaient donc pas pour les comptes bascules. Les helpers de v1 sont
REUTILISES; ces tests verifient la traduction du registre et le branchement,
pas le calcul des creneaux lui-meme.
"""
from datetime import date, time as dtime
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase

from core.models import ConversationMessage, RecurringBlock, UploadedDocument
from services.agent.tools.base import ToolResult
from services.agent_v2.boutons import appels_outils, boutons_forces
from services.agent_v2.redaction import ReponseDire
from services.agent_v2.registre import Registre

# Un jour lointain: jamais « aujourd'hui », donc aucun rognage des creneaux
# deja passes ne vient dependre de l'heure a laquelle on lance les tests.
JOUR = date(2030, 6, 10)

PHRASE_FIN = "jusqu'à quand veux-tu le garder à l'horaire ?"
CHIPS_FIN_PHYSIQUE = [
    {"label": "🏁 Je te donne la date de fin",
     "value": "Je vais te donner la date de fin pour Physique."},
    {"label": "♾️ Pas de fin prévue",
     "value": "Physique n'a pas de date de fin, garde-le tel quel."},
]


def _registre_conflit():
    r = Registre()
    r.ajouter("schedule_task_at",
              {"title": "Révision", "date": JOUR.isoformat(),
               "start_time": "10:00", "end_time": "11:00"},
              ToolResult(success=False, message="Conflit avec Cours",
                         data={"conflict": {"title": "Cours"}}))
    return r


def _consultation(**args):
    """Un find_free_slots reussi, dans la forme exacte que rend l'outil."""
    return ("find_free_slots", {"date": JOUR.isoformat(), **args}, ToolResult(
        success=True, message="2 créneau(x) libre(s).",
        data={"date": JOUR.isoformat(), "day_name": "mardi",
              "free_slots": [
                  {"start_time": "15:00", "end_time": "17:00", "duration_minutes": 120},
                  {"start_time": "18:00", "end_time": "20:00", "duration_minutes": 120}],
              "total_free_minutes": 240}))


class TraductionDuRegistreTests(TestCase):
    def test_le_registre_prend_le_format_des_helpers_de_v1(self):
        r = Registre()
        r.ajouter("create_block", {"title": "Maths"},
                  ToolResult(success=True, message="ok", data={"created": [{"id": 1}]}))
        self.assertEqual(appels_outils(r), [{
            "tool": "create_block",
            "args": {"title": "Maths"},
            "result": {"success": True, "data": {"created": [{"id": 1}]}},
        }])


class PlanificationBloqueeTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="chips", password="x")
        # La matinee est prise: le premier creneau libre commence a 15:00,
        # et c'est ce que les boutons doivent porter, pas l'heure refusee.
        RecurringBlock.objects.create(
            user=self.user, title="Cours", block_type="course",
            day_of_week=JOUR.weekday(), flexibility="fixed",
            start_time=dtime(7, 0), end_time=dtime(15, 0))

    def test_un_conflit_impose_des_creneaux_aux_heures_concretes(self):
        texte, chips = boutons_forces(
            self.user, "planifie ma révision", None, _registre_conflit(),
            "Le créneau est déjà occupé.", False)
        self.assertTrue(texte.startswith("Le créneau est déjà occupé."))
        self.assertIn("Ce créneau est pris", texte)
        self.assertIn("choisis", texte)
        self.assertTrue(1 <= len(chips) <= 3, chips)
        self.assertTrue(chips[0]["label"].startswith("🕐 15:00"), chips[0])
        for chip in chips:
            self.assertRegex(chip["label"], r"\d{2}:\d{2}")
            self.assertIn(f"Planifie « Révision » le {JOUR.isoformat()}", chip["value"])
            self.assertRegex(chip["value"], r"de \d{2}:\d{2} à \d{2}:\d{2}\.")

    def test_une_ecriture_reussie_n_impose_rien(self):
        r = _registre_conflit()
        r.ajouter("schedule_task_at",
                  {"title": "Révision", "date": JOUR.isoformat(),
                   "start_time": "15:00", "end_time": "16:00"},
                  ToolResult(success=True, message="cale",
                             data={"scheduled_block": {"id": 3}}))
        self.assertEqual(
            boutons_forces(self.user, "planifie ma révision", None, r, "Calé.", False),
            ("Calé.", []))

    def test_une_creation_reussie_sans_conflit_n_impose_rien(self):
        r = Registre()
        r.ajouter("create_block", {"title": "Maths"},
                  ToolResult(success=True, message="ok", data={"created": [{"id": 1}]}))
        self.assertEqual(
            boutons_forces(self.user, "ajoute maths", None, r, "Ajouté.", False),
            ("Ajouté.", []))

    def test_un_tour_sans_action_ni_intention_n_impose_rien(self):
        self.assertEqual(
            boutons_forces(self.user, "bonjour", None, Registre(), "Salut.", False),
            ("Salut.", []))


    def test_la_phrase_des_creneaux_porte_leurs_libelles_une_seule_fois(self):
        """Les chips ne sont pas persistees, le texte l'est: il nomme donc
        les creneaux proposes, a la suite de la phrase, sans les doubler."""
        texte, chips = boutons_forces(
            self.user, "planifie ma révision", None, _registre_conflit(),
            "Le créneau est déjà occupé.", False)
        self.assertIn("en voici de libres", texte)
        self.assertTrue(texte.endswith(", ".join(c["label"] for c in chips)), texte)
        for chip in chips:
            self.assertEqual(texte.count(chip["label"]), 1, texte)


class LectureDeCreneauxTests(TestCase):
    """La deuxieme jambe de v1 (find_free_slots reussi sans ecriture) ne
    distingue ni une lecture d'une planification, ni les mutations absentes
    de sa liste: v2 tranche AVANT de l'appeler."""

    def setUp(self):
        self.user = User.objects.create_user(username="lecture", password="x")

    def test_une_lecture_pure_de_creneaux_libres_n_impose_rien(self):
        """Un tap creerait un evenement que personne n'a demande."""
        r = Registre()
        r.ajouter(*_consultation())
        self.assertEqual(
            boutons_forces(self.user, "quand suis-je libre demain ?", None, r,
                           "Tu es libre de 15h à 17h et de 18h à 20h.", False),
            ("Tu es libre de 15h à 17h et de 18h à 20h.", []))

    def test_une_consultation_avec_intention_de_planifier_impose_des_creneaux(self):
        r = Registre()
        r.ajouter(*_consultation(min_duration_minutes=60))
        texte, chips = boutons_forces(
            self.user, "planifie ma révision demain", None, r,
            "Voici ce qui est libre.", False)
        self.assertEqual([c["label"] for c in chips],
                         ["🕐 15:00–16:00", "🕐 18:00–19:00"])
        self.assertTrue(texte.startswith("Voici ce qui est libre."))
        self.assertTrue(texte.endswith("🕐 15:00–16:00, 🕐 18:00–19:00"), texte)
        for chip in chips:
            self.assertIn(f"le {JOUR.isoformat()}", chip["value"])

    def test_une_consultation_apres_une_ecriture_tentee_impose_des_creneaux(self):
        """Aucun verbe dans le message, mais le tour a tente d'ecrire (echec
        sans conflit) puis consulte: l'intention est dans le registre."""
        r = Registre()
        r.ajouter("schedule_task_at",
                  {"title": "Révision", "date": JOUR.isoformat(),
                   "start_time": "25:00", "end_time": "26:00"},
                  ToolResult(success=False, message="Heure invalide."))
        r.ajouter(*_consultation(min_duration_minutes=60))
        texte, chips = boutons_forces(self.user, "et donc ?", None, r, "Voici.", False)
        self.assertEqual([c["label"] for c in chips],
                         ["🕐 15:00–16:00", "🕐 18:00–19:00"])

    def test_une_mutation_reussie_hors_liste_v1_puis_find_free_slots_n_impose_rien(self):
        """organize_day, optimize_week et cancel_scheduled_block ecrivent mais
        manquent a MUTATION_TOOLS de v1. Le verbe est dans le message pour que
        seule cette garde puisse expliquer le silence."""
        for outil in ("organize_day", "optimize_week", "cancel_scheduled_block"):
            with self.subTest(outil=outil):
                r = Registre()
                r.ajouter(outil, {"date": JOUR.isoformat()},
                          ToolResult(success=True, message="fait",
                                     data={"date": JOUR.isoformat()}))
                r.ajouter(*_consultation(min_duration_minutes=60))
                self.assertEqual(
                    boutons_forces(self.user, "organise et planifie ma journée",
                                   None, r, "Fait.", False),
                    ("Fait.", []))


class FinDeRecurrenceTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="import", password="x")
        self.doc = UploadedDocument.objects.create(
            user=self.user, file_name="horaire.pdf",
            document_type="course_schedule", processed=True,
            extracted_data={"courses": [{"name": "Physique"}]})
        RecurringBlock.objects.create(
            user=self.user, title="Physique", block_type="course", day_of_week=2,
            start_time=dtime(9, 0), end_time=dtime(11, 0), source_document=self.doc)

    def test_un_import_sans_date_de_fin_pose_la_question_avec_ses_boutons(self):
        texte, chips = boutons_forces(
            self.user, "voici mon horaire", self.doc, Registre(), "Import terminé.", True)
        self.assertTrue(texte.startswith("Import terminé."))
        self.assertIn("« Physique » n'a pas de date de fin", texte)
        self.assertIn(PHRASE_FIN, texte)
        self.assertEqual(chips, CHIPS_FIN_PHYSIQUE)

    def test_la_phrase_n_est_pas_doublee_si_le_texte_la_pose_deja(self):
        """Les chips restent forcees: seule la phrase est evitee."""
        deja = "Jusqu'à quand veux-tu garder Physique ?"
        texte, chips = boutons_forces(
            self.user, "voici mon horaire", self.doc, Registre(), deja, True)
        self.assertEqual(texte, deja)
        self.assertEqual(chips, CHIPS_FIN_PHYSIQUE)

    def test_un_import_traite_a_un_tour_precedent_n_impose_rien(self):
        self.assertEqual(
            boutons_forces(self.user, "c'est bon ?", self.doc, Registre(), "Oui.", False),
            ("Oui.", []))

    def test_un_import_dont_les_blocs_ont_une_fin_n_impose_rien(self):
        RecurringBlock.objects.filter(source_document=self.doc).update(
            end_date=date(2030, 12, 20))
        self.assertEqual(
            boutons_forces(self.user, "voici mon horaire", self.doc, Registre(), "Ok.", True),
            ("Ok.", []))

    def test_la_fin_de_recurrence_prime_sur_les_creneaux(self):
        texte, chips = boutons_forces(
            self.user, "voici mon horaire", self.doc, _registre_conflit(), "Ok.", True)
        self.assertEqual(chips, CHIPS_FIN_PHYSIQUE)
        self.assertNotIn("Ce créneau est pris", texte)


class BranchementDansLeFluxTests(TestCase):
    """Le drapeau « traite ce tour » se lit autour de _contexte_document, et
    le resultat doit atteindre done ET le message persiste."""

    def setUp(self):
        self.user = User.objects.create_user(username="flux", password="x")
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def _document(self, processed):
        doc = UploadedDocument.objects.create(
            user=self.user, file_name="horaire.pdf",
            document_type="course_schedule", processed=processed,
            extracted_data={"courses": [{"name": "Physique"}]})
        RecurringBlock.objects.create(
            user=self.user, title="Physique", block_type="course", day_of_week=2,
            start_time=dtime(9, 0), end_time=dtime(11, 0), source_document=doc)
        return doc

    def _done(self, doc, agir=None, message="voici mon horaire"):
        def _muet(self_agent, user, message, registre):
            return ""
        with patch.object(self.Agent, "_agir", agir or _muet), \
             patch.object(self.Agent, "_dire", return_value=ReponseDire(ouverture="Reçu.")):
            return self.Agent().process_message(self.user, message, doc)

    def test_le_document_qui_finit_son_analyse_pendant_le_tour_declenche_la_question(self):
        doc = self._document(processed=False)

        def _analyse_terminee(_secondes):
            UploadedDocument.objects.filter(pk=doc.pk).update(processed=True)

        with self.settings(ATTACHMENT_WAIT_SECONDS=1), \
             patch("services.agent_v2.agent.time.sleep", _analyse_terminee):
            done = self._done(doc)
        self.assertEqual(done["quick_replies"], CHIPS_FIN_PHYSIQUE)
        self.assertIn(PHRASE_FIN, done["response"])
        persiste = ConversationMessage.objects.filter(
            user=self.user, role="assistant").latest("created_at")
        self.assertEqual(persiste.content, done["response"])

    def test_un_document_deja_traite_a_l_arrivee_ne_declenche_rien(self):
        """Equivalence avec v1: le meme document, envoye a un tour ulterieur
        (« c'est bon ? »), ne repose pas la question a chaque fois."""
        done = self._done(self._document(processed=True))
        self.assertEqual(done["quick_replies"], [])
        self.assertNotIn(PHRASE_FIN, done["response"])


    def test_boutons_forces_recoit_le_message_brut_et_non_le_message_enrichi(self):
        """La troisieme jambe de v1 cherche un verbe et deux heures dans le
        message, et le contexte d'un import en est plein. Le document est deja
        traite: son contexte enrichit le message vu par AGIR sans declencher
        la question de fin, ce qui isole l'argument observe."""
        doc = self._document(processed=True)
        vus = {}

        def _observe(self_agent, user, message, registre):
            vus["agir"] = message
            return ""

        from services.agent_v2 import boutons
        with patch("services.agent_v2.agent.boutons_forces",
                   wraps=boutons.boutons_forces) as espion:
            done = self._done(doc, _observe)
        self.assertTrue(vus["agir"].startswith("voici mon horaire\n\n"), vus["agir"])
        espion.assert_called_once()
        self.assertEqual(espion.call_args.args[1], "voici mon horaire")
        self.assertEqual(done["quick_replies"], [])

    def test_la_phrase_persistee_contient_les_creneaux(self):
        """Les chips ne sont pas persistees. Le texte de done et la ligne en
        base sont le meme texte, et il nomme les creneaux proposes."""
        RecurringBlock.objects.create(
            user=self.user, title="Cours", block_type="course",
            day_of_week=JOUR.weekday(), flexibility="fixed",
            start_time=dtime(7, 0), end_time=dtime(15, 0))

        def _conflit(self_agent, user, message, registre):
            registre.ajouter("schedule_task_at",
                             {"title": "Révision", "date": JOUR.isoformat(),
                              "start_time": "10:00", "end_time": "11:00"},
                             ToolResult(success=False, message="Conflit avec Cours",
                                        data={"conflict": {"title": "Cours"}}))
            return ""

        done = self._done(None, _conflit, message="planifie ma révision")
        chips = done["quick_replies"]
        self.assertTrue(chips, done)
        self.assertTrue(chips[0]["label"].startswith("🕐 15:00"), chips[0])
        self.assertIn("en voici de libres", done["response"])
        self.assertTrue(
            done["response"].endswith(", ".join(c["label"] for c in chips)),
            done["response"])
        persiste = ConversationMessage.objects.filter(
            user=self.user, role="assistant").latest("created_at")
        self.assertEqual(persiste.content, done["response"])


class MesureDesFormulairesTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="mesure", password="x")
        from services.agent_v2 import PlannerAgentV2
        self.Agent = PlannerAgentV2

    def _logs(self, message, agir=None):
        def _muet(self_agent, user, message, registre):
            return ""
        with patch.object(self.Agent, "_agir", agir or _muet), \
             patch.object(self.Agent, "_dire", return_value=ReponseDire(ouverture="Ok.")), \
             self.assertLogs("services.agent_v2.agent", level="INFO") as logs:
            self.Agent().process_message(self.user, message)
        return [ligne for ligne in logs.output if "formulaire" in ligne]

    def test_un_formulaire_presente_est_journalise_avec_ses_types(self):
        def _presente(self_agent, user, message, registre):
            registre.ajouter("present_form", {}, ToolResult(
                success=True, message="Formulaire présenté.",
                data={"interactive_inputs": [
                    {"id": "duree", "type": "duration", "label": "Durée",
                     "question": "Combien de temps ?"},
                    {"id": "jour", "type": "date", "label": "Date",
                     "question": "Quel jour ?"},
                ]}))
            return ""
        lignes = self._logs("planifie une séance", _presente)
        self.assertEqual(len(lignes), 1)
        self.assertIn(
            f"formulaire presente user={self.user.id} champs=2 types=duration,date",
            lignes[0])

    def test_un_formulaire_echoue_n_est_pas_compte(self):
        def _echoue(self_agent, user, message, registre):
            registre.ajouter("present_form", {}, ToolResult(
                success=False, message="Aucun champ spécifié pour le formulaire."))
            return ""
        self.assertEqual(self._logs("planifie une séance", _echoue), [])

    def test_une_reponse_au_formulaire_est_journalisee(self):
        lignes = self._logs("Voici mes réponses :\nDurée: 1 h 30\nDate: demain (2026-09-01)")
        self.assertEqual(len(lignes), 1)
        self.assertIn(f"formulaire repondu user={self.user.id}", lignes[0])

    def test_un_formulaire_passe_est_journalise(self):
        lignes = self._logs("On verra ça plus tard, continuons sans formulaire.")
        self.assertEqual(len(lignes), 1)
        self.assertIn(f"formulaire passe user={self.user.id}", lignes[0])

    def test_un_message_ordinaire_ne_journalise_aucune_mesure(self):
        self.assertEqual(self._logs("qu'est-ce que j'ai demain ?"), [])
