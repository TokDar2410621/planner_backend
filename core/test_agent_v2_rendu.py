"""
Le rendu en francais des faits, des lectures et des questions du code.

Chaque sortie est ce que l'utilisateur lit: aucune forme machine (« (s) »,
ISO, HH:MM, « Refus », nom d'outil, anglais) ne doit y passer. Les formes
viennent du catalogue de l'enquete answer-rendering (S02 a S19).

Date de reference: lundi 2026-09-14, passee a chaque appel.
"""
from datetime import date

from django.test import SimpleTestCase

from services.agent.tools.base import ToolResult
from services.agent_v2 import rendu
from services.agent_v2.rendu import (
    date_courte, heure, jour, jours, marqueurs_bruts, plage, pluriel,
    rendre_demandes, rendre_faits, rendre_lecture,
)
from services.agent_v2.registre import Registre

AUJ = date(2026, 9, 14)
EM_DASH = "\u2014"


def registre(*entrees):
    """entrees: (outil, parametres, succes, donnees[, message])."""
    r = Registre()
    for e in entrees:
        outil, params, succes, donnees = e[:4]
        message = e[4] if len(e) > 4 else ""
        r.ajouter(outil, params, ToolResult(success=succes, data=donnees, message=message))
    return r


def faits(r, cles_posees=None):
    return rendre_faits(r, aujourdhui=AUJ, cles_posees=cles_posees)


def lecture(r):
    return rendre_lecture(r, aujourdhui=AUJ)


def cree(titre, dow, debut, fin, ident=None):
    return {"id": ident, "title": titre, "day_of_week": dow,
            "day_name": rendu.JOURS[dow].capitalize(), "start_time": debut, "end_time": fin}


class FormateursTests(SimpleTestCase):
    def test_heures(self):
        self.assertEqual(heure("09:00"), "9 h")
        self.assertEqual(heure("09:30"), "9 h 30")
        self.assertEqual(heure("12:00"), "12 h")
        self.assertEqual(heure("00:00"), "minuit")
        self.assertEqual(heure("09:05"), "9 h 05")
        self.assertEqual(plage("10:00", "11:50"), "10 h à 11 h 50")
        self.assertEqual(plage("19:00", "02:00"), "19 h à 2 h")
        self.assertEqual(plage("22:00", "23:59"), "22 h à minuit")

    def test_dates(self):
        self.assertEqual(date_courte("2026-09-14", AUJ), "aujourd'hui")
        self.assertEqual(date_courte("2026-09-15", AUJ), "demain")
        self.assertEqual(date_courte("2026-09-13", AUJ), "hier")
        self.assertEqual(date_courte("2026-09-24", AUJ), "jeu. 24 sept.")
        self.assertEqual(date_courte("2026-10-15", AUJ), "jeu. 15 oct.")
        self.assertEqual(date_courte("2026-10-01", AUJ), "jeu. 1er oct.")

    def test_echeance_utc_convertie_en_heure_murale(self):
        # 2026-09-16 03:59 UTC = 2026-09-15 23:59 a Toronto: c'est demain.
        self.assertEqual(date_courte("2026-09-16T03:59:00+00:00", AUJ), "demain")

    def test_jours_et_pluriels(self):
        self.assertEqual(jours([0, 2, 4]), "lundi, mercredi et vendredi")
        self.assertEqual(jours([0, 1, 2, 3, 4]), "du lundi au vendredi")
        self.assertEqual(jours([5, 6]), "samedi et dimanche")
        self.assertEqual(jour(3, pluriel=True), "jeudis")
        self.assertEqual(jour(0), "lundi")
        self.assertEqual(pluriel(1, "cours", "cours"), "1 cours")
        self.assertEqual(pluriel(2, "bloc"), "2 blocs")
        self.assertEqual(pluriel(0, "bloc"), "0 bloc")

    def test_jours_nommes_ou_abreges(self):
        self.assertEqual(rendu._dow("jeu."), 3)
        self.assertEqual(rendu._dow("Mardi"), 1)
        self.assertEqual(rendu._dow("lundis"), 0)
        self.assertEqual(rendu._dow("Sunday"), 6)
        self.assertIsNone(rendu._dow("?"))
        self.assertIsNone(rendu._dow(7))

    def test_dimanche_est_six(self):
        """dow 0 = lundi: le decalage americain dimanche=0 ne doit pas revenir."""
        self.assertEqual(jour(6), "dimanche")
        self.assertEqual(date_courte("2026-09-20", AUJ), "dim. 20 sept.")

    def test_marqueurs_bruts(self):
        brut = marqueurs_bruts(
            "- Refus: 0 bloc(s) créé(s) 2026-09-17 10:00 create_block (e1) 16 x create_block #12")
        for nom in ("refus", "pluriel_machine", "date_iso", "heure_hhmm", "nom_outil",
                    "ref_registre", "compte_outil", "id_interne"):
            self.assertIn(nom, brut)
        self.assertEqual(brut, sorted(set(brut)))
        self.assertEqual(marqueurs_bruts("Mardi, 8 h à 9 h 50 · Philosophie"), [])
        self.assertIn("anglais", marqueurs_bruts("1 block created"))
        self.assertIn("ecart", marqueurs_bruts("- Ecart: CREE mais dans le passe"))
        self.assertEqual(marqueurs_bruts(""), [])


class CreationDeBlocsTests(SimpleTestCase):
    def test_creation_partielle_lisible(self):
        r = registre(("create_block",
                      {"title": "Statistiques", "days": ["lundi", "vendredi"],
                       "start_time": "10:00", "end_time": "12:00"}, True,
                      {"created": [{"title": "Statistiques", "day_name": "Vendredi",
                                    "day_of_week": 4, "start_time": "10:00", "end_time": "12:00"}],
                       "skipped": [{"day": 0, "day_name": "Lundi", "motif": "chevauchement",
                                    "titre": "Statistiques", "debut": "10:00", "fin": "12:00",
                                    "avec": {"titre": "Calcul différentiel", "debut": "10:00",
                                             "fin": "11:50"}}]},
                      "1 bloc(s) créé(s): Statistiques (10:00-12:00) les Vendredi. 1 sauté(s): x"))
        sortie = faits(r)
        for attendu in ("Statistiques", "vendredi", "10 h à 12 h", "Calcul différentiel",
                        "10 h à 11 h 50"):
            self.assertIn(attendu, sortie)
        for interdit in ("(s)", "Refus", "sauté", "10:00", "create_block", "Vendredi", EM_DASH):
            self.assertNotIn(interdit, sortie)
        self.assertEqual(marqueurs_bruts(sortie), [])

    def test_refus_total_de_creation(self):
        r = registre(("create_block",
                      {"title": "Statistiques", "days": ["lundi"],
                       "start_time": "10:00", "end_time": "12:00"}, False,
                      {"created": [], "skipped": [{
                          "day": 0, "day_name": "Lundi", "motif": "chevauchement",
                          "titre": "Statistiques", "debut": "10:00", "fin": "12:00",
                          "avec": {"titre": "Calcul différentiel", "debut": "10:00", "fin": "11:50"}}]},
                      "0 bloc(s) créé(s): Statistiques (10:00-12:00). 1 sauté(s): ..."))
        sortie = faits(r)
        self.assertIn("pas", sortie)
        self.assertIn("Calcul différentiel", sortie)
        self.assertEqual(marqueurs_bruts(sortie), [])

    def test_refus_sans_motif_structure_lit_la_raison(self):
        """Repli tant que l'outil ne donne pas encore `motif` et `avec`."""
        r = registre(("create_block",
                      {"title": "Statistiques", "days": [0], "start_time": "10:00", "end_time": "12:00"},
                      False,
                      {"created": [], "skipped": [{
                          "day": 0, "day_name": "Lundi",
                          "reason": "Chevauchement avec 'Calcul différentiel' (10:00-11:50)"}]}))
        sortie = faits(r)
        self.assertIn("Calcul différentiel (10 h à 11 h 50)", sortie)
        self.assertIn("le lundi", sortie)
        self.assertEqual(marqueurs_bruts(sortie), [])

    def test_doublon(self):
        r = registre(("create_block",
                      {"title": "Philosophie", "days": ["mardi"], "start_time": "08:00",
                       "end_time": "09:50"}, False,
                      {"created": [], "skipped": [{
                          "day": 1, "day_name": "Mardi", "motif": "doublon", "titre": "Philosophie",
                          "debut": "08:00", "fin": "09:50",
                          "avec": {"titre": "Philosophie", "debut": "08:00", "fin": "09:50"}}]}))
        sortie = faits(r)
        for attendu in ("déjà", "mardi", "8 h"):
            self.assertIn(attendu, sortie)
        self.assertEqual(marqueurs_bruts(sortie), [])

    def test_doublon_sans_motif_structure(self):
        r = registre(("create_block",
                      {"title": "Philosophie", "days": [1], "start_time": "08:00", "end_time": "09:50"},
                      False,
                      {"created": [], "skipped": [{
                          "day": 1, "day_name": "Mardi",
                          "reason": "'Philosophie' existe déjà le Mardi à 08:00 (aucun doublon créé)"}]}))
        sortie = faits(r)
        self.assertIn("Philosophie est déjà à ton horaire le mardi à 8 h", sortie)

    def test_groupement_sans_nom_d_outil(self):
        titres = [("Anglais", 0), ("Biologie", 1), ("Biologie", 3), ("Histoire", 4),
                  ("Chimie", 2), ("Physique", 5)]
        r = registre(*[("create_block", {"title": t}, True,
                        {"created": [cree(t, d, "08:00", "09:30", ident=i)], "skipped": []})
                       for i, (t, d) in enumerate(titres)])
        sortie = faits(r)
        self.assertNotIn(" x ", sortie)
        self.assertNotIn("create_block", sortie)
        self.assertTrue("6 blocs" in sortie or all(t in sortie for t, _ in titres))
        self.assertIn("les mardis et jeudis", sortie)  # Biologie fusionnee
        self.assertEqual(marqueurs_bruts(sortie), [])

    def test_au_dela_de_cinq_titres_une_ligne_par_famille(self):
        titres = ["Anglais", "Biologie", "Histoire", "Chimie", "Physique", "Philosophie"]
        r = registre(*[("create_block", {"title": t}, True,
                        {"created": [cree(t, i, "08:00", "09:30", ident=i)], "skipped": []})
                       for i, t in enumerate(titres)])
        sortie = faits(r)
        self.assertIn("6 blocs ajoutés à ton horaire", sortie)
        self.assertIn("Anglais, Biologie, Histoire, Chimie, Physique et Philosophie", sortie)
        self.assertEqual(sortie.count("\n"), 0)

    def test_rejeu_idempotent_ne_compte_pas_double(self):
        donnees = {"created": [cree("Yoga", 6, "09:00", "10:00", ident=9)], "skipped": []}
        r = registre(("create_block", {"title": "Yoga"}, True, donnees),
                     ("create_block", {"title": "Yoga"}, True, donnees))
        self.assertEqual(faits(r), "Ajouté : Yoga, les dimanches de 9 h à 10 h.")

    def test_borne_de_la_semaine_dite(self):
        r = registre(("create_block", {"title": "Étude"}, True,
                      {"created": [cree("Étude", 0, "16:00", "18:00", 3)], "skipped": [],
                       "borne_auto": {"end_date": "2026-09-20"}}))
        sortie = faits(r)
        self.assertIn("cette semaine seulement (jusqu'au dim. 20 sept.)", sortie)
        self.assertEqual(marqueurs_bruts(sortie), [])

    def test_quart_de_nuit_garde_ses_heures(self):
        r = registre(("create_block", {"title": "Quart au dépanneur"}, True,
                      {"created": [cree("Quart au dépanneur", 3, "19:00", "02:00", 1)], "skipped": []}))
        self.assertEqual(faits(r), "Ajouté : Quart au dépanneur, les jeudis de 19 h à 2 h.")

    def test_exception_brute_jamais_montree(self):
        r = registre(("create_block", {"title": "Statistiques"}, False, {},
                      "Création annulée (aucun bloc créé): database is locked"))
        sortie = faits(r)
        self.assertIn("pas pu", sortie)
        self.assertIn("Statistiques", sortie)
        self.assertNotIn("database", sortie)

    def test_tentative_corrigee_dans_le_tour_se_tait(self):
        r = registre(("create_block", {"title": "Gym", "block_type": "shift"}, False, {},
                      "block_type invalide: 'shift'."),
                     ("create_block", {"title": "Gym", "block_type": "sport"}, True,
                      {"created": [cree("Gym", 1, "07:00", "08:00", 4)], "skipped": []}))
        sortie = faits(r)
        self.assertNotIn("pas pu", sortie)
        self.assertIn("Ajouté : Gym", sortie)


class AutresMutationsTests(SimpleTestCase):
    def test_mise_a_jour_dit_le_changement(self):
        bloc = {"id": 3, "title": "Gym", "day_of_week": 0, "day_name": "Lundi",
                "start_time": "18:00", "end_time": "19:00", "flexibility": "flexible"}
        avant = {"title": "Gym", "day_of_week": 0, "start_time": "07:00", "end_time": "08:00",
                 "flexibility": "flexible"}
        un = registre(("update_block", {"block_id": 3, "start_time": "18:00"}, True,
                       {"block": bloc, "avant": avant}))
        sortie = faits(un)
        for attendu in ("Gym", "18 h à 19 h", "lundis"):
            self.assertIn(attendu, sortie)
        self.assertIn("7 h à 8 h", sortie)
        trois = registre(*[("update_block", {"block_id": 3, "start_time": "18:00"}, True,
                            {"block": bloc, "avant": avant})] * 3)
        self.assertEqual(faits(trois).count("Gym"), 1)
        self.assertEqual(faits(trois).count("\n"), 0)

    def test_mise_a_jour_sans_avant(self):
        bloc = {"id": 3, "title": "Gym", "day_of_week": 0, "start_time": "18:00", "end_time": "19:00"}
        sortie = faits(registre(("update_block", {"block_id": 3}, True, {"block": bloc})))
        self.assertEqual(sortie, "Gym mis à jour : les lundis de 18 h à 19 h.")

    def test_action_du_code_rendue(self):
        r = registre(("skip_block_occurrence", {"date": "2026-09-17", "title": "Quart au dépanneur"},
                      True, {"date": "2026-09-17", "title": "Quart au dépanneur", "block_type": "work",
                             "cle_demande": "k", "par_le_code": True}))
        sortie = faits(r)
        self.assertIn("Quart au dépanneur", sortie)
        self.assertIn("jeu. 17 sept.", sortie)
        self.assertIn("les autres jeudis restent", sortie)
        self.assertEqual(marqueurs_bruts(sortie), [])

    def test_deja_fait_par_le_code_ne_se_repete_pas(self):
        donnees = {"block": {"id": 7, "title": "Quart au dépanneur", "day_of_week": 3,
                             "start_time": "19:00", "end_time": "02:00"},
                   "cle_demande": "k", "par_le_code": True}
        r = registre(("delete_block", {"block_id": 7}, True, donnees),
                     ("delete_block", {"block_id": 7}, True, {}, "Deja fait par le code ce tour."))
        self.assertEqual(faits(r), "Retiré de ton horaire : Quart au dépanneur, les jeudis de 19 h à 2 h.")

    def test_planification_et_annulation(self):
        r = registre(
            ("schedule_task_at", {"title": "Dentiste", "date": "2026-09-24"}, True,
             {"scheduled_block": {"id": 1, "title": "Rendez-vous chez le dentiste", "date": "2026-09-24",
                                  "start_time": "09:00", "end_time": "10:00", "overnight": False}}),
            ("cancel_scheduled_block", {"date": "2026-09-15", "title": "Étude"}, True,
             {"cancelled": [{"title": "Étude", "date": "2026-09-15", "start_time": "22:00", "end_time": "23:59"},
                            {"title": "Étude", "date": "2026-09-16", "start_time": "00:00", "end_time": "01:00"}]}),
        )
        sortie = faits(r)
        self.assertIn("Planifié : Rendez-vous chez le dentiste, le jeu. 24 sept. de 9 h à 10 h.", sortie)
        self.assertIn("Annulé : Étude, demain de 22 h à 1 h.", sortie)
        self.assertEqual(marqueurs_bruts(sortie), [])

    def test_conflit_de_planification(self):
        r = registre(("schedule_task_at", {"title": "Étude en groupe", "date": "2026-09-21",
                                           "start_time": "10:30", "end_time": "12:00"}, False,
                      {"conflict": {"start_time": "10:00", "end_time": "11:50",
                                    "titre": "Calcul différentiel", "sommeil": False}},
                      "Ce créneau (10:30-12:00) chevauche ... Choisis un autre horaire libre."))
        sortie = faits(r)
        self.assertIn("c'est pris par Calcul différentiel (10 h à 11 h 50)", sortie)
        self.assertNotIn("Choisis", sortie)
        self.assertEqual(marqueurs_bruts(sortie), [])

    def test_taches(self):
        r = registre(
            ("create_task", {"title": "Remettre le labo"}, True,
             {"task": {"id": 1, "title": "Remettre le labo", "priority": 8,
                       "deadline": "2026-09-18T12:00:00-04:00"}},
             "Tâche 'Remettre le labo' créée (priorité 8)."),
            ("create_task", {"title": "Lire"}, True,
             {"task": {"id": 2, "title": "Lire"}, "deja_presente": True}),
            ("complete_task", {"task_id": 3}, True, {"task": {"id": 3, "title": "Payer le loyer"}}),
            ("delete_task", {"task_id": 4, "confirm": True}, True, {"deleted_id": 4, "title": "Vieux devoir"}),
        )
        r.ajouter_ecart("a2", "tache deja presente, rien n'a ete cree", genre="tache_existante",
                        donnees={"titre": "Lire"})
        sortie = faits(r)
        self.assertIn("Ajouté à ta liste : Remettre le labo (pour ven. 18 sept.).", sortie)
        self.assertNotIn("priorité", sortie)
        self.assertEqual(sortie.count("Lire était déjà dans ta liste."), 1)
        self.assertIn("Coché : Payer le loyer.", sortie)
        self.assertIn("Retiré de ta liste : Vieux devoir.", sortie)
        self.assertEqual(marqueurs_bruts(sortie), [])

    def test_vider_le_planning(self):
        r = registre(("clear_all_blocks", {"confirm": True}, True, {"deleted_count": 23, "reversible": True}))
        self.assertEqual(faits(r), "Ton planning est vidé : 23 blocs archivés, tu peux les récupérer.")

    def test_organisation_proposee(self):
        r = registre(("organize_day", {"date": "2026-09-23"}, True,
                      {"applied": False, "date": "2026-09-23",
                       "placed": [{"title": "Entraînement de soccer", "start_time": "18:00", "end_time": "19:30"}],
                       "skipped": [], "moved": []}))
        r.ajouter_ecart("a1", "plan seulement propose, rien n'a ete applique", genre="plan_propose")
        sortie = faits(r)
        self.assertEqual(sortie.count("rien n'a changé"), 1)
        self.assertIn("Entraînement de soccer de 18 h à 19 h 30", sortie)
        self.assertNotIn("J'ai réorganisé", sortie)

    def test_semaine_optimisee_proposee(self):
        r = registre(("optimize_week", {}, True,
                      {"applied": False, "start_date": "2026-09-14", "moved_count": 0, "skipped_count": 0,
                       "days": [{"date": "2026-09-16",
                                 "placed": [{"title": "Sport", "start_time": "08:00", "end_time": "09:00"}],
                                 "overnight_kept": [], "skipped": [], "moved": []}]}))
        r.ajouter_ecart("a1", "plan seulement propose", genre="plan_propose")
        sortie = faits(r)
        self.assertEqual(sortie.count("rien n'a changé"), 1)
        self.assertIn("- mer. 16 sept. : Sport de 8 h à 9 h", sortie)
        self.assertEqual(marqueurs_bruts(sortie), [])

    def test_preferences(self):
        r = registre(("update_preferences", {}, True, {"updated_fields": ["min_sleep_hours"]}),
                     ("update_preferences", {}, True, {"updated_fields": []}))
        r.ajouter_ecart("a2", "aucune preference n'a change", genre="preferences_inchangees")
        sortie = faits(r)
        self.assertIn("Préférences mises à jour : heures de sommeil.", sortie)
        self.assertEqual(sortie.count("Aucune de tes préférences n'a changé."), 1)
        self.assertNotIn("min_sleep_hours", sortie)

    def test_echec_generique_nomme_le_titre(self):
        r = registre(("update_block", {"block_id": 9999}, False, {}, "Bloc #9999 introuvable."))
        sortie = faits(r)
        self.assertEqual(sortie, "Je n'ai pas pu modifier ce bloc, rien n'a changé.")
        self.assertEqual(marqueurs_bruts(sortie), [])
        # Pour une modification, `title` est le NOUVEAU nom: il ne designe pas la cible.
        renomme = registre(("update_block", {"block_id": 9999, "title": "X"}, False, {}, "introuvable"))
        self.assertEqual(faits(renomme), "Je n'ai pas pu modifier ce bloc, rien n'a changé.")
        cree_rate = registre(("schedule_task_at", {"title": "Gym", "start_time": "9h"}, False, {},
                              "Heure invalide (attendu HH:MM)."))
        self.assertEqual(faits(cree_rate), "Je n'ai pas pu planifier Gym, rien n'a changé.")

    def test_ancienne_garde_sans_demande(self):
        r = registre(("delete_block", {"block_id": 4}, False, {"needs_confirmation": True},
                      "Action destructrice: il faut une confirmation explicite. Demande-la, n'invente pas."))
        sortie = faits(r)
        self.assertIn("pas encore", sortie)
        self.assertNotIn("Demande-la", sortie)


class LecturesRateesEtRetenuesTests(SimpleTestCase):
    def test_lectures_ratees_invisibles(self):
        r = registre(*[(outil, {}, False, {}, "Format de date invalide.")
                       for outil in ("find_free_slots", "get_week_schedule", "list_blocks")])
        self.assertEqual(faits(r), "")
        self.assertEqual(lecture(r), "")

    def test_retenue_non_posee_dit_une_ligne(self):
        portee = {"type": "confirmation", "motif": "portee_jour", "cle": "p1", "outil": "delete_block",
                  "parametres": {"block_id": 7},
                  "cible": {"titre": "Quart au dépanneur", "jour": 3, "date": "2026-09-17"},
                  "options": [{"id": "occurrence"}, {"id": "serie"}, {"id": "annuler"}]}
        masse = {"type": "confirmation", "motif": "creation_en_masse", "cle": "c1",
                 "outil": "schedule_task_at", "parametres": {"title": "Lecture"},
                 "cible": {"titre": "Lecture", "nombre": 6, "deja": 5},
                 "options": [{"id": "confirmer"}, {"id": "annuler"}]}
        r = registre(("delete_block", {"block_id": 7}, False, {"needs_confirmation": True, "demande": portee}),
                     ("schedule_task_at", {"title": "Lecture"}, False, {"demande": masse}))
        sortie = faits(r, cles_posees={"p1"})
        self.assertIn("pas encore", sortie)
        self.assertIn("Lecture", sortie)
        self.assertNotIn("Quart", sortie)
        self.assertNotIn("jeudi", sortie)
        self.assertEqual(sortie.count("pas encore"), 1)
        muet = faits(r, cles_posees=None)
        self.assertNotIn("pas encore", muet)
        self.assertNotIn("Lecture", muet)

    def test_retenues_de_meme_cle_une_seule_ligne(self):
        destructif = {"motif": "destructif", "cle": "d1", "outil": "clear_all_blocks",
                      "cible": {}, "options": [{"id": "confirmer"}, {"id": "annuler"}]}
        r = registre(("clear_all_blocks", {"confirm": True}, False, {"demande": destructif}),
                     ("clear_all_blocks", {"confirm": True}, False, {"demande": destructif}))
        self.assertEqual(faits(r, cles_posees=set()),
                         "Je n'ai pas encore vidé ton planning : redemande-le-moi après ta réponse.")

    def test_heure_refusee_en_ligne_de_fait(self):
        demande = {"type": "choix", "motif": "heure_refusee", "cle": "h1", "outil": "schedule_task_at",
                   "cible": {"titre": "Dentiste", "date": "2026-09-17", "debut": "10:30", "fin": "11:30"},
                   "options": [{"id": "creneau_1", "cible": {"date": "2026-09-17", "debut": "11:50", "fin": "12:50"}},
                               {"id": "autre_jour"}]}
        r = registre(
            ("schedule_task_at", {"title": "Dentiste", "date": "2026-09-17", "start_time": "10:30",
                                  "end_time": "11:30"}, False,
             {"conflict": {"start_time": "10:00", "end_time": "11:50", "titre": "Calcul différentiel",
                           "sommeil": False}, "demande": demande}),
            ("schedule_task_at", {"title": "RDV dentiste", "date": "2026-09-17", "start_time": "12:00",
                                  "end_time": "13:00"}, False, {"demande": demande}),
        )
        sortie = faits(r)
        self.assertEqual(sortie, "10 h 30 jeu. 17 sept., c'est pris par Calcul différentiel.")

    def test_heure_refusee_create_block_garde_les_jours_crees(self):
        demande = {"motif": "heure_refusee", "cle": "h2", "outil": "create_block",
                   "cible": {"titre": "Statistiques", "jour": 0, "date": "2026-09-21", "debut": "10:00"}}
        r = registre(("create_block", {"title": "Statistiques", "start_time": "10:00", "end_time": "12:00"},
                      True,
                      {"created": [cree("Statistiques", 4, "10:00", "12:00", 5)],
                       "skipped": [{"day": 0, "motif": "chevauchement", "titre": "Statistiques",
                                    "debut": "10:00", "fin": "12:00",
                                    "avec": {"titre": "Calcul différentiel", "debut": "10:00", "fin": "11:50"}}],
                       "demande": demande}))
        sortie = faits(r)
        self.assertIn("Ajouté : Statistiques, les vendredis de 10 h à 12 h.", sortie)
        self.assertIn("10 h lun. 21 sept., c'est pris par Calcul différentiel.", sortie)
        self.assertEqual(sortie.count("Calcul différentiel"), 1)

    def test_sommeil_reporte_dit_simplement(self):
        demande = {"motif": "heure_refusee", "cle": "h3", "cible": {"date": "2026-09-15", "debut": "09:00"}}
        r = registre(("schedule_task_at", {"title": "Lecture"}, False,
                      {"conflict": {"start_time": "02:00", "end_time": "10:00", "titre": None, "sommeil": True},
                       "demande": demande}))
        self.assertEqual(faits(r), "9 h demain, c'est pendant ton sommeil.")


class EcartsEtInterruptionsTests(SimpleTestCase):
    def test_ecart_passe_humain(self):
        r = registre(("schedule_task_at", {"title": "Révision"}, True,
                      {"scheduled_block": {"title": "Révision", "date": "2026-09-13",
                                           "start_time": "14:00", "end_time": "15:00"}}))
        r.ajouter_ecart("a1", "CREE mais dans le passe (2026-09-13 15:00): il existe", genre="passe",
                        donnees={"date": "2026-09-13", "fin": "15:00", "titre": "Révision"})
        sortie = faits(r)
        self.assertIn("hier", sortie)
        self.assertIn("déjà passé", sortie)
        for interdit in ("CREE", "Ecart", "2026-"):
            self.assertNotIn(interdit, sortie)
        self.assertEqual(marqueurs_bruts(sortie), [])

    def test_ecarts_par_genre(self):
        r = registre(("schedule_task_at", {}, True, {}), ("restore_block_occurrence", {}, True,
                                                         {"restored": False, "title": "Gym", "date": "2026-09-16"}))
        r.ajouter_ecart("a1", "date demandee x", genre="date_differente",
                        donnees={"demandee": "2026-09-18", "obtenue": "2026-09-19", "titre": "Lecture"})
        r.ajouter_ecart("a2", "aucune occurrence", genre="rien_a_restaurer",
                        donnees={"titre": "Gym", "date": "2026-09-16"})
        r.ajouter_ecart("a1", "ancien sans genre")
        sortie = faits(r)
        self.assertIn("Lecture : placé le sam. 19 sept., pas le ven. 18 sept. comme demandé.", sortie)
        self.assertEqual(sortie.count("rien à remettre"), 1)
        self.assertNotIn("ancien", sortie)
        self.assertEqual(marqueurs_bruts(sortie), [])

    def test_budget_epuise_accentue(self):
        r = Registre()
        r.budget_epuise = True
        sortie = faits(r)
        self.assertIn("arrêté", sortie)
        self.assertNotIn("etapes", sortie)

    def test_boucle_interrompue(self):
        r = Registre()
        r.boucle_interrompue = True
        self.assertEqual(faits(r), "Je me suis arrêté : je répétais la même étape.")

    def test_rien_ne_rend_rien(self):
        self.assertEqual(faits(Registre()), "")
        self.assertEqual(lecture(Registre()), "")


class ImportTests(SimpleTestCase):
    DONNEES = {
        "fichier": "horaire_automne_2026.pdf",
        "blocs": [
            {"id": 1, "titre": "Calcul différentiel", "jour": "lundi", "debut": "10:00", "fin": "11:50"},
            {"id": 2, "titre": "Calcul différentiel", "jour": "mercredi", "debut": "10:00", "fin": "11:50"},
        ],
        "en_attente": 1,
        "dates": [{"titre": "Examen intra de chimie", "date": "2026-10-15", "debut": "13:00", "fin": "15:00"}],
        "ignores": [{"titre": "Statistiques", "jour": "lundi", "debut": "10:00", "fin": "12:00"}],
    }

    def test_recap_import(self):
        r = registre(("import_document", {"document": "horaire_automne_2026.pdf"}, True, self.DONNEES,
                      "Horaire importé depuis « horaire_automne_2026.pdf » : 3 entrées ajoutées"))
        sortie = faits(r)
        for attendu in ("C'est importé", "Calcul différentiel · lun. et mer., 10 h à 11 h 50",
                        "jeu. 15 oct.", "À vérifier", "Statistiques"):
            self.assertIn(attendu, sortie)
        for interdit in ("horaire_automne_2026.pdf", "  - ", "2026-10-15", EM_DASH):
            self.assertNotIn(interdit, sortie)
        self.assertTrue(sortie.startswith("C'est importé : 1 cours et 1 examen."))
        self.assertEqual(marqueurs_bruts(sortie), [])

    def test_quarts_importes(self):
        donnees = {"fichier": "x.pdf", "blocs": [
            {"titre": "Quart au dépanneur", "type": "work", "jour": "mardi", "debut": "16:00", "fin": "22:00"},
            {"titre": "Chimie générale", "type": "course", "jour": "jeudi", "debut": "13:00", "fin": "15:50"}],
            "en_attente": 0, "dates": [], "ignores": []}
        sortie = faits(registre(("import_document", {}, True, donnees)))
        self.assertTrue(sortie.startswith("C'est importé : 1 quart et 1 cours."))
        self.assertNotIn("À vérifier", sortie)

    def test_import_recent_silencieux(self):
        r = registre(("import_recent", {"document": "x.pdf"}, True, self.DONNEES, "Horaire importé..."))
        self.assertEqual(faits(r), "")
        self.assertEqual(lecture(r), "")


def semaine(avec_detail=True):
    jours_data = []
    for i in range(7):
        detail = []
        if i == 1:
            detail = [
                {"title": "Philosophie", "start_time": "08:00", "end_time": "09:50",
                 "block_type": "course", "is_flexible": False},
                {"title": "Biologie", "start_time": "10:00", "end_time": "11:30",
                 "block_type": "course", "is_flexible": False},
                {"title": "Quart au dépanneur", "start_time": "16:00", "end_time": "22:00",
                 "block_type": "work", "is_flexible": False},
            ]
        if i == 0:
            detail = [{"title": "Calcul différentiel", "start_time": "10:00", "end_time": "11:50",
                       "block_type": "course", "is_flexible": False}]
        detail.append({"title": "Sommeil", "start_time": "23:00", "end_time": "07:00",
                       "block_type": "sleep", "is_flexible": True})
        jour_data = {"date": f"2026-09-{14 + i}", "day_name": rendu.JOURS[i].capitalize(),
                     "block_count": len(detail), "occupied_hours": 0,
                     "blocks": [f"{b['title']} ({b['start_time']}-{b['end_time']})" for b in detail]}
        if avec_detail:
            jour_data["detail"] = detail
        jours_data.append(jour_data)
    return {"week_start": "2026-09-14", "days": jours_data, "total_hours": 93.7}


class LecturesTests(SimpleTestCase):
    def test_semaine_lisible(self):
        sortie = lecture(registre(("get_week_schedule", {}, True, semaine())))
        premiere = sortie.split("\n")[0]
        self.assertTrue(premiere.endswith("."))
        self.assertFalse(premiere.startswith(("-", "**")))
        self.assertIn("**Mardi**", sortie)
        self.assertIn("8 h à 9 h 50 · Philosophie", sortie)
        self.assertLessEqual(sortie.count("Sommeil"), 1)
        self.assertIn("Sommeil : 23 h à 7 h", sortie)
        self.assertNotIn("Mardi :", sortie)
        self.assertIn("Rien de prévu mercredi, jeudi, vendredi, samedi et dimanche.", sortie)
        self.assertIn("la journée la plus chargée est mardi", sortie)
        self.assertEqual(marqueurs_bruts(sortie), [])

    def test_semaine_sans_detail_lit_les_chaines(self):
        sortie = lecture(registre(("get_week_schedule", {}, True, semaine(avec_detail=False))))
        self.assertIn("**Mardi**\n- 8 h à 9 h 50 · Philosophie", sortie)
        self.assertLessEqual(sortie.count("Sommeil"), 1)
        self.assertEqual(marqueurs_bruts(sortie), [])

    def test_liste_jamais_collee_a_l_entete_suivant(self):
        """Sans ligne vide, markdown avale l'en-tete suivant dans le dernier point."""
        sortie = lecture(registre(("get_week_schedule", {}, True, semaine())))
        self.assertIn("· Calcul différentiel\n\n**Mardi**", sortie)

    def test_journee_avec_libre(self):
        donnees = {"day_name": "Jeudi",
                   "blocks": [{"id": 1, "title": "Calcul", "type": "recurring", "block_type": "course",
                               "start_time": "10:00", "end_time": "11:50"},
                              {"id": 2, "title": "Sommeil", "type": "recurring", "block_type": "sleep",
                               "start_time": "23:00", "end_time": "07:00"}],
                   "free_slots": [{"start_time": "07:00", "end_time": "10:00", "duration_minutes": 180},
                                  {"start_time": "11:50", "end_time": "17:00", "duration_minutes": 310}]}
        sortie = lecture(registre(("get_today_schedule", {}, True, donnees)))
        self.assertIn("Libre : 7 h à 10 h, 11 h 50 à 17 h", sortie)
        self.assertEqual(sortie.count("Jeudi"), 1)
        self.assertNotIn("Sommeil", sortie)
        self.assertEqual(marqueurs_bruts(sortie), [])

    def test_journee_datee_demain(self):
        donnees = {"date": "2026-09-15", "day_name": "Mardi", "blocks": [], "free_slots": []}
        sortie = lecture(registre(("get_today_schedule", {}, True, donnees)))
        self.assertTrue(sortie.startswith("**Mardi** (demain)\nRien de prévu."))

    def test_creneaux_et_taches_rendus(self):
        creneaux = {"date": "2026-09-17", "day_name": "Jeudi",
                    "free_slots": [{"start_time": "07:00", "end_time": "08:30", "duration_minutes": 90}]}
        sortie = lecture(registre(("find_free_slots", {"date": "2026-09-17"}, True, creneaux)))
        self.assertIn("Libre", sortie)
        self.assertEqual(sortie, "Libre jeu. 17 sept. : 7 h à 8 h 30")
        taches = {"tasks": [{"id": 1, "title": "Remettre le labo", "priority": 8,
                             "deadline": "2026-09-15", "completed": False}], "count": 1}
        sortie = lecture(registre(("list_tasks", {}, True, taches)))
        self.assertIn("demain", sortie)
        self.assertNotIn("8)", sortie)
        self.assertNotIn("priorité", sortie)
        self.assertEqual(marqueurs_bruts(sortie), [])

    def test_aucun_creneau(self):
        sortie = lecture(registre(("find_free_slots", {}, True,
                                   {"date": "2026-09-15", "free_slots": []})))
        self.assertEqual(sortie, "Aucun créneau libre demain.")

    def test_horaire_liste(self):
        blocs = [
            {"id": 1, "title": "Anglais", "block_type": "course", "day_of_week": 0, "day_name": "Lundi",
             "start_time": "08:00", "end_time": "09:30"},
            {"id": 2, "title": "Soccer", "block_type": "sport", "day_of_week": 5, "day_name": "Samedi",
             "start_time": "18:00", "end_time": "19:30"},
        ]
        sortie = lecture(registre(("list_blocks", {}, True, {"blocks": blocs, "count": 2})))
        self.assertTrue(sortie.startswith("Ton horaire compte 2 blocs"))
        self.assertIn("**Samedi**\n- 18 h à 19 h 30 · Soccer", sortie)
        self.assertNotIn("Rien de prévu", sortie)
        self.assertEqual(marqueurs_bruts(sortie), [])

    def test_lecture_tue_par_une_mutation(self):
        r = registre(("get_week_schedule", {}, True, semaine()),
                     ("create_block", {"title": "Yoga"}, True,
                      {"created": [cree("Yoga", 6, "09:00", "10:00", 1)], "skipped": []}))
        self.assertEqual(lecture(r), "")

    def test_la_derniere_lecture_fait_foi(self):
        creneaux = {"date": "2026-09-17", "free_slots": [
            {"start_time": "07:00", "end_time": "08:30", "duration_minutes": 90}]}
        r = registre(("get_week_schedule", {}, True, semaine()),
                     ("find_free_slots", {}, True, creneaux))
        self.assertTrue(lecture(r).startswith("Libre"))


def portee(titre, cle, jour_semaine=3, iso="2026-09-17"):
    return {"type": "confirmation", "motif": "portee_jour", "cle": cle, "outil": "delete_block",
            "parametres": {"block_id": 1},
            "cible": {"titre": titre, "jour": jour_semaine, "date": iso},
            "options": [{"id": "occurrence", "effet": {"outil": "skip_block_occurrence", "parametres": {}}},
                        {"id": "serie", "effet": {"outil": "delete_block", "parametres": {}}},
                        {"id": "annuler", "effet": None}],
            "emise_le": "2026-09-14T12:00:00+00:00"}


class DemandesTests(SimpleTestCase):
    def test_demande_portee_jour(self):
        question, chips, cles = rendre_demandes([portee("Quart au dépanneur", "p1")], AUJ)
        self.assertTrue(question.endswith("?"))
        self.assertIn("jeudi", question)
        self.assertIn("Quart au dépanneur", question)
        self.assertEqual([c["label"] for c in chips], ["Seulement ce jeudi", "Tous les jeudis", "Non, garde tout"])
        self.assertEqual([c["option"] for c in chips], ["occurrence", "serie", "annuler"])
        self.assertEqual(len({c["value"] for c in chips}), 3)
        self.assertEqual(chips[0]["value"], "Seulement ce jeudi 17 sept. (sauter l'occurrence).")
        self.assertEqual(chips[1]["value"], "Tous les jeudis (supprimer la série).")
        self.assertEqual(chips[2]["value"], "Non, ne change rien.")
        self.assertEqual(cles, ["p1"])

    def test_demandes_fusionnees(self):
        demandes = [portee("Chimie générale", "p1"), portee("Quart au dépanneur", "p2"),
                    portee("Sommeil", "p3"), portee("Sommeil", "p3")]
        question, chips, cles = rendre_demandes(demandes, AUJ)
        for titre in ("Chimie générale", "Quart au dépanneur", "Sommeil"):
            self.assertIn(titre, question)
        self.assertEqual(question.count("?"), 1)
        self.assertEqual(len(chips), 3)
        self.assertEqual(cles, ["p1", "p2", "p3"])

    def test_portee_sur_deux_jours_differents(self):
        question, chips, _ = rendre_demandes(
            [portee("Gym", "p1", 0, "2026-09-21"), portee("Quart", "p2", 3, "2026-09-17")], AUJ)
        self.assertEqual([c["label"] for c in chips], ["Seulement cette fois", "Toute la série", "Non, garde tout"])
        self.assertTrue(question.endswith("?"))

    def test_seul_le_motif_prioritaire_est_pose(self):
        masse = {"motif": "creation_en_masse", "cle": "c1", "outil": "schedule_task_at",
                 "cible": {"titre": "Lecture", "nombre": 6, "deja": 5},
                 "options": [{"id": "confirmer"}, {"id": "annuler"}]}
        question, chips, cles = rendre_demandes([masse, portee("Quart au dépanneur", "p1")], AUJ)
        self.assertEqual(cles, ["p1"])
        self.assertNotIn("Lecture", question)
        self.assertNotIn("continue", question)
        self.assertEqual([c["option"] for c in chips], ["occurrence", "serie", "annuler"])

    def test_demande_destructive(self):
        demande = {"motif": "destructif", "cle": "d1", "outil": "delete_block",
                   "cible": {"titre": "Chimie", "jour": 0, "debut": "13:00", "fin": "15:50"},
                   "options": [{"id": "confirmer"}, {"id": "annuler"}]}
        vider = {"motif": "destructif", "cle": "d2", "outil": "clear_all_blocks", "cible": {},
                 "options": [{"id": "confirmer"}, {"id": "annuler"}]}
        question, chips, cles = rendre_demandes([demande, vider], AUJ)
        self.assertEqual(question,
                         "Tu veux vraiment supprimer Chimie (les lundis de 13 h à 15 h 50) et vider tout ton planning ?")
        self.assertEqual([(c["label"], c["value"], c["option"]) for c in chips],
                         [("Oui, confirme", "Oui, je confirme.", "confirmer"),
                          ("Non, garde tout", "Non, ne change rien.", "annuler")])
        self.assertEqual(cles, ["d1", "d2"])
        self.assertEqual(marqueurs_bruts(question), [])

    def test_demande_heure_refusee(self):
        demande = {"type": "choix", "motif": "heure_refusee", "cle": "h1", "outil": "schedule_task_at",
                   "cible": {"titre": "Dentiste", "date": "2026-09-17", "debut": "10:30"},
                   "options": [
                       {"id": "creneau_1", "effet": None, "cible": {"debut": "11:50", "fin": "12:50", "date": "2026-09-17"}},
                       {"id": "creneau_2", "effet": None, "cible": {"debut": "15:50", "fin": "16:50", "date": "2026-09-17"}},
                       {"id": "autre_jour", "effet": None}]}
        question, chips, cles = rendre_demandes([demande], AUJ)
        self.assertEqual([c["label"] for c in chips], ["11 h 50 à 12 h 50", "15 h 50 à 16 h 50", "Un autre jour"])
        self.assertEqual(chips[0]["value"], "Va pour 11 h 50 à 12 h 50 jeu. 17 sept.")
        self.assertEqual(chips[2]["value"], "Je préfère un autre jour.")
        self.assertEqual([c["option"] for c in chips], ["creneau_1", "creneau_2", "autre_jour"])
        self.assertTrue(question.endswith("?"))
        self.assertIn("Dentiste", question)
        self.assertEqual(cles, ["h1"])

    def test_demande_choix_modele(self):
        demande = {"type": "choix", "motif": "choix_modele", "cle": "choix:abc", "outil": "present_choices",
                   "question": "Lequel de tes cours ?", "source": "blocs",
                   "options": [{"id": "o1", "libelle": "Calcul différentiel", "valeur": "Le cours de Calcul différentiel", "effet": None},
                               {"id": "o2", "libelle": "Physique mécanique", "valeur": "Le cours de Physique mécanique", "effet": None}]}
        autre = dict(demande, cle="choix:def", question="Quel jour ?")
        question, chips, cles = rendre_demandes([demande, autre], AUJ)
        self.assertEqual(question, "Lequel de tes cours ?")
        self.assertEqual(chips, [
            {"label": "Calcul différentiel", "value": "Le cours de Calcul différentiel", "option": "o1"},
            {"label": "Physique mécanique", "value": "Le cours de Physique mécanique", "option": "o2"}])
        self.assertEqual(cles, ["choix:abc"])

    def test_creation_en_masse_et_optimisation(self):
        masse = {"motif": "creation_en_masse", "cle": "creation_en_masse",
                 "cible": {"titres": ["Lecture", "Yoga"], "nombre": 7, "deja": 5},
                 "options": [{"id": "confirmer"}, {"id": "annuler"}]}
        question, chips, _ = rendre_demandes([masse], AUJ)
        self.assertEqual(question, "Ça fait déjà 5 ajouts d'un coup. Je continue avec Lecture et Yoga ?")
        self.assertEqual([(c["label"], c["value"]) for c in chips],
                         [("Oui, continue", "Oui, continue les ajouts."), ("Non, arrête là", "Non, arrête là.")])
        optim = {"motif": "optimisation", "cle": "optimize_week:apply", "outil": "optimize_week",
                 "options": [{"id": "confirmer"}, {"id": "annuler"}]}
        question, chips, cles = rendre_demandes([optim], AUJ)
        self.assertTrue(question.endswith("?"))
        self.assertEqual([(c["label"], c["value"]) for c in chips],
                         [("Applique le plan", "Oui, applique le plan."),
                          ("Montre d'abord", "Montre-moi d'abord la proposition.")])
        self.assertEqual(cles, ["optimize_week:apply"])

    def test_chevauchement(self):
        demande = {"motif": "chevauchement", "cle": "x1", "cible": {"titre": "Statistiques", "jour": 0},
                   "options": [{"id": "autre_heure"}, {"id": "annuler"}]}
        question, chips, _ = rendre_demandes([demande], AUJ)
        self.assertTrue(question.endswith("?"))
        self.assertEqual([(c["label"], c["value"]) for c in chips],
                         [("Trouve une autre heure", "Trouve une autre heure pour Statistiques."),
                          ("Laisse faire", "Non, laisse faire.")])

    def test_rien_a_demander(self):
        self.assertEqual(rendre_demandes([], AUJ), ("", [], []))
        self.assertEqual(rendre_demandes([{"motif": "inconnu"}, "x"], AUJ), ("", [], []))


class AucunTiretLongTests(SimpleTestCase):
    def test_le_module_n_a_aucun_tiret_long(self):
        import inspect
        self.assertNotIn(EM_DASH, inspect.getsource(rendu))
