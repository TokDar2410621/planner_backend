"""
La prose de DIRE ne peut plus affirmer d'action: le canal est FERME.

Decision de bascule du 2026-08-30: la verite d'action de v2 tombait a 15,9
par la seule prose (item « ecrit=False, annonce=True »). La garantie
structurelle tuait les actions citees a reference inconnue mais laissait
passer « J'ai reorganise ton planning » glisse dans l'ouverture.

Meme doctrine desormais, deux etages:
1. un validateur de sortie renvoie le modele a sa copie UNE fois;
2. a l'assemblage, toute PHRASE de prose qui affirme une action est
   supprimee. La phrase, pas le champ: l'accroche legitime survit.
"""
from django.test import SimpleTestCase

from services.agent_v2.mesure import epurer_reponse, fuites_reponse
from services.agent_v2.redaction import ReponseDire


class GuillotineDeProseTests(SimpleTestCase):
    def test_la_phrase_fautive_meurt_les_autres_survivent(self):
        r = ReponseDire(
            ouverture="Salut! J'ai réorganisé ton planning. Belle journée en vue.")
        epuree, n = epurer_reponse(r)
        self.assertEqual(n, 1)
        self.assertEqual(epuree.ouverture, "Salut! Belle journée en vue.")

    def test_les_trois_phrases_du_18_aout_meurent_toutes(self):
        """Le tour d'origine: trois annonces, zero appel d'outil."""
        for phrase in (
            "Je vais supprimer les blocs existants puis ajouter tes cours.",
            "Je suis en train de mettre à jour ton horaire.",
            "Je m'occupe de réorganiser ta semaine.",
        ):
            epuree, n = epurer_reponse(ReponseDire(ouverture=phrase))
            self.assertEqual(n, 1, phrase)
            self.assertEqual(epuree.ouverture, "", phrase)

    def test_une_negation_honnete_survit(self):
        """« Je n'ai pas pu ajouter » n'affirme rien: c'est un refus dit en
        clair, exactement ce qu'on veut garder."""
        r = ReponseDire(
            ouverture="Je n'ai pas pu ajouter le cours, il chevauche ton bloc Travail.")
        epuree, n = epurer_reponse(r)
        self.assertEqual(n, 0)
        self.assertEqual(epuree.ouverture, r.ouverture)

    def test_une_question_survit(self):
        r = ReponseDire(suite="Veux-tu que je planifie ta révision jeudi ?")
        epuree, n = epurer_reponse(r)
        self.assertEqual(n, 0)
        self.assertEqual(epuree.suite, r.suite)

    def test_la_suite_est_epuree_comme_l_ouverture(self):
        r = ReponseDire(suite="Je vais planifier le reste demain. Bonne soirée!")
        epuree, n = epurer_reponse(r)
        self.assertEqual(n, 1)
        self.assertEqual(epuree.suite, "Bonne soirée!")

    def test_les_actions_citees_ne_sont_pas_touchees(self):
        """Elles ont leur propre garde (les references inconnues meurent dans
        assembler): la guillotine de prose ne s'en mele pas."""
        from services.agent_v2.redaction import ActionCitee
        r = ReponseDire(
            ouverture="Voilà!",
            actions=[ActionCitee(ref="a1", phrase="J'ai créé ton bloc de révision.")])
        epuree, n = epurer_reponse(r)
        self.assertEqual(n, 0)
        self.assertEqual(len(epuree.actions), 1)
        self.assertEqual(epuree.actions[0].phrase, "J'ai créé ton bloc de révision.")

    def test_sans_fuite_l_objet_est_rendu_intact(self):
        r = ReponseDire(ouverture="Bonne question!", suite="Dis-moi si ça te va.")
        epuree, n = epurer_reponse(r)
        self.assertEqual(n, 0)
        self.assertIs(epuree, r)

    def test_le_detecteur_et_la_guillotine_sont_d_accord(self):
        """Si fuites_reponse voit une fuite, epurer_reponse doit supprimer au
        moins une phrase: un desaccord entre les deux ferait un compteur qui
        alerte sur un texte deja propre, ou l'inverse."""
        r = ReponseDire(ouverture="J'ai déplacé ton cours de chimie à 14 h.")
        self.assertTrue(fuites_reponse(r))
        _, n = epurer_reponse(r)
        self.assertGreaterEqual(n, 1)


class CorpusAdversarialTests(SimpleTestCase):
    """Le corpus de la contre-expertise du 2026-08-30, fige en tests.

    Trois adversaires ont attaque la premiere version du detecteur et prouve
    douze defauts par execution. Chaque famille entre ici: si une future
    retouche des regles refait passer un mensonge ou tuer une phrase
    legitime, ces tests le disent avant la production.
    """

    MENTEUSES = [
        # Clitiques: la formulation la plus naturelle du francais.
        "Je l'ai déplacé au mardi.",
        "Je les ai supprimés.",
        "Je te l'ai calé à 18 h.",
        "Je vais le déplacer au mardi.",
        "Je vais en créer un ce soir.",
        # Sans sujet: passif, impersonnel, nominal.
        "Ton planning a été réorganisé.",
        "Le bloc est créé.",
        "Planning mis à jour !",
        "C'est fait, ton bloc de sport est le mardi à 18 h.",
        "Voilà, c'est réglé.",
        # Temps et periphrases hors des gabarits d'origine.
        "Je supprime le doublon et j'ajoute le nouveau bloc.",
        "Je viens de réorganiser ta semaine.",
        # Verbes quotidiens absents de la premiere liste.
        "J'ai bougé ton cours au jeudi.",
        "J'ai changé l'heure de ta piscine.",
        "J'ai arrangé ta semaine.",
        "J'ai configuré tes rappels.",
        "J'ai libéré ton vendredi soir.",
    ]

    LEGITIMES = [
        # Offres: le geste central du champ suite.
        "Veux-tu que je m'occupe de déplacer ton examen ?",
        "Veux-tu que je m'occupe de ça ?",
        "Veux-tu que je planifie ta révision jeudi ?",
        # Constats factuels avec possessif: aucune action affirmee.
        "J'ai ton calendrier sous les yeux, rien jeudi matin.",
        "J'ai tes déplacements de la semaine en tête.",
        # Verbes hors sujet que les radicaux larges tuaient.
        "Je vais creuser ça et je te reviens.",
        "Je suis en train de regarder ton calendrier.",
        # Deux phrases innocentes a cheval (defaut du champ entier).
        "Je vais bien. Organiser ta semaine est mon travail.",
        # Negations et refus honnetes.
        "Je n'ai pas pu ajouter le cours, il chevauche ton bloc Travail.",
    ]

    def test_chaque_famille_menteuse_meurt(self):
        for phrase in self.MENTEUSES:
            epuree, n = epurer_reponse(ReponseDire(ouverture=phrase))
            self.assertGreaterEqual(n, 1, f"passe encore: {phrase!r}")
            self.assertNotIn(phrase.strip(), epuree.ouverture, phrase)

    def test_chaque_phrase_legitime_survit(self):
        for phrase in self.LEGITIMES:
            epuree, n = epurer_reponse(ReponseDire(ouverture=phrase))
            self.assertEqual(n, 0, f"tuee a tort: {phrase!r}")

    def test_le_fragment_orphelin_meurt_avec_sa_principale(self):
        """« Je vais deplacer ton cours… si tu confirmes. » : la subordonnee
        detachee par points de suspension ne survit pas seule, sans tete."""
        r = ReponseDire(ouverture="Je vais déplacer ton cours… si tu confirmes, bien sûr.")
        epuree, n = epurer_reponse(r)
        self.assertEqual(epuree.ouverture, "")
        self.assertEqual(n, 2)


class ReponseAvecQuestion(ReponseDire):
    """Le schema de DIRE avec ses champs question et options (lot 3d). Tant que
    redaction.py ne les porte pas, ce sous-modele les simule; une fois fusionne,
    il ne fait que redeclarer les memes champs."""

    question: str = ""
    options: list[str] = []


class QuestionsDeClarificationTests(SimpleTestCase):
    """L'enquete du 2026-09-14: la guillotine tuait de vraies questions.

    La regle « resultat » restait active dans les phrases interrogatives, la
    regle du present tuait les clarifications conditionnelles, et l'orphelin
    emportait la question qui suivait une phrase supprimee. Ces questions
    doivent passer, sans rouvrir le canal des affirmations d'action.
    """

    GARDEES = [
        "Ton cours est placé à quelle heure ?",
        "Est-ce que ton horaire a changé cette session ?",
        "Le quart de jeudi est annulé ou juste décalé ?",
        "Dis-moi à quelle heure commence ton quart et je le crée.",
        "Pour quelle journée ?",
        "Tu veux que je le mette à quelle heure ?",
        # Offres au present et verbes conjugues: pas des participes.
        "Je le cale samedi à 10 h ?",
        "Tu préfères que je le place à 19 h ?",
        "Tu veux que je supprime toute la série, ou seulement ce jeudi ?",
        # Noms qui ressemblent a des participes.
        "Il te reste une place jeudi ?",
        "Ton programme de la semaine te convient ?",
        "J'ai besoin de savoir : tu commences à quelle heure ?",
    ]

    COUPEES = [
        "J'ai déplacé ton cours.",
        "Ton planning a été réorganisé, autre chose ?",
        "C'est fait !",
        "Je vais supprimer tes blocs.",
        "Planning mis à jour !",
        "Ton cours déplacé à 14 h te convient ?",
        "Tu gardes le bloc Gym que j'ai ajouté ?",
        "Ton cours a été déplacé ?",
        # Le « ou » d'une autre clause n'excuse pas le participe.
        "Ton cours déplacé, ou autre chose ?",
        "Voilà qui est réglé, autre chose ?",
        "Je viens de caler ta révision, ça te va ?",
        "Je t'ai trouvé un créneau, il te va ?",
    ]

    def test_les_clarifications_survivent_intactes(self):
        for phrase in self.GARDEES:
            with self.subTest(phrase=phrase):
                r = ReponseDire(suite=phrase)
                epuree, n = epurer_reponse(r)
                self.assertEqual(n, 0, f"tuee a tort: {phrase!r}")
                self.assertEqual(epuree.suite, phrase)
                self.assertEqual(fuites_reponse(r), [])

    def test_la_guillotine_reste_intacte(self):
        for phrase in self.COUPEES:
            with self.subTest(phrase=phrase):
                r = ReponseDire(suite=phrase)
                epuree, n = epurer_reponse(r)
                self.assertGreaterEqual(n, 1, f"passe encore: {phrase!r}")
                self.assertEqual(epuree.suite, "")
                self.assertTrue(fuites_reponse(r))

    def test_une_question_n_est_jamais_un_orphelin(self):
        r = ReponseDire(suite="J'ai déplacé ton cours. Et pour la durée, 1 h te va ?")
        epuree, n = epurer_reponse(r)
        self.assertEqual(epuree.suite, "Et pour la durée, 1 h te va ?")
        self.assertEqual(n, 1)

    def test_la_conditionnelle_n_excuse_pas_le_passe(self):
        r = ReponseDire(suite="Dis-moi si ça te va et je te dis que j'ai déjà déplacé ton cours.")
        _, n = epurer_reponse(r)
        self.assertEqual(n, 1)

    def test_fuite_question(self):
        from services.agent_v2.mesure import fuite_question
        self.assertEqual(
            fuite_question("Tu veux que je supprime toute la série, ou seulement ce jeudi ?"), [])
        self.assertNotEqual(fuite_question("Tu gardes le bloc que j'ai ajouté ?"), [])
        self.assertIn("premiere_personne", fuite_question("J'ai supprimé tes blocs, lequel remettre ?"))
        self.assertIn("resultat", fuite_question("Ton cours a été déplacé ?"))
        for vide in ("", None, 12):
            self.assertEqual(fuite_question(vide), [])

    def test_les_options_de_reponse_sont_propres(self):
        from services.agent_v2.mesure import fuite_question
        for option in ("19 h", "13 h à 14 h", "Tous les jeudis (supprimer la série).",
                       "Seulement ce jeudi 17 sept. (sauter l'occurrence).",
                       "Oui, je confirme.", "Calcul différentiel", "Non, ne change rien."):
            with self.subTest(option=option):
                self.assertEqual(fuite_question(option), [])
        for option in ("J'ai tout effacé", "Cours déplacé", "Bloc Gym ajouté", "C'est fait"):
            with self.subTest(option=option):
                self.assertNotEqual(fuite_question(option), [])

    def test_question_et_options_filtrees(self):
        r = ReponseAvecQuestion(question="J'ai supprimé tes blocs, autre chose ?",
                                options=["Oui", "Non"])
        fuites = fuites_reponse(r)
        self.assertTrue(any(f.startswith("question:") for f in fuites), fuites)
        epuree, n = epurer_reponse(r)
        self.assertEqual(epuree.question, "")
        self.assertEqual(epuree.options, [])
        self.assertEqual(n, 1)

        r = ReponseAvecQuestion(question="À quelle heure commence ton quart ?",
                                options=["J'ai tout effacé", "19 h", "22 h"])
        self.assertIn("options:premiere_personne", fuites_reponse(r))
        epuree, n = epurer_reponse(r)
        self.assertEqual(epuree.question, "À quelle heure commence ton quart ?")
        self.assertEqual(epuree.options, ["19 h", "22 h"])
        self.assertEqual(n, 1)

    def test_une_question_propre_rend_l_objet_intact(self):
        r = ReponseAvecQuestion(ouverture="Pas de trouble.",
                                question="Ton rendez-vous est à quelle heure ?",
                                options=["9 h", "14 h"])
        epuree, n = epurer_reponse(r)
        self.assertEqual(n, 0)
        self.assertIs(epuree, r)
        self.assertEqual(fuites_reponse(r), [])

    def test_sans_champ_question_le_schema_actuel_est_respecte(self):
        """question et options ne sont lus que s'ils existent dans le schema."""
        r = ReponseDire(ouverture="Salut.")
        self.assertEqual(fuites_reponse(r), [])
        self.assertIs(epurer_reponse(r)[0], r)


class ExemptionParParticipeTests(SimpleTestCase):
    """Revue de verite du 2026-09-14: un mot interrogatif n'importe ou dans la
    clause exemptait le participe. Ces six phrases affirmaient une action
    absente du registre et passaient, en suite, en question et dans
    present_choices."""

    AFFIRMATIONS = (
        "Ton cours est déplacé à 14 h ou tu préfères 15 h ?",
        "Quel autre bloc veux-tu ajouter maintenant que ton quart est placé ?",
        "Où veux-tu mettre la révision maintenant que le Gym est calé à 9 h ?",
        "Ton cours est bien ajouté pour quel jour déjà ?",
        "Ton bloc Gym a été créé quand tu voulais, veux-tu autre chose ?",
        "Ta séance est déplacée à 14 h quel autre changement veux-tu ?",
        "Je supprime ton cours de jeudi, ça te va ?",
    )
    QUESTIONS = (
        "Ton cours est placé à quelle heure ?",
        "Est-ce que ton horaire a changé cette session ?",
        "Le quart de jeudi est annulé ou juste décalé ?",
        "Pour quelle journée ?",
        "Tu veux que je le mette à quelle heure ?",
        "Quelle place veux-tu lui donner ?",
        "Tu veux qu'il soit déplacé à 15 h ?",
    )

    def test_les_affirmations_deguisees_sont_coupees_partout(self):
        from services.agent.tools.interactive import PresentChoicesTool
        from services.agent_v2.mesure import fuite_question

        class AvecQuestion(ReponseDire):
            pass

        for phrase in self.AFFIRMATIONS:
            with self.subTest(phrase=phrase):
                self.assertTrue(fuite_question(phrase))
                epuree, n = epurer_reponse(ReponseDire(suite=phrase))
                self.assertEqual(epuree.suite, "")
                self.assertGreaterEqual(n, 1)
                epuree, _ = epurer_reponse(ReponseDire(question=phrase, options=["14 h", "15 h"]))
                self.assertEqual(epuree.question, "")
                self.assertEqual(epuree.options, [])
                r = PresentChoicesTool().execute(
                    None, question=phrase if len(phrase) <= 140 else phrase[:139] + "?",
                    source="jours", options=[{"label": "Lundi", "value": "Lundi."},
                                             {"label": "Jeudi", "value": "Jeudi."}])
                self.assertFalse(r.success)

    def test_les_vraies_questions_passent(self):
        from services.agent_v2.mesure import fuite_question

        for phrase in self.QUESTIONS + ("Dis-moi à quelle heure et je le crée.",
                                        "Dis-moi à quelle heure commence ton quart et je le crée."):
            with self.subTest(phrase=phrase):
                self.assertEqual(fuite_question(phrase), [])

    def test_une_valeur_de_puce_a_l_imperatif_passe(self):
        """Le banc du 2026-09-14: « Place ma révision jeudi. » faisait refuser
        un choix de jours reel."""
        from services.agent.tools.interactive import PresentChoicesTool

        r = PresentChoicesTool().execute(
            None, question="Quel jour pour ta révision ?", source="jours",
            options=[{"label": "Lundi (aujourd'hui)", "value": "Place ma révision aujourd'hui."},
                     {"label": "Jeudi", "value": "Place ma révision jeudi."}])
        self.assertTrue(r.success, r.message)
        epuree, n = epurer_reponse(ReponseDire(ouverture="Ton cours est placé."))
        self.assertEqual(n, 1)
