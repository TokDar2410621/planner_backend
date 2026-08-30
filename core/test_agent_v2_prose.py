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
