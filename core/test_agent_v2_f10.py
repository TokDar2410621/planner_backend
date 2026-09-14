"""
Round 10, fixeur f10. Chaque classe a ete ecrite AVANT son correctif et vue
en echec.

P1  Fuite du banc: le prompt citait mot pour mot le scenario du banc (« mon
    cours de maths », « Calcul differentiel », « mardi et jeudi de 16 h a
    17 h 50 ») et ses donnees; la présélection « 16h-17h50 » est apparue en
    s03-1. Aucun exemple des prompts ni des descriptions d'outils ne reprend
    une phrase du banc ou des donnees de la semaine type ou du compte vitrine.
P2  k4-1 « souper jeudi soir a 6 h » a cree une serie hebdomadaire. Un jour
    nomme sans mot de recurrence donne UN evenement date.
P3  Prose sous une question destructive du code: « Je comprends, attends ta
    confirmation. » (k2-1), « Parfait. Reponds seulement ce jeudi ou tous les
    jeudis, et je m'en occupe. » (k1b-1), liste de 2 sous une question sur 3
    (s05-1).
P4  s06-1 ne demandait plus l'heure et s06-2 a choisi 16 h seul. Un
    formulaire qui demande les jours d'une activite demande aussi sa plage
    horaire, sans defaut invente.

Horloge du harnais: lundi 14 septembre 2026, 8 h (jour 0 = lundi).
"""
import json
import re
import unicodedata

from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase, TransactionTestCase

from core.models import ConversationMessage, RecurringBlock, ScheduledBlock
from core.test_agent_v2_gardes import HarnaisGardes
from services.agent.tools.base import ToolResult
from services.agent_v2.redaction import ReponseDire, bloc_factuel, composer
from services.agent_v2.registre import Registre


def _plat(texte: str) -> str:
    sans = unicodedata.normalize("NFKD", texte or "").encode("ascii", "ignore").decode("ascii")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", sans.lower()).split())


# ── P1: aucune phrase du banc dans les prompts ni les descriptions ──────────

# Messages et donnees des bancs (live-runs-apres-r5.md, apres-r8/notation.md,
# apres-r9/notation.md, tour.py, lot-suite-r9.sh) et du compte vitrine. Une
# phrase par scenario ou par donnee de semaine type; comparees aplaties.
PHRASES_DU_BANC = (
    # s01, rendez-vous
    "rendez-vous chez le dentiste", "dentiste",
    # s02, etude
    "je veux etudier plus cette semaine", "etudier plus",
    # s03 et K6, cours de maths
    "mets mon cours de maths", "cours de maths", "calcul differentiel",
    "mardi et jeudi de 16 h a 17 h 50", "16 h a 17 h 50", "16h-17h50", "17 h 50",
    # formulation reservee au prochain banc (notation r9, constat 2)
    "lundi et mercredi de 18 h 30 a 20 h 20", "18 h 30 a 20 h 20",
    # s04
    "deplace mon cours",
    # s05, n*, f*
    "efface tout jeudi", "supprime ces trois blocs", "trois blocs le jeudi",
    "seulement ce jeudi", "tous les jeudis", "ce jeudi seulement",
    # s06
    "ajoute gym 3 fois par semaine", "3 fois par semaine", "gym commence a 18h",
    # s07, s08, k5
    "qu'est-ce que j'ai cette semaine", "c'est quoi mon horaire", "j'ai quoi samedi",
    # s09
    "place ma revision de chimie", "revision de chimie", "avant vendredi",
    # s10 et quart de la semaine type
    "j'ai un quart de 19h a 2h jeudi", "19 h a 2 h", "19h a 2h",
    "quart au depanneur", "depanneur",
    # k1b, k2, k3, k4, f1, p3
    "annule mes lectures de samedi", "lecture ce samedi", "lectures de samedi",
    "deplace mon gym a 7 h", "souper jeudi", "jeudi soir a 6 h", "gym demain matin",
    "gym jeudi 10h", "lecture jeudi 15h", "marche aux puces",
    # semaine type du banc
    "chimie generale", "chimie organique", "physique mecanique", "programmation",
    "cours d'anglais",
    # compte vitrine
    "litterature quebecoise", "cafe depot", "souper d'equipe", "soccer",
)


def _textes_envoyes_au_modele() -> dict:
    from services.agent.tools import ALL_TOOLS
    from services.agent_v2 import prompts

    textes = {nom: getattr(prompts, nom) for nom in ("REGLES_AGIR", "PREMIER_CONTACT", "PROMPT_DIRE")}
    textes["ReponseDire"] = json.dumps(ReponseDire.model_json_schema(), ensure_ascii=False)
    for outil in ALL_TOOLS:
        textes[f"outil {outil.name}"] = (
            f"{outil.name}\n{outil.description}\n"
            f"{json.dumps(outil.parameters, ensure_ascii=False)}")
    return textes


class P1FuiteDuBancTests(SimpleTestCase):

    def test_aucune_phrase_du_banc(self):
        for nom, texte in _textes_envoyes_au_modele().items():
            plat = f" {_plat(texte)} "
            for phrase in PHRASES_DU_BANC:
                with self.subTest(texte=nom, phrase=phrase):
                    self.assertNotIn(f" {_plat(phrase)} ", plat)

    def test_la_liste_couvre_tous_les_textes(self):
        textes = _textes_envoyes_au_modele()
        self.assertIn("outil create_block", textes)
        self.assertIn("outil present_choices", textes)
        self.assertIn("outil cancel_scheduled_block", textes)
        self.assertIn("PROMPT_DIRE", textes)

    def test_les_regles_restent(self):
        from services.agent_v2.prompts import PROMPT_DIRE, REGLES_AGIR
        for requis in ("COURS EXISTANT", "AJOUT AVEC JOURS ET HEURES",
                       "c'est un nouveau cours: create_block dans ce tour",
                       "un cours revient chaque semaine par defaut",
                       'present_choices (source "blocs") avec ces seuls cours',
                       "Exemples: « L", "echeance sans jour choisi"):
            with self.subTest(requis=requis):
                self.assertIn(requis, REGLES_AGIR)
        self.assertIn("seulement cette fois ou chaque semaine", " ".join(PROMPT_DIRE.split()))


# ── P2: un jour nomme sans mot de recurrence donne un evenement unique ──────


class P2RegleTests(SimpleTestCase):

    def test_le_prompt_dit_la_regle(self):
        from services.agent_v2.prompts import REGLES_AGIR
        for requis in ("UN JOUR NOMME SANS MOT DE RECURRENCE",
                       "un seul evenement date -> schedule_task_at",
                       "« chaque », « tous les », « toutes les », « les lundis »"):
            with self.subTest(requis=requis):
                self.assertIn(requis, REGLES_AGIR)
        # La vieille regle attrapait tout jour + heure pour create_block.
        self.assertNotIn("l'utilisateur decrit ses horaires habituels AVEC jours et heures -> create_block.",
                         REGLES_AGIR)

    def test_les_descriptions_disent_la_regle(self):
        from services.agent.tools import TOOL_MAP
        self.assertIn("sans mot de récurrence", TOOL_MAP["create_block"].description)
        self.assertIn("sans mot de récurrence", TOOL_MAP["schedule_task_at"].description)

    def test_lecture_des_mots_de_recurrence(self):
        from services.agent_v2 import outils as outils_v2
        uniques = ("souper jeudi soir a 6 h", "souper ce jeudi a 18 h", "mets un souper jeudi a 18 h",
                   "gym mardi et jeudi 7 h a 8 h", "yoga samedi a 10 h")
        recurrents = ("souper chaque jeudi a 18 h", "souper tous les jeudis a 18 h",
                      "souper les jeudis a 18 h", "ajoute gym le jeudi a 15 h",
                      "yoga toutes les semaines le samedi", "gym jeudi a 7 h, chaque semaine",
                      "ajoute gym 3 fois par semaine", "gym lundis et mercredis a 7 h",
                      "yoga samedi a 10 h a partir du 3 octobre", "mon horaire: yoga samedi 10 h",
                      "d'habitude je nage le samedi a 10 h")
        for texte in uniques:
            with self.subTest(texte=texte):
                self.assertTrue(outils_v2.jour_sans_recurrence(texte))
        for texte in recurrents + ("ajoute du yoga", "Voici mes réponses :\nJours: Lundi"):
            with self.subTest(texte=texte):
                self.assertFalse(outils_v2.jour_sans_recurrence(texte))


class P2GardeTests(HarnaisGardes, TransactionTestCase):

    def _creer(self, brut, tache, titre='Souper', block_type='meal', days=('jeudi',),
               debut='18:00', fin='19:00'):
        registre, tools = self.outils(brut, tache=tache)
        self.appeler(tools, 'create_block', title=titre, block_type=block_type, days=list(days),
                     start_time=debut, end_time=fin)
        return registre, tools, registre.actions[-1]

    def test_k4_souper_jeudi_soir_devient_un_evenement(self):
        brut = 'souper jeudi soir a 6 h'
        self.message_courant(brut)
        registre, tools, action = self._creer(brut, 'p2:1')
        self.assertFalse(action.succes, action.message)
        self.assertEqual(action.donnees.get('evenement_unique'), ['2026-09-17'])
        self.assertIn('schedule_task_at', action.message)
        self.assertIn('2026-09-17', action.message)
        self.assertNotIn(chr(0x2014), action.message)
        self.assertFalse(RecurringBlock.objects.filter(user=self.user, title='Souper').exists())

        self.appeler(tools, 'schedule_task_at', title='Souper', date='2026-09-17',
                     start_time='18:00', end_time='19:00')
        self.assertTrue(registre.actions[-1].succes, registre.actions[-1].message)
        self.assertTrue(ScheduledBlock.objects.filter(user=self.user, date='2026-09-17').exists())
        faits = bloc_factuel(registre)
        self.assertIn('Souper', faits)
        self.assertNotIn("Je n'ai pas pu", faits)
        self.assertNotIn('les jeudis', faits)

    def test_un_refus_sans_suite_reste_dit(self):
        brut = 'souper jeudi soir a 6 h'
        self.message_courant(brut)
        registre, _tools, action = self._creer(brut, 'p2:2')
        self.assertFalse(action.succes)
        self.assertIn("Je n'ai pas pu", bloc_factuel(registre))

    def test_recurrence_dite_passe(self):
        for i, brut in enumerate(('souper chaque jeudi a 18 h', 'souper tous les jeudis a 18 h',
                                  'souper les jeudis a 18 h')):
            with self.subTest(brut=brut):
                RecurringBlock.objects.filter(user=self.user, title='Souper').delete()
                self.message_courant(brut)
                _r, _t, action = self._creer(brut, f'p2:r{i}')
                self.assertTrue(action.succes, action.message)

    def test_cours_et_quart_reviennent_par_defaut(self):
        brut = 'lundi et mercredi de 13 h a 15 h'
        self.message_courant(brut)
        _r, _t, action = self._creer(brut, 'p2:c', titre='Cours d\'histoire', block_type='course',
                                     days=('lundi', 'mercredi'), debut='13:00', fin='15:00')
        self.assertTrue(action.succes, action.message)
        brut = 'mon quart samedi de 9 h a 13 h'
        self.message_courant(brut)
        _r, _t, action = self._creer(brut, 'p2:w', titre='Quart', block_type='work',
                                     days=('samedi',), debut='09:00', fin='13:00')
        self.assertTrue(action.succes, action.message)

    def test_reponse_de_formulaire_passe(self):
        brut = "Voici mes réponses :\nJours de yoga: Samedi\nPlage horaire: 10:00 - 11:00"
        self.message_courant(brut)
        _r, _t, action = self._creer(brut, 'p2:f', titre='Yoga', block_type='sport',
                                     days=('samedi',), debut='10:00', fin='11:00')
        self.assertTrue(action.succes, action.message)

    def test_reponse_a_une_question_sur_une_habitude_passe(self):
        u1 = ConversationMessage.objects.create(user=self.user, role='user',
                                                content='ajoute du yoga chaque semaine')
        ConversationMessage.objects.create(
            user=self.user, role='assistant', content='Quel jour et à quelle heure ?',
            metadata={'en_reponse_a': u1.pk, 'question_posee': True})
        brut = 'samedi a 10 h'
        self.message_courant(brut)
        _r, _t, action = self._creer(brut, 'p2:q', titre='Yoga', block_type='sport',
                                     days=('samedi',), debut='10:00', fin='11:00')
        self.assertTrue(action.succes, action.message)


# ── P3: la prose sous une question destructive du code ──────────────────────


def _question_code(motif, question, titres):
    demandes = [{"motif": motif, "cle": f"c{i}", "cible": {"titre": t}} for i, t in enumerate(titres)]
    return {"source": "demande", "motif": motif, "question": question,
            "chips": [{"label": "Non, garde tout", "value": "Non, ne change rien.", "option": "annuler"}],
            "demandes": demandes, "cles_posees": [d["cle"] for d in demandes]}


class P3ProseSousQuestionDuCodeTests(SimpleTestCase):

    def _registre_retenu(self, outil):
        r = Registre()
        r.ajouter("list_blocks", {"day_of_week": "jeudi"}, ToolResult(success=True, data={"blocks": []}, message="ok"))
        r.ajouter(outil, {}, ToolResult(success=False, data={"demande": {"motif": "portee_jour", "cle": "c0"}},
                                        message="retenu"))
        return r

    def test_les_prose_du_banc_ne_passent_pas(self):
        cas = (
            ("destructif", "Tu veux vraiment annuler Lecture le sam. 19 sept. ?",
             ReponseDire(ouverture="Je comprends, attends ta confirmation.")),
            ("portee_jour", "Tu veux enlever A, B et Sommeil seulement ce jeudi 17 sept. ou tous les jeudis ?",
             ReponseDire(ouverture="Parfait.",
                         suite="Réponds seulement ce jeudi ou tous les jeudis, et je m'en occupe.")),
            ("portee_jour", "Tu veux enlever A, B et Sommeil seulement ce jeudi 17 sept. ou tous les jeudis ?",
             ReponseDire(suite="Ton Sommeil est aussi concerné.")),
        )
        for motif, question, brut in cas:
            with self.subTest(prose=brut.ouverture or brut.suite):
                compo = composer(brut, self._registre_retenu("delete_block"), "faits",
                                 _question_code(motif, question, ["A"]))
                self.assertEqual(compo.prose, "")
                self.assertEqual(compo.question, question)

    def test_une_question_de_formulaire_garde_sa_prose(self):
        compo = composer(ReponseDire(ouverture="Bonne idée."), Registre(), "",
                         {"source": "formulaire", "motif": "formulaire", "question": "",
                          "chips": [], "demandes": [], "cles_posees": []})
        self.assertEqual(compo.prose, "Bonne idée.")

    def test_le_prompt_dire_ne_fait_plus_parler_d_attente(self):
        from services.agent_v2.prompts import PROMPT_DIRE
        self.assertIn("QUESTION DEJA POSEE PAR LE CODE: laisse ouverture et suite vides", PROMPT_DIRE)
        self.assertNotIn("seulement ce jeudi ou tous les jeudis", PROMPT_DIRE)


class P3ListeEtQuestionTests(SimpleTestCase):

    def _registre(self):
        r = Registre()
        blocs = [
            {"title": "Labo", "day_of_week": 3, "day_name": "Jeudi", "start_time": "08:00",
             "end_time": "09:50", "block_type": "course"},
            {"title": "Quart", "day_of_week": 3, "day_name": "Jeudi", "start_time": "19:00",
             "end_time": "02:00", "block_type": "work"},
            {"title": "Sommeil", "day_of_week": 3, "day_name": "Jeudi", "start_time": "23:00",
             "end_time": "07:00", "block_type": "sleep"},
        ]
        r.ajouter("list_blocks", {"day_of_week": "jeudi"},
                  ToolResult(success=True, data={"blocks": blocs}, message="3 blocs"))
        return r

    def test_la_liste_montre_chaque_element_de_la_question(self):
        faits = bloc_factuel(self._registre(), titres_vises=["Labo", "Quart", "Sommeil"])
        for titre in ("Labo", "Quart", "Sommeil"):
            with self.subTest(titre=titre):
                self.assertIn(titre, faits)
        self.assertIn("23 h à 7 h · Sommeil", faits)

    def test_sans_question_le_sommeil_reste_discret(self):
        self.assertNotIn("Sommeil", bloc_factuel(self._registre()))

    def test_deux_evenements_du_meme_titre_sont_comptes(self):
        from services.agent_v2.rendu import rendre_demandes
        demande = {"type": "confirmation", "motif": "destructif", "cle": "k",
                   "outil": "cancel_scheduled_block",
                   "parametres": {"date": "2026-09-19", "title": "Atelier"},
                   "cible": {"titre": "Atelier", "date": "2026-09-19", "ids": [7, 8]},
                   "options": [{"id": "confirmer"}, {"id": "annuler"}]}
        question, _chips, _cles = rendre_demandes([demande])
        self.assertIn("les 2 créneaux de Atelier", question)
        demande["cible"]["ids"] = [7]
        question, _chips, _cles = rendre_demandes([demande])
        self.assertNotIn("créneaux", question)


# ── P4: un formulaire sur les jours d'une activite demande aussi l'heure ────


JOURS_OPTIONS = [{'value': str(i), 'label': j} for i, j in enumerate(
    ('Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'))]


class P4FormulaireTests(HarnaisGardes, TransactionTestCase):

    def _formulaire(self, brut, inputs, tache):
        self.message_courant(brut)
        registre, tools = self.outils(brut, tache=tache)
        self.appeler(tools, 'present_form', inputs=inputs)
        action = registre.actions[-1]
        self.assertTrue(action.succes, action.message)
        return action.donnees['interactive_inputs']

    def test_s06_1_du_banc_demande_la_plage_horaire(self):
        # Arguments exacts de apres-r9/runs/s06-1.json.
        champs = self._formulaire('ajoute gym 3 fois par semaine', [
            {'id': 'gym_days', 'type': 'checkbox', 'label': 'Jours de gym',
             'question': 'Quels jours veux-tu aller à la gym ?', 'options': JOURS_OPTIONS},
            {'id': 'gym_duration', 'type': 'duration', 'label': "Durée d'une séance",
             'question': 'Combien de temps par séance ?', 'default': 60, 'presets': [30, 60, 90, 120]},
        ], 'p4:1')
        plages = [c for c in champs if c['type'] == 'time_range']
        self.assertEqual(len(plages), 1, champs)
        self.assertNotIn('default', plages[0])
        self.assertTrue(plages[0]['question'].endswith('?'))
        self.assertNotIn(chr(0x2014), json.dumps(champs, ensure_ascii=False))

    def test_r8_une_plage_inventee_perd_son_defaut(self):
        champs = self._formulaire('ajoute gym 3 fois par semaine', [
            {'id': 'gym_days', 'type': 'checkbox', 'label': 'Jours de gym',
             'question': 'Quels jours ?', 'options': JOURS_OPTIONS, 'default': ['1', '3', '5']},
            {'id': 'gym_time', 'type': 'time_range', 'label': "Heure d'entraînement",
             'question': 'À quelle heure ?', 'default': {'start': '17:00', 'end': '18:00'}},
        ], 'p4:2')
        plages = [c for c in champs if c['type'] == 'time_range']
        self.assertEqual(len(plages), 1)
        self.assertNotIn('default', plages[0])

    def test_une_heure_dite_n_est_pas_redemandee(self):
        champs = self._formulaire('ajoute gym 3 fois par semaine à 18 h', [
            {'id': 'gym_days', 'type': 'checkbox', 'label': 'Jours de gym',
             'question': 'Quels jours ?', 'options': JOURS_OPTIONS},
        ], 'p4:3')
        self.assertFalse([c for c in champs if c['type'] == 'time_range'])

    def test_le_formulaire_d_etude_en_heures_totales_reste_court(self):
        champs = self._formulaire('je veux lire davantage cette semaine', [
            {'id': 'heures', 'type': 'duration', 'label': "Heures de lecture",
             'question': "Combien d'heures en tout ?", 'default': 240},
            {'id': 'jours', 'type': 'checkbox', 'label': 'Jours', 'question': 'Quels jours ?',
             'options': JOURS_OPTIONS},
        ], 'p4:4')
        self.assertFalse([c for c in champs if c['type'] == 'time_range'])

    def test_le_sommeil_garde_son_defaut(self):
        champs = self._formulaire('aide-moi a demarrer', [
            {'id': 'sommeil', 'type': 'time_range', 'label': 'Sommeil',
             'question': 'Quand dors-tu ?', 'default': {'start': '23:00', 'end': '07:00'}},
            {'id': 'jours', 'type': 'checkbox', 'label': 'Jours travaillés',
             'question': 'Quels jours travailles-tu ?', 'options': JOURS_OPTIONS},
        ], 'p4:5')
        self.assertEqual([c.get('default') for c in champs if c['type'] == 'time_range'],
                         [{'start': '23:00', 'end': '07:00'}])

    def test_un_deplacement_par_liste_ne_change_pas(self):
        champs = self._formulaire('deplace mon atelier', [
            {'id': 'jour', 'type': 'select', 'label': 'Nouveau jour', 'question': 'Vers quel jour ?',
             'options': JOURS_OPTIONS},
        ], 'p4:6')
        self.assertEqual(len(champs), 1)


class P4RegleTests(SimpleTestCase):

    def test_le_prompt_demande_toutes_les_valeurs_d_un_coup(self):
        from services.agent_v2.prompts import REGLES_AGIR
        for requis in ("Quand tu demandes, demande TOUTES les valeurs qui manquent dans le meme formulaire",
                       "une valeur que tu n'as pas demandee ne se devine pas au tour suivant"):
            with self.subTest(requis=requis):
                self.assertIn(requis, REGLES_AGIR)
