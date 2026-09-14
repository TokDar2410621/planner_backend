"""
La garantie structurelle, testee sans le moindre appel LLM.

Le modele DIRE ne redige pas les faits. Le code rend le compte rendu (rendu.py,
branche par la couture de redaction.py), et la prose de DIRE est ecartee en
entier des qu'elle cite une reference absente du registre.

Depuis le 2026-09-14, UN SEUL NARRATEUR: les phrases `actions` de DIRE ne sont
plus jamais rendues. Les tests qui verifiaient qu'une phrase referencee
survivait ont ete retournes: elle ne doit PAS apparaitre, les faits suffisent.
"""
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase

from services.agent.tools.base import ToolResult
from services.agent_v2 import redaction
from services.agent_v2.redaction import (
    ActionCitee, ReponseDire, assembler, bloc_factuel, bloc_lecture,
    question_code,
)
from services.agent_v2.registre import Registre


def _deux_creations() -> Registre:
    r = Registre()
    r.ajouter('create_block', {'title': 'Maths'},
              ToolResult(success=True, message="Bloc 'Maths' cree (09:00-12:00) les Lundi",
                         data={'created': [{'title': 'Maths', 'day_of_week': 0,
                                            'day_name': 'Lundi', 'start_time': '09:00',
                                            'end_time': '12:00'}]}))
    r.ajouter('create_block', {'title': 'Sport'},
              ToolResult(success=False, message="Chevauchement avec 'Travail' (09:00-17:00)"))
    return r


def _faux_rendu(journal=None):
    """rendu.py simule: FAITS sur mutation reussie, LECTURE sinon."""
    journal = journal if journal is not None else {}

    def rendre_faits(registre, aujourdhui=None, cles_posees=None):
        journal.setdefault('faits', []).append((aujourdhui, cles_posees))
        return 'FAITS' if any(a.succes and a.est_mutation for a in registre.actions) else ''

    def rendre_lecture(registre, aujourdhui=None):
        journal.setdefault('lecture', []).append(aujourdhui)
        return 'LECTURE' if any(a.succes and a.outil == 'list_blocks'
                                for a in registre.actions) else ''

    def rendre_demandes(demandes, aujourdhui=None):
        journal.setdefault('demandes', []).append(demandes)
        return ('Question ?', [{'label': 'A', 'value': 'a', 'option': 'x'}],
                [demandes[0]['cle']])

    return SimpleNamespace(rendre_faits=rendre_faits, rendre_lecture=rendre_lecture,
                           rendre_demandes=rendre_demandes,
                           marqueurs_bruts=lambda texte: ['refus'] if 'Refus' in texte else [])


class CoutureRenduTests(SimpleTestCase):
    """bloc_factuel, bloc_lecture et question_code deleguent a rendu.py."""

    def test_les_faits_viennent_de_rendre_faits(self):
        journal = {}
        with patch.object(redaction, '_charger_rendu', return_value=_faux_rendu(journal)):
            self.assertEqual(bloc_factuel(_deux_creations(), cles_posees={'k'}), 'FAITS')
        self.assertEqual(journal['faits'], [(None, {'k'})])

    def test_sans_fait_la_lecture_prend_le_relais(self):
        r = Registre()
        r.ajouter('list_blocks', {}, ToolResult(success=True, message='3 blocs trouves'))
        with patch.object(redaction, '_charger_rendu', return_value=_faux_rendu()):
            self.assertEqual(bloc_factuel(r), 'LECTURE')
            self.assertEqual(bloc_lecture(r), 'LECTURE')

    def test_rien_a_dire_rend_une_chaine_vide(self):
        with patch.object(redaction, '_charger_rendu', return_value=_faux_rendu()):
            self.assertEqual(bloc_factuel(Registre()), '')

    def test_question_code_delegue_et_rend_les_cles(self):
        demandes = [{'cle': 'p1', 'motif': 'portee_jour'}]
        with patch.object(redaction, '_charger_rendu', return_value=_faux_rendu()):
            question, chips, cles = question_code(demandes)
        self.assertEqual(question, 'Question ?')
        self.assertEqual(chips[0]['option'], 'x')
        self.assertEqual(cles, ['p1'])

    def test_aucune_demande_aucune_question(self):
        with patch.object(redaction, '_charger_rendu', return_value=_faux_rendu()):
            self.assertEqual(question_code([]), ('', [], []))

    def test_marqueurs_bruts_delegue(self):
        with patch.object(redaction, '_charger_rendu', return_value=_faux_rendu()):
            self.assertEqual(redaction.marqueurs_bruts('- Refus: x'), ['refus'])


class InvariantsDuRenduTests(SimpleTestCase):
    """Vrais sur le vrai rendu.py."""

    def test_un_registre_vide_ne_produit_aucun_bloc(self):
        self.assertEqual(bloc_factuel(Registre()), '')

    def test_une_creation_reussie_nomme_son_titre(self):
        self.assertIn('Maths', bloc_factuel(_deux_creations()))

    def test_un_budget_epuise_est_dit(self):
        r = _deux_creations()
        r.budget_epuise = True
        texte = bloc_factuel(r).lower()
        self.assertTrue('interrompu' in texte or 'arrêté' in texte, texte)


class AssemblageTests(SimpleTestCase):
    """assembler: faits, puis prose, puis question. Rendu simule."""

    def setUp(self):
        patcher = patch.object(redaction, '_charger_rendu', return_value=_faux_rendu())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_une_reference_valide_garde_la_prose_mais_pas_la_phrase(self):
        """Un seul narrateur: la phrase citee n'est jamais rendue, les faits
        la remplacent. La reference valide ne coupe rien."""
        r = _deux_creations()
        brut = ReponseDire(ouverture="Voila.",
                           actions=[ActionCitee(ref='a1', phrase="Maths est cale le lundi.")],
                           suite="")
        texte, rejetees = assembler(brut, r)
        self.assertEqual(rejetees, 0)
        self.assertNotIn('Maths est cale', texte)
        self.assertEqual(texte, "FAITS\n\nVoila.")

    def test_une_action_inventee_est_supprimee_et_comptee(self):
        """LE test central du projet: une reference inventee fait tomber
        toute la prose, pas seulement sa phrase."""
        r = _deux_creations()
        brut = ReponseDire(
            ouverture="C'est note.",
            actions=[
                ActionCitee(ref='a1', phrase="Maths est cale le lundi."),
                ActionCitee(ref='a42', phrase="J'ai aussi reorganise toute ta semaine."),
            ],
            suite="")
        texte, rejetees = assembler(brut, r)
        self.assertEqual(rejetees, 1)
        self.assertNotIn('reorganise toute ta semaine', texte)
        self.assertNotIn("C'est note", texte)
        self.assertEqual(texte, 'FAITS')

    def test_une_ref_inventee_dans_refs_coupe_aussi(self):
        r = _deux_creations()
        texte, rejetees = assembler(ReponseDire(ouverture="Top.", refs=['a9']), r)
        self.assertEqual(rejetees, 1)
        self.assertNotIn('Top', texte)

    def test_un_registre_vide_ne_laisse_passer_aucune_action(self):
        """Le cas du 18 aout: tour sans le moindre outil."""
        brut = ReponseDire(
            ouverture="C'est note,",
            actions=[ActionCitee(ref='a1', phrase="j'ai supprime les blocs qui chevauchent.")],
            suite="")
        texte, rejetees = assembler(brut, Registre())
        self.assertEqual(rejetees, 1)
        self.assertNotIn('supprime', texte)

    def test_une_reference_vide_est_rejetee(self):
        r = _deux_creations()
        brut = ReponseDire(ouverture="", actions=[ActionCitee(ref='', phrase="J'ai tout fait.")], suite="")
        texte, rejetees = assembler(brut, r)
        self.assertEqual(rejetees, 1)
        self.assertNotIn('tout fait', texte)

    def test_le_bloc_factuel_ouvre_le_texte_final(self):
        texte, _ = assembler(ReponseDire(ouverture="Salut."), _deux_creations())
        self.assertTrue(texte.startswith('FAITS'), texte)

    def test_la_question_ferme_le_texte(self):
        texte, _ = assembler(
            ReponseDire(ouverture="Salut.", question="Autre chose ?"), _deux_creations())
        self.assertEqual(texte, "FAITS\n\nSalut.\n\nAutre chose ?")


class VocabulaireInterneTests(SimpleTestCase):
    """Banc du 2026-09-14: « Il est marqué flexible », « Veux-tu le verrouiller
    à 17 h ? », « Je dois d'abord clarifier la portée. »"""

    def test_les_mots_internes_ne_partent_pas(self):
        from services.agent_v2.redaction import ReponseDire, composer

        brut = ReponseDire(ouverture="Il est marqué flexible. Tu es libre jeudi.",
                           suite="Je dois d'abord clarifier la portée.",
                           question="Veux-tu le verrouiller à 17 h ?", options=["Oui", "Non"])
        compo = composer(brut, Registre(), "", None)
        self.assertEqual(compo.prose, "Tu es libre jeudi.")
        self.assertEqual(compo.question, "")
        self.assertEqual(compo.chips, [])

    def test_bloc_et_formulaire_ne_partent_pas(self):
        """Banc du 2026-09-14, round 2: s02-1, s03-2, s08-1, s10-1."""
        from services.agent_v2.redaction import ReponseDire, composer

        brut = ReponseDire(
            ouverture="Remplis le formulaire et je m'occupe du reste.",
            suite="Par contre, le bloc de sommeil chevauche ton quart. Samedi reste libre.")
        compo = composer(brut, Registre(), "FAITS", None)
        self.assertEqual(compo.prose, "Samedi reste libre.")
        brut = ReponseDire(suite="Le bloc du soir est bien dégagé.",
                           question="Lequel de tes blocs veux-tu déplacer ?",
                           options=["Chimie", "Physique"])
        compo = composer(brut, Registre(), "FAITS", None)
        self.assertEqual((compo.prose, compo.question, compo.chips), ("", "", []))
