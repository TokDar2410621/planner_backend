"""
Postback structure des puces (tap): au tap d'une puce, le front envoie aussi
{demande: cle, option: id}. L'egalite d'identifiants remplace la comparaison
de texte normalisee, et RIEN d'autre ne change: un tap forge equivaut a taper
le texte exact de la puce (aucune surface nouvelle), le texte libre garde
exactement sa semantique, la lecture reste PAR demande.

Horloge morale: les regles de septembre 2026 (rondes 2 a 6) restent la loi.
"""
from django.test import SimpleTestCase

from core.test_agent_v2_gardes import puces
from services.agent_v2 import demandes as dem
from services.agent_v2.agent import _chips_reponse

PORTEE = puces({
    'motif': 'portee_jour', 'cle': 'p1',
    'cible': {'titre': 'Quart au dépanneur', 'jour': 3, 'date': '2026-09-17'},
    'options': [{'id': 'occurrence'}, {'id': 'serie'}, {'id': 'annuler'}],
})
DESTR = puces({
    'motif': 'destructif', 'cle': 'd1',
    'options': [{'id': 'confirmer'}, {'id': 'annuler'}],
})


class TapOptionChoisieTests(SimpleTestCase):

    def test_le_tap_exact_donne_l_option_sans_lire_le_texte(self):
        # « ok » seul ne choisirait jamais « serie »; le tap, si.
        self.assertIsNone(dem.option_choisie('ok', PORTEE))
        self.assertEqual(
            dem.option_choisie('ok', PORTEE, tap={'demande': 'p1', 'option': 'serie'}),
            'serie')

    def test_le_tap_destructif_vaut_la_puce_exacte(self):
        # Un motif destructif n'accepte aucun oui libre; le tap est l'egal
        # byte-exact du bouton, il confirme.
        self.assertIsNone(dem.option_choisie('ok', DESTR))
        self.assertEqual(
            dem.option_choisie('ok', DESTR, tap={'demande': 'd1', 'option': 'confirmer'}),
            'confirmer')

    def test_une_mauvaise_cle_retombe_sur_le_texte(self):
        tap = {'demande': 'autre-cle', 'option': 'serie'}
        self.assertIsNone(dem.option_choisie('ok', PORTEE, tap=tap))
        # Le texte exact de la puce garde tout son pouvoir malgre le tap errone.
        self.assertEqual(dem.option_choisie('Tous les jeudis', PORTEE, tap=tap), 'serie')

    def test_une_option_hors_demande_retombe_sur_le_texte(self):
        self.assertIsNone(
            dem.option_choisie('ok', PORTEE, tap={'demande': 'p1', 'option': 'inconnue'}))

    def test_la_lecture_reste_par_demande(self):
        # Le tap de la demande p1 ne choisit rien pour d1, meme si l'option
        # existe dans les deux (« annuler »).
        tap = {'demande': 'p1', 'option': 'annuler'}
        self.assertEqual(dem.option_choisie('ok', PORTEE, tap=tap), 'annuler')
        self.assertIsNone(dem.option_choisie('ok', DESTR, tap=tap))

    def test_un_tap_malforme_est_ignore(self):
        for tap in ('serie', {}, {'option': 'serie'}, {'demande': 'p1'},
                    {'demande': '', 'option': 'serie'}, ['p1', 'serie'], 42):
            with self.subTest(tap=tap):
                self.assertIsNone(dem.option_choisie('ok', PORTEE, tap=tap))

    def test_une_demande_sans_cle_ne_matche_jamais_un_tap(self):
        sans_cle = {**PORTEE}
        sans_cle.pop('cle')
        self.assertIsNone(
            dem.option_choisie('ok', sans_cle, tap={'demande': '', 'option': 'serie'}))

    def test_sans_tap_regression_zero(self):
        # Le comportement texte du round 5 est inchange, tap absent ou None.
        self.assertEqual(dem.option_choisie('Tous les jeudis', PORTEE), 'serie')
        self.assertEqual(dem.option_choisie('Tous les jeudis', PORTEE, tap=None), 'serie')
        self.assertIsNone(dem.option_choisie('Tous les jeudis ?', PORTEE, tap=None))
        self.assertIsNone(dem.option_choisie('oui', DESTR, tap=None))


class ChipsReponseTests(SimpleTestCase):

    def test_les_chips_d_une_demande_rendue_portent_le_postback(self):
        propres = _chips_reponse(PORTEE['chips'], [PORTEE])
        serie = next(c for c in propres if c['option'] == 'serie')
        self.assertEqual(serie['demande'], 'p1')
        self.assertEqual(serie['label'], 'Tous les jeudis')
        # Toutes les puces de la demande sont appariees.
        self.assertTrue(all(c.get('demande') == 'p1' for c in propres))

    def test_sans_demande_rendue_les_chips_restent_nues(self):
        propres = _chips_reponse(PORTEE['chips'], [])
        self.assertTrue(propres)
        for chip in propres:
            self.assertNotIn('demande', chip)
            self.assertNotIn('option', chip)
            self.assertEqual(set(chip), {'label', 'value'})

    def test_une_chip_etrangere_a_la_demande_reste_nue(self):
        chips = PORTEE['chips'] + [{'label': 'Voir mon planning',
                                    'value': 'Voir mon planning'}]
        propres = _chips_reponse(chips, [PORTEE])
        etrangere = next(c for c in propres if c['label'] == 'Voir mon planning')
        self.assertNotIn('demande', etrangere)
        self.assertNotIn('option', etrangere)
