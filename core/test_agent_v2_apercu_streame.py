"""
Apercu streame (2026-09-18): en mode voix_agir, la reponse s'ecrit pendant
qu'AGIR la produit, au lieu d'apparaitre d'un bloc a la fin.

L'invariant de septembre tient ENTIER: chaque phrase passe par le MEME
chemin que la composition finale (epurer_reponse puis composer) avant de
partir. Une phrase qui affirme une action, parle vocabulaire interne ou
pose une question ne s'affiche JAMAIS, meme une seconde. Et `done.response`
reste la verite canonique (faits, prose, question), que le client applique.
"""
import queue
from unittest.mock import MagicMock, patch

from core.test_agent_v2_narrateur import NarrateurBase, demande, ok, refus
from services.agent_v2.agent import PlannerAgentV2
from services.agent_v2.redaction import ReponseDire
from services.agent_v2.registre import Registre


class ApercuBase(NarrateurBase):

    def setUp(self):
        super().setUp()
        profil = self.user.profile
        profil.voix_agir = True
        profil.save(update_fields=["voix_agir"])

    def tour_streame(self, fragments=(), actions=(), message="bonjour"):
        """AGIR pousse `fragments` comme le ferait le flux de PydanticAI,
        puis laisse son texte complet en brouillon."""
        dire = MagicMock(return_value=ReponseDire(ouverture="DIRE A PARLE"))

        def _agir(self_agent, user, msg, registre):
            for outil, params, res in actions:
                self_agent.signaler_outil(registre.ajouter(outil, params, res))
            for fragment in fragments:
                self_agent.pousser_brouillon(fragment)
            self_agent._brouillon_agir = "".join(fragments).strip()
            return ""

        with patch.object(PlannerAgentV2, "_agir", _agir), \
             patch.object(PlannerAgentV2, "_dire", dire):
            evts = list(PlannerAgentV2().process_message_stream(self.user, message))
        return evts, evts[-1]

    def deltas_de(self, evts):
        return [e["text"] for e in evts if e["type"] == "delta"]


class ApercuStreameTests(ApercuBase):

    def test_les_phrases_partent_au_fil_de_l_ecriture(self):
        evts, done = self.tour_streame(
            fragments=["Bonne ", "idée. ", "Le matin ", "te laisse ", "plus d'élan. "])
        # Deux phrases, deux deltas: le texte s'ecrit au lieu d'arriver en bloc.
        self.assertEqual(self.deltas_de(evts),
                         ["Bonne idée.", " Le matin te laisse plus d'élan."])
        # Et ces deltas partent AVANT la fin du tour.
        self.assertLess(evts.index(next(e for e in evts if e["type"] == "delta")),
                        len(evts) - 1)
        self.assertEqual(done["response"], "Bonne idée. Le matin te laisse plus d'élan.")

    def test_une_phrase_qui_affirme_une_action_ne_s_affiche_jamais(self):
        evts, done = self.tour_streame(
            fragments=["J'ai ajouté ton gym jeudi. ", "Bonne séance ! "],
            actions=[ok("create_block", created=[{"title": "Gym"}])])
        # Le mensonge potentiel ne part pas, meme une seconde; le reste passe,
        # et sans espace parasite en tete puisque rien ne le precede.
        self.assertEqual(self.deltas_de(evts), ["Bonne séance !"])
        self.assertNotIn("J'ai ajout", "".join(self.deltas_de(evts)))
        # done.response porte les faits du registre, jamais l'affirmation.
        self.assertIn("FAITS", done["response"])
        self.assertNotIn("J'ai ajout", done["response"])

    def test_une_question_attend_la_fin_du_tour(self):
        evts, done = self.tour_streame(
            fragments=["Parfait. ", "Tu préfères le matin ou le soir ? "])
        # La question n'est pas streamee: sa place est la fin, et le code
        # peut encore la remplacer par la sienne.
        self.assertEqual(self.deltas_de(evts), ["Parfait."])
        self.assertTrue(done["response"].endswith("Tu préfères le matin ou le soir ?"))

    def test_sous_une_garde_rien_ne_s_affiche_en_apercu(self):
        evts, done = self.tour_streame(
            fragments=["Je peux le faire. ", "Voyons ensemble. "],
            actions=[refus("delete_block", demande=demande("portee_jour", "p1"))])
        # La question du code va gagner et tailler la prose: on n'ecrit rien
        # qu'il faudrait reprendre. Le tour retombe sur les sections finales,
        # qui ne portent que la question du code.
        self.assertNotIn("Je peux le faire", "".join(self.deltas_de(evts)))
        self.assertNotIn("Voyons ensemble", done["response"])
        self.assertEqual(done["question_motif"], "portee_jour")

    def test_une_phrase_incomplete_attend_sa_ponctuation(self):
        agent = PlannerAgentV2()
        agent._registre_courant = Registre()
        agent._file_pensees = queue.Queue()
        agent._apercu_actif, agent._apercu_tampon, agent._apercu_emis = True, "", False
        agent.pousser_brouillon("Je regarde ça")
        self.assertTrue(agent._file_pensees.empty())
        self.assertFalse(agent._apercu_emis)
        # La ponctuation ET son espace arrivent: la phrase part alors.
        agent.pousser_brouillon(" tout de suite. ")
        self.assertEqual(agent._file_pensees.get_nowait(),
                         ("delta", "Je regarde ça tout de suite."))

    def test_flag_eteint_les_sections_partent_comme_avant(self):
        profil = self.user.profile
        profil.voix_agir = False
        profil.save(update_fields=["voix_agir"])
        evts, done = self.tour()  # harnais NarrateurBase: DIRE simule dit « Ok. »
        self.assertEqual(self.deltas_de(evts), ["Ok."])
        self.assertEqual(done["response"], "Ok.")

    def test_la_reponse_finale_reste_la_verite_canonique(self):
        # Faits d'abord, prose ensuite: l'ordre de done.response ne bouge
        # pas, meme quand l'apercu a ecrit la prose en premier.
        _, done = self.tour_streame(
            fragments=["Bonne séance ! "],
            actions=[ok("create_block", created=[{"title": "Gym"}])])
        self.assertEqual(done["response"], "FAITS\n\nBonne séance !")
