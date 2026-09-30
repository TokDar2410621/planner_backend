"""La couche de jugement (services/agent_v2/jugement.py).

Des decisions typees au lieu des regex d'intention: Jev en premier, repli
LLM en sortie structuree quand la cle manque, indisponible sinon. Un echec
ne vaut jamais une approbation; sur une action destructive, le doute impose
le comportement prudent.
"""
import json
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, override_settings

from services.agent_v2 import demandes as dem
from services.agent_v2 import jugement


def _rep_jev(answers: dict):
    """Une reponse requests.post simulee."""
    rep = MagicMock()
    rep.json.return_value = {"answers": answers}
    rep.raise_for_status.return_value = None
    return rep


def _choice(valeur, confiance=0.9):
    return {"choice": valeur, "confidence": confiance,
            "probabilities": {valeur: confiance}}


def _noul(p, confiance=None):
    rep = {"noul": p}
    if confiance is not None:
        rep["confidence"] = confiance
    return rep


PORTEE = {'motif': 'portee_jour', 'cle': 'p', 'outil': 'delete_block',
          'question': 'Supprimer le Quart au dépanneur : seulement jeudi, ou tous les jeudis ?',
          'cible': {'titre': 'Quart au dépanneur', 'jour': 3, 'date': '2026-09-17'},
          'options': [{'id': 'occurrence'}, {'id': 'serie'}, {'id': 'annuler'}]}
DESTR = {'motif': 'destructif', 'cle': 'd', 'outil': 'delete_block',
         'question': 'Supprimer le Quart au dépanneur ?',
         'cible': {'titre': 'Quart au dépanneur', 'jour': 3},
         'options': [{'id': 'confirmer'}, {'id': 'annuler'}]}
MASSE = {'motif': 'creation_en_masse', 'cle': 'm', 'outil': 'create_block',
         'question': 'Continuer les ajouts en série ?',
         'cible': {'titre': 'Yoga'},
         'options': [{'id': 'confirmer'}, {'id': 'annuler'}]}


def _script(reponses: dict):
    """Un juger scripte: {qid: (valeur, confiance)} -> resultats types."""
    def faux_juger(etat, questions):
        return {qid: {"valeur": v, "confiance": c, "probabilites": None,
                      "statut": ("decision" if c >= 0.8 else "incertain")}
                for qid, (v, c) in reponses.items()}
    return faux_juger


def juger_script(par_message: dict):
    """Un faux `jugement.juger` scripte par message.

    par_message: {message: {qid: (valeur, confiance)}}. Pour les questions
    choice, la valeur scriptee doit etre un ID d'option du contrat (comme un
    vrai juge la rendrait). Les qids non scriptes rendent indisponible:
    jamais d'approbation par defaut. Les cles sont comparees apres strip
    (les gardes decoupent parfois le message). Partage avec les autres
    modules de tests qui rejouent des tours avec un juge fige.
    """
    def faux(etat, questions):
        message = etat.get("message") if isinstance(etat, dict) else etat
        if isinstance(message, str):
            message = message.strip()
        cas = par_message.get(message, {})
        sortie = {}
        for qid in questions:
            if qid not in cas:
                sortie[qid] = {"valeur": None, "confiance": 0.0,
                               "probabilites": None, "statut": "indisponible"}
                continue
            valeur, confiance = cas[qid]
            sortie[qid] = {"valeur": valeur, "confiance": float(confiance),
                           "probabilites": None,
                           "statut": ("decision" if confiance >= 0.8
                                      else "incertain")}
        return sortie
    return faux


class JugementJevTests(SimpleTestCase):
    """Le contrat Jev, valide strictement au bord reseau."""

    def setUp(self):
        jugement.vider_cache()

    def _juger_jev(self, questions, answers):
        with override_settings(JEV_API_KEY="cle-test"):
            with patch("requests.post", return_value=_rep_jev(answers)) as post:
                res = jugement.juger({"message": "x"}, questions)
        return res, post

    def test_choice_valide_devient_decision(self):
        res, _ = self._juger_jev(
            {"portee": jugement.q_portee()},
            {"portee": _choice("serie", 0.9)})
        self.assertEqual(res["portee"]["valeur"], "serie")
        self.assertEqual(res["portee"]["statut"], "decision")
        self.assertGreaterEqual(res["portee"]["confiance"], 0.8)

    def test_noul_valide(self):
        res, _ = self._juger_jev(
            {"suppression": jugement.q_suppression()},
            {"suppression": _noul(0.92)})
        self.assertTrue(res["suppression"]["valeur"])
        self.assertEqual(res["suppression"]["statut"], "decision")

    def test_probabilite_hors_bornes_rejetee(self):
        res, _ = self._juger_jev(
            {"portee": jugement.q_portee()},
            {"portee": {"choice": "serie", "confidence": 0.9,
                        "probabilities": {"serie": 1.5}}})
        self.assertEqual(res["portee"]["statut"], "indisponible")

    def test_option_inconnue_rejetee(self):
        res, _ = self._juger_jev(
            {"portee": jugement.q_portee()},
            {"portee": _choice("pluie", 0.9)})
        self.assertEqual(res["portee"]["statut"], "indisponible")

    def test_question_manquante_rejetee(self):
        res, _ = self._juger_jev(
            {"portee": jugement.q_portee()}, {})
        self.assertEqual(res["portee"]["statut"], "indisponible")

    def test_type_de_reponse_incompatible_rejete(self):
        res, _ = self._juger_jev(
            {"portee": jugement.q_portee()},
            {"portee": _noul(0.9)})
        self.assertEqual(res["portee"]["statut"], "indisponible")

    def test_timeout_rend_indisponible(self):
        import requests
        with override_settings(JEV_API_KEY="cle-test"):
            with patch("requests.post",
                       side_effect=requests.Timeout("trop lent")):
                with patch("services.agent_v2.jugement._juger_llm",
                           return_value=None):
                    res = jugement.juger("supprime tout",
                                         {"s": jugement.q_suppression()})
        self.assertEqual(res["s"]["statut"], "indisponible")

    def test_http_500_rend_indisponible(self):
        import requests
        rep = MagicMock()
        rep.raise_for_status.side_effect = requests.HTTPError("500")
        with override_settings(JEV_API_KEY="cle-test"):
            with patch("requests.post", return_value=rep):
                with patch("services.agent_v2.jugement._juger_llm",
                           return_value=None):
                    res = jugement.juger("supprime tout",
                                         {"s": jugement.q_suppression()})
        self.assertEqual(res["s"]["statut"], "indisponible")

    def test_cle_jamais_journalisee_ni_dans_le_corps(self):
        import requests
        with override_settings(JEV_API_KEY="cle-ultra-secrete"):
            with patch("requests.post",
                       side_effect=requests.Timeout("boom")) as post:
                with self.assertLogs("services.agent_v2.jugement",
                                     level="WARNING") as logs:
                    with patch("services.agent_v2.jugement._juger_llm",
                               return_value=None):
                        jugement.juger("supprime tout",
                                       {"s": jugement.q_suppression()})
        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer cle-ultra-secrete")
        corps = json.dumps(post.call_args.kwargs["json"])
        self.assertNotIn("cle-ultra-secrete", corps)
        for ligne in logs.output:
            self.assertNotIn("cle-ultra-secrete", ligne)

    def test_confiance_haute_decision(self):
        res, _ = self._juger_jev(
            {"portee": jugement.q_portee()},
            {"portee": _choice("serie", 0.85)})
        self.assertEqual(res["portee"]["statut"], "decision")

    def test_confiance_basse_incertain(self):
        res, _ = self._juger_jev(
            {"portee": jugement.q_portee()},
            {"portee": _choice("serie", 0.5)})
        self.assertEqual(res["portee"]["statut"], "incertain")
        self.assertEqual(res["portee"]["valeur"], "serie")

    def test_seuil_configurable(self):
        with override_settings(JUGEMENT_SEUIL_DECISION=0.95):
            res, _ = self._juger_jev(
                {"portee": jugement.q_portee()},
                {"portee": _choice("serie", 0.9)})
        self.assertEqual(res["portee"]["statut"], "incertain")


class JugementRepliTests(SimpleTestCase):
    """Sans cle Jev: le repli LLM en sortie structuree, sinon indisponible."""

    def setUp(self):
        jugement.vider_cache()

    def test_sans_cle_le_repli_llm_tranche(self):
        with override_settings(JEV_API_KEY="", JUGEMENT_REPLI_LLM="1"):
            with patch("services.agent_v2.jugement._juger_llm",
                       return_value={"s": {"valeur": True, "confiance": 0.9,
                                           "probabilites": None}}):
                res = jugement.juger("supprime mon cours",
                                     {"s": jugement.q_suppression()})
        self.assertEqual(res["s"]["statut"], "decision")
        self.assertTrue(res["s"]["valeur"])

    def test_repli_llm_invalide_rend_indisponible(self):
        with override_settings(JEV_API_KEY=""):
            with patch("services.agent_v2.jugement._juger_llm",
                       return_value=None):
                res = jugement.juger("supprime mon cours",
                                     {"s": jugement.q_suppression()})
        self.assertEqual(res["s"]["statut"], "indisponible")

    def test_jev_et_llm_hs_rend_indisponible(self):
        import requests
        with override_settings(JEV_API_KEY="cle-test"):
            with patch("requests.post",
                       side_effect=requests.ConnectionError("coupé")):
                with patch("services.agent_v2.jugement._juger_llm",
                           return_value=None):
                    res = jugement.juger("supprime tout",
                                         {"s": jugement.q_suppression()})
        self.assertEqual(res["s"]["statut"], "indisponible")
        self.assertIsNone(res["s"]["valeur"])

    def test_juger_ne_leve_jamais(self):
        with override_settings(JEV_API_KEY="cle-test"):
            with patch("requests.post", side_effect=RuntimeError("inattendu")):
                with patch("services.agent_v2.jugement._juger_llm",
                           side_effect=RuntimeError("inattendu")):
                    res = jugement.juger("x", {"s": jugement.q_suppression()})
        self.assertEqual(res["s"]["statut"], "indisponible")


class JugementPrudenceTests(SimpleTestCase):
    """Un echec ne vaut jamais une approbation; le doute est prudent."""

    def _indisponible(self, etat, questions):
        return {qid: {"valeur": None, "confiance": 0.0, "probabilites": None,
                      "statut": "indisponible"}
                for qid in questions}

    def test_indisponible_jamais_approbation(self):
        with patch("services.agent_v2.jugement.juger", self._indisponible):
            # Aucune option ne sort d'un juge muet, meme sur un oui franc.
            self.assertIsNone(dem.option_choisie("oui", DESTR))
            self.assertIsNone(dem.option_choisie("oui, je confirme", MASSE))
            self.assertFalse(dem.annulation_libre("garde", DESTR))
            self.assertEqual(dem.classification_reponse("oui", DESTR),
                             "incertain")
            self.assertTrue(dem.reponse_plausible("oui", DESTR))

    def test_doute_destructif_toujours_prudent(self):
        with patch("services.agent_v2.jugement.juger", self._indisponible):
            # Le doute se traite comme une suppression possible, un jour
            # vise, un saut suspect: jamais comme un feu vert.
            self.assertTrue(dem.suppression_demandee("bonjour"))
            self.assertTrue(dem.jour_vise("bonjour"))
            self.assertTrue(dem.saut_suspect("saute mon gym demain"))
            visee, tranchee = dem.jour_vise_tranche("bonjour")
            self.assertFalse(tranchee)


class JugementMappingTests(SimpleTestCase):
    """Le code applique correctement les decisions typees du juge."""

    def test_refuse_ferme_sur_annuler(self):
        with patch("services.agent_v2.jugement.juger",
                   _script({"intention": ("refuse", 0.9)})):
            self.assertTrue(dem.annulation_libre("garde", PORTEE))
            self.assertEqual(dem.option_choisie("non, garde-le", PORTEE),
                             "annuler")

    def test_precise_portee_ne_tranche_jamais_en_texte_libre(self):
        # D1: la portee ne se lit que sur la puce exacte, meme quand le juge
        # est sur. « seulement jeudi » repose la question.
        with patch("services.agent_v2.jugement.juger",
                   _script({"intention": ("precise", 0.95)})):
            self.assertIsNone(dem.option_choisie("seulement jeudi", PORTEE))
            self.assertEqual(dem.classification_reponse("seulement jeudi",
                                                       PORTEE),
                             "reponse")

    def test_accepte_confirme_masse_mais_jamais_destructif(self):
        with patch("services.agent_v2.jugement.juger",
                   _script({"intention": ("accepte", 0.9)})):
            self.assertEqual(
                dem.option_choisie("d'accord, on continue comme ça", MASSE),
                "confirmer")
            # Une suppression ne se confirme jamais en texte libre.
            self.assertIsNone(dem.option_choisie("oui, vas-y", DESTR))

    def test_precise_choisit_une_option_sans_effet(self):
        libre = {'motif': 'question_libre', 'cle': 'q', 'outil': '',
                 'question': 'Tu préfères le matin ou le soir ?',
                 'cible': {},
                 'options': [{'id': 'matin', 'libelle': 'Le matin'},
                             {'id': 'soir', 'libelle': 'Le soir'},
                             {'id': 'annuler', 'libelle': 'Annuler'}]}
        with patch("services.agent_v2.jugement.juger",
                   _script({"intention": ("precise", 0.9),
                            "choix": ("soir", 0.92)})):
            self.assertEqual(dem.option_choisie("le soir", libre), "soir")
        with patch("services.agent_v2.jugement.juger",
                   _script({"intention": ("refuse", 0.9)})):
            self.assertIsNone(dem.option_choisie("aucun des deux", libre))

    def test_nouvelle_requete_abandonne(self):
        with patch("services.agent_v2.jugement.juger",
                   _script({"intention": ("nouvelle_requete", 0.9)})):
            self.assertEqual(
                dem.classification_reponse("c'est quoi mon horaire demain ?",
                                           PORTEE),
                "nouvelle_requete")
            self.assertFalse(dem.reponse_plausible(
                "c'est quoi mon horaire demain ?", PORTEE))

    def test_incertain_repose_au_lieu_d_abandonner(self):
        with patch("services.agent_v2.jugement.juger",
                   _script({"intention": ("incertain", 0.5)})):
            self.assertEqual(dem.classification_reponse("euh", PORTEE),
                             "incertain")
            self.assertTrue(dem.reponse_plausible("euh", PORTEE))
            self.assertIsNone(dem.option_choisie("euh", PORTEE))

    def test_annuler_evenement_ne_se_lit_pas_en_texte_libre(self):
        annulation = {'motif': 'destructif', 'cle': 'c',
                      'outil': 'cancel_scheduled_block',
                      'question': 'Annuler le Dentiste ?',
                      'cible': {'titre': 'Dentiste', 'date': '2026-09-16'},
                      'options': [{'id': 'confirmer'}, {'id': 'annuler'}]}
        with patch("services.agent_v2.jugement.juger",
                   _script({"intention": ("accepte", 0.95)})):
            # « annule » face a « J'annule le dentiste ? » repose la question.
            self.assertIsNone(dem.option_choisie("annule", annulation))
            self.assertFalse(dem.annulation_libre("annule", annulation))
