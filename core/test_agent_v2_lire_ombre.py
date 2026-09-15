"""
LIRE en mode ombre: une lecture typee du message tape qui MESURE et ne decide
rien (services/agent_v2/lecture.py, lecture_schema.py).

Pourquoi ces tests. Les gardes de v2 lisent le francais brut par des regex, et
une faute (« aujourdui ») saute une garantie sans bruit. Avant qu'une regle ne
s'appuie sur la lecture typee, elle tourne en production pour compter ses
desaccords avec les lecteurs regex geles. Ce mode n'est acceptable que s'il
tient cinq promesses, verrouillees ici:

1. Le tour est identique octet pour octet, LIRE actif ou coupe.
2. Chaque statut sort tel que prevu, et un credit DeepSeek mort se voit.
3. Aucun ORM dans le pool LIRE, qui est dedie et n'affame pas AGIR.
4. Aucune donnee de l'utilisateur dans les journaux.
5. Les comparaisons avec les lecteurs geles sont des fonctions pures, et les
   dates typees y passent par le resolveur du code (une date, ou une question).

Aucun appel reseau: les fournisseurs sont des FunctionModel de pydantic-ai,
qui passent par la vraie validation de sortie.
"""
import asyncio
import json
import re
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from datetime import date, time as heure, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth.models import User
from django.db import connections
from django.db.backends.utils import CursorWrapper
from django.test import SimpleTestCase, TransactionTestCase, override_settings
from django.utils import timezone
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from core.models import ConversationMessage, RecurringBlock
from core.serializers import ConversationMessageSerializer
from services.agent_v2 import demandes as dem
from services.agent_v2 import lecture
from services.agent_v2 import lecture_schema as schema

LUNDI = date(2026, 9, 14)
JOURNAL = "services.agent_v2.lecture"
TIRET_LONG, DEMI_CADRATIN = chr(0x2014), chr(0x2013)
LIGNE_TOUR = re.compile(
    r"^agent_v2 lire statut=[a-z_]+ fournisseur=[a-z0-9.\-]+ ms=\d+ accords="
    r"heures:(ok|diff|na),dates:(ok|diff|na),jour_vise:(ok|diff|na),date_visee:(ok|diff|na),"
    r"suppression:(ok|diff|na),evenement_unique:(ok|diff|na),cette_semaine:(ok|diff|na),"
    r"puces_date:(ok|diff|na) attente=\d+ rejets=\d+ erreur=[A-Za-z0-9:_\-]+$")


# ------------------------------------------------------------------ fabriques

def _element(**champs):
    base = {"mention": "", "polarite": "demande", "operation": "ajouter", "genre": "other"}
    base.update(champs)
    return base


def _ref(extrait, sorte, genre="placement", **champs):
    """Une ReferenceJour: le modele dit le jour, le code calcule la date."""
    base = {"extrait": extrait, "genre": genre, "sorte": sorte}
    base.update(champs)
    return base


def _heure(extrait, *lectures, role="debut", genre="ferme"):
    return {"extrait": extrait, "role": role, "genre": genre, "lectures": list(lectures)}


def _lecture(*elements, reponses=()):
    return schema.LectureTour.model_validate(
        {"elements": list(elements), "reponses": list(reponses)})


def _prep(message, modeles=("deepseek-flash", "gemini-2.5-flash"), semaine=(), aujourdhui=LUNDI):
    texte, refs = schema.contexte_lire(aujourdhui, "10:42", "tape", list(semaine), [], [], [], "")
    return lecture.Preparation(message=message, aujourdhui=aujourdhui, contexte=texte,
                               refs=refs, modeles=tuple(modeles))


def _lecteur(args=None, *, attente=0.0, erreur=None, appels=None):
    """Un fournisseur simule: rend l'appel d'outil de sortie, ou leve."""
    async def repondre(messages, info: AgentInfo):
        if appels is not None:
            appels.append(threading.current_thread().name)
        if attente:
            await asyncio.sleep(attente)
        if erreur is not None:
            raise erreur
        return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=args)])
    return FunctionModel(repondre)


def _fournisseurs(modeles: dict):
    return patch.object(lecture, "_modele", side_effect=lambda nom: modeles.get(nom))


GYM_JEUDI = {"elements": [_element(
    mention="gym", genre="sport",
    dates=[_ref("jeudi", "jour_semaine", jour_semaine="jeudi")],
    heures=[_heure("18h", "18:00")], recurrence="unique")], "reponses": []}


# ------------------------------------------------------------ schema et contrat

class SchemaTests(SimpleTestCase):
    def test_l_enveloppe_arguments_est_refusee(self):
        with self.assertRaises(ValidationError):
            schema.LectureTour.model_validate({"arguments": GYM_JEUDI})

    def test_ni_tiret_long_ni_demi_cadratin_dans_le_prompt_et_le_schema(self):
        expose = json.dumps(schema.LectureTour.model_json_schema(), ensure_ascii=False)
        for texte in (schema.PROMPT_LIRE, expose):
            self.assertNotIn(TIRET_LONG, texte)
            self.assertNotIn(DEMI_CADRATIN, texte)

    def test_le_resolveur_passe_ses_propres_controles(self):
        self.assertEqual(schema.verifier_resolveur(), [])

    def test_un_jour_non_precise_encore_a_venir_vaut_une_question(self):
        e = _lecture(_element(dates=[_ref("jeudi", "jour_semaine", jour_semaine="jeudi")])).elements[0]
        self.assertEqual(schema.jours_resolus(e, LUNDI), [("placement", schema.QUESTION)])

    def test_un_jour_deja_passe_donne_la_prochaine_occurrence(self):
        e = _lecture(_element(dates=[_ref("lundi", "jour_semaine", jour_semaine="lundi")])).elements[0]
        self.assertEqual(schema.jours_resolus(e, date(2026, 9, 18)), [("placement", date(2026, 9, 21))])

    def test_la_semaine_dite_tranche(self):
        e = _lecture(_element(dates=[_ref("jeudi prochain", "jour_semaine", jour_semaine="jeudi",
                                          semaine="semaine_prochaine")])).elements[0]
        self.assertEqual(schema.jours_resolus(e, LUNDI), [("placement", date(2026, 9, 24))])

    def test_une_periode_rend_ses_deux_bornes(self):
        e = _lecture(_element(dates=[_ref("cette semaine", "periode", "fenetre_debut",
                                          periode="semaine", semaine="cette_semaine")])).elements[0]
        self.assertEqual(schema.jours_resolus(e, LUNDI),
                         [("fenetre_debut", date(2026, 9, 14)), ("fenetre_fin", date(2026, 9, 20))])

    def test_une_reference_irresolvable_est_ignoree(self):
        e = _lecture(_element(dates=[_ref("bientot", "delai")])).elements[0]
        self.assertEqual(schema.jours_resolus(e, LUNDI), [])

    def test_les_jours_d_une_habitude_sont_numerotes_depuis_lundi(self):
        e = _lecture(_element(jours_semaine=[{"extrait": "jeudis", "jour": "jeudi"},
                                             {"extrait": "lundis", "jour": "lundi"}])).elements[0]
        self.assertEqual(schema.jours_de_semaine(e), [0, 3])

    def test_les_cles_internes_sont_celles_que_l_api_retire(self):
        self.assertEqual(tuple(ConversationMessageSerializer.METADONNEES_INTERNES),
                         lecture.CLES_METADONNEES)


# ------------------------------------------------------------------ reglages

class ReglagesTests(SimpleTestCase):
    def test_actif_quand_la_variable_est_absente(self):
        with patch.object(lecture, "settings", SimpleNamespace()):
            self.assertTrue(lecture.ombre_active())
            self.assertEqual(lecture.modeles_lire(), lecture.MODELES_DEFAUT)
            self.assertEqual(lecture.attente_s(), 1.5)

    def test_l_interrupteur(self):
        for valeur, attendu in (("0", False), ("false", False), (" OFF ", False), (False, False),
                                ("1", True), ("", True), ("true", True), (True, True)):
            with override_settings(LIRE_OMBRE=valeur):
                self.assertEqual(lecture.ombre_active(), attendu, valeur)

    def test_une_attente_mal_formee_ne_casse_rien(self):
        for valeur, attendu in (("abc", 1.5), ("nan", 1.5), ("-3", 0.0), ("99", 10.0), ("0.4", 0.4)):
            with override_settings(LIRE_ATTENTE_S=valeur):
                self.assertEqual(lecture.attente_s(), attendu, valeur)

    def test_les_modeles_viennent_des_settings(self):
        with override_settings(LIRE_MODELES=" deepseek-flash , gemini-2.5-flash,"):
            self.assertEqual(lecture.modeles_lire(), ("deepseek-flash", "gemini-2.5-flash"))

    def test_gemini_porte_son_delai_dans_les_reglages_et_deepseek_coupe_la_reflexion(self):
        """Sonde 2: google-genai ignore le delai du client httpx; DeepSeek rend
        HTTP 400 sur la sortie outil forcee quand la reflexion est active."""
        self.assertEqual(schema.reglages_lire("gemini-2.5-flash", 6.0)["timeout"], 6.0)
        self.assertEqual(
            schema.reglages_lire("gemini-2.5-flash", 6.0)["google_thinking_config"]["thinking_budget"], 0)
        self.assertEqual(
            schema.reglages_lire("deepseek-flash", 6.0)["extra_body"]["thinking"]["type"], "disabled")

    @override_settings(DEEPSEEK_API_KEY="x", GEMINI_API_KEY="y")
    def test_le_client_deepseek_est_borne_et_sans_relance(self):
        client = lecture._modele("deepseek-flash")._provider.client
        self.assertEqual(client.max_retries, 0)
        self.assertLessEqual(float(getattr(client.timeout, "read", client.timeout)), lecture.DELAI_LIRE)
        self.assertIsNone(lecture._modele("inconnu"))


# ---------------------------------------------------------------- les statuts

class StatutsTests(SimpleTestCase):
    """lire() tourne dans le pool: SimpleTestCase y interdit la base, ce qui
    prouve au passage qu'il n'en touche rien."""

    def test_ok(self):
        with _fournisseurs({"deepseek-flash": _lecteur(GYM_JEUDI)}):
            r = lecture.lire(_prep("ajoute gym jeudi a 18h"), time.perf_counter())
        self.assertEqual((r.statut, r.fournisseur, r.rejets), ("ok", "deepseek-flash", 0))
        meta = r.metadonnees()
        self.assertEqual(meta["lecture"]["elements"][0]["dates"][0]["jour_semaine"], "jeudi")
        self.assertEqual(json.loads(json.dumps(meta)), meta)

    def test_partielle_quand_l_ancrage_retire_un_champ(self):
        args = json.loads(json.dumps(GYM_JEUDI))
        args["elements"][0]["dates"][0]["extrait"] = "vendredi"
        with _fournisseurs({"deepseek-flash": _lecteur(args)}):
            r = lecture.lire(_prep("ajoute gym jeudi a 18h"))
        self.assertEqual((r.statut, r.rejets), ("partielle", 1))
        self.assertEqual(r.lecture.elements[0].dates, [])

    def test_l_enveloppe_est_absente_sans_repli(self):
        appels_gemini: list = []
        with _fournisseurs({"deepseek-flash": _lecteur({"arguments": GYM_JEUDI}),
                            "gemini-2.5-flash": _lecteur(GYM_JEUDI, appels=appels_gemini)}):
            r = lecture.lire(_prep("ajoute gym jeudi a 18h"))
        self.assertEqual((r.statut, r.fournisseur), ("absente", "deepseek-flash"))
        self.assertTrue(r.erreur.startswith("UnexpectedModelBehavior"), r.erreur)
        self.assertIsNone(r.metadonnees()["lecture"])
        self.assertEqual(appels_gemini, [])

    def test_une_lecture_vide_sur_un_message_qui_dit_quelque_chose_est_absente(self):
        vide = {"elements": [], "reponses": []}
        with _fournisseurs({"deepseek-flash": _lecteur(vide)}):
            r = lecture.lire(_prep("ajoute gym jeudi a 18h"))
            r_blanc = lecture.lire(_prep("   "))
        self.assertEqual((r.statut, r.erreur), ("absente", "lecture_vide"))
        self.assertEqual(r_blanc.statut, "ok")

    def test_402_alerte_et_passe_au_repli(self):
        with _fournisseurs({"deepseek-flash": _lecteur(erreur=ModelHTTPError(402, "deepseek-flash")),
                            "gemini-2.5-flash": _lecteur(GYM_JEUDI)}), \
                self.assertLogs(JOURNAL, "WARNING") as journal:
            r = lecture.lire(_prep("ajoute gym jeudi a 18h"))
        self.assertEqual((r.statut, r.fournisseur), ("ok", "gemini-2.5-flash"))
        self.assertTrue(any("http=402" in l and "deepseek-flash" in l for l in journal.output))

    def test_tous_en_panne_rend_erreur_avec_la_classe(self):
        with _fournisseurs({"deepseek-flash": _lecteur(erreur=ModelHTTPError(401, "deepseek-flash")),
                            "gemini-2.5-flash": _lecteur(erreur=ModelHTTPError(503, "gemini-2.5-flash"))}), \
                self.assertLogs(JOURNAL, "WARNING") as journal:
            r = lecture.lire(_prep("ajoute gym jeudi a 18h"))
        self.assertEqual((r.statut, r.erreur, r.fournisseur), ("erreur", "ModelHTTPError:401", ""))
        self.assertTrue(any("http=401" in l for l in journal.output))

    def test_une_panne_hors_credit_n_alerte_pas(self):
        with _fournisseurs({"deepseek-flash": _lecteur(erreur=ModelHTTPError(503, "deepseek-flash"))}), \
                self.assertNoLogs(JOURNAL, "WARNING"):
            r = lecture.lire(_prep("ajoute gym jeudi a 18h"))
        self.assertEqual((r.statut, r.erreur), ("erreur", "ModelHTTPError:503"))

    @override_settings(DEEPSEEK_API_KEY="", GEMINI_API_KEY="")
    def test_sans_cle_aucun_fournisseur(self):
        r = lecture.lire(_prep("ajoute gym jeudi a 18h"))
        self.assertEqual((r.statut, r.erreur), ("erreur", "AucunFournisseur"))

    def test_hors_budget_puis_journal_tardif_sans_texte(self):
        prep = _prep("ajoute zorglub jeudi a 18h")
        args = json.loads(json.dumps(GYM_JEUDI).replace('"gym"', '"zorglub"'))
        with _fournisseurs({"deepseek-flash": _lecteur(args, attente=0.5)}), \
                override_settings(LIRE_ATTENTE_S="0.05"), \
                self.assertLogs(JOURNAL, "INFO") as journal:
            depart = time.perf_counter()
            suivi = lecture.Suivi(preparation=prep, futur=lecture._POOL_LIRE.submit(lecture.lire, prep, depart),
                                  depart=depart)
            r = lecture.conclure(suivi)
            self.assertEqual(r.statut, "hors_budget")
            self.assertLess(time.perf_counter() - depart, 0.4)
            limite = time.perf_counter() + 5
            while not any("tardive" in l for l in journal.output) and time.perf_counter() < limite:
                time.sleep(0.02)
        tardives = [l for l in journal.output if "tardive" in l]
        self.assertEqual(len(tardives), 1)
        self.assertIn("statut=ok", tardives[0])
        self.assertNotIn("zorglub", " ".join(journal.output))

    def test_une_lecture_encore_en_file_est_annulee(self):
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lire_test")
        verrou = threading.Event()
        try:
            pool.submit(verrou.wait, 5)
            with patch.object(lecture, "_modele") as modele, override_settings(LIRE_ATTENTE_S="0.05"):
                prep = _prep("ajoute gym jeudi a 18h")
                suivi = lecture.Suivi(preparation=prep, futur=pool.submit(lecture.lire, prep), depart=1.0)
                r = lecture.conclure(suivi)
                verrou.set()
                self.assertTrue(suivi.futur.cancelled())
                modele.assert_not_called()
            self.assertEqual(r.statut, "hors_budget")
        finally:
            verrou.set()
            pool.shutdown(wait=True)

    @override_settings(LIRE_OMBRE="0")
    def test_desactivee_ne_soumet_rien(self):
        with patch.object(lecture, "preparer") as preparer, patch.object(lecture, "_POOL_LIRE") as pool:
            suivi = lecture.demarrer(None, "ajoute gym jeudi a 18h")
        preparer.assert_not_called()
        pool.submit.assert_not_called()
        self.assertEqual(lecture.finir(suivi),
                         {"lecture": None, "lecture_statut": "desactivee",
                          "lecture_fournisseur": None, "lecture_ms": 0})

    def test_finir_ne_leve_jamais(self):
        with patch.object(lecture, "conclure", side_effect=RuntimeError("boum")), \
                self.assertLogs(JOURNAL, "WARNING"):
            meta = lecture.finir(lecture.sautee())
        self.assertEqual(meta["lecture_statut"], "erreur")


# ------------------------------------------------ accords avec les lecteurs geles

SEMAINE_GYM_MERCREDI = [{"titre": "Gym", "jours": [2], "debut": "18:00", "fin": "19:00", "type": "sport"}]


class AccordsTests(SimpleTestCase):
    """LUNDI est un lundi: un jeudi ou un mercredi non precise est encore a
    venir, donc le resolveur demande entre deux dates."""

    def test_aujourdui_regex_muette_lecture_typee_lue(self):
        """Le cas fondateur A: la faute fait taire les deux lecteurs geles."""
        message = "planifie revision aujourdui de 14h a 16h"
        lu = _lecture(_element(mention="revision", genre="revision",
                               dates=[_ref("aujourdui", "jour_relatif", jour_relatif="aujourdhui")],
                               heures=[_heure("14h", "14:00"), _heure("16h", "16:00", role="fin")]))
        lu, rejets = schema.ancrer(lu, message, LUNDI, _prep(message).refs)
        self.assertEqual(rejets, [])
        self.assertEqual(schema.jours_resolus(lu.elements[0], LUNDI), [("placement", LUNDI)])
        self.assertEqual(dem._dates_nommees(message, LUNDI), [])
        self.assertIsNone(lecture.date_puces_v1(message, LUNDI))
        self.assertEqual(lecture.accord_dates(lu, message, LUNDI), "diff")
        self.assertEqual(lecture.accord_puces_date(lu, message, LUNDI), "diff")
        self.assertEqual(lecture.accord_heures(lu, message), "ok")

        juste = "planifie revision aujourd'hui de 14h a 16h"
        lu_juste = _lecture(_element(dates=[_ref("aujourd'hui", "jour_relatif", jour_relatif="aujourdhui")]))
        self.assertEqual(lecture.accord_dates(lu_juste, juste, LUNDI), "ok")
        self.assertEqual(lecture.accord_puces_date(lu_juste, juste, LUNDI), "ok")

    def test_heures(self):
        deux = _lecture(_element(heures=[_heure("7h", "07:00", "19:00")]))
        self.assertEqual(lecture.accord_heures(deux, "gym a 7h"), "ok")
        une_fausse = _lecture(_element(heures=[_heure("7h", "19:00")]))
        self.assertEqual(lecture.accord_heures(une_fausse, "gym a 7h"), "diff")
        pm = _lecture(_element(heures=[_heure("3pm", "15:00")]))
        self.assertEqual(lecture.accord_heures(pm, "gym a 3pm"), "diff")
        self.assertEqual(lecture.accord_heures(_lecture(_element()), "salut"), "na")

    def test_dates_contre_la_sortie_du_resolveur(self):
        # La regex tranche « jeudi » en silence la ou le code demanderait.
        ouvert = _lecture(_element(dates=[_ref("jeudi", "jour_semaine", jour_semaine="jeudi")]))
        self.assertEqual(lecture.accord_dates(ouvert, "gym jeudi", LUNDI), "diff")
        prochain = _lecture(_element(dates=[_ref("jeudi prochain", "jour_semaine", jour_semaine="jeudi",
                                                 semaine="semaine_prochaine")]))
        self.assertEqual(lecture.accord_dates(prochain, "gym jeudi prochain", LUNDI), "diff")
        ce_jeudi = _lecture(_element(dates=[_ref("ce jeudi", "jour_semaine", jour_semaine="jeudi",
                                                 semaine="cette_semaine")]))
        self.assertEqual(lecture.accord_dates(ce_jeudi, "gym ce jeudi", LUNDI), "ok")
        fenetre = _lecture(_element(dates=[_ref("cette semaine", "periode", "fenetre_debut",
                                                periode="semaine", semaine="cette_semaine")]))
        self.assertEqual(lecture.accord_dates(fenetre, "gym cette semaine", LUNDI), "na")
        habitude = _lecture(_element(jours_semaine=[{"extrait": "jeudis", "jour": "jeudi"}]))
        self.assertEqual(lecture.accord_dates(habitude, "gym tous les jeudis", LUNDI), "ok")

    def test_suppression(self):
        retrait = _lecture(_element(operation="supprimer"))
        self.assertEqual(lecture.accord_suppression(retrait, "suprime mon gym"), "diff")
        self.assertEqual(lecture.accord_suppression(retrait, "supprime mon gym"), "ok")
        self.assertEqual(lecture.accord_suppression(_lecture(_element()), "ajoute gym"), "na")

    def test_jour_vise_ne_vaut_que_pour_un_retrait(self):
        faute = _lecture(_element(operation="supprimer",
                                  dates=[_ref("mercedi", "jour_semaine", jour_semaine="mercredi")]))
        self.assertEqual(lecture.accord_jour_vise(faute, "efface mon gym de mercedi", LUNDI), "diff")
        juste = _lecture(_element(operation="supprimer",
                                  dates=[_ref("mercredi", "jour_semaine", jour_semaine="mercredi")]))
        self.assertEqual(lecture.accord_jour_vise(juste, "efface mon gym de mercredi", LUNDI), "ok")
        ajout = _lecture(_element(dates=[_ref("mercredi", "jour_semaine", jour_semaine="mercredi")]))
        self.assertEqual(lecture.accord_jour_vise(ajout, "ajoute gym mercredi", LUNDI), "na")

    def test_date_visee(self):
        refs = _prep("x", semaine=SEMAINE_GYM_MERCREDI).refs
        tranche = _lecture(_element(operation="supprimer", candidats=["s1"], dates=[
            _ref("ce mercredi", "jour_semaine", "occurrence_visee", jour_semaine="mercredi",
                 semaine="cette_semaine")]))
        self.assertEqual(lecture.accord_date_visee(tranche, "efface mon gym de ce mercredi", LUNDI, refs), "ok")
        ouverte = _lecture(_element(operation="supprimer", candidats=["s1"], dates=[
            _ref("mercredi", "jour_semaine", "occurrence_visee", jour_semaine="mercredi")]))
        self.assertEqual(lecture.accord_date_visee(ouverte, "efface mon gym de mercredi", LUNDI, refs), "diff")
        sans_cible = _lecture(_element(operation="supprimer"))
        self.assertEqual(lecture.accord_date_visee(sans_cible, "efface mon gym", LUNDI, refs), "na")

    def test_evenement_unique(self):
        unique = _lecture(_element(recurrence="unique"))
        self.assertEqual(lecture.accord_evenement_unique(unique, "souper jeudi soir a 6h"), "ok")
        hebdo = _lecture(_element(recurrence="hebdomadaire"))
        self.assertEqual(lecture.accord_evenement_unique(hebdo, "souper chak jeudi a 18h"), "diff")
        self.assertEqual(lecture.accord_evenement_unique(_lecture(_element()), "ajoute gym"), "na")

    def test_cette_semaine(self):
        bornee = _lecture(_element(recurrence="cette_semaine_seulement"))
        self.assertEqual(lecture.accord_cette_semaine(bornee, "gym cette semaine"), "ok")
        self.assertEqual(lecture.accord_cette_semaine(bornee, "gym cette sem"), "diff")
        hebdo = _lecture(_element(recurrence="hebdomadaire"))
        self.assertEqual(lecture.accord_cette_semaine(hebdo, "gym cette semaine et les suivantes"), "na")

    def test_le_miroir_de_la_jambe_date_des_puces_suit_l_original(self):
        """date_puces_v1 recopie la jambe date de _chips_from_message: si
        l'original bouge, cette parite casse avant la mesure."""
        from services.agent import agent as v1

        aujourdhui = timezone.localdate()
        messages = [
            "Planifie « Révision » aujourd’hui de 14h à 16h",
            "planifie revision aujourdhui de 14h a 16h",
            "planifie revision aujourdui de 14h a 16h",
            "planifie revision demain de 14h a 16h",
            "planifie revision apres-demain de 14h a 16h",
            "planifie revision le 2026-09-16 de 14h a 16h",
            "planifie revision mercredi de 14h a 16h",
        ]
        for message in messages:
            vu: dict = {}

            def capter(user, appels, *a, **kw):
                vu["date"] = date.fromisoformat(appels[0]["args"]["date"])
                return None

            with patch("services.scheduling.placement.open_intervals", return_value=[]), \
                    patch.object(v1, "_ambiguous_scheduling_chips", side_effect=capter):
                v1._chips_from_message(None, message)
            self.assertEqual(lecture.date_puces_v1(message, aujourdhui), vu.get("date"), message)

    def test_sans_lecture_ou_lecture_cassee_tout_est_na(self):
        attendu = dict.fromkeys(lecture.LECTEURS, "na")
        self.assertEqual(lecture.accords(None, "supprime gym", LUNDI, {}), attendu)
        with self.assertLogs(JOURNAL, "WARNING"):
            self.assertEqual(lecture.accords(SimpleNamespace(), "supprime gym", LUNDI, {}), attendu)

    def test_la_ligne_du_tour_est_categorielle_et_sans_texte(self):
        message = "efface zorglub de mercredi a 18h cette semaine"
        prep = _prep(message, semaine=SEMAINE_GYM_MERCREDI)
        lu = _lecture(_element(mention="zorglub", operation="supprimer", candidats=["s1"],
                               titre_propose="Zorglub",
                               dates=[_ref("mercredi", "jour_semaine", jour_semaine="mercredi")],
                               heures=[_heure("18h", "18:00")]))
        suivi = lecture.Suivi(preparation=prep)
        with self.assertLogs(JOURNAL, "INFO") as journal:
            lecture.journaliser(suivi, lecture.Resultat("absente", lecture=lu, fournisseur="deepseek-flash",
                                                        ms=12, attente_ms=3, erreur="lecture_vide"))
        self.assertEqual(len(journal.records), 1)
        ligne = journal.records[0].getMessage()
        self.assertRegex(ligne, LIGNE_TOUR)
        self.assertNotIn("zorglub", ligne.lower())
        self.assertIn("suppression:ok", ligne)


# ------------------------------------------------------- concurrence du pool

class ConcurrenceTests(SimpleTestCase):
    """Huit tours concurrents, lecteur lent, pool LIRE de quatre. La base est
    interdite ici: ni le pool LIRE ni le pool AGIR n'y touchent."""

    def _manche(self):
        from services.agent_v2 import agent as module_agent

        def agir(soumis):
            debut = time.perf_counter()

            async def repondre(messages, info):
                await asyncio.sleep(0.2)
                return ModelResponse(parts=[TextPart(content="ok")])

            sortie = Agent(FunctionModel(repondre)).run_sync("x").output
            return debut - soumis, threading.current_thread().name, sortie

        def tour(_i):
            suivi = lecture.demarrer(None, "ajoute gym jeudi a 18h")
            soumis = time.perf_counter()
            retard, nom, sortie = module_agent._POOL_AGIR.submit(agir, soumis).result(timeout=10)
            return suivi, retard, nom, sortie, lecture.finir(suivi)

        requetes = ThreadPoolExecutor(max_workers=8, thread_name_prefix="requete")
        try:
            depart = time.perf_counter()
            tours = [f.result(timeout=15) for f in [requetes.submit(tour, i) for i in range(8)]]
            duree = time.perf_counter() - depart
        finally:
            requetes.shutdown(wait=True)
        for suivi, *_ in tours:
            try:
                suivi.futur.result(timeout=10)
            except CancelledError:
                pass
        return tours, duree

    @override_settings(LIRE_OMBRE="1", LIRE_ATTENTE_S="1.0")
    def test_huit_tours_concurrents_sans_famine_ni_interblocage(self):
        noms_lire: list = []
        lent = _lecteur(GYM_JEUDI, attente=0.8, appels=noms_lire)
        with patch.object(lecture, "preparer", side_effect=lambda user, m: _prep(m)), \
                _fournisseurs({"deepseek-flash": lent}):
            for _manche in range(2):
                tours, duree = self._manche()
                self.assertEqual(len(tours), 8)
                self.assertLess(duree, 8.0)
                statuts = [meta["lecture_statut"] for *_, meta in tours]
                self.assertTrue(set(statuts) <= {"ok", "hors_budget"}, statuts)
                self.assertIn("ok", statuts)
                for _suivi, retard, nom, sortie, _meta in tours:
                    self.assertLess(retard, 0.5, "AGIR a attendu derriere LIRE")
                    self.assertTrue(nom.startswith("agir"), nom)
                    self.assertEqual(sortie, "ok")
        self.assertTrue(noms_lire)
        self.assertTrue(all(n.startswith("lire") for n in noms_lire), noms_lire)
        self.assertEqual(lecture._POOL_LIRE._max_workers, 4)


# ------------------------------------------------------------- tours complets

def _agir_scripte(appel=None):
    """AGIR simule: un appel d'outil au premier pas s'il y en a un, puis du texte."""
    def deja_appele(messages):
        return any(isinstance(p, ToolReturnPart) for m in messages for p in getattr(m, "parts", []))

    def repondre(messages, info: AgentInfo):
        if appel and not deja_appele(messages):
            return ModelResponse(parts=[ToolCallPart(tool_name=appel[0], args=appel[1])])
        return ModelResponse(parts=[TextPart(content="C'est note.")])

    async def en_flux(messages, info: AgentInfo):
        if appel and not deja_appele(messages):
            yield {0: DeltaToolCall(name=appel[0], json_args=json.dumps(appel[1]))}
        else:
            yield "C'est note."

    return FunctionModel(repondre, stream_function=en_flux)


def _dire_scripte(sortie: dict):
    def repondre(messages, info: AgentInfo):
        return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=sortie)])
    return FunctionModel(repondre)


_VOLATILES = {"id", "block_id", "pk", "en_reponse_a", "emise_le", "cle", "task_id",
              "created_at", "updated_at"}


def _normaliser(valeur):
    """Seuls les identifiants de base et les horodatages different d'un compte a l'autre."""
    if isinstance(valeur, dict):
        return {k: ("<volatil>" if k in _VOLATILES else _normaliser(v)) for k, v in valeur.items()}
    if isinstance(valeur, list):
        return [_normaliser(v) for v in valeur]
    return valeur


class TourCompletTests(TransactionTestCase):
    """AGIR et DIRE scriptes par FunctionModel, outils et gardes reels."""

    SCENARIOS = {
        "mutation": ("ajoute gym tous les jeudis de 18h a 19h",
                     ("create_block", {"title": "Gym", "block_type": "sport", "days": ["jeudi"],
                                       "start_time": "18:00", "end_time": "19:00"}),
                     {"ouverture": "Voila.", "suite": "", "question": "", "options": [], "refs": []}),
        "puces_forcees": ("planifie revision demain de 14h a 16h", None,
                          {"ouverture": "", "suite": "", "question": "", "options": [], "refs": []}),
    }

    def _utilisateur(self, nom):
        user = User.objects.create_user(username=nom, password="x")
        demain = timezone.localdate() + timedelta(days=1)
        RecurringBlock.objects.create(user=user, title="Cours", block_type="course",
                                      day_of_week=demain.weekday(), start_time=heure(13, 0),
                                      end_time=heure(17, 0))
        return user

    @staticmethod
    def _lecture_valide(scenario):
        if scenario == "mutation":
            return {"elements": [_element(
                mention="gym", genre="sport", jours_semaine=[{"extrait": "jeudis", "jour": "jeudi"}],
                heures=[_heure("18h", "18:00"), _heure("19h", "19:00", role="fin")],
                recurrence="hebdomadaire")], "reponses": []}
        return {"elements": [_element(
            mention="revision", genre="revision",
            dates=[_ref("demain", "jour_relatif", jour_relatif="demain")],
            heures=[_heure("14h", "14:00"), _heure("16h", "16:00", role="fin")],
            recurrence="unique")], "reponses": []}

    def _tour(self, user, scenario, lire_actif, appels_lecteur=None):
        from services.agent_v2 import agent as module_agent
        from services.agent_v2.agent import PlannerAgentV2

        message, appel, dire = self.SCENARIOS[scenario]
        # Tous les reglages LIRE sont fixes ici: une variable du poste (LIRE_MODELES)
        # ne doit pas decider du fournisseur simule que le tour interroge.
        reglages = {"LIRE_OMBRE": "1" if lire_actif else "0", "LIRE_ATTENTE_S": "5",
                    "LIRE_MODELES": "deepseek-flash,gemini-2.5-flash"}
        lecteur = _lecteur(self._lecture_valide(scenario), appels=appels_lecteur)
        with override_settings(**reglages), \
                patch.object(module_agent, "modele_agir", side_effect=lambda: _agir_scripte(appel)), \
                patch.object(module_agent, "modele_dire", side_effect=lambda: _dire_scripte(dire)), \
                _fournisseurs({"deepseek-flash": lecteur}):
            evenements = list(PlannerAgentV2().process_message_stream(user, message))
        assistant = ConversationMessage.objects.filter(user=user, role="assistant").latest("pk")
        return evenements, assistant

    def test_le_tour_est_identique_lire_actif_ou_coupe(self):
        for scenario in self.SCENARIOS:
            with self.subTest(scenario=scenario):
                coupe, msg_coupe = self._tour(self._utilisateur(f"coupe-{scenario}"), scenario, False)
                actif, msg_actif = self._tour(self._utilisateur(f"actif-{scenario}"), scenario, True)

                self.assertEqual(msg_actif.metadata["lecture_statut"], "ok")
                self.assertEqual(msg_coupe.metadata["lecture_statut"], "desactivee")
                self.assertEqual(actif[-1]["type"], "done")
                self.assertEqual(actif[-1]["response"], coupe[-1]["response"])
                self.assertEqual(msg_actif.content, msg_coupe.content)
                self.assertEqual(json.dumps(_normaliser(actif), sort_keys=True, ensure_ascii=False),
                                 json.dumps(_normaliser(coupe), sort_keys=True, ensure_ascii=False))

                sans_lire = [{k: v for k, v in m.metadata.items() if k not in lecture.CLES_METADONNEES}
                             for m in (msg_actif, msg_coupe)]
                self.assertEqual(_normaliser(sans_lire[0]), _normaliser(sans_lire[1]))
                if scenario == "mutation":
                    self.assertTrue(actif[-1]["blocks_created"])
                else:
                    self.assertTrue(actif[-1]["quick_replies"])

    def test_metadonnees_et_une_ligne_par_tour_sans_texte(self):
        user = self._utilisateur("meta")
        with self.assertLogs(JOURNAL, "INFO") as journal:
            _evts, assistant = self._tour(user, "mutation", True)
        meta = assistant.metadata
        self.assertEqual(meta["lecture_statut"], "ok")
        self.assertEqual(meta["lecture_fournisseur"], "deepseek-flash")
        self.assertIsInstance(meta["lecture_ms"], int)
        self.assertEqual(meta["lecture"]["elements"][0]["recurrence"], "hebdomadaire")
        lignes = [r.getMessage() for r in journal.records if r.getMessage().startswith("agent_v2 lire statut=")]
        self.assertEqual(len(lignes), 1)
        self.assertRegex(lignes[0], LIGNE_TOUR)
        tout = " ".join(r.getMessage() for r in journal.records).lower()
        self.assertNotIn("gym", tout)
        self.assertNotIn("jeudis", tout)

    def test_l_api_des_conversations_ne_renvoie_pas_la_lecture(self):
        from rest_framework.test import APIClient

        user = self._utilisateur("api")
        self._tour(user, "puces_forcees", True)
        client = APIClient()
        client.force_authenticate(user)
        reponse = client.get("/api/conversations/")
        self.assertEqual(reponse.status_code, 200)
        assistant = [m for m in reponse.json() if m["role"] == "assistant"][-1]
        self.assertTrue(assistant["metadata"]["quick_replies"])
        for cle in lecture.CLES_METADONNEES:
            self.assertNotIn(cle, assistant["metadata"])

    def test_aucun_orm_dans_le_pool_lire(self):
        requetes: list = []
        lecteurs: list = []
        original = CursorWrapper._execute_with_wrappers

        def espion(curseur, sql, params, many, executor):
            requetes.append(threading.current_thread().name)
            return original(curseur, sql, params, many, executor)

        user = self._utilisateur("orm")
        with patch.object(CursorWrapper, "_execute_with_wrappers", espion):
            _evts, assistant = self._tour(user, "mutation", True, appels_lecteur=lecteurs)
        self.assertEqual(assistant.metadata["lecture_statut"], "ok")
        self.assertTrue(lecteurs and all(n.startswith("lire") for n in lecteurs), lecteurs)
        self.assertTrue(requetes)
        self.assertFalse([n for n in requetes if n.startswith("lire")])

    def test_le_chemin_rapide_ne_soumet_pas_lire(self):
        from services.agent_v2.agent import PlannerAgentV2

        user = self._utilisateur("rapide")
        with override_settings(LIRE_OMBRE="1"), \
                patch.object(PlannerAgentV2, "_tour_decide", staticmethod(lambda registre, message: True)), \
                patch.object(lecture, "demarrer") as demarrer:
            list(PlannerAgentV2().process_message_stream(user, "oui"))
        demarrer.assert_not_called()
        assistant = ConversationMessage.objects.filter(user=user, role="assistant").latest("pk")
        self.assertEqual(assistant.metadata["lecture_statut"], "sautee")
        self.assertIsNone(assistant.metadata["lecture"])

    def test_trois_tours_de_suite_ne_gelent_pas(self):
        """Le gel du 2026-08-28 frappait le DEUXIEME tour d'un processus."""
        user = self._utilisateur("suite")
        fini: list = []

        def tours():
            try:
                for _ in range(3):
                    _evts, assistant = self._tour(user, "mutation", True)
                    fini.append(assistant.metadata["lecture_statut"])
            finally:
                connections.close_all()

        fil = threading.Thread(target=tours, name="requete-suite")
        fil.start()
        fil.join(timeout=60)
        self.assertFalse(fil.is_alive(), "un tour s'est fige")
        self.assertEqual(fini, ["ok", "ok", "ok"])
