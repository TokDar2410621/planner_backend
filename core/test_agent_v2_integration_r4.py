"""
Round 4, integration des deux correctifs: la demande reposee par le code
apres une reponse floue (B1, deux chemins: registre d'outils et agent) et la
lecture de portee qui ne supprime jamais sur un verbe de garde (G1).
"""
from core import test_agent_v2_voix_r4 as voix_r4
from core.models import RecurringBlock
from services.agent_v2.redaction import ReponseDire


class ReemiseEtGardeTests(voix_r4.DemandeReemiseTests):
    """Herite du harnais; les tests du parent tournent deja dans leur module."""

    def _flou(self):
        premier = self._tour("efface tout jeudi", agir=self._supprimer)
        self.assertEqual(premier["question_motif"], "portee_jour")
        cle = self._meta()["demandes"][0]["cle"]
        flou = self._tour("Oui, supprime ces trois blocs.",
                          dire=ReponseDire(ouverture="D'accord."))
        self.assertEqual(flou["question_motif"], "portee_jour")
        self.assertEqual([d["cle"] for d in self._meta()["demandes"]], [cle])
        return cle

    def test_une_seule_demande_malgre_les_deux_chemins_de_reemission(self):
        self._flou()
        self.assertEqual(len(self._meta()["demandes"]), 1)
        self.assertTrue(RecurringBlock.all_objects.get(pk=self.quart.pk).active)

    def test_garder_tous_les_jeudis_apres_une_reemission_ne_supprime_rien(self):
        cle = self._flou()
        done = self._tour("garde tous les jeudis")
        self.assertTrue(RecurringBlock.all_objects.get(pk=self.quart.pk).active)
        # Reponse ambigue dans le sens sur: la question revient, rien n'est fait.
        self.assertEqual(done["question_motif"], "portee_jour")
        self.assertEqual([d["cle"] for d in self._meta()["demandes"]], [cle])

    def test_la_reponse_claire_apres_reemission_ferme_la_boucle(self):
        self._flou()
        done = self._tour("Tous les jeudis")
        self.assertFalse(RecurringBlock.all_objects.get(pk=self.quart.pk).active)
        self.assertNotEqual(done["question_motif"], "portee_jour")
        self.assertEqual(self._meta()["demandes"], [])

    # Les tests herites du parent ne sont pas rejoues ici.
    test_oui_flou_puis_tous_les_jeudis = None
    test_une_reponse_claire_n_est_pas_reemise = None
