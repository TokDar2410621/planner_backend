"""
Ordre des fournisseurs et reglages. Aucun appel reseau: on verifie la
CONSTRUCTION, pas le service.
"""
from django.test import SimpleTestCase, override_settings

from services.agent_v2 import modeles


@override_settings(DEEPSEEK_API_KEY='x', GEMINI_API_KEY='y', ANTHROPIC_API_KEY='z',
                   DEEPSEEK_MODEL='deepseek-v4-pro')
class ConstructionTests(SimpleTestCase):
    def test_l_ordre_commence_par_deepseek(self):
        noms = modeles.noms_de(modeles.modele_agir())
        self.assertEqual(len(noms), 3)
        self.assertIn('deepseek', noms[0].lower())

    def test_le_modele_vient_des_settings(self):
        """Un nom en dur ferait comparer deux MODELES et non deux boucles."""
        self.assertIn('deepseek-v4-pro', modeles.noms_de(modeles.modele_agir())[0])

    def test_dire_coupe_le_raisonnement(self):
        """Verifie par sonde: en mode thinking, DeepSeek refuse
        tool_choice=required, que PydanticAI utilise pour forcer la sortie
        structuree. Sans ce reglage, DIRE echoue 10 fois sur 10."""
        self.assertEqual(
            modeles.REGLAGES_DIRE['extra_body']['thinking']['type'], 'disabled')


class AbsenceDeClesTests(SimpleTestCase):
    @override_settings(DEEPSEEK_API_KEY='', GEMINI_API_KEY='y', ANTHROPIC_API_KEY='')
    def test_un_fournisseur_sans_cle_disparait(self):
        self.assertEqual(len(modeles.noms_de(modeles.modele_agir())), 1)

    @override_settings(DEEPSEEK_API_KEY='', GEMINI_API_KEY='', ANTHROPIC_API_KEY='')
    def test_aucune_cle_leve_une_erreur_explicite(self):
        with self.assertRaises(RuntimeError):
            modeles.modele_agir()


class DependanceDeclareeTests(SimpleTestCase):
    """La prod n'installe QUE requirements.txt.

    Vecu le 2026-08-27: pydantic-ai etait installe a la main dans le venv
    local pendant la sonde et jamais declare. Les 626 tests passaient au vert
    pendant que la production levait ModuleNotFoundError a chaque message des
    comptes bascules. Un import qui marche en local ne prouve rien sur ce qui
    est deploye.
    """

    def test_pydantic_ai_est_declare_dans_requirements(self):
        import pathlib
        racine = pathlib.Path(__file__).resolve().parent.parent
        texte = (racine / 'requirements.txt').read_text(encoding='utf-8')
        self.assertIn('pydantic-ai-slim', texte)
        # Sans cet epinglage, `import pydantic_ai` casse sur
        # ModuleNotFoundError: opentelemetry._events (mesure du 2026-08-24).
        self.assertIn('opentelemetry-api<1.38', texte)

    def test_le_paquet_v2_s_importe_vraiment(self):
        from services.agent_v2 import PlannerAgentV2
        self.assertTrue(callable(PlannerAgentV2))
