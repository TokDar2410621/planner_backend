"""
Un fournisseur lent ne doit pas pouvoir immobiliser un tour.

Mesure du 2026-08-29. Le matin, en production: mediane de 57 s par tour, un
tour a 397 s, jusqu'a 99 s pour un seul appel. L'apres-midi, MEME CODE:
mediane de 8,2 s, maximum 16,7 s, avec 15 a 350 jetons de raisonnement par
etape. Rien n'avait change chez nous. DeepSeek etait lent, voila tout.

Ce qui a laisse durer le tour de 397 s, c'est le client par defaut du
fournisseur: 600 secondes de delai en lecture et deux reprises. Personne
n'arretait rien.

Ces tests verrouillent les trois proprietes qui rendent la lenteur bornee:

1. le delai est fini et court au regard d'un usage conversationnel;
2. aucune reprise, parce que rejouer un appel expire chez un fournisseur qui
   rame revient a attendre trois fois; la resilience vient de la chaine de
   repli, pas de l'acharnement;
3. un depassement leve bien une exception que FallbackModel intercepte, sinon
   le tour echouerait au lieu de basculer.

Aucun appel reseau ici: on inspecte la configuration et la hierarchie
d'exceptions.
"""
from django.test import SimpleTestCase


class DelaiDuFournisseurTests(SimpleTestCase):
    def setUp(self):
        from services.agent_v2 import modeles
        self.modeles = modeles

    def _client_deepseek(self):
        modele = self.modeles._deepseek()
        self.assertIsNotNone(
            modele, "DEEPSEEK_API_KEY doit etre configuree pour ce test")
        fournisseur = getattr(modele, "_provider", None) or getattr(modele, "provider", None)
        return fournisseur.client

    def test_le_delai_est_borne(self):
        """600 s par defaut, c'est dix minutes d'immobilisation possible."""
        client = self._client_deepseek()
        delai = client.timeout
        secondes = getattr(delai, "read", delai)
        self.assertIsNotNone(secondes)
        self.assertLessEqual(
            float(secondes), 90.0,
            "un tour conversationnel ne doit pas pouvoir attendre plus longtemps")

    def test_aucune_reprise(self):
        """Rejouer un appel expire multiplie l'attente par le nombre d'essais."""
        self.assertEqual(self._client_deepseek().max_retries, 0)

    def test_un_depassement_fait_basculer_et_non_echouer(self):
        """FallbackModel intercepte ModelAPIError par defaut. Un depassement de
        delai en leve un: verifie par sonde reseau le 2026-08-29, le message
        etant « Request timed out. ». Si cette parente disparaissait, un
        fournisseur lent ferait echouer le tour au lieu de passer au suivant."""
        from pydantic_ai.exceptions import ModelAPIError
        from pydantic_ai.models.fallback import FallbackModel
        import inspect

        defaut = inspect.signature(FallbackModel.__init__).parameters["fallback_on"].default
        self.assertIn(ModelAPIError, defaut)

    def test_la_chaine_a_bien_un_repli(self):
        """Borner ne sert a rien si personne ne prend la releve."""
        chaine = self.modeles._chaine()
        membres = getattr(chaine, "models", None)
        self.assertIsNotNone(
            membres, "la chaine doit compter plusieurs modeles pour que le repli existe")
        self.assertGreaterEqual(len(membres), 2)
