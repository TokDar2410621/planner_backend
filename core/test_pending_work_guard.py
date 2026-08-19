"""
Le mensonge du « travail en cours » (vécu prod 2026-08-18, 21:59-22:02):
l'agent a promis « je vais supprimer les blocs puis ajouter tes cours »,
puis « je suis en train », puis « je n'ai pas encore terminé », trois tours
streamés d'affilée avec ZÉRO appel d'outil. Ces tests verrouillent le
détecteur qui force le rejeu non-streamé de ces tours.
"""
from django.test import SimpleTestCase

from services.agent.agent import _claims_pending_work


class PendingWorkDetectorTests(SimpleTestCase):
    def test_les_phrases_exactes_du_18_aout_sont_detectees(self):
        vecues = [
            "Absolument. Je vais ajuster ton planning pour que tes cours soient "
            "prioritaires. Cela signifie que je vais supprimer les blocs existants "
            "qui chevauchent tes cours, puis ajouter tes cours.",
            "Je suis en train de mettre à jour ton planning.",
            "Je dois d'abord identifier et supprimer les blocs existants qui "
            "chevauchent tes cours, puis ajouter les nouveaux cours. Cela peut "
            "prendre quelques instants.",
            "Je n'ai pas encore terminé. Je te tiendrai informé dès que ce sera fait.",
        ]
        for phrase in vecues:
            self.assertTrue(_claims_pending_work(phrase), phrase)

    def test_les_tournures_legitimes_passent(self):
        legitimes = [
            "Veux-tu que je supprime le bloc « Travail » de mardi ?",
            "Je peux optimiser ta semaine si tu veux.",
            "Tu vas recevoir une notification avant tes blocs.",
            "J'ai supprimé le bloc et ajouté tes cours.",  # affaire du filet COMPLETED
            "Ton planning du mercredi est vide.",
            "",
        ]
        for phrase in legitimes:
            self.assertFalse(_claims_pending_work(phrase), phrase)

    def test_les_accents_ne_cachent_rien(self):
        self.assertTrue(_claims_pending_work("Je vais déplacer ton cours puis le caler."))
        self.assertTrue(_claims_pending_work("Je n'ai pas encore TERMINÉ."))
