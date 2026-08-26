"""
Parite des outils entre v1 et v2, et garde destructive.

Les regles produit vivent DANS les descriptions et les schemas des outils
(segregation recurrent contre ponctuel, jamais demander un identifiant, jamais
supprimer puis recreer pour deplacer). Une reecriture qui garde le code mais
perd ces phrases perd le comportement de l'agent.

Ce test verrouille le contrat COMPLET, schema serialise compris. Verifie par
sonde le 2026-08-24: Tool(fonction, name=..., description=...) sur une
fonction **kwargs produit un schema VIDE, donc 30 outils sans le moindre
parametre. Seul Tool.from_schema transmet le vrai schema.
"""
import json

from django.contrib.auth.models import User
from django.test import TestCase, TransactionTestCase

from services.agent.tools import ALL_TOOLS, TOOL_MAP
from services.agent_v2 import outils as outils_v2
from services.agent_v2.registre import Registre


class PariteDesOutilsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='parite', password='x')
        self.registre = Registre()
        self.exposes = {
            t.name: t for t in outils_v2.outils_pour(self.user, self.registre)
        }

    def test_les_30_outils_sont_exposes(self):
        attendus = {t.name for t in ALL_TOOLS}
        self.assertEqual(len(attendus), 30, 'ALL_TOOLS a change de taille')
        self.assertEqual(set(self.exposes), attendus)

    def test_chaque_description_est_identique_octet_pour_octet(self):
        for outil in ALL_TOOLS:
            with self.subTest(outil=outil.name):
                self.assertEqual(
                    self.exposes[outil.name].description.encode('utf-8'),
                    outil.description.encode('utf-8'),
                )

    def test_chaque_schema_serialise_est_identique(self):
        for outil in ALL_TOOLS:
            with self.subTest(outil=outil.name):
                expose = outils_v2.schema_expose(self.exposes[outil.name])
                self.assertEqual(
                    json.dumps(expose, sort_keys=True, ensure_ascii=False),
                    json.dumps(outil.parameters, sort_keys=True, ensure_ascii=False),
                )

    def test_aucun_schema_n_est_vide(self):
        """Le piege exact de la sonde: un schema vide passerait le test de
        description tout en privant le modele de tous les parametres."""
        avec_params = [t for t in ALL_TOOLS if t.parameters.get('properties')]
        self.assertGreater(len(avec_params), 20)
        for outil in avec_params:
            with self.subTest(outil=outil.name):
                expose = outils_v2.schema_expose(self.exposes[outil.name])
                self.assertTrue(
                    expose.get('properties'),
                    f'{outil.name} expose un schema sans proprietes',
                )


class GardeDestructiveTests(TestCase):
    """agent.py applique requires_confirmation (lignes 921 a 944), PAS
    execute_tool. Un adaptateur qui appelle les outils en direct contourne la
    garde, et clear_all_blocks efface un planning sans confirmation."""

    def test_les_outils_destructifs_portent_le_drapeau(self):
        for nom in ('clear_all_blocks', 'delete_task'):
            self.assertTrue(
                getattr(TOOL_MAP[nom], 'requires_confirmation', False),
                f'{nom} devrait porter requires_confirmation',
            )

    def test_une_intention_claire_autorise(self):
        for message in ("supprime ma tache X", "oui vas-y", "efface tout",
                        "c'est bon, confirme"):
            with self.subTest(message=message):
                self.assertTrue(outils_v2.autorise_destructif(message))

    def test_les_accents_ne_contournent_pas_la_garde(self):
        self.assertTrue(outils_v2.autorise_destructif("supprimé mon cours"))

    def test_une_demande_anodine_n_autorise_pas(self):
        for message in ("c'est quoi mon planning ?", "ajoute un cours", "", None):
            with self.subTest(message=message):
                self.assertFalse(outils_v2.autorise_destructif(message))


class ExecutionTests(TransactionTestCase):
    """L'adaptateur doit alimenter le registre a CHAQUE appel, y compris quand
    l'outil echoue ou leve, sinon le bloc factuel se tait sur une mutation
    tentee.

    TransactionTestCase et non TestCase, pour une raison de fond: l'adaptateur
    executera l'ORM via sync_to_async(thread_sensitive=True), donc dans un
    thread d'executeur distinct. TestCase enveloppe chaque test dans une
    transaction non validee, que ce thread ne peut pas voir: l'utilisateur
    cree en setUp serait invisible et l'outil echouerait pour une raison qui
    n'a rien a voir avec le code teste.

    La production n'a pas ce probleme: ATOMIC_REQUESTS vaut False et
    AUTOCOMMIT vaut True, verifie le 2026-08-25. Chaque ecriture est validee
    immediatement, donc visible de tout thread.
    """

    def setUp(self):
        self.user = User.objects.create_user(username='exec', password='x')

    def _appeler(self, nom, message='', **kwargs):
        import asyncio
        registre = Registre()
        outils = {t.name: t for t in outils_v2.outils_pour(self.user, registre, message)}
        fonction = outils[nom].function_schema.function
        asyncio.run(fonction(**kwargs))
        return registre

    def test_un_appel_reussi_laisse_une_entree(self):
        registre = self._appeler(
            'create_block', title='Maths', block_type='course',
            days=[0], start_time='09:00', end_time='12:00',
        )
        self.assertEqual(len(registre.actions), 1)
        self.assertTrue(registre.actions[0].succes)
        self.assertEqual(registre.actions[0].outil, 'create_block')

    def test_un_appel_echoue_laisse_AUSSI_une_entree(self):
        """Un refus doit pouvoir etre annonce, donc il doit etre consigne."""
        registre = self._appeler(
            'create_block', title='Vide', block_type='course',
            days=[], start_time='09:00', end_time='12:00',
        )
        self.assertEqual(len(registre.actions), 1)
        self.assertFalse(registre.actions[0].succes)

    def test_un_outil_destructif_sans_confirmation_est_refuse_ET_consigne(self):
        registre = self._appeler('clear_all_blocks', message='bonjour', confirm=True)
        self.assertEqual(len(registre.actions), 1)
        action = registre.actions[0]
        self.assertFalse(action.succes)
        self.assertTrue(action.donnees.get('needs_confirmation'))

    def test_un_outil_destructif_avec_confirmation_passe(self):
        registre = self._appeler(
            'clear_all_blocks', message='oui efface tout', confirm=True)
        self.assertEqual(len(registre.actions), 1)
        self.assertNotIn('needs_confirmation', registre.actions[0].donnees)
