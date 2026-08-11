"""
L'API REST des objectifs.

Contexte du bug d'origine: le modèle `Goal` existait, l'agent savait le lire et
l'écrire (`list_goals`, `create_goal`, `update_goal`, contexte du jour, cerveau
quotidien), mais AUCUNE route REST ne l'exposait. La page Objectifs vivait donc
dans un store local (`goals-storage`, localStorage). Conséquence visible pour
l'utilisateur: un objectif créé par l'agent ou par le MCP n'apparaissait jamais
dans l'interface, et l'inverse était vrai aussi. Les tests ci-dessous fixent le
contrat des deux côtés.
"""
from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken

from core.models import Goal


def _client(user):
    api = APIClient()
    api.credentials(HTTP_AUTHORIZATION=f'Bearer {RefreshToken.for_user(user).access_token}')
    return api


class GoalApiTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='dar', password='x')
        self.other = User.objects.create_user(username='autre', password='x')
        self.client_ = _client(self.user)

    # --- Le cas qui a motivé tout ceci ---------------------------------------

    def test_un_objectif_cree_hors_interface_est_visible_par_lapi(self):
        """Un objectif posé directement en base (agent, MCP, admin) se lit."""
        Goal.objects.create(
            user=self.user,
            title='Planner AI sur l\'App Store',
            goal_type='short_term',
            deadline='2026-08-25',
        )

        r = self.client_.get('/api/goals/')
        self.assertEqual(r.status_code, 200)
        results = r.json()['results']
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['title'], 'Planner AI sur l\'App Store')
        self.assertEqual(results[0]['deadline'], '2026-08-25')

    def test_un_objectif_cree_par_lapi_est_visible_par_lagent(self):
        """Le chemin inverse: l'ORM (donc l'agent) voit ce que l'API écrit."""
        r = self.client_.post('/api/goals/', {
            'title': 'Lancer la v2',
            'type': 'long_term',
        }, format='json')
        self.assertEqual(r.status_code, 201, r.content)

        goal = Goal.objects.get(user=self.user)
        self.assertEqual(goal.title, 'Lancer la v2')
        self.assertEqual(goal.goal_type, 'long_term')

    # --- Contrat de champs ----------------------------------------------------

    def test_le_champ_sappelle_type_cote_api_et_goal_type_en_base(self):
        r = self.client_.post('/api/goals/', {'title': 'A', 'type': 'short_term'}, format='json')
        self.assertEqual(r.status_code, 201, r.content)
        self.assertEqual(r.json()['type'], 'short_term')
        self.assertNotIn('goal_type', r.json())

    def test_type_par_defaut_court_terme(self):
        r = self.client_.post('/api/goals/', {'title': 'Sans type'}, format='json')
        self.assertEqual(r.status_code, 201, r.content)
        self.assertEqual(r.json()['type'], 'short_term')

    def test_type_invalide_rejete(self):
        r = self.client_.post('/api/goals/', {'title': 'A', 'type': 'moyen_terme'}, format='json')
        self.assertEqual(r.status_code, 400)

    def test_progression_hors_bornes_rend_400_pas_500(self):
        """Sans validation, la CheckConstraint en base ferait planter la requête."""
        r = self.client_.post(
            '/api/goals/', {'title': 'A', 'type': 'short_term', 'progress': 150}, format='json'
        )
        self.assertEqual(r.status_code, 400)

    def test_cent_pour_cent_bascule_en_complete(self):
        """Même règle que l'agent: 100% et « actif » ne coexistent pas."""
        r = self.client_.post(
            '/api/goals/',
            {'title': 'A', 'type': 'short_term', 'progress': 100, 'status': 'active'},
            format='json',
        )
        self.assertEqual(r.status_code, 201, r.content)
        self.assertEqual(r.json()['status'], 'completed')

    def test_progression_partielle_ne_bascule_pas(self):
        r = self.client_.post(
            '/api/goals/', {'title': 'A', 'type': 'short_term', 'progress': 99}, format='json'
        )
        self.assertEqual(r.json()['status'], 'active')

    # --- Mise à jour et suppression ------------------------------------------

    def test_patch_partiel(self):
        goal = Goal.objects.create(user=self.user, title='A', goal_type='short_term')
        r = self.client_.patch(f'/api/goals/{goal.id}/', {'progress': 40}, format='json')
        self.assertEqual(r.status_code, 200, r.content)
        goal.refresh_from_db()
        self.assertEqual(goal.progress, 40)
        self.assertEqual(goal.title, 'A')

    def test_patch_a_cent_pour_cent_complete_lobjectif(self):
        goal = Goal.objects.create(user=self.user, title='A', goal_type='short_term', progress=10)
        r = self.client_.patch(f'/api/goals/{goal.id}/', {'progress': 100}, format='json')
        self.assertEqual(r.status_code, 200, r.content)
        goal.refresh_from_db()
        self.assertEqual(goal.status, 'completed')

    def test_suppression(self):
        goal = Goal.objects.create(user=self.user, title='A', goal_type='short_term')
        r = self.client_.delete(f'/api/goals/{goal.id}/')
        self.assertEqual(r.status_code, 204)
        self.assertFalse(Goal.objects.filter(id=goal.id).exists())

    # --- Cloisonnement --------------------------------------------------------

    def test_on_ne_voit_pas_les_objectifs_dautrui(self):
        Goal.objects.create(user=self.other, title='Secret', goal_type='short_term')
        r = self.client_.get('/api/goals/')
        self.assertEqual(r.json()['results'], [])

    def test_on_ne_peut_pas_lire_lobjectif_dautrui_par_son_id(self):
        goal = Goal.objects.create(user=self.other, title='Secret', goal_type='short_term')
        # 404 et non 403: pas d'énumération d'ids.
        self.assertEqual(self.client_.get(f'/api/goals/{goal.id}/').status_code, 404)
        self.assertEqual(self.client_.patch(f'/api/goals/{goal.id}/', {'progress': 1}, format='json').status_code, 404)
        self.assertEqual(self.client_.delete(f'/api/goals/{goal.id}/').status_code, 404)

    def test_on_ne_peut_pas_creer_un_objectif_au_nom_dautrui(self):
        r = self.client_.post(
            '/api/goals/', {'title': 'A', 'type': 'short_term', 'user': self.other.id}, format='json'
        )
        self.assertEqual(r.status_code, 201)
        self.assertEqual(Goal.objects.get(id=r.json()['id']).user, self.user)

    def test_anonyme_refuse(self):
        self.assertEqual(APIClient().get('/api/goals/').status_code, 401)

    # --- Filtres --------------------------------------------------------------

    def test_filtres_status_et_type(self):
        Goal.objects.create(user=self.user, title='A', goal_type='short_term', status='active')
        Goal.objects.create(user=self.user, title='B', goal_type='long_term', status='paused')

        self.assertEqual(len(self.client_.get('/api/goals/?status=paused').json()['results']), 1)
        self.assertEqual(len(self.client_.get('/api/goals/?type=long_term').json()['results']), 1)
        self.assertEqual(len(self.client_.get('/api/goals/').json()['results']), 2)
