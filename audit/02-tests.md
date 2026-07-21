# Audit Planner AI — Phase 2 : Tests

> Exécuté le 2026-07-20. Backend via venv (`python 3.10`, Django 5.2.10, pytest 7.4.4). Frontend `node` + `vitest 3.2.4`. Tests manuels via `curl` contre un serveur local sur une **base SQLite jetable** (scratchpad), la vraie `db.sqlite3` n'a pas été touchée.

## 1. Suite backend (`pytest`)

```
33 passed in 21.25s
```

Tous les tests existants passent (`core/tests.py` = 21 tests, `services/scheduling/test_overlap.py` = 12 tests). Ils couvrent les modèles, l'overlap, et une tranche fine des endpoints. **Couverture réelle des chemins critiques : faible** :
- Chat IA : **0 test** (feature centrale).
- Traitement de document / OCR / `run_in_background` : **0 test**.
- Correction de la génération de planning : le test existant n'assert que `HTTP 200` + présence de la clé `created_blocks`, jamais la correction (pas de chevauchement, respect des slots).
- Isolation multi-tenant (user A vs user B) : **0 test**.
- 4 endpoints de partage (dont les 2 publics) : **0 test**.
- `GoogleAuthView`, `CheckEmailView`, `MeView`, les 6 endpoints insights : **0 test**.

### Vérifications Django complémentaires
- `makemigrations --check --dry-run` → **No changes detected** (aucune migration manquante, schéma cohérent).
- `migrate` sur base neuve → 9 migrations appliquées proprement.
- `manage.py check --deploy` → **6 warnings** : `SECURE_HSTS_SECONDS`, `SECURE_SSL_REDIRECT`, `SECRET_KEY` (préfixe `django-insecure-`), `SESSION_COOKIE_SECURE`, `CSRF_COOKIE_SECURE`, `DEBUG`. Aucun réglage `SECURE_*`/cookie n'existe dans `settings.py`.

## 2. Frontend

- `npm ci` : OK (node_modules absent au départ).
- `vitest run` : **1 test** (`src/test/example.test.ts`, `expect(true).toBe(true)`). Couverture ≈ 0.
- `vite build` : **OK** (bundle principal 494 KB / 155 KB gzip, Schedule 194 KB, Chat 149 KB). Attention : `vite build` (esbuild) **ne typecheck pas**.
- `tsc --noEmit` : **0 erreur**. Mais `tsconfig.app.json` a `strict:false`, `noImplicitAny:false`, et ESLint `no-unused-vars:off` : la sécurité de type n'est ni compilée ni lintée.
- `npm audit` : **35 vulnérabilités** (2 critical, 23 high, 8 moderate, 2 low), majoritairement des dépendances **dev transitives** (vitest UI, esbuild, ws, yaml). Impact prod limité, mais à corriger via `npm audit fix`.

## 3. Tests manuels `curl` (endpoints critiques)

Serveur `runserver 8010` sur base jetable, user `demo/demo123`.

| # | Endpoint | Résultat | Verdict |
|---|---|---|---|
| 1 | `POST /auth/login/` | 200, `{user, tokens:{access,refresh}}` | ✅ |
| 2 | `POST /auth/refresh/` | 200, nouveaux access+refresh (rotation) | ✅ |
| 3 | `POST /auth/register/` | 201 | ✅ |
| 4 | `GET /tasks/` (authed) | 200, 6 tâches | ✅ |
| 5 | `POST /tasks/` | 201 | ✅ |
| 6 | `GET /schedule/` | 200 | ✅ |
| 7 | `POST /schedule/generate/` | 200, blocs créés, **rapide** (déterministe) | ✅ |
| 8 | `GET /recurring-blocks/` | 200 | ✅ |
| 9 | `GET /recurring-completions/` | 200 | ✅ |
| 10 | IDOR : user B lit la tâche de demo | 404 « No Task matches » | ✅ (tâches isolées) |
| 11 | `GET /planning/demo/` (SANS auth) | **200, fuite du planning complet** (titres, salles `A-101`/`B-205`, horaires) | ❌ **FUITE PII** |
| 12 | `GET /tasks/` (sans token) | 401 | ✅ |
| 13 | `POST /chat/` (multipart, 1 appel LLM) | **200 mais `"Erreur lors de la communication avec l'IA"` en 0.42s** | ❌ **CHAT CASSÉ** |
| 14 | `GET /insights/suggestions/` | 200, vraies suggestions (règle-based, rapide) | ✅ |

### Bug reproduit #13 (le plus important) — Chat IA mort
Logs serveur pendant le `POST /chat/` :
```
INFO claude ClaudeProvider initialized with model: claude-sonnet-4-20250514
INFO _client HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 404 Not Found"
ERROR claude Claude API error: 404 - not_found_error, 'message': 'model: claude-sonnet-4-20250514'
```
Cause : `services/llm/claude.py:29` `DEFAULT_MODEL = 'claude-sonnet-4-20250514'` (modèle Claude Sonnet 4, en cours de retrait, renvoie 404 pour ce compte). La clé API est valide (le 404 porte sur le **modèle**, pas l'auth). Aggravé par `services/agent/agent.py:38` qui **fige `self.llm = ClaudeProvider()`**, ignorant `LLM_PROVIDER=gemini` (défaut) et `profile.preferred_llm`. Aucun fallback : l'erreur est renvoyée 200 comme réponse assistant.

**Reproduction** : login demo → `curl -F "message=..." /api/chat/` → réponse d'erreur générique.

### Bug reproduit #11 — Fuite PII publique
`GET /api/planning/<username>/` (`views.py:1011`, `AllowAny`) : lookup de **n'importe quel** username, renvoie tous ses `RecurringBlock` actifs (titre, type, jour, horaires, **salle**). Aucun opt-in, aucun token, username énumérable. Reproduction : `curl http://127.0.0.1:8010/api/planning/demo/` → 200 + planning.

## 4. Tests ajoutés (Phase 2, sans corriger les bugs)

Nouveau fichier `core/test_audit.py` (14 tests) couvrant les chemins critiques non testés. Les tests qui **reproduisent un bug confirmé** sont marqués `@unittest.expectedFailure` : ils échouent tant que le bug existe (Phase 4), ce qui garde la suite verte tout en documentant chaque défaut par un repro exécutable.

```
9 passed, 5 xfailed in 11.81s
```

**9 nouveaux tests verts (vraie couverture ajoutée)** :
- Isolation : user B ne peut ni lire ni modifier la tâche/le bloc de user A (404).
- Chat : auth requise (401), payload vide rejeté (400), câblage vue OK (agent mocké → 200).
- Génération : aucun bloc généré ne chevauche le RecurringBlock du jour ; blocs dans la fenêtre 08:00-22:00.
- Public planning : username inconnu → 404.

**5 `xfail` (bugs confirmés, repro exécutable)** :
| Test | Bug reproduit |
|---|---|
| `test_user_cannot_complete_others_recurring_block` | IDOR sur `recurring-completions` (référence le bloc d'un autre user) |
| `test_planning_by_username_is_not_public_without_optin` | Fuite PII `/api/planning/<username>/` |
| `test_agent_respects_configured_provider` | Provider figé en dur (`ClaudeProvider`), ignore la config |
| `test_claude_default_model_is_not_retired` | Modèle Claude retiré (404) |
| `test_token_blacklist_app_installed` | `BLACKLIST_AFTER_ROTATION` sans l'app `token_blacklist` |

> Quand un bug sera corrigé en Phase 4, son test `xfail` deviendra `xpass` : signal pour retirer le décorateur.

## 5. Résumé

- Backend : suite existante verte (33/33), mais couverture des chemins critiques faible. Ajout de 14 tests (9 verts + 5 repros de bugs). Aucune migration manquante.
- Frontend : build OK, `tsc` propre, mais `strict:false` et couverture de test ≈ 0.
- **2 bugs bloquants reproduits au runtime** : chat IA mort (modèle Claude 404 + provider figé), fuite PII publique par username. Détail et correctifs proposés en `03-plan.md`.
