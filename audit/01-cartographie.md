# Audit Planner AI — Phase 1 : Cartographie

> Généré le 2026-07-20 sur la branche `audit-remise-a-niveau`. Lecture intégrale du code (backend `planner_backend/`, frontend `day-wise-bot/`), hors `venv/`, `node_modules/`, `dist/`, `__pycache__/`, `media/`.

## 1. Vue d'ensemble

| | Backend | Frontend |
|---|---|---|
| Stack | Django 5.2.10 + DRF + SimpleJWT | React 18 + Vite 5 + TypeScript |
| Auth | JWT (access 1h, refresh 7j, rotation + blacklist activés) | JWT en `localStorage`, Zustand persist |
| Base | PostgreSQL (Railway) / SQLite (local) | — |
| IA | `anthropic` + `google-genai` (+ HF vision) | — |
| Déploiement | Railway (aucun start-command déclaré) | Vercel (`day-wise-bot.vercel.app`) + Capacitor |
| Tests | `pytest` (33 tests, tous verts) | `vitest` (1 test factice) |

Les deux dossiers sont des **dépôts git distincts** (chacun son `.git`, branche `main` avec historique) ; le dossier parent `planner/` n'a aucun commit. Chaque dépôt est maintenant sur la branche `audit-remise-a-niveau`.

Constat de départ (« l'app ne marche pas bien ») : l'audit identifie **plusieurs causes racines concrètes et cumulatives**, détaillées en `03-plan.md`. Les trois plus probables :
1. **Le chat IA est mort à 100 %** : modèle Claude retiré (404) + provider figé en dur ignorant Gemini (voir §4).
2. **Le SPA déployé peut pointer vers `localhost`** : `.env` frontend committé avec `VITE_API_URL=http://localhost:8000/api`.
3. **Déconnexions aléatoires** : course de rafraîchissement 401 concurrent côté frontend.

## 2. Architecture réelle

### 2.1 Backend (`planner_backend/`)

```
planner_backend/
├── planner/            # config projet
│   ├── settings.py     # env, JWT, CORS, Cloudinary(mort), logging
│   ├── urls.py         # /admin, /api/, /api/auth/refresh/, /api/health/
│   └── wsgi.py         # patch psycogreen (inerte sans worker gevent)
├── core/               # app principale
│   ├── models.py       # 10 modèles, tous FK User CASCADE
│   ├── views.py        # 1035 lignes : toute la surface API
│   ├── serializers.py  # DRF serializers (plusieurs morts)
│   ├── urls.py         # routes + router DRF
│   ├── admin.py        # 6 modèles enregistrés sur 10
│   └── management/commands/setup_demo_user.py  # seed demo/demo123 (committé)
├── services/
│   ├── agent/          # PlannerAgent (boucle agentique Claude) + 9 outils
│   ├── llm/            # base / claude / gemini (abstraction provider)
│   ├── scheduling/     # overlap.py (testé) + test_overlap.py
│   ├── ai_scheduler.py # génération déterministe de planning (594 l.)
│   ├── ai_insights.py  # insights règle-based (1107 l.)
│   ├── document_processor.py # upload → OCR/vision → LLM → RecurringBlock (1044 l.)
│   ├── pdf_extractor.py # pdfplumber + OCR fallback
│   └── calendar_sync.py # placeholder mort (NotImplementedError partout)
├── utils/helpers.py    # retry_with_backoff, run_in_background (thread), time-math
├── clear_blocks.py     # ⚠ script committé : delete GLOBAL non scopé
├── check_extraction.py # script dev committé
├── amelioration.md     # notes dev committées
└── =24.0               # ⚠ fichier junk 0 octet committé (artefact `pip >=24.0`)
```

**Modèles de données** (`core/models.py`, schéma cohérent avec les 9 migrations, `makemigrations --check` = « No changes ») :
`UserProfile` (préférences, auto-créé via signal post_save), `UploadedDocument`, `RecurringBlock` + `RecurringBlockCompletion`, `Task` + `ScheduledBlock`, `TaskHistory`, `ConversationMessage`, `Goal`, `SharedSchedule`.

Point structurel : **4 représentations parallèles de la complétion** (`Task.completed`, `ScheduledBlock.actually_completed`, `RecurringBlockCompletion`, `TaskHistory`) non réconciliées, source probable d'incohérences.

### 2.2 Frontend (`day-wise-bot/src/`)

```
src/
├── services/api.ts     # 742 l. : fetchWithAuth + tous les groupes d'appels + flow 401→refresh
├── services/notifications.ts # Web/Capacitor, aucun appel backend
├── hooks/              # useAuth, useSchedule, useChat, useInsights (TanStack Query)
├── stores/             # authStore, chatStore, scheduleStore, onboardingStore (Zustand persist)
├── pages/              # Index, Login, Profile, Goals, Knowledge, SharedSchedule (1082 l.)
├── components/
│   ├── schedule/       # WeekView (908 l.), DayColumn, TaskBlock, MobileDayView, AgendaView...
│   ├── chat/           # ChatContainer, ChatInput, InteractiveInputs, MarkdownRenderer...
│   ├── insights/, goals/, onboarding/, settings/, profile/, auth/, landing/
│   └── ui/             # primitives shadcn (non auditées en détail)
├── middleware.ts       # Vercel edge : rewrite bots /monplanning → OG serverless
└── api/monplanning/[username].ts # OG serverless (fetch /planning/<username>/)
```

State : Zustand (persisté) + TanStack Query. La frontière réseau tient entièrement dans `services/api.ts`.

## 3. Flux de données

### Auth
`Login.tsx` → `checkEmail` (oracle d'existence) → `login`/`register` → réponse `{user, tokens:{access,refresh}}` → tokens en `localStorage` + mirror partiel dans `auth-storage` Zustand. Aucun `ProtectedRoute` : la protection des pages n'est qu'un effet de bord du `401 → forceLogout`. Rafraîchissement token **sans single-flight** (cause de déconnexions, §03).

### Chat (feature centrale, cassée)
`ChatInput` → `useChat.sendMessage` → **multipart** `POST /api/chat/` → `ChatView` → `PlannerAgent.process_message` → boucle agentique (max 8 tours) → `ClaudeProvider.generate_with_history` → **Anthropic 404 (modèle retiré)** → exception avalée en `LLMResponse(text="Erreur ... IA")` → renvoyée 200 comme réponse de l'assistant. Aucun fallback Gemini. Le contenu du document uploadé n'est jamais transmis au LLM (seul le nom de fichier l'est).

### Génération de planning (fonctionne, déterministe)
`POST /api/schedule/generate/` (et **chaque** `POST /api/tasks/`) → `AIScheduler.generate_schedule(num_days=7)` **synchrone dans la requête**. Algorithme glouton : intervalles occupés (RecurringBlock + ScheduledBlock) → slots libres 08:00-22:00 → scoring tâche/slot → placement. Pas de LLM sur ce chemin (`optimize_with_ai` est du code mort). Bugs de dates UTC vs Europe/Paris (§03).

### Insights (fonctionne, règle-based)
`/api/insights/*` : suggestions/patterns/conflicts/reschedule calculés en Python + ORM. Seul `parse_scheduling_request` appelle Gemini. `AIInsightsService()` reconstruit un `genai.Client()` à chaque requête.

### Partage public
`PublicScheduleView` (`/api/shared/<uuid>/`, token non devinable = OK) et **`PublicPlanningByUsernameView`** (`/api/planning/<username>/`, `AllowAny`, aucun opt-in = **fuite PII**, §03).

## 4. Contrat d'API (front ↔ back)

Le cross-check exhaustif (42 routes backend vs 44 appels frontend) donne **5 écarts** :

| Sévérité | Écart |
|---|---|
| **Bloquant** | Le groupe `goals` du frontend (`api.ts:696-728`) appelle `GET/POST/PATCH/DELETE /api/goals/` : **aucune route ni ViewSet côté backend**. Atténué car code mort (Goals.tsx utilise un store local). Les objectifs ne sont donc jamais synchronisés serveur. |
| Majeur | `tasks.list` envoie `?task_type=` que `TaskViewSet.get_queryset` ignore (seul `completed` est lu). Filtre silencieusement inopérant. |
| Mineur | `ChatView` renvoie `blocks_created` non déclaré dans l'interface `ChatResponse`. |
| Mineur | `GET /api/health/` jamais appelé (normal, sonde). |
| Mineur | Routes detail `documents/tasks` partiellement inutilisées. |

Le reste des contrats est aligné. La pagination DRF (`PAGE_SIZE=20`) tronque cependant silencieusement toutes les collections : `extractResults` ne lit que `data.results` et jamais `data.next` (perte des lignes au-delà de 20).

## 5. Zones à risque (synthèse, détail chiffré en 03-plan.md)

1. **Chemin IA (chat)** : modèle Claude retiré + provider figé + zéro fallback + erreurs avalées → feature morte.
2. **Config déploiement** : `.env` frontend `localhost` committé ; `DEFAULT_FILE_STORAGE` ignoré par Django 5.2 (Cloudinary mort → uploads perdus à chaque redeploy Railway) ; aucun start-command (gevent/gunicorn jamais activé) ; pas de whitenoise (statics admin 404).
3. **Auth/robustesse** : course de refresh 401 concurrent (déconnexions) ; `token_blacklist` non installé alors que `BLACKLIST_AFTER_ROTATION=True` ; `LoginView` manuel qui ignore `is_active`.
4. **Sécurité/vie privée** : `/api/planning/<username>/` expose horaires + salles de n'importe qui ; aucun throttling (brute-force login, coût API IA illimité) ; uploads sans validation (bombe PDF, stored-XSS) ; SECRET_KEY fallback en dur.
5. **Autorisation** : IDOR sur `recurring-completions` (référence le bloc d'un autre user) ; outils agent destructifs (`clear_all`, `delete_task`) déclenchables par un simple booléen fourni par le LLM.
6. **Correction du planning** : dates UTC vs Europe/Paris (off-by-one), heure de deadline ignorée, `max_deep_work_hours_per_day` jamais appliqué.
7. **Dette** : SQLite en prod (verrous d'écriture), jobs LLM/scheduler synchrones dans la requête (p95 lié aux API tierces), `run_in_background` (thread daemon) qui fuit des connexions DB et est tué au redeploy → documents bloqués en `processed=False`, near-zéro couverture de test frontend, TypeScript `strict:false`.

> Détail bug par bug, failles et plan ordonné : voir `audit/03-plan.md`. Résultats de test : `audit/02-tests.md`.
