# Audit Planner AI — Phase 5 : Intégrations (étude d'écosystème)

> Recensement des apps/services dont l'intégration apporte de la valeur à Planner, avec le **comment** (API / MCP / webhook / export), l'effort, les variables Railway et le plan de rollback. Classé **indispensable / forte valeur / gadget**. Recherche web faite le 2026-07-21 (sources en fin de doc). Adapté à ton contexte : étudiant (Moodle), cerveau Obsidian (MCP existant), topics ntfy, Google Calendar (stub `calendar_sync` + MCP Google connecté), colocataires.

Effort : **S** ≈ 0.5-1 j, **M** ≈ 1-3 j, **L** ≈ 3-5 j+.

## 0. Vision : de l'agenda au « life assistant manager »

Chaque canal fait passer Planner d'un agenda passif à un **assistant de vie proactif** : il voit tes échéances (Omnivox/Léa), tes mails, ton sommeil, et il agit tout seul (crée les blocs, te rappelle via push, ajuste ton planning) derrière une seule interface en langage naturel : le **chat** + le **serveur MCP** (Phase 4). Plus il y a de sources, plus il devient le point de contrôle central de ta vie étudiante. Cette étude est la feuille de route de ce pivot : chaque intégration = un **capteur** (entrée) ou un **effecteur** (sortie) de plus.

- **Capteurs (entrées)** qui nourrissent l'agent : Omnivox/ColNet, Google Calendar, Gmail, Moodle/Canvas, santé/sommeil, Todoist/Notion.
- **Effecteurs (sorties)** qui le rendent proactif : ntfy / Web Push (rappels), cerveau (mémoire longue), Telegram/Discord (dialogue), Google Calendar (écriture).

L'ordre de priorité ci-dessous suit cette logique : d'abord brancher les capteurs les plus riches, puis rendre l'agent proactif.

## 1. Architecture d'intégration (comment tout se branche)

Trois canaux, tolérants à la panne (timeout court + fallback silencieux, comme le cerveau) :

- **Sortant périodique** : un **cron natif Railway** (pas de Celery, cf. étape 3 D3) `python manage.py <sync>` qui pousse/pull (résumé quotidien vers le cerveau, sync calendrier, rappels ntfy).
- **Entrant temps réel** : des **webhooks** entrants (endpoints Django dédiés, HMAC/secret) pour Google Calendar (watch), Telegram/WhatsApp (bot), Moodle (polling faute de webhook).
- **Interactif** : le `calendar_sync.py` (actuellement stub `NotImplementedError`) devient le point d'entrée sync ; le serveur MCP (Phase 4) et le chat consomment ces données.

Principe transversal : **chaque intégration est optionnelle et dégrade en silence** si sa variable d'env est absente ou l'API indisponible.

## 2. Niveau INDISPENSABLE

| Service | Valeur | Comment | Effort |
|---|---|---|---|
| **Google Calendar (sync bidir.)** | Le planning de Planner et l'agenda réel de l'étudiant doivent être une seule vérité. | **API** `google-api-python-client` + OAuth (tu as déjà `GOOGLE_CLIENT_ID`). Sortant : créer/mettre à jour les events depuis les ScheduledBlock/RecurringBlock. Entrant temps réel : **watch channels** (POST « ça a changé » sur un endpoint HTTPS) + resync incrémental via **syncToken** (ne resync que le delta). Renouveler les canaux (ils expirent). | **L** |
| **ntfy (rappels push)** | Rappels de tâches/blocs sans app à installer ; tu as déjà des topics. | **HTTP** pur : `POST https://ntfy.sh/<topic>` (ou ton instance), corps = message, en-têtes `Title`/`Priority` (1-5)/`Tags`. Déclenché par le cron (X min avant un bloc/deadline). Zéro SDK. | **S** |
| **Omnivox / Léa + ColNet (TON école, Québec/CEGEP)** | LE cas d'usage « récupération d'emploi du temps » : devoirs, examens, horaire → blocs/tâches Planner, automatiquement. | **Pas d'API publique** (Skytech/CEGEP). Trois voies, par robustesse : **(1) export iCal/webcal** si ton cégep l'offre sur le module Horaire/Léa/« Mon agenda » (le plus propre) ; **(2) upload d'une capture/PDF de l'horaire → l'extraction OCR+LLM DÉJÀ dans Planner** (`services/document_processor.py`) crée les RecurringBlocks : **zéro nouveau code, marche tout de suite** ; **(3) scraping** (login + parse ; des scrapers Omnivox open-source existent comme base) : fragile, identifiants chiffrés + opt-in + respect des CGU. | **S** (upload/iCal) → **L** (scraping) |
| **Cerveau (Obsidian MCP)** | Contexte perso/objectifs/notes pour enrichir la génération de planning + résumé quotidien poussé. | **MCP existant** (`obsidian-mcp` sur Railway `gracious-joy`) + son API HTTP. Sortant : résumé quotidien dans `08-auto/` (cron). Entrant : le backend interroge le cerveau (token, timeout court, fallback). C'est l'étape 5.1/5.2 de ton prompt d'origine. | **M** |

## 3. Niveau FORTE VALEUR

| Service | Valeur | Comment | Effort |
|---|---|---|---|
| **Telegram (bot de saisie rapide)** | « Ajoute réviser maths demain » en 2s depuis le téléphone → passe dans le chat/agent Planner. | **Bot API** gratuit : webhook (`setWebhook` → POST des updates sur un endpoint Django) ou long-polling (`getUpdates`). `sendMessage` pour répondre. Mappe le chat_id ↔ user Planner (token MCP/lien de compte). **Le plus rentable des messageries** (gratuit, simple). | **M** |
| **Gmail (extraction d'échéances)** | Détecter deadlines dans les mails (fac, admin) → propositions de tâches. | **Gmail API** (tu as le MCP Gmail connecté) : lecture ciblée (labels/filtres), extraction via l'agent LLM existant. Prudence vie privée : opt-in + filtres stricts. | **M** |
| **WhatsApp (saisie rapide)** | Idem Telegram si tu préfères WhatsApp. | **Cloud API** (Meta, gratuit à l'accès) : webhook entrant + `messages` sortant. **Nuance coût** : 1000 conversations « service » gratuites/mois, et toute réponse dans la fenêtre 24h après un message entrant est gratuite → un bot « inbound-first » reste gratuit en pratique. Plus lourd que Telegram (vérif business, templates). | **M-L** |
| **Todoist / TickTick (import de tâches)** | Migrer/synchroniser depuis un gestionnaire existant. | **Todoist REST v2** : `https://api.todoist.com/rest/v2/tasks`, `Authorization: Bearer <token>` (token perso simple). **TickTick Open API** : OAuth2, `https://api.ticktick.com/open/v1` (plus lourd). Import ponctuel (cron/one-shot) → tâches Planner, dédup par titre. | **S** (Todoist) / **M** (TickTick) |
| **Notion** | Si tes cours/projets vivent dans Notion. | **API** (`notion-client`, integration token, bases de données → pages) ou le **MCP Notion** (déjà connecté). Pull des items datés → tâches. | **M** |

## 4. Niveau GADGET (plus tard / faible ROI)

| Service | Pourquoi gadget | Comment (si un jour) |
|---|---|---|
| **Outlook / Microsoft To Do** | Redondant avec Google Calendar si tu es sur l'écosystème Google. | Microsoft Graph API (OAuth Azure) — effort **L** pour peu de gain. |
| **Calendriers partagés colocataires** | Déjà couvert par Google Calendar (calendriers partagés natifs). | Rien à coder : partager un Google Calendar + le sync existant. |
| **Banques / budget** | Hors périmètre planning ; friction (agrégateurs type Plaid/Bridge, KYC, coût). | Agrégateur (Plaid/Bridge/Nordigen) — **L**, à éviter sauf pivot produit. |
| **Trackers d'habitudes** | Les RecurringBlock + completions de Planner FONT déjà le tracking d'habitudes. | À construire dans Planner plutôt qu'intégrer un tiers. |

## 4bis. Catalogue étendu (les autres intégrations)

### A. Autres LMS (hors Québec / si tu changes d'établissement) — ton école est sur Omnivox+ColNet (voir §2)
| Service | Comment | Effort |
|---|---|---|
| **Canvas LMS** | REST API, **Bearer token** (expire fin d'année civile) : `GET /api/v1/users/self/calendar_events`, `GET /api/v1/courses/:id/assignments`. Pull par cron → tâches/blocs. | M |
| **Google Classroom** | API CourseWork (champs `dueDate`/`dueTime`), OAuth Google (tu as déjà le client). Pull des devoirs datés → tâches. | M |

### B. Calendriers (au-delà de Google)
| Service | Comment | Effort |
|---|---|---|
| **Abonnement iCal / .ics (read-only)** — *le fallback universel* | S'abonner à N'IMPORTE quel `.ics` (fac, Moodle, calendrier de sport, agenda des colocs) : `pip install icalendar`, fetch + parse par cron → blocs. Marche partout où il n'y a pas d'API. **Quasi indispensable, très peu d'effort.** | S |
| **CalDAV** (Apple iCloud, Nextcloud, Fastmail) | Pour les non-Google : lib `caldav` python, lecture/écriture d'events. | M |

### C. Push natif (complète ntfy — ton front est PWA + Capacitor)
| Service | Comment | Effort |
|---|---|---|
| **Web Push (VAPID)** | Standard W3C, web/PWA sans dépendance tierce ni app à installer. Le `services/notifications.ts` du front s'abonne, le backend stocke les subscriptions + un cron envoie via `pywebpush`. **La vraie push web.** | M |
| **FCM (Firebase)** | Push natif iOS/Android via le wrapper Capacitor (clés VAPID côté Firebase). Cross-plateforme mais dépendance Google. | M-L |

*ntfy = zéro-effort/self-host (mais l'app ntfy requise) ; Web Push = intégré à ta PWA sans app tierce.*

### D. Messagerie (au-delà de Telegram/WhatsApp)
| Service | Comment | Effort |
|---|---|---|
| **Discord** | Bot + webhook, gratuit ; les groupes étudiants y vivent. Saisie rapide + notifs de groupe. | M |
| **Slack** | Idem si contexte asso/pro (slash-commands → chat Planner). | M |

### E. Méta-connecteurs (débloquent des centaines d'apps d'un coup) — forte valeur
| Service | Comment | Effort |
|---|---|---|
| **n8n** *(tu l'as déjà)* | Un **webhook générique** entrant/sortant sur Planner + tes workflows n8n → connecte Planner à Sheets, Airtable, RSS, etc. sans coder chaque intégration. Le meilleur rapport valeur/effort si tu utilises déjà n8n. | S (côté Planner) |
| **Zapier / Make / IFTTT** | Même principe, hébergé, pour le no-code. | S |

### F. Dev (tu es développeur) — forte valeur perso
| Service | Comment | Effort |
|---|---|---|
| **GitHub / GitLab** | Issues/PR qui te sont assignées → tâches ; deadlines de milestones → blocs. API REST + webhooks entrants. | M |

### G. Étude / recherche
| Service | Comment | Tier / Effort |
|---|---|---|
| **Anki** | Créer des blocs de révision selon les cartes dues (spaced repetition). Pas d'API cloud officielle (AnkiConnect = local) → export/plugin. | gadget / M |
| **Zotero** | Deadlines de rendu / bibliographie. API REST simple. | gadget / S |

### H. Santé & sommeil (informe le scheduling) — le profil a déjà `min_sleep_hours`, `peak_productivity_time`
| Service | Comment | Tier / Effort |
|---|---|---|
| **Google Fit / Apple Health / Fitbit / Oura** | Données réelles de sommeil/énergie → ajuster automatiquement les créneaux deep-work et l'heure de coucher. Colle à l'angle « productivité + repos » de Planner. OAuth variable selon le fournisseur. | forte valeur / L |
| **Strava** | Compléter les blocs sport automatiquement (webhook Strava). | gadget / M |

### I. Réunions
| Service | Comment | Tier / Effort |
|---|---|---|
| **Google Meet / Zoom / Teams** | Générer/attacher un lien de visio à un bloc. Meet est gratuit via l'API Google Calendar (conferenceData). | gadget / M |

### J. Capture rapide
| Service | Comment | Tier / Effort |
|---|---|---|
| **Google Tasks** | API officielle ; import/sync des tâches Google. | forte valeur / S |
| **Apple Reminders / Shortcuts** | Un raccourci iOS qui POST vers un webhook Planner (ou vers le bot Telegram). | forte valeur / S |
| **Google Keep** | Capture, mais API non officielle → passer par Google Tasks/Shortcuts. | gadget / M |

### Synthèse des ajouts par priorité
- **À ajouter au niveau indispensable** : **abonnement iCal/.ics** (fallback universel, effort S) et **Web Push** (vraie push de ta PWA).
- **Forte valeur** : **n8n** (tu l'as, débloque tout), **Canvas/Classroom** (selon ta fac), **Discord**, **GitHub**, **Google Tasks**, **santé/sommeil**.
- **Gadget** : Zotero, Anki, Strava, Zoom/Teams, Keep.

## 5. Variables d'environnement Railway (récap)

À poser sur le service backend, **toutes optionnelles** (absence = intégration désactivée) :

| Intégration | Variables |
|---|---|
| Google Calendar | `GOOGLE_CLIENT_ID` (déjà), `GOOGLE_CLIENT_SECRET`, `GOOGLE_OAUTH_REDIRECT_URI`, `GCAL_WEBHOOK_URL` |
| ntfy | `NTFY_BASE_URL` (défaut `https://ntfy.sh`), `NTFY_DEFAULT_TOPIC`, `NTFY_TOKEN` (si topic protégé) |
| Omnivox / ColNet (ton école) | pas d'API : voie upload (aucune var) ; ou iCal `OMNIVOX_ICAL_URL` ; ou scraping `OMNIVOX_BASE_URL`+identifiants chiffrés |
| Moodle | `MOODLE_BASE_URL`, `MOODLE_WS_TOKEN` |
| Cerveau | `CERVEAU_API_URL`, `CERVEAU_TOKEN` |
| Telegram | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_WEBHOOK_SECRET` |
| WhatsApp | `WHATSAPP_TOKEN`, `WHATSAPP_PHONE_ID`, `WHATSAPP_VERIFY_TOKEN` |
| Todoist / TickTick | `TODOIST_TOKEN` / `TICKTICK_CLIENT_ID`+`TICKTICK_CLIENT_SECRET` |
| Notion | `NOTION_TOKEN`, `NOTION_DATABASE_ID` |
| Canvas / Classroom | `CANVAS_BASE_URL`+`CANVAS_TOKEN` / (OAuth Google réutilisé) |
| iCal / CalDAV | `ICAL_FEED_URLS` (liste) / `CALDAV_URL`+`CALDAV_USER`+`CALDAV_PASSWORD` |
| Web Push / FCM | `VAPID_PUBLIC_KEY`+`VAPID_PRIVATE_KEY`+`VAPID_SUBJECT` / `FCM_SERVER_KEY` |
| Discord / Slack | `DISCORD_BOT_TOKEN`+`DISCORD_WEBHOOK_URL` / `SLACK_BOT_TOKEN`+`SLACK_SIGNING_SECRET` |
| n8n (webhook générique) | `N8N_WEBHOOK_SECRET` |
| GitHub | `GITHUB_TOKEN`, `GITHUB_WEBHOOK_SECRET` |
| Google Tasks | (OAuth Google réutilisé) |
| Santé (Fitbit/Oura/Fit) | `FITBIT_*` / `OURA_TOKEN` / (OAuth Google Fit) |

## 6. Plan de rollback

- Chaque intégration = un module isolé (`services/integrations/<nom>.py`) + un cron ou un webhook dédié. La désactiver = **retirer sa variable d'env** (le code fallback en silence) ou supprimer son cron Railway. Aucun impact sur le cœur.
- Les webhooks entrants (Google/Telegram/WhatsApp) sont des endpoints séparés protégés par secret ; les désenregistrer côté fournisseur (`setWebhook` vide, stopper le watch channel) coupe le flux sans toucher au backend.
- Les écritures externes (events Google, tâches) sont **additives et idempotentes** (clé externe stockée sur le modèle Planner pour éviter les doublons au re-sync) → un rollback ne corrompt pas les données Planner.
- Migration de schéma nécessaire seulement pour stocker les ID externes (ex. `ScheduledBlock.gcal_event_id`) : additif, réversible.

## 7. Reco de priorité

1. **ntfy** (S) : gain immédiat, quasi zéro effort, tu as déjà les topics.
2. **Cerveau** (M) : c'est l'étape 5 de ton prompt d'origine (résumé sortant + contexte entrant), forte valeur perso.
3. **Google Calendar** (L) : la plus structurante, mais la plus lourde (OAuth + watch + syncToken). À faire une fois le reste stable.
4. **Telegram** (M) : saisie rapide, très rentable, gratuit.
5. **Moodle** (M) : forte valeur étudiant, **mais** dépend de l'activation des web services par ta fac (à vérifier d'abord ; fallback iCal).

Le reste (Gmail, Todoist, Notion, WhatsApp) selon ton usage réel ; les gadgets seulement sur besoin explicite.

---

## Sources

- Google Calendar push/watch : [developers.google.com/workspace/calendar/api/guides/push](https://developers.google.com/workspace/calendar/api/guides/push)
- Moodle Web Services : [docs.moodle.org/dev/Web_service_API_functions](https://docs.moodle.org/dev/Web_service_API_functions)
- Omnivox/Léa (pas d'API ; scrapers OSS) : [github.com/Simard302/Omnivox-Web-Scraping](https://github.com/Simard302/Omnivox-Web-Scraping), [github.com/solonovamax/Omnivox-Scraper](https://github.com/solonovamax/Omnivox-Scraper) ; ColNet « Mon agenda » : [collegealma.ca/plateformes-etudiants](https://www.collegealma.ca/plateformes-etudiants/)
- Todoist REST v2 : [github.com/Doist/todoist-api-python](https://github.com/Doist/todoist-api-python) ; TickTick Open API : [help.ticktick.com](https://help.ticktick.com/articles/7055781495671095296)
- ntfy publish : [docs.ntfy.sh/publish](https://docs.ntfy.sh/publish/)
- Telegram Bot API : [core.telegram.org/bots/api](https://core.telegram.org/bots/api)
- WhatsApp Cloud API pricing : [authgear.com/post/whatsapp-api-pricing](https://www.authgear.com/post/whatsapp-api-pricing/)
- Canvas LMS REST API : [canvas.instructure.com/doc/api](https://www.canvas.instructure.com/doc/api/all_resources.html) ; Google Classroom CourseWork : [developers.google.com/workspace/classroom](https://developers.google.com/workspace/classroom/guides/manage-coursework)
- Web Push VAPID vs FCM : [firebase.google.com/docs/cloud-messaging/web](https://firebase.google.com/docs/cloud-messaging/web/get-started) ; Capacitor push : [capawesome.io/blog/the-push-notifications-guide-for-capacitor](https://capawesome.io/blog/the-push-notifications-guide-for-capacitor/)
