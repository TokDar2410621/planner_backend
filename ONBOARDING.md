# Onboarding : backend Planner AI

Bienvenue. Si ton mandat porte d'abord sur l'interface (c'est le cas au
départ), commence par l'onboarding du dépôt `day-wise-bot` : ce document-ci
est ta référence pour comprendre le système, ses conventions et l'API que le
client consomme. Public québécois, produit en français, tutoiement partout.

L'essentiel de l'API pour un dev d'interface : endpoints dans
`core/urls.py`, vues dans `core/views.py`, la vérité de ce que le client
reçoit. Le client les appelle via `src/services/api.ts` côté frontend.

## Le produit en une ligne

Planner AI (planneria.app) : un planner intelligent pour étudiants et
travailleurs; blocs récurrents hebdomadaires, tâches, agent IA conversationnel,
import d'horaire en photo, alarmes iOS réelles.

## La carte du système

- **Ce dépôt** : Django 5 + DRF, PostgreSQL. Service Railway `planner_backend`
  (web, ASGI) dans le projet Railway `meticulous-strength`.
- **`send-reminders`** : second service Railway, MÊME code, autre commande de
  démarrage (voir `railway.toml`, dispatch par `RAILWAY_SERVICE_NAME`) :
  rappels push et cerveau quotidien toutes les 15 min.
- **`planner-mcp`** : troisième service Railway. Serveur MCP (FastMCP, 27
  outils) qui expose l'API au nom d'un utilisateur. Attention : son code
  (`mcp-server/` dans le dépôt racine) est GITIGNORÉ; la seule copie vit sur
  le poste de Darius. Premier chantier sain : le mettre dans son propre dépôt.
- **Frontend** : `day-wise-bot` (React/Vite), déployé par Vercel sur
  planneria.app, et embarqué tel quel dans la coque iOS (Capacitor).
- **Dépôt racine `planner-racine`** : docs (guides App Store, audits, specs
  agent v2), tests e2e. À demander aussi, c'est la mémoire du projet.

API de prod : `https://plannerbackend-production.up.railway.app/api`.

## Démarrer en local

```bash
python -m venv venv
venv/Scripts/activate            # Windows; source venv/bin/activate ailleurs
pip install -r requirements.txt
cp .env.example .env             # puis remplis (voir le fichier, noms commentés)
python manage.py migrate
python manage.py runserver
```

Tests : `python manage.py test` (900 et quelques, ~10 min; un module :
`python manage.py test core.test_partage_journee`).

**Règle absolue sur les tests** : jamais de pipe qui avale le code de sortie
(`test | tail && commit`). Écris la sortie dans un fichier, lis le RC
séparément, et rien ne se commit sur un RC non nul. Une PR rouge a déjà été
mergée en silence à cause d'un pipe.

## Où vit quoi

- `core/` : modèles (RecurringBlock, Task, ScheduledBlock, SharedSchedule,
  AppareilPush, ReveilPlanning...), vues API, migrations, ~60 fichiers de
  tests `test_*.py`.
- `services/agent/` : agent v1 (héritage, encore référencé par v2 pour les
  outils et certains contextes).
- `services/agent_v2/` : l'agent par défaut de tous les comptes. Doctrine :
  AGIR (boucle d'outils PydanticAI) écrit chaque action dans un REGISTRE;
  DIRE (rédaction, sortie structurée) ne peut citer que des références du
  registre; le compte rendu factuel est rendu par du CODE (`redaction.py`),
  jamais par le modèle. Toute écriture faite par le système hors de la boucle
  d'outils (ex. import de document) doit entrer au registre, sinon DIRE la
  nie (`importation.py` raconte le cas vécu). Specs complètes dans le dépôt
  racine, `docs/superpowers/specs/`.
- `services/apns.py` : réveil silencieux. Chaque écriture de bloc (signal ou
  on_commit) pousse un push APNs sans contenu; l'iPhone se réveille, repose
  ses alarmes, confirme (`ReveilPlanning.confirme_a`); relance jusqu'à 3 fois
  sinon. Guide détaillé : racine, `docs/reveil-silencieux-apns-guide.md`.
- `services/document_processor.py` : import d'horaires (vision Gemini).
- `benchmarks/` : banc de l'agent (harness, juges). Il consomme du crédit
  LLM réel : ne le lance pas pour le plaisir.

## Déployer

Un push sur `main` déclenche le build Railway. Dans tous les cas : **suivre
les logs jusqu'au boot sain** (`railway logs -s planner_backend`; attendu :
migrations, puis `Listening at`, puis 3 `Started server process`). Jamais
« c'est déployé » sur la seule foi du push.

- Déploiement immédiat sans push : `railway up -s planner_backend --detach`
  DEPUIS ce dossier (lancé d'un autre dossier, Railway reçoit le mauvais code).
- **Après tout `railway variables --set`** : enchaîne immédiatement
  `railway up -s planner_backend --detach`, sinon le prochain rebuild partira
  du GitHub périmé (rollback silencieux vécu).
- Le worker : `railway up -s send-reminders`.
- Railway s'opère au CLI (`railway link -p 97ee808f-062c-4bc1-8402-71124ab34837
  -e production -s planner_backend`). Ne jamais afficher la VALEUR d'un
  secret; noms et présence seulement.

## Conventions capitales (à ne jamais deviner)

- `day_of_week` : **0 = lundi**. Un vieux commentaire du frontend dit
  « 0=Sunday » : il ment.
- Heures murales `HH:MM`, fuseau produit `America/Toronto` (TIME_ZONE). Pas
  de conversion artisanale.
- **Overnight voulu** : `end_time < start_time` = bloc qui traverse minuit
  (quart de nuit, sommeil). Ne jamais « corriger » en inversant.
- Vérité API après chaque écriture importante : relis l'état avant d'affirmer
  qu'une action a réussi.
- Destructif (delete_block, delete_task, clear) = confirmation humaine.
- Zéro tiret long (em dash) dans tout texte destiné à l'utilisateur.

## Pièges déjà payés (ne les repaie pas)

- **ASGI obligatoire** : l'agent v2 tourne sur asyncio (PydanticAI); gevent
  et asyncio ne cohabitent pas. Le service tourne en gunicorn + UvicornWorker
  (voir `railway.toml`, le pourquoi est en commentaire). Pas de `--preload`.
- **DeepSeek à sec = 402 silencieux** : si les deux agents s'effondrent au
  banc en même temps, c'est le fournisseur (crédit), pas le code.
- **CORS local** : seul `http://localhost:5173` est admis par la prod. Un
  front local sur un autre port sera rejeté.
- **iOS rationne les pushes silencieux** (~6 réveils en 35 min puis plus
  rien) : espace les tests d'alarme de 15-20 min; lis les lignes `bilan:` et
  `repli alarmes` dans les logs plutôt que « ça a sonné ».
- **AlarmKit** a une limite système dynamique : le client plafonne à 4
  alarmes posées, purge les fantômes avant de poser, sérialise les syncs.
- Diagnostics agent : ligne `agent_v2 tour actions=... rejetees=... fuites=...`
  (une par tour); `rejetees`/`fuites` non nulles = le modèle a tenté de
  mentir et la garantie a coupé.

## Les accès qu'il te faut (Darius les accorde, valeurs jamais par écrit)

- GitHub : collaborateur sur `planner_backend`, `day-wise-bot`,
  `planner-racine` (compte TokDar2410621).
- Railway : membre du projet `meticulous-strength` (3 services + Postgres).
- Clés LLM en local : utilise TES propres clés (Gemini obligatoire, DeepSeek
  pour l'agent v2, Anthropic optionnel). Les clés de prod restent en place.
- Comptes de test : `demo.reviewer@planneria.app` (revue Apple) et
  `camille.vitrine@planneria.app` (captures marketing); mots de passe par
  gestionnaire, jamais par écrit. Ne crée jamais de données de test sur la
  prod sans le signaler.
- Sur demande selon les chantiers : Cloudinary (médias), boîte d'envoi email
  (variables EMAIL_*), App Store Connect et Codemagic (voir l'onboarding du
  dépôt `day-wise-bot`).
