# Audit Planner AI — Phase 4 : Serveur MCP

> Expose Planner comme serveur MCP pour que Claude (Desktop, Code, Cowork) lise et pilote ton planning directement. Implémenté dans `mcp-server/`, vérifié end-to-end en prod le 2026-07-21.

## 1. Ce que ça fait

Le serveur MCP (`mcp-server/server.py`, basé sur le SDK `mcp`/FastMCP) traduit chaque appel d'outil de Claude en un appel à l'API REST de Planner, **au nom de l'utilisateur** (jamais un compte partagé). Transport **HTTP streamable** (hébergeable sur Railway) ou **stdio** (local).

### Outils exposés (8)

| Outil | Action | Endpoint appelé |
|---|---|---|
| `list_tasks(completed?)` | Liste les tâches | `GET /tasks/` |
| `create_task(title, priority?, task_type?, deadline?, estimated_duration_minutes?)` | Crée une tâche | `POST /tasks/` |
| `complete_task(task_id, actual_duration_minutes?)` | Termine une tâche | `POST /tasks/{id}/complete/` |
| `get_schedule(start_date?)` | Planning de la semaine | `GET /schedule/` |
| `generate_schedule()` | Auto-planifie les tâches | `POST /schedule/generate/` |
| `complete_block(recurring_block, date)` | Coche un bloc récurrent | `POST /recurring-completions/` |
| `get_completions(start_date?, end_date?)` | Liste les complétions | `GET /recurring-completions/` |
| `chat(message)` | Parle à l'agent IA (crée/modifie via langage naturel) | `POST /chat/` |

## 2. Authentification (token par utilisateur)

Le backend expose un token long-lived par utilisateur (DRF `TokenAuthentication`, ajouté en Phase 4) :

```bash
# 1. Se connecter (JWT)
curl -X POST https://plannerbackend-production.up.railway.app/api/auth/login/ \
  -H 'Content-Type: application/json' -d '{"username":"...","password":"..."}'
# 2. Récupérer son token MCP (avec l'access JWT)
curl https://plannerbackend-production.up.railway.app/api/auth/mcp-token/ \
  -H "Authorization: Bearer <ACCESS_JWT>"
# -> {"token": "<40 chars>", "username": "..."}
#    POST sur le même endpoint = rotation du token.
```

Le serveur MCP utilise ce token via `Authorization: Token <key>` :
- **Hébergé (HTTP)** : le client Claude envoie l'en-tête `Authorization: Token <key>` à chaque requête ; le serveur agit comme cet utilisateur. Multi-utilisateur sans compte partagé.
- **Local (stdio)** : `PLANNER_TOKEN` dans l'environnement (mono-utilisateur).

## 3. Utilisation locale (stdio) — le plus simple

`mcp-server/` :
```bash
python -m venv venv && source venv/Scripts/activate   # (Scripts/ sous Windows, bin/ sous *nix)
pip install -r requirements.txt
```

Config client (ex. Claude Desktop `claude_desktop_config.json`, ou `claude mcp add`) :
```json
{
  "mcpServers": {
    "planner": {
      "command": "python",
      "args": ["C:/Users/Darius/Desktop/planner/mcp-server/server.py"],
      "env": {
        "MCP_TRANSPORT": "stdio",
        "PLANNER_TOKEN": "<ton token MCP>",
        "PLANNER_API_URL": "https://plannerbackend-production.up.railway.app/api"
      }
    }
  }
}
```
Ou en CLI : `claude mcp add --scope user planner -- python C:/Users/Darius/Desktop/planner/mcp-server/server.py` (puis poser `PLANNER_TOKEN`/`MCP_TRANSPORT=stdio` dans l'env).

## 4. Déploiement hébergé (Railway, HTTP streamable)

Le dossier `mcp-server/` est prêt (`railway.toml` : `python server.py`, écoute sur `$PORT` en `streamable-http`). Comme pour ton obsidian-mcp :

1. Nouveau service dans le projet `meticulous-strength` (ou repo séparé) pointant sur `mcp-server/`.
2. Variables Railway du service MCP :
   - `PLANNER_API_URL=https://plannerbackend-production.up.railway.app/api`
   - `MCP_TRANSPORT=streamable-http`
   - (ne PAS poser `PLANNER_TOKEN` : chaque utilisateur envoie le sien par en-tête)
3. `railway domain` pour générer l'URL publique, ex. `https://planner-mcp-production.up.railway.app/mcp`.
4. Config client (transport HTTP) :
```json
{
  "mcpServers": {
    "planner": {
      "type": "http",
      "url": "https://planner-mcp-production.up.railway.app/mcp",
      "headers": { "Authorization": "Token <ton token MCP>" }
    }
  }
}
```

## 5. Vérifié (end-to-end, prod)

Token obtenu (40 car.), puis via les outils MCP réels sur la prod : `list_tasks` (vide → puis peuplé), `create_task` (tâche + deadline correcte), `chat` (« C'est noté ! J'ai ajouté "Appeler le dentiste" pour demain » + tâche créée). Auth par-utilisateur confirmée. 4 tests backend sur l'endpoint token (`core/test_mcp_token.py`), suite à 203 tests.

## 6. Variables d'environnement (récap)

| Variable | Où | Valeur |
|---|---|---|
| `PLANNER_API_URL` | service MCP | URL de l'API backend |
| `MCP_TRANSPORT` | service MCP | `streamable-http` (hébergé) / `stdio` (local) |
| `PLANNER_TOKEN` | local stdio uniquement | token de l'utilisateur |
| `PLANNER_HTTP_TIMEOUT` / `PLANNER_CHAT_TIMEOUT` | optionnel | timeouts (s) |

Côté backend (déjà déployé) : `rest_framework.authtoken` + `TokenAuthentication`, endpoint `GET/POST /api/auth/mcp-token/`.

## 7. Plan de rollback

- **Serveur MCP** : c'est un service séparé, sans état. Le supprimer/arrêter côté Railway n'affecte pas le backend ni le frontend.
- **Backend** : le seul changement est additif (app `authtoken` + un endpoint + une classe d'auth en plus de JWT). Pour revenir en arrière : retirer `'rest_framework.authtoken'` de `INSTALLED_APPS`, retirer `TokenAuthentication` des `DEFAULT_AUTHENTICATION_CLASSES`, et l'endpoint `mcp-token`. La table `authtoken_token` peut rester (inerte) ou être supprimée par une migration inverse. Aucune donnée existante n'est touchée.
- Révoquer un accès : `POST /api/auth/mcp-token/` (rotation) invalide l'ancien token de l'utilisateur.
