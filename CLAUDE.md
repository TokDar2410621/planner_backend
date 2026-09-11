# Planner AI backend : notes pour l'agent

Lis `ONBOARDING.md` pour la carte du système. Ce fichier condense les règles
que ton agent doit respecter dans CE dépôt.

## Commandes

```bash
venv/Scripts/activate            # Windows; source venv/bin/activate ailleurs
python manage.py test            # suite complete
python manage.py test core.test_agent_v2_documents   # un module
python manage.py makemigrations core --name <nom>
python manage.py runserver
```

## Règles non négociables

- **Codes de sortie vérifiés séparément.** Jamais `tests | tail && commit` :
  le pipe avale le RC. Sortie dans un fichier, RC lu explicitement, aucun
  commit ni merge sur un RC non nul.
- **Après chaque déploiement, suivre les logs** (`railway logs -s
  planner_backend`) jusqu'au boot sain (migrations, Listening at, workers
  démarrés). Un push mergé n'est pas un déploiement réussi.
- **Après `railway variables --set`** : enchaîner `railway up -s
  planner_backend --detach`, sinon le prochain rebuild repart d'un GitHub
  périmé.
- **Secrets** : jamais matérialiser une valeur (ni en clair, ni dans un log,
  ni dans une URL; `str(e)` d'une HTTPError porte l'URL entière). Noms et
  présence seulement. Une clé fuitée se rote, le correctif ne suffit pas.
- **Conventions produit** : `day_of_week` 0 = lundi; heures murales
  `America/Toronto`; overnight `end_time < start_time` voulu, jamais
  « corrigé »; textes utilisateur en français québécois, tutoiement, zéro
  tiret long (em dash).
- **Agent v2** : toute action réelle passe par le registre
  (`services/agent_v2/registre.py`); les faits montrés à l'utilisateur sont
  rendus par du code (`redaction.py`); une écriture faite par le système hors
  de la boucle d'outils doit être inscrite au registre (voir
  `importation.py`), sinon la phase DIRE la niera.
- **Vérité API après écriture** : relire l'état avant d'affirmer qu'une
  mutation a réussi. Les outils destructifs exigent une confirmation humaine.
- Le banc (`benchmarks/`) consomme du crédit LLM réel : ne le lancer que sur
  demande. Un effondrement simultané v1+v2 = fournisseur (402 DeepSeek), pas
  le code.
