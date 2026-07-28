# Matrice de capacites de l'agent Planner AI

Date d'audit: 2026-07-28. Source de verite: `services/agent/tools/__init__.py`, `services/agent/agent.py`, `services/agent/system_prompt.py`, et implementations dans `services/agent/tools/*.py`.

## Surface exposee au LLM

`ALL_TOOLS` expose 27 outils. `ProcessUploadedDocumentTool` (`documents.py`) et `PresentQuickRepliesTool` (`interactive.py`) existent dans le code mais ne sont pas importes dans `ALL_TOOLS`; ils ne sont donc pas des capacites agent.

Retour commun: tous les outils renvoient `ToolResult(success: bool, data: dict, message: str)`. `execute_tool` capture les exceptions et renvoie `success=false` avec un message d'erreur. Les validations JSON Schema sont indicatives; seules les validations executees dans `execute()` sont garanties.

## Inventaire exhaustif des outils

| Outil | Fichier | Parametres | Retour / erreurs possibles | Validations faites | Donnees mutees | Verification apres ecriture / manques |
|---|---|---|---|---|---|---|
| `list_blocks` | `tools/blocks.py` | `day_of_week?`, `block_type?` | `blocks`, `count`; pas d'erreur explicite | Filtrage ORM seulement; enum non revalide cote outil | Aucune | N/A. Ne lit que les blocs recurrents visibles; pas les occurrences effectives ni one-off. |
| `create_block` | `tools/blocks.py` | `title`, `block_type`, `days[]`, `start_time`, `end_time`, `location?`, `flexibility?`, `is_night_shift?` | `created[]`, `skipped[]`; erreurs enum, longueur, heure invalide, rollback exception | Type/flex enum, max length titre/lieu, parse heure, jours 0-6, anti-doublon exact, conflits fixes via `find_recurring_conflicts`, transaction multi-jours | Cree des `RecurringBlock` hebdomadaires actifs | Pas de read-back post-commit; pas de cle d'idempotence; pas de date de fin, intervalle bihebdo, portee future, permission externe. `success=true` si au moins un jour cree, meme avec skips. |
| `update_block` | `tools/blocks.py` | `block_id`, `title?`, `start_time?`, `end_time?`, `location?`, `block_type?`, `flexibility?`, `day_of_week?` | `block`; erreurs introuvable, enum, longueur, heure, jour, conflit | Ownership user + active, parse heures, jour 0-6, conflits fixes en excluant soi-meme | Modifie le `RecurringBlock` lui-meme, donc toute la serie hebdomadaire | Pas de read-back; pas d'`expected_version`; pas de portee occurrence/futures; pas de note/description/couleur; ambiguite resolue hors outil par le LLM. |
| `delete_block` | `tools/blocks.py` | `block_id` | `deleted_id`; erreur introuvable | Ownership user + active | Soft-delete de toute la serie (`active=false`) | Pas de confirmation forte dans l'agent; pas de read-back; pas de restauration de serie exposee; dangereux si le LLM choisit le mauvais bloc. |
| `clear_all_blocks` | `tools/blocks.py` | `confirm` | `deleted_count`, `reversible`; erreur si `confirm` absent/faux | `confirm` booleen dans l'appel | Soft-delete de tous les blocs recurrents actifs | `requires_confirmation=True` existe mais l'orchestrateur chat ne l'applique pas; un LLM peut fournir `confirm=true`. Pas de read-back. |
| `skip_block_occurrence` | `tools/blocks.py` | `date`, `block_type?`, `title?` | `date`, `title`, `block_type`; erreurs date, type, aucun bloc, plusieurs candidats | Date YYYY-MM-DD, enum type, resolution par jour/type/titre, candidat multiple bloque | Cree un `RecurringBlockException` pour une occurrence datee | `get_or_create` idempotent; pas de read-back; pas d'ID exception; ne couvre pas one-off ni serie supprimee. |
| `restore_block_occurrence` | `tools/blocks.py` | `date`, `block_type?`, `title?` | `date`, `title`, `restored`; erreurs date, type, aucun bloc, plusieurs candidats | Meme resolution que skip | Supprime une `RecurringBlockException` | `success=true` meme si aucune exception n'existait; pas de restauration de serie soft-deleted. |
| `list_tasks` | `tools/tasks.py` | `completed?`, `task_type?`, `limit?` | `tasks`, `count` | Filtrage ORM; enum/limites seulement schema | Aucune | N/A. Ne retourne pas explicitement les blocs planifies sauf via task relation. |
| `create_task` | `tools/tasks.py` | `title`, `task_type?`, `priority?`, `description?`, `deadline?`, `estimated_duration_minutes?`, `place_*?` | `task`; erreurs enum/longueur; geocodage best-effort | Type enum, longueur titre, deadline parsee, idempotence par titre actif, lieu user unique | Cree `Task`; peut creer/mettre a jour `UserPlace` | Pas de cle d'idempotence; pas de verification post-ecriture; `priority` hors schema est clamp cote modele seulement; creer une tache ne planifie pas un evenement. |
| `update_task` | `tools/tasks.py` | `task_id`, `title?`, `priority?`, `deadline?`, `description?`, `task_type?`, `place_*?` | `task`; erreurs introuvable, enum, longueur | Ownership, type enum, longueur titre, deadline parsee | Modifie `Task`; peut creer/associer `UserPlace` | Pas de `estimated_duration_minutes` en update; pas de version/read-back; resolution par nom laissee au LLM. |
| `delete_task` | `tools/tasks.py` | `task_id`, `confirm` | `deleted_id`; erreurs confirmation ou introuvable | `confirm` dans l'appel, ownership | Hard-delete `Task` et cascade ses `ScheduledBlock` | `requires_confirmation=True` non force par l'agent; irreversible; pas de read-back. |
| `complete_task` | `tools/tasks.py` | `task_id` | `task`; erreur introuvable | Ownership | Marque `Task` complete, ses `ScheduledBlock`, cree `TaskHistory` | Helper idempotent; pas de verification externe. |
| `get_today_schedule` | `tools/schedule.py` | `date?` | `date`, `day_name`, `blocks`, `free_slots`, `total_blocks`, `total_free_minutes`; erreur date | Date YYYY-MM-DD | Aucune | Lit l'etat effectif d'un jour: recurrents actifs moins exceptions, placements souples, one-offs; bon outil de verification lecture. |
| `get_week_schedule` | `tools/schedule.py` | `start_date?` | Resume par jour, `total_hours`; erreur date | Date YYYY-MM-DD | Aucune | Resume les recurrents, pas les `ScheduledBlock`; ne donne pas tous les details d'evenements one-off/free slots. |
| `find_free_slots` | `tools/schedule.py` | `date`, `min_duration_minutes?` | `free_slots`, `total_free_minutes`; erreur date | Date YYYY-MM-DD | Aucune | Utilise la fenetre 07:00-23:00 et l'occupation effective; pas multi-jour ni contraintes avancees. |
| `schedule_task_at` | `tools/schedule.py` | `title`, `date`, `start_time`, `end_time`, `task_type?`, `priority?`, `description?` | `scheduled_block`; erreurs titre, date, heure, duree nulle, conflit | Date/heure, conflit avec murs fixes et sommeil protege, upsert par tache active titre + date; task_type invalide replie vers `shallow` | Cree/reutilise `Task`, cree/met a jour `ScheduledBlock`, split overnight en deux morceaux | Pas de transaction globale; pas de lieu/trajet; pas de suppression one-off; pas de cle idempotence; pas de read-back; peut planifier hors fenetre 07:00-23:00 si horaire fourni. |
| `check_feasibility` | `tools/schedule.py` | `date`, `activities[{title,duration_minutes}]` | `feasible`, `placed`, `unplaced`; erreur date/activite invalide | Date, durees positives, solveur CP-SAT | Aucune | Verifie seulement; le LLM doit ensuite creer. Risque de fausse confirmation si la reponse assimile proposition et ecriture. |
| `organize_day` | `tools/schedule.py` | `date`, `apply?` | `applied`, `placed`, `skipped`, `moved`; erreur date | Date, solveur placement | Si `apply=true`, modifie les heures des `RecurringBlock` souples du jour de semaine | Portee dangereuse: une demande datee modifie la serie hebdomadaire, pas une occurrence. Pas de transaction ni version; pas de read-back. |
| `get_preferences` | `tools/preferences.py` | aucun | Preferences profil | Aucune | Aucune | N/A. |
| `update_preferences` | `tools/preferences.py` | `min_sleep_hours?`, `peak_productivity_time?`, `transport_time_minutes?`, `max_deep_work_hours_per_day?`, `onboarding_completed?` | `updated_fields` | Enum productivite seulement | Modifie `UserProfile` | Pas de validation numerique cote outil malgre schema; pas de read-back/version. |
| `list_goals` | `tools/goals.py` | `status?` | `goals`, `count` | Filtre ORM; enum non revalide | Aucune | N/A. |
| `create_goal` | `tools/goals.py` | `title`, `goal_type`, `description?`, `deadline?` | `goal`; erreurs enum/longueur; risque exception deadline string | Type enum, longueur titre | Cree `Goal` | Pas de parse deadline avant `_goal_to_dict`; risque `BACKEND_BUG` si deadline reste string en memoire. Pas d'idempotence/read-back. |
| `update_goal` | `tools/goals.py` | `goal_id`, `title?`, `description?`, `progress?`, `status?`, `deadline?` | `goal`; erreurs introuvable, enum, longueur | Ownership, status enum, longueur titre; modele clamp progress | Modifie `Goal` | Deadline non parsee; pas de read-back/version; progress range surtout garantie par modele. |
| `suggest_schedule_optimization` | `tools/planning.py` | `focus?` | `analysis`, `suggestions` | Aucune robuste cote outil | Aucune | Analyse simple des recurrents et taches; aucune application; pas de solveur semaine, trajet, contraintes complexes. |
| `detect_conflicts` | `tools/planning.py` | `day_of_week?` | `conflicts`, `count` | Aucune revalidation jour | Aucune | Detecte seulement chevauchements simples de blocs recurrents non-overnight; ignore `ScheduledBlock`, trajets, overnight, exceptions et logique flexible/fixe avancee. |
| `get_productivity_stats` | `tools/analytics.py` | `period_days?` | Stats taches/blocs/streak | Aucune revalidation range cote outil | Aucune | N/A pour agenda; pas de verification de contraintes. |
| `present_form` | `tools/interactive.py` | `inputs[]` | `interactive_inputs`; erreurs inputs vides/options manquantes | Validation partielle des options pour choice types | Aucune DB; renvoie UI | Pas de garde schema robuste si provider envoie un type invalide; ne resout aucune capacite agenda seule. |

## Orchestration agent

- Garde anti-fausse confirmation: `COMPLETED_MUTATION_RE` bloque une affirmation de succes seulement si une mutation a ete tentee ce tour et qu'aucune n'a reussi. Il ne bloque pas une hallucination sans appel d'outil, ni un succes partiel maquille si au moins une mutation a reussi.
- Garde anti-doublon: `executed_calls` evite deux appels identiques dans le meme tour de chat, mais pas entre deux requetes, deux appareils ou apres retry reseau.
- Footer realite: apres `schedule_task_at`/`create_block`, l'agent peut ajouter un planning reel du jour. Cela aide pour les creations datees, mais ne verifie pas toutes les mutations.
- Les flags `requires_confirmation` des outils destructifs ne sont pas appliques par `execute_tool` ou `ChatView`.

## Matrice intentions-outils

`OK` signifie qu'aucun manque structurel majeur n'est visible dans le code, sous reserve d'une bonne orchestration LLM. Les autres valeurs sont des codes de la taxonomie du brief.

| Demande utilisateur | Intention | Action reelle attendue | Outil necessaire | Outil existant | Resultat attendu | Risque |
|---|---|---|---|---|---|---|
| 1,5,8: aujourd'hui, midi, premier truc demain | Lecture jour/date | Lire planning effectif et slots | get schedule day | Oui: `get_today_schedule` | Reponse fondee sur blocs effectifs, sans mutation | OK |
| 2,4: prochain cours, heure de fin | Calculer prochain/dernier evenement | Lire jour/semaine puis trier | get_next_event / get schedule | Partiel: `get_today_schedule`, `get_week_schedule`, `list_blocks` | Calcul humain par LLM; pas d'outil dedie | BAD_ORCHESTRATION |
| 3,9,10: travail/sport/cours filtres | Filtrer type de bloc | Lire recurrents par type/jour | search/list events | Oui: `list_blocks` | Liste uniquement les blocs demandes | OK |
| 6,7: journee chargee, temps libre | Resume charge/free time | Lire slots et total libre | get free slots / summary | Oui: `get_today_schedule`, `find_free_slots` | Reponse avec heures libres reelles | OK |
| 11,16: ajouter cours/rdv a heure sans duree | Creation one-off | Creer evenement date avec fin | create event | Partiel: `schedule_task_at` exige `end_time` | Demander/choisir une duree clairement | MISSING_PARAMETER |
| 12,17: sport lundi soir, bloquer samedi matin | Creation imprecise | Choisir ou demander plage | create/block time | Partiel: `create_block`/`schedule_task_at` exigent heures | Proposition ou action avec hypothese explicite | AMBIGUITY_HANDLING |
| 13: deux heures d'etude demain | Planifier duree sans heure | Trouver slot puis creer | find slots + create event | Oui: `find_free_slots`, `schedule_task_at` | Creer un one-off si slot complet existe | OK |
| 14: pause a 15h sans duree | Creation incomplete | Demander/choisir duree | create event duration | Partiel | Ne pas inventer une duree comme fait certain | MISSING_PARAMETER |
| 15: travail vendredi 18-22 | Ajouter travail | Creer recurrent ou one-off selon contexte | create block/event | Oui, mais choix de portee par LLM | Clarifier si vendredi unique ou chaque vendredi | AMBIGUITY_HANDLING |
| 18: rappelle-moi d'etudier ce soir | Rappel/notification | Creer rappel notifie | create_reminder | Non | Dire limite ou creer tache sans promettre rappel | MISSING_TOOL |
| 19,20,26,27: decaler/avancer/changer debut-fin | Modifier horaire | Identifier evenement puis update/upsert | update/reschedule event | Partiel: `update_block`, `schedule_task_at` | Demander precision si plusieurs; modifier bonne portee | AMBIGUITY_HANDLING |
| 21: prolonger etude d'une heure | Modifier duree | Ajuster end_time/duration | update duration | Partiel: `update_block`; pas `estimated_duration` update task | Modifier si cible unique, sinon demander | MISSING_PARAMETER |
| 22,23: changer titre/lieu | Metadata bloc | Update bloc | update metadata | Oui: `update_block` titre/lieu | Mutation de la serie identifiee | OK |
| 24: ajouter une note | Note/description d'evenement recurrent | Ajouter champ note | update metadata | Non pour `RecurringBlock` | Dire limite ou convertir en tache si pertinent | MISSING_TOOL |
| 25: changer couleur sport | UI metadata | Modifier couleur | update metadata color | Non | Dire non supporte | MISSING_TOOL |
| 28-30,35: annuler occurrence recurrente | Sauter une date, garder serie | Creer exception datee | cancel occurrence | Oui: `skip_block_occurrence` | Exception datee; serie active | OK |
| 31: annuler rdv dentiste one-off | Annuler evenement ponctuel | Supprimer/cancel scheduled task | cancel/delete event | Non dedie; `delete_task` hard-delete avec confirm | Ne pas simuler; demander confirmation si hard delete | MISSING_TOOL |
| 32: enlever toutes etudes demain | Bulk occurrence delete | Skip plusieurs blocs | bulk cancel | Partiel: plusieurs `skip_block_occurrence` | Detail succes/echecs, pas de transaction | PARTIAL_FAILURE_HANDLING |
| 33: supprimer prochain evenement | Identifier prochaine occurrence puis annuler/supprimer | get_next_event + cancel scoped | get_event_by_id/cancel | Non dedie | Ne pas supprimer une serie par erreur | MISSING_TOOL |
| 34,106-110: enlever ce qui vient d'etre ajoute / undo | Historique/rollback | Revenir action precedente/version | undo/history | Non expose a l'agent | Dire impossible ou utiliser vrai historique | MISSING_TOOL |
| 36: supprimer tous cours lundi | Supprimer plusieurs series | `delete_block` sur chaque cours lundi | delete series bulk | Partiel: `list_blocks` + `delete_block` | Supprimer seulement cours lundi; detailler partiels | PARTIAL_FAILURE_HANDLING |
| 37-44: demandes ambigues/pronoms | Resolution ambiguite | Lister candidats ou demander precision avant mutation | disambiguation | Partiel: outils bas niveau, pas de garde centrale | Aucune mutation tant que cible/portee non confirmees | AMBIGUITY_HANDLING |
| 45: remets comme avant | Undo implicite | Restaurer etat precedent | undo/version | Non | Ne pas reconstituer de memoire | MISSING_TOOL |
| 46: flemme sport demain | Annuler occurrence sport | Skip date/type | cancel occurrence | Oui si unique | Exception datee; sinon demander lequel | OK |
| 47: prof ne vient pas | Annuler cours implicite | Identifier cours concerne | cancel occurrence | Partiel | Demander si plusieurs cours/profs possibles | AMBIGUITY_HANDLING |
| 48: boss une heure avant | Modifier travail | Update heure debut ou occurrence | update event | Partiel | Clarifier occurrence vs serie | AMBIGUITY_HANDLING |
| 49,51,54: etre libre autour contrainte | Contrainte/pref persistante | Deplacer/contraindre planning | constraints engine | Non | Proposer, ne pas promettre contrainte durable | MISSING_TOOL |
| 50: fais-moi respirer mercredi | Optimisation jour | Proposer/deplacer souples | organize day | Oui partiel: `organize_day` | `apply=false` proposition, `apply=true` seulement souples | OK |
| 52,64-73,104: optimiser/reorganiser semaine | Optimisation multi-jour | Solveur semaine avec objectifs | optimize_schedule | Non; seulement `suggest_schedule_optimization` et `organize_day` | Proposition honnete, pas application globale pretendue | MISSING_TOOL |
| 53: plus de temps pour reviser | Ajouter temps d'etude | Trouver slots et creer | find slots + create | Partiel | Demander volume/duree si absent | MISSING_PARAMETER |
| 55: dormir plus | Sommeil/prefs | Creer/update sommeil + pref | update prefs/block | Partiel | Hypothese modifiable; pas garantie d'optimisation | AMBIGUITY_HANDLING |
| 56-59,88: ajout/deplacement en conflit fixe | Detecter conflit | Refuser ou proposer alternative | conflict detection | Oui: `create_block`, `update_block`, `schedule_task_at`, `check_feasibility` | Aucun chevauchement fixe masque | OK |
| 60: travailler 12h lundi | Validation charge/duree | Verifier max travail, sommeil, contraintes | validate constraints | Partiel: conflits seulement | Alerter sur charge excessive | MISSING_VALIDATION |
| 61,99,101,103,105: multiples mutations | Transaction multi-action | Tout ou rien ou succes partiel clair | transaction/bulk | Non global | Detailler chaque action, rollback ou partiel | PARTIAL_FAILURE_HANDLING |
| 62,69,79,100,158,160: trajets | Temps de trajet entre evenements | Estimer/inserer trajets | estimate_travel_time | Non; seulement travel_minutes lieu pour occupancy | Ne pas pretendre optimiser les deplacements | MISSING_TOOL |
| 63: placer meme sans espace | Faisabilite stricte | Verifier puis refuser si aucun slot | check feasibility + create | Oui partiel | Ne creer aucun bloc hors contraintes | MISSING_VALIDATION |
| 74-84: contraintes complexes | Optimisation sous contraintes | Modele obligatoire/preference/verrou | constraint solver | Non general | Expliquer limites/compromis | MISSING_TOOL |
| 85-90: contraintes impossibles | Detection impossibilite | Valider et expliquer conflit | validate constraints | Partiel: `check_feasibility` jour simple | Ne rien modifier en silence | MISSING_VALIDATION |
| 91: cours tous les lundis 9h | Recurrent hebdomadaire simple | Creer bloc weekly | create recurring | Oui: `create_block` | Bloc hebdomadaire actif | OK |
| 92,98: supprimer/restaurer occurrence | Exception datee | Skip puis restore exception | delete/restore occurrence | Oui: `skip_block_occurrence`, `restore_block_occurrence` | Serie conservee, exception creee/supprimee | OK |
| 93: decaler tous cours lundi | Update plusieurs series | Modifier chaque bloc lundi | update recurring event | Partiel: plusieurs `update_block` | Detailler conflit/partiel | PARTIAL_FAILURE_HANDLING |
| 94: occurrence et toutes les suivantes | Scope future | Split serie ou recurrence range | update future occurrences | Non | Dire non supporte ou demander conversion | MISSING_TOOL |
| 95: arreter recurrence fin du mois | Date de fin RRULE | Mettre `until` | update recurring rule | Non | Dire non supporte; proposer skips manuels | MISSING_TOOL |
| 96: un lundi sur deux | Recurrence interval | RRULE interval=2 | create recurring advanced | Non | Dire non supporte | MISSING_TOOL |
| 97: vacances | Exceptions par plage | Bulk skip range | bulk exceptions | Non dedie | Ne pas promettre sans creer toutes exceptions | MISSING_TOOL |
| 102: liberer vendredi apres-midi | Bulk delete/move scoped | Lister puis skip/delete/move | bulk scoped mutation | Partiel | Clarifier supprimer vs deplacer; partiels detailles | PARTIAL_FAILURE_HANDLING |
| 111-115: demain, vendredi prochain, ce soir, apres travail, dans deux semaines | Dates relatives simples | Convertir en date absolue puis outil | date parser | Partiel: LLM + prompt date | Correct si date absolue bien derivee | BAD_ORCHESTRATION |
| 116-118: deuxiemes lundis, dernier vendredi vacances, semaine prochaine nuancee | Regles calendrier avancees | Parser recurrence/holidays | recurrence/calendar parser | Non | Demander precision ou dire limite | MISSING_TOOL |
| 119-122: modifier calendrier ecole/equipe/prof/ami | Source externe permissionnee | Verifier permission puis muter externe | verify_permissions + external API | Non | Refuser ou demander integration/permission | PERMISSION_FAILURE |
| 123: synchroniser Google | Integration calendrier | OAuth + sync | Google Calendar sync | Non expose; placeholder `NotImplementedError` | Dire non disponible | MISSING_TOOL |
| 124: importer horaire scolaire | Import document | Upload + extraction auto | document import | Hors outils LLM; endpoint chat peut joindre fichier | Demander fichier si absent; ne pas pretendre import sans fichier | OK |
| 125: envoyer nouvel horaire employeur | Communication externe | Email/message | send tool | Non | Dire non disponible; fournir recap | UNSUPPORTED_REQUEST |
| 126-128: erreur reseau/success false/timeout | Echec technique | Ne pas confirmer; retry option | robust tool errors | Partiel: exceptions capturees, guard failure | Message "aucune modification" si echec | FALSE_SUCCESS |
| 129: id vide retourne | Validation resultat | Refuser succes sans id | verify event state | Non general | Echec technique | MISSING_VALIDATION |
| 130-132: deja supprime/inaccessible/perte auth | Etat/permission | Lire et dire rien modifie | state/permission checks | Partiel ownership local; pas externe | Pas de fausse reussite | PERMISSION_FAILURE |
| 133,136: ecriture OK mais verification echoue ou etat inchange | Verification post-write | Read-back obligatoire | verify_event_state | Non general | Ne pas annoncer etat non verifie | STATE_MISMATCH |
| 134-135: outils partiels/mal formes | Robustesse orchestration | Detailler partiel/schema error | transaction + validation | Partiel | Pas de succes complet si un outil echoue | PARTIAL_FAILURE_HANDLING |
| 137: double ajout sport demain 18h | Idempotence creation | Eviter doublon | idempotency key/upsert | Partiel: `schedule_task_at` upsert, `create_block` exact guard | Pas de doublon exact | OK |
| 138: suppression repetee | Idempotence delete | No-op sur deja supprime | idempotent delete | Partiel: skip idempotent; delete series second fail | Ne pas toucher une autre entite | IDEMPOTENCY_FAILURE |
| 139-141: retry reseau/double confirmation/2 appareils | Idempotence distribuee | Cle idempotence et verrou | idempotency key | Non | Eviter doubles ecritures cross-request | IDEMPOTENCY_FAILURE |
| 142-146: concurrence/stale state | Controle optimiste | expected_version/updated_at | verify version | Non pour outils agent | Ne pas ecraser en silence | CONCURRENCY_FAILURE |
| 147-153: hors perimetre physique/institutionnel/deviner sans acces | Limite capacite | Refuser et ramener au planning | none | Oui: pas d'outil mutation externe | Explication honnete, aucune mutation | UNSUPPORTED_REQUEST |
| 154-157,159: raisonnement sans modif | Analyse planning | Lire et calculer | schedule/stats | Oui partiel | Aucune ecriture; chiffres fondes sur etat | OK |
| 158,160: deplacements/retards | Analyse trajet/risque | Estimer travel entre events | estimate travel | Non | Reponse heuristique seulement, signaler limite | MISSING_TOOL |

## Manques structurels confirmes

- Annuler vs supprimer: occurrence recurrente couverte, mais one-off cancel/delete dedie absent; `delete_block` supprime la serie.
- Recurrents avances: pas de portee "occurrence et suivantes", pas de date de fin, pas d'intervalle bihebdo, pas de vacances/plage.
- Undo/historique agent: `SchedulePlanChange` et `/schedule/undo/` existent pour le replan backend, mais aucun outil agent `get_action_history`/`undo_last_action`/`restore_schedule_version`.
- Permissions/calendriers externes: aucun `verify_permissions`, aucun outil Google/Apple/ecole/equipe/ami, `CalendarSync` est placeholder.
- Idempotence: protections locales partielles, pas de cle idempotence cross-request.
- Concurrence: pas de `expected_version`, controle optimiste, `STALE_STATE`, ni read-back general.
- Trajets: occupancy peut consommer `travel_minutes` associe aux lieux, mais aucun outil `estimate_travel_time` ni optimisation de deplacements entre evenements.
