# 06 — Moteur de planification intelligent (spec → architecture → phases)

Réponse d'ingénierie à la spec « planner intelligent » (objectifs en langage
naturel → planning réaliste, trajets, réajustement continu). Ce doc fige
l'architecture et découpe la livraison ; la Phase 1 est implémentée.

## Architecture retenue (hybride, conforme §2 de la spec)

```
Langage naturel ──> LLM (PlannerAgent, existant)          [comprendre]
                        │  extrait objectifs/durées/contraintes
                        ▼
Contraintes ──────> Moteur DÉTERMINISTE (AIScheduler)      [décider]
  fixes: RecurringBlock (cours/travail, verrouillés)
  protégées: sommeil (min_sleep_hours), trajets (commute), marges
  flexibles: Task (sport, lecture, études)
  préférences: energy_levels, peak_productivity_time
                        │
Lieux/trajets ────> services/commute.py                    [calculer]
  UserPlace.travel_minutes (déclaré ; Phase 3: API cartographique)
  départ = début - trajet - marge ; indispo = départ - préparation
                        │
                        ▼
              ScheduledBlocks + rapport `unplaced`         [expliquer]
              (jamais de planning impossible silencieux)
```

Le LLM ne produit **jamais** l'horaire final (risque de chevauchements) ; il
alimente le moteur déterministe. C'est déjà la topologie du code (agent → tools
→ AIScheduler), la spec la confirme.

## État par exigence

| Spec | État | Où |
|---|---|---|
| §1 formule départ/indisponibilité | ✅ Phase 1 | `services/commute.py` (test = exemple 17h18/17h03) |
| §4 contraintes fixes | ✅ existant | RecurringBlock actifs = imperméables |
| §4 sommeil protégé | ✅ existant | min_sleep_hours + récupération post-nuit |
| §4 flexibles + préférences | ✅ existant | Task + energy_levels/peak_productivity |
| §5 chevauchements/marges | ✅ Ph1 | `_get_blocked_times` (fenêtre par lieu) |
| §5 cap deep-work / répartition | ✅ existant | B12 (max_deep_work_hours_per_day) |
| §6 localisation | Phase 1 = lieux déclarés (mode « localisation désactivée » de la spec) ; live = Ph3 | UserPlace |
| §7 réajustement auto | Ph2 (événementiel) | aujourd'hui: régénération à la demande + reminders cron |
| §9 notifications départ | Ph2 | brancher commute_window sur send_reminders (« pars maintenant ») |
| §10 conflits honnêtes | ✅ Ph1 | `generate_schedule.last_unplaced` → réponse API `unplaced` |
| §8 niveaux d'automatisation | Ph2 (mode suggestion d'abord) | statut pending (0014) = brique déjà en place |
| §12 confidentialité | lieux = déclaratifs, zéro GPS ; EFVP Loi 25 à faire avant tout live-tracking | — |

## OR-Tools CP-SAT (étude demandée §2)

Verdict : **pas maintenant**. Le placement actuel est un glouton trié
(priorité/deadline/énergie) sur des créneaux libres — correct, explicable, et
O(n·slots). CP-SAT devient pertinent quand on aura : objectifs hebdomadaires
fractionnables (« 3×/semaine »), fenêtres multi-lieux chaînées (deux villes le
même jour), et optimisation globale multi-critères. À ce stade on remplacera
`_match_task_to_slot` par un modèle CP-SAT (variables d'intervalle par tâche,
NoOverlap, poids = préférences) **sans changer le contrat** : mêmes entrées
(slots, contraintes), mêmes sorties (blocs + unplaced). L'API `unplaced` est
déjà la forme du « core IIS » (contraintes irréconciliables) qu'un solveur
sait produire proprement.

## Phases

- **Phase 1 (LIVRÉE)** : UserPlace (+ CRUD `/api/places/`), profil
  `prep_time_minutes`/`safety_margin_minutes`, `services/commute.py`,
  intégration `_get_blocked_times` (fenêtre prépa+trajet+marge avant, retour
  après ; fallback transport plat sans lieu), rapport `unplaced` dans
  `/schedule/generate/`. Migration 0015.
- **Phase 2 — réajustement + effecteurs** : notifications « commence à te
  préparer / pars maintenant » (commute_window × send_reminders × Web Push
  livré) ; endpoint « je termine 30 min plus tard » → replanification partielle
  (verrouiller fixes, ne bouger que les flexibles affectés) ; modes
  suggestion/semi-auto (réutilise le statut pending).
- **Phase 3 — trajets vivants** : refresh `travel_minutes` via API
  cartographique (Google Routes / Mapbox) sur déclencheurs sobres (avant un
  événement, à l'ouverture de l'app) ; jamais de tracking continu ; les 3 modes
  de localisation de la spec ; EFVP Loi 25 préalable.
- **Phase 4 — optimisation globale** : objectifs récurrents fractionnables
  (« 1h de sport/jour, soir de préférence ») en vrais objets, bascule CP-SAT si
  la combinatoire le justifie.
