# Boucle unique : rebuild de l'architecture agent v2 (2026-09-29)

## Constat (recherche bonnes pratiques 2026)

Le pipeline actuel fait 3 a 4 appels LLM sequentiels par tour : LIRE (lecture
typee, aujourd'hui en ombre + 2 regles), AGIR (boucle ReAct multi-episodes),
DIRE (redaction), plus les jugements Jev types. La norme prod 2026 : 2 a 5
appels par tour, plus de 3 sequentiels = odeur de design. Anthropic, OpenAI,
LangChain et pydantic-ai convergent : **une seule boucle d'agent avec des
outils**, entouree de code deterministe. Les etapes NLU et NLG separees sont
les deux sauts que les sources 2026 appellent explicitement du gaspillage.

## Cible

Un seul appel de boucle par tour (hors jugements Jev types, qui restent le
garde-fou deterministe « le modele juge, le code agit »).

```
message -> [code] memoire, document, choix en attente (D6, inchange)
        -> [LLM] BOUCLE UNIQUE : comprend + agit, rend ReponseDire typee
                 (prose, question, refs d'actions, lecture)
        -> [code] VERIFIER : verifier_prose() croise la prose avec le
                 Registre (recus d'actions reels) avant rendu
        -> [code] RENDRE : composer() assemble faits verifies + prose + question
```

## Ce qui disparait

- L'appel LLM de LIRE par tour (`lecture.demarrer`) : la lecture typee
  vient desormais de `ReponseDire.lecture`, renseignee par la boucle elle-meme.
  Le mode ombre (scaffolding de migration) est retire.
- L'appel LLM de DIRE (`_dire`, PROMPT_DIRE, REGLAGES_DIRE, `modele_dire`) :
  la prose vient de la boucle unique, verifiee par `verifier_prose()`
  (redaction.py) avant rendu. Pas de second appel LLM.
- `voix_agir` et l'apercu streame : supprimes. Seul le raisonnement est
  diffuse en direct ; la reponse structuree est verifiee avant les deltas.

## Ce qui reste (inchange)

- `demandes.py` + `jugement.py` : questions en attente, puces, gardes
  destructives, jugements Jev types. Seuil 0.8, repli prudent.
- `outils.py` : outils typés + gardes. Les lectures restent parallelisables,
  les mutations sequentielles.
- `registre.py` : le registre d'actions, etendu (recus, pending_confirmation).
- `mesure.py` : detection deterministe des fuites + `epurer_reponse`,
  desormais appliques au brouillon de la boucle (verifier-puis-rendre).
- `redaction.py` (`composer`), `rendu.py`, `prompts.py` (prompt AGIR),
  `memoire.py`, `boutons.py`, `suggestions.py`.

## Ce qui est ajoute

1. **Sortie structuree** (`output_type=ReponseDire` de la boucle) :
   `{ouverture, actions: [ActionCitee(ref, phrase)], suite, question,
   options, refs, lecture}` — la prose cite les ids d'action du Registre
   qu'elle affirme. Le prompt l'exige.
2. **verifier_prose** (`redaction.py`) : croise la prose avec les recus reels
   du registre avant rendu. Garantie actuelle : une prose qui affirme une
   action sans aucune reference valide est refusee (repli sur les faits) ;
   les phrases adossees a des refs verifiees survivent. Le croisement
   phrase-par-phrase des explications causales (« car ... ») est une cible
   documentee (test_23 en xfail), pas encore implementee.
3. **Recus d'idempotence** (`registre.py`) : chaque mutation porte
   `cle_operation` (tache + outil + empreinte parametres) et rend un recu
   stable. Timeout ambigu -> etat `pending_confirmation`, reconciliation par
   requete avant reemission (jamais de reexecution aveugle).
4. **Instrumentation P99** : la ligne du tour gagne les phases
   (jugement, boucle, verification, rendu) en plus d'agir/dire.
5. **Reprise bornee** : 1 retry sur stream vide (`UnexpectedModelBehavior`)
   dans la boucle unique.
6. **Suite d'evals FR** (`core/test_agent_v2_scenarios_fr.py`) : ~30 scenarios
   tires d'echecs reels, asserts sur l'ETAT FINAL en base + absence
   d'affirmations mensongeres. Poids le plus lourd aux checks d'etat.

## Ce qui est volontairement limite

- Prompt caching : borne au fournisseur (seul Anthropic le supporte
  vraiment) ; prompts deja statiques d'abord, knob documente.
- Streaming : la boucle unique streame sa reponse finale (contrat SSE
  inchange : status, thinking, tool, delta, done).
- FallbackModel, UsageLimits, retries bornes : deja en place dans
  `modeles.py` (chaine DeepSeek -> Gemini -> Muse, delai 60 s,
  max_retries=0) ; on ajoute le retry stream-vide et on garde.

## Ordre de construction

1. Evals FR (mesurer avant/apres).
2. Recus + pending_confirmation dans le registre et le runner d'outils.
3. ResultatTour + boucle unique (suppression LIRE/DIRE, voix_agir, apercu).
4. verifier_prose (redaction.py) + rendu.
5. Instrumentation, retry, cache-knob.
6. Suite verte, PR.
