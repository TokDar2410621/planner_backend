# Recherche : un outil `ask_question_custom` pour Planner

Date : 2026-10-25. Lecture d'OpenCode sur sa branche par défaut `dev`, commit [`9e8c5f5`](https://github.com/anomalyco/opencode/commit/9e8c5f5402c77001368ba7497eaff1759d6b9147). Sources : code, tests et documentation officiels. Aucun comportement d'OpenCode n'a été exécuté localement dans cette recherche; les constats portent sur son code à ce commit.

## Décision proposée

Créer **un outil de question exposé au modèle**, avec une interface courte et des variantes typées `choix`, `texte`, `formulaire`, `confirmation`. Il produit une demande persistée et attend une réponse utilisateur. Il n'exécute aucune mutation. Le backend valide chaque question avant affichage et chaque réponse avant reprise. Pour une confirmation, le modèle choisit le moment et peut proposer le libellé; seul le backend peut créer l'autorisation liée à une action exacte déjà retenue. Ne pas donner à une option `Oui` ni à un texte libre le pouvoir de confirmer une suppression.

Cette interface réunirait les usages actuels de `poser_question`, `present_choices` et `present_form` sans dupliquer leurs contrôles. Le passage doit être progressif : anciens outils acceptés pendant une migration et nouveaux échanges testés de bout en bout avant retrait. C'est une proposition de conception pour Planner, pas une fonction existante d'OpenCode.

## Ce que fait OpenCode

1. Son outil `question` reçoit un tableau non vide de questions. Chaque entrée a une question complète, un en-tête court, des options `{label, description}` et `multiple` facultatif. La sortie est un tableau d'ensembles de réponses. La réponse libre est ajoutée par l'interface, donc le modèle ne doit pas créer une option `Autre` lui-même. Sources : [schéma](https://github.com/anomalyco/opencode/blob/9e8c5f5402c77001368ba7497eaff1759d6b9147/packages/schema/src/question.ts#L6-L22), [outil](https://github.com/anomalyco/opencode/blob/9e8c5f5402c77001368ba7497eaff1759d6b9147/packages/core/src/tool/plugin/question.ts#L11-L29), [documentation](https://github.com/anomalyco/opencode/blob/9e8c5f5402c77001368ba7497eaff1759d6b9147/services/www/src/docs/content/tools.mdx).

2. Avant l'affichage, l'outil vérifie sa permission `question`, puis convertit chaque question en champ `string` ou `multiselect` avec `custom: true`. Le formulaire attend la réponse; celle-ci revient au modèle dans le résultat de l'appel, sous forme structurée et de texte lisible. Un refus de permission devient un échec d'outil. Une fermeture sans message met fin à l'interaction par erreur typée; une annulation avec message devient un retour d'échec au modèle. Sources : [permission, conversion et retour](https://github.com/anomalyco/opencode/blob/9e8c5f5402c77001368ba7497eaff1759d6b9147/packages/core/src/tool/plugin/question.ts#L38-L133), [tests de réponse et annulation](https://github.com/anomalyco/opencode/blob/9e8c5f5402c77001368ba7497eaff1759d6b9147/packages/core/test/tool-question.test.ts#L133-L264).

3. Le formulaire a un identifiant, un propriétaire de session et des états `pending`, `answered`, `cancelled`. `reply` refuse un champ inconnu, une réponse invalide et une seconde réponse. Pour les champs à choix fermés, une valeur hors options est rejetée; `custom: true` rend la valeur libre recevable. L'endpoint de réponse vérifie que le formulaire appartient à la session demandée. Sources : [schéma d'état](https://github.com/anomalyco/opencode/blob/9e8c5f5402c77001368ba7497eaff1759d6b9147/packages/schema/src/form.ts#L136-L179), [validation et transition](https://github.com/anomalyco/opencode/blob/9e8c5f5402c77001368ba7497eaff1759d6b9147/packages/core/src/form.ts#L180-L340), [contrôle de session de l'endpoint](https://github.com/anomalyco/opencode/blob/9e8c5f5402c77001368ba7497eaff1759d6b9147/packages/server/src/handlers/session.ts#L36-L43), [endpoint de réponse](https://github.com/anomalyco/opencode/blob/9e8c5f5402c77001368ba7497eaff1759d6b9147/packages/server/src/handlers/session.ts#L656-L670).

4. Côté interface, la question reste dans un panneau interactif, avec sélections, saisie libre et états d'envoi. La réponse et l'annulation passent par des mutations distinctes; les échecs sont montrés dans un toast. Sources : [panneau](https://github.com/anomalyco/opencode/blob/9e8c5f5402c77001368ba7497eaff1759d6b9147/packages/app/src/session/requests/session-question-dock.tsx#L95-L166), [envoi et annulation](https://github.com/anomalyco/opencode/blob/9e8c5f5402c77001368ba7497eaff1759d6b9147/packages/app/src/session/requests/session-question-dock.tsx#L260-L320).

### Deux limites du modèle OpenCode pour Planner

- Le schéma de `question` ne borne pas le nombre ni la longueur des questions et des options. Le formulaire sous-jacent valide leur type et la réponse, mais il faudrait ajouter des limites produit à Planner. Source : [schéma](https://github.com/anomalyco/opencode/blob/9e8c5f5402c77001368ba7497eaff1759d6b9147/packages/schema/src/question.ts#L6-L22).
- Le formulaire d'OpenCode attend via un `Deferred` d'un service vivant en mémoire et un cache local. Une question en attente reste dans ce cache jusqu'à sa résolution ou l'arrêt du service. Copier cette attente dans un tour HTTP Planner serait fragile face à une fin de requête ou un redémarrage; Planner doit persister la demande et reprendre lors d'une nouvelle requête. Source : [cache, `ask` et `reply`](https://github.com/anomalyco/opencode/blob/9e8c5f5402c77001368ba7497eaff1759d6b9147/packages/core/src/form.ts#L8-L11) et [implémentation](https://github.com/anomalyco/opencode/blob/9e8c5f5402c77001368ba7497eaff1759d6b9147/packages/core/src/form.ts#L103-L210). Cette conclusion sur Planner est une inférence d'architecture.

L'outil `question` d'OpenCode sert à recueillir une réponse, pas à autoriser par lui-même une action risquée. Son contrôle `permission.assert(action: "question")` porte sur l'usage de l'outil de question. Planner doit conserver une autorisation de mutation indépendante. Source : [outil](https://github.com/anomalyco/opencode/blob/9e8c5f5402c77001368ba7497eaff1759d6b9147/packages/core/src/tool/plugin/question.ts#L49-L78).

## État de Planner et écart à combler

- `poser_question` demande 2 à 4 choix libres; `present_choices` demande 2 à 4 choix ancrés dans les blocs, tâches, jours ou créneaux. Après ancrage, `present_choices` coupe à quatre options : des candidats peuvent être cachés si le modèle en donne davantage. Sources locales : `services/agent/tools/poser_question.py`, `services/agent/tools/present_choices.py`.
- `present_form` permet des champs structurés et une option `Autre` pour `radio`. Cependant `ChatContainer.tsx` transmet actuellement un résumé textuel du formulaire, pas un identifiant stable et les valeurs structurées comme canal de contrôle. Source locale : `day-wise-bot/src/components/chat/ChatContainer.tsx:139-144` et `day-wise-bot/src/types/index.ts`.
- Les puces d'une question transmettent déjà un postback `{demande, option}`. Le backend garde la demande dans les métadonnées du message assistant; `demandes_en_attente` ne considère que les trois derniers messages de cet utilisateur et les demandes émises depuis moins de 30 minutes. Sources locales : `day-wise-bot/src/components/chat/ChatMessage.tsx:249-269`, `services/agent_v2/demandes.py:108-142`.

## Interface proposée

```python
class AskQuestionCustom(BaseModel):
    kind: Literal["choice", "text", "form", "confirmation"]
    question: str                 # une question, bornée, sans fait inventé
    options: list[Option] = Field(default_factory=list)  # 2 à 4 pour choice
    allow_other: bool = False     # interdit pour confirmation
    fields: list[Field] = Field(default_factory=list)  # uniquement pour form
    pending_action_id: str | None = None  # obligatoire pour confirmation
```

Le modèle ne fournit jamais `question_id`, `user_id`, `conversation_id`, `tool_call_id`, `expires_at`, le hash de cible ni les effets d'une option de confirmation. Ces valeurs sont créées et signées ou conservées côté serveur. Le modèle peut proposer `question` et les libellés de clarification. Pour `confirmation`, le code peut reformuler une question trompeuse et fixe les actions `confirmer` et `annuler` liées au `pending_action_id`.

La réponse de l'outil devrait être typée : `{status: "awaiting_user", question_id, expires_at}`. Le tour se termine alors sans prétendre qu'une réponse a déjà été reçue. La prochaine requête porte `{question_id, selected_option_id}` ou `{question_id, text/form_values}`. Le serveur valide la réponse, exécute si autorisé, produit un résultat d'outil avec reçu, et relance le modèle pour sa réponse finale. Cela adapte à Planner le principe [PydanticAI des outils différés](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/) sans supposer qu'une requête HTTP attend le clic de l'utilisateur. La documentation demande de reprendre avec l'historique et les résultats corrélés par ID et précise que l'approbation soumise par le client ne remplace jamais l'autorisation côté serveur.

## Validations et moyens de contrôle

| Étape | Vérification côté serveur | Retour si rejet |
| --- | --- | --- |
| Appel du modèle | `kind` valide; une seule question par tour; question courte; options uniques, 2 à 4 si choix fermé; champs de formulaire connus; pas d'effet d'action dans un libellé | Erreur d'outil explicite au modèle; il peut corriger dans la même boucle |
| Choix de planning | Chaque option reliée à un identifiant réel et encore accessible, sans troncature silencieuse. Si plus de quatre possibilités, demander un critère de tri ou proposer `Autre` | Candidats rejetés et raison au modèle |
| Confirmation | Action en attente créée par le garde; cible, portée, arguments et empreinte stockés côté serveur; option `Autre` absente; le modèle n'invente ni l'action ni son autorisation | Demande invalide, aucune écriture |
| Réponse utilisateur | Session, utilisateur, conversation, question et option corrélés; demande non expirée, non déjà consommée; valeurs du formulaire validées; contrôle de cible et permissions refait | Erreur visible, aucune écriture; si cible changée, nouvelle question |
| Mutation confirmée | Clé d'idempotence; une seule exécution; relecture de l'état réel; résultat typé transmis au modèle | Cause précise ou effet incertain, sans annoncer un succès |

Une question de clarification peut permettre une réponse libre. Une confirmation destructive exige un tap sur l'option exacte attachée à la demande, jamais un `oui` interprété librement. Les options du modèle sont des propositions d'interface; elles ne constituent pas une preuve métier.

## Découpage de mise en œuvre suggéré

1. Définir un schéma `AskQuestionCustom` et un adaptateur vers les contrôles actuels des trois outils. Ne pas ajouter un quatrième chemin de validation concurrent.
2. Persister `PendingInteraction` ou un équivalent explicite dans la base, lié à l'utilisateur, la conversation, le tour, la cible et, au besoin, la mutation retenue. Réutiliser les demandes existantes durant la migration. Conserver un délai d'expiration et une consommation atomique.
3. Faire rendre le même objet `question` au web et à l'app iOS, avec options, texte libre, champs et état. Transporter les valeurs structurées du formulaire. Afficher chargement et erreur par action.
4. Sur la réponse, valider puis relancer la boucle avec le résultat réel. Retirer ensuite, un motif à la fois, les questions que `_choisir_question` fabrique après la boucle. Les questions du code qui expriment un garde nécessaire restent disponibles en secours si le modèle ne pose rien d'utile.
5. Tester le contrat de bout en bout : choix ancré valide/inventé, `Autre`, >4 possibilités, formulaire structuré, confirmation avant/après expiration, double tap, cible modifiée, autre conversation, annulation, panne réseau après écriture et reprise avec résultat au modèle.

## Mise en œuvre du 25 octobre 2026

L'outil `ask_question_custom` est maintenant le seul outil de question exposé au modèle V2. Les types `choice`, `text` et `form` réutilisent les validations existantes. Les choix de plus de quatre possibilités sont refusés avant toute troncature. Une demande `confirmation` isolée est refusée : l'outil d'action doit produire la confirmation liée à sa cible.

Le chat transmet les réponses de formulaire avec l'identifiant de question et les valeurs structurées. Le serveur vérifie le propriétaire, la conversation, la dernière question, le délai de 30 minutes, le format des valeurs et la consommation unique. Il transmet ensuite la réponse validée au modèle. Le frontend conserve la question dans les métadonnées après réouverture de la conversation.

La section « Interface proposée » ci-dessus décrit la cible étudiée. Cette livraison n'introduit pas de table `PendingInteraction` ni de signature de confirmation. Elle conserve le mécanisme de confirmation des actions déjà présent dans Planner. La suite backend complète a exécuté 2 127 tests et trouvé deux régressions liées au paramètre facultatif `question_answer` sur une sous-classe de l'agent; la compatibilité a été corrigée, puis les deux cas ont été rejoués avec succès. La suite complète n'a pas été relancée après ce correctif. Les tests frontend ont passé 366 cas, le contrôle TypeScript et le build.

## Limites de vérification

OpenCode a été lu au commit indiqué, pas exécuté. La documentation PydanticAI consultée est actuelle, tandis que Planner installe localement PydanticAI `1.30.1`; les classes `ApprovalRequired`, `DeferredToolRequests` et `DeferredToolResults` y sont importables, mais aucune compatibilité avec le mode réflexion et le transport HTTP/SSE de Planner n'a été testée ici. Le code de Planner évite aujourd'hui `output_type` pour ce mode; la documentation [PydanticAI](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/) précise que son arrêt sur `DeferredToolRequests` exige ce type de sortie. Le chemin le plus court est donc d'adapter les demandes persistées de Planner, puis de tester une éventuelle adoption directe des outils différés dans un prototype isolé.
