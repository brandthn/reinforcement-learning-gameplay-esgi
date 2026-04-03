# Guide d'Implémentation — Projet Deep Reinforcement Learning

> **Objectif de ce document :** Référence partagée pour tous les membres et contributeurs de ce projet. Il documente les décisions architecturales convenues, les conventions à suivre, et les limites entre ce qui est décidé et ce qui reste ouvert. **Si quelque chose n'est pas indiqué ici, ce n'est pas encore décidé — ne pas inventer de contraintes.**

> **Ce que ce document N'EST PAS :** Ce n'est pas le syllabus du professeur (voir `docs/PROJECT_INSTRUCTIONS.md`). C'est le contrat d'ingénierie et de conception.

---

## 1. Périmètre du projet

**Objectif :** Implémenter et évaluer 16 algorithmes de reinforcement learning sur 4 environnements, avec des modèles entraînés, des métriques, une GUI et un rapport.

**Environnements :** LineWorld, GridWorld, TicTacToe, Bobail

**Framework :** PyTorch (tout le code réseau de neurones)

**Convention de langue :** Code (noms de variables, fonctions) en anglais. Commentaires, docstrings en français. 
Fichiers de documentation dans `docs/` en français. Commenter de manière factuelle, simple et claire.

---

## 2. Structure du Dépôt

```
drl-project/
│
├── environments/
│   ├── __init__.py              # Registre + factory get_env()
│   ├── base.py                  # Classe abstraite Environment
│   ├── line_world.py
│   ├── grid_world.py
│   ├── tictactoe.py
│   └── bobail.py
│
├── agents/
│   ├── __init__.py              # Registre + factory get_agent()
│   ├── base.py                  # Classe abstraite Agent
│   ├── random_agent.py
│   ├── human_agent.py
│   ├── tabular_q.py
│   ├── value_based/             # Implémenté : dqn, ddqn, ddqn_er, ddqn_per
│   │   ├── __init__.py
│   │   ├── dqn.py
│   │   ├── ddqn.py
│   │   ├── ddqn_er.py
│   │   └── ddqn_per.py
│   ├── policy_gradient/         # À implémenter
│   │   ├── __init__.py
│   │   ├── reinforce.py         # 3 variantes REINFORCE
│   │   └── ppo.py
│   └── planning/                # À implémenter
│       ├── __init__.py
│       ├── random_rollout.py
│       ├── mcts.py
│       ├── expert_apprentice.py
│       ├── alpha_zero.py
│       └── muzero.py            # MuZero + variante stochastique
│
├── training/
│   ├── __init__.py
│   ├── trainer.py               # Boucle d'entraînement mono-joueur
│   ├── self_play.py             # Boucle d'entraînement bi-joueur (self-play)
│   ├── replay_buffer.py         # ReplayBuffer + PrioritizedReplayBuffer
│   └── networks.py              # Constructeur MLP partagé
│
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py             # Évaluation en mode inférence aux checkpoints
│
├── gui/
│   ├── __init__.py
│   └── app.py                   # GUI Pygame
│
├── docs/
│   ├── encoding.md              # Spécifications d'encodage état & action (livrable)
│   ├── decisions.md             # Journal des décisions d'architecture
│   └── algorithms.md            # Explications des algorithmes en langage courant
│
├── configs/                     # Fichiers de config YAML, organisés par agent
│   ├── random/
│   │   ├── line_world.yaml
│   │   └── ...
│   ├── tabular_q/
│   ├── dqn/
│   │   ├── grid_world.yaml          # expérience de base
│   │   ├── grid_world_sweep_lr.yaml # expérience de sweep
│   │   └── ...
│   ├── ddqn/
│   ├── ddqn_er/
│   └── ddqn_per/
│
├── results/                     # CSV de métriques, courbes d'entraînement, checkpoints modèles (gitignored)
│   └── .gitkeep
│
├── notebooks/                   # À créer : visualisation et analyse des résultats
│   └── results_analysis.ipynb
│
├── scripts/
│   ├── train.py                 # Entraîner un agent sur un env depuis une config
│   ├── train_sweep.py           # Expanser une config sweep → appeler train_single() par combinaison
│   ├── train_all.py             # Batch : lancer train.py ou train_sweep.py pour toutes les configs
│   ├── evaluate_all.py          # Réévaluation batch des modèles sauvegardés
│   ├── promote_best.py          # Promouvoir le meilleur modèle vers best/
│   ├── benchmark.py             # Benchmark du jeu aléatoire (parties/sec)
│   └── run_gui.py               # Lancer la GUI
│
├── tests/
│   ├── test_environments.py
│   ├── test_agents.py
│   ├── test_training.py
│   └── test_value_based.py
│
├── pyproject.toml               # Dépendances + métadonnées (géré par uv)
├── uv.lock                      # Versions verrouillées des dépendances
├── README.md
└── main.py
```

### Principes de structure

**Les sous-dossiers dans `agents/` sont purement organisationnels.** Ils regroupent les familles d'algorithmes pour faciliter la navigation. Ils N'impliquent PAS de code utilitaire partagé entre fichiers d'une même famille. Chaque fichier d'algorithme est autonome.

**Les agents de premier niveau** (`random_agent.py`, `human_agent.py`, `tabular_q.py`) sont en dehors des sous-dossiers car ils n'appartiennent à aucune famille.

**`training/networks.py`** contient le constructeur MLP partagé utilisé par plusieurs familles d'algorithmes. Il est dans `training/` car c'est un utilitaire d'entraînement, pas un algorithme.

**Le dossier `notebooks/`** est exclusivement dédié à la visualisation et l'analyse des résultats. Les notebooks lisent depuis `results/`, ils ne produisent jamais d'artefacts d'entraînement. Leurs sorties (graphiques, tableaux) alimentent directement le rapport et les slides.

### Décisions de structure encore ouvertes

- Si MuZero / AlphaZero nécessiteront un découpage interne supplémentaire (ex: sous-dossier avec des définitions de réseaux séparées) — à décider pendant l'implémentation selon la taille et la complexité des fichiers.
- Si des modules utilitaires supplémentaires sont nécessaires dans `training/` — à évaluer au fur et à mesure.

---

## 3. Interfaces Fondamentales

Ces deux classes abstraites sont le socle du projet. **Tout le reste — entraînement, évaluation, GUI, agents — dépend de ces interfaces.** Elles doivent être implémentées en premier et rester stables.

### 3.1 Interface Environnement

```python
# environments/base.py
from abc import ABC, abstractmethod
import numpy as np
import copy

class Environment(ABC):

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Réinitialise à l'état initial. Retourne le vecteur d'état."""
        ...

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Exécute une action.
        Retourne (next_state, reward, done).

        Pour les jeux à deux joueurs :
        - reward est du point de vue du joueur qui vient d'agir
        - next_state est du point de vue du NOUVEAU joueur courant
        """
        ...

    @abstractmethod
    def available_actions(self) -> list[int]:
        """Indices des actions légales pour le joueur/phase courant(e)."""
        ...

    @abstractmethod
    def state_description(self) -> np.ndarray:
        """État courant sous forme de vecteur float32 plat."""
        ...

    @abstractmethod
    def action_space_size(self) -> int:
        """Nombre total d'actions possibles (y compris les illégales)."""
        ...

    @abstractmethod
    def state_space_size(self) -> int:
        """Dimensionnalité de la sortie de state_description()."""
        ...

    def is_adversarial(self) -> bool:
        """Surcharger pour retourner True dans les jeux à deux joueurs."""
        return False

    def current_player(self) -> int:
        """0 pour mono-joueur. 0 ou 1 pour bi-joueur."""
        return 0

    def clone(self):
        """Copie profonde. Requis pour MCTS/AlphaZero/MuZero."""
        return copy.deepcopy(self)

    def render_text(self) -> str:
        """Représentation texte optionnelle pour le debug."""
        return ""
```

### 3.2 Interface Agent

```python
# agents/base.py
from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):

    @abstractmethod
    def act(self, state: np.ndarray, available_actions: list[int],
            training: bool = False) -> int:
        """
        Sélectionne une action.
        - training=True : peut explorer (epsilon-greedy, stochastique, etc.)
        - training=False : exploitation pure (greedy, déterministe)
        """
        ...

    def observe(self, state, action, reward, next_state, done):
        """Appelée après env.step() pendant l'entraînement. Par défaut : no-op."""
        pass

    def end_episode(self):
        """Appelée en fin d'épisode d'entraînement. Par défaut : no-op."""
        pass

    def save(self, path: str) -> None:
        """Sauvegarder le modèle/poids/tables."""
        pass

    def load(self, path: str) -> None:
        """Charger le modèle/poids/tables."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
```

### Pourquoi c'est important

La classe `Agent` est un **contrat** : la GUI, l'Evaluator, le Trainer et la boucle de self-play appellent tous `agent.act()` sans savoir quel algorithme est derrière. C'est ce qui permet à chaque algorithme de fonctionner avec chaque environnement et chaque pipeline d'évaluation/GUI sans cas particuliers.

La classe `Environment` est le même type de contrat : chaque boucle d'entraînement appelle `env.step()`, `env.available_actions()`, `env.state_description()` sans savoir si c'est LineWorld ou Bobail.

**Règle : Si vous implémentez un nouvel algorithme, vous devez respecter l'interface Agent. Si vous implémentez un nouvel environnement, vous devez respecter l'interface Environment. Sans exception.**

### Décisions d'interface encore ouvertes

- Si `observe()` est suffisant pour tous les agents apprenants, ou si certains algorithmes (ex: REINFORCE collectant des trajectoires complètes) nécessiteront un mécanisme différent de passage de données — à évaluer pendant l'implémentation. Le hook `end_episode()` existe à cet effet mais son usage exact par algorithme n'est pas prescrit.
- Si des méthodes de commodité supplémentaires sont nécessaires sur `Environment` (ex: `game_result()` pour le résultat final bi-joueur) — à décider lors de l'implémentation de la première boucle d'entraînement bi-joueur.

---

## 4. Architecture Bi-Joueur (Approche 1 — Perspective du Joueur Courant)

### Décision

Pour les jeux à deux joueurs (TicTacToe, Bobail), l'environnement présente toujours l'état du point de vue du **joueur courant.**

### Ce que ça signifie concrètement

- `state_description()` retourne un vecteur où le canal/section « mes pièces » désigne toujours les pièces du joueur dont c'est le tour, et « pièces adverses » désigne l'autre joueur.
- Quand `step(action)` est appelé, l'environnement exécute le coup, change le joueur actif, et le prochain appel à `state_description()` montre le plateau du point de vue du nouveau joueur.
- Le `reward` retourné par `step()` est du point de vue du joueur qui vient d'agir.
- Les agents ne savent jamais s'ils sont « joueur 1 » ou « joueur 2 ». Ils voient toujours « mes pièces » vs « pièces adverses ».

### Pourquoi cette approche

- Les agents sont agnostiques au joueur — le même DQN entraîné fonctionne en tant que joueur 1 ou 2.
- Le code d'entraînement pour les envs mono-joueur et bi-joueur partage la même interface agent.
- La simulation MCTS fonctionne naturellement — chaque `step()` dans l'arbre donne la vue du prochain joueur.
- Permet des appairages flexibles : agent vs random, agent vs heuristique, agent vs agent entraîné, agent vs humain — avec la même classe d'environnement.

### Attention pour les implémenteurs

**Transitions du replay buffer dans les jeux bi-joueur :** Quand l'agent apprenant (joueur 0) agit dans l'état `s`, le `next_state` retourné par `env.step()` est du point de vue du joueur 1. Pour les méthodes off-policy (famille DQN), la boucle d'entraînement doit suivre et apparier correctement les états « même-joueur ». Ce problème est résolu dans `self_play.py` par le mécanisme de **deferred observe** : la transition n'est livrée à l'agent que lorsqu'il rejoue (prochain état de sa perspective) ou en fin de partie.

**Convention de récompense :** Quand une partie se termine, la récompense terminale est positive pour le gagnant (de son point de vue). La boucle d'entraînement est responsable d'assigner la bonne récompense aux transitions de chaque joueur. Documenté dans `docs/decisions.md`.

---

## 5. Bobail — Conception Spécifique au Jeu

### Tours en deux phases

Chaque tour à Bobail se compose de deux sous-actions :
1. **Phase bobail :** Déplacer le jeton bobail d'une case (dans l'une des 8 directions)
2. **Phase pièce :** Déplacer l'une de vos 5 pièces aussi loin que possible dans une direction

**Exception :** Au tout premier tour de la partie, le joueur 1 ne fait que la phase pièce (pas de déplacement du bobail).

### Approche d'implémentation

L'environnement suit une variable `phase` interne. Chaque appel à `step(action)` gère une seule sous-action. L'espace d'actions et `available_actions()` dépendent de la phase courante. Le tour ne passe à l'adversaire qu'après les deux phases complétées.

**Récompense :** La sous-action intermédiaire (phase bobail) retourne `reward = 0` et `done = False`. Ce n'est qu'après la phase pièce que l'environnement vérifie les conditions de victoire et retourne potentiellement une récompense terminale.

### Encodage de l'état

C'est un livrable requis (voir `docs/encoding.md`). Approche générale :

- Plateau 5×5 → 25 cases
- 3 canaux : pièces du joueur courant, pièces de l'adversaire, position du bobail
- Vecteur d'état total : 75 valeurs float32

Les détails exacts de l'encodage (y compris si des informations supplémentaires comme la phase ou le numéro de tour sont incluses) sont documentés dans `docs/encoding.md`.

### Encodage des actions

Approche générale : encoder les actions comme `(case_départ, case_arrivée)` mappées sur un entier unique. Cela fonctionne uniformément pour la phase bobail et la phase pièce. Taille de l'espace d'actions : 625 (25 × 25).

### Conditions de victoire

1. Le bobail est déplacé sur votre rangée de départ (rangée 0 pour joueur 0, rangée 4 pour joueur 1) → vous gagnez
2. Le joueur courant ne peut pas déplacer le bobail (il est complètement entouré) → le joueur courant perd

---

## 6. Conventions Expérimentales

### Fichiers de config (YAML)

Chaque expérience est définie par un fichier YAML. **Aucun hyperparamètre n'est codé en dur dans les fichiers d'algorithmes.** Les constructeurs d'algorithmes reçoivent leurs paramètres depuis la config.

Le schéma exact des fichiers de config n'est pas prescrit — il varie selon la famille d'algorithmes. Mais chaque config doit inclure au minimum :
- `env` : nom de l'environnement
- `agent` : nom de l'algorithme
- `seed` ou `seeds` : graine(s) aléatoire(s)
- `training.num_episodes` : nombre d'épisodes d'entraînement

### Organisation des dossiers de configs

Les configs sont organisées par agent sous `configs/` :

```
configs/
├── random/
│   ├── line_world.yaml
│   └── ...
├── tabular_q/
├── dqn/
│   ├── grid_world.yaml              # baseline
│   ├── grid_world_sweep_lr.yaml     # variante sweep
│   └── ...
├── ddqn/
├── ddqn_er/
└── ddqn_per/
```

**Une config = une expérience** au moment de l'exécution. Les configs de base utilisent `scripts/train.py`. Les configs sweep (voir ci-dessous) utilisent `scripts/train_sweep.py`. `scripts/train_all.py` détecte quel type est chaque config et route en conséquence.

### Configs sweep

Une config sweep est une config normale avec une section `sweep:` supplémentaire qui déclare des axes de variation en notation pointée :

```yaml
sweep:
  agent_params.lr: [0.001, 0.0005, 0.0001]
  agent_params.batch_size: [32, 64]
```

`train_sweep.py` calcule le produit cartésien (ici 3 × 2 = 6 expériences), construit une config concrète pour chaque combinaison, et appelle `train_single()` depuis `train.py` directement (in-process). Le script de sweep est une pure couche d'expansion de config — `Trainer`, `SelfPlayTrainer`, agents et environnements ignorent les sweeps.

La clé `sweep:` est explicite pour éviter l'ambiguïté avec les paramètres à valeur liste comme `hidden_layers: [64, 64]`. Seules les clés déclarées sous `sweep:` sont expansées.

Chaque run expansé écrit son propre snapshot `config.yaml` dans `results/`, préservant l'invariant une-config-une-expérience pour la traçabilité.

### Nommage des expériences et stockage des résultats

```
results/
└── {env_name}/
    └── {agent_name}/
        └── {résumé_hyperparams}_seed{N}/
            ├── config.yaml          # Snapshot complet de la config (copié au lancement)
            ├── metrics.csv          # Métriques d'évaluation aux checkpoints
            ├── training_curve.csv   # Données d'entraînement par épisode
            └── model_{checkpoint}.pt
```

Le `{résumé_hyperparams}` est généré à partir des paramètres clés (ex: `lr0.001_gamma0.99_eps50k`). Le YAML complet à l'intérieur du dossier est la source de vérité — le nom du dossier sert uniquement à l'identification humaine rapide.

Chaque config peut optionnellement inclure un champ `experiment_label` pour des légendes de graphiques lisibles.

### Stratégie de graines (seeds)

- **Pendant le développement :** Utiliser une graine fixe unique pour la reproductibilité et le debug.
- **Pour les résultats finaux rapportés :** Lancer chaque expérience avec **plusieurs graines** (3-5) et rapporter moyenne ± écart-type. Cela démontre la rigueur statistique.
- Le script d'entraînement accepte une liste de graines et boucle dessus.
- Le CSV de résultats enregistre quelle graine a été utilisée.

### Checkpoints d'évaluation

Le syllabus exige des métriques à : 1 000 / 10 000 / 100 000 / 1 000 000 (si possible) épisodes d'entraînement.

À chaque checkpoint, l'**Evaluator** exécute l'agent en mode inférence pure (`training=False`) pour N parties et enregistre :
- Récompense moyenne (± std)
- Longueur moyenne d'épisode (± std)
- Temps moyen par action (ms)

**Distinction critique :** Ce sont des métriques pour la **policy obtenue**, pas des métriques d'entraînement. La courbe d'entraînement (récompense par épisode pendant l'entraînement) est stockée séparément.

---

## 7. Utilitaires Partagés

### Constructeur MLP (`training/networks.py`)

Une fonction unique qui construit un réseau feedforward à partir de la dim d'entrée, la dim de sortie et les tailles des couches cachées. Utilisé par DQN, DDQN, REINFORCE, PPO, et potentiellement les sous-réseaux AlphaZero/MuZero. Supporte les activations ReLU, Tanh et ELU.

### Replay Buffer (`training/replay_buffer.py`)

Deux classes :
- `ReplayBuffer` — échantillonnage uniforme (utilisé par DQN, DDQN, DDQN+ER)
- `PrioritizedReplayBuffer` — échantillonnage pondéré par priorité avec sum-tree (utilisé par DDQN+PER)

Partagés par toute la famille value-based. Les autres familles d'algorithmes peuvent ou non les utiliser (AlphaZero/MuZero pourraient nécessiter leur propre stockage de trajectoires — à décider pendant l'implémentation).

### Trainer (`training/trainer.py`)

Boucle d'entraînement mono-joueur générique. Appelle `agent.act()`, `env.step()`, `agent.observe()`, `agent.end_episode()`. Déclenche l'évaluation aux checkpoints configurés. Écrit `training_curve.csv` et `metrics.csv`.

### Self-Play (`training/self_play.py`)

Boucle d'entraînement bi-joueur. Gère l'alternance des tours entre l'agent apprenant et un adversaire. L'adversaire peut être n'importe quelle instance d'`Agent` (RandomAgent, heuristique, autre agent entraîné).

Utilise le mécanisme de **deferred observe** : la transition de l'agent n'est livrée que lorsqu'il a de nouveau la main (next_state correct de sa propre perspective) ou en fin de partie. Cela garantit que les cibles Q-learning utilisent des états du même joueur.

### Evaluator (`evaluation/evaluator.py`)

Exécute un agent entraîné en mode inférence. Collecte les statistiques de récompense, longueur d'épisode et temps par action. Supporte les modes mono-joueur et bi-joueur (avec adversaire). Écrit les résultats au format CSV.

---

## 8. GUI

**Technologie :** Pygame

**Fonctionnalités implémentées :**
- Sélection de l'environnement (LineWorld, GridWorld, TicTacToe, Bobail)
- Sélection de l'agent (ou humain) pour chaque joueur
- Sélecteur de modèle : affiche les runs entraînés disponibles (scannés depuis `results/`)
- Regarder l'agent jouer / jouer en tant qu'humain
- Fonctionne avec n'importe quelle sous-classe Agent via `agent.act()`
- Contrôles humains : touches fléchées (LineWorld/GridWorld), clic (TicTacToe), clic deux temps source→destination (Bobail)

**Pas de hiérarchie abstraite de renderers.** Chaque représentation visuelle d'environnement est gérée directement dans le code GUI via un dispatch dict.

**Chargement des modèles :** Convention `best/` (modèle promu par `scripts/promote_best.py`) + sélecteur permettant de choisir un run spécifique. Le `config.yaml` est lu depuis le même dossier que le modèle pour garantir la correspondance architecture/poids.

---

## 9. Stratégie de Documentation

### `docs/encoding.md` — Livrable REQUIS

Pour chaque environnement :
- Ce que le vecteur d'état représente, élément par élément
- Ce que chaque indice d'action signifie
- Pourquoi cet encodage a été choisi, quelles alternatives ont été considérées
- Rédigé en français

### `docs/decisions.md` — Journal des Décisions d'Architecture

Entrées courtes documentant les choix d'implémentation non-évidents. Format :

```markdown
## Titre de la décision

**Contexte :** Quel problème on essaie de résoudre.

**Décision :** Ce qu'on a choisi de faire.

**Pourquoi :** Justification. Pourquoi pas les alternatives.

**Références :** Sources si applicable.
```

Écrire une entrée **à chaque fois** qu'un choix est fait que quelqu'un pourrait questionner plus tard, notamment lors de la soutenance orale. Exemples : stratégie de masquage d'actions, reward shaping, fonctionnement du epsilon decay, choix d'architecture réseau, etc.

### `docs/algorithms.md` — Explications des algorithmes

Pour chaque algorithme, une explication en langage courant **avec vos propres mots** que vous pouvez présenter en confiance à l'oral. Structure par algorithme :
- Idée centrale (2-3 phrases)
- Comment il s'entraîne (le mécanisme clé)
- Ce qu'il ajoute par rapport à la version précédente/plus simple
- Limites connues ou points de vigilance

Rédigé en français à un niveau adapté à la soutenance orale.

### Documentation progressive

**Ne pas écrire toute la documentation à la fin.** Écrire les entrées de `docs/decisions.md` au fil de l'implémentation. Écrire les entrées de `docs/algorithms.md` en implémentant chaque algorithme. Cela sert à la fois de matériel d'étude et de préparation à la soutenance.

---

## 10. Règles de Sécurité d'Implémentation

Ces règles existent pour éviter que le travail d'une personne ne casse celui d'une autre.

### Règle 1 : Ne jamais modifier les interfaces de base sans consensus de l'équipe

`environments/base.py` et `agents/base.py` sont les contrats. Si vous pensez qu'une méthode doit être ajoutée ou modifiée, discutez d'abord. Modifier ces fichiers impacte tout.

### Règle 2 : Chaque algorithme doit passer le smoke test

Avant de considérer un algorithme comme « terminé », vérifier :
```python
agent = VotreAgent(state_size=env.state_space_size(), action_size=env.action_space_size(), ...)
state = env.reset()
action = agent.act(state, env.available_actions())
assert action in env.available_actions()
```

Cela doit fonctionner pour chaque environnement que l'algorithme est censé supporter.

### Règle 3 : Les algorithmes doivent gérer le masquage d'actions

Si l'environnement retourne `available_actions()` comme un sous-ensemble de toutes les actions, l'agent ne doit retourner que des actions de ce sous-ensemble. **Comment** chaque algorithme y parvient (ex: masquage des Q-values à -inf, filtrage de la sortie softmax, etc.) est un détail d'implémentation à décider par algorithme et à documenter dans `docs/decisions.md`.

### Règle 4 : Les environnements doivent implémenter clone() pour les algorithmes de planning

MCTS, AlphaZero et MuZero ont besoin de `env.clone()` pour simuler les états futurs. Le `copy.deepcopy` par défaut devrait fonctionner mais peut être lent. Si le profiling révèle que `clone()` est un goulot d'étranglement, l'environnement peut le surcharger avec une implémentation plus rapide.

### Règle 5 : Aucun hyperparamètre codé en dur

Tous les paramètres ajustables doivent provenir de la config. Les fichiers d'algorithmes ne doivent pas contenir de valeurs par défaut pour learning rate, gamma, epsilon, etc. qui s'appliquent silencieusement quand aucune config n'est fournie. Si un paramètre manque dans la config, lever une erreur plutôt que d'utiliser une valeur cachée.

### Règle 6 : Les résultats sont en écriture seule pendant l'entraînement

Les scripts d'entraînement écrivent dans `results/`. Les notebooks et scripts d'analyse lisent depuis `results/`. Rien ne doit à la fois lire et écrire dans le même dossier de résultats pendant un run d'entraînement.

---

## 11. Ce Qui N'Est PAS Encore Décidé

Les aspects suivants sont intentionnellement laissés ouverts et doivent être décidés pendant l'implémentation, puis documentés dans `docs/decisions.md` :

- Comment les variantes REINFORCE sont organisées en interne (une classe avec flags, ou trois classes)
- Si AlphaZero / MuZero nécessitent un découpage interne supplémentaire
- Comment l'adversaire heuristique de Bobail fonctionne (basé sur des règles ? quelles règles ?)
- Si des features d'état supplémentaires sont nécessaires dans l'encodage (numéro de tour, indicateur de phase, etc.)
- Si le replay de MuZero nécessite une implémentation séparée du ReplayBuffer partagé
- La structure des notebooks (un gros notebook vs plusieurs ciblés)
- Le contenu exact de la structure du rapport final

**Décidé depuis la rédaction initiale (voir `docs/decisions.md` pour les détails) :**

- ~~Architectures réseau exactes~~ → MLP configurable via param `hidden_layers` (D-013)
- ~~Plages d'hyperparamètres exactes~~ → Définies par fichier de config ; les configs sweep permettent l'exploration systématique
- ~~Layout exact de la GUI~~ → Implémenté dans `gui/app.py` (D-009, D-010)
- ~~Gestion des transitions bi-joueur dans le replay buffer~~ → Deferred observe dans `self_play.py` (D-002)
- ~~Parallélisation de `train_all.py`~~ → Séquentiel avec subprocess ; détecte automatiquement sweep vs régulier
- ~~Organisation des fichiers de config~~ → Sous-répertoires par agent sous `configs/` (voir §6)
- ~~Chargement des modèles dans la GUI~~ → Convention `best/` + sélecteur de modèle (D-018)
- ~~Hiérarchie d'héritage value-based~~ → DQN → DDQN → DDQN+ER / DDQN+PER (D-014, D-015, D-016, D-017)
- ~~Masquage d'actions dans le calcul des cibles~~ → Masquage dans `act()` uniquement, pas dans les cibles (D-012)
- ~~Décroissance epsilon~~ → Linéaire par step (D-011)

**Quand vous prenez l'une de ces décisions pendant l'implémentation, documentez-la dans `docs/decisions.md`.**

---

## 12. Pattern Registre

Les environnements et les agents utilisent un registre pour une instanciation propre depuis des chaînes de config.

```python
# environments/__init__.py
from .line_world import LineWorldEnv
from .grid_world import GridWorldEnv
from .tictactoe import TicTacToeEnv
from .bobail import BobailEnv

ENV_REGISTRY = {
    "line_world": LineWorldEnv,
    "grid_world": GridWorldEnv,
    "tictactoe": TicTacToeEnv,
    "bobail": BobailEnv,
}

def get_env(name: str, **kwargs):
    return ENV_REGISTRY[name](**kwargs)
```

```python
# agents/__init__.py
AGENT_REGISTRY = {
    "random": RandomAgent,
    "human": HumanAgent,
    "tabular_q": TabularQAgent,
    "dqn": DQNAgent,
    "ddqn": DDQNAgent,
    "ddqn_er": DDQNERAgent,
    "ddqn_per": DDQNPERAgent,
}

def get_agent(name: str, env, params: dict = None):
    params = params or {}
    return AGENT_REGISTRY[name](
        state_size=env.state_space_size(),
        action_size=env.action_space_size(),
        **params,
    )
```

Cela permet à `main.py` et tous les scripts d'instancier depuis des chaînes de config sans importer de classes spécifiques. Le registre sera étendu au fur et à mesure que de nouveaux agents sont implémentés.

---

## 13. Stratégie de Tests

### Tests de conformité des environnements (`tests/test_environments.py`)

Pour chaque environnement, vérifier :
- `reset()` retourne un array de taille `state_space_size()`
- `available_actions()` retourne une liste non-vide après reset
- `step(action)` avec une action légale ne plante pas
- `step(action)` retourne un tuple (ndarray, float, bool)
- Une partie avec des coups aléatoires finit par se terminer (done=True)
- `clone()` produit une copie indépendante (modifier le clone n'affecte pas l'original)

### Smoke tests des agents (`tests/test_agents.py`)

Pour chaque agent :
- `act()` retourne une action dans `available_actions()`
- `save()` puis `load()` produit un agent qui agit de manière identique

### Quand lancer les tests

Lancer `test_environments.py` après avoir implémenté ou modifié un environnement. Lancer `test_agents.py` après avoir implémenté un nouvel agent. Ces tests sont rapides (pas d'entraînement) et détectent immédiatement les violations d'interface.

---

*Ce document évolue au fur et à mesure que les décisions d'implémentation sont prises.*
