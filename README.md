# Projet DRL — Guide d'Entraînement & Expérimentation

## Getting Started

```bash
git clone <repo>
cd <repo>
uv sync                                                        # installer les dépendances

uv run pytest                                                  # vérifier que tout fonctionne

uv run python scripts/run_gui.py                              # lancer la GUI (PyGame)

uv run python scripts/train.py configs/dqn/grid_world.yaml   # entraîner un agent
uv run python scripts/train.py configs/dqn/grid_world.yaml --quick  # mode rapide (dev)
uv run python scripts/train_all.py                            # tout entraîner
uv run python scripts/evaluate_all.py                         # réévaluer tous les modèles
uv run python scripts/promote_best.py --all                   # promouvoir les meilleurs modèles vers best/
```

> **Prérequis :** Python ≥ 3.11, [`uv`](https://docs.astral.sh/uv/) installé.

---

## Installation

```bash
uv sync
```

Cela installe toutes les dépendances (`torch`, `numpy`, `pygame`, `pyyaml`) depuis `pyproject.toml` dans un environnement virtuel isolé géré par `uv`.

---

## Démarrage Rapide

Entraîner une seule expérience :

```bash
uv run python scripts/train.py configs/dqn/grid_world.yaml
```

Entraîner toutes les expériences d'un agent :

```bash
uv run python scripts/train_all.py configs/dqn/
```

Tout entraîner :

```bash
uv run python scripts/train_all.py
```

---

## Agents et Environnements Disponibles

**Agents implémentés :** `random`, `tabular_q`, `dqn`, `ddqn`, `ddqn_er`, `ddqn_per`

**Agents à implémenter :** REINFORCE (3 variantes), PPO, RandomRollout, MCTS, Expert Apprentice, AlphaZero, MuZero (+stochastique)

**Environnements :** `line_world`, `grid_world`, `tictactoe`, `bobail`

`line_world` et `grid_world` sont mono-joueur. `tictactoe` et `bobail` sont adversariaux (nécessitent un champ `opponent` dans la config).

---

## Fichiers de Configuration

### Structure

Chaque fichier YAML définit une expérience (un agent, un environnement, un jeu d'hyperparamètres). Les configs sont organisées par agent :

```
configs/
├── random/
│   ├── line_world.yaml
│   ├── grid_world.yaml
│   ├── tictactoe.yaml
│   └── bobail.yaml
├── tabular_q/
│   └── ...
├── dqn/
│   ├── grid_world.yaml          # baseline
│   ├── grid_world_lr0005.yaml   # variante d'hyperparamètre
│   └── ...
├── ddqn/
├── ddqn_er/
└── ddqn_per/
```

Pour ajouter une variante d'hyperparamètre (ex: un learning rate différent), copier la config de base, modifier la valeur, et sauvegarder dans le même dossier agent. Chaque fichier s'exécute comme une expérience séparée avec son propre dossier de résultats.

### Exemple : `configs/dqn/grid_world.yaml`

```yaml
env: grid_world
agent: dqn

agent_params:
  lr: 0.001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay_steps: 20000
  hidden_layers: [64, 64]
  batch_size: 64
  buffer_capacity: 10000
  target_update_freq: 200

training:
  num_episodes: 100000
  max_steps_per_episode: 200

eval:
  checkpoints: [1000, 10000, 100000]
  num_games: 100

seeds: [42, 123, 456]
```

Champ par champ :

| Champ | Description |
|---|---|
| `env` | Nom de l'environnement (doit correspondre à une clé dans `ENV_REGISTRY`) |
| `agent` | Nom de l'agent (doit correspondre à une clé dans `AGENT_REGISTRY`) |
| `opponent` | Pour les envs adversariaux uniquement. Agent utilisé comme adversaire (ex: `random`) |
| `agent_params` | Tous les hyperparamètres passés au constructeur de l'agent. Varient selon l'agent (voir ci-dessous) |
| `training.num_episodes` | Nombre total d'épisodes d'entraînement |
| `training.max_steps_per_episode` | Limite de troncature d'épisode (défaut : 1000) |
| `eval.checkpoints` | Numéros d'épisodes auxquels évaluer la policy gelée et sauvegarder le modèle |
| `eval.num_games` | Nombre de parties d'évaluation par checkpoint |
| `seeds` | Liste de graines. L'entraînement complet est lancé une fois par graine |
| `seed` | Alternative à `seeds` pour une graine unique |

### Référence des paramètres par agent

| Paramètre | Agents | Description |
|---|---|---|
| `lr` | `tabular_q`, `dqn`, `ddqn`, `ddqn_er`, `ddqn_per` | Taux d'apprentissage |
| `gamma` | tous les agents apprenants | Facteur d'escompte |
| `epsilon_start` | tous les agents apprenants | Taux d'exploration initial |
| `epsilon_end` | tous les agents apprenants | Taux d'exploration final |
| `epsilon_decay_steps` | tous les agents apprenants | Steps pour la décroissance linéaire d'epsilon |
| `hidden_layers` | `dqn`, `ddqn`, `ddqn_er`, `ddqn_per` | Tailles des couches cachées (ex: `[128, 128]`) |
| `batch_size` | `dqn`, `ddqn`, `ddqn_er`, `ddqn_per` | Taille du minibatch pour l'échantillonnage du replay |
| `buffer_capacity` | `dqn`, `ddqn`, `ddqn_er`, `ddqn_per` | Max de transitions dans le replay buffer |
| `target_update_freq` | `dqn`, `ddqn`, `ddqn_er`, `ddqn_per` | Copier le réseau online vers le réseau target toutes les N épisodes |
| `learning_starts` | `ddqn_er`, `ddqn_per` | Min d'échantillons dans le buffer avant de commencer l'entraînement |
| `per_alpha` | `ddqn_per` | Exposant de priorisation PER |
| `per_beta_start` | `ddqn_per` | Correction IS initiale |
| `per_beta_end` | `ddqn_per` | Correction IS finale |
| `per_beta_steps` | `ddqn_per` | Steps pour le scheduling linéaire de beta |

---

## Lancer les Expériences

### Entraîner une seule config

```bash
uv run python scripts/train.py configs/ddqn_per/tictactoe.yaml
```

Cela entraîne `ddqn_per` sur `tictactoe` pour 100k épisodes, une fois par graine dans la config (ex: graines 42, 123, 456 = 3 runs). Résultats dans :

```
results/tictactoe/ddqn_per/..._seed42/
results/tictactoe/ddqn_per/..._seed123/
results/tictactoe/ddqn_per/..._seed456/
```

### Remplacer la graine

```bash
uv run python scripts/train.py configs/dqn/grid_world.yaml --seed 99
```

Ignore les graines de la config. Lance une seule fois avec la graine 99.

### Mode rapide (développement)

```bash
uv run python scripts/train.py configs/dqn/grid_world.yaml --quick
uv run python scripts/train.py configs/dqn/grid_world.yaml --quick --quick-episodes 500
```

Mode itération rapide : 1 graine, moins d'épisodes (défaut 1000), résultats dans `results_dev/`.

### Entraîner toutes les configs d'un agent

```bash
uv run python scripts/train_all.py configs/dqn/
```

Lance `scripts/train.py` séquentiellement sur chaque `.yaml` dans `configs/dqn/`.

### Entraîner toutes les configs du projet

```bash
uv run python scripts/train_all.py
```

Trouve récursivement tous les fichiers `.yaml` sous `configs/` et entraîne chacun. Détecte automatiquement les configs sweep et les route vers `train_sweep.py`.

### Réévaluer les modèles sauvegardés

```bash
uv run python scripts/evaluate_all.py
uv run python scripts/evaluate_all.py --results-dir results --num-games 200
```

Parcourt chaque expérience dans `results/`, charge chaque checkpoint sauvegardé, réévalue avec la policy gelée, et écrit `metrics_reeval.csv`.

### Promouvoir les meilleurs modèles pour la GUI

Après l'entraînement, promouvoir le meilleur checkpoint par combo agent/env vers `results/{env}/{agent}/best/` :

```bash
uv run scripts/promote_best.py --all
```

Ou cibler une combinaison spécifique :

```bash
uv run scripts/promote_best.py --env line_world --agent dqn
```

Ou promouvoir un run/checkpoint spécifique manuellement :

```bash
uv run scripts/promote_best.py --run results/line_world/dqn/..._seed42/ --checkpoint 100000
```

La GUI charge depuis `best/` par défaut. Un sélecteur de modèle dans le menu permet aussi de choisir n'importe quel run entraîné.

### Lancer la GUI

```bash
uv run scripts/run_gui.py
```

Sélectionner l'environnement, l'agent et le modèle (si applicable), puis cliquer Start.

---

## Structure des Sorties

Chaque run d'entraînement écrit dans `results/{env}/{agent}/{params}_seed{N}/` :

| Fichier | Contenu |
|---|---|
| `config.yaml` | Config exacte utilisée (snapshot pour la reproductibilité) |
| `training_curve.csv` | Par épisode : `episode, reward, steps` |
| `metrics.csv` | Par checkpoint : `mean_reward, std_reward, mean_steps, std_steps, mean_action_time_ms, std_action_time_ms` |
| `model_{N}.pt` | Modèle sauvegardé à l'épisode N |

Le `metrics.csv` contient les métriques d'**évaluation** (policy gelée, pas d'exploration), pas les métriques d'entraînement. C'est ce que le syllabus exige.

---

## Sweeps d'Hyperparamètres

Pour comparer plusieurs valeurs d'un hyperparamètre, utiliser une **config sweep** avec `scripts/train_sweep.py`. Une config sweep est une config normale avec une section `sweep:` supplémentaire qui déclare des axes de variation.

### Exemple : `configs/dqn/grid_world_sweep.yaml`

```yaml
env: grid_world
agent: dqn

agent_params:
  lr: 0.001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay_steps: 20000
  hidden_layers: [64, 64]
  batch_size: 64
  buffer_capacity: 10000
  target_update_freq: 200

sweep:
  agent_params.lr: [0.001, 0.0005, 0.0001]
  agent_params.batch_size: [32, 64]

training:
  num_episodes: 100000
  max_steps_per_episode: 200

eval:
  checkpoints: [1000, 10000, 100000]
  num_games: 100

seeds: [42, 123, 456]
```

La section `sweep:` utilise des clés en notation pointée. Chaque clé correspond à une liste de valeurs à tester. Le script calcule le produit cartésien : ici 3 valeurs de lr × 2 tailles de batch = 6 expériences, chacune lancée avec 3 graines = 18 runs au total.

### Lancer un sweep

Prévisualiser ce qui sera lancé (pas d'entraînement) :

```bash
uv run python scripts/train_sweep.py configs/dqn/grid_world_sweep.yaml --dry-run
```

Lancer le sweep complet :

```bash
uv run python scripts/train_sweep.py configs/dqn/grid_world_sweep.yaml
```

Remplacer la graine :

```bash
uv run python scripts/train_sweep.py configs/dqn/grid_world_sweep.yaml --seed 99
```

En interne, chaque combinaison est une config concrète passée à la même fonction `train_single()` utilisée par `scripts/train.py`. Les résultats atterrissent dans des dossiers séparés (le nom du dossier encode tous les paramètres). Le snapshot `config.yaml` dans chaque dossier de résultats est la version concrète expansée — pas le fichier sweep.

### Approche manuelle

Vous pouvez toujours créer des fichiers de config individuels pour des expériences isolées. Copier la baseline, changer une valeur, entraîner avec `train.py` ou `train_all.py`. L'invariant une-config-une-expérience est maintenu au moment de l'exécution.

---

## Tests

```bash
uv run pytest
```

Les tests vérifient la conformité des environnements (interface, terminaison) et le bon fonctionnement des agents (actions légales, save/load).
