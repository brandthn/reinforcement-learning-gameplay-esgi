# Projet DRL — 5IABD T2

Tous les modèles ne sont pas sur le remote (dû au volume accepté sur GitHub) :
```
- Repo GitHub : https://github.com/brandthn/reinforcement-learning-gameplay-esgi/tree/main
- Repo HuggingFace pour Modèles [public]: https://huggingface.co/datasets/Brand066/rl-results/tree/main
```.

Évaluation des algorithmes de Deep Reinforcement Learning sur 4 environnements (`line_world`, `grid_world`, `tictactoe`, `bobail`).


---

## Prérequis

- Python ≥ 3.11
- [`uv`](https://docs.astral.sh/uv/)

```bash
uv sync
```

---

## Environnements

| Nom | Type | Adversaire requis |
|---|---|---|
| `line_world` | Solo | non |
| `grid_world` | Solo | non |
| `tictactoe` | Adversarial | oui |
| `bobail` | Adversarial | oui |

## Agents

| Famille | Clés | Apprenant |
|---|---|---|
| Baseline | `random`, `human` | non |
| Tabulaire | `tabular_q` | oui |
| Value-based | `dqn`, `ddqn`, `ddqn_er`, `ddqn_per` | oui |
| Policy gradient | `reinforce`, `reinforce_mean_baseline`, `reinforce_critic`, `ppo` | oui |
| Planning | `random_rollout`, `mcts` | non (online) |

Les agents de planification ne s'entraînent pas dans le temps : on évalue leur courbe **budget → score** via un script dédié (voir plus bas).

---

## Configs

Une expérience = un fichier YAML dans `configs/{agent}/{env}.yaml`.

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

Champs principaux :

| Champ | Rôle |
|---|---|
| `env`, `agent` | Doivent matcher `ENV_REGISTRY` / `AGENT_REGISTRY` |
| `opponent` | Requis pour `tictactoe` / `bobail` (ex : `random`) |
| `agent_params` | Hyperparamètres passés au constructeur de l'agent |
| `training.num_episodes` | Épisodes totaux d'entraînement |
| `eval.checkpoints` | Épisodes où la policy gelée est évaluée et sauvegardée |
| `eval.num_games` | Parties d'évaluation par checkpoint |
| `seeds` | Liste de graines (un run par graine) |

`configs_done/` contient les configs déjà exécutées (résultats dans `results/`). `configs/` regroupe les agents restants à entraîner.

---

## Commandes

### Entraîner

```bash
# Une config (toutes les graines)
uv run scripts/train.py configs/reinforce/line_world.yaml

# Forcer une graine
uv run scripts/train.py configs/reinforce/line_world.yaml --seed 99

# Mode dev (1 graine, 1000 épisodes, écrit dans results_dev/)
uv run scripts/train.py configs/reinforce/line_world.yaml --quick
uv run scripts/train.py configs/reinforce/line_world.yaml --quick --quick-episodes 500

# Tous les YAML d'un dossier (récursif)
uv run scripts/train_all.py configs/reinforce/

# Tous les YAML sous configs/
uv run scripts/train_all.py
```

### Évaluer les modèles sauvegardés

```bash
# Re-évalue chaque checkpoint dans results/, écrit metrics_reeval.csv + results/summary.csv
uv run scripts/evaluate_all.py
uv run scripts/evaluate_all.py --num-games 200
```

Pour les environnements adversariaux, l'évaluation est équilibrée (moitié des parties en P0, moitié en P1).

### Évaluer les agents de planification (budget sweep)

```bash
uv run scripts/evaluate_planning_agents.py configs/mcts/bobail.yaml
uv run scripts/evaluate_planning_agents.py configs/random_rollout/tictactoe.yaml --budgets 10 50 100 200 --num-games 100
```

Écrit `results/{env}/{agent}/budget_sweep_seed{S}/metrics.csv`.

### Promouvoir les meilleurs modèles (utilisé par la GUI)

```bash
# Tout l'arbre results/
uv run scripts/promote_best.py --all

# Un combo précis
uv run scripts/promote_best.py --env line_world --agent dqn

# Un run + checkpoint manuel
uv run scripts/promote_best.py --run results/line_world/dqn/<run>/ --checkpoint 100000
```

Copie `config.yaml` + `model.pt` vers `results/{env}/{agent}/best/`.

### GUI

```bash
uv run scripts/run_gui.py
uv run scripts/run_gui.py --debug    # affiche l'encoding de l'état
```

Permet de regarder un agent jouer ou de jouer soi-même (`human`). Charge `best/` par défaut, sélecteur intégré pour basculer sur un autre run.

### Benchmark des environnements

```bash
uv run scripts/benchmark.py        # 1000 parties random / env
uv run scripts/benchmark.py 5000
```

### Tests

```bash
uv run pytest
```

---

## Structure des sorties

Un run = un dossier `results/{env}/{agent}/{params}_seed{N}/` contenant :

| Fichier | Contenu |
|---|---|
| `config.yaml` | Snapshot exact de la config |
| `training_curve.csv` | `episode, reward, steps` (par épisode) |
| `metrics.csv` | Métriques d'évaluation par checkpoint (policy gelée) |
| `metrics_reeval.csv` | Réévaluation a posteriori (généré par `evaluate_all.py`) |
| `model_{N}.pt` | Modèle sauvegardé à l'épisode N |

`results/summary.csv` agrège toutes les évaluations (un run × checkpoint × side par ligne).

---

## Structure du repo

```
agents/             # Implémentations (random, tabular, value-based, policy-based, planning)
environments/       # 4 environnements + base.py
training/           # Trainer (solo) + SelfPlayTrainer (adversarial)
evaluation/         # Evaluator (single-side + balanced both-sides)
scripts/            # train, evaluate, promote, gui, benchmark
gui/                # Pygame app
configs/            # Configs des agents en cours
configs_done/       # Configs déjà exécutées (résultats dans results/)
results/            # Sorties d'entraînement et modèles
notebooks/          # Analyses par environnement
docs/               # Sujet, encoding, décisions, observations
tests/              # pytest (envs + agents)
```
