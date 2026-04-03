# DRL Project -- Training & Experimentation Guide

## Setup

```bash
uv sync
```

This installs all dependencies (`torch`, `numpy`, `pygame`, `pyyaml`) from `pyproject.toml` into an isolated virtual environment managed by `uv`.

---

## Quick Start

Train a single experiment:

```bash
uv run python scripts/train.py configs/dqn/grid_world.yaml
```

Train all experiments for one agent:

```bash
uv run python scripts/train_all.py configs/dqn/
```

Train everything:

```bash
uv run python scripts/train_all.py
```

---

## Available Agents and Environments

**Agents:** `random`, `tabular_q`, `dqn`, `ddqn`, `ddqn_er`, `ddqn_per`

**Environments:** `line_world`, `grid_world`, `tictactoe`, `bobail`

`line_world` and `grid_world` are single-player. `tictactoe` and `bobail` are adversarial (require an `opponent` field in the config).

---

## Config Files

### Structure

Each YAML file defines one experiment (one agent, one environment, one set of hyperparameters). Configs are organized by agent:

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
│   ├── grid_world_lr0005.yaml   # hyperparameter variant
│   └── ...
├── ddqn/
├── ddqn_er/
└── ddqn_per/
```

To add a hyperparameter variant (e.g. a different learning rate), copy the baseline config, change the value, and save it in the same agent folder. Each file runs as a separate experiment with its own results directory.

### Example: `configs/dqn/grid_world.yaml`

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

Field by field:

| Field | Description |
|---|---|
| `env` | Environment name (must match a key in `ENV_REGISTRY`) |
| `agent` | Agent name (must match a key in `AGENT_REGISTRY`) |
| `opponent` | For adversarial envs only. Agent used as opponent (e.g. `random`) |
| `agent_params` | All hyperparameters passed to the agent constructor. Varies by agent (see below) |
| `training.num_episodes` | Total training episodes |
| `training.max_steps_per_episode` | Episode truncation limit (default: 1000) |
| `eval.checkpoints` | Episode numbers at which to evaluate the frozen policy and save the model |
| `eval.num_games` | Number of evaluation games per checkpoint |
| `seeds` | List of seeds. The full training runs once per seed |
| `seed` | Alternative to `seeds` for a single seed |

### Agent parameters reference

| Parameter | Agents | Description |
|---|---|---|
| `lr` | `tabular_q`, `dqn`, `ddqn`, `ddqn_er`, `ddqn_per` | Learning rate |
| `gamma` | all learning agents | Discount factor |
| `epsilon_start` | all learning agents | Initial exploration rate |
| `epsilon_end` | all learning agents | Final exploration rate |
| `epsilon_decay_steps` | all learning agents | Steps for linear epsilon decay |
| `hidden_layers` | `dqn`, `ddqn`, `ddqn_er`, `ddqn_per` | Network hidden layer sizes (e.g. `[128, 128]`) |
| `batch_size` | `dqn`, `ddqn`, `ddqn_er`, `ddqn_per` | Minibatch size for replay sampling |
| `buffer_capacity` | `dqn`, `ddqn`, `ddqn_er`, `ddqn_per` | Max transitions in replay buffer |
| `target_update_freq` | `dqn`, `ddqn`, `ddqn_er`, `ddqn_per` | Copy online network to target network every N episodes |
| `learning_starts` | `ddqn_er`, `ddqn_per` | Min buffer samples before training begins |
| `per_alpha` | `ddqn_per` | PER prioritization exponent |
| `per_beta_start` | `ddqn_per` | Initial importance-sampling correction |
| `per_beta_end` | `ddqn_per` | Final IS correction |
| `per_beta_steps` | `ddqn_per` | Steps for linear beta annealing |

---

## Running Experiments

### Train a single config

```bash
uv run python scripts/train.py configs/ddqn_per/tictactoe.yaml
```

This trains `ddqn_per` on `tictactoe` for 100k episodes, once per seed in the config (e.g. seeds 42, 123, 456 = 3 runs). Results go to:

```
results/tictactoe/ddqn_per/..._seed42/
results/tictactoe/ddqn_per/..._seed123/
results/tictactoe/ddqn_per/..._seed456/
```

### Override the seed

```bash
uv run python scripts/train.py configs/dqn/grid_world.yaml --seed 99
```

Ignores the seeds in the config. Runs only once with seed 99.

### Train all configs for one agent

```bash
uv run python scripts/train_all.py configs/dqn/
```

Runs `scripts/train.py` sequentially on every `.yaml` in `configs/dqn/`.

### Train all configs in the project

```bash
uv run python scripts/train_all.py
```

Recursively finds all `.yaml` files under `configs/` and trains each one.

### Re-evaluate saved models

```bash
uv run python scripts/evaluate_all.py
uv run python scripts/evaluate_all.py --results-dir results --num-games 200
```

Walks every experiment in `results/`, loads each saved checkpoint, re-evaluates with frozen policy, and writes `metrics_reeval.csv`.

### Promote best models for the GUI

After training, promote the best checkpoint per agent/env to `results/{env}/{agent}/best/`:

```bash
uv run scripts/promote_best.py --all
```

Or target a specific combo:

```bash
uv run scripts/promote_best.py --env line_world --agent dqn
```

Or promote a specific run/checkpoint manually:

```bash
uv run scripts/promote_best.py --run results/line_world/dqn/..._seed42/ --checkpoint 100000
```

The GUI loads from `best/` by default. A model picker in the menu also lets you select any trained run.

### Run the GUI

```bash
uv run scripts/run_gui.py
```

Select environment, agent, and model (if applicable), then click Start.

---

## Output Structure

Each training run writes to `results/{env}/{agent}/{params}_seed{N}/`:

| File | Content |
|---|---|
| `config.yaml` | Exact config used (snapshot for reproducibility) |
| `training_curve.csv` | Per-episode: `episode, reward, steps` |
| `metrics.csv` | Per-checkpoint: `mean_reward, std_reward, mean_steps, std_steps, mean_action_time_ms, std_action_time_ms` |
| `model_{N}.pt` | Saved model at episode N |

The `metrics.csv` contains **evaluation** metrics (frozen policy, no exploration), not training metrics. This is what the syllabus requires.

---

## Hyperparameter Sweeps

For comparing multiple values of a hyperparameter, use a **sweep config** with `scripts/train_sweep.py`. A sweep config is a normal config with an extra `sweep:` section that declares axes of variation.

### Example: `configs/dqn/grid_world_sweep.yaml`

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

The `sweep:` section uses dot-notation keys. Each key maps to a list of values to try. The script computes the cartesian product: here 3 lr values x 2 batch sizes = 6 experiments, each run with 3 seeds = 18 total training runs.

### Running a sweep

Preview what will run (no training):

```bash
uv run python scripts/train_sweep.py configs/dqn/grid_world_sweep.yaml --dry-run
```

Run the full sweep:

```bash
uv run python scripts/train_sweep.py configs/dqn/grid_world_sweep.yaml
```

Override the seed:

```bash
uv run python scripts/train_sweep.py configs/dqn/grid_world_sweep.yaml --seed 99
```

Under the hood, each combination is a concrete config passed to the same `train_single()` function used by `scripts/train.py`. Results land in separate directories (the folder name encodes all parameters). The `config.yaml` snapshot in each results directory is the concrete, expanded version -- not the sweep file.

### Manual approach

You can still create individual config files for single experiments. Copy the baseline, change one value, train with `train.py` or `train_all.py`. One config = one experiment remains the invariant at execution time.

---

## Tests

```bash
uv run pytest
```
