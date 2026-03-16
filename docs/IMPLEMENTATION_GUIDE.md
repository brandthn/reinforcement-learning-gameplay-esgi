# docs/PROJECT_INSTRUCTIONS.md

# Implementation Guide — Deep Reinforcement Learning Project

> **Purpose of this document:** This is the shared reference for all members and contributors implementing this project. It documents the architectural decisions we have explicitly agreed on, the conventions to follow, and the boundaries of what has been decided vs. what remains open. **If something is not stated here, it is not yet decided — do not assume or invent constraints.**

> **What this document is NOT:** This is not the project syllabus (see professor's instructions separately in file `docs/PROJECT_INSTRUCTIONS.md`). This is not a deadline tracker. This is the engineering and design contract.

---

## 1. Project Scope

**Goal:** Implement and evaluate 16 reinforcement learning algorithms across 4 environments, with trained models, metrics, a GUI, and a report.

**Environments:** LineWorld, GridWorld, TicTacToe, Bobail

**Framework:** PyTorch (all neural network code)

**Language convention:** Code (variable names, comments, docstrings) in English. Documentation files in `docs/` in French. Comment factually, not generic verbose as LLM do. Factual, simple and clear.

---

## 2. Repository Structure

```
aidrl-project/
│
├── environments/
│   ├── __init__.py              # Registry + get_env() factory
│   ├── base.py                  # Abstract Environment class
│   ├── line_world.py
│   ├── grid_world.py
│   ├── tictactoe.py
│   └── bobail.py
│
├── agents/
│   ├── __init__.py              # Registry + get_agent() factory
│   ├── base.py                  # Abstract Agent class
│   ├── random_agent.py
│   ├── human_agent.py
│   ├── tabular_q.py
│   ├── value_based/
│   │   ├── __init__.py
│   │   ├── dqn.py
│   │   ├── ddqn.py
│   │   ├── ddqn_er.py
│   │   └── ddqn_per.py
│   ├── policy_gradient/
│   │   ├── __init__.py
│   │   ├── reinforce.py         # All 3 REINFORCE variants
│   │   └── ppo.py
│   └── planning/
│       ├── __init__.py
│       ├── random_rollout.py
│       ├── mcts.py
│       ├── expert_apprentice.py
│       ├── alpha_zero.py
│       └── muzero.py            # MuZero + stochastic variant
│
├── training/
│   ├── __init__.py
│   ├── trainer.py               # Generic training loop
│   ├── self_play.py             # Two-player training loop
│   ├── replay_buffer.py         # ReplayBuffer + PrioritizedReplayBuffer
│   └── networks.py              # Shared MLP builder
│
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py             # Frozen-policy evaluation at checkpoints
│
├── gui/
│   ├── __init__.py
│   └── app.py                   # Pygame GUI
│
├── docs/
│   ├── encoding.md              # State & action encoding specs (deliverable)
│   ├── decisions.md             # Architecture decision log
│   └── algorithms.md            # Plain-language algorithm explanations
│
├── configs/                     # YAML config files, organized by agent
│   ├── random/
│   │   ├── line_world.yaml
│   │   └── ...
│   ├── dqn/
│   │   ├── grid_world.yaml          # baseline experiment
│   │   ├── grid_world_sweep_lr.yaml # sweep experiment
│   │   └── ...
│   ├── ddqn/
│   ├── ddqn_er/
│   ├── ddqn_per/
│   └── tabular_q/
│
├── models/                      # Saved .pt checkpoints (gitignored)
│   └── .gitkeep
│
├── results/                     # Metrics CSVs, training curves (gitignored)
│   └── .gitkeep
│
├── notebooks/
│   └── results_analysis.ipynb   # Comparative plots, report figures
│
├── scripts/
│   ├── train.py                 # Train one agent on one env from a config
│   ├── train_sweep.py           # Expand sweep config → call train_single() per combo
│   ├── train_all.py             # Batch: run train.py or train_sweep.py for all configs
│   ├── evaluate_all.py          # Batch evaluation of saved models
│   ├── benchmark.py             # Benchmark random play (games/sec)
│   └── run_gui.py               # Launch GUI
│
├── tests/
│   ├── test_environments.py
│   ├── test_agents.py
│   ├── test_training.py
│   └── test_value_based.py
│
├── pyproject.toml               # Dependencies + project metadata (managed by uv)
├── uv.lock                      # Locked dependency versions
├── README.md
└── main.py
```

### Structure principles

**The subfolders inside `agents/` are purely organizational.** They group algorithm families so contributors working on one family can locate their files easily. They do NOT imply shared utilities between files in the same folder. Each algorithm file is self-contained.

**Top-level agents** (`random_agent.py`, `human_agent.py`, `tabular_q.py`) sit outside the subfolders because they don't belong to any family.

`**training/networks.py`** contains the shared MLP builder function used by multiple algorithm families. It lives in `training/` because it's a training utility, not an algorithm.

**The `notebooks/` folder** is exclusively for results visualization and analysis. Notebooks read from `results/`, they never produce training artifacts. Their output (plots, tables) feeds directly into the report and slides.

### Open decisions about structure

- Whether MuZero / AlphaZero will need further internal splitting (e.g. into a subfolder with separate network definitions) — decide during implementation based on file size and complexity.
- Whether additional utility modules are needed in `training/` — to be assessed as implementation progresses.

---

## 3. Core Interfaces

These two abstract classes are the foundation. **Everything else in the project — training, evaluation, GUI, agents — depends on these interfaces.** They must be implemented first and must remain stable.

### 3.1 Environment Interface

```python
# environments/base.py
from abc import ABC, abstractmethod
import numpy as np
import copy

class Environment(ABC):

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset to initial state. Returns state vector."""
        ...

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Execute action.
        Returns (next_state, reward, done).
        
        For two-player games:
        - reward is from the perspective of the player who just acted
        - next_state is from the perspective of the NEW current player
        """
        ...

    @abstractmethod
    def available_actions(self) -> list[int]:
        """Legal action indices for the current player/phase."""
        ...

    @abstractmethod
    def state_description(self) -> np.ndarray:
        """Current state as flat float32 vector."""
        ...

    @abstractmethod
    def action_space_size(self) -> int:
        """Total number of possible actions (including currently illegal ones)."""
        ...

    @abstractmethod
    def state_space_size(self) -> int:
        """Dimensionality of state_description() output."""
        ...

    def is_adversarial(self) -> bool:
        """Override to return True for two-player games."""
        return False

    def current_player(self) -> int:
        """0 for single-player. 0 or 1 for two-player."""
        return 0

    def clone(self):
        """Deep copy. Required for MCTS/AlphaZero/MuZero."""
        return copy.deepcopy(self)

    def render_text(self) -> str:
        """Optional text representation for debugging."""
        return ""
```

### 3.2 Agent Interface

```python
# agents/base.py
from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):

    @abstractmethod
    def act(self, state: np.ndarray, available_actions: list[int],
            training: bool = False) -> int:
        """
        Select an action.
        - training=True: may explore (epsilon-greedy, stochastic, etc.)
        - training=False: pure exploitation (greedy, deterministic)
        """
        ...

    def observe(self, state, action, reward, next_state, done):
        """Called after env.step() during training. Default: no-op."""
        pass

    def end_episode(self):
        """Called at end of each training episode. Default: no-op."""
        pass

    def save(self, path: str) -> None:
        """Save model/weights/tables."""
        pass

    def load(self, path: str) -> None:
        """Load model/weights/tables."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
```

### Why this matters

The `Agent` class is a **contract**: the GUI, Evaluator, Trainer, and self-play loop all call `agent.act()` without knowing which algorithm is behind it. This is what allows every algorithm to work with every environment and every evaluation/GUI pipeline without special-casing.

The `Environment` class is the same kind of contract: every training loop calls `env.step()`, `env.available_actions()`, `env.state_description()` without knowing if it's LineWorld or Bobail.

**Rule: If you are implementing a new algorithm, you must conform to the Agent interface. If you are implementing a new environment, you must conform to the Environment interface. No exceptions.**

### Open decisions about interfaces

- Whether `observe()` is sufficient for all learning agents, or whether some algorithms (e.g. REINFORCE collecting full trajectories) will need a different data-passing mechanism — to be assessed during implementation. The `end_episode()` hook exists for this purpose but its exact usage pattern per algorithm is not prescribed.
- Whether additional convenience methods are needed on `Environment` (e.g. `game_result()` for two-player final outcome) — decide when implementing the first two-player training loop.

---

## 4. Two-Player Architecture (Approach 1 — Current Player Perspective)

### Decision

For two-player games (TicTacToe, Bobail), the environment always presents the state from the **current player's perspective.**

### What this means concretely

- `state_description()` returns a vector where channel/section "my pieces" always refers to the pieces of whoever's turn it is, and "opponent pieces" refers to the other player.
- When `step(action)` is called, the environment executes the move, switches the active player, and the next `state_description()` call shows the board from the new player's viewpoint.
- `reward` returned by `step()` is from the perspective of the player who just acted.
- Agents never know whether they are "player 1" or "player 2." They always see "my pieces" vs "opponent pieces."

### Why this approach

- Agents are player-agnostic — the same trained DQN works as either player.
- Training code for single-player envs and two-player envs shares the same agent interface.
- MCTS simulation works naturally — each `step()` in the tree gives the next player's view.
- Allows flexible opponent pairing: agent vs random, agent vs heuristic, agent vs trained agent, agent vs human — all with the same environment class.

### Caution for implementers

**Replay buffer transitions in two-player games:** When the learning agent (player 0) acts in state `s`, the `next_state` returned by `env.step()` is from player 1's perspective. For off-policy methods (DQN family), the training loop must track and pair "same-player" states correctly. The exact mechanism for this is to be determined during the implementation of `self_play.py`, but the problem must be kept in mind.

**Reward convention:** When a game ends, the terminal reward should be positive for the winner (from the winner's perspective). The training loop is responsible for assigning the correct reward to each player's transitions. This is a detail to document carefully in `docs/decisions.md` when implemented.

---

## 5. Bobail — Game-Specific Design

### Two-Phase Turns

Each Bobail turn consists of two sub-actions:

1. **Bobail phase:** Move the bobail token one square (any of 8 directions)
2. **Piece phase:** Move one of your 5 pieces as far as it can go in one direction

**Exception:** On the very first turn of the game, player 1 only performs the piece phase (no bobail move).

### Implementation approach

The environment tracks an internal `phase` variable. Each call to `step(action)` handles one sub-action. The action space and `available_actions()` return depends on the current phase. The turn switches to the opponent only after both phases are complete.

**Reward:** The intermediate sub-action (bobail phase) returns `reward = 0` and `done = False`. Only after the piece phase does the environment check win conditions and potentially return a terminal reward.

### State encoding

This is a required deliverable (see `docs/encoding.md`). The general approach:

- 5×5 board → 25 cells
- 3 channels: current player's pieces, opponent's pieces, bobail position
- Total state vector: 75 float32 values

The exact encoding details (including whether additional information like phase or turn number is included) are to be finalized during implementation and documented in `docs/encoding.md`.

### Action encoding

General approach: encode actions as `(from_cell, to_cell)` mapped to a single integer. This works uniformly for both the bobail phase and the piece phase. The action space size and exact mapping are to be finalized during implementation.

### Win conditions

1. The bobail is moved to your home row (row 0 for player 0, row 4 for player 1) → you win
2. The current player cannot move the bobail (it's completely surrounded) → current player loses

---

## 6. Experimental Conventions

### Config files (YAML)

Every experiment is defined by a YAML config file. **No hyperparameters are hardcoded in algorithm files.** Algorithm constructors receive their parameters from the config.

The exact schema of config files is not prescribed — it will vary by algorithm family. But every config must include at minimum:

- `env`: environment name
- `agent`: algorithm name
- `seed` or `seeds`: random seed(s)
- `training.num_episodes`: how many episodes to train

### Config folder organization

Configs are organized by agent under `configs/`:

```
configs/
├── random/
│   ├── line_world.yaml
│   └── ...
├── dqn/
│   ├── grid_world.yaml              # baseline
│   ├── grid_world_sweep_lr.yaml     # sweep variant
│   └── ...
├── ddqn/
├── ddqn_er/
├── ddqn_per/
└── tabular_q/
```

**One config = one experiment** at execution time. Baseline configs use `scripts/train.py`. Sweep configs (see below) use `scripts/train_sweep.py`. `scripts/train_all.py` detects which type each config is and routes accordingly.

### Sweep configs

A sweep config is a regular config with an additional `sweep:` section that declares axes of variation using dot-notation paths:

```yaml
sweep:
  agent_params.lr: [0.001, 0.0005, 0.0001]
  agent_params.batch_size: [32, 64]
```

`train_sweep.py` computes the cartesian product (here 3 × 2 = 6 experiments), builds a concrete config for each combination, and calls `train_single()` from `train.py` directly (in-process, Option A). The sweep script is a pure config-expansion layer — `Trainer`, `SelfPlayTrainer`, agents, and environments are unaware of sweeps.

The `sweep:` key is explicit to avoid ambiguity with list-valued parameters like `hidden_layers: [64, 64]`. Only keys declared under `sweep:` are expanded.

Each expanded run writes its own concrete `config.yaml` snapshot to `results/`, preserving the one-config-one-experiment invariant for traceability.

### Experiment naming and results storage

```
results/
└── {env_name}/
    └── {agent_name}/
        └── {hyperparam_summary}_seed{N}/
            ├── config.yaml          # Full config snapshot (copied at run start)
            ├── metrics.csv          # Evaluation metrics at checkpoints
            ├── training_curve.csv   # Per-episode training data
            └── model_{checkpoint}.pt
```

The `{hyperparam_summary}` is generated from key parameters (e.g. `lr0.001_gamma0.99_eps50k`). The full config YAML inside the directory is the authoritative record — the directory name is for quick human identification only.

Each config may optionally include an `experiment_label` field for readable plot legends.

### Seed strategy

- **During development:** Use a single fixed seed for reproducibility and debugging.
- **For final reported results:** Run each experiment with **multiple seeds** (3-5) and report mean ± standard deviation. This demonstrates statistical rigor.
- The training script must accept a list of seeds and loop over them.
- Results CSV must record which seed was used.

### Evaluation checkpoints

The syllabus requires metrics at: 1,000 / 10,000 / 100,000 / 1,000,000 (if possible) training episodes.

At each checkpoint, the **Evaluator** runs the agent in pure inference mode (`training=False`) for N games and records:

- Mean reward (± std)
- Mean episode length (± std)
- Mean action time (ms)

**Critical distinction:** These are metrics for the **obtained policy**, not training-time metrics. The training curve (per-episode reward during training) is stored separately.

---

## 7. Shared Utilities

### MLP Builder (`training/networks.py`)

A single function that builds a feedforward network given input dim, output dim, and hidden layer sizes. Used by DQN, DDQN, REINFORCE, PPO, and potentially AlphaZero/MuZero sub-networks.

### Replay Buffer (`training/replay_buffer.py`)

Two classes:

- `ReplayBuffer` — uniform sampling (used by DQN, DDQN, DDQN+ER)
- `PrioritizedReplayBuffer` — priority-weighted sampling (used by DDQN+PER)

These are shared across the value-based family. Other algorithm families may or may not use them (AlphaZero/MuZero may need their own trajectory-based storage — to be decided during implementation).

### Trainer (`training/trainer.py`)

Generic single-player training loop. Calls `agent.act()`, `env.step()`, `agent.observe()`, `agent.end_episode()`. Triggers evaluation at configured checkpoints.

### Self-Play (`training/self_play.py`)

Two-player training loop. Manages turn alternation between the learning agent and an opponent. The opponent can be any `Agent` instance (RandomAgent, a heuristic, another trained agent).

### Evaluator (`evaluation/evaluator.py`)

Runs a trained agent in inference mode. Collects reward, episode length, and action timing statistics. Writes results to CSV.

---

## 8. GUI

**Technology:** Pygame

**Minimum requirements:**

- Select environment
- Select agent (or human)
- Watch the agent play / play as human
- Works with any Agent subclass via `agent.act()`

**No abstract renderer hierarchy.** Each environment's visual representation is handled directly in the GUI code (if/else or a simple dispatch dict for 4 environments).

The exact GUI design (layout, interaction flow, visual style) is not prescribed and will be determined during implementation.

---

## 9. Documentation Strategy

### `docs/encoding.md` — REQUIRED deliverable

For each environment:

- What the state vector represents, element by element
- What each action index means
- Why this encoding was chosen, what alternatives were considered
- Written in French

### `docs/decisions.md` — Architecture Decision Log

Short entries documenting non-obvious implementation choices. Format:

```markdown
## Titre de la décision

**Contexte :** Quel problème on essaie de résoudre.

**Décision :** Ce qu'on a choisi de faire.

**Pourquoi :** Justification. Pourquoi pas les alternatives.

**Références :** Sources si applicable.
```

Write an entry **every time** you make a choice that someone might question later, especially during the oral defense. Examples: action masking strategy, reward shaping, how epsilon decay works, why a certain network architecture, etc.

### `docs/algorithms.md` — Algorithm explanations

For each algorithm, a plain-language explanation **in your own words** that you can confidently present orally. Structure per algorithm:

- Core idea (2-3 sentences)
- How it trains (the key mechanism)
- What it adds over the previous/simpler version
- Known limitations or things to watch out for

Written in French at a level suitable for the oral defense.

### Progressive documentation

**Do not write all documentation at the end.** Write `docs/decisions.md` entries as you implement. Write `docs/algorithms.md` entries as you implement each algorithm. This serves both as study material and defense preparation.

---

## 10. Implementation Safety Rules

These rules exist to prevent one person's work from breaking another's.

### Rule 1: Never modify the base interfaces without team consensus

`environments/base.py` and `agents/base.py` are the contracts. If you think a method needs to be added or changed, discuss first. Changing these affects everything.

### Rule 2: Every algorithm must pass the smoke test

Before considering an algorithm "done," verify:

```python
agent = YourAgent(state_size=env.state_space_size(), action_size=env.action_space_size(), ...)
state = env.reset()
action = agent.act(state, env.available_actions())
assert action in env.available_actions()
```

This must work for every environment the algorithm is supposed to support.

### Rule 3: Algorithms must handle action masking

If the environment returns `available_actions()` as a subset of all actions, the agent must only return actions from that subset. **How** each algorithm achieves this (e.g. masking Q-values to -inf, filtering the softmax output, etc.) is an implementation detail to be decided per algorithm and documented in `docs/decisions.md`.

### Rule 4: Environments must implement clone() for planning algorithms

MCTS, AlphaZero, and MuZero need `env.clone()` to simulate future states. The default `copy.deepcopy` should work but may be slow. If profiling reveals `clone()` is a bottleneck, the environment can override it with a faster implementation.

### Rule 5: No hardcoded hyperparameters

All tunable parameters must come from the config. Algorithm files should not contain default values for learning rate, gamma, epsilon, etc. that silently apply when no config is provided. If a parameter is missing from the config, raise an error rather than using a hidden default.

### Rule 6: Results are write-only during training

Training scripts write to `results/`. Notebooks and analysis scripts read from `results/`. Nothing should both read and write to the same results directory during a training run.

---

## 11. What Is NOT Decided Yet

The following aspects are intentionally left open and should be decided during implementation, then documented in `docs/decisions.md`:

- Exact hyperparameter ranges for experiments
- How REINFORCE variants are organized internally (one class with flags, or three classes)
- Whether AlphaZero / MuZero need additional internal file splitting
- How the heuristic opponent for Bobail works (rule-based? what rules?)
- Whether additional state features are needed in the encoding (turn number, phase indicator, etc.)
- Whether MuZero's replay needs a separate implementation from the shared ReplayBuffer
- The notebook structure (one big notebook vs multiple focused ones)
- Exact contents of the final report structure

**Decided since initial writing (see `docs/decisions.md` for details):**

- ~~Exact neural network architectures~~ → Configurable MLP via `hidden_layers` param (D-013)
- ~~Exact hyperparameter ranges~~ → Defined per config file; sweep configs enable systematic exploration
- ~~Exact GUI layout~~ → Implemented in `gui/app.py` (D-009, D-010)
- ~~How the replay buffer handles two-player transitions~~ → Deferred observe in `self_play.py` (D-002)
- ~~How `train_all.py` parallelizes~~ → Sequential with subprocess; detects sweep vs regular configs automatically
- ~~Config file organization~~ → Agent-based subdirectories under `configs/` (see §6)
- ~~Model loading in the GUI~~ → `best/` convention + model picker (D-018)
- ~~Value-based inheritance hierarchy~~ → DQN → DDQN → DDQN+ER / DDQN+PER (D-014, D-015, D-016, D-017)
- ~~Action masking in target computation~~ → Masking in `act()` only, not in targets (D-012)
- ~~Epsilon decay strategy~~ → Linear per step (D-011)

**When you make one of these decisions during implementation, document it in `docs/decisions.md`.**

---

## 12. Registry Pattern

Both environments and agents use a registry for clean instantiation from config strings.

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
    "tabular_q": TabularQAgent,
    "dqn": DQNAgent,
    # ... etc
}

def get_agent(name: str, env, params: dict = None):
    params = params or {}
    return AGENT_REGISTRY[name](
        state_size=env.state_space_size(),
        action_size=env.action_space_size(),
        **params,
    )
```

This allows `main.py` and all scripts to instantiate from config strings without importing specific classes.

---

## 13. Testing Strategy

### Environment conformance tests (`tests/test_environments.py`) [THE FUNCTION SIGNATURE TO BE CONFIRMED - NOT FIXED YET]

For every environment, verify:

- `reset()` returns an array of size `state_space_size()`
- `available_actions()` returns a non-empty list after reset
- `step(action)` with a legal action doesn't crash
- `step(action)` returns a tuple of (ndarray, float, bool) [FUNCTION SIGNATURE TO BE CONFIRMED - NOT FIXED YET]
- A game with random moves eventually terminates (done=True)
- `clone()` produces an independent copy (modifying clone doesn't affect original)

### Agent smoke tests (`tests/test_agents.py`)

For every agent:

- `act()` returns an action within `available_actions()`
- `save()` then `load()` produces an agent that acts identically

### When to run tests

Run `test_environments.py` after implementing or modifying any environment. Run `test_agents.py` after implementing a new agent. These tests are fast (no training) and catch interface violations immediately.

---

*This document will evolve as implementation decisions are made.*