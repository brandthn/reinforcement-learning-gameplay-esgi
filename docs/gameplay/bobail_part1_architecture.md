# Part 1 — Architecture & Chaîne d'appels

> Objectif : comprendre **qui appelle quoi**, avec quelles signatures, et dans quel ordre,
> depuis le fichier de config jusqu'au `step()` de l'environnement.

---

## 1. Vue d'ensemble — Diagramme de dépendances

```
configs/random/bobail.yaml
        │
        ▼
┌───────────────────┐     ┌──────────────────────────┐
│ environments/     │     │ agents/                   │
│   __init__.py     │     │   __init__.py             │
│   ENV_REGISTRY    │     │   AGENT_REGISTRY          │
│   get_env()       │     │   get_agent()             │
└────────┬──────────┘     └────────┬─────────────────┘
         │                         │
         ▼                         ▼
┌────────────────┐        ┌────────────────┐
│ BobailEnv      │        │ RandomAgent    │
│ (Environment)  │        │ (Agent)        │
└────────┬───────┘        └────────┬───────┘
         │                         │
         └──────────┬──────────────┘
                    ▼
         ┌─────────────────────┐
         │ SelfPlayTrainer     │
         │ training/self_play  │
         │ _run_episode()      │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │ Evaluator           │
         │ evaluation/         │
         │ _play_adversarial() │
         └─────────────────────┘
```

---

## 2. Le fichier de configuration

**Fichier** : `configs/random/bobail.yaml`

```yaml
env: bobail          # clé dans ENV_REGISTRY → BobailEnv
agent: random        # clé dans AGENT_REGISTRY → RandomAgent
opponent: random     # clé dans AGENT_REGISTRY → RandomAgent (adversaire)

training:
  num_episodes: 1000
  max_steps_per_episode: 500

eval:
  checkpoints: [100, 500, 1000]
  num_games: 50

seed: 42
```

| Champ | Rôle | Consommé par |
|-------|------|-------------|
| `env` | Nom de l'environnement | `get_env()` |
| `agent` | Nom de l'agent principal (joueur 0) | `get_agent()` |
| `opponent` | Nom de l'adversaire (joueur 1) | `get_agent()` |
| `training.num_episodes` | Nombre d'épisodes d'entraînement | `SelfPlayTrainer.__init__()` |
| `training.max_steps_per_episode` | Sécurité anti-boucle infinie | `SelfPlayTrainer.__init__()` |
| `eval.checkpoints` | Épisodes où on évalue | `SelfPlayTrainer.__init__()` |
| `eval.num_games` | Parties par évaluation | `SelfPlayTrainer.__init__()` |
| `seed` | Graine aléatoire | Script d'entraînement |

---

## 3. Les Registries (Factories)

### 3.1 Environment Registry

**Fichier** : `environments/__init__.py`

```python
# Imports
from .bobail import BobailEnv

# Dictionnaire nom → classe
ENV_REGISTRY = {
    "line_world": LineWorldEnv,
    "grid_world": GridWorldEnv,
    "tictactoe": TicTacToeEnv,
    "bobail": BobailEnv,          # ← celui qui nous intéresse
}

def get_env(name: str, **kwargs):
    """Instancie un environnement par son nom."""
    return ENV_REGISTRY[name](**kwargs)
```

**Appel** : `get_env("bobail")` → `BobailEnv()` → appelle `BobailEnv.__init__()`

### 3.2 Agent Registry

**Fichier** : `agents/__init__.py`

```python
from .random_agent import RandomAgent

AGENT_REGISTRY = {
    "random": RandomAgent,
    "human": HumanAgent,
    "tabular_q": TabularQAgent,
}

def get_agent(name: str, env, params: dict = None):
    """Instancie un agent avec les dimensions de l'env."""
    params = params or {}
    return AGENT_REGISTRY[name](
        state_size=env.state_space_size(),   # BobailEnv → 75
        action_size=env.action_space_size(), # BobailEnv → 625
        **params,
    )
```

**Appel** : `get_agent("random", env)` → `RandomAgent(state_size=75, action_size=625)`

---

## 4. Toutes les classes et leurs signatures

### 4.1 Classe abstraite `Environment`

**Fichier** : `environments/base.py`

```python
class Environment(ABC):
    def reset(self) -> np.ndarray                                    # → state (vecteur 1D)
    def step(self, action: int) -> tuple[np.ndarray, float, bool]    # → (state, reward, done)
    def available_actions(self) -> list[int]                          # → [action_id, ...]
    def state_description(self) -> np.ndarray                        # → state (vecteur 1D)
    def action_space_size(self) -> int                               # → taille totale
    def state_space_size(self) -> int                                # → dimensionnalité du state
    def is_adversarial(self) -> bool                                 # → True/False
    def current_player(self) -> int                                  # → 0 ou 1
    def clone(self)                                                  # → deep copy
    def render_text(self) -> str                                     # → affichage texte
```

### 4.2 Classe `BobailEnv(Environment)`

**Fichier** : `environments/bobail.py`

```python
class BobailEnv(Environment):
    def __init__(self)                                                # pas de paramètre
    def reset(self) -> np.ndarray                                     # → vecteur float32[75]
    def step(self, action: int) -> tuple[np.ndarray, float, bool]     # action ∈ [0, 624]
    def available_actions(self) -> list[int]                           # → sous-ensemble de [0, 624]
    def state_description(self) -> np.ndarray                         # → float32[75]
    def action_space_size(self) -> int                                # → 625
    def state_space_size(self) -> int                                 # → 75
    def is_adversarial(self) -> bool                                  # → True
    def current_player(self) -> int                                   # → 0 ou 1
    def render_text(self) -> str                                      # → plateau 5x5 en texte

    # Méthodes privées :
    def _bobail_moves(self) -> list[int]                              # mouvements bobail légaux
    def _piece_moves(self) -> list[int]                               # mouvements pions légaux
```

### 4.3 Classe abstraite `Agent`

**Fichier** : `agents/base.py`

```python
class Agent(ABC):
    def act(self, state: np.ndarray, available_actions: list[int],
            training: bool = False) -> int                            # → action choisie
    def observe(self, state, action, reward, next_state, done)        # transition pour apprentissage
    def end_episode(self)                                             # fin d'épisode
    def save(self, path: str) -> None                                 # sauvegarde
    def load(self, path: str) -> None                                 # chargement
    @property
    def name(self) -> str                                             # → nom de la classe
```

### 4.4 Classe `RandomAgent(Agent)`

**Fichier** : `agents/random_agent.py`

```python
class RandomAgent(Agent):
    def __init__(self, state_size: int = 0, action_size: int = 0, **kwargs):
        pass  # ← ne stocke rien, n'a besoin de rien

    def act(self, state: np.ndarray, available_actions: list[int],
            training: bool = False) -> int:
        return random.choice(available_actions)  # ← choix uniforme
```

> **Point clé** : `RandomAgent` ignore complètement `state`, `state_size`, `action_size`,
> et `training`. Il pioche au hasard parmi les actions légales.

### 4.5 Classe `SelfPlayTrainer`

**Fichier** : `training/self_play.py`

```python
class SelfPlayTrainer:
    def __init__(self, env: Environment, agent: Agent, opponent: Agent, config: dict)
    def train(self, results_dir: str) -> dict           # boucle principale
    def _run_episode(self) -> tuple[float, int]          # un épisode complet
```

### 4.6 Classe `Evaluator`

**Fichier** : `evaluation/evaluator.py`

```python
class Evaluator:
    def evaluate(self, agent: Agent, env: Environment, num_games: int,
                 opponent: Agent = None) -> dict
    def _play_adversarial(self, agent, opponent, env, max_steps=10_000)
    def _play_single(self, agent, env, max_steps=10_000)
```

---

## 5. Chaîne d'appels complète — Qui appelle qui ?

```
Script d'entraînement
│
├─ 1. config = yaml.load("configs/random/bobail.yaml")
├─ 2. env = get_env("bobail")                        → BobailEnv.__init__()
├─ 3. agent = get_agent("random", env)                → RandomAgent(state_size=75, action_size=625)
├─ 4. opponent = get_agent("random", env)             → RandomAgent(state_size=75, action_size=625)
├─ 5. trainer = SelfPlayTrainer(env, agent, opponent, config)
│
└─ 6. trainer.train(results_dir)
     │
     └─ Pour chaque épisode (1..1000) :
        │
        ├─ trainer._run_episode()
        │   │
        │   ├─ env.reset()                            → BobailEnv.reset()
        │   │                                            → retourne state float32[75]
        │   │
        │   └─ Boucle (max 500 steps) :
        │       │
        │       ├─ player = env.current_player()       → 0 ou 1
        │       ├─ available = env.available_actions()  → BobailEnv.available_actions()
        │       │                                         → _bobail_moves() ou _piece_moves()
        │       │
        │       ├─ Si player == 0 :
        │       │   action = agent.act(state, available, training=True)
        │       │                                       → RandomAgent.act() → random.choice()
        │       ├─ Sinon :
        │       │   action = opponent.act(state, available, training=False)
        │       │                                       → RandomAgent.act() → random.choice()
        │       │
        │       ├─ next_state, reward, done = env.step(action)
        │       │                                       → BobailEnv.step()
        │       │
        │       ├─ agent.observe(...)                   → RandomAgent: no-op (hérité de Agent)
        │       │
        │       └─ Si done → break
        │
        ├─ agent.end_episode()                          → RandomAgent: no-op (hérité de Agent)
        │
        └─ Si épisode ∈ checkpoints :
            └─ evaluator.evaluate(agent, env, 50, opponent)
                └─ _play_adversarial() × 50 parties
```

---

## 6. Pourquoi `SelfPlayTrainer` et pas `Trainer` ?

| | `Trainer` | `SelfPlayTrainer` |
|---|-----------|-------------------|
| **Fichier** | `training/trainer.py` | `training/self_play.py` |
| **Joueurs** | 1 seul agent | 2 agents (agent + opponent) |
| **Envs** | Single-player (LineWorld, GridWorld) | Adversarial (TicTacToe, **Bobail**) |
| **Sélection** | `env.is_adversarial()` retourne `False` | `env.is_adversarial()` retourne `True` |
| **Observe** | Immédiat après chaque `step()` | **Différé** (deferred observe) |
| **Reward** | Direct | Perspective du joueur 0 |

Bobail est **adversarial** (`is_adversarial() → True`), donc c'est `SelfPlayTrainer` qui est utilisé.
