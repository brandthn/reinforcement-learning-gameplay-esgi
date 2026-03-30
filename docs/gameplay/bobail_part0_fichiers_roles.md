# Part 0 — Carte complète du projet : chaque fichier, son rôle, qui l'invoque

> Objectif : pour **chaque fichier** du projet, répondre à 4 questions :
> 1. **C'est quoi ?** — Que contient ce fichier ?
> 2. **Pourquoi existe-t-il ?** — Quel problème résout-il ?
> 3. **Qui l'invoque ?** — Quel fichier / quelle entité l'appelle ?
> 4. **Qui invoque-t-il ?** — De quels fichiers dépend-il ?

---

## Vue d'ensemble — Arbre du projet

```
projet/
├── main.py                          ← Point d'entrée CLI (stub)
│
├── configs/                         ← Fichiers de configuration YAML
│   └── random/
│       └── bobail.yaml
│
├── environments/                    ← Les environnements de jeu
│   ├── __init__.py                  ← Registry + factory
│   ├── base.py                      ← Classe abstraite Environment
│   └── bobail.py                    ← Implémentation Bobail
│
├── agents/                          ← Les agents (joueurs)
│   ├── __init__.py                  ← Registry + factory
│   ├── base.py                      ← Classe abstraite Agent
│   ├── random_agent.py              ← Agent aléatoire
│   └── human_agent.py               ← Agent humain (pour la GUI)
│
├── training/                        ← Boucles d'entraînement
│   ├── __init__.py                  ← Exports
│   ├── trainer.py                   ← Boucle single-player
│   └── self_play.py                 ← Boucle 2 joueurs (Bobail)
│
├── evaluation/                      ← Évaluation aux checkpoints
│   ├── __init__.py                  ← Export
│   └── evaluator.py                 ← Joue N parties, calcule stats
│
├── gui/                             ← Interface graphique Pygame
│   ├── __init__.py
│   └── app.py                       ← Application Pygame complète
│
├── scripts/                         ← Scripts exécutables
│   ├── run_gui.py                   ← Lance la GUI
│   └── benchmark.py                 ← Benchmark de performance
│
└── tests/                           ← Tests
    ├── test_environments.py
    └── test_agents.py
```

---

## Diagramme de qui-appelle-qui

```
L'UTILISATEUR tape une commande
         │
         ├──── uv run scripts/run_gui.py ─────────► scripts/run_gui.py
         │                                                │
         │                                          gui/app.py
         │                                           │       │
         │                                           ▼       ▼
         │                                    environments/  agents/
         │                                    __init__.py    __init__.py
         │                                        │              │
         │                                        ▼              ▼
         │                                    bobail.py    random_agent.py
         │                                        │        human_agent.py
         │                                        ▼
         │                                    base.py (Environment)
         │
         │
         ├──── uv run scripts/benchmark.py ───────► scripts/benchmark.py
         │                                                │
         │                                          environments/
         │                                          __init__.py → bobail.py → base.py
         │
         │
         └──── (futur) uv run main.py ────────────► main.py (stub vide)
                  avec un config YAML                    │
                                                   training/self_play.py
                                                    │          │
                                                    ▼          ▼
                                              environments/  agents/
                                                    │          │
                                                    ▼          ▼
                                              evaluation/evaluator.py
```

---

## Fiches détaillées par fichier

---

### `configs/random/bobail.yaml`

| Question | Réponse |
|----------|---------|
| **C'est quoi ?** | Fichier de configuration YAML qui définit les paramètres d'une expérience |
| **Pourquoi ?** | Séparer la config du code : on change les paramètres sans toucher au Python |
| **Qui l'invoque ?** | Un script d'entraînement (futur) le charge avec `yaml.load()`. La GUI (`gui/app.py`) scanne aussi les dossiers `configs/` pour proposer les agents disponibles |
| **Qui invoque-t-il ?** | Rien — c'est un fichier de données passif |

```yaml
env: bobail              # → clé pour ENV_REGISTRY
agent: random            # → clé pour AGENT_REGISTRY
opponent: random         # → clé pour AGENT_REGISTRY
training:
  num_episodes: 1000     # → SelfPlayTrainer._num_episodes
  max_steps_per_episode: 500  # → SelfPlayTrainer._max_steps
eval:
  checkpoints: [100, 500, 1000]  # → SelfPlayTrainer._checkpoints
  num_games: 50                  # → SelfPlayTrainer._eval_games
seed: 42
```

---

### `environments/base.py`

| Question | Réponse |
|----------|---------|
| **C'est quoi ?** | Classe abstraite `Environment` — le **contrat** que tout environnement doit respecter |
| **Pourquoi ?** | Pour que `Trainer`, `SelfPlayTrainer`, `Evaluator` et `gui/app.py` puissent utiliser N'IMPORTE QUEL environnement sans connaître son implémentation. C'est du **polymorphisme** |
| **Qui l'invoque ?** | `BobailEnv` en **hérite**. `Trainer`, `SelfPlayTrainer`, `Evaluator` l'utilisent comme **type hint** dans leurs signatures |
| **Qui invoque-t-il ?** | Seulement `copy.deepcopy` (pour `clone()`) et `numpy` |

```python
class Environment(ABC):
    def reset(self) -> np.ndarray                    # OBLIGATOIRE à implémenter
    def step(self, action: int) -> (state, reward, done)  # OBLIGATOIRE
    def available_actions(self) -> list[int]          # OBLIGATOIRE
    def state_description(self) -> np.ndarray         # OBLIGATOIRE
    def action_space_size(self) -> int                # OBLIGATOIRE
    def state_space_size(self) -> int                 # OBLIGATOIRE
    def is_adversarial(self) -> bool                  # Optionnel (défaut: False)
    def current_player(self) -> int                   # Optionnel (défaut: 0)
    def clone(self)                                   # Optionnel (deep copy)
    def render_text(self) -> str                      # Optionnel (debug)
```

**Chaîne d'héritage :**
```
Environment (ABC)           ← environments/base.py
    ├── LineWorldEnv         ← environments/line_world.py
    ├── GridWorldEnv         ← environments/grid_world.py
    ├── TicTacToeEnv         ← environments/tictactoe.py
    └── BobailEnv            ← environments/bobail.py
```

---

### `environments/__init__.py`

| Question | Réponse |
|----------|---------|
| **C'est quoi ?** | Le **registry** (registre) et la **factory** (usine) des environnements |
| **Pourquoi ?** | Pour pouvoir créer un environnement à partir d'une **string** (`"bobail"`) lue dans un fichier YAML, sans avoir à écrire `if name == "bobail": return BobailEnv()` partout |
| **Qui l'invoque ?** | `gui/app.py` (pour lister et créer les envs), `scripts/benchmark.py` (pour itérer sur tous les envs), et le futur script d'entraînement |
| **Qui invoque-t-il ?** | Importe les 4 classes d'environnement (`BobailEnv`, `LineWorldEnv`, etc.) |

```python
ENV_REGISTRY = {
    "bobail": BobailEnv,        # string → classe
    "tictactoe": TicTacToeEnv,
    ...
}

def get_env(name: str, **kwargs):       # "bobail" → BobailEnv()
    return ENV_REGISTRY[name](**kwargs)
```

**Qui appelle `get_env()` :**
```
scripts/benchmark.py:18  →  env = get_env(env_name)
gui/app.py:17            →  from environments import get_env
(futur script)           →  env = get_env(config["env"])
```

---

### `environments/bobail.py`

| Question | Réponse |
|----------|---------|
| **C'est quoi ?** | L'implémentation complète du jeu Bobail : plateau, pièces, règles, phases, conditions de victoire |
| **Pourquoi ?** | C'est le **simulateur** du jeu. Sans lui, impossible de jouer une partie. Il implémente le contrat `Environment` |
| **Qui l'invoque ?** | `environments/__init__.py` l'importe et le met dans `ENV_REGISTRY`. Ensuite, tout le monde utilise l'objet via les méthodes de `Environment` : `SelfPlayTrainer`, `Evaluator`, `gui/app.py` |
| **Qui invoque-t-il ?** | `environments/base.py` (héritage), `numpy` |

```
Les méthodes appelées et par qui :
┌──────────────────────────┬────────────────────────────────────────────────┐
│ Méthode                  │ Appelée par                                    │
├──────────────────────────┼────────────────────────────────────────────────┤
│ __init__()               │ get_env("bobail") → ENV_REGISTRY["bobail"]()  │
│ reset()                  │ SelfPlayTrainer._run_episode():97              │
│                          │ Evaluator._play_adversarial():73               │
│                          │ gui/app.py (quand on lance une partie)         │
│                          │ scripts/benchmark.py:23                        │
│ step(action)             │ SelfPlayTrainer._run_episode():133             │
│                          │ Evaluator._play_adversarial():91               │
│                          │ gui/app.py (après choix d'action)              │
│                          │ scripts/benchmark.py:28                        │
│ available_actions()      │ SelfPlayTrainer._run_episode():113             │
│                          │ Evaluator._play_adversarial():80               │
│                          │ gui/app.py (pour afficher les coups possibles) │
│                          │ scripts/benchmark.py:27                        │
│ state_description()      │ Appelée en interne par reset() et step()      │
│ action_space_size()      │ agents/__init__.py:15 (via get_agent)          │
│ state_space_size()       │ agents/__init__.py:14 (via get_agent)          │
│ is_adversarial()         │ Evaluator.evaluate():25                        │
│                          │ gui/app.py (pour adapter l'interface)          │
│ current_player()         │ SelfPlayTrainer._run_episode():112             │
│                          │ Evaluator._play_adversarial():79               │
│                          │ gui/app.py (pour savoir à qui le tour)         │
│ render_text()            │ gui/app.py (affichage debug)                   │
└──────────────────────────┴────────────────────────────────────────────────┘
```

---

### `agents/base.py`

| Question | Réponse |
|----------|---------|
| **C'est quoi ?** | Classe abstraite `Agent` — le **contrat** que tout agent doit respecter |
| **Pourquoi ?** | Même raison que `Environment` : polymorphisme. `SelfPlayTrainer` et `Evaluator` travaillent avec N'IMPORTE QUEL agent sans connaître son implémentation |
| **Qui l'invoque ?** | `RandomAgent` et `HumanAgent` en **héritent**. `Trainer`, `SelfPlayTrainer`, `Evaluator` l'utilisent comme type hint |
| **Qui invoque-t-il ?** | Seulement `numpy` (pour le type `np.ndarray`) |

```python
class Agent(ABC):
    def act(state, available_actions, training) -> int  # OBLIGATOIRE
    def observe(state, action, reward, next_state, done) # Optionnel (défaut: no-op)
    def end_episode()                                    # Optionnel (défaut: no-op)
    def save(path)                                       # Optionnel (défaut: no-op)
    def load(path)                                       # Optionnel (défaut: no-op)
    @property name -> str                                # Retourne le nom de la classe
```

**Chaîne d'héritage :**
```
Agent (ABC)               ← agents/base.py
    ├── RandomAgent        ← agents/random_agent.py
    └── HumanAgent         ← agents/human_agent.py
```

---

### `agents/__init__.py`

| Question | Réponse |
|----------|---------|
| **C'est quoi ?** | Le **registry** et la **factory** des agents |
| **Pourquoi ?** | Pour créer un agent à partir d'une string (`"random"`) + les dimensions de l'environnement, sans if/else |
| **Qui l'invoque ?** | `gui/app.py` (pour lister et créer les agents), le futur script d'entraînement |
| **Qui invoque-t-il ?** | Importe `RandomAgent`, `HumanAgent`. Appelle `env.state_space_size()` et `env.action_space_size()` |

```python
def get_agent(name: str, env, params: dict = None):
    params = params or {}
    return AGENT_REGISTRY[name](
        state_size=env.state_space_size(),    # ← appelle BobailEnv → 75
        action_size=env.action_space_size(),  # ← appelle BobailEnv → 625
        **params,
    )
```

**Flux d'appels :**
```
get_agent("random", env)
    │
    ├── env.state_space_size()      → BobailEnv.state_space_size() → 75
    ├── env.action_space_size()     → BobailEnv.action_space_size() → 625
    │
    └── AGENT_REGISTRY["random"](state_size=75, action_size=625)
        └── RandomAgent.__init__(state_size=75, action_size=625)
            └── pass  (ignore tout)
```

---

### `agents/random_agent.py`

| Question | Réponse |
|----------|---------|
| **C'est quoi ?** | Un agent qui choisit une action au hasard parmi les actions légales |
| **Pourquoi ?** | C'est le **baseline** le plus simple. Il sert de référence : tout agent intelligent doit faire mieux qu'un agent aléatoire. Il sert aussi d'adversaire pour le self-play |
| **Qui l'invoque ?** | `agents/__init__.py` l'importe. `SelfPlayTrainer` l'appelle via `agent.act()` et `opponent.act()`. `Evaluator` l'appelle via `agent.act()` |
| **Qui invoque-t-il ?** | `agents/base.py` (héritage), `random.choice()` |

```python
class RandomAgent(Agent):
    def __init__(self, state_size=0, action_size=0, **kwargs):
        pass                                    # N'a besoin de RIEN

    def act(self, state, available_actions, training=False):
        return random.choice(available_actions)  # Pioche au hasard

    # observe() → hérité de Agent → pass (no-op)
    # end_episode() → hérité de Agent → pass (no-op)
    # save() → hérité de Agent → pass (no-op)
```

---

### `agents/human_agent.py`

| Question | Réponse |
|----------|---------|
| **C'est quoi ?** | Un agent dont l'action est définie par un humain via la GUI |
| **Pourquoi ?** | Pour permettre à un humain de **jouer** contre une IA dans la GUI Pygame |
| **Qui l'invoque ?** | `gui/app.py` crée un `HumanAgent`, appelle `set_action()` quand l'humain clique, puis la boucle de jeu appelle `act()` |
| **Qui invoque-t-il ?** | `agents/base.py` (héritage) |

```
Flux dans la GUI :
1. Humain clique sur une case         → gui/app.py détecte le clic
2. gui/app.py calcule l'action        → human_agent.set_action(action)
3. Boucle de jeu appelle              → human_agent.act(state, available)
4. act() retourne l'action pendante   → env.step(action)
```

---

### `training/trainer.py`

| Question | Réponse |
|----------|---------|
| **C'est quoi ?** | Boucle d'entraînement pour environnements **single-player** (1 seul agent) |
| **Pourquoi ?** | Pour entraîner un agent sur LineWorld ou GridWorld où il n'y a qu'un seul joueur |
| **Qui l'invoque ?** | Le futur script d'entraînement, quand `env.is_adversarial()` retourne `False` |
| **Qui invoque-t-il ?** | `env.reset()`, `env.step()`, `env.available_actions()`, `agent.act()`, `agent.observe()`, `agent.end_episode()`, `agent.save()`, `Evaluator.evaluate()` |

**PAS utilisé pour Bobail** (Bobail est adversarial → `SelfPlayTrainer`).

```python
class Trainer:
    def __init__(self, env, agent, config)
    def train(self, results_dir) → dict
    def _run_episode(self) → (reward, steps)
```

---

### `training/self_play.py`

| Question | Réponse |
|----------|---------|
| **C'est quoi ?** | Boucle d'entraînement pour environnements **2 joueurs** avec self-play et deferred observe |
| **Pourquoi ?** | Parce que dans un jeu à 2 joueurs, l'agent n'agit pas à chaque step. Le `next_state` immédiat est dans la perspective de l'adversaire. Il faut **différer** l'observe jusqu'au prochain tour de l'agent pour que Q-learning fonctionne correctement |
| **Qui l'invoque ?** | Le futur script d'entraînement, quand `env.is_adversarial()` retourne `True`. C'est le cas pour **Bobail** et **TicTacToe** |
| **Qui invoque-t-il ?** | Voir tableau ci-dessous |

```
┌─────────────────────────────────┬───────────────────────────────────────────┐
│ Appel dans self_play.py         │ Fichier/Méthode invoquée                  │
├─────────────────────────────────┼───────────────────────────────────────────┤
│ self._env.reset()               │ environments/bobail.py → BobailEnv.reset()│
│ self._env.current_player()      │ environments/bobail.py → current_player() │
│ self._env.available_actions()   │ environments/bobail.py → available_actions │
│ self._env.step(action)          │ environments/bobail.py → step()           │
│ self._agent.act(s, a, True)     │ agents/random_agent.py → RandomAgent.act()│
│ self._agent.observe(s,a,r,s',d) │ agents/base.py → Agent.observe() (no-op)  │
│ self._agent.end_episode()       │ agents/base.py → Agent.end_episode()(noop)│
│ self._agent.save(path)          │ agents/base.py → Agent.save() (no-op)     │
│ self._opponent.act(s, a, False) │ agents/random_agent.py → RandomAgent.act()│
│ self._evaluator.evaluate(...)   │ evaluation/evaluator.py → Evaluator       │
│ yaml.dump(...)                  │ bibliothèque PyYAML                       │
│ csv.writer(...)                 │ bibliothèque standard Python              │
└─────────────────────────────────┴───────────────────────────────────────────┘
```

---

### `training/__init__.py`

| Question | Réponse |
|----------|---------|
| **C'est quoi ?** | Simple fichier d'export qui rend `Trainer` et `SelfPlayTrainer` importables |
| **Pourquoi ?** | Pour pouvoir écrire `from training import SelfPlayTrainer` au lieu de `from training.self_play import SelfPlayTrainer` |
| **Qui l'invoque ?** | Le futur script d'entraînement |
| **Qui invoque-t-il ?** | `training/trainer.py`, `training/self_play.py` (imports) |

```python
from .trainer import Trainer
from .self_play import SelfPlayTrainer
```

---

### `evaluation/evaluator.py`

| Question | Réponse |
|----------|---------|
| **C'est quoi ?** | Classe qui joue N parties avec une politique **gelée** (training=False) et collecte des statistiques |
| **Pourquoi ?** | Pour mesurer **objectivement** la performance de l'agent à des moments précis de l'entraînement (checkpoints). Séparer évaluation et entraînement évite que l'exploration (epsilon-greedy) pollue les métriques |
| **Qui l'invoque ?** | `SelfPlayTrainer.train()` (ligne 74) et `Trainer.train()` (ligne 67), uniquement aux checkpoints |
| **Qui invoque-t-il ?** | Voir tableau ci-dessous |

```
┌──────────────────────────────────┬──────────────────────────────────────────┐
│ Appel dans evaluator.py          │ Fichier/Méthode invoquée                 │
├──────────────────────────────────┼──────────────────────────────────────────┤
│ env.is_adversarial()             │ environments/bobail.py → True            │
│ env.reset()                      │ environments/bobail.py → reset()         │
│ env.current_player()             │ environments/bobail.py → current_player()│
│ env.available_actions()          │ environments/bobail.py → available_act.. │
│ env.step(action)                 │ environments/bobail.py → step()          │
│ agent.act(s, avail, False)       │ agents/random_agent.py → act()           │
│ opponent.act(s, avail, False)    │ agents/random_agent.py → act()           │
│ time.perf_counter()              │ bibliothèque standard Python             │
│ np.mean(), np.std(), np.array()  │ bibliothèque NumPy                       │
└──────────────────────────────────┴──────────────────────────────────────────┘
```

**Quand exactement est-il invoqué :**
```python
# training/self_play.py:68-78
for ep in range(1, 1001):
    reward, steps = self._run_episode()          # ← CHAQUE épisode
    if ep in self._checkpoints:                  # ← {100, 500, 1000} seulement
        eval_result = self._evaluator.evaluate(  # ← ICI
            self._agent, self._env, 50,
            opponent=self._opponent,
        )
```

---

### `evaluation/__init__.py`

| Question | Réponse |
|----------|---------|
| **C'est quoi ?** | Simple fichier d'export |
| **Pourquoi ?** | Pour écrire `from evaluation import Evaluator` |
| **Qui l'invoque ?** | Personne directement — c'est `evaluation.evaluator` qui est importé par `Trainer` et `SelfPlayTrainer` |
| **Qui invoque-t-il ?** | `evaluation/evaluator.py` (import) |

---

### `gui/app.py`

| Question | Réponse |
|----------|---------|
| **C'est quoi ?** | Application Pygame complète : menu, sélection env/agents, rendu du plateau, gestion des clics, boucle de jeu |
| **Pourquoi ?** | Pour **visualiser** les parties (IA vs IA) ou **jouer** (humain vs IA). C'est l'outil de debug et de démo |
| **Qui l'invoque ?** | `scripts/run_gui.py` appelle `gui.app.run()` |
| **Qui invoque-t-il ?** | Presque tout le projet : |

```
gui/app.py importe et utilise :
├── environments/__init__.py  → ENV_REGISTRY, get_env()
├── environments/bobail.py    → BOARD_SIZE, PHASE_BOBAIL, _idx_to_rc, _rc_to_idx
├── agents/__init__.py        → AGENT_REGISTRY, get_agent()
├── agents/human_agent.py     → HumanAgent, set_action()
└── pygame, numpy, yaml       → bibliothèques externes
```

---

### `scripts/run_gui.py`

| Question | Réponse |
|----------|---------|
| **C'est quoi ?** | Script exécutable qui lance la GUI |
| **Pourquoi ?** | Point d'entrée pour l'utilisateur. Configure le `sys.path` et appelle `gui.app.run()` |
| **Qui l'invoque ?** | L'**utilisateur** tape `uv run scripts/run_gui.py` dans le terminal |
| **Qui invoque-t-il ?** | `gui/app.py` → `run()` |

```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # ajoute projet/ au path
from gui.app import run
run()     # ← lance la boucle Pygame
```

---

### `scripts/benchmark.py`

| Question | Réponse |
|----------|---------|
| **C'est quoi ?** | Script qui joue N parties aléatoires sur chaque environnement et mesure la vitesse (parties/sec) |
| **Pourquoi ?** | Pour vérifier que les environnements sont assez rapides pour l'entraînement. Un env lent = entraînement lent |
| **Qui l'invoque ?** | L'**utilisateur** tape `uv run scripts/benchmark.py` |
| **Qui invoque-t-il ?** | `environments/__init__.py` → `ENV_REGISTRY`, `get_env()`. Utilise `random.choice()` directement (pas d'agent) |

```python
for env_name in ENV_REGISTRY:             # itère sur tous les envs
    env = get_env(env_name)               # crée l'env
    for _ in range(n_games):
        env.reset()
        while not done:
            action = random.choice(env.available_actions())
            _, _, done = env.step(action)
```

---

### `main.py`

| Question | Réponse |
|----------|---------|
| **C'est quoi ?** | Point d'entrée CLI prévu pour le futur — actuellement un **stub vide** |
| **Pourquoi ?** | Destiné à devenir le script principal qui charge un YAML, crée env + agents + trainer, et lance l'entraînement |
| **Qui l'invoque ?** | Personne pour l'instant (`main()` fait `pass`) |
| **Qui invoque-t-il ?** | Rien pour l'instant |

---

## Diagramme final — Chaîne d'invocation complète pour Bobail + RandomAgent

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  L'utilisateur lance l'entraînement (futur main.py ou script)           │
│                                                                          │
│  1. Charge configs/random/bobail.yaml                                   │
│         │                                                                │
│  2. environments/__init__.py                                            │
│         │  get_env("bobail")                                            │
│         ▼                                                                │
│     environments/bobail.py                                              │
│         │  BobailEnv() hérite de                                        │
│         ▼                                                                │
│     environments/base.py                                                │
│         │  class Environment(ABC)                                       │
│         │                                                                │
│  3. agents/__init__.py                                                  │
│         │  get_agent("random", env)                                     │
│         │  appelle env.state_space_size() → 75                          │
│         │  appelle env.action_space_size() → 625                        │
│         ▼                                                                │
│     agents/random_agent.py                                              │
│         │  RandomAgent(75, 625) hérite de                               │
│         ▼                                                                │
│     agents/base.py                                                      │
│         │  class Agent(ABC)                                             │
│         │                                                                │
│  4. training/self_play.py                                               │
│         │  SelfPlayTrainer(env, agent, opponent, config)                │
│         │  .train(results_dir)                                          │
│         │    │                                                           │
│         │    ├── _run_episode() × 1000                                  │
│         │    │     appelle: env.reset()                                  │
│         │    │     appelle: env.current_player()                        │
│         │    │     appelle: env.available_actions()                     │
│         │    │     appelle: agent.act() / opponent.act()                │
│         │    │     appelle: env.step()                                  │
│         │    │     appelle: agent.observe()                             │
│         │    │     appelle: agent.end_episode()                         │
│         │    │                                                           │
│         │    └── evaluator.evaluate() × 3 (aux checkpoints)            │
│         │          │                                                     │
│         ▼          ▼                                                     │
│     evaluation/evaluator.py                                             │
│         │  Evaluator._play_adversarial() × 50                          │
│         │  appelle: env.reset(), env.step(), agent.act(), opponent.act()│
│         │  retourne: {mean_reward, std_reward, mean_steps, ...}         │
│         │                                                                │
│  5. Résultats écrits dans results_dir/ :                                │
│         ├── config.yaml                                                 │
│         ├── training_curve.csv   (1000 lignes)                          │
│         ├── metrics.csv          (3 lignes : checkpoints 100,500,1000)  │
│         └── model_100.pt, model_500.pt, model_1000.pt (vides pour Random)│
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```
