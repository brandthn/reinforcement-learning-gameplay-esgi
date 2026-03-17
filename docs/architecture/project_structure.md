# Architecture du Projet

## Arborescence

```
projet/
├── agents/                        # Implementations des agents
│   ├── __init__.py               # AGENT_REGISTRY + get_agent()
│   ├── base.py                   # Classe abstraite Agent
│   ├── random_agent.py           # Politique aleatoire uniforme
│   ├── human_agent.py            # Pont GUI → interface Agent
│   ├── tabular_q.py              # Q-Learning tabulaire
│   └── value_based/              # Agents Deep RL
│       ├── dqn.py                # Deep Q-Network
│       ├── ddqn.py               # Double DQN
│       ├── ddqn_er.py            # DDQN + warm-up Experience Replay
│       └── ddqn_per.py           # DDQN + Prioritized Experience Replay
│
├── environments/                  # Implementations des jeux
│   ├── __init__.py               # ENV_REGISTRY + get_env()
│   ├── base.py                   # Classe abstraite Environment
│   ├── line_world.py             # Navigation 1D
│   ├── grid_world.py             # Navigation 2D
│   ├── tictactoe.py              # Morpion 3x3
│   └── bobail.py                 # Bobail 5x5
│
├── training/                      # Infrastructure d'entrainement
│   ├── trainer.py                # Boucle single-player
│   ├── self_play.py              # Boucle adversarial (self-play)
│   ├── networks.py               # Constructeur MLP (PyTorch)
│   └── replay_buffer.py          # Uniform + Prioritized replay buffers
│
├── evaluation/                    # Pipeline d'evaluation
│   └── evaluator.py              # Metriques : reward, steps, latence
│
├── gui/                           # Interface graphique
│   └── app.py                    # Application PyGame complete
│
├── scripts/                       # Points d'entree CLI
│   ├── train.py                  # Entrainer 1 configuration
│   ├── train_all.py              # Entrainer toutes les configs
│   ├── train_sweep.py            # Sweep d'hyperparametres
│   ├── evaluate_all.py           # Re-evaluer les modeles sauvegardes
│   ├── promote_best.py           # Promouvoir le meilleur modele
│   ├── run_gui.py                # Lancer la GUI
│   └── benchmark.py              # Benchmark parties/seconde
│
├── configs/                       # Configurations YAML
│   ├── random/                   # 4 configs baseline random
│   ├── tabular_q/                # 4 configs Q-learning
│   ├── dqn/                      # 5 configs + sweeps
│   ├── ddqn/                     # 4 configs
│   ├── ddqn_er/                  # 4 configs
│   └── ddqn_per/                 # 5 configs + sweeps
│
├── tests/                         # Tests unitaires & integration
│   ├── test_environments.py
│   ├── test_agents.py
│   ├── test_training.py
│   └── test_value_based.py
│
├── results/                       # Modeles entraines (production)
├── results_dev/                   # Modeles (iteration rapide)
├── docs/                          # Documentation
├── README.md
├── pyproject.toml                 # Dependances (UV)
└── uv.lock
```

---

## Couches architecturales

```mermaid
graph TB
    subgraph "Couche Presentation"
        GUI["gui/app.py<br/>PyGame GUI"]
        CLI["scripts/*.py<br/>CLI"]
    end

    subgraph "Couche Orchestration"
        TR["training/trainer.py<br/>Single-Player Trainer"]
        SP["training/self_play.py<br/>Self-Play Trainer"]
        EV["evaluation/evaluator.py<br/>Evaluator"]
    end

    subgraph "Couche Metier"
        AG["agents/*<br/>7 types d'agents"]
        EN["environments/*<br/>4 environnements"]
    end

    subgraph "Couche Infrastructure"
        NET["training/networks.py<br/>MLP Builder"]
        BUF["training/replay_buffer.py<br/>Replay Buffers"]
        CFG["configs/*.yaml<br/>Configurations"]
    end

    GUI --> AG
    GUI --> EN
    CLI --> TR
    CLI --> SP
    TR --> AG
    TR --> EN
    TR --> EV
    SP --> AG
    SP --> EN
    SP --> EV
    AG --> NET
    AG --> BUF
    TR --> CFG
    SP --> CFG
```

---

## Systeme de Registres

L'architecture utilise un pattern **Registry** pour decoupler la configuration de l'instanciation.

```mermaid
graph LR
    subgraph "ENV_REGISTRY"
        E1["'line_world' → LineWorldEnv"]
        E2["'grid_world' → GridWorldEnv"]
        E3["'tictactoe' → TicTacToeEnv"]
        E4["'bobail' → BobailEnv"]
    end

    subgraph "AGENT_REGISTRY"
        A1["'random' → RandomAgent"]
        A2["'human' → HumanAgent"]
        A3["'tabular_q' → TabularQAgent"]
        A4["'dqn' → DQNAgent"]
        A5["'ddqn' → DDQNAgent"]
        A6["'ddqn_er' → DDQNERAgent"]
        A7["'ddqn_per' → DDQNPERAgent"]
    end

    YAML["config.yaml<br/>env: bobail<br/>agent: ddqn"]
    YAML -->|"get_env(name)"| E4
    YAML -->|"get_agent(name, env, params)"| A5
```

### Instanciation d'un agent

```python
def get_agent(name: str, env, params: dict = None):
    params = params or {}
    return AGENT_REGISTRY[name](
        state_size=env.state_space_size(),   # fourni par l'env
        action_size=env.action_space_size(),  # fourni par l'env
        **params,                             # hyperparametres du YAML
    )
```

---

## Flux de donnees : Entrainement

```mermaid
graph TD
    A["config.yaml"] -->|"charge"| B["scripts/train.py"]
    B --> C{"env.is_adversarial() ?"}
    C -->|"Non"| D["Trainer"]
    C -->|"Oui"| E["SelfPlayTrainer"]
    D --> F["Boucle episodes"]
    E --> F
    F -->|"checkpoint"| G["Evaluator"]
    G -->|"metriques"| H["metrics.csv"]
    F -->|"episode"| I["training_curve.csv"]
    F -->|"checkpoint"| J["model_N.pt"]

    style A fill:#FFD700,color:#000
    style H fill:#4CAF50,color:#fff
    style I fill:#4CAF50,color:#fff
    style J fill:#2196F3,color:#fff
```

---

## Flux de donnees : GUI

```mermaid
graph TD
    A["Utilisateur lance run_gui.py"] --> B["Menu : Selection env + agents"]
    B -->|"Start"| C["Charge l'environnement"]
    B -->|"Start"| D["Instancie les agents<br/>+ charge modeles .pt"]

    C --> E["Boucle de jeu"]
    D --> E

    E --> F{"Tour de qui ?"}
    F -->|"IA"| G["agent.act(state, available)"]
    G --> H["env.step(action)"]
    F -->|"Humain"| I["Attend input GUI"]
    I --> J["HumanAgent.set_action()"]
    J --> H

    H --> K["Render PyGame"]
    K --> L{"done ?"}
    L -->|"Non"| E
    L -->|"Oui"| M["Ecran Game Over"]

    style I fill:#FF9800,color:#fff
    style G fill:#2196F3,color:#fff
```

---

## Dependances

| Package | Usage |
|---------|-------|
| `torch` | Reseaux de neurones (DQN, DDQN) |
| `numpy` | Vecteurs d'etats, calculs numeriques |
| `pygame` | Interface graphique |
| `pyyaml` | Lecture des configurations |
| `pytest` | Tests (dev) |
