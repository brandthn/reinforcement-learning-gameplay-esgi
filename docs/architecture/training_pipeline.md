# Pipeline d'Entrainement

## Vue d'ensemble

Deux trainers sont implementes selon le type d'environnement :

| Trainer | Fichier | Utilisation |
|---------|---------|-------------|
| `Trainer` | `training/trainer.py` | Environnements single-player (LineWorld, GridWorld) |
| `SelfPlayTrainer` | `training/self_play.py` | Environnements adversariaux (TicTacToe, Bobail) |

```mermaid
graph TD
    A["Configuration YAML"] --> B{"env.is_adversarial() ?"}
    B -->|"Non<br/>(LineWorld, GridWorld)"| C["Trainer"]
    B -->|"Oui<br/>(TicTacToe, Bobail)"| D["SelfPlayTrainer"]

    C --> E["Agent unique"]
    D --> F["Agent (J0) + Opponent (J1)"]

    style C fill:#2196F3,color:#fff
    style D fill:#F44336,color:#fff
```

---

## 1. Trainer (Single-Player)

### Sequence d'un episode

```mermaid
sequenceDiagram
    participant T as Trainer
    participant E as Environment
    participant A as Agent

    T->>E: reset()
    E-->>T: state

    loop max_steps_per_episode fois
        T->>E: available_actions()
        E-->>T: actions legales
        T->>A: act(state, available, training=True)
        Note over A: Exploration epsilon-greedy
        A-->>T: action
        T->>E: step(action)
        E-->>T: (next_state, reward, done)
        T->>A: observe(state, action, reward, next_state, done)
        Note over A: Stocke dans replay buffer<br/>+ train step
        T->>T: state = next_state
        alt done == True
            T->>T: break
        end
    end

    T->>A: end_episode()
    Note over A: Decay epsilon, etc.
```

### Cycle complet d'entrainement

```mermaid
graph TD
    A["Debut entrainement"] --> B["Sauvegarder config.yaml"]
    B --> C["Creer CSV headers"]
    C --> D["Episode 1"]

    D --> E["_run_episode()"]
    E --> F["Ecrire (ep, reward, steps) dans training_curve.csv"]
    F --> G{"ep dans checkpoints ?"}
    G -->|"Non"| H["Episode suivant"]
    G -->|"Oui"| I["Evaluator.evaluate()"]
    I --> J["Ecrire metriques dans metrics.csv"]
    J --> K["agent.save(model_{ep}.pt)"]
    K --> H
    H --> L{"Dernier episode ?"}
    L -->|"Non"| E
    L -->|"Oui"| M["Retourner all_metrics"]

    style I fill:#FF9800,color:#fff
    style K fill:#4CAF50,color:#fff
```

---

## 2. SelfPlayTrainer (Adversarial)

### Le probleme du "Deferred Observe"

Dans un jeu a 2 joueurs, l'agent (joueur 0) **ne joue pas a chaque step**. Le probleme : quand l'agent joue a l'instant `t`, le `next_state` qu'il verra est celui quand ce sera **a nouveau son tour**, pas le state juste apres son action.

```mermaid
sequenceDiagram
    participant A as Agent (J0)
    participant E as Environment
    participant O as Opponent (J1)

    Note over E: Tour de J0
    A->>E: step(action_A) → state_1, reward_A, done
    Note over E: Tour de J1
    O->>E: step(action_O) → state_2, reward_O, done
    Note over E: Tour de J0 (encore)
    Note over A: MAINTENANT on peut observer<br/>la transition (state_0, action_A, reward_A, state_2, done)
```

### Mecanisme du Pending State

```mermaid
graph TD
    A["Debut episode"] --> B["pending_state = None"]
    B --> C["Tour du joueur X"]
    C --> D{"Joueur 0 ?"}

    D -->|"Oui"| E{"pending_state != None ?"}
    E -->|"Oui"| F["agent.observe(pending_state,<br/>pending_action, pending_reward,<br/>state, False)"]
    F --> G["action = agent.act(state, available, training=True)"]
    E -->|"Non"| G
    G --> H["pending_state = state<br/>pending_action = action<br/>pending_reward = 0.0"]

    D -->|"Non (Joueur 1)"| I["action = opponent.act(state, available, training=False)"]

    H --> J["next_state, reward, done = env.step(action)"]
    I --> J

    J --> K{"Joueur 0 qui a joue ?"}
    K -->|"Oui"| L["pending_reward += reward"]
    K -->|"Non"| M["(rien)"]

    L --> N{"done ?"}
    M --> N
    N -->|"Non"| C
    N -->|"Oui"| O["Flusher la transition terminale"]

    style F fill:#FF9800,color:#fff
    style O fill:#F44336,color:#fff
```

### Gestion des rewards a la terminaison

```mermaid
graph TD
    A["Partie terminee (done=True)"] --> B{"Qui a fait le dernier coup ?"}
    B -->|"Joueur 0 (agent)"| C["pending_reward += reward<br/>(sa propre victoire)"]
    B -->|"Joueur 1 (opponent)"| D["pending_reward -= reward<br/>(victoire adversaire = defaite)"]
    C --> E["agent.observe(pending_state,<br/>pending_action, pending_reward,<br/>next_state, True)"]
    D --> E

    style C fill:#4CAF50,color:#fff
    style D fill:#F44336,color:#fff
```

**Pourquoi `-= reward` quand l'adversaire gagne ?**

- Le reward de `env.step()` est du point de vue du **joueur qui vient de jouer**
- Si l'adversaire gagne, son reward = +1.0
- Du point de vue de l'agent, c'est une defaite = -1.0
- Donc : `pending_reward -= reward` (soit `0 - 1.0 = -1.0`)

---

## Configuration YAML

### Structure

```yaml
env: bobail                         # Nom dans ENV_REGISTRY
agent: ddqn                         # Nom dans AGENT_REGISTRY
opponent: random                    # (adversarial) Nom de l'opposant

agent_params:                       # Hyperparametres specifiques a l'agent
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
  num_episodes: 100000              # Nombre total d'episodes
  max_steps_per_episode: 200        # Limite par episode

eval:
  checkpoints: [1000, 10000, 100000]  # Episodes ou evaluer
  num_games: 100                       # Parties d'evaluation par checkpoint

seeds: [42, 123, 456]               # Seeds pour reproductibilite
```

### Scripts d'entrainement

```bash
# Entrainer une seule configuration
uv run python scripts/train.py configs/dqn/grid_world.yaml

# Mode rapide (pour iteration)
uv run python scripts/train.py configs/dqn/grid_world.yaml --quick --quick-episodes 500

# Entrainer toutes les configs d'un dossier
uv run python scripts/train_all.py configs/dqn/

# Sweep d'hyperparametres
uv run python scripts/train_sweep.py configs/dqn/grid_world_sweep.yaml
```

---

## Fichiers de sortie

```
results/{env}/{agent}/{run_name}_seed{N}/
├── config.yaml              # Snapshot de la configuration exacte
├── training_curve.csv       # episode, reward, steps
├── metrics.csv              # checkpoint, mean_reward, std_reward, ...
├── model_1000.pt            # Checkpoint a 1000 episodes
├── model_10000.pt           # Checkpoint a 10000 episodes
└── model_100000.pt          # Checkpoint a 100000 episodes
```

| Fichier | Contenu | Utilisation |
|---------|---------|-------------|
| `training_curve.csv` | `[episode, reward, steps]` par episode | Courbes d'apprentissage |
| `metrics.csv` | `[checkpoint, mean_reward, std_reward, mean_steps, std_steps, mean_action_time_ms, std_action_time_ms]` | Metriques du sujet |
| `model_N.pt` | Poids du reseau ou Q-table | Chargement dans la GUI |
| `config.yaml` | Config exacte de ce run | Reproductibilite |
