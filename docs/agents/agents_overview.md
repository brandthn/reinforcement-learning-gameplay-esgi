# Vue d'ensemble des Agents

## Hierarchie des classes

```mermaid
classDiagram
    class Agent {
        <<abstract>>
        +act(state, available_actions, training) int
        +observe(state, action, reward, next_state, done)
        +end_episode()
        +save(path)
        +load(path)
        +name : str
    }

    class RandomAgent {
        +act() → random.choice(available)
    }

    class HumanAgent {
        -_pending_action: int|None
        +set_action(action)
        +act() → pending_action
    }

    class TabularQAgent {
        -_q: dict[tuple → ndarray]
        -_lr: float
        -_gamma: float
        -_step_count: int
        +act() → epsilon-greedy
        +observe() → Q-update
    }

    class DQNAgent {
        -_online_net: nn.Sequential
        -_target_net: nn.Sequential
        -_buffer: ReplayBuffer
        -_optimizer: Adam
        +act() → epsilon-greedy + masking
        +observe() → buffer + train_step
        #_compute_targets() → r + γ max Q_target
        #_train_step()
    }

    class DDQNAgent {
        #_compute_targets() → r + γ Q_target(s', argmax Q_online)
    }

    class DDQNERAgent {
        -_learning_starts: int
        +observe() → buffer + delayed train
    }

    class DDQNPERAgent {
        -_buffer: PrioritizedReplayBuffer
        -_per_alpha, _per_beta_*: float
        +observe() → PER buffer + IS-weighted train
        #_train_step() → weighted loss + priority update
    }

    Agent <|-- RandomAgent
    Agent <|-- HumanAgent
    Agent <|-- TabularQAgent
    Agent <|-- DQNAgent
    DQNAgent <|-- DDQNAgent
    DDQNAgent <|-- DDQNERAgent
    DDQNAgent <|-- DDQNPERAgent
```

---

## Tableau comparatif des agents

| Agent | Methode | Representation Q | Exploration | Replay | Complexite |
|-------|---------|-----------------|-------------|--------|------------|
| **Random** | Aucune | Aucune | 100% random | Non | O(1) |
| **TabularQ** | Q-Learning | Dict de Q-values | ε-greedy decay | Non | O(1) lookup |
| **DQN** | Deep Q-Network | MLP online + target | ε-greedy decay | Uniform | O(forward pass) |
| **DDQN** | Double DQN | Idem DQN | Idem DQN | Uniform | O(2 forward pass) |
| **DDQN+ER** | DDQN + warm-up | Idem | Idem | Uniform + warm-up | Idem |
| **DDQN+PER** | DDQN + priorite | Idem | Idem | Prioritized (sum-tree) | O(log N) sample |

---

## Interface Agent : le contrat

Chaque agent doit implementer `act()`. Les autres methodes sont optionnelles (no-op par defaut).

```mermaid
sequenceDiagram
    participant T as Trainer
    participant A as Agent

    Note over T,A: Phase d'action
    T->>A: act(state, available_actions, training=True)
    A-->>T: action (int)

    Note over T,A: Phase d'apprentissage
    T->>A: observe(state, action, reward, next_state, done)
    Note over A: Stocke transition<br/>Fait un pas d'optimisation

    Note over T,A: Fin d'episode
    T->>A: end_episode()
    Note over A: Target update (DQN)<br/>Decay epsilon
```

---

## Exploration : Epsilon-Greedy avec decay lineaire

Utilise par TabularQ, DQN, DDQN et variantes.

```
ε(t) = ε_start + (ε_end - ε_start) × min(t / decay_steps, 1.0)
```

```mermaid
graph LR
    A["t=0<br/>ε = 1.0<br/>100% exploration"] --> B["t = decay_steps/2<br/>ε = 0.5<br/>50% exploration"]
    B --> C["t = decay_steps<br/>ε = 0.01<br/>99% exploitation"]
    C --> D["t > decay_steps<br/>ε = 0.01<br/>(stable)"]
```

### Logique de decision

```mermaid
graph TD
    A["act(state, available, training)"]
    A --> B{"training = True ?"}
    B -->|"Non"| C["Mode inference : greedy pur"]
    B -->|"Oui"| D["Incrementer step_count"]
    D --> E{"random() < ε ?"}
    E -->|"Oui"| F["random.choice(available_actions)"]
    E -->|"Non"| G["Greedy : argmax Q avec masking"]

    C --> G

    style F fill:#FF9800,color:#fff
    style G fill:#4CAF50,color:#fff
```

---

## Sauvegarde et chargement

| Agent | Format | Contenu |
|-------|--------|---------|
| **Random** | Rien | Pas de parametres |
| **Human** | Rien | Pas de parametres |
| **TabularQ** | `.pt` (pickle) | Q-table (dict) + step_count |
| **DQN/DDQN/...** | `.pt` (torch) | online_net, target_net, optimizer, step_count, update_count |

---

## Compatibilite Agent × Environnement

| | LineWorld | GridWorld | TicTacToe | Bobail |
|---|---|---|---|---|
| **Random** | Oui | Oui | Oui (vs opponent) | Oui (vs opponent) |
| **Human** | Oui (clavier) | Oui (clavier) | Oui (clic) | Oui (clic 2 etapes) |
| **TabularQ** | Oui | Oui | Theorique (trop d'etats) | Non (espace trop grand) |
| **DQN** | Oui | Oui | Oui | Oui |
| **DDQN** | Oui | Oui | Oui | Oui |
| **DDQN+ER** | Oui | Oui | Oui | Oui |
| **DDQN+PER** | Oui | Oui | Oui | Oui |
