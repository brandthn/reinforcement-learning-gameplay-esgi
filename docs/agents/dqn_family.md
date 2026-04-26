# Famille DQN : Deep Q-Networks

## Evolution des algorithmes

```mermaid
graph TD
    A["DQN<br/>Deep Q-Network<br/>(Mnih et al., 2015)"]
    A --> B["DDQN<br/>Double DQN<br/>(van Hasselt et al., 2016)"]
    B --> C["DDQN + ER<br/>+ Warm-up<br/>Experience Replay"]
    B --> D["DDQN + PER<br/>+ Prioritized<br/>Experience Replay<br/>(Schaul et al., 2016)"]

    A1["Probleme: surestimation Q"] --> B
    B1["Probleme: echantillons peu divers<br/>en debut d'entrainement"] --> C
    B2["Probleme: echantillonnage<br/>uniforme sous-optimal"] --> D

    style A fill:#2196F3,color:#fff
    style B fill:#4CAF50,color:#fff
    style C fill:#FF9800,color:#fff
    style D fill:#F44336,color:#fff
```

---

## 1. DQN — Deep Q-Network

### Architecture

```mermaid
graph LR
    subgraph "Online Network"
        I1["Input<br/>state (s)"] --> H1["Hidden 1<br/>Linear + ReLU"]
        H1 --> H2["Hidden 2<br/>Linear + ReLU"]
        H2 --> O1["Output<br/>Q(s, a) pour chaque a"]
    end

    subgraph "Target Network"
        I2["Input<br/>next_state (s')"] --> H3["Hidden 1<br/>(copie figee)"]
        H3 --> H4["Hidden 2<br/>(copie figee)"]
        H4 --> O2["Output<br/>Q_target(s', a)"]
    end

    O1 -.->|"Copie periodique<br/>des poids"| I2
```

### Formule de la target DQN

```
target = r + γ × max_a' Q_target(s', a') × (1 - done)
```

### Pas d'optimisation

```mermaid
graph TD
    A["Replay Buffer"] -->|"Sample batch"| B["Batch de transitions<br/>(s, a, r, s', done)"]
    B --> C["Q_online(s) → q_values"]
    B --> D["Q_target(s') → next_q"]

    C --> E["q_selected = q_values[a]"]
    D --> F["target = r + γ × max(next_q) × (1-done)"]

    E --> G["loss = MSE(q_selected, target)"]
    F --> G

    G --> H["Backprop + Adam"]
    H --> I["Update online_net"]

    I --> J{"update_count % freq == 0 ?"}
    J -->|"Oui"| K["target_net ← online_net"]
    J -->|"Non"| L["Continuer"]

    style A fill:#FFD700,color:#000
    style K fill:#FF9800,color:#fff
```

### Hyperparametres DQN

| Param | Description | Typique |
|-------|-------------|---------|
| `lr` | Learning rate Adam | 0.0001 - 0.001 |
| `gamma` | Facteur de discount | 0.99 |
| `epsilon_start` | ε initial | 1.0 |
| `epsilon_end` | ε final | 0.01 |
| `epsilon_decay_steps` | Steps pour le decay | 10000 - 50000 |
| `hidden_layers` | Architecture MLP | [64, 64] ou [128, 128] |
| `batch_size` | Taille du mini-batch | 32 ou 64 |
| `buffer_capacity` | Taille max du replay buffer | 10000 - 50000 |
| `target_update_freq` | Frequence de copie target | 100 - 500 |

---

## 2. DDQN — Double Deep Q-Network

### Probleme resolu : surestimation des Q-values

DQN utilise le **meme reseau** pour selectionner ET evaluer la meilleure action :
```
target_DQN = r + γ × max_a' Q_target(s', a')
                      ↑ sélection ET évaluation par target_net
```

Cela cause une **surestimation systematique** car les erreurs d'estimation sont toujours dans la direction positive (max d'estimations bruitees > vraie valeur).

### Solution DDQN

**Decoupler** la selection et l'evaluation :

```
best_a = argmax_a' Q_online(s', a')       ← online choisit la meilleure action
target_DDQN = r + γ × Q_target(s', best_a)  ← target evalue cette action
```

```mermaid
graph LR
    subgraph "DQN (probleme)"
        A1["Q_target(s')"] --> B1["max → action + valeur"]
        B1 --> C1["target = r + γ × valeur"]
    end

    subgraph "DDQN (solution)"
        A2["Q_online(s')"] --> B2["argmax → meilleure action"]
        A3["Q_target(s')"] --> C2["evaluer a cette action"]
        B2 --> C2
        C2 --> D2["target = r + γ × valeur"]
    end

    style B1 fill:#F44336,color:#fff
    style B2 fill:#4CAF50,color:#fff
    style C2 fill:#4CAF50,color:#fff
```

### Code (la seule methode qui change)

```python
class DDQNAgent(DQNAgent):
    def _compute_targets(self, rewards_t, next_states_t, dones_t):
        with torch.no_grad():
            best_actions = self._online_net(next_states_t).argmax(dim=1)  # online choisit
            next_q = self._target_net(next_states_t)
            next_q_max = next_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)  # target evalue
        return rewards_t + self._gamma * next_q_max * (~dones_t).float()
```

---

## 3. DDQN+ER — Warm-up Experience Replay

### Probleme resolu : echantillons peu divers en debut d'entrainement

Au debut de l'entrainement, le replay buffer ne contient que quelques transitions. L'agent apprend sur des donnees tres correlees et peu representatives.

### Solution : delai avant apprentissage

```mermaid
graph LR
    A["Phase de collecte<br/>(learning_starts steps)"] --> B["Phase d'entrainement<br/>(buffer assez rempli)"]

    A1["Buffer vide"] --> A2["Buffer partiellement rempli<br/>Pas d'apprentissage"]
    A2 --> A3["Buffer >= learning_starts"]
    A3 --> B1["Apprentissage commence<br/>Buffer diversifie"]

    style A fill:#FFD700,color:#000
    style B fill:#4CAF50,color:#fff
```

### Condition d'entrainement

```python
def observe(self, state, action, reward, next_state, done):
    self._buffer.push(state, action, reward, next_state, done)
    # Ne pas entrainer tant que :
    # 1. Le buffer n'a pas assez de samples pour un batch
    # 2. On n'a pas atteint learning_starts steps
    if len(self._buffer) >= self._batch_size and self._step_count >= self._learning_starts:
        self._train_step()
```

---

## 4. DDQN+PER — Prioritized Experience Replay

### Probleme resolu : echantillonnage uniforme

L'echantillonnage uniforme du replay buffer traite toutes les transitions de maniere egale. Or, certaines transitions sont plus **informatives** (TD-error eleve) et devraient etre rejoues plus souvent.

### Solution : priorite proportionnelle au TD-error

```mermaid
graph TD
    A["Transition (s, a, r, s', done)"]
    A --> B["Calculer TD-error<br/>δ = |Q(s,a) - target|"]
    B --> C["Priorite = (|δ| + ε)^α"]
    C --> D["Stocker dans Sum-Tree"]
    D --> E["Echantillonner proportionnellement<br/>aux priorites"]

    E --> F["Correction Importance Sampling<br/>weight = (N × p_i)^(-β) / max_weight"]
    F --> G["Loss pondere<br/>= mean(weight × (Q - target)^2)"]

    style C fill:#FF9800,color:#fff
    style F fill:#2196F3,color:#fff
```

### Structure Sum-Tree

```
                    [Total: 15.0]           ← Racine (somme totale)
                   /              \
          [8.0]                    [7.0]    ← Noeuds internes (sommes partielles)
         /     \                  /     \
      [3.0] [5.0]            [4.0] [3.0]   ← Feuilles (priorites)
        ↓      ↓                ↓      ↓
      trans_0 trans_1        trans_2 trans_3  ← Transitions stockees
```

**Echantillonnage O(log N)** : descendre l'arbre avec une valeur aleatoire uniforme dans [0, Total].

### Annealing de Beta (Importance Sampling)

```
β(t) = β_start + (β_end - β_start) × min(t / beta_steps, 1.0)
```

| Phase | β | Effet |
|-------|---|-------|
| **Debut** | β_start ≈ 0.4 | Correction partielle (plus de biais, mais plus stable) |
| **Milieu** | ~0.7 | Correction croissante |
| **Fin** | β_end = 1.0 | Correction complete (pas de biais) |

### Hyperparametres supplementaires PER

| Param | Description | Typique |
|-------|-------------|---------|
| `per_alpha` | Exposant de priorite (0=uniforme, 1=full) | 0.6 |
| `per_beta_start` | β initial (IS correction) | 0.4 |
| `per_beta_end` | β final | 1.0 |
| `per_beta_steps` | Steps pour anneal β | = num_episodes |
| `learning_starts` | Warm-up (comme DDQN+ER) | 5000 |

---

## Comparaison des formules de target

| Agent | Formule de target |
|-------|-------------------|
| **DQN** | `r + γ × max_a' Q_target(s', a')` |
| **DDQN** | `r + γ × Q_target(s', argmax_a' Q_online(s', a'))` |
| **DDQN+ER** | Identique a DDQN (seul le timing d'entrainement change) |
| **DDQN+PER** | Identique a DDQN (seul l'echantillonnage et la loss changent) |

---

## Composants partages

### MLP Builder (`training/networks.py`)

```python
build_mlp(input_dim, output_dim, hidden_layers, activation="relu")
```

Exemple : `build_mlp(80, 625, [128, 128])` pour Bobail :

```
Input(80) → Linear(80→128) → ReLU → Linear(128→128) → ReLU → Linear(128→625)
```

> Bobail a un etat de 80 dims : 75 canaux spatiaux binaires + 5 features strategiques (`phase`, `dist_my`, `dist_opp`, `mobilite`, `first_turn`). Voir `docs/encoding.md` pour le detail.

### Replay Buffer (`training/replay_buffer.py`)

| Type | Echantillonnage | Complexite | Utilise par |
|------|----------------|------------|-------------|
| `ReplayBuffer` | Uniforme | O(1) push, O(k) sample | DQN, DDQN, DDQN+ER |
| `PrioritizedReplayBuffer` | Proportionnel aux priorites | O(log N) push/sample | DDQN+PER |
