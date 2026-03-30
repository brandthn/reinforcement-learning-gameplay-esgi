# Part 6 — La boucle Self-Play et le Deferred Observe

> Objectif : comprendre pourquoi et comment `SelfPlayTrainer` diffère du `Trainer` classique,
> le mécanisme de deferred observe, et la gestion des rewards dans un jeu à 2 joueurs.

---

## 1. Pourquoi pas le Trainer classique ?

### 1.1 Le problème

Dans un jeu single-player (LineWorld, GridWorld), l'agent agit à **chaque step** :

```
Step 1 : agent.act() → env.step() → agent.observe(s, a, r, s', done)
Step 2 : agent.act() → env.step() → agent.observe(s, a, r, s', done)
Step 3 : ...
```

L'observe est **immédiat** : après chaque `step()`, on donne à l'agent la transition
`(state, action, reward, next_state, done)` et `next_state` est bien l'état que l'agent
reverra quand il devra agir.

Dans un jeu à 2 joueurs, l'agent n'agit **qu'un step sur deux** (environ) :

```
Step 1 : AGENT  act → env.step() → state' est pour l'ADVERSAIRE
Step 2 : ADVERSAIRE act → env.step() → state' est pour l'AGENT ← ce state est le vrai next_state
Step 3 : AGENT  act → ...
```

Le `next_state` immédiat après l'action de l'agent est dans la **perspective de l'adversaire**.
L'agent ne le reverra jamais. Son vrai `next_state` est celui qui arrive quand c'est à nouveau
son tour.

### 1.2 Diagramme du problème

```
                    Trainer classique (FAUX pour 2 joueurs)
                    ──────────────────────────────────────
Step 1: agent.act(s₁) → env.step() → s₂ (perspective adversaire !)
        agent.observe(s₁, a₁, r₁, s₂, done)
                                      ↑
                              MAUVAIS ! s₂ est pour l'adversaire
                              L'agent ne verra jamais ce s₂

                    SelfPlayTrainer (CORRECT)
                    ────────────────────────
Step 1: agent.act(s₁) → env.step()      → sauvegarder (s₁, a₁)
Step 2: opponent.act() → env.step()      → obtenir s₃
Step 3: agent.observe(s₁, a₁, r, s₃, done)  ← s₃ est le vrai next_state
                                                de l'agent
        agent.act(s₃) → env.step()      → sauvegarder (s₃, a₃)
Step 4: opponent.act() → env.step()      → obtenir s₅
Step 5: agent.observe(s₃, a₃, r, s₅, done) ...
```

---

## 2. Le mécanisme du Deferred Observe

### 2.1 Les variables de bookkeeping

```python
# training/self_play.py:100-103
pending_state = None       # l'état que l'agent a vu quand il a agi
pending_action = None      # l'action qu'il a choisie
pending_reward = 0.0       # le reward accumulé depuis son action
```

### 2.2 Quand l'agent agit (player == 0)

```python
# training/self_play.py:117-129
if player == 0:
    # 1) D'abord, livrer la transition PRÉCÉDENTE (si elle existe)
    if pending_state is not None:
        self._agent.observe(
            pending_state,        # état quand l'agent a agi la dernière fois
            pending_action,       # action choisie
            pending_reward,       # reward accumulé
            state,                # état ACTUEL = vrai next_state de l'agent
            False,                # pas terminal (sinon on serait sorti)
        )
        pending_state = None
        pending_reward = 0.0

    # 2) Puis, choisir une nouvelle action
    action = self._agent.act(state, available, training=True)

    # 3) Sauvegarder pour la prochaine livraison
    pending_state = state
    pending_action = action
```

### 2.3 Quand l'adversaire agit (player == 1)

```python
# training/self_play.py:130-131
else:
    action = self._opponent.act(state, available, training=False)
```

Pas d'observe, pas de pending. L'adversaire joue et c'est tout.

### 2.4 Après env.step()

```python
# training/self_play.py:133-138
next_state, reward, done = self._env.step(action)
steps += 1

if player == 0:
    pending_reward += reward    # accumule le reward de l'agent
    agent_reward += reward
```

### 2.5 À la fin de la partie (terminal)

```python
# training/self_play.py:140-151
if done and pending_state is not None:
    if player != 0:
        # L'adversaire a terminé la partie
        # reward est de la perspective de l'adversaire → inverser pour l'agent
        pending_reward -= reward
        agent_reward -= reward
    self._agent.observe(
        pending_state, pending_action, pending_reward,
        next_state, True,        # done = True
    )
    pending_state = None
```

---

## 3. Diagramme de séquence complet

```
Temps  Joueur  Phase    Action dans _run_episode()
─────  ──────  ───────  ─────────────────────────────────────────────────
  │
  │    reset()  →  state = s₀ (perspective J0)
  │              pending = None
  │
  ▼    J0      PIECE    pending=None → pas d'observe
  1                     action = agent.act(s₀, avail, True)  → a₁
                        pending = (s₀, a₁, 0.0)
                        env.step(a₁) → s₁, r=0, done=F
                        pending_reward += 0 → 0.0
                        state = s₁ (perspective J1)
  │
  ▼    J1      BOBAIL   action = opponent.act(s₁, avail, False) → a₂
  2                     env.step(a₂) → s₂, r=0, done=F
                        state = s₂ (perspective J1)
  │
  ▼    J1      PIECE    action = opponent.act(s₂, avail, False) → a₃
  3                     env.step(a₃) → s₃, r=0, done=F
                        state = s₃ (perspective J0)
  │
  ▼    J0      BOBAIL   pending=(s₀, a₁, 0.0) ≠ None
  4                     ┌──────────────────────────────────────┐
                        │ agent.observe(s₀, a₁, 0.0, s₃, F)  │  ← DEFERRED !
                        │ La transition (s₀→s₃) saute les     │
                        │ 2 steps de l'adversaire              │
                        └──────────────────────────────────────┘
                        action = agent.act(s₃, avail, True)  → a₄
                        pending = (s₃, a₄, 0.0)
                        env.step(a₄) → s₄, r=0, done=F
                        pending_reward += 0 → 0.0
                        state = s₄ (perspective J0)
  │
  ▼    J0      PIECE    pending=(s₃, a₄, 0.0) ≠ None
  5                     ┌──────────────────────────────────────┐
                        │ agent.observe(s₃, a₄, 0.0, s₄, F)  │
                        └──────────────────────────────────────┘
                        action = agent.act(s₄, avail, True)  → a₅
                        pending = (s₄, a₅, 0.0)
                        env.step(a₅) → s₅, r=0, done=F
                        state = s₅ (perspective J1)
  │
  ...  (la partie continue)
  │
  ▼    J0      BOBAIL   L'agent déplace le bobail sur sa home_row
  N                     env.step(aₙ) → sₙ, r=1.0, done=True
                        pending_reward += 1.0
                        ┌──────────────────────────────────────────────┐
                        │ Terminal flush :                              │
                        │ agent.observe(sₚ, aₚ, 1.0, sₙ, True)      │
                        └──────────────────────────────────────────────┘
                        BREAK
```

---

## 4. La gestion du reward terminal

### 4.1 Quand l'agent (J0) termine la partie

```python
# L'agent vient de jouer et a gagné
pending_reward += reward    # += 1.0
# player == 0, pas d'inversion
agent.observe(pending_state, pending_action, pending_reward, next_state, True)
```

Le reward est directement positif : **+1.0** = victoire.

### 4.2 Quand l'adversaire (J1) termine la partie

```python
# L'adversaire vient de jouer et reward=1.0 (de SA perspective = il gagne)
if player != 0:
    pending_reward -= reward    # -= 1.0 → reward agent = -1.0
    agent_reward -= reward
```

Si l'adversaire gagne, le reward de son `step()` est **+1.0** de sa perspective.
Mais pour l'agent, c'est une **défaite** : on soustrait → `pending_reward = -1.0`.

### 4.3 Tableau récapitulatif des rewards

| Qui termine ? | reward de step() | Pour l'agent (J0) | Explication |
|:---:|:---:|:---:|:---|
| Agent (J0) gagne | 1.0 | +1.0 | Directement ajouté |
| Adversaire (J1) gagne | 1.0 | -1.0 | Inversé (`-= reward`) |
| Pas terminal | 0.0 | 0.0 | Accumulé, reste 0 |

---

## 5. Le RandomAgent et l'observe

Pour le `RandomAgent`, `observe()` et `end_episode()` sont des **no-ops** (hérités de `Agent`) :

```python
# agents/base.py:18-23
def observe(self, state, action, reward, next_state, done):
    """Called after env.step() during training. Default: no-op."""
    pass

def end_episode(self):
    """Called at end of each training episode. Default: no-op."""
    pass
```

Le deferred observe fonctionne quand même (le code l'appelle), mais l'information est
simplement **ignorée**. Cela n'a d'importance que pour les agents qui **apprennent**
(TabularQ, DQN, DDQN...).

---

## 6. La boucle d'entraînement complète

### 6.1 `SelfPlayTrainer.train()`

```python
# training/self_play.py:42-94
def train(self, results_dir: str) -> dict:
    # 1. Créer le dossier et les fichiers CSV
    os.makedirs(results_dir, exist_ok=True)

    # 2. Écrire la config
    yaml.dump(self._config, f)

    # 3. Initialiser les CSV (headers)
    #    training_curve.csv : [episode, reward, steps]
    #    metrics.csv : [checkpoint, mean_reward, std_reward, ...]

    # 4. Boucle principale
    for ep in range(1, self._num_episodes + 1):     # 1..1000

        # 4a. Jouer un épisode
        reward, steps = self._run_episode()
        # → écrit dans training_curve.csv

        # 4b. Évaluation aux checkpoints
        if ep in self._checkpoints:                  # {100, 500, 1000}
            eval_result = self._evaluator.evaluate(
                self._agent, self._env, self._eval_games,  # 50 parties
                opponent=self._opponent,
            )
            # → écrit dans metrics.csv
            # → sauvegarde model_<ep>.pt

    return all_metrics
```

### 6.2 L'évaluation aux checkpoints

```python
# evaluation/evaluator.py:72-103
def _play_adversarial(self, agent, opponent, env, max_steps=10_000):
    state = env.reset()
    agent_reward = 0.0

    for _ in range(max_steps):
        player = env.current_player()
        available = env.available_actions()

        if player == 0:
            action = agent.act(state, available, training=False)   # ← training=False !
        else:
            action = opponent.act(state, available, training=False)

        next_state, reward, done = env.step(action)

        if player == 0:
            agent_reward += reward
        elif done:
            agent_reward -= reward     # adversaire gagne → -reward

        state = next_state
        if done:
            break

    return agent_reward, steps, action_times
```

**Différences avec _run_episode()** :
1. `training=False` → pas d'exploration (mais pour RandomAgent, aucune différence)
2. Pas de deferred observe → pas d'apprentissage pendant l'évaluation
3. Mesure du temps par action (`time.perf_counter()`)

---

## 7. Flux de données complet — Du config au CSV

```
configs/random/bobail.yaml
        │
        ▼
┌──────────────────────────────────────────────────┐
│           SelfPlayTrainer.train()                 │
│                                                    │
│   Pour chaque épisode (1..1000) :                 │
│   ├─ _run_episode()                               │
│   │   ├─ env.reset()                              │
│   │   ├─ [boucle agent/opponent actions]          │
│   │   └─ retourne (reward, steps)                 │
│   │                                               │
│   ├─ Écrit → training_curve.csv                   │
│   │   episode | reward | steps                    │
│   │   1       | 1.0    | 47                       │
│   │   2       | -1.0   | 52                       │
│   │   ...                                         │
│   │                                               │
│   └─ Si checkpoint (100, 500, 1000) :             │
│       ├─ Evaluator.evaluate() × 50 parties        │
│       ├─ Écrit → metrics.csv                      │
│       │   checkpoint | mean_reward | std_reward    │
│       │   100        | 0.02        | 0.98          │
│       │   500        | -0.04       | 1.01          │
│       │   ...                                     │
│       └─ Sauvegarde → model_<ep>.pt               │
│           (vide pour RandomAgent)                  │
└──────────────────────────────────────────────────┘
```

---

## 8. Résumé — Points essentiels

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  1. Bobail est ADVERSARIAL → utilise SelfPlayTrainer            │
│                                                                  │
│  2. L'agent (J0) ne voit un observe que quand c'est à nouveau   │
│     SON TOUR → deferred observe                                  │
│                                                                  │
│  3. Le next_state dans observe() est toujours du point de vue   │
│     de l'agent (J0), jamais de l'adversaire                      │
│                                                                  │
│  4. Rewards :                                                    │
│     • Agent gagne   → +1.0                                       │
│     • Adversaire gagne → -1.0 (inversion dans _run_episode)    │
│     • En cours      → 0.0                                        │
│                                                                  │
│  5. RandomAgent ignore tout (observe = no-op), mais la          │
│     mécanique est là pour les agents qui apprennent             │
│                                                                  │
│  6. Évaluation aux checkpoints : 50 parties, training=False,    │
│     pas de deferred observe, mesure du temps par action          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```
