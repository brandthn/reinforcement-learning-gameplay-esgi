# Part 5 — Walkthrough complet d'un épisode (RandomAgent vs RandomAgent)

> Objectif : suivre pas à pas un épisode complet en traçant chaque appel de fonction,
> chaque valeur intermédiaire, et chaque changement d'état du plateau.

---

## 1. Initialisation de l'épisode

L'épisode commence dans `SelfPlayTrainer._run_episode()` :

```python
# training/self_play.py:96-106
def _run_episode(self) -> tuple[float, int]:
    state = self._env.reset()          # ← APPEL 1
    done = False
    pending_state = None               # pour le deferred observe
    pending_action = None
    pending_reward = 0.0
    agent_reward = 0.0
    steps = 0
```

### APPEL 1 : `BobailEnv.reset()`

```python
# environments/bobail.py:52-61
def reset(self) -> np.ndarray:
    self._pieces[0] = {20, 21, 22, 23, 24}    # Joueur 0 en bas
    self._pieces[1] = {0, 1, 2, 3, 4}         # Joueur 1 en haut
    self._bobail = 12                          # Centre (2,2)
    self._current = 0                          # Joueur 0 commence
    self._phase = PHASE_PIECE                  # Skip bobail au 1er tour !
    self._done = False
    self._turn_number = 0
    self._first_turn = True
    return self.state_description()            # → float32[80]
```

**Plateau après reset :**
```
1 1 1 1 1       indices: {0, 1, 2, 3, 4}
. . . . .
. . B . .       bobail: 12
. . . . .
0 0 0 0 0       indices: {20, 21, 22, 23, 24}
```

**État retourné** (current=0, donc canal 0 = joueur 0) :
```
state = [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 1,1,1,1,1,  ← canal 0 (mes pions)
         1,1,1,1,1, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0,  ← canal 1 (adversaire)
         0,0,0,0,0, 0,0,0,0,0, 0,0,1,0,0, 0,0,0,0,0, 0,0,0,0,0]  ← canal 2 (bobail=12)
```

---

## 2. Step 1 — Joueur 0, Phase PIECE (premier tour, bobail skip)

```python
# training/self_play.py:108-134 (boucle principale)
player = self._env.current_player()        # → 0
available = self._env.available_actions()   # ← APPEL 2
```

### APPEL 2 : `BobailEnv.available_actions()`

```python
# environments/bobail.py:105-111
def available_actions(self) -> list[int]:
    if self._phase == PHASE_BOBAIL:
        return self._bobail_moves()
    return self._piece_moves()       # ← phase = PIECE → ici
```

→ Appelle `_piece_moves()` pour le joueur 0.

**Calcul** : Les 5 pions en {20,21,22,23,24} cherchent dans les 8 directions.
`occupied = {0,1,2,3,4} ∪ {20,21,22,23,24} ∪ {12}`

Pour le pion 20 (row=4, col=0) :
```
↑  (dr=-1,dc=0) : 15 libre → while: 10 libre, 5 libre, 0 occupé → stop → target=5  → action=505
↗ (dr=-1,dc=1) : 16 libre → while: 12 occupé(bobail) → stop → target=16 → action=516
→  (dr=0,dc=1)  : 21 occupé(allié) → skip
↓  (dr=1,dc=0)  : hors limites → skip
↘ (dr=1,dc=1)  : hors limites → skip
↙ (dr=1,dc=-1) : hors limites → skip
←  (dr=0,dc=-1) : hors limites → skip
↖ (dr=-1,dc=-1): hors limites → skip
```
Pion 20 → actions [505, 516]

*(Calculs similaires pour les pions 21, 22, 23, 24)*

Supposons que `available_actions()` retourne une liste de ~14 actions légales.

### Choix de l'agent

```python
# training/self_play.py:117-129
if player == 0:
    if pending_state is not None:       # None au premier step
        ...
    action = self._agent.act(state, available, training=True)
    pending_state = state               # sauvegarde pour deferred observe
    pending_action = action
```

→ `RandomAgent.act(state, available, training=True)` → `random.choice(available)`

**Supposons** que l'agent choisit action **505** (pion 20 → cellule 5, glissement vers le nord).

### APPEL 3 : `BobailEnv.step(505)`

```python
# environments/bobail.py:63-103
def step(self, action: int):
    from_cell = 505 // 25    # = 20
    to_cell = 505 % 25       # = 5
    # phase == PHASE_PIECE, donc on va à la ligne 81

    # PHASE_PIECE
    self._pieces[0].discard(20)     # retire pion de cellule 20
    self._pieces[0].add(5)          # place pion en cellule 5

    self._first_turn = False         # plus jamais True

    # Switch joueur
    self._current = 1                # au tour du joueur 1
    self._turn_number = 1

    # Prochain tour commence par bobail
    self._phase = PHASE_BOBAIL

    # Vérifier si joueur 1 peut bouger le bobail
    bobail_moves = self._bobail_moves()  # → liste non vide → continue
    return self.state_description(), 0.0, False
```

**Plateau après step 1 :**
```
1 1 1 1 1       Joueur 1: {0, 1, 2, 3, 4}
0 . . . .       ← Pion joueur 0 a glissé de 20 à 5
. . B . .       Bobail: 12
. . . . .
. 0 0 0 0       Joueur 0: {5, 21, 22, 23, 24}  (20 retiré, 5 ajouté)
```

**État retourné** (current=1 maintenant → canal 0 = joueur 1) :
```
Canal 0 (joueur 1) : pions en {0,1,2,3,4} → state[0..4] = 1.0
Canal 1 (joueur 0) : pions en {5,21,22,23,24} → state[30]=1.0, state[46..49]=1.0
Canal 2 (bobail)   : state[62] = 1.0
```

**Retour** : `(state, reward=0.0, done=False)` — la partie continue.

```python
# training/self_play.py:133-138
steps += 1                          # steps = 1
# player == 0, donc :
pending_reward += 0.0               # pending_reward = 0.0
agent_reward += 0.0                 # agent_reward = 0.0
state = next_state                  # état mis à jour
```

---

## 3. Step 2 — Joueur 1, Phase BOBAIL

```python
player = self._env.current_player()        # → 1
available = self._env.available_actions()   # → _bobail_moves()
```

### APPEL 4 : `BobailEnv._bobail_moves()`

Bobail en 12 (row=2, col=2). `occupied = {0,1,2,3,4} ∪ {5,21,22,23,24}`.

```
Cell 5 est maintenant occupée (pion joueur 0 y a glissé) !

↖ idx=6  → libre → action = 12*25 + 6  = 306
↑  idx=7  → libre → action = 12*25 + 7  = 307
↗ idx=8  → libre → action = 12*25 + 8  = 308
←  idx=11 → libre → action = 12*25 + 11 = 311
→  idx=13 → libre → action = 12*25 + 13 = 313
↙ idx=16 → libre → action = 12*25 + 16 = 316
↓  idx=17 → libre → action = 12*25 + 17 = 317
↘ idx=18 → libre → action = 12*25 + 18 = 318
```

**Résultat** : `[306, 307, 308, 311, 313, 316, 317, 318]` — 8 mouvements légaux.

### Choix de l'adversaire

```python
# training/self_play.py:130-131
else:  # player == 1
    action = self._opponent.act(state, available, training=False)
```

→ `RandomAgent.act()` → `random.choice([306, 307, 308, 311, 313, 316, 317, 318])`

**Supposons** action **317** (bobail de 12 vers 17, direction ↓).

### APPEL 5 : `BobailEnv.step(317)`

```python
from_cell = 317 // 25    # = 12
to_cell = 317 % 25       # = 17
# phase == PHASE_BOBAIL

self._bobail = 17                    # bobail passe de 12 à 17
self._phase = PHASE_PIECE            # prochaine phase = pion

# Vérifier si bobail atteint home_row du joueur 1
br, _ = _idx_to_rc(17)              # → (3, 2)
home_row = 0   # (joueur 1, home = row 0)
# 3 != 0 → pas de victoire

return self.state_description(), 0.0, False
```

**Plateau après step 2 :**
```
1 1 1 1 1
0 . . . .
. . . . .       ← bobail a quitté la cellule 12
. . B . .       ← bobail maintenant en 17 (row=3, col=2)
. 0 0 0 0
```

**Note** : `current` est toujours 1 (pas de changement de joueur en phase bobail).

```python
# training/self_play.py:136-138
# player == 1, pas de pending_reward modif
steps += 1     # steps = 2
state = next_state
```

---

## 4. Step 3 — Joueur 1, Phase PIECE

```python
player = self._env.current_player()        # → 1 (toujours)
available = self._env.available_actions()   # → _piece_moves() pour joueur 1
```

Le joueur 1 a ses pions en {0, 1, 2, 3, 4}. Chaque pion cherche à glisser.
`occupied = {0,1,2,3,4} ∪ {5,21,22,23,24} ∪ {17}`

**Supposons** que le RandomAgent choisit action **100** : pion 4 (row=0, col=4) → cellule 0... Non, vérifions : `100 // 25 = 4`, `100 % 25 = 0`.

Pion 4 (row=0, col=4) vers cellule 0 (row=0, col=0) ? Direction (0,-1) = ←.
- (0,3)=idx 3 → occupé(allié) → skip. Pas valide.

Prenons plutôt action **119** : `119 // 25 = 4`, `119 % 25 = 19`. Pion 4 (0,4) → cellule 19 (3,4).
Direction ↓ (dr=1, dc=0) : (1,4)=9 libre, (2,4)=14 libre, (3,4)=19 libre, (4,4)=24 occupé → stop.
Action 119 est légale.

### APPEL 6 : `BobailEnv.step(119)`

```python
from_cell = 119 // 25    # = 4
to_cell = 119 % 25       # = 19
# phase == PHASE_PIECE

self._pieces[1].discard(4)      # retire pion de cellule 4
self._pieces[1].add(19)         # place pion en cellule 19

# Switch joueur
self._current = 0               # au tour du joueur 0
self._turn_number = 2
self._phase = PHASE_BOBAIL      # prochain tour = bobail

# Vérifier bobail bloqué ? _bobail_moves() non vide → continue
return self.state_description(), 0.0, False
```

**Plateau après step 3 :**
```
1 1 1 1 .       Joueur 1: {0, 1, 2, 3, 19} (4 retiré, 19 ajouté)
0 . . . .       Joueur 0: {5, 21, 22, 23, 24}
. . . . .
. . B . 1       ← pion joueur 1 a glissé de 4 à 19
. 0 0 0 0
```

---

## 5. Step 4 — Joueur 0, Phase BOBAIL

C'est maintenant au joueur 0 de déplacer le bobail.

```python
player = self._env.current_player()        # → 0
available = self._env.available_actions()   # → _bobail_moves()
```

Le bobail est en 17 (row=3, col=2).
`occupied = {0,1,2,3,19} ∪ {5,21,22,23,24}` (sans le bobail lui-même).

```
↖ (2,1)=11 → libre → action = 17*25 + 11 = 436
↑  (2,2)=12 → libre → action = 17*25 + 12 = 437
↗ (2,3)=13 → libre → action = 17*25 + 13 = 438
←  (3,1)=16 → libre → action = 17*25 + 16 = 441
→  (3,3)=18 → libre → action = 17*25 + 18 = 443
↙ (4,1)=21 → occupé(J0) → skip
↓  (4,2)=22 → occupé(J0) → skip
↘ (4,3)=23 → occupé(J0) → skip
```

**Résultat** : `[436, 437, 438, 441, 443]` — 5 actions légales (3 cases bloquées par les pions du joueur 0).

### Deferred observe

```python
# training/self_play.py:117-129
if player == 0:
    # pending_state est non-None (sauvegardé au step 1)
    if pending_state is not None:
        self._agent.observe(
            pending_state,      # état vu au step 1
            pending_action,     # action 505
            pending_reward,     # 0.0
            state,              # état actuel (perspective joueur 0)
            False,              # pas terminal
        )
        pending_state = None
        pending_reward = 0.0

    action = self._agent.act(state, available, training=True)
    pending_state = state
    pending_action = action
```

> **Point clé** : c'est ici que le deferred observe se produit. L'agent apprend
> la transition (s_step1, a_step1, r=0, s_step4, done=False).
> Voir Part 6 pour les détails.

**Supposons** action **437** (bobail de 17 vers 12, direction ↑).

### APPEL 7 : `BobailEnv.step(437)`

```python
from_cell = 437 // 25    # = 17
to_cell = 437 % 25       # = 12
# phase == PHASE_BOBAIL

self._bobail = 12
self._phase = PHASE_PIECE

# home_row joueur 0 = row 4. Bobail en row 2. 2 != 4 → continue
return self.state_description(), 0.0, False
```

**Plateau après step 4 :**
```
1 1 1 1 .
0 . . . .
. . B . .       ← bobail revenu en 12
. . . . 1
. 0 0 0 0
```

---

## 6. Steps suivants — Le pattern se répète

La boucle continue avec le même pattern :

```
Step 5 : Joueur 0, Phase PIECE → glisse un pion → switch à joueur 1, phase → BOBAIL
Step 6 : Joueur 1, Phase BOBAIL → déplace bobail → phase → PIECE
Step 7 : Joueur 1, Phase PIECE → glisse un pion → switch à joueur 0, phase → BOBAIL
Step 8 : Joueur 0, Phase BOBAIL → déplace bobail → phase → PIECE
Step 9 : Joueur 0, Phase PIECE → ...
```

### Pattern de la boucle

```
┌─────────────────────────────────────────────────────┐
│  Step N   │ Joueur │ Phase  │ Action          │ Qui │
├───────────┼────────┼────────┼─────────────────┼─────┤
│  1        │ 0      │ PIECE  │ glisse pion     │ agt │  ← exception 1er tour
│  2        │ 1      │ BOBAIL │ bouge bobail    │ opp │
│  3        │ 1      │ PIECE  │ glisse pion     │ opp │
│  4        │ 0      │ BOBAIL │ bouge bobail    │ agt │
│  5        │ 0      │ PIECE  │ glisse pion     │ agt │
│  6        │ 1      │ BOBAIL │ bouge bobail    │ opp │
│  7        │ 1      │ PIECE  │ glisse pion     │ opp │
│  8        │ 0      │ BOBAIL │ bouge bobail    │ agt │
│  9        │ 0      │ PIECE  │ glisse pion     │ agt │
│  ...      │ ...    │ ...    │ ...             │ ... │
└─────────────────────────────────────────────────────┘

Après step 1 : 2 steps par tour (bobail + pion), en alternance J1/J0
```

---

## 7. Fin de partie — Exemple de victoire

Supposons qu'à un moment donné, le joueur 0 déplace le bobail sur **row 4** :

```
Plateau hypothétique :
. 1 1 . .
0 . . . .
. . . . .
. 1 . 1 1
0 0 B 0 0       ← Bobail en 22 (row=4, col=2) — HOME ROW du joueur 0 !
```

### APPEL : `BobailEnv.step(action)`

```python
# Phase BOBAIL
self._bobail = 22
self._phase = PHASE_PIECE

br, _ = _idx_to_rc(22)          # → (4, 2)
home_row = 4   # (joueur 0)
# 4 == 4 → VICTOIRE !

self._done = True
self._current = 1               # convention: current = perdant
return self.state_description(), 1.0, True
```

### Retour dans _run_episode()

```python
next_state, reward, done = self._env.step(action)
# reward = 1.0, done = True
steps += 1

# player == 0
pending_reward += 1.0
agent_reward += 1.0

# Terminal: flush pending transition
if done and pending_state is not None:
    # player == 0, donc on flush directement
    self._agent.observe(
        pending_state, pending_action, pending_reward,
        next_state, True,        # done = True
    )
    pending_state = None

# Sortie de la boucle (done = True)
self._agent.end_episode()
return agent_reward, steps       # ex: (1.0, 47)
```

---

## 8. Fin de partie — Exemple de blocage du Bobail

Si après le step d'un pion, le bobail est entièrement encerclé :

```
Plateau hypothétique :
. . 1 . .
0 0 . . .
. 0 B 1 .       ← Bobail en 12, entouré de pions
. . 1 . .
. . . . 1

Cellules adjacentes au bobail (12) :
  6  → occupé (J0)    7  → libre?    8  → ?
  11 → occupé (J0)               13 → occupé (J1)
  16 → ?              17 → occupé (J1)   18 → ?
```

Si toutes les 8 cases adjacentes sont occupées :

```python
# environments/bobail.py:96-101
if not self._bobail_moves():     # retourne [] → not [] = True
    self._done = True
    # current = l'adversaire qui ne peut pas bouger → il PERD
    return self.state_description(), 1.0, True
```

---

## 9. Résumé du flux d'un épisode

```
env.reset()
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    BOUCLE PRINCIPALE                         │
│                                                              │
│  ┌──── player = env.current_player()                        │
│  │     available = env.available_actions()                    │
│  │                                                           │
│  │     Si player == 0 (agent) :                              │
│  │       ├─ deferred observe (si pending)                    │
│  │       ├─ action = agent.act(state, available, True)       │
│  │       └─ sauvegarde pending_state/action                  │
│  │                                                           │
│  │     Si player == 1 (opponent) :                           │
│  │       └─ action = opponent.act(state, available, False)   │
│  │                                                           │
│  │     next_state, reward, done = env.step(action)           │
│  │     steps += 1                                            │
│  │                                                           │
│  │     Si done : flush pending + break                       │
│  │     Sinon   : state = next_state, continuer               │
│  └────────────────────────────────────────────────────────── │
│                                                              │
│  agent.end_episode()                                         │
│  return (agent_reward, steps)                                │
└─────────────────────────────────────────────────────────────┘
```
