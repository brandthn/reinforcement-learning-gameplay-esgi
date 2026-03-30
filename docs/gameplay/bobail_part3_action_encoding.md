# Part 3 — Encodage des Actions (Deep-Dive)

> Objectif : maîtriser parfaitement la formule `action = from_cell * 25 + to_cell`,
> comprendre pourquoi 625, comment décoder, et voir des exemples concrets pour chaque phase.

---

## 1. La formule fondamentale

### 1.1 Encodage

```
action = from_cell × 25 + to_cell
```

- `from_cell` : index de la cellule de **départ** (0..24)
- `to_cell` : index de la cellule d'**arrivée** (0..24)

### 1.2 Décodage

```python
from_cell = action // 25    # division entière
to_cell   = action % 25     # modulo (reste)
```

### 1.3 Pourquoi 25 ?

```python
# environments/bobail.py:5-6
BOARD_SIZE = 5
NUM_CELLS = BOARD_SIZE * BOARD_SIZE   # = 25
```

Le plateau a 25 cellules. Chaque action est un couple (départ, arrivée) parmi 25×25 = **625**
combinaisons possibles.

```python
# environments/bobail.py:161-162
def action_space_size(self) -> int:
    return NUM_CELLS * NUM_CELLS   # = 625
```

---

## 2. Pourquoi un espace de 625 alors que peu d'actions sont légales ?

### 2.1 Le problème

À chaque instant, seules ~20-60 actions sont légales sur les 625 possibles. Pourquoi ne pas
utiliser un espace plus petit ?

### 2.2 La raison : compatibilité avec les réseaux de neurones

Les réseaux de neurones (DQN, DDQN...) produisent un vecteur de Q-values de **taille fixe**.
Si l'espace d'action changeait de taille à chaque step, le réseau devrait changer d'architecture
en cours de partie — impossible.

```
Réseau de neurones DQN :
    Input: state[75] → Hidden layers → Output: Q[625]
                                                  │
                                         Q-value pour chaque
                                         action possible
```

### 2.3 Action Masking

Pour empêcher le réseau de choisir une action illégale :

```python
# Pseudo-code d'un agent DQN :
q_values = network(state)                    # vecteur de 625 Q-values
mask = torch.full((625,), float('-inf'))     # tout à -∞
for a in available_actions:
    mask[a] = 0.0                            # seules les actions légales restent
q_values = q_values + mask                   # illégales → -∞
action = q_values.argmax()                   # choisir la meilleure action légale
```

Le `RandomAgent`, lui, ignore complètement l'espace de 625 et pioche directement dans
`available_actions` :

```python
# agents/random_agent.py:15-16
def act(self, state, available_actions, training=False):
    return random.choice(available_actions)   # parmi les ~20-60 actions légales
```

---

## 3. Matrice complète des actions (625)

Chaque cellule de cette matrice correspond à une action :

```
              to_cell →
              0    1    2    3    4    5    6  ...  24
from_cell  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬───┬──────┐
    ↓    0 │  0  │  1  │  2  │  3  │  4  │  5  │  6  │...│  24  │
         1 │ 25  │ 26  │ 27  │ 28  │ 29  │ 30  │ 31  │...│  49  │
         2 │ 50  │ 51  │ 52  │ 53  │ 54  │ 55  │ 56  │...│  74  │
         3 │ 75  │ 76  │ 77  │ 78  │ 79  │ 80  │ 81  │...│  99  │
         4 │100  │101  │102  │103  │104  │105  │106  │...│ 124  │
         5 │125  │126  │127  │128  │129  │130  │131  │...│ 149  │
         . │ .   │ .   │ .   │ .   │ .   │ .   │ .   │...│  .   │
        12 │300  │301  │302  │303  │304  │305  │306  │...│ 324  │
         . │ .   │ .   │ .   │ .   │ .   │ .   │ .   │...│  .   │
        20 │500  │501  │502  │503  │504  │505  │506  │...│ 524  │
        24 │600  │601  │602  │603  │604  │605  │606  │...│ 624  │
            └─────┴─────┴─────┴─────┴─────┴─────┴─────┴───┴──────┘
```

**Lecture** : action 307 = ligne 12, colonne 7 → from=12, to=7.

---

## 4. Encodage dans _bobail_moves()

### 4.1 Code source

```python
# environments/bobail.py:113-124
def _bobail_moves(self) -> list[int]:
    br, bc = _idx_to_rc(self._bobail)              # position du bobail
    occupied = self._pieces[0] | self._pieces[1]    # pions qui bloquent
    moves = []
    for dr, dc in DIRECTIONS:
        nr, nc = br + dr, bc + dc
        if _in_bounds(nr, nc):
            target = _rc_to_idx(nr, nc)
            if target not in occupied:
                moves.append(self._bobail * NUM_CELLS + target)  # ← ENCODAGE
    return moves
```

### 4.2 Trace d'exécution — Bobail en cellule 12

Supposons : Bobail en 12 (row=2, col=2), aucun pion adjacent.

```
Itération 1 : dr=-1, dc=-1 → (1, 1) → idx=6  → libre → action = 12*25 + 6  = 306
Itération 2 : dr=-1, dc=0  → (1, 2) → idx=7  → libre → action = 12*25 + 7  = 307
Itération 3 : dr=-1, dc=1  → (1, 3) → idx=8  → libre → action = 12*25 + 8  = 308
Itération 4 : dr=0,  dc=-1 → (2, 1) → idx=11 → libre → action = 12*25 + 11 = 311
Itération 5 : dr=0,  dc=1  → (2, 3) → idx=13 → libre → action = 12*25 + 13 = 313
Itération 6 : dr=1,  dc=-1 → (3, 1) → idx=16 → libre → action = 12*25 + 16 = 316
Itération 7 : dr=1,  dc=0  → (3, 2) → idx=17 → libre → action = 12*25 + 17 = 317
Itération 8 : dr=1,  dc=1  → (3, 3) → idx=18 → libre → action = 12*25 + 18 = 318
```

**Résultat** : `[306, 307, 308, 311, 313, 316, 317, 318]` — 8 actions légales.

### 4.3 Trace d'exécution — Bobail en coin (cellule 0)

Bobail en 0 (row=0, col=0). Pions du joueur 1 en {1, 2, 3, 4} (occupent la rangée).

```
Itération 1 : dr=-1, dc=-1 → (-1,-1) → HORS LIMITES → skip
Itération 2 : dr=-1, dc=0  → (-1, 0) → HORS LIMITES → skip
Itération 3 : dr=-1, dc=1  → (-1, 1) → HORS LIMITES → skip
Itération 4 : dr=0,  dc=-1 → (0, -1) → HORS LIMITES → skip
Itération 5 : dr=0,  dc=1  → (0,  1) → idx=1 → OCCUPÉ (joueur 1) → skip
Itération 6 : dr=1,  dc=-1 → (1, -1) → HORS LIMITES → skip
Itération 7 : dr=1,  dc=0  → (1,  0) → idx=5 → libre → action = 0*25 + 5 = 5
Itération 8 : dr=1,  dc=1  → (1,  1) → idx=6 → libre → action = 0*25 + 6 = 6
```

**Résultat** : `[5, 6]` — seulement 2 actions légales.

---

## 5. Encodage dans _piece_moves()

### 5.1 Code source

```python
# environments/bobail.py:126-145
def _piece_moves(self) -> list[int]:
    occupied = self._pieces[0] | self._pieces[1] | {self._bobail}  # inclut le bobail !
    moves = []
    for cell in self._pieces[self._current]:     # pour chaque pion du joueur courant
        r, c = _idx_to_rc(cell)
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not _in_bounds(nr, nc):           # direction bloquée immédiatement
                continue
            if _rc_to_idx(nr, nc) in occupied:   # bloqué par pion/bobail
                continue
            # On peut bouger d'au moins 1 case → glisser
            while _in_bounds(nr + dr, nc + dc) and _rc_to_idx(nr + dr, nc + dc) not in occupied:
                nr += dr
                nc += dc
            target = _rc_to_idx(nr, nc)
            moves.append(cell * NUM_CELLS + target)  # ← ENCODAGE
    return moves
```

### 5.2 Trace d'exécution — Pion en cellule 22 vers le Nord

Position initiale. Joueur 0 (current=0), pion en cellule 22 (row=4, col=2).

```
Plateau :    1 1 1 1 1         occupied = {0,1,2,3,4, 20,21,22,23,24, 12}
             . . . . .
             . . B . .
             . . . . .
             0 0 0 0 0

Pion 22 (4,2), direction ↑ (dr=-1, dc=0) :
  Étape 1 : nr=3, nc=2 → idx=17 → pas dans occupied → continue
  While :   nr+dr=2, nc+dc=2 → idx=12 → 12 ∈ occupied (BOBAIL) → STOP
  → target = 17
  → action = 22 * 25 + 17 = 567

Pion 22 (4,2), direction ↗ (dr=-1, dc=1) :
  Étape 1 : nr=3, nc=3 → idx=18 → pas dans occupied → continue
  While :   nr+dr=2, nc+dc=4 → idx=14 → pas dans occupied → nr=2, nc=4
            nr+dr=1, nc+dc=5 → HORS LIMITES → STOP
  → target = 14
  → action = 22 * 25 + 14 = 564

Pion 22 (4,2), direction → (dr=0, dc=1) :
  Étape 1 : nr=4, nc=3 → idx=23 → 23 ∈ occupied (pion allié) → skip

Pion 22 (4,2), direction ← (dr=0, dc=-1) :
  Étape 1 : nr=4, nc=1 → idx=21 → 21 ∈ occupied (pion allié) → skip
```

Et ainsi de suite pour les autres directions.

### 5.3 Visualisation du glissement

```
Pion en 22, direction ↗ :

     1  1  1  1  1
     .  .  .  .  .
     .  .  B  . [●]  ← glisse jusqu'ici (idx=14), bloqué par le bord
     .  .  .  ↗  .
     0  0 [0] 0  0   ← départ (idx=22)

action = 22 × 25 + 14 = 564
décodage : 564 // 25 = 22 (from), 564 % 25 = 14 (to)
```

---

## 6. Vérification croisée : décodage dans step()

Quand `step(action)` est appelé, il décode immédiatement :

```python
# environments/bobail.py:63-65
def step(self, action: int) -> tuple[np.ndarray, float, bool]:
    from_cell = action // NUM_CELLS    # NUM_CELLS = 25
    to_cell = action % NUM_CELLS
```

C'est exactement l'inverse de l'encodage :

| Opération | Phase Bobail | Phase Pion |
|-----------|:---:|:---:|
| `from_cell` | Position actuelle du bobail | Cellule du pion à déplacer |
| `to_cell` | Nouvelle position du bobail | Destination du pion après glissement |

---

## 7. Table de référence rapide

### 7.1 Plages d'actions par cellule de départ

| from_cell | Plage d'actions | Nombre |
|:---------:|:---------------:|:------:|
| 0 | 0 — 24 | 25 |
| 1 | 25 — 49 | 25 |
| 2 | 50 — 74 | 25 |
| ... | ... | 25 |
| 12 | 300 — 324 | 25 |
| ... | ... | 25 |
| 20 | 500 — 524 | 25 |
| 24 | 600 — 624 | 25 |

### 7.2 Exemples de décodage

| Action | from_cell | to_cell | Interprétation |
|:------:|:---------:|:-------:|:---------------|
| 0 | 0 | 0 | cellule 0 → cellule 0 (identité, jamais légal) |
| 5 | 0 | 5 | cellule 0 → cellule 5 |
| 307 | 12 | 7 | cellule 12 → cellule 7 (bobail monte) |
| 505 | 20 | 5 | cellule 20 → cellule 5 (pion glisse au nord) |
| 564 | 22 | 14 | cellule 22 → cellule 14 (pion glisse NE) |
| 567 | 22 | 17 | cellule 22 → cellule 17 (pion monte, bloqué par bobail) |
| 624 | 24 | 24 | cellule 24 → cellule 24 (identité, jamais légal) |

### 7.3 Formule résumée

```
┌─────────────────────────────────────────────────┐
│                                                  │
│   ENCODAGE :  action = from × 25 + to           │
│   DÉCODAGE :  from = action // 25                │
│               to   = action % 25                 │
│                                                  │
│   Espace total : 625 (25 × 25)                  │
│   Actions légales : ~20-60 par step              │
│                                                  │
│   Invariant :  0 ≤ action ≤ 624                  │
│                0 ≤ from ≤ 24                     │
│                0 ≤ to ≤ 24                       │
│                from ≠ to (toujours, en pratique) │
│                                                  │
└─────────────────────────────────────────────────┘
```
