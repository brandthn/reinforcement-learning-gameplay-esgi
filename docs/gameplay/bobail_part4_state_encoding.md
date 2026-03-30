# Part 4 — Encodage de l'État (State Encoding)

> Objectif : comprendre le vecteur de 75 floats retourné par `state_description()`,
> ses 3 canaux, la perspective du joueur courant, et comment il change à chaque step.

---

## 1. Structure du vecteur d'état

`state_description()` retourne un `np.ndarray` de shape `(75,)` et dtype `float32`.

Ce vecteur est la **concaténation de 3 canaux** de 25 éléments chacun :

```
state[75] = [ canal_mes_pions[25] | canal_pions_adverses[25] | canal_bobail[25] ]
              indices 0..24           indices 25..49             indices 50..74
```

```python
# environments/bobail.py:147-159
def state_description(self) -> np.ndarray:
    my_pieces = np.zeros(NUM_CELLS, dtype=np.float32)    # canal 0 : mes pions
    opp_pieces = np.zeros(NUM_CELLS, dtype=np.float32)   # canal 1 : pions adverses
    bobail = np.zeros(NUM_CELLS, dtype=np.float32)       # canal 2 : bobail

    for idx in self._pieces[self._current]:              # pions du joueur COURANT
        my_pieces[idx] = 1.0
    for idx in self._pieces[1 - self._current]:          # pions de l'ADVERSAIRE
        opp_pieces[idx] = 1.0
    bobail[self._bobail] = 1.0                           # position du bobail

    return np.concatenate([my_pieces, opp_pieces, bobail])
```

```python
# environments/bobail.py:164-165
def state_space_size(self) -> int:
    return NUM_CELLS * 3   # = 75
```

---

## 2. Les 3 canaux en détail

### 2.1 Canal 0 — Mes pions (indices 0..24)

```
state[i] = 1.0  si le joueur COURANT a un pion sur la cellule i
state[i] = 0.0  sinon
```

### 2.2 Canal 1 — Pions adverses (indices 25..49)

```
state[25 + i] = 1.0  si l'ADVERSAIRE a un pion sur la cellule i
state[25 + i] = 0.0  sinon
```

### 2.3 Canal 2 — Bobail (indices 50..74)

```
state[50 + i] = 1.0  si le bobail est sur la cellule i
state[50 + i] = 0.0  sinon
```

> Le canal bobail est un **one-hot** : exactement 1 valeur à 1.0, les 24 autres à 0.0.

---

## 3. La perspective du joueur courant

**Point crucial** : l'état est TOUJOURS encodé du point de vue du joueur courant
(`self._current`). Canal 0 = "mes" pions, Canal 1 = pions de "l'autre".

### 3.1 Exemple — État initial (current = 0)

```
Plateau :    1 1 1 1 1     Joueur 0 = pions en {20,21,22,23,24}
             . . . . .     Joueur 1 = pions en {0,1,2,3,4}
             . . B . .     Bobail en 12
             . . . . .     current_player = 0
             0 0 0 0 0
```

```
Canal 0 (Mes pions = Joueur 0) :
  idx:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
                                                                       ↑            ↑
                                                                   row 4, col 0-4

Canal 1 (Pions adverses = Joueur 1) :
  idx: 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
       [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ↑            ↑
    row 0, col 0-4

Canal 2 (Bobail) :
  idx: 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                              ↑
                                          idx 62 = cellule 12 (le bobail)
```

### 3.2 Même plateau, mais vu par le Joueur 1 (current = 1)

Si c'est au tour du joueur 1 (même positions physiques), les canaux 0 et 1 **s'inversent** :

```
Canal 0 (Mes pions = Joueur 1) :
  idx:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
       [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ↑            ↑
    MES pions (joueur 1) sont en haut

Canal 1 (Pions adverses = Joueur 0) :
  idx: 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
                                                                       ↑            ↑
    Pions de l'ADVERSAIRE (joueur 0) sont en bas

Canal 2 (Bobail) : identique — le bobail ne change pas de perspective
```

### 3.3 Pourquoi cette convention ?

```
┌─────────────────────────────────────────────────────────────┐
│  L'agent voit TOUJOURS :                                     │
│    Canal 0 = "mes pions"                                     │
│    Canal 1 = "pions ennemis"                                 │
│    Canal 2 = "bobail"                                        │
│                                                               │
│  → L'agent n'a PAS besoin de savoir s'il est joueur 0 ou 1  │
│  → Il apprend UNE SEULE politique qui fonctionne des 2 côtés│
│  → Pas besoin d'encoder le numéro du joueur dans l'état      │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Mapping index d'état → cellule du plateau

Pour convertir un index du vecteur d'état en cellule du plateau :

```
Si state[i] est dans le canal 0 (i < 25) :
    cellule = i
    signification = "mon pion en cellule i"

Si state[i] est dans le canal 1 (25 ≤ i < 50) :
    cellule = i - 25
    signification = "pion adverse en cellule (i-25)"

Si state[i] est dans le canal 2 (50 ≤ i < 75) :
    cellule = i - 50
    signification = "bobail en cellule (i-50)"
```

### Tableau de correspondance

| Index état | Canal | Cellule plateau | Exemple (état initial, current=0) |
|:---:|:---:|:---:|:---|
| 0 | 0 (moi) | 0 | 0.0 — pas de pion à moi en (0,0) |
| 20 | 0 (moi) | 20 | 1.0 — j'ai un pion en (4,0) |
| 25 | 1 (adv) | 0 | 1.0 — adversaire a un pion en (0,0) |
| 44 | 1 (adv) | 19 | 0.0 — pas de pion adverse en (3,4) |
| 62 | 2 (bob) | 12 | 1.0 — bobail en (2,2) |
| 70 | 2 (bob) | 20 | 0.0 — bobail pas en (4,0) |

---

## 5. Propriétés mathématiques du vecteur d'état

```
┌─────────────────────────────────────────────────────┐
│  Propriétés invariantes :                            │
│                                                       │
│  • sum(state[0:25])  = 5   (toujours 5 pions à moi) │
│  • sum(state[25:50]) = 5   (toujours 5 pions adv.)  │
│  • sum(state[50:75]) = 1   (toujours 1 bobail)      │
│  • sum(state)        = 11  (toujours)                │
│                                                       │
│  • state[i] ∈ {0.0, 1.0}  (binaire)                 │
│  • Pas de chevauchement entre canaux pour une        │
│    même cellule (un pion et le bobail ne peuvent     │
│    pas être sur la même case)                        │
└─────────────────────────────────────────────────────┘
```

---

## 6. Quand l'état change-t-il ?

`state_description()` est appelé par `step()` à chaque retour. L'état change **après chaque
action** (bobail OU pion) :

```
1. step() déplace bobail ou pion
2. (éventuellement) switch joueur courant
3. return self.state_description()  ← recalcule l'état depuis zéro
```

### 6.1 Changement après phase bobail

Le bobail bouge, mais le joueur courant **ne change pas** (il doit encore jouer son pion).
→ Les canaux 0 et 1 restent dans la même perspective.
→ Seul le canal 2 (bobail) change.

### 6.2 Changement après phase pion

Le pion bouge, puis le joueur courant **change** (`self._current = 1 - self._current`).
→ Les canaux 0 et 1 **s'inversent** (car la perspective change).
→ Le canal 0 (anciens pions adverses) devient canal 0 (mes pions), etc.
→ Le canal 2 peut aussi avoir changé si un pion a bougé.

```
Exemple : Joueur 0 déplace un pion, puis current passe à 1

AVANT step (current=0) :
  Canal 0 = pions joueur 0
  Canal 1 = pions joueur 1

APRÈS step (current=1) :
  Canal 0 = pions joueur 1    ← INVERSÉ
  Canal 1 = pions joueur 0    ← INVERSÉ
```

---

## 7. Visualisation — Reconstruire le plateau depuis l'état

```python
def state_to_board(state: np.ndarray, current_player: int) -> str:
    """Reconstruit le plateau depuis le vecteur d'état."""
    my = state[0:25]
    opp = state[25:50]
    bob = state[50:75]

    lines = []
    for r in range(5):
        row = []
        for c in range(5):
            idx = r * 5 + c
            if bob[idx] == 1.0:
                row.append('B')
            elif my[idx] == 1.0:
                row.append(str(current_player))      # 'mes' pions
            elif opp[idx] == 1.0:
                row.append(str(1 - current_player))  # pions adverses
            else:
                row.append('.')
        lines.append(' '.join(row))
    return '\n'.join(lines)
```

---

## 8. Résumé visuel

```
                        state[75]
    ┌──────────────────────────────────────────────────┐
    │  Canal 0: Mes pions   │  Canal 1: Pions adv.   │  Canal 2: Bobail     │
    │  state[0..24]         │  state[25..49]          │  state[50..74]       │
    │  5 × 1.0, 20 × 0.0   │  5 × 1.0, 20 × 0.0    │  1 × 1.0, 24 × 0.0  │
    └───────────┬───────────┴───────────┬─────────────┴──────────┬───────────┘
                │                       │                         │
                ▼                       ▼                         ▼
    ┌───────────────────┐  ┌────────────────────┐   ┌────────────────────┐
    │  0  1  2  3  4    │  │  0  1  2  3  4     │   │  0  1  2  3  4     │
    │  5  6  7  8  9    │  │  5  6  7  8  9     │   │  5  6  7  8  9     │
    │ 10 11 12 13 14    │  │ 10 11 12 13 14     │   │ 10 11 12 13 14     │
    │ 15 16 17 18 19    │  │ 15 16 17 18 19     │   │ 15 16 17 18 19     │
    │ 20 21 22 23 24    │  │ 20 21 22 23 24     │   │ 20 21 22 23 24     │
    └───────────────────┘  └────────────────────┘   └────────────────────┘
      Mes pions = 1.0        Ses pions = 1.0          Bobail = 1.0
      Tout le reste = 0.0    Tout le reste = 0.0      Tout le reste = 0.0
```
