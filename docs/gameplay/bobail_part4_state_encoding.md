# Part 4 — Encodage de l'État (State Encoding)

> Objectif : comprendre le vecteur de **80 floats** retourné par `state_description()`,
> ses 3 canaux spatiaux + 5 features stratégiques, la perspective du joueur courant,
> et comment il change à chaque step.

---

## 1. Structure du vecteur d'état

`state_description()` retourne un `np.ndarray` de shape `(80,)` et dtype `float32`.

Ce vecteur est la **concaténation de 3 canaux spatiaux de 25 éléments + 5 features stratégiques** :

```
state[80] = [ canal_mes_pions[25] | canal_pions_adverses[25] | canal_bobail[25] | features[5] ]
              indices 0..24         indices 25..49              indices 50..74     indices 75..79
```

```python
# environments/bobail.py (state_description)
def state_description(self) -> np.ndarray:
    my_pieces = np.zeros(NUM_CELLS, dtype=np.float32)    # canal 0 : mes pions
    opp_pieces = np.zeros(NUM_CELLS, dtype=np.float32)   # canal 1 : pions adverses
    bobail = np.zeros(NUM_CELLS, dtype=np.float32)       # canal 2 : bobail

    for idx in self._pieces[self._current]:
        my_pieces[idx] = 1.0
    for idx in self._pieces[1 - self._current]:
        opp_pieces[idx] = 1.0
    bobail[self._bobail] = 1.0

    # --- Features strategiques ---
    phase       = float(self._phase)
    br          = self._bobail // BOARD_SIZE
    my_home     = 4 if self._current == 0 else 0
    opp_home    = 0 if self._current == 0 else 4
    dist_my     = abs(br - my_home)  / (BOARD_SIZE - 1)
    dist_opp    = abs(br - opp_home) / (BOARD_SIZE - 1)
    mobilite    = len(self.available_actions()) / 40.0
    premier     = float(self._first_turn)

    extras = np.array([phase, dist_my, dist_opp, mobilite, premier], dtype=np.float32)
    return np.concatenate([my_pieces, opp_pieces, bobail, extras])
```

```python
def state_space_size(self) -> int:
    return NUM_CELLS * 3 + 5   # = 80
```

---

## 2. Les 3 canaux spatiaux en détail

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

## 3. Les 5 features stratégiques (indices 75..79)

Contrairement aux canaux spatiaux, ces 5 valeurs n'encodent pas des cases mais des **indicateurs de haut niveau** calculés à partir du plateau. L'idée : donner au réseau des signaux qu'il pourrait théoriquement réapprendre mais qui sont coûteux à extraire des 75 canaux spatiaux seuls.

| Index | Nom           | Plage                            | Calcul                                      | Rôle sémantique                                            |
|:-----:|:--------------|:---------------------------------|:--------------------------------------------|:-----------------------------------------------------------|
| 75    | `phase`       | {0.0, 1.0}                       | `float(self._phase)`                        | 0 = coup bobail à jouer, 1 = coup pion à jouer             |
| 76    | `dist_my`     | {0.0, 0.25, 0.5, 0.75, 1.0}      | `|ligne_bobail − ma_rangée_maison| / 4`     | Plus petit = plus proche de **ma** victoire                |
| 77    | `dist_opp`    | {0.0, 0.25, 0.5, 0.75, 1.0}      | `|ligne_bobail − rangée_adverse| / 4`       | Plus petit = plus proche de la victoire **adverse**        |
| 78    | `mobilite`    | **continue** (~ `[0, 1.0+]`)     | `len(available_actions()) / 40.0`           | Proxy de la condition "bobail bloqué"                      |
| 79    | `first_turn`  | {0.0, 1.0}                       | `float(self._first_turn)`                   | 1 pendant le tout premier coup du joueur 0 (pas de bobail) |

### 3.1 Rangée maison (home row)

- Joueur 0 : rangée maison = **ligne 4** (bas du plateau)
- Joueur 1 : rangée maison = **ligne 0** (haut du plateau)

Chaque joueur gagne en amenant le bobail sur **sa propre** rangée maison. La "distance" utilisée pour `dist_my` / `dist_opp` est simplement la différence en lignes, normalisée par `BOARD_SIZE − 1 = 4`.

### 3.2 Pourquoi `mobilite` est la seule feature continue

`mobilite` dépend du nombre total de coups légaux dans la configuration actuelle :
- en phase bobail : au plus 8 (cases adjacentes au bobail)
- en phase pion : jusqu'à ~40 (5 pions × ≤ 8 directions de glissade)

Le dénominateur 40 est choisi pour que la valeur tombe typiquement dans `[0, 1]`. Elle peut **légèrement dépasser 1.0** si plus de 40 coups sont disponibles (rare en pratique). C'est la seule valeur non strictement dans `{0, 0.25, 0.5, 0.75, 1.0}` de tout le vecteur.

### 3.3 Redondance et justification

Un réseau suffisamment profond pourrait déduire `dist_my`, `dist_opp` et `mobilite` des 75 canaux spatiaux. Mais :
- **`phase` / `first_turn`** ne sont PAS déductibles du plateau seul : deux plateaux identiques peuvent correspondre à des phases différentes. Ces deux features sont donc **nécessaires**, pas optionnelles.
- **`dist_my` / `dist_opp` / `mobilite`** sont des **shortcuts** qui accélèrent l'apprentissage en fournissant directement des signaux liés à la condition de victoire.

---

## 4. La perspective du joueur courant

**Point crucial** : l'état est TOUJOURS encodé du point de vue du joueur courant
(`self._current`). Canal 0 = "mes" pions, Canal 1 = pions de "l'autre", et `dist_my` / `dist_opp` sont eux aussi relatifs au joueur courant.

### 4.1 Exemple — État initial (current = 0)

```
Plateau :    1 1 1 1 1     Joueur 0 = pions en {20,21,22,23,24}
             . . . . .     Joueur 1 = pions en {0,1,2,3,4}
             . . B . .     Bobail en 12
             . . . . .     current_player = 0
             0 0 0 0 0     phase = PHASE_PIECE (premier tour)
```

```
Canal 0 (Mes pions = Joueur 0) :
  idx:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

Canal 1 (Pions adverses = Joueur 1) :
  idx: 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
       [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Canal 2 (Bobail) :
  idx: 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                                              ↑
                                          idx 62 = cellule 12 (le bobail)

Features stratégiques :
  idx:    75       76       77        78        79
       [ 1.0 ,   0.5 ,    0.5 ,   ≥ nb_coups/40 , 1.0 ]
         phase  dist_my  dist_opp   mobilite     first_turn
```

Calcul des features pour l'état initial (current=0) :
- `phase = 1.0` (premier tour du J0 = phase pion directement)
- `dist_my = |2 - 4| / 4 = 0.5` (bobail en ligne 2, ma maison = ligne 4)
- `dist_opp = |2 - 0| / 4 = 0.5` (maison adverse = ligne 0)
- `mobilite = len(piece_moves) / 40` (au premier tour, J0 a un certain nombre de glissades disponibles)
- `first_turn = 1.0`

### 4.2 Même plateau, mais vu par le Joueur 1 (current = 1)

Les canaux 0 et 1 **s'inversent**, et `dist_my` / `dist_opp` aussi (car `my_home` devient 0 au lieu de 4) :

```
Canal 0 (Mes pions = Joueur 1) : 1..1 en indices 0..4
Canal 1 (Pions adverses = Joueur 0) : 1..1 en indices 45..49
Canal 2 : identique (le bobail n'a pas de perspective)

Features :
  dist_my  = |2 - 0| / 4 = 0.5   (pour J1, ma maison = ligne 0)
  dist_opp = |2 - 4| / 4 = 0.5
```

Sur ce plateau symétrique, `dist_my` et `dist_opp` valent toutes deux 0.5 pour les deux joueurs — mais dès que le bobail bouge, elles divergent.

### 4.3 Pourquoi cette convention ?

```
┌─────────────────────────────────────────────────────────────┐
│  L'agent voit TOUJOURS :                                     │
│    Canal 0   = "mes pions"                                   │
│    Canal 1   = "pions ennemis"                               │
│    Canal 2   = "bobail"                                      │
│    dist_my   = distance vers ma victoire                     │
│    dist_opp  = distance vers la victoire adverse             │
│                                                               │
│  → L'agent n'a PAS besoin de savoir s'il est joueur 0 ou 1  │
│  → Il apprend UNE SEULE politique qui fonctionne des 2 côtés│
│  → Pas besoin d'encoder le numéro du joueur dans l'état      │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Mapping index d'état → interprétation

```
Si  i < 25           → canal 0 (mes pions), cellule = i
Si  25 ≤ i < 50      → canal 1 (pions adverses), cellule = i - 25
Si  50 ≤ i < 75      → canal 2 (bobail), cellule = i - 50
Si  i == 75          → phase
Si  i == 76          → dist_my
Si  i == 77          → dist_opp
Si  i == 78          → mobilite
Si  i == 79          → first_turn
```

### Tableau de correspondance

| Index état | Zone            | Cellule / sémantique | Exemple (état initial, current=0) |
|:----------:|:----------------|:---------------------|:-----------------------------------|
| 0          | canal 0 (moi)   | cellule 0            | 0.0 — pas de pion en (0,0)         |
| 20         | canal 0 (moi)   | cellule 20           | 1.0 — pion en (4,0)                |
| 25         | canal 1 (adv)   | cellule 0            | 1.0 — adversaire en (0,0)          |
| 44         | canal 1 (adv)   | cellule 19           | 0.0                                |
| 62         | canal 2 (bob)   | cellule 12           | 1.0 — bobail en (2,2)              |
| 70         | canal 2 (bob)   | cellule 20           | 0.0                                |
| 75         | feature `phase` | —                    | 1.0 (premier tour J0 = phase pion) |
| 76         | feature `dist_my`  | —                 | 0.5                                |
| 77         | feature `dist_opp` | —                 | 0.5                                |
| 78         | feature `mobilite` | —                 | ~ len(actions)/40                  |
| 79         | feature `first_turn` | —               | 1.0                                |

---

## 6. Propriétés du vecteur d'état

```
┌─────────────────────────────────────────────────────────────┐
│  Canaux spatiaux (indices 0..74) :                           │
│                                                               │
│  • sum(state[0:25])  = 5   (toujours 5 pions à moi)          │
│  • sum(state[25:50]) = 5   (toujours 5 pions adverses)       │
│  • sum(state[50:75]) = 1   (toujours 1 bobail)               │
│  • state[0:75] ∈ {0.0, 1.0}  (binaire)                       │
│  • Pas de chevauchement entre canaux                         │
│                                                               │
│  Features stratégiques (indices 75..79) :                    │
│                                                               │
│  • phase, first_turn ∈ {0.0, 1.0}                            │
│  • dist_my, dist_opp ∈ {0.0, 0.25, 0.5, 0.75, 1.0}           │
│  • mobilite ∈ [0.0, ~1.0+]  (continue, peut >1 rarement)     │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Quand l'état change-t-il ?

`state_description()` est appelé par `step()` à chaque retour. L'état change **après chaque action** (bobail OU pion) :

```
1. step() déplace bobail ou pion
2. (éventuellement) switch joueur courant
3. recalcule phase, first_turn
4. return self.state_description()  ← recalcule l'état complet depuis zéro
```

### 7.1 Changement après phase bobail

Le bobail bouge, mais le joueur courant **ne change pas** (il doit encore jouer son pion).
→ Canaux 0 et 1 inchangés (même perspective).
→ Canal 2 (bobail) change.
→ `phase` passe de 0 à 1.
→ `dist_my` / `dist_opp` changent (nouvelle ligne du bobail).
→ `mobilite` recalculée.

### 7.2 Changement après phase pion

Le pion bouge, puis le joueur courant **change** (`self._current = 1 - self._current`).
→ Canaux 0 et 1 **s'inversent** (perspective change).
→ `dist_my` / `dist_opp` **s'inversent aussi** (my_home change).
→ `phase` passe de 1 à 0 (prochain coup = bobail).
→ `first_turn` devient 0 dès la fin du premier tour.

---

## 8. Visualisation — Reconstruire le plateau depuis l'état

```python
def state_to_board(state: np.ndarray, current_player: int) -> str:
    """Reconstruit le plateau depuis le vecteur d'etat (80 dims)."""
    my  = state[0:25]
    opp = state[25:50]
    bob = state[50:75]
    phase, dist_my, dist_opp, mobilite, first_turn = state[75:80]

    lines = []
    for r in range(5):
        row = []
        for c in range(5):
            idx = r * 5 + c
            if bob[idx] == 1.0:
                row.append('B')
            elif my[idx] == 1.0:
                row.append(str(current_player))
            elif opp[idx] == 1.0:
                row.append(str(1 - current_player))
            else:
                row.append('.')
        lines.append(' '.join(row))
    board = '\n'.join(lines)
    meta = (f"phase={int(phase)}  dist_my={dist_my:.2f}  dist_opp={dist_opp:.2f}  "
            f"mobilite={mobilite:.2f}  first_turn={int(first_turn)}")
    return board + "\n" + meta
```

---

## 9. Résumé visuel

```
                                  state[80]
 ┌────────────────────────────────────────────────────────────────────────┐
 │  Canal 0 (moi)  │ Canal 1 (adv) │ Canal 2 (bobail) │ Features[5]       │
 │  state[0..24]   │ state[25..49] │ state[50..74]    │ state[75..79]     │
 │  5 × 1.0        │ 5 × 1.0       │ 1 × 1.0          │ phase, dist_my,   │
 │  20 × 0.0       │ 20 × 0.0      │ 24 × 0.0         │ dist_opp, mobilite│
 │                 │               │                  │ first_turn        │
 └─────────────────┴───────────────┴──────────────────┴───────────────────┘
       binaire           binaire          binaire          majoritairement
                                                         discret + mobilite
                                                              continue
```
