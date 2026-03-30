# Part 2 — Plateau, Pions & Règles du jeu

> Objectif : comprendre le plateau 5x5, les pions, les phases de jeu,
> et les conditions de victoire tels qu'implémentés dans `BobailEnv`.

---

## 1. Le plateau 5x5

Le plateau est une grille de 5 lignes × 5 colonnes = **25 cellules**.

### 1.1 Coordonnées (row, col) → Index

Chaque cellule a un index linéaire calculé par :

```python
# environments/bobail.py:22-23
def _rc_to_idx(r: int, c: int) -> int:
    return r * BOARD_SIZE + c       # BOARD_SIZE = 5
```

Et l'inverse :

```python
# environments/bobail.py:26-27
def _idx_to_rc(idx: int) -> tuple[int, int]:
    return divmod(idx, BOARD_SIZE)  # → (row, col)
```

### 1.2 Carte des index

```
         col=0  col=1  col=2  col=3  col=4
        ┌──────┬──────┬──────┬──────┬──────┐
 row=0  │  0   │  1   │  2   │  3   │  4   │  ← Rangée du Joueur 1
        ├──────┼──────┼──────┼──────┼──────┤
 row=1  │  5   │  6   │  7   │  8   │  9   │
        ├──────┼──────┼──────┼──────┼──────┤
 row=2  │ 10   │ 11   │ 12   │ 13   │ 14   │  ← Bobail au centre
        ├──────┼──────┼──────┼──────┼──────┤
 row=3  │ 15   │ 16   │ 17   │ 18   │ 19   │
        ├──────┼──────┼──────┼──────┼──────┤
 row=4  │ 20   │ 21   │ 22   │ 23   │ 24   │  ← Rangée du Joueur 0
        └──────┴──────┴──────┴──────┴──────┘
```

### 1.3 Exemples de conversion

| (row, col) | Index | Calcul |
|:---:|:---:|:---|
| (0, 0) | 0 | 0×5 + 0 |
| (0, 4) | 4 | 0×5 + 4 |
| (2, 2) | 12 | 2×5 + 2 |
| (4, 0) | 20 | 4×5 + 0 |
| (4, 4) | 24 | 4×5 + 4 |
| (3, 1) | 16 | 3×5 + 1 |

---

## 2. Les 3 types de pièces

### 2.1 Positions initiales dans le code

```python
# environments/bobail.py:12-16
_P0_START = [(4, c) for c in range(5)]   # Joueur 0 : [(4,0), (4,1), (4,2), (4,3), (4,4)]
_P1_START = [(0, c) for c in range(5)]   # Joueur 1 : [(0,0), (0,1), (0,2), (0,3), (0,4)]
_BOBAIL_START = (2, 2)                   # Bobail : centre du plateau
```

### 2.2 Plateau initial (render_text)

```
1 1 1 1 1       ← 5 pions du Joueur 1 (indices 0,1,2,3,4)
. . . . .
. . B . .       ← Bobail (index 12)
. . . . .
0 0 0 0 0       ← 5 pions du Joueur 0 (indices 20,21,22,23,24)
```

### 2.3 Stockage interne

```python
# environments/bobail.py:43-44
self._pieces = [set(), set()]   # _pieces[0] = pions joueur 0, _pieces[1] = pions joueur 1
self._bobail = 0                # index de la cellule du bobail (int)
```

Après `reset()` :
```python
self._pieces[0] = {20, 21, 22, 23, 24}   # row 4, colonnes 0-4
self._pieces[1] = {0, 1, 2, 3, 4}        # row 0, colonnes 0-4
self._bobail = 12                         # centre (2,2)
```

---

## 3. Les 8 directions de mouvement

```python
# environments/bobail.py:7-9
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]
```

Représentation visuelle :

```
    (-1,-1)  (-1, 0)  (-1,+1)
        ↖       ↑       ↗
         \      |      /
  (0,-1) ← ── [X] ── → (0,+1)
         /      |      \
        ↙       ↓       ↘
    (+1,-1)  (+1, 0)  (+1,+1)
```

| (dr, dc) | Direction | Nom |
|:---------:|:---------:|:---:|
| (-1, -1) | ↖ | Nord-Ouest |
| (-1, 0)  | ↑ | Nord |
| (-1, +1) | ↗ | Nord-Est |
| (0, -1)  | ← | Ouest |
| (0, +1)  | → | Est |
| (+1, -1) | ↙ | Sud-Ouest |
| (+1, 0)  | ↓ | Sud |
| (+1, +1) | ↘ | Sud-Est |

---

## 4. Les deux phases d'un tour

Chaque tour comporte **2 phases** exécutées séquentiellement :

```
┌─────────────────────────────────────────────────────────────┐
│                     UN TOUR COMPLET                          │
│                                                              │
│  Phase 0 (PHASE_BOBAIL)    Phase 1 (PHASE_PIECE)           │
│  ┌─────────────────────┐   ┌──────────────────────────┐    │
│  │ Déplacer le Bobail   │ → │ Déplacer un de ses pions  │   │
│  │ de 1 case            │   │ en glissant               │   │
│  └─────────────────────┘   └──────────────────────────┘    │
│                                                              │
│  Exception : Joueur 0, tout premier tour → skip Phase 0     │
└─────────────────────────────────────────────────────────────┘
```

### 4.1 Variables internes de phase

```python
# environments/bobail.py:18-19
PHASE_BOBAIL = 0
PHASE_PIECE = 1

# Dans __init__ et reset() :
self._phase = PHASE_PIECE    # Premier tour → on saute le bobail
self._first_turn = True
```

### 4.2 Transition de phase dans step()

```python
# Quand phase == PHASE_BOBAIL (ligne 67-78) :
#   1. Déplace le bobail
#   2. self._phase = PHASE_PIECE    ← passe à la phase suivante
#   3. Vérifie si le bobail a atteint la rangée maison
#   4. Retourne (state, reward, done)
#   ⚠️ PAS de changement de joueur !

# Quand phase == PHASE_PIECE (ligne 80-103) :
#   1. Déplace le pion du joueur courant
#   2. self._current = 1 - self._current   ← change de joueur
#   3. self._phase = PHASE_BOBAIL           ← prochain tour commence par bobail
#   4. Vérifie si l'adversaire peut bouger le bobail
#   5. Retourne (state, reward, done)
```

### 4.3 Diagramme de la machine à états

```
                    ┌──────────────────────────────┐
                    │          reset()              │
                    │  current=0, phase=PIECE       │
                    │  first_turn=True              │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │      PHASE_PIECE              │
                    │  Joueur 0 déplace un pion     │
                    │  first_turn → False           │
              ┌─────│  current → 1 (switch)         │
              │     │  phase → BOBAIL               │
              │     └──────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│      PHASE_BOBAIL           │
│  Joueur 1 déplace le bobail │
│  phase → PIECE              │
│  (pas de switch joueur)     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      PHASE_PIECE            │
│  Joueur 1 déplace un pion   │
│  current → 0 (switch)       │──────► Et ainsi de suite...
│  phase → BOBAIL             │
└─────────────────────────────┘
```

---

## 5. Règles de mouvement

### 5.1 Mouvement du Bobail (Phase 0)

**Règle** : Le bobail avance d'**exactement 1 case** dans l'une des 8 directions.

**Contraintes** :
- La case cible doit être **dans les limites** du plateau
- La case cible doit être **inoccupée** (ni pion joueur 0, ni pion joueur 1)

```python
# environments/bobail.py:113-124
def _bobail_moves(self) -> list[int]:
    br, bc = _idx_to_rc(self._bobail)
    occupied = self._pieces[0] | self._pieces[1]     # union des 2 sets de pions
    moves = []
    for dr, dc in DIRECTIONS:                         # 8 directions
        nr, nc = br + dr, bc + dc                     # case adjacente
        if _in_bounds(nr, nc):                        # dans le plateau ?
            target = _rc_to_idx(nr, nc)
            if target not in occupied:                 # case libre ?
                moves.append(self._bobail * NUM_CELLS + target)  # encodage action
    return moves
```

### 5.2 Mouvement d'un pion (Phase 1) — Le Glissement

**Règle** : Un pion **glisse** dans une direction donnée **aussi loin que possible** jusqu'à
être bloqué. Il doit avancer d'**au moins 1 case**.

**Ce qui bloque** :
- Le bord du plateau
- Un pion adverse
- Un pion allié
- Le Bobail

```python
# environments/bobail.py:126-145
def _piece_moves(self) -> list[int]:
    occupied = self._pieces[0] | self._pieces[1] | {self._bobail}  # tout ce qui bloque
    moves = []
    for cell in self._pieces[self._current]:          # pour chaque pion du joueur
        r, c = _idx_to_rc(cell)
        for dr, dc in DIRECTIONS:                     # 8 directions
            nr, nc = r + dr, c + dc
            if not _in_bounds(nr, nc):                # hors-limites dès la 1ère case ?
                continue                              # → pas de mouvement possible
            if _rc_to_idx(nr, nc) in occupied:        # bloqué dès la 1ère case ?
                continue                              # → pas de mouvement possible
            # Glisser jusqu'au blocage
            while _in_bounds(nr + dr, nc + dc) and _rc_to_idx(nr + dr, nc + dc) not in occupied:
                nr += dr
                nc += dc
            target = _rc_to_idx(nr, nc)               # case finale
            moves.append(cell * NUM_CELLS + target)
    return moves
```

### 5.3 Différence clé : Bobail vs Pion

```
┌──────────────────────────────────────────────────────────┐
│           BOBAIL (Phase 0)        PION (Phase 1)         │
│                                                           │
│  Mouvement :  1 case exactement   Glisse jusqu'au bout   │
│  Bloqué par : pions uniquement    pions + bobail          │
│  Contrôlé :  occupé = P0 ∪ P1    occupé = P0 ∪ P1 ∪ {B} │
│  Minimum :   1 case              1 case                   │
└──────────────────────────────────────────────────────────┘
```

> **Subtilité** : Le Bobail **ne se bloque pas lui-même** quand on calcule ses mouvements
> (il n'est pas dans `occupied`). Mais un pion **est bloqué par le Bobail** (le Bobail
> est dans `occupied` pour `_piece_moves`).

---

## 6. Conditions de victoire / défaite

Il y a **2 façons** de gagner :

### 6.1 Amener le Bobail sur sa rangée maison

```python
# environments/bobail.py:71-76 (dans step(), après phase bobail)
br, _ = _idx_to_rc(self._bobail)
home_row = 4 if self._current == 0 else 0
if br == home_row:
    self._done = True
    self._current = 1 - self._current   # convention: current = perdant
    return self.state_description(), 1.0, True
```

| Joueur courant | Sa rangée maison | Victoire si Bobail atteint row... |
|:-:|:-:|:-:|
| 0 | row 4 (bas) | row 4 |
| 1 | row 0 (haut) | row 0 |

> **Attention** : la rangée "maison" est la rangée **de départ** du joueur.
> Le joueur 0 commence en bas (row 4), donc sa maison = row 4.
> Si le joueur 0 déplace le bobail sur row 4, il **gagne** (le bobail est "rentré chez lui").

### 6.2 Bloquer le Bobail

Si après qu'un joueur a déplacé son pion, l'adversaire ne peut **aucunement** bouger le Bobail
(toutes les cases adjacentes au Bobail sont occupées ou hors-limites), alors l'adversaire **perd**.

```python
# environments/bobail.py:96-101 (dans step(), après phase pièce)
if not self._bobail_moves():         # aucun mouvement possible pour le bobail
    self._done = True
    # current pointe déjà sur le joueur bloqué → il perd
    return self.state_description(), 1.0, True
```

### 6.3 Résumé des fins de partie

```
┌──────────────────────────────────────────────────────────┐
│                    FIN DE PARTIE                          │
│                                                           │
│  Cas 1 : Bobail atteint home_row du joueur courant       │
│    → reward = 1.0, done = True                           │
│    → Le joueur courant GAGNE                              │
│                                                           │
│  Cas 2 : Adversaire ne peut pas bouger le Bobail          │
│    → reward = 1.0, done = True                           │
│    → L'adversaire PERD (celui qui vient de jouer gagne)   │
│                                                           │
│  Convention : après fin, current_player() = le PERDANT    │
└──────────────────────────────────────────────────────────┘
```

---

## 7. Le premier tour — L'exception

```python
# environments/bobail.py:57
self._phase = PHASE_PIECE   # player 0's first turn skips bobail phase
self._first_turn = True
```

Au tout premier tour :
1. `current_player() = 0`
2. `_phase = PHASE_PIECE` (pas BOBAIL)
3. Le joueur 0 déplace directement un de ses pions
4. Après ce step, `_first_turn` passe à `False`
5. `_current` passe à 1, `_phase` passe à `PHASE_BOBAIL`
6. **Tous les tours suivants** commencent par la phase Bobail

**Pourquoi ?** Parce que le Bobail est au centre et les 8 cases autour sont libres — le premier
mouvement du bobail serait arbitraire et sans intérêt stratégique. La règle officielle de Bobail
impose au joueur 0 de commencer par déplacer un pion.
