# Conditions de victoire dans Bobail

> Objectif : comprendre les 2 facons de gagner une partie de Bobail,
> la regle de jeu correspondante, et comment chacune est implementee
> dans le code (`environments/bobail.py`).

---

## Rappels necessaires avant de commencer

### Qui est le "joueur courant" ?

A chaque instant, `self._current` indique quel joueur doit jouer :
- `self._current == 0` → c'est au Joueur 0
- `self._current == 1` → c'est au Joueur 1

### Qu'est-ce qu'une "rangee maison" (home row) ?

Chaque joueur a une rangee de depart, celle ou ses 5 pions sont places au debut :

```
  row 0 :  1  1  1  1  1     ← rangee maison du Joueur 1
  row 1 :  .  .  .  .  .
  row 2 :  .  .  B  .  .
  row 3 :  .  .  .  .  .
  row 4 :  0  0  0  0  0     ← rangee maison du Joueur 0
```

Dans le code :

```python
# environments/bobail.py:74
home_row = 4 if self._current == 0 else 0
```

- Joueur 0 → `home_row = 4` (ligne du bas)
- Joueur 1 → `home_row = 0` (ligne du haut)

### Que signifie `reward` ?

Le contrat est defini dans `environments/base.py:16-21` :

```python
def step(self, action: int) -> tuple[np.ndarray, float, bool]:
    """Execute une action. Retourne (next_state, reward, done).

    Pour les jeux a deux joueurs :
    - reward est du point de vue du joueur qui vient d'agir
    - next_state est du point de vue du NOUVEAU joueur courant
    """
```

Donc :
- `reward = +1.0` → le joueur qui vient de jouer a **gagne**
- `reward = 0.0` → rien de special, la partie continue
- `done = True` → la partie est **terminee**

### Qu'est-ce que la convention `current = perdant` ?

Apres la fin de partie, `self._current` est positionne pour pointer vers le **perdant**.
C'est une convention interne qui permet de savoir, apres coup, qui a perdu.

---

## Victoire n.1 — Le bobail atteint la rangee maison du joueur

### La regle

Pendant la **phase bobail**, le joueur courant deplace le bobail d'une case.
Si apres ce deplacement, le bobail se retrouve sur la **rangee maison du joueur courant**,
alors ce joueur **GAGNE**.

Dit autrement : le but du jeu est de ramener le bobail "chez soi" (sur sa propre ligne de depart).

### Le code — `environments/bobail.py:67-80`

```python
if self._phase == PHASE_BOBAIL:
    self._bobail = to_cell                          # 1. Deplace le bobail
    self._phase = PHASE_PIECE

    br, _ = _idx_to_rc(self._bobail)                # 2. Recupere la LIGNE d'arrivee
    home_row = 4 if self._current == 0 else 0       # 3. Determine la rangee maison
    if br == home_row:                               # 4. Le bobail est sur ma rangee ?
        self._done = True                            #    OUI → partie terminee
        self._current = 1 - self._current            #    Convention : current = perdant
        return self.state_description(), 1.0, True   #    reward = +1.0 (je gagne)
```

### Etape par etape

| Etape | Ligne | Ce qui se passe |
|:-----:|:-----:|:----------------|
| 1 | 68 | Le bobail est deplace vers la case `to_cell` |
| 2 | 72 | On extrait la **ligne** (row) de la nouvelle position du bobail. `_idx_to_rc` fait `divmod(index, 5)` et renvoie `(row, col)`. On ne garde que `row` (la colonne est ignoree avec `_`) |
| 3 | 74 | On determine la rangee maison du joueur courant. Joueur 0 → ligne 4. Joueur 1 → ligne 0 |
| 4 | 76 | On compare : **est-ce que la ligne du bobail == ma rangee maison ?** |
| | 77 | Si **OUI** : `self._done = True` → la partie est terminee |
| | 79 | `self._current = 1 - self._current` → on bascule current vers l'adversaire (convention : current = perdant) |
| | 80 | On retourne `reward = 1.0` (positif = victoire du point de vue du joueur qui vient d'agir) et `done = True` |

### Exemple concret — Le Joueur 0 gagne

```
AVANT le coup :                         C'est le tour du Joueur 0
                                         Phase = BOBAIL
  1  .  1  1  1                          self._current = 0
  .  .  .  .  .
  .  .  .  .  .
  .  .  B  1  .                          Bobail en cellule 17 → position (3, 2)
  0  0  .  0  0

Le Joueur 0 deplace le bobail vers le BAS (direction ↓, dr=+1, dc=0)
  → nouvelle position : (3+1, 2) = (4, 2) = cellule 22

APRES le coup :

  1  .  1  1  1
  .  .  .  .  .
  .  .  .  .  .
  .  .  .  1  .
  0  0  B  0  0                          Bobail en cellule 22 → position (4, 2)

Verification dans le code :
  br = 4                                 (ligne du bobail apres deplacement)
  home_row = 4                           (rangee maison du Joueur 0)
  br == home_row → 4 == 4 → VRAI

  → self._done = True
  → self._current = 1 - 0 = 1           (convention : current = perdant = Joueur 1)
  → return (state, 1.0, True)            (reward +1.0 pour le Joueur 0 qui a agi)

RESULTAT : Le Joueur 0 a ramene le bobail sur sa rangee → JOUEUR 0 GAGNE
```

### Exemple concret — Le Joueur 1 gagne

```
AVANT le coup :                         C'est le tour du Joueur 1
                                         Phase = BOBAIL
  1  .  1  1  1                          self._current = 1
  0  .  B  .  .                          Bobail en cellule 7 → position (1, 2)
  .  .  .  .  .
  .  .  .  .  .
  .  0  0  0  0

Le Joueur 1 deplace le bobail vers le HAUT (direction ↑, dr=-1, dc=0)
  → nouvelle position : (1-1, 2) = (0, 2) = cellule 2

APRES le coup :

  1  .  B  1  1                          Bobail en cellule 2 → position (0, 2)
  0  .  .  .  .
  .  .  .  .  .
  .  .  .  .  .
  .  0  0  0  0

Verification dans le code :
  br = 0                                 (ligne du bobail apres deplacement)
  home_row = 0                           (rangee maison du Joueur 1)
  br == home_row → 0 == 0 → VRAI

  → self._done = True
  → self._current = 1 - 1 = 0           (convention : current = perdant = Joueur 0)
  → return (state, 1.0, True)            (reward +1.0 pour le Joueur 1 qui a agi)

RESULTAT : Le Joueur 1 a ramene le bobail sur sa rangee → JOUEUR 1 GAGNE
```

---

## Victoire n.2 — Le bobail est bloque

### La regle

Apres qu'un joueur a deplace un de ses pions (phase piece), on verifie si l'adversaire
**pourra** deplacer le bobail au debut de son prochain tour.

Si le bobail est **completement entoure** — toutes ses cases adjacentes sont soit occupees
par des pions, soit hors des limites du plateau — alors l'adversaire ne peut pas jouer.
Il **perd**.

### Le code — `environments/bobail.py:92-105`

```python
# Passage a l'adversaire
opponent = 1 - self._current
self._current = opponent                            # 1. Le tour passe a l'adversaire
self._turn_number += 1

self._phase = PHASE_BOBAIL                          # 2. Son tour commencerait par le bobail

if not self._bobail_moves():                        # 3. Peut-il bouger le bobail ?
    self._done = True                               #    NON → partie terminee
    return self.state_description(), 1.0, True      #    reward = +1.0 (le joueur precedent gagne)
```

### Etape par etape

| Etape | Ligne | Ce qui se passe |
|:-----:|:-----:|:----------------|
| 1 | 93-94 | Le joueur courant a fini de deplacer son pion. On passe la main a l'adversaire : `self._current = opponent` |
| 2 | 98 | La phase est mise a `PHASE_BOBAIL` : le prochain joueur devrait bouger le bobail |
| 3 | 101 | On appelle `_bobail_moves()` pour verifier si le nouveau joueur courant a **au moins 1 mouvement legal** pour le bobail |
| | | Si la liste est **vide** (`not []` est `True`) → ce joueur est bloque → il **perd** |
| | 102 | `self._done = True` → partie terminee |
| | 105 | `reward = 1.0` → positif du point de vue du joueur **precedent** (celui qui a deplace son pion et provoque le blocage). Current pointe deja vers le perdant (convention respectee) |

### Comment `_bobail_moves()` detecte le blocage — `environments/bobail.py:117-128`

```python
def _bobail_moves(self) -> list[int]:
    br, bc = _idx_to_rc(self._bobail)                # Position du bobail
    occupied = self._pieces[0] | self._pieces[1]     # Toutes les cases avec des pions
    moves = []
    for dr, dc in DIRECTIONS:                        # Pour chacune des 8 directions
        nr, nc = br + dr, bc + dc                    # Case adjacente dans cette direction
        if _in_bounds(nr, nc):                       # Est-elle dans le plateau ?
            target = _rc_to_idx(nr, nc)
            if target not in occupied:               # Est-elle libre ?
                moves.append(...)                    # OUI → mouvement legal
    return moves                                     # Si vide → bobail completement bloque
```

La fonction teste les 8 cases autour du bobail. Pour chaque case :
1. Est-elle **dans les limites** du plateau ? (pas hors de la grille 5x5)
2. Est-elle **libre** ? (pas occupee par un pion du joueur 0 ou du joueur 1)

Si **aucune** des 8 cases ne passe ces 2 tests → liste vide → bobail bloque.

**Remarque importante** : le bobail lui-meme n'est **pas** dans `occupied` ici.
`occupied = self._pieces[0] | self._pieces[1]` ne contient que les pions des joueurs.
C'est logique : le bobail ne se bloque pas lui-meme, il EST l'objet qu'on essaie de deplacer.

### Exemple concret — Le Joueur 1 perd (bobail bloque)

```
AVANT le coup :                         C'est le tour du Joueur 0
                                         Phase = PIECE
  1  .  1  1  1                          self._current = 0
  .  0  1  0  .
  .  1  B  1  .                          Bobail en cellule 12 → position (2, 2)
  .  0  .  0  .
  .  .  0  .  .

Le Joueur 0 deplace son pion de cellule 22 vers cellule 17 (glissement ↑)

APRES le deplacement du pion :

  1  .  1  1  1
  .  0  1  0  .
  .  1  B  1  .                          Bobail en cellule 12
  .  0  0  0  .                          ← pion arrive en 17
  .  .  .  .  .

Maintenant le code execute les lignes 92-105 :

  1. self._current = 1                   (passage au Joueur 1)
  2. self._phase = PHASE_BOBAIL

  3. Appel _bobail_moves() :
     Bobail en (2,2). Verification des 8 cases adjacentes :

     ┌──────────┬─────────────┬──────────────────────────┬────────┐
     │ Direction │ Case testee │ Contenu                  │ Libre? │
     ├──────────┼─────────────┼──────────────────────────┼────────┤
     │ ↖ (1,1)  │ cellule 6   │ Pion Joueur 0            │ NON    │
     │ ↑ (1,2)  │ cellule 7   │ Pion Joueur 1            │ NON    │
     │ ↗ (1,3)  │ cellule 8   │ Pion Joueur 0            │ NON    │
     │ ← (2,1)  │ cellule 11  │ Pion Joueur 1            │ NON    │
     │ → (2,3)  │ cellule 13  │ Pion Joueur 1            │ NON    │
     │ ↙ (3,1)  │ cellule 16  │ Pion Joueur 0            │ NON    │
     │ ↓ (3,2)  │ cellule 17  │ Pion Joueur 0 (nouveau!) │ NON    │
     │ ↘ (3,3)  │ cellule 18  │ Pion Joueur 0            │ NON    │
     └──────────┴─────────────┴──────────────────────────┴────────┘

     8 directions testees, 0 case libre
     → _bobail_moves() retourne []

  not [] → True → le Joueur 1 ne peut pas jouer

  → self._done = True
  → return (state, 1.0, True)

RESULTAT : Le Joueur 1 ne peut pas bouger le bobail → JOUEUR 1 PERD
           Le Joueur 0 (qui a cree le blocage) GAGNE
```

---

## Resume

```
┌───────────────────────────────────────────────────────────────────────────┐
│                      2 CONDITIONS DE VICTOIRE                             │
├───────────────────────┬───────────────────────────────────────────────────┤
│                       │                                                   │
│  N.1 : BOBAIL RAMENE  │  Le joueur deplace le bobail sur sa propre        │
│  CHEZ SOI             │  rangee maison (la ou ses pions demarrent)        │
│                       │  → CE JOUEUR GAGNE                                │
│                       │                                                   │
│  Quand :              │  Pendant la phase bobail, dans step()             │
│  Code :               │  environments/bobail.py lignes 72-80             │
│  Test :               │  br == home_row                                   │
│  Reward :             │  +1.0 (du point de vue du gagnant)               │
│                       │                                                   │
├───────────────────────┼───────────────────────────────────────────────────┤
│                       │                                                   │
│  N.2 : BOBAIL BLOQUE  │  Apres un deplacement de pion, l'adversaire ne   │
│                       │  peut pas bouger le bobail (8 cases adjacentes    │
│                       │  toutes occupees ou hors-limites)                 │
│                       │  → L'ADVERSAIRE PERD                              │
│                       │                                                   │
│  Quand :              │  Apres la phase piece, dans step()                │
│  Code :               │  environments/bobail.py lignes 100-105           │
│  Test :               │  not self._bobail_moves()                        │
│  Reward :             │  +1.0 (du point de vue du joueur qui a joue)     │
│                       │                                                   │
├───────────────────────┼───────────────────────────────────────────────────┤
│                       │                                                   │
│  CONVENTION COMMUNE   │  Apres fin de partie :                            │
│                       │  - done = True                                    │
│                       │  - reward = +1.0 (toujours positif pour le        │
│                       │    gagnant, car reward = point de vue de celui     │
│                       │    qui vient d'agir, cf. base.py:19)              │
│                       │  - current_player() retourne le PERDANT           │
│                       │                                                   │
└───────────────────────┴───────────────────────────────────────────────────┘
```
