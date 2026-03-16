# Spécifications d'Encodage — États et Actions

> **Ce document est un livrable obligatoire** pour l'étape intermédiaire. Pour chaque environnement, il décrit précisément comment l'état du jeu est représenté en vecteur numérique et comment les actions sont encodées en entiers.

---

## LineWorld

### État

Vecteur one-hot de taille N (nombre de cases), dtype float32.

| Index | Signification |
|-------|---------------|
| 0     | Agent en case 0 (1.0 si oui, 0.0 sinon) |
| 1     | Agent en case 1 |
| ...   | ... |
| N-1   | Agent en case N-1 (case objectif) |

**Taille du vecteur d'état :** N (par défaut N=5)

**Exemple :** Agent en case 2 sur une grille de 5 → `[0, 0, 1, 0, 0]`

### Actions

| Index | Action |
|-------|--------|
| 0     | Aller à gauche |
| 1     | Aller à droite |

**Taille de l'espace d'actions :** 2

**Masquage :** L'agent en case 0 ne peut pas aller à gauche. L'agent en case N-1 ne peut pas aller à droite (mais cette situation ne se produit pas car c'est la condition de victoire).

### Justification des choix

Encodage one-hot : représentation la plus directe pour un agent dont la seule information pertinente est sa position discrète. Pas de coordonnée continue nécessaire, pas d'information supplémentaire à encoder. Le one-hot est compatible avec toutes les familles d'algorithmes (tabulaire, DQN, policy gradient).

---

## GridWorld

### État

Vecteur one-hot de taille rows × cols, dtype float32.

| Index | Signification |
|-------|---------------|
| 0     | Agent en position (0,0) — coin supérieur gauche |
| 1     | Agent en position (0,1) |
| ...   | ... |
| r*cols+c | Agent en position (r,c) |
| rows*cols-1 | Agent en position (rows-1, cols-1) — objectif |

**Taille du vecteur d'état :** rows × cols (par défaut 25 pour une grille 5×5)

**Mapping :** Position (ligne r, colonne c) → index `r * cols + c`

**Exemple :** Agent en (1,2) sur une grille 5×5 → index 7, vecteur de 25 zéros avec un 1.0 à la position 7.

### Actions

| Index | Action |
|-------|--------|
| 0     | Haut (ligne - 1) |
| 1     | Bas (ligne + 1) |
| 2     | Gauche (colonne - 1) |
| 3     | Droite (colonne + 1) |

**Taille de l'espace d'actions :** 4

**Masquage :** Les actions menant hors de la grille sont retirées de `available_actions()`. Exemple : en (0,0), seuls bas (1) et droite (3) sont disponibles.

### Justification des choix

Même logique que LineWorld étendue en 2D. Le one-hot évite d'introduire un biais de proximité entre cases (que des coordonnées (x, y) normalisées introduiraient). Pour une grille 5×5, un vecteur de 25 éléments reste très compact.

---

## TicTacToe

### État

3 canaux de 9 valeurs chacun, concaténés en un vecteur de 27 float32.

| Canal | Index | Signification |
|-------|-------|---------------|
| 0 | 0–8 | Pièces du joueur courant (1.0 si marquée, 0.0 sinon) |
| 1 | 9–17 | Pièces de l'adversaire (1.0 si marquée, 0.0 sinon) |
| 2 | 18–26 | Cases vides (1.0 si vide, 0.0 sinon) |

**Mapping des cases :**
```
Index : 0 | 1 | 2
        3 | 4 | 5
        6 | 7 | 8
```

**Taille du vecteur d'état :** 27

**Perspective :** Joueur courant (D-002). Les canaux 0 et 1 permutent automatiquement quand le tour change. Le joueur voit toujours "mes pièces" en canal 0 et "pièces adverses" en canal 1.

### Actions

| Index | Action |
|-------|--------|
| 0–8   | Placer sa marque dans la case correspondante |

**Taille de l'espace d'actions :** 9

**Masquage :** Seules les cases vides (board[i] == 0) sont dans `available_actions()`.

### Justification des choix

L'encodage multi-canal sépare clairement les pièces alliées, ennemies, et les cases vides. Le canal "vide" est techniquement redondant (c'est le complément des deux autres) mais il facilite le travail du réseau en rendant l'information directement accessible. C'est l'encodage standard pour les jeux de plateau en RL (utilisé dans AlphaGo, AlphaZero).

---

## Bobail

### État

3 canaux de 25 valeurs chacun (grille 5×5), concaténés en un vecteur de 75 float32.

| Canal | Index | Signification |
|-------|-------|---------------|
| 0 | 0–24 | Pièces du joueur courant (1.0 si présente, 0.0 sinon) |
| 1 | 25–49 | Pièces de l'adversaire (1.0 si présente, 0.0 sinon) |
| 2 | 50–74 | Position du bobail (1.0 à la position du bobail, 0.0 ailleurs) |

**Mapping des cases :**
```
Index :  0 |  1 |  2 |  3 |  4
         5 |  6 |  7 |  8 |  9
        10 | 11 | 12 | 13 | 14
        15 | 16 | 17 | 18 | 19
        20 | 21 | 22 | 23 | 24
```

Position (ligne r, colonne c) → index `r * 5 + c`

**Taille du vecteur d'état :** 75

**Perspective :** Joueur courant (D-002). Les canaux 0 et 1 permutent automatiquement quand le tour change.

### Actions

Encodage `(case_départ, case_arrivée)` : `from_cell * 25 + to_cell`

| Phase | Actions possibles |
|-------|-------------------|
| Phase bobail | Déplacements du bobail (1 case, 8 directions) |
| Phase pièce | Déplacements d'une pièce (glissement maximal dans 1 direction) |

**Taille de l'espace d'actions :** 625 (25 × 25)

**Masquage :** `available_actions()` retourne uniquement les déplacements légaux selon la phase courante. En phase bobail : les cases adjacentes non-occupées autour du bobail. En phase pièce : pour chaque pièce du joueur courant, les destinations atteignables par glissement.

### Règles du jeu (référence)

- Grille 5×5, deux joueurs de 5 pièces chacun, un pion neutre (le bobail) au centre
- Chaque tour : déplacer le bobail (1 case, toute direction) puis déplacer une de ses pièces (aussi loin que possible dans une direction)
- Exception : premier tour, le joueur 0 ne déplace qu'une pièce (pas de bobail)
- Victoire : amener le bobail sur sa rangée de départ OU bloquer le bobail pour que l'adversaire ne puisse pas le déplacer
- Les pièces ne peuvent pas sauter par-dessus d'autres pièces

### Justification des choix

**Encodage d'état :** Même approche multi-canal que TicTacToe, étendue à une grille 5×5 avec un pion neutre. Le bobail a son propre canal car c'est un élément distinct du jeu (ni allié ni ennemi). 75 valeurs restent compactes pour un réseau de neurones.

**Encodage d'actions :** Le mapping `from * 25 + to` est uniforme pour les deux phases (bobail et pièce). Ça évite d'avoir deux espaces d'actions distincts. L'espace de 625 est clairsemé (la plupart des combinaisons sont illégales) mais le masquage via `available_actions()` garantit que seules les actions légales sont choisies.

---

*Ce document sera enrichi avec des diagrammes et exemples concrets au fur et à mesure de l'implémentation.*
