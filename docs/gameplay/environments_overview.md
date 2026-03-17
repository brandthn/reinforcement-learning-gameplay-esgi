# Vue d'ensemble des Environnements

## Les 4 environnements implementes

```mermaid
graph LR
    subgraph "Environnements de test"
        LW["LineWorld<br/>1D, single-player"]
        GW["GridWorld<br/>2D, single-player"]
        TTT["TicTacToe<br/>3x3, adversarial"]
    end
    subgraph "Gameplay choisi"
        BOB["Bobail<br/>5x5, adversarial"]
    end

    LW -->|"Complexite croissante"| GW
    GW -->|"Introduction adversarial"| TTT
    TTT -->|"Jeu complexe"| BOB

    style LW fill:#4CAF50,color:#fff
    style GW fill:#2196F3,color:#fff
    style TTT fill:#FF9800,color:#fff
    style BOB fill:#F44336,color:#fff
```

## Tableau comparatif

| Propriete | LineWorld | GridWorld | TicTacToe | Bobail |
|-----------|-----------|-----------|-----------|--------|
| **Type** | Navigation 1D | Navigation 2D | Jeu de plateau | Jeu de plateau |
| **Joueurs** | 1 (single-player) | 1 (single-player) | 2 (adversarial) | 2 (adversarial) |
| **Taille de grille** | 5 cellules | 5x5 = 25 cellules | 3x3 = 9 cellules | 5x5 = 25 cellules |
| **Taille du vecteur d'etat** | 5 | 25 | 27 (3 x 9) | 75 (3 x 25) |
| **Taille de l'espace d'actions** | 2 | 4 | 9 | 625 |
| **Actions legales typiques** | 1-2 | 2-4 | 5-9 | ~20-60 |
| **Reward victoire** | +1.0 | +1.0 | +1.0 | +1.0 |
| **Reward defaite** | N/A | N/A | -1.0 (implicite) | -1.0 (implicite) |
| **Reward autre** | 0.0 | 0.0 | 0.0 | 0.0 |
| **Encodage etat** | One-hot | One-hot | 3 canaux binaires | 3 canaux binaires |
| **Perspective** | N/A | N/A | Joueur courant | Joueur courant |
| **Action masking** | Oui (bords) | Oui (bords) | Oui (cases occupees) | Oui (mouvements legaux) |

## Interface commune : `Environment` (ABC)

```mermaid
classDiagram
    class Environment {
        <<abstract>>
        +reset() ndarray
        +step(action: int) tuple[ndarray, float, bool]
        +available_actions() list[int]
        +state_description() ndarray
        +action_space_size() int
        +state_space_size() int
        +is_adversarial() bool
        +current_player() int
        +clone() Environment
        +render_text() str
    }

    class LineWorldEnv {
        -_size: int = 5
        -_pos: int
    }
    class GridWorldEnv {
        -_rows: int = 5
        -_cols: int = 5
        -_row: int
        -_col: int
    }
    class TicTacToeEnv {
        -_board: ndarray[9]
        -_current: int
        -_done: bool
    }
    class BobailEnv {
        -_pieces: list[set]
        -_bobail: int
        -_current: int
        -_phase: int
        -_first_turn: bool
    }

    Environment <|-- LineWorldEnv
    Environment <|-- GridWorldEnv
    Environment <|-- TicTacToeEnv
    Environment <|-- BobailEnv
```

## Cycle de vie d'un episode

```mermaid
sequenceDiagram
    participant C as Caller (Trainer / GUI)
    participant E as Environment
    participant A as Agent

    C->>E: reset()
    E-->>C: state (ndarray)

    loop Chaque step jusqu'a done=True
        C->>E: available_actions()
        E-->>C: [action_0, action_1, ...]
        C->>A: act(state, available_actions)
        A-->>C: action (int)
        C->>E: step(action)
        E-->>C: (next_state, reward, done)
    end
```

## Registre des environnements

Le fichier `environments/__init__.py` definit un registre qui lie les noms aux classes :

```python
ENV_REGISTRY = {
    "line_world": LineWorldEnv,
    "grid_world": GridWorldEnv,
    "tictactoe":  TicTacToeEnv,
    "bobail":     BobailEnv,
}
```

Instanciation : `env = get_env("bobail")` cree une instance prete a l'emploi.

---

## LineWorld : Navigation 1D

```
Position initiale:        Position finale (victoire):
[A|.|.|.|G]               [.|.|.|.|A]
 0 1 2 3 4                 0 1 2 3 4
```

- **Action 0** : aller a gauche (si pos > 0)
- **Action 1** : aller a droite (si pos < 4)
- **Objectif** : atteindre la cellule 4 (goal)
- **Reward** : 1.0 quand pos == 4, sinon 0.0

---

## GridWorld : Navigation 2D

```
Position initiale:           Position finale:
 A . . . .                    . . . . .
 . . . . .                    . . . . .
 . . . . .        -->         . . . . .
 . . . . .                    . . . . .
 . . . . G                    . . . . A
```

| Action | Direction | Delta |
|--------|-----------|-------|
| 0 | Haut | row - 1 |
| 1 | Bas | row + 1 |
| 2 | Gauche | col - 1 |
| 3 | Droite | col + 1 |

- **Objectif** : atteindre (4, 4) depuis (0, 0)
- **Action masking** : impossible de sortir de la grille

---

## TicTacToe : Morpion adversarial

```
Indexes des cellules:     Exemple en cours:
  0 | 1 | 2                X | O | .
  ---------                ---------
  3 | 4 | 5                . | X | .
  ---------                ---------
  6 | 7 | 8                O | . | .
```

- **9 actions** : une par cellule (0 a 8)
- **Action masking** : seules les cases vides sont jouables
- **Lignes gagnantes** : 8 combinaisons (3 lignes + 3 colonnes + 2 diagonales)

---

## Bobail : Le jeu choisi

Voir [bobail.md](bobail.md) pour la documentation complete.
