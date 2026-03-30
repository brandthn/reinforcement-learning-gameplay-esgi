# `scripts/run_gui.py`

Lance une interface graphique Pygame pour **jouer** ou **observer des agents jouer** sur les 4 environnements du projet.

## Utilisation

```bash
uv run scripts/run_gui.py
```

Aucun argument CLI. La fenetre s'ouvre a 900x700 pixels, 60 FPS.

## Architecture du script

Le script lui-meme est un simple point d'entree :

```python
from gui.app import run   # tout le code est dans gui/app.py
run()                      # pygame.init() -> App().main_loop() -> pygame.quit()
```

Toute la logique est dans la classe `App` de `gui/app.py`.

## Machine a etats de l'application

L'application fonctionne avec 3 etats :

```
┌────────────┐   clic "Start"   ┌──────────────┐   done=True   ┌──────────────┐
│  STATE_MENU │ ───────────────► │ STATE_PLAYING │ ────────────► │STATE_GAME_OVER│
│             │                  │               │               │               │
│ Selection:  │                  │ Boucle de jeu │               │ Affiche       │
│ - env       │                  │ tour par tour │               │ le gagnant    │
│ - agent(s)  │                  │               │               │               │
│ - model(s)  │                  │               │  clic "Menu"  │               │
└────────────┘ ◄─────────────────┴───────────────┘ ◄─────────────┴───────────────┘
```

## Phase 1 : Le menu (`STATE_MENU`)

Le menu permet de configurer la partie avant de lancer.

### Selecteurs disponibles

| Selecteur | Options | Details |
|---|---|---|
| **Environnement** | `Line`, `Grid`, `TicTac`, `Bobail` | Boutons radio, 1 seul actif (vert) |
| **Agent (Player 1)** | Tous les agents de `AGENT_REGISTRY` | Ex: `random`, `human`, `tabular_q`, `dqn`... |
| **Model (P1)** | Modeles trouves dans `results/<env>/<agent>/` | Seulement si l'agent necessite un modele |
| **Agent (Player 2)** | Idem P1 | Apparait uniquement si l'env est **adversarial** (`tictactoe`, `bobail`) |
| **Model (P2)** | Idem | Idem |

### Decouverte automatique des modeles (`_scan_models`)

L'application scanne le dossier `results/` pour trouver les modeles entraines :

```
results/
└── bobail/
    └── dqn/
        ├── best/              -> label "Best"
        │   └── model.pt
        ├── lr0.001_seed42/    -> label "seed 42"
        │   └── model_5000.pt
        └── lr0.01_seed7/      -> label "seed 7"
            └── model_3000.pt
```

**Regles de labeling :**

| Situation | Label genere | Exemple |
|---|---|---|
| Dossier `best/` avec `model.pt` | `"Best"` | `Best` |
| 1 seule config de params | `"seed {N}"` | `seed 42` |
| Plusieurs configs de params | `"cfg{X} s{N}"` | `cfg1 s42`, `cfg2 s7` |

Si un agent n'a aucun modele entraine -> le bouton "Start" est grise, impossible de lancer.

### Navigation modeles

Quand il y a plus de 4 modeles, des fleches `<` `>` permettent de paginer (max 4 boutons visibles a la fois).

## Phase 2 : Le jeu (`STATE_PLAYING`)

### Boucle de jeu (a chaque frame)

```
main_loop() [60 FPS]
│
├── Gerer les evenements Pygame (clic, clavier, fermeture)
│
├── Si c'est le tour d'un AGENT IA :
│   └── _try_ai_step()
│       ├── Verifier que 400ms se sont ecoulees depuis le dernier coup
│       └── _do_step_ai()
│           ├── state = env.state_description()
│           ├── available = env.available_actions()
│           ├── action = agent.act(state, available)
│           └── state, reward, done = env.step(action)
│
├── Si c'est le tour d'un HUMAIN :
│   └── Attendre un clic/touche du joueur
│       └── _do_step_human(action)
│           ├── agent.set_action(action)   # pre-charge le choix
│           ├── agent.act(state, available) # consomme le choix
│           └── env.step(action)
│
└── Dessiner l'ecran (plateau + info)
```

### Delai entre coups IA

`AI_STEP_DELAY_MS = 400` — les agents IA jouent un coup toutes les 400ms pour que le spectateur puisse suivre visuellement la partie.

### Input humain par environnement

| Environnement | Mode d'input | Details |
|---|---|---|
| `line_world` | Clavier | Fleches gauche/droite |
| `grid_world` | Clavier | Fleches directionnelles (4 directions) |
| `tictactoe` | Clic souris | Clic sur la case vide souhaitee |
| `bobail` | Clic souris (2 clics) | 1er clic = selectionner la piece, 2e clic = choisir la destination |

### Agents sans modele

Certains agents n'ont pas besoin de fichier `.pt` :

```python
AGENTS_WITHOUT_MODELS = {"random", "human"}
```

- `random` : choisit un coup legal au hasard
- `human` : attend l'input du joueur (clic/clavier)

## Phase 3 : Fin de partie (`STATE_GAME_OVER`)

### Determination du gagnant

```python
if reward > 0:
    winner = "Le joueur qui vient de jouer"
elif reward < 0:
    winner = "L'autre joueur"
else:
    "Draw!" (adversarial) ou "Game over!" (solo)
```

| Reward | Env adversarial | Env solo |
|---|---|---|
| `> 0` | Player X wins! | Game over! |
| `< 0` | Player Y wins! | Game over! |
| `= 0` | Draw! | Game over! |

Un bouton "Menu" ramene au `STATE_MENU`.

## Constantes visuelles

| Constante | Valeur | Role |
|---|---|---|
| `WINDOW_W, WINDOW_H` | 900 x 700 | Taille de la fenetre |
| `FPS` | 60 | Images par seconde |
| `AI_STEP_DELAY_MS` | 400 | Delai entre coups IA (ms) |
| `MAX_MODEL_BUTTONS` | 4 | Boutons modeles visibles avant pagination |
| `BG` | `(30, 30, 40)` | Fond sombre |
| `ACCENT` | `(100, 140, 255)` | Bleu pour boutons actifs |
| `GREEN` | `(80, 200, 120)` | Selection active |

## Exemples concrets

### Exemple 1 : Observer DQN vs Random sur Bobail

1. Lancer `uv run scripts/run_gui.py`
2. Selectionner `Bobail`
3. Player 1 : `dqn` -> choisir modele `Best`
4. Player 2 : `random`
5. Clic `Start`
6. L'IA DQN joue un coup toutes les 400ms, Random repond aleatoirement

### Exemple 2 : Jouer contre un agent sur TicTacToe

1. Lancer `uv run scripts/run_gui.py`
2. Selectionner `TicTac`
3. Player 1 : `human`
4. Player 2 : `tabular_q` -> choisir modele `seed 42`
5. Clic `Start`
6. Cliquer sur les cases pour jouer, l'agent repond apres 400ms

### Exemple 3 : Tester LineWorld en solo

1. Selectionner `Line`
2. Player 1 : `human`
3. Pas de Player 2 (env non-adversarial)
4. Clic `Start`
5. Utiliser les fleches gauche/droite pour naviguer
