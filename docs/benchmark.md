# `scripts/benchmark.py`

Mesure la **vitesse d'execution** (parties/sec) et la **duree moyenne** (coups/partie) de chaque environnement, en jouant N parties avec des actions aleatoires.

## Utilisation

```bash
# 1000 parties par defaut
uv run scripts/benchmark.py

# Ou specifier le nombre de parties
uv run scripts/benchmark.py 5000
```

## Flux d'execution

```
main()
│
├── 1. Lire argv[1] (ou defaut = 1000)
├── 2. Afficher l'en-tete du tableau
│
└── 3. Pour CHAQUE env dans ENV_REGISTRY :
     │
     └── benchmark_env(env_name, n)
          │
          ├── a. get_env(env_name) ──► instancie l'env
          ├── b. time.perf_counter() ──► chrono START
          │
          ├── c. Repeter n fois :
          │     ┌──────────────────────────────────────┐
          │     │  env.reset()                         │
          │     │  while not done:                     │
          │     │    action = random.choice(            │
          │     │               env.available_actions())│
          │     │    state, reward, done = env.step()   │
          │     │    steps += 1                         │
          │     └──────────────────────────────────────┘
          │
          ├── d. time.perf_counter() ──► chrono STOP
          └── e. Retourner (n/elapsed, total_steps/n)
```

## Detail des fonctions

### `benchmark_env(env_name, n_games)` — ligne 16

| Etape | Code | Description |
|---|---|---|
| Creer l'env | `get_env(env_name)` | `ENV_REGISTRY` mappe `"bobail"` -> `BobailEnv()` |
| Chrono | `time.perf_counter()` | Chronometre haute precision (nanosecondes) |
| Boucle parties | `for _ in range(n_games)` | Joue `n` parties completes |
| Reset | `env.reset()` | Remet le plateau a l'etat initial |
| Action | `random.choice(env.available_actions())` | Coup legal aleatoire |
| Step | `env.step(action)` -> `(state, reward, done)` | Applique l'action |
| Resultat | `n / elapsed`, `total_steps / n` | Parties/sec et coups moyens/partie |

### `main()` — ligne 38

| Etape | Code | Description |
|---|---|---|
| Lire N | `int(sys.argv[1])` ou `1000` | Argument CLI optionnel |
| Boucle | `for env_name in ENV_REGISTRY` | Itere sur les 4 envs : `line_world`, `grid_world`, `tictactoe`, `bobail` |
| Affichage | `print(f"{env_name:<15} ...")` | Tableau aligne en colonnes |

## Sortie attendue

```
Running 1000 random games per environment...

Environment       Games/sec    Avg steps
------------------------------------------
line_world         85000.3          4.2
grid_world         42000.7         12.8
tictactoe          31000.5          6.1
bobail              8500.2         28.4
```

> Les valeurs dependent de la machine et de la complexite de chaque env.

## A quoi ca sert

| Cas d'usage | Exemple |
|---|---|
| **Comparer les envs** | `bobail` est ~10x plus lent que `line_world` -> normal, le plateau est plus complexe |
| **Detecter des regressions** | Apres modif de `BobailEnv`, les games/sec chutent -> probleme de performance |
| **Estimer le temps d'entrainement** | 1M parties a 8000 games/sec -> ~2 min de simulation |
