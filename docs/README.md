# Documentation du Projet Deep Reinforcement Learning

## Index de la documentation

Ce dossier contient la documentation technique complète du projet. Chaque sous-dossier couvre un aspect spécifique.

### Gameplay & Environnements (`gameplay/`)

| Document | Description |
|----------|-------------|
| [bobail.md](gameplay/bobail.md) | Jeu Bobail : regles, phases, conditions de victoire, implementation |
| [environments_overview.md](gameplay/environments_overview.md) | Vue d'ensemble des 4 environnements (Line, Grid, TicTacToe, Bobail) |
| [state_encoding.md](gameplay/state_encoding.md) | Vecteurs d'encoding de l'etat du jeu pour chaque environnement |
| [action_encoding.md](gameplay/action_encoding.md) | Vecteurs d'encoding des actions pour chaque environnement |
| [random_simulation_benchmark.md](gameplay/random_simulation_benchmark.md) | Benchmark des parties/seconde avec joueur random |
| [human_gui.md](gameplay/human_gui.md) | Jeu avec joueur humain + interface graphique PyGame |

### Architecture (`architecture/`)

| Document | Description |
|----------|-------------|
| [project_structure.md](architecture/project_structure.md) | Arborescence, modules, registres, flux de donnees |
| [training_pipeline.md](architecture/training_pipeline.md) | Pipeline d'entrainement single-player et self-play |
| [evaluation_pipeline.md](architecture/evaluation_pipeline.md) | Pipeline d'evaluation, metriques, checkpoints |

### Agents (`agents/`)

| Document | Description |
|----------|-------------|
| [agents_overview.md](agents/agents_overview.md) | Hierarchie des agents, interfaces, comparaisons |
| [dqn_family.md](agents/dqn_family.md) | DQN, DDQN, DDQN+ER, DDQN+PER en detail |

### Documents existants

| Document | Description |
|----------|-------------|
| [PROJECT_INSTRUCTIONS.md](PROJECT_INSTRUCTIONS.md) | Sujet et consignes du projet (syllabus) |
| [decisions.md](decisions.md) | Decisions de conception prises au fil du developpement |
| [encoding.md](encoding.md) | Specification initiale des encodings |
| [algorithms.md](algorithms.md) | Notes sur les algorithmes |
