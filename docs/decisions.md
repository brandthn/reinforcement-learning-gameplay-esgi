# docs/decisions.md
# Journal des Décisions d'Architecture

> Ce document est rempli **progressivement** au fil de l'implémentation. Chaque fois qu'un choix non-trivial est fait, une entrée est ajoutée ici. C'est la mémoire du projet et un outil de préparation pour la soutenance orale.

---

## Comment utiliser ce document

À chaque fois que vous prenez une décision technique que quelqu'un pourrait questionner plus tard :
1. Ajoutez une entrée ci-dessous avec la date
2. Expliquez le contexte, la décision, et pourquoi
3. Si des alternatives ont été considérées, expliquez pourquoi elles ont été rejetées

---

## Décisions prises en phase de conception (avant code)

### D-001 : Framework PyTorch (Mars 2026)

**Contexte :** Choix du framework de deep learning pour le projet.

**Décision :** PyTorch.

**Pourquoi :** Graphe de calcul dynamique (facilite le debug des boucles RL custom), standard de facto en recherche RL (AlphaZero, MuZero, PPO — les implémentations de référence sont en PyTorch), contrôle complet sur les fonctions de perte et boucles d'entraînement.

---

### D-002 : Perspective du joueur courant pour les jeux à deux joueurs (Mars 2026)

**Contexte :** TicTacToe et Bobail sont des jeux à deux joueurs. Les algorithmes RL sont conçus pour un seul agent. Comment gérer cette différence ?

**Décision :** L'environnement présente toujours l'état du point de vue du joueur dont c'est le tour. Le vecteur d'état contient "mes pièces" et "pièces adverses" — jamais "pièces du joueur 1" et "pièces du joueur 2".

**Pourquoi :** Les agents sont agnostiques au joueur — le même DQN peut jouer en tant que joueur 1 ou 2 sans modification. Le code d'entraînement pour les environnements mono-joueur et bi-joueur partage la même interface agent. Permet de confronter facilement agent vs random, agent vs heuristique, agent vs agent entraîné, agent vs humain.

**Alternatives rejetées :**
- Perspective fixe (toujours du point de vue joueur 1) — oblige les agents à savoir quel joueur ils sont
- Adversaire intégré dans l'environnement — limite la flexibilité (il faudrait des classes séparées par type d'adversaire)

---

### D-003 : Tours en deux phases pour Bobail (Mars 2026)

**Contexte :** À Bobail, chaque tour se compose de deux sous-actions : déplacer le bobail, puis déplacer une pièce. Comment modéliser ça dans l'interface RL standard ?

**Décision :** Chaque appel à `step(action)` traite une seule sous-action. L'environnement gère une variable `phase` interne. L'espace d'actions et `available_actions()` dépendent de la phase courante.

**Pourquoi :** Maintient un espace d'actions de taille raisonnable. L'alternative (produit cartésien des deux sous-actions) aurait fait exploser l'espace d'actions de manière combinatoire et rendu l'apprentissage beaucoup plus difficile.

---

### D-004 : Sous-dossiers dans agents/ pour l'organisation (Mars 2026)

**Contexte :** 16 algorithmes à implémenter. Comment organiser les fichiers ?

**Décision :** Sous-dossiers par famille (`value_based/`, `policy_gradient/`, `planning/`) à l'intérieur de `agents/`. Chaque fichier d'agent est autonome — pas de code utilitaire partagé à l'intérieur d'une même famille.

**Pourquoi :** Évite d'avoir 16+ fichiers dans un dossier plat. Permet à chaque contributeur de travailler dans sa zone. Les sous-dossiers sont purement organisationnels.

---

## Décisions prises pendant l'implémentation

### D-005 : uv comme gestionnaire de paquets (Mars 2026)

**Contexte :** Le guide d'implémentation mentionne `requirements.txt` pour la gestion des dépendances. Choix d'un outil de gestion de paquets.

**Décision :** Utilisation de `uv` avec `pyproject.toml` au lieu de `pip` + `requirements.txt`.

**Pourquoi :** `uv` gère l'installation Python, la résolution de dépendances, le lockfile (`uv.lock`) et les environnements virtuels en un seul outil. Installation reproductible grâce au lockfile. Rétrocompatible avec l'écosystème Python standard (`pyproject.toml` est le format PEP 621). Aucun impact sur l'architecture du code.

---

### D-006 : Encodage one-hot pour LineWorld et GridWorld (Mars 2026)

**Contexte :** Choix de la représentation de l'état pour les environnements mono-joueur simples.

**Décision :** Vecteur one-hot de la taille de la grille. La position de l'agent est le seul 1.0 dans le vecteur.

**Pourquoi :** Représentation la plus directe pour un espace d'états discret et fini. Pas de biais de proximité (contrairement à des coordonnées normalisées). Compatible avec tous les algorithmes (tabulaire, DQN, policy gradient). Pour des grilles de petite taille (5 cases, 25 cases), la dimension du vecteur reste très raisonnable.

---

### D-007 : Canal "cases vides" dans l'encodage TicTacToe (Mars 2026)

**Contexte :** L'encodage 3-canaux de TicTacToe inclut un troisième canal pour les cases vides, qui est techniquement redondant (complémentaire des deux premiers canaux).

**Décision :** Conserver le canal redondant.

**Pourquoi :** Rend l'information directement accessible au réseau sans calcul implicite. C'est la convention standard des implémentations AlphaGo/AlphaZero. Le coût en mémoire est négligeable (9 float32 supplémentaires).

---

### D-008 : HumanAgent avec pending action (Mars 2026)

**Contexte :** L'interface `Agent.act()` est synchrone, mais la GUI Pygame est event-driven. Comment un joueur humain peut-il fournir une action à travers l'interface Agent ?

**Décision :** `HumanAgent` stocke un attribut `_pending_action` que la GUI remplit via `set_action()` avant d'appeler `act()`. La boucle de jeu ne demande `act()` au HumanAgent que quand une action a été sélectionnée par l'utilisateur.

**Pourquoi :** Évite le threading ou les callbacks complexes. La GUI contrôle le flux : elle détecte `isinstance(agent, HumanAgent)` et attend l'input utilisateur avant de faire avancer le jeu. L'interface Agent n'est pas modifiée.

---

### D-009 : GUI Pygame — architecture simple sans hiérarchie de renderers (Mars 2026)

**Contexte :** 4 environnements à visualiser dans la GUI. Comment organiser le code de rendu ?

**Décision :** Un dispatch dict dans `App._draw_game()` appelle une méthode de dessin par environnement (`_draw_line_world`, `_draw_tictactoe`, etc.). Pas de classe abstraite Renderer.

**Pourquoi :** Conforme au guide d'implémentation (§8 : "No abstract renderer hierarchy"). 4 environnements seulement — une dispatch map est suffisante. Chaque méthode lit directement les attributs internes de l'environnement pour le rendu.

---

### D-010 : Input humain par environnement (Mars 2026)

**Contexte :** Chaque environnement a un espace d'actions différent. Comment mapper les clics/touches du joueur vers des actions ?

**Décision :** LineWorld/GridWorld : touches fléchées. TicTacToe : clic sur une case vide. Bobail : clic sur une pièce source puis clic sur la destination (les destinations valides sont surlignées).

**Pourquoi :** Interaction la plus naturelle pour chaque type de jeu. Le système deux-clics pour Bobail est nécessaire car chaque action encode (from_cell, to_cell).

---

### D-011 : Décroissance epsilon linéaire par step (Mars 2026)

**Contexte :** Les agents epsilon-greedy (TabularQ, DQN, DDQN) ont besoin d'une stratégie de décroissance d'epsilon. Plusieurs options : par épisode, par step, exponentielle, linéaire.

**Décision :** Décroissance linéaire par step d'entraînement (chaque appel à `act(training=True)`). Le paramètre `epsilon_decay_steps` définit le nombre de steps pour aller de `epsilon_start` à `epsilon_end`. Au-delà, epsilon reste à `epsilon_end`.

**Pourquoi :** Linéaire est le plus simple et le plus prévisible. Par step (et non par épisode) car la durée des épisodes varie selon les environnements — une décroissance par step garantit un comportement d'exploration cohérent quelle que soit la durée moyenne d'un épisode. C'est aussi l'approche utilisée dans l'implémentation originale de DQN (Mnih et al. 2015).

---

### D-012 : Masquage d'actions uniquement dans act(), pas dans le calcul des cibles (Mars 2026)

**Contexte :** L'interface `observe(state, action, reward, next_state, done)` ne fournit pas les actions légales du `next_state`. Or, pour calculer la cible Q-learning `max_a Q(s', a)`, idéalement on ne devrait maximiser que sur les actions légales en s'.

**Décision :** Le masquage d'actions (mettre les Q-values à `-inf` pour les actions illégales) est appliqué dans `act()` pour garantir qu'on ne sélectionne jamais d'action illégale (conforme à la Règle 3 du guide). En revanche, le calcul des cibles dans le replay buffer utilise `max` sur toutes les actions sans masquage.

**Pourquoi :** L'interface `observe()` ne passe pas `next_available_actions`. Stocker les actions légales avec chaque transition compliquerait le replay buffer et l'interface. En pratique, le réseau apprend à assigner de faibles Q-values aux actions qui ne mènent jamais à des récompenses. Pour nos environnements (états discrets, espaces d'actions raisonnables), cette approche fonctionne. Si nécessaire, on pourra ajouter le masquage des cibles dans une version future.

**Alternatives rejetées :**
- Modifier `observe()` pour inclure `next_available_actions` — modifie l'interface Agent, impact sur tous les agents existants et futurs
- Stocker le masque dans le replay buffer — complexifie le buffer et le code de sampling

---

### D-013 : Architecture réseau DQN/DDQN — MLP configurable (Mars 2026)

**Contexte :** Choix de l'architecture de réseau pour les agents DQN et DDQN.

**Décision :** MLP (feedforward) construit via `training/networks.py::build_mlp()`. La liste des couches cachées est un paramètre de config (`hidden_layers`). Activation ReLU par défaut. Optimiseur Adam. Perte MSE.

**Pourquoi :** Un MLP est suffisant pour des espaces d'états en vecteur plat (one-hot, encodages binaires, avec pour Bobail quelques features stratégiques additionnelles). Nos environnements ont des espaces d'états de faible dimension (5 à 80). La configurabilité via `hidden_layers` permet d'ajuster la capacité par environnement (ex: `[64, 64]` pour LineWorld, `[256, 128]` pour Bobail).

---

### D-014 : DDQN hérite de DQN (Mars 2026)

**Contexte :** DQN et DDQN partagent 95% de leur code. La seule différence est le calcul de la cible : DQN utilise `max_a Q_target(s', a)`, DDQN utilise `Q_target(s', argmax_a Q_online(s', a))`.

**Décision :** `DDQNAgent` hérite de `DQNAgent` et surcharge uniquement `_compute_targets()`. Le fichier `ddqn.py` fait ~20 lignes.

**Pourquoi :** Évite la duplication de ~120 lignes de code identique. Le guide dit que les fichiers sont "self-contained" au sens de ne pas dépendre d'utilitaires partagés — ici il s'agit d'héritage naturel (DDQN EST-UN DQN avec un calcul de cible différent), pas d'un utilitaire.

---

### D-015 : DDQN+ER distinct de DDQN malgré l'ER déjà présent (Mars 2026)

**Contexte :** Le syllabus liste séparément « DoubleDeepQLearning » (#4) et « DoubleDeepQLearningWithExperienceReplay » (#5). Or, notre DQN/DDQN utilisent déjà un `ReplayBuffer` (experience replay), conformément à l'implémentation standard (Mnih et al. 2015). DDQN+ER est donc fonctionnellement très proche du DDQN existant.

**Décision :** Créer `DDQNERAgent` comme sous-classe de `DDQNAgent` avec un paramètre `learning_starts` (nombre minimum de steps avant de commencer l'entraînement). Ce warm-up du buffer est un ajustement standard d'ER (présent dans stable-baselines3, CleanRL, etc.) qui donne à DDQN+ER une identité propre et un levier de configuration distinct.

**Pourquoi :** Le syllabus exige 16 agents distincts — DDQN+ER doit exister comme entrée séparée dans le registre. Le `learning_starts` est un ajout concret (pas juste un alias) : il permet d'accumuler des transitions diversifiées avant le premier gradient update, ce qui améliore la stabilité de l'entraînement initial. C'est documenté comme paramètre de config dans les fichiers YAML.

---

### D-016 : DDQN+PER — pondération IS et scheduling beta (Mars 2026)

**Contexte :** Le PER (Proportional Prioritized Experience Replay, Schaul et al. 2015) introduit un biais de sampling : les transitions à haute erreur TD sont échantillonnées plus souvent. Ce biais doit être corrigé pour que les gradients restent non biaisés.

**Décision :** `DDQNPERAgent` utilise `PrioritizedReplayBuffer` (sum-tree déjà implémenté dans `training/replay_buffer.py`). La perte MSE est pondérée par les importance-sampling weights : `loss = mean(w_i * (Q - target)^2)`. Le paramètre beta (contrôle de la correction IS) suit un scheduling linéaire de `per_beta_start` à `per_beta_end` sur `per_beta_steps` steps. Après chaque train step, les priorités sont mises à jour avec les erreurs TD courantes.

**Pourquoi :** La correction IS est essentielle pour la convergence — sans elle, le réseau sur-apprend les transitions à haute priorité. Le scheduling linéaire de beta (typiquement 0.4 → 1.0) est la convention de l'article original : au début, on privilégie la vitesse d'apprentissage (beta bas, correction partielle), à la fin on corrige complètement le biais pour la convergence finale.

**Paramètres de config :** `per_alpha` (0.6 par défaut), `per_beta_start` (0.4), `per_beta_end` (1.0), `per_beta_steps`, `learning_starts`.

---

### D-017 : Hiérarchie d'héritage pour la famille value-based (Mars 2026)

**Contexte :** La famille value-based comprend maintenant DQN → DDQN → DDQN+ER et DDQN+PER. L'héritage est-il toujours le bon choix ? (cf. D-014)

**Décision :** Chaîne d'héritage : `DQNAgent` → `DDQNAgent` (surcharge `_compute_targets`) → `DDQNERAgent` (surcharge `observe`, ajoute `learning_starts`) et `DDQNPERAgent` (surcharge `observe`, `_train_step`, remplace le buffer). Chaque sous-classe surcharge uniquement ce qui change.

**Pourquoi :** La factorisation reste naturelle : chaque extension ajoute une fonctionnalité précise. DDQN+PER est le plus personnalisé (buffer, loss, priorités) mais hérite quand même de l'epsilon-greedy, du `act()`, de `_compute_targets()` DDQN, et du save/load. Le code ajouté total est ~40 lignes (DDQN+ER) + ~60 lignes (DDQN+PER).

**Risque identifié :** Si l'interface de `DQNAgent._train_step()` change (ex: passage à Huber loss), il faudra aussi mettre à jour `DDQNPERAgent._train_step()`. Ce couplage est acceptable vu le nombre réduit de classes.

---

### D-018 : Chargement des modèles dans la GUI via convention `best/` + sélecteur (Mars 2026)

**Contexte :** La GUI doit instancier des agents entraînés (DQN, DDQN, etc.) pour les regarder jouer. Ces agents nécessitent des hyperparamètres (architecture réseau, lr, gamma...) pour être construits, puis un modèle sauvegardé pour être chargés. Le dossier `configs/` est réservé à l'expérimentation (sweeps, variantes) et ne constitue pas une source de vérité unique.

**Décision :** Deux mécanismes complémentaires :

1. **Convention `best/`** : `results/{env}/{agent}/best/` contient le modèle promu (`model.pt`, `config.yaml`, `source.txt`). Un script `scripts/promote_best.py` le peuple (auto par métriques ou manuel). La GUI charge `best/` par défaut.

2. **Sélecteur de modèle dans la GUI** : un row de boutons "Model (P1)" apparaît pour les agents nécessitant un modèle. Il affiche les runs entraînés disponibles (scanné depuis `results/`). L'utilisateur peut choisir "Best" ou un run spécifique (par seed). Si aucun modèle n'existe, le bouton Start est désactivé.

La source de vérité est toujours `results/` — chaque run contient son `config.yaml` (snapshot des paramètres utilisés à l'entraînement) et ses checkpoints `model_*.pt`.

**Pourquoi :** `configs/` contient des fichiers d'expérimentation (sweeps avec plusieurs valeurs d'hyperparamètres), pas un modèle canonique. Lire le `config.yaml` depuis le même dossier que le modèle garantit que l'architecture réseau correspond aux poids sauvegardés. La convention `best/` évite d'avoir à choisir parmi des dizaines de runs.

**Alternatives rejetées :**
- Lire depuis `configs/{agent}/{env}.yaml` — ne gère pas les sweeps, pas de correspondance garantie avec le modèle sauvegardé
- Auto-sélection par métriques dans la GUI — logique opinionée ("best" = highest reward?), complexité dans le code GUI
- Défauts hardcodés pour le mode inférence — fragile, chaque agent a des paramètres différents
