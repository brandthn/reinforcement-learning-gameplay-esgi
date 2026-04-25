# agents/planning/expert_apprentice.py
"""
Implémentation de l'algorithme Expert Iteration (EXIT).
Référence : Anthony et al., "Thinking Fast and Slow with Deep Learning and Tree Search", NeurIPS 2017.

Principe : alterne entre deux étapes —
  1. Expert (MCTS guidé par l'apprenti) génère des coups forts → dataset
  2. Apprenti (réseau de neurones) apprend à imiter l'expert → améliore le MCTS

L'apprenti est un réseau qui produit une distribution sur les actions (policy head)
et optionnellement une estimation de valeur (value head).
"""

import math
import random
import time
import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agents.base import Agent


# ---------------------------------------------------------------------------
# Réseau apprenti — policy head + optional value head
# ---------------------------------------------------------------------------

class ApprenticeNetwork(nn.Module):
    """
    MLP avec une tête de politique (softmax sur les actions)
    et une tête de valeur optionnelle (sigmoid → [0,1]).
    """

    def __init__(self, state_size: int, action_size: int, hidden_layers: list[int],
                 use_value_head: bool = True):
        super().__init__()
        self.use_value_head = use_value_head

        # Tronc partagé
        layers = []
        in_dim = state_size
        for h in hidden_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        # Tête politique
        self.policy_head = nn.Linear(in_dim, action_size)

        # Tête valeur (optionnelle)
        if use_value_head:
            self.value_head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        Retourne (logits_politique, valeur_scalaire_ou_None).
        Les logits ne sont PAS masqués ici ; le masquage se fait dans act().
        """
        features = self.trunk(x)
        policy_logits = self.policy_head(features)
        value = None
        if self.use_value_head:
            value = torch.sigmoid(self.value_head(features))  # prob de victoire
        return policy_logits, value


# ---------------------------------------------------------------------------
# Nœud MCTS
# ---------------------------------------------------------------------------

class MCTSNode:
    """Nœud de l'arbre de recherche Monte Carlo."""

    __slots__ = ("state", "parent", "action_from_parent", "children",
                 "n_visits", "total_value", "prior", "available_actions",
                 "is_terminal", "terminal_value")

    def __init__(self, state: np.ndarray, available_actions: list[int],
                 parent=None, action_from_parent: int = None, prior: float = 1.0,
                 is_terminal: bool = False, terminal_value: float = 0.0):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children: dict[int, "MCTSNode"] = {}
        self.n_visits = 0
        self.total_value = 0.0
        self.prior = prior                     # prior de l'apprenti π(a|s)
        self.available_actions = available_actions
        self.is_terminal = is_terminal
        self.terminal_value = terminal_value   # valeur connue si terminal

    @property
    def q_value(self) -> float:
        """Valeur moyenne estimée de ce nœud."""
        if self.n_visits == 0:
            return 0.0
        return self.total_value / self.n_visits

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(self.available_actions)

    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ---------------------------------------------------------------------------
# MCTS guidé par le réseau apprenti
# ---------------------------------------------------------------------------

class NeuralMCTS:
    """
    MCTS avec prior neural network (Policy Network) et estimation de valeur optionnelle.
    Formule UCT modifiée (style PUCT d'AlphaGo) :
        PUCT(s, a) = Q(s,a) + c_puct * π(a|s) * sqrt(N(s)) / (1 + N(s,a))
    """

    def __init__(self, network: ApprenticeNetwork, num_simulations: int,
                 c_puct: float, device: torch.device, use_value_head: bool = True,
                 discount: float = 1.0):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        self.use_value_head = use_value_head
        self.discount = discount  # pour les environnements avec rewards intermédiaires

    @torch.no_grad()
    def _evaluate_network(self, state: np.ndarray,
                          available_actions: list[int]) -> tuple[dict[int, float], float]:
        """
        Passe l'état dans le réseau.
        Retourne (priors_masqués, valeur_estimée).
        Les priors sont normalisés sur les actions légales uniquement.
        """
        x = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, value_tensor = self.network(x)
        logits = logits.squeeze(0)

        # Masquer les actions illégales → -inf
        mask = torch.full((logits.shape[0],), float('-inf'), device=self.device)
        for a in available_actions:
            mask[a] = logits[a]

        priors = F.softmax(mask, dim=0).cpu().numpy()
        prior_dict = {a: float(priors[a]) for a in available_actions}

        value = 0.5  # valeur neutre par défaut
        if self.use_value_head and value_tensor is not None:
            value = float(value_tensor.squeeze())

        return prior_dict, value

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Descend dans l'arbre en choisissant l'action maximisant PUCT."""
        while not node.is_leaf() and not node.is_terminal:
            best_score = -float('inf')
            best_child = None
            sqrt_n = math.sqrt(node.n_visits + 1)

            for action, child in node.children.items():
                # PUCT
                exploit = child.q_value
                explore = self.c_puct * child.prior * sqrt_n / (1 + child.n_visits)
                score = exploit + explore
                if score > best_score:
                    best_score = score
                    best_child = child

            node = best_child
        return node

    def _expand(self, node: MCTSNode, env_clone) -> float:
        """
        Expand un nœud feuille non terminal.
        Utilise le réseau pour obtenir les priors et la valeur.
        Ajoute tous les enfants légaux.
        Retourne la valeur estimée du nœud.
        """
        if node.is_terminal:
            return node.terminal_value

        priors, value = self._evaluate_network(node.state, node.available_actions)

        for action in node.available_actions:
            env_child = env_clone.clone()
            next_state, reward, done = env_child.step(action)

            if done:
                # Nœud terminal : la valeur est le reward réel
                child = MCTSNode(
                    state=next_state,
                    available_actions=[],
                    parent=node,
                    action_from_parent=action,
                    prior=priors.get(action, 0.0),
                    is_terminal=True,
                    terminal_value=reward,
                )
            else:
                child = MCTSNode(
                    state=next_state,
                    available_actions=env_child.available_actions(),
                    parent=node,
                    action_from_parent=action,
                    prior=priors.get(action, 0.0),
                )
            node.children[action] = child

        return value

    def _backup(self, node: MCTSNode, value: float):
        """
        Remonte la valeur depuis le nœud feuille jusqu'à la racine.
        Pour les jeux à deux joueurs, on inverse la valeur à chaque niveau
        (convention : valeur depuis la perspective du joueur courant).
        """
        current = node
        while current is not None:
            current.n_visits += 1
            current.total_value += value
            value = -value  # inversion pour jeux adversariaux
            current = current.parent

    def search(self, env) -> tuple[dict[int, int], np.ndarray]:
        """
        Lance `num_simulations` simulations depuis l'état courant de l'env.
        Retourne :
          - visit_counts : {action → nombre de visites}
          - tree_policy_target : distribution normalisée sur les actions légales (TPT)
        """
        state = env.state_description()
        available = env.available_actions()

        priors_root, _ = self._evaluate_network(state, available)

        root = MCTSNode(
            state=state,
            available_actions=available,
            prior=1.0,
        )
        # Initialise les enfants de la racine avec les priors
        for action in available:
            env_child = env.clone()
            next_state, reward, done = env_child.step(action)
            if done:
                child = MCTSNode(
                    state=next_state, available_actions=[],
                    parent=root, action_from_parent=action,
                    prior=priors_root.get(action, 0.0),
                    is_terminal=True, terminal_value=reward,
                )
            else:
                child = MCTSNode(
                    state=next_state,
                    available_actions=env_child.available_actions(),
                    parent=root, action_from_parent=action,
                    prior=priors_root.get(action, 0.0),
                )
            root.children[action] = child

        for _ in range(self.num_simulations):
            leaf = self._select(root)

            if leaf.is_terminal:
                value = leaf.terminal_value
            elif leaf.n_visits == 0:
                # Évaluation par le réseau (pas d'expansion complète pour économiser la mémoire)
                _, value = self._evaluate_network(leaf.state, leaf.available_actions)
            else:
                # Expansion puis évaluation
                env_clone = env.clone()
                # Rejouer les actions depuis la racine jusqu'à la feuille
                # (on utilise le state stocké directement → pas de replay nécessaire)
                value = self._expand(leaf, env.clone())

            self._backup(leaf, value)

        # Calcule les statistiques de visite
        visit_counts = {a: root.children[a].n_visits for a in available}
        total_visits = sum(visit_counts.values()) or 1

        tpt = np.zeros(env.action_space_size(), dtype=np.float32)
        for a, n in visit_counts.items():
            tpt[a] = n / total_visits

        return visit_counts, tpt

    def best_action(self, env, temperature: float = 0.0) -> int:
        """
        Choisit le meilleur coup selon les visites MCTS.
        temperature=0 → coup le plus visité (greedy).
        temperature>0 → échantillonnage proportionnel à n^(1/T).
        """
        visit_counts, _ = self.search(env)
        if temperature == 0.0:
            return max(visit_counts, key=visit_counts.get)

        actions = list(visit_counts.keys())
        counts = np.array([visit_counts[a] for a in actions], dtype=np.float64)
        counts = counts ** (1.0 / temperature)
        counts /= counts.sum()
        return int(np.random.choice(actions, p=counts))


# ---------------------------------------------------------------------------
# Buffer d'expérience pour l'apprentissage imitation
# ---------------------------------------------------------------------------

class ImitationBuffer:
    """
    Stocke les paires (état, tree_policy_target, valeur_MC) générées par l'expert.
    Implémente le dataset aggregation (style online EXIT / DAgger).
    """

    def __init__(self, max_size: int):
        self.buffer: deque = deque(maxlen=max_size)

    def add(self, state: np.ndarray, tpt: np.ndarray, value: float):
        self.buffer.append((state.copy(), tpt.copy(), float(value)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, tpts, values = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(tpts, dtype=np.float32),
            np.array(values, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Agent Expert Apprentice (EXIT)
# ---------------------------------------------------------------------------

class ExpertApprenticeAgent(Agent):
    """
    Agent Expert Iteration (EXIT) — Anthony et al. 2017.

    Cycle d'entraînement :
      1. Joue `episodes_per_iter` parties avec le MCTS (expert) guidé par l'apprenti.
         Pour chaque état rencontré, stocke (état, TPT, valeur_MC).
      2. Entraîne le réseau apprenti sur le buffer accumulé (imitation learning).
      3. Le nouvel apprenti améliore les priors du MCTS → itère.

    En inférence (`training=False`) : utilise le MCTS avec le réseau courant.
    En mode rapide (`act()` sans training) : utilise directement le réseau (fast policy).
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers: list[int],
        num_simulations: int,
        c_puct: float,
        lr: float,
        batch_size: int,
        buffer_size: int,
        train_epochs_per_iter: int,
        episodes_per_iter: int,
        use_value_head: bool = True,
        temperature_train: float = 1.0,
        temperature_eval: float = 0.0,
        device: str = "cpu",
    ):
        """
        Paramètres
        ----------
        state_size            : taille du vecteur d'état
        action_size           : nombre total d'actions (y compris illégales)
        hidden_layers         : liste des tailles des couches cachées du MLP
        num_simulations       : nombre de simulations MCTS par coup
        c_puct                : constante d'exploration PUCT
        lr                    : taux d'apprentissage Adam
        batch_size            : taille du mini-batch pour l'entraînement
        buffer_size           : taille max du buffer d'imitation (dataset aggregation)
        train_epochs_per_iter : passes sur les données à chaque itération EXIT
        episodes_per_iter     : parties générées par l'expert à chaque itération EXIT
        use_value_head        : entraîner aussi une tête de valeur
        temperature_train     : température pour le choix de coup lors de la génération
        temperature_eval      : température lors de l'évaluation (0 = greedy)
        device                : "cpu" ou "cuda"
        """
        self.state_size = state_size
        self.action_size = action_size
        self.use_value_head = use_value_head
        self.batch_size = batch_size
        self.train_epochs_per_iter = train_epochs_per_iter
        self.episodes_per_iter = episodes_per_iter
        self.temperature_train = temperature_train
        self.temperature_eval = temperature_eval

        self.device = torch.device(device)

        # Réseau apprenti
        self.network = ApprenticeNetwork(
            state_size, action_size, hidden_layers, use_value_head
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Expert MCTS
        self.mcts = NeuralMCTS(
            network=self.network,
            num_simulations=num_simulations,
            c_puct=c_puct,
            device=self.device,
            use_value_head=use_value_head,
        )

        # Buffer d'imitation (dataset aggregation)
        self.buffer = ImitationBuffer(max_size=buffer_size)

        # Statistiques d'entraînement
        self.training_losses: list[float] = []
        self.iteration: int = 0

    # ------------------------------------------------------------------
    # Interface Agent
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray, available_actions: list[int],
            training: bool = False) -> int:
        """
        En inférence (training=False) : utilise directement le réseau (rapide).
        En entraînement (training=True) : aussi le réseau rapide — le MCTS est
        appelé explicitement dans `run_exit_iteration()`.
        """
        return self._fast_act(state, available_actions)

    @torch.no_grad()
    def _fast_act(self, state: np.ndarray, available_actions: list[int]) -> int:
        """Sélectionne l'action via le réseau seul (pas de MCTS)."""
        x = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, _ = self.network(x)
        logits = logits.squeeze(0)

        # Masquer les actions illégales
        mask = torch.full((self.action_size,), float('-inf'), device=self.device)
        for a in available_actions:
            mask[a] = logits[a]

        probs = F.softmax(mask, dim=0).cpu().numpy()
        return int(np.argmax(probs))

    def act_with_mcts(self, env, temperature: float = None) -> int:
        """Sélectionne l'action via MCTS (expert). Utilisé pendant la génération de données."""
        t = self.temperature_train if temperature is None else temperature
        return self.mcts.best_action(env, temperature=t)

    # ------------------------------------------------------------------
    # Boucle EXIT principale
    # ------------------------------------------------------------------

    def run_exit_iteration(self, env, opponent_agent=None) -> dict:
        """
        Effectue une itération EXIT complète :
          1. Génère `episodes_per_iter` parties avec le MCTS → alimente le buffer.
          2. Entraîne le réseau sur le buffer.

        Paramètres
        ----------
        env            : environnement (doit implémenter clone())
        opponent_agent : agent adverse pour jeux à deux joueurs.
                         None pour jeux à un joueur.

        Retourne un dict de métriques de l'itération.
        """
        self.iteration += 1

        # --- Phase 1 : génération de données expert ---
        episode_rewards = []
        episode_lengths = []

        for _ in range(self.episodes_per_iter):
            reward, length = self._generate_expert_episode(env, opponent_agent)
            episode_rewards.append(reward)
            episode_lengths.append(length)

        # --- Phase 2 : entraînement de l'apprenti ---
        losses = self._train_apprentice()

        metrics = {
            "iteration": self.iteration,
            "mean_episode_reward": float(np.mean(episode_rewards)),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "buffer_size": len(self.buffer),
        }
        return metrics

    def _generate_expert_episode(self, env, opponent_agent=None) -> tuple[float, int]:
        """
        Joue une partie complète avec le MCTS expert.
        Collecte (état, TPT, valeur_MC) pour chaque état visité par l'agent apprenti.
        """
        env_episode = env.clone()
        state = env_episode.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        # Historique pour le calcul de la valeur MC : liste de (état, tpt)
        trajectory: list[tuple[np.ndarray, np.ndarray, int]] = []

        is_two_player = env_episode.is_adversarial() if hasattr(env_episode, 'is_adversarial') else False

        while not done:
            available = env_episode.available_actions()
            if not available:
                break

            # Détermine si c'est le tour de l'agent apprenti
            current_player = env_episode.current_player() if hasattr(env_episode, 'current_player') else 0
            is_agent_turn = (not is_two_player) or (current_player == 0)

            if is_agent_turn:
                # Expert : MCTS → collecte (état, TPT)
                _, tpt = self.mcts.search(env_episode)
                trajectory.append((state.copy(), tpt, current_player))

                # Choisit le coup selon la distribution de visite
                action = self.mcts.best_action(env_episode, temperature=self.temperature_train)
            else:
                # Adversaire
                if opponent_agent is not None:
                    action = opponent_agent.act(state, available, training=False)
                else:
                    action = random.choice(available)

            next_state, reward, done = env_episode.step(action)
            total_reward += reward
            state = next_state
            step_count += 1

        # Calcule les valeurs MC et alimente le buffer
        # Convention : +1 = victoire pour l'agent (player 0), -1 = défaite
        final_reward = total_reward  # dans les jeux simples c'est suffisant

        for traj_state, tpt, player in trajectory:
            # Valeur MC depuis la perspective du joueur qui a joué dans cet état
            mc_value = final_reward if (not is_two_player or player == 0) else -final_reward
            # Normalise dans [0, 1] pour la tête sigmoid
            value_target = (mc_value + 1.0) / 2.0
            self.buffer.add(traj_state, tpt, value_target)

        return total_reward, step_count

    def _train_apprentice(self) -> list[float]:
        """
        Entraîne le réseau sur le buffer d'imitation.
        Pertes :
          - L_policy = KL(TPT || π_réseau)  [tree-policy targets]
          - L_value  = BCE(valeur_réseau, valeur_MC)  [si value head]
        """
        if len(self.buffer) < self.batch_size:
            return []

        self.network.train()
        losses = []

        # Calcule le nombre de batches pour couvrir `train_epochs_per_iter` passes
        n_samples = len(self.buffer)
        steps_per_epoch = max(1, n_samples // self.batch_size)
        total_steps = self.train_epochs_per_iter * steps_per_epoch

        for _ in range(total_steps):
            states, tpts, values = self.buffer.sample(self.batch_size)

            states_t = torch.FloatTensor(states).to(self.device)
            tpts_t = torch.FloatTensor(tpts).to(self.device)
            values_t = torch.FloatTensor(values).unsqueeze(1).to(self.device)

            logits, value_pred = self.network(states_t)

            # Perte politique : KL(TPT || softmax(logits))
            # = -sum_a TPT(a) * log(softmax(logits)[a])  [cross-entropy pondérée]
            log_probs = F.log_softmax(logits, dim=1)
            loss_policy = -(tpts_t * log_probs).sum(dim=1).mean()

            loss = loss_policy

            # Perte valeur : BCE
            if self.use_value_head and value_pred is not None:
                loss_value = F.binary_cross_entropy(value_pred, values_t)
                loss = loss_policy + loss_value

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            losses.append(float(loss.item()))

        self.network.eval()
        self.training_losses.extend(losses)
        return losses

    # ------------------------------------------------------------------
    # Sauvegarde / chargement
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Sauvegarde le réseau et les statistiques d'entraînement."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "training_losses": self.training_losses,
        }, path)

    def load(self, path: str) -> None:
        """Charge le réseau depuis un checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.iteration = checkpoint.get("iteration", 0)
        self.training_losses = checkpoint.get("training_losses", [])
        self.network.eval()

    @property
    def name(self) -> str:
        return "ExpertApprentice"


# ---------------------------------------------------------------------------
# Script de démonstration rapide (smoke test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Test de fumée minimal : vérifie que l'agent instancie, génère un coup,
    effectue une itération EXIT sur un environnement TicTacToe minimal.

    Pour un vrai entraînement, utiliser scripts/train.py avec un config YAML.
    """

    # Environnement TicTacToe simplifié pour le test
    class _MinimalTicTacToe:
        """TicTacToe 3x3 minimal pour test standalone."""

        def __init__(self):
            self._board = None
            self._current_player = 0
            self._done = False

        def reset(self):
            self._board = np.zeros(9, dtype=np.float32)
            self._current_player = 0
            self._done = False
            return self.state_description()

        def state_description(self):
            # 9 cases : +1 joueur courant, -1 adversaire, 0 vide
            view = np.zeros(9, dtype=np.float32)
            for i in range(9):
                if self._board[i] == self._current_player + 1:
                    view[i] = 1.0
                elif self._board[i] != 0:
                    view[i] = -1.0
            return view

        def available_actions(self):
            return [i for i in range(9) if self._board[i] == 0]

        def step(self, action: int):
            assert self._board[action] == 0
            self._board[action] = self._current_player + 1
            winner = self._check_winner()
            if winner is not None:
                reward = 1.0 if winner == self._current_player + 1 else -1.0
                self._done = True
                self._current_player = 1 - self._current_player
                return self.state_description(), reward, True
            if not self.available_actions():
                self._done = True
                return self.state_description(), 0.0, True
            self._current_player = 1 - self._current_player
            return self.state_description(), 0.0, False

        def _check_winner(self):
            b = self._board
            lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
            for l in lines:
                if b[l[0]] == b[l[1]] == b[l[2]] != 0:
                    return b[l[0]]
            return None

        def action_space_size(self): return 9
        def state_space_size(self): return 9
        def is_adversarial(self): return True
        def current_player(self): return self._current_player
        def clone(self): return copy.deepcopy(self)

    # --- Instanciation ---
    env = _MinimalTicTacToe()
    env.reset()

    agent = ExpertApprenticeAgent(
        state_size=9,
        action_size=9,
        hidden_layers=[64, 64],
        num_simulations=50,        # réduit pour le test rapide
        c_puct=1.0,
        lr=1e-3,
        batch_size=32,
        buffer_size=5000,
        train_epochs_per_iter=2,
        episodes_per_iter=10,      # réduit pour le test
        use_value_head=True,
        temperature_train=1.0,
        temperature_eval=0.0,
        device="cpu",
    )

    # --- Smoke test : act() ---
    state = env.reset()
    available = env.available_actions()
    action = agent.act(state, available)
    assert action in available, f"Action {action} hors des actions légales {available}"
    print(f"[OK] act() → action {action} (parmi {available})")

    # --- Smoke test : une itération EXIT ---
    print("Lancement d'une itération EXIT (10 parties, 50 simulations MCTS)…")
    t0 = time.time()
    metrics = agent.run_exit_iteration(env)
    elapsed = time.time() - t0
    print(f"[OK] Itération EXIT terminée en {elapsed:.1f}s")
    print(f"     Métriques : {metrics}")

    # --- Smoke test : save/load ---
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "exit_test.pt")
        agent.save(path)
        agent.load(path)
    print("[OK] save() / load() fonctionnels")

    print("\nTous les tests sont passés.")
