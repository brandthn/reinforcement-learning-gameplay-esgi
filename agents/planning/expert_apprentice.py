"""Expert Iteration (ExIt) with iterative apprentice-improves-expert loop.

Implementation inspired by:
"Thinking Fast and Slow with Deep Learning and Tree Search"
(Anthony, Tian, Barber, NeurIPS 2017; arXiv:1705.08439).
"""

import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from environments.base import Environment

from ..base import Agent
from .mcts import MCTSAgent, _Node
from training.networks import build_mlp


class NeuralMCTSAgent(MCTSAgent):
    """MCTS expert biased by apprentice priors in the UCT score.

    Score(a) = Q(a) + c * sqrt(log(N) / N_a) + w_a * pi(a|s) / (N_a + 1)
    """

    def __init__(
        self,
        env: Environment,
        n_simulations: int,
        c_uct: float,
        max_rollout_depth: int,
        gamma: float,
        policy_network: torch.nn.Module,
        action_size: int,
        device: torch.device,
        policy_bonus_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            env=env,
            n_simulations=n_simulations,
            c_uct=c_uct,
            max_rollout_depth=max_rollout_depth,
            gamma=gamma,
            **kwargs,
        )
        self._policy_network = policy_network
        self._action_size = int(action_size)
        self._device = device
        self._policy_bonus_weight = float(policy_bonus_weight)
        self._last_root_visits: dict[int, int] = {}
        self._last_root_total_visits = 0

    def get_root_visit_counts(self) -> dict[int, int]:
        """Returns n(s,a) at root from the last act() call."""
        return dict(self._last_root_visits)

    def get_root_total_visits(self) -> int:
        return int(self._last_root_total_visits)

    def act(
        self, state: np.ndarray, available_actions: list[int], training: bool = False
    ) -> int:
        if len(available_actions) == 1:
            a = available_actions[0]
            self._last_root_visits = {a: 1}
            self._last_root_total_visits = 1
            return a

        root_env = self._env.clone()
        root = _Node(
            player=root_env.current_player(),
            legal_actions=list(available_actions),
        )
        root_prior = self._compute_root_prior(state, available_actions)

        for _ in range(self._n_simulations):
            sim = root_env.clone()
            path: list[tuple[_Node, int]] = []
            node = root

            while node.fully_expanded() and not node.is_terminal:
                action = self._ucb_select_with_prior(node, root, root_prior)
                path.append((node, action))
                actor = sim.current_player()
                _, reward, done = sim.step(action)
                child = node.children[action]
                node = child
                if child.is_terminal:
                    break

            if not node.is_terminal and node.untried:
                action = random.choice(node.untried)
                node.untried.remove(action)
                path.append((node, action))
                actor = sim.current_player()
                _, reward, done = sim.step(action)
                if done:
                    child = _Node(
                        player=sim.current_player(),
                        legal_actions=[],
                        is_terminal=True,
                        terminal_reward=reward,
                        terminal_actor=actor,
                    )
                    node.children[action] = child
                    terminal_reward = reward
                    terminal_actor = actor
                else:
                    legal = sim.available_actions()
                    child = _Node(
                        player=sim.current_player(),
                        legal_actions=legal,
                    )
                    node.children[action] = child
                    terminal_reward, terminal_actor = self._rollout(sim)
            elif node.is_terminal:
                terminal_reward = node.terminal_reward
                terminal_actor = node.terminal_actor
            else:
                terminal_reward, terminal_actor = 0.0, node.player

            for pnode, paction in path:
                val = (
                    terminal_reward
                    if pnode.player == terminal_actor
                    else -terminal_reward
                )
                pnode.N[paction] += 1
                pnode.W[paction] += val

        self._last_root_visits = dict(root.N)
        self._last_root_total_visits = sum(root.N.values())
        return max(root.N, key=lambda a: root.N[a])

    def _compute_root_prior(
        self, state: np.ndarray, available_actions: list[int]
    ) -> dict[int, float]:
        with torch.no_grad():
            logits = self._policy_network(
                torch.as_tensor(
                    state, dtype=torch.float32, device=self._device
                ).unsqueeze(0)
            ).squeeze(0)
            mask = torch.full_like(logits, -1e9)
            mask[available_actions] = 0.0
            probs = torch.softmax(logits + mask, dim=0).detach().cpu().numpy()
        return {a: float(probs[a]) for a in available_actions}

    def _ucb_select_with_prior(
        self, node: _Node, root: _Node, root_prior: dict[int, float]
    ) -> int:
        total_n = node.total_visits()
        log_total = math.log(total_n) if total_n > 0 else 0.0
        best_a = None
        best_score = -float("inf")
        for a, n_a in node.N.items():
            if n_a == 0:
                return a
            q = node.W[a] / n_a
            u = self._c * math.sqrt(log_total / n_a)
            prior = root_prior.get(a, 0.0) if node is root else 0.0
            score = q + u + self._policy_bonus_weight * prior / (n_a + 1)
            if score > best_score:
                best_score = score
                best_a = a
        return best_a


class ExpertApprenticeAgent(Agent):
    """True ExIT: iterative self-play -> expert labels -> train from scratch."""

    def __init__(
        self,
        env: Environment,
        lr: float,
        hidden_layers: list[int],
        n_simulations: int,
        c_uct: float,
        max_rollout_depth: int,
        buffer_capacity: int,
        batch_size: int,
        learning_starts: int,
        train_steps_per_observe: int,
        mlp_activation: str,
        gamma: float,
        device: str = "cpu",
        max_iterations: int = 5,
        policy_bonus_weight: float = 1.0,
        val_fraction: float = 0.1,
        max_epochs: int = 30,
        early_stopping_patience: int = 5,
        min_delta: float = 1e-4,
        opening_moves_to_skip: int = 2,
        **kwargs,
    ):
        _ = train_steps_per_observe
        _ = kwargs
        self._env = env
        self._state_size = env.state_space_size()
        self._action_size = env.action_space_size()
        self._device = torch.device(device)

        self._lr = float(lr)
        self._hidden_layers = list(hidden_layers)
        self._activation = mlp_activation
        self._batch_size = int(batch_size)
        self._max_iterations = int(max_iterations)
        self._current_iteration = 0
        self._max_epochs = int(max_epochs)
        self._val_fraction = float(val_fraction)
        self._early_stopping_patience = int(early_stopping_patience)
        self._min_delta = float(min_delta)
        self._opening_moves_to_skip = int(opening_moves_to_skip)

        self._n_simulations = int(n_simulations)
        self._c_uct = float(c_uct)
        self._max_rollout_depth = int(max_rollout_depth)
        self._gamma = float(gamma)
        self._policy_bonus_weight = float(policy_bonus_weight)

        # Kept for compatibility with existing configs; in ExIT this is
        # interpreted as games-per-iteration (one sampled state per game).
        self._games_per_iteration = max(1, int(learning_starts))
        # Optional cap to avoid unbounded memory if training loop runs long.
        self._dataset_capacity = max(self._games_per_iteration, int(buffer_capacity))

        self._policy = self._build_policy()
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=self._lr)

        self._expert = self._build_expert(self._env.clone())

        # One sampled state per game from apprentice-only self-play.
        self._current_game_reservoir = None
        self._current_game_seen = 0
        self._current_game_step = 0
        self._dataset_states: list[tuple[np.ndarray, tuple[int, ...], Environment]] = []

    def _build_policy(self) -> torch.nn.Module:
        return build_mlp(
            self._state_size,
            self._action_size,
            self._hidden_layers,
            activation=self._activation,
        ).to(self._device)

    def _build_expert(self, env_snapshot: Environment) -> NeuralMCTSAgent:
        return NeuralMCTSAgent(
            env=env_snapshot,
            n_simulations=self._n_simulations,
            c_uct=self._c_uct,
            max_rollout_depth=self._max_rollout_depth,
            gamma=self._gamma,
            policy_network=self._policy,
            action_size=self._action_size,
            device=self._device,
            policy_bonus_weight=self._policy_bonus_weight,
        )

    def act(
        self, state: np.ndarray, available_actions: list[int], training: bool = False
    ) -> int:
        # During training, ExIT self-play is apprentice-only (fast policy).
        if training:
            self._maybe_sample_state_for_dataset(state, available_actions)
            self._current_game_step += 1
        return self._policy_action(state, available_actions)

    def _policy_action(self, state: np.ndarray, available_actions: list[int]) -> int:
        with torch.no_grad():
            logits = self._policy(
                torch.as_tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0)
            ).squeeze(0)
        mask = torch.full((self._action_size,), -float("inf"), device=self._device)
        mask[available_actions] = 0.0
        return int((logits + mask).argmax().item())

    def _maybe_sample_state_for_dataset(
        self, state: np.ndarray, available_actions: list[int]
    ) -> None:
        if self._current_game_step < self._opening_moves_to_skip:
            return
        self._current_game_seen += 1
        # Reservoir sampling: exactly one random post-opening state per game.
        if random.randint(1, self._current_game_seen) == 1:
            self._current_game_reservoir = (
                np.asarray(state, dtype=np.float32).copy(),
                tuple(available_actions),
                self._env.clone(),
            )

    def observe(self, state, action, reward, next_state, done):
        _ = (state, action, reward, next_state, done)

    def end_episode(self):
        if (
            self._current_game_reservoir is not None
            and len(self._dataset_states) < self._dataset_capacity
        ):
            self._dataset_states.append(self._current_game_reservoir)
        self._current_game_reservoir = None
        self._current_game_seen = 0
        self._current_game_step = 0

        if (
            self._current_iteration < self._max_iterations
            and len(self._dataset_states) >= self._games_per_iteration
        ):
            self._run_single_iteration()

    def run_exit(self) -> None:
        """Runs complete ExIT iterations based on currently collected states."""
        while (
            self._current_iteration < self._max_iterations
            and len(self._dataset_states) >= self._games_per_iteration
        ):
            self._run_single_iteration()

    def _run_single_iteration(self) -> None:
        # D_i: one random state per game, no replay across iterations.
        dataset_i = self._dataset_states[: self._games_per_iteration]
        self._dataset_states = []
        labeled = self._label_dataset_with_expert(dataset_i)
        if not labeled:
            return
        self._reinitialize_apprentice()
        self._train_apprentice_on_tpt(labeled)
        self._current_iteration += 1
        self._expert = self._build_expert(self._env.clone())

    def _label_dataset_with_expert(
        self,
        dataset: list[tuple[np.ndarray, tuple[int, ...], Environment]],
    ) -> list[tuple[np.ndarray, np.ndarray, tuple[int, ...]]]:
        labeled: list[tuple[np.ndarray, np.ndarray, tuple[int, ...]]] = []
        for state, legal, env_snapshot in dataset:
            expert = self._build_expert(env_snapshot.clone())
            expert.act(state, list(legal), training=False)
            root_counts = expert.get_root_visit_counts()
            total = expert.get_root_total_visits()
            if total <= 0:
                continue
            # Tree-Policy Target (TPT): pi^target(a|s) = n(s,a)/n(s)
            target = np.zeros(self._action_size, dtype=np.float32)
            for a, n_a in root_counts.items():
                target[a] = float(n_a) / float(total)
            labeled.append((state, target, legal))
        return labeled

    def _reinitialize_apprentice(self) -> None:
        self._policy = self._build_policy()
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=self._lr)

    def _train_apprentice_on_tpt(
        self,
        labeled: list[tuple[np.ndarray, np.ndarray, tuple[int, ...]]],
    ) -> None:
        if len(labeled) == 1:
            train_data = labeled
            val_data = labeled
        else:
            idx = np.arange(len(labeled))
            np.random.shuffle(idx)
            val_size = max(1, int(len(labeled) * self._val_fraction))
            if val_size >= len(labeled):
                val_size = len(labeled) - 1
            val_idx = set(idx[:val_size].tolist())
            train_data = [labeled[i] for i in range(len(labeled)) if i not in val_idx]
            val_data = [labeled[i] for i in range(len(labeled)) if i in val_idx]

        best_state = None
        best_val = float("inf")
        patience_left = self._early_stopping_patience

        for _ in range(self._max_epochs):
            self._policy.train()
            random.shuffle(train_data)
            for start in range(0, len(train_data), self._batch_size):
                batch = train_data[start : start + self._batch_size]
                loss = self._batch_kl_loss(batch)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            val_loss = self._evaluate_dataset_loss(val_data)
            if val_loss + self._min_delta < best_val:
                best_val = val_loss
                best_state = {
                    k: v.detach().clone() for k, v in self._policy.state_dict().items()
                }
                patience_left = self._early_stopping_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self._policy.load_state_dict(best_state)

    def _batch_kl_loss(
        self,
        batch: list[tuple[np.ndarray, np.ndarray, tuple[int, ...]]],
    ) -> torch.Tensor:
        states = torch.as_tensor(
            np.stack([b[0] for b in batch]),
            dtype=torch.float32,
            device=self._device,
        )
        targets = torch.as_tensor(
            np.stack([b[1] for b in batch]),
            dtype=torch.float32,
            device=self._device,
        )
        legal_per_sample = [list(b[2]) for b in batch]
        logits = self._policy(states)
        masked = logits.clone()
        for i, legal in enumerate(legal_per_sample):
            m = torch.full_like(masked[i], -1e9)
            m[legal] = 0.0
            masked[i] = masked[i] + m
        log_probs = torch.log_softmax(masked, dim=1)
        # KL(p_target || p_model) up to additive const = cross-entropy with soft targets.
        return -(targets * log_probs).sum(dim=1).mean()

    def _evaluate_dataset_loss(
        self,
        dataset: list[tuple[np.ndarray, np.ndarray, tuple[int, ...]]],
    ) -> float:
        self._policy.eval()
        losses = []
        with torch.no_grad():
            for start in range(0, len(dataset), self._batch_size):
                batch = dataset[start : start + self._batch_size]
                losses.append(float(self._batch_kl_loss(batch).item()))
        return float(np.mean(losses)) if losses else 0.0

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy": self._policy.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "iteration": self._current_iteration,
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self._device, weights_only=False)
        self._policy.load_state_dict(data["policy"])
        self._optimizer.load_state_dict(data["optimizer"])
        self._current_iteration = int(data.get("iteration", 0))
        self._expert = self._build_expert(self._env.clone())
