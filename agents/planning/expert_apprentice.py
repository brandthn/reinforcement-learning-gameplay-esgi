"""Expert Iteration (ExIt) : MCTS comme expert, reseau comme apprenti.

Inspire d'Anthony, Tian, Barber (NeurIPS 2017), arXiv:1705.08439
("Thinking Fast and Slow with Deep Learning and Tree Search") :
l'arbre UCT fournit des etiquettes (s, a_expert), l'apprenti minimise la
cross-entropie sur les actions legales ; en inference (training=False),
seul l'apprenti est utilise (pensee rapide).

Reference PDF : docs/ExpertApprentice_1705.08439v4.pdf
"""

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from environments.base import Environment

from ..base import Agent
from .mcts import MCTSAgent
from training.networks import build_mlp


class ExpertApprenticeAgent(Agent):
    """Apprenti policy (MLP) entraine par imitation de MCTS a la racine."""

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
        **kwargs,
    ):
        _ = kwargs
        self._env = env
        self._state_size = env.state_space_size()
        self._action_size = env.action_space_size()
        self._device = torch.device(device)
        self._batch_size = int(batch_size)
        self._learning_starts = int(learning_starts)
        self._train_steps_per_observe = int(train_steps_per_observe)

        self._expert = MCTSAgent(
            env,
            n_simulations=int(n_simulations),
            c_uct=float(c_uct),
            max_rollout_depth=int(max_rollout_depth),
            gamma=float(gamma),
        )

        self._policy = build_mlp(
            self._state_size,
            self._action_size,
            hidden_layers,
            activation=mlp_activation,
        ).to(self._device)
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=float(lr))

        self._buffer: deque[tuple[np.ndarray, int, tuple[int, ...]]] = deque(
            maxlen=int(buffer_capacity)
        )
        self._train_steps = 0

    def act(self, state: np.ndarray, available_actions: list[int],
            training: bool = False) -> int:
        if training:
            expert_action = self._expert.act(
                state, available_actions, training=False
            )
            self._buffer.append(
                (
                    np.asarray(state, dtype=np.float32).copy(),
                    int(expert_action),
                    tuple(available_actions),
                )
            )
            return expert_action

        with torch.no_grad():
            logits = self._policy(
                torch.as_tensor(
                    state, dtype=torch.float32, device=self._device
                ).unsqueeze(0)
            ).squeeze(0)
        mask = torch.full(
            (self._action_size,), -float("inf"), device=self._device
        )
        mask[available_actions] = 0.0
        return int((logits + mask).argmax().item())

    def observe(self, state, action, reward, next_state, done):
        if len(self._buffer) < self._learning_starts:
            return
        if len(self._buffer) < self._batch_size:
            return
        for _ in range(self._train_steps_per_observe):
            self._train_step()

    def end_episode(self):
        pass

    def _train_step(self) -> None:
        batch = random_sample(self._buffer, self._batch_size)
        states = torch.as_tensor(
            np.stack([b[0] for b in batch]),
            dtype=torch.float32,
            device=self._device,
        )
        actions = torch.as_tensor(
            [b[1] for b in batch], dtype=torch.long, device=self._device
        )
        avail = [list(b[2]) for b in batch]

        logits = self._policy(states)
        masked = logits.clone()
        for i, legal in enumerate(avail):
            row = masked[i]
            m = torch.full_like(row, -1e9)
            m[legal] = 0.0
            masked[i] = row + m

        loss = F.cross_entropy(masked, actions)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._train_steps += 1

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy": self._policy.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "train_steps": self._train_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self._device, weights_only=False)
        self._policy.load_state_dict(data["policy"])
        self._optimizer.load_state_dict(data["optimizer"])
        self._train_steps = int(data.get("train_steps", 0))


def random_sample(
    buf: deque[tuple[np.ndarray, int, tuple[int, ...]]], k: int
) -> list[tuple[np.ndarray, int, tuple[int, ...]]]:
    """Echantillonnage uniforme avec remise dans un deque."""
    n = len(buf)
    if n == 0:
        return []
    idx = np.random.randint(0, n, size=k)
    items = list(buf)
    return [items[i] for i in idx]
