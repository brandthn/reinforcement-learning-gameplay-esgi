"""Shared base class for REINFORCE-style policy-gradient agents."""

from typing import Sequence

import numpy as np
import torch
import torch.optim as optim

from agents.base import Agent
from training.networks import build_mlp


class BaseREINFORCEAgent(Agent):
    """Common episodic storage and return computation for REINFORCE variants."""

    def __init__(self, state_size: int, action_size: int,
                 hidden_layers: Sequence[int] = (64, 64), lr: float = 1e-3,
                 gamma: float = 0.99):
        self._state_size = state_size
        self._action_size = action_size
        self._gamma = gamma
        self._device = torch.device("cpu")

        self._policy = build_mlp(
            state_size, action_size, list(hidden_layers)
        ).to(self._device)
        self._optimizer = optim.Adam(self._policy.parameters(), lr=lr)

        self._states = []
        self._actions = []
        self._rewards = []

    def act(self, state: np.ndarray, available_actions: list[int],
            training: bool = False) -> int:
        state_t = torch.from_numpy(np.asarray(state)).float().to(
            self._device
        ).unsqueeze(0)
        logits = self._policy(state_t).squeeze(0)

        # Mask illegal actions so they are never sampled/selected.
        if available_actions is not None:
            mask = torch.full_like(logits, float("-inf"))
            mask[available_actions] = 0.0
            logits = logits + mask

        probs = torch.softmax(logits, dim=-1)
        if training:
            dist = torch.distributions.Categorical(probs)
            return int(dist.sample().item())
        return int(torch.argmax(probs).item())

    def observe(self, state, action, reward, next_state, done):
        self._states.append(np.asarray(state))
        self._actions.append(int(action))
        self._rewards.append(float(reward))

    def end_episode(self):
        if not self._rewards:
            return

        returns = []
        ret = 0.0
        for reward in reversed(self._rewards):
            ret = reward + self._gamma * ret
            returns.insert(0, ret)

        returns_t = torch.tensor(
            returns, dtype=torch.float32, device=self._device
        )
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        self._update_policy(self._states, self._actions, returns_t)

        self._states.clear()
        self._actions.clear()
        self._rewards.clear()

    def _update_policy(self, states, actions, returns: torch.Tensor):
        raise NotImplementedError()

    def _log_prob(self, state, action) -> torch.Tensor:
        state_t = torch.from_numpy(np.asarray(state)).float().to(
            self._device
        ).unsqueeze(0)
        logits = self._policy(state_t).squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(torch.tensor(action, device=self._device))

    def save(self, path: str) -> None:
        torch.save({"policy_state": self._policy.state_dict()}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self._device, weights_only=False)
        self._policy.load_state_dict(data["policy_state"])
