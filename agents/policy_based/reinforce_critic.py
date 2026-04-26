"""REINFORCE agent with a learned critic baseline."""

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.policy_based.base_reinforce import BaseREINFORCEAgent
from training.networks import build_mlp


class REINFORCECriticAgent(BaseREINFORCEAgent):
    """Updates a value network baseline (critic) at each episode end."""

    def __init__(self, state_size: int, action_size: int,
                 hidden_layers: Sequence[int] = (64, 64), lr: float = 1e-3,
                 gamma: float = 0.99, critic_lr: float = 1e-3):
        super().__init__(state_size, action_size, hidden_layers, lr, gamma)
        self._value_net = build_mlp(
            state_size, 1, list(hidden_layers)
        ).to(self._device)
        self._critic_optimizer = optim.Adam(
            self._value_net.parameters(), lr=critic_lr
        )

    def _update_policy(self, states, actions, returns):
        self._critic_optimizer.zero_grad()

        value_preds = []
        for state in states:
            state_t = torch.from_numpy(np.asarray(state)).float().to(
                self._device
            ).unsqueeze(0)
            value = self._value_net(state_t).squeeze(0)
            value_preds.append(value)

        value_preds_t = torch.stack(value_preds).squeeze(-1)
        value_loss = F.mse_loss(value_preds_t, returns)
        value_loss.backward()
        self._critic_optimizer.step()

        advantages = returns - value_preds_t.detach()

        self._optimizer.zero_grad()
        policy_loss = 0.0
        for state, action, advantage in zip(states, actions, advantages):
            log_prob = self._log_prob(state, action)
            policy_loss = policy_loss - log_prob * advantage

        policy_loss = policy_loss / len(advantages)
        policy_loss.backward()
        self._optimizer.step()

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_state": self._policy.state_dict(),
                "value_state": self._value_net.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self._device, weights_only=False)
        self._policy.load_state_dict(data["policy_state"])
        if "value_state" in data:
            self._value_net.load_state_dict(data["value_state"])
