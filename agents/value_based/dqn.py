"""Deep Q-Network (DQN) agent."""

import random

import numpy as np
import torch
import torch.nn.functional as F

from agents.base import Agent
from training.networks import build_mlp
from training.replay_buffer import ReplayBuffer


class DQNAgent(Agent):
    """DQN with target network and experience replay.

    Action masking in act() ensures only legal actions are selected.
    Target computation uses max over all actions (no masking).
    """

    def __init__(self, state_size: int, action_size: int,
                 lr: float, gamma: float,
                 epsilon_start: float, epsilon_end: float,
                 epsilon_decay_steps: int,
                 hidden_layers: list[int],
                 batch_size: int, buffer_capacity: int,
                 target_update_freq: int):
        self._state_size = state_size
        self._action_size = action_size
        self._gamma = gamma
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay_steps = epsilon_decay_steps
        self._batch_size = batch_size
        self._target_update_freq = target_update_freq

        self._online_net = build_mlp(state_size, action_size, hidden_layers)
        self._target_net = build_mlp(state_size, action_size, hidden_layers)
        self._target_net.load_state_dict(self._online_net.state_dict())

        self._optimizer = torch.optim.Adam(self._online_net.parameters(), lr=lr)
        self._buffer = ReplayBuffer(buffer_capacity)

        self._step_count = 0
        self._update_count = 0

    def _get_epsilon(self) -> float:
        frac = min(self._step_count / self._epsilon_decay_steps, 1.0)
        return self._epsilon_start + frac * (self._epsilon_end - self._epsilon_start)

    def act(self, state: np.ndarray, available_actions: list[int],
            training: bool = False) -> int:
        if training:
            self._step_count += 1
            if random.random() < self._get_epsilon():
                return random.choice(available_actions)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self._online_net(state_t).squeeze(0).numpy()

        mask = np.full(self._action_size, -np.inf)
        for a in available_actions:
            mask[a] = 0.0
        return int(np.argmax(q_values + mask))

    def observe(self, state, action, reward, next_state, done):
        self._buffer.push(state, action, reward, next_state, done)
        if len(self._buffer) >= self._batch_size:
            self._train_step()

    def _train_step(self):
        states, actions, rewards, next_states, dones = self._buffer.sample(
            self._batch_size
        )

        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.BoolTensor(dones)

        q_values = self._online_net(states_t)
        q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        targets = self._compute_targets(rewards_t, next_states_t, dones_t)

        loss = F.mse_loss(q_selected, targets)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._update_count += 1
        if self._update_count % self._target_update_freq == 0:
            self._target_net.load_state_dict(self._online_net.state_dict())

    def _compute_targets(self, rewards_t, next_states_t, dones_t):
        """DQN target: r + gamma * max_a Q_target(s', a)."""
        with torch.no_grad():
            next_q = self._target_net(next_states_t)
            next_q_max = next_q.max(dim=1).values
        return rewards_t + self._gamma * next_q_max * (~dones_t).float()

    def save(self, path: str) -> None:
        torch.save({
            "online_net": self._online_net.state_dict(),
            "target_net": self._target_net.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "step_count": self._step_count,
            "update_count": self._update_count,
        }, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu", weights_only=False)
        self._online_net.load_state_dict(data["online_net"])
        self._target_net.load_state_dict(data["target_net"])
        self._optimizer.load_state_dict(data["optimizer"])
        self._step_count = data["step_count"]
        self._update_count = data["update_count"]
