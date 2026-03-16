"""Double DQN with Prioritized Experience Replay (DDQN+PER) agent."""

import random

import numpy as np
import torch
import torch.nn.functional as F

from .ddqn import DDQNAgent
from training.replay_buffer import PrioritizedReplayBuffer


class DDQNPERAgent(DDQNAgent):
    """DDQN with proportional Prioritized Experience Replay.

    Replaces uniform replay sampling with priority-weighted sampling
    (sum-tree). Uses importance-sampling weights to correct the bias
    introduced by non-uniform sampling. Beta anneals linearly from
    per_beta_start toward per_beta_end over per_beta_steps.
    """

    def __init__(self, state_size: int, action_size: int,
                 lr: float, gamma: float,
                 epsilon_start: float, epsilon_end: float,
                 epsilon_decay_steps: int,
                 hidden_layers: list[int],
                 batch_size: int, buffer_capacity: int,
                 target_update_freq: int,
                 per_alpha: float,
                 per_beta_start: float,
                 per_beta_end: float,
                 per_beta_steps: int,
                 learning_starts: int):
        super().__init__(
            state_size=state_size, action_size=action_size,
            lr=lr, gamma=gamma,
            epsilon_start=epsilon_start, epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            hidden_layers=hidden_layers,
            batch_size=batch_size, buffer_capacity=buffer_capacity,
            target_update_freq=target_update_freq,
        )
        self._buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=per_alpha)
        self._per_beta_start = per_beta_start
        self._per_beta_end = per_beta_end
        self._per_beta_steps = per_beta_steps
        self._learning_starts = learning_starts

    def _get_beta(self) -> float:
        frac = min(self._step_count / self._per_beta_steps, 1.0)
        return self._per_beta_start + frac * (self._per_beta_end - self._per_beta_start)

    def observe(self, state, action, reward, next_state, done):
        self._buffer.push(state, action, reward, next_state, done)
        if (len(self._buffer) >= self._batch_size
                and self._step_count >= self._learning_starts):
            self._train_step()

    def _train_step(self):
        beta = self._get_beta()
        states, actions, rewards, next_states, dones, indices, weights = (
            self._buffer.sample(self._batch_size, beta=beta)
        )

        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.BoolTensor(dones)
        weights_t = torch.FloatTensor(weights)

        q_values = self._online_net(states_t)
        q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        targets = self._compute_targets(rewards_t, next_states_t, dones_t)

        td_errors = (q_selected - targets).detach()

        # IS-weighted MSE loss
        loss = (weights_t * (q_selected - targets).pow(2)).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._buffer.update_priorities(indices, td_errors.numpy())

        self._update_count += 1
        if self._update_count % self._target_update_freq == 0:
            self._target_net.load_state_dict(self._online_net.state_dict())
