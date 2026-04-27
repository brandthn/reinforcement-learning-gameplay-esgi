"""Tabular Q-Learning agent."""

import pickle
import random

import numpy as np

from .base import Agent


class TabularQAgent(Agent):
    """Q-learning with a dict-based Q-table.

    Works on environments with discrete, finite state spaces where the
    state vector is hashable (e.g. one-hot or binary encodings).
    """

    def __init__(self, state_size: int, action_size: int,
                 lr: float, gamma: float,
                 epsilon_start: float, epsilon_end: float,
                 epsilon_decay_steps: int, device: str = "cpu"):
        self._state_size = state_size
        self._action_size = action_size
        self._lr = lr
        self._gamma = gamma
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay_steps = epsilon_decay_steps
        self._q: dict[tuple, np.ndarray] = {}
        self._step_count = 0

    def _get_epsilon(self) -> float:
        frac = min(self._step_count / self._epsilon_decay_steps, 1.0)
        return self._epsilon_start + frac * (self._epsilon_end - self._epsilon_start)

    def _state_key(self, state: np.ndarray) -> tuple:
        return tuple(state.tolist())

    def _get_q(self, state: np.ndarray) -> np.ndarray:
        key = self._state_key(state)
        if key not in self._q:
            self._q[key] = np.zeros(self._action_size)
        return self._q[key]

    def act(self, state: np.ndarray, available_actions: list[int],
            training: bool = False) -> int:
        if training:
            self._step_count += 1
            if random.random() < self._get_epsilon():
                return random.choice(available_actions)

        q = self._get_q(state)
        best_a = available_actions[0]
        best_q = q[best_a]
        for a in available_actions[1:]:
            if q[a] > best_q:
                best_q = q[a]
                best_a = a
        return best_a

    def observe(self, state, action, reward, next_state, done):
        q = self._get_q(state)
        if done:
            target = reward
        else:
            next_q = self._get_q(next_state)
            target = reward + self._gamma * np.max(next_q)
        q[action] += self._lr * (target - q[action])

    def save(self, path: str) -> None:
        data = {
            "q_table": dict(self._q),
            "step_count": self._step_count,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._q = data["q_table"]
        self._step_count = data["step_count"]
