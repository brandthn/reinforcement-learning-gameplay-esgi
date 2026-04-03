"""Replay buffers for value-based methods."""

import random
import numpy as np


class ReplayBuffer:
    """Uniform-sampling replay buffer for DQN / DDQN / DDQN+ER."""

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._buffer: list[tuple] = []
        self._pos = 0

    def push(self, state, action, reward, next_state, done):
        transition = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        if len(self._buffer) < self._capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._pos] = transition
        self._pos = (self._pos + 1) % self._capacity

    def sample(self, batch_size: int):
        batch = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
        )

    def __len__(self):
        return len(self._buffer)


class PrioritizedReplayBuffer:
    """Proportional prioritized experience replay using a sum-tree.

    - alpha controls how much prioritization is used (0 = uniform, 1 = full)
    - beta (passed at sample time) controls importance-sampling correction
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        self._capacity = capacity
        self._alpha = alpha
        # Binary sum-tree stored in a flat array (2*capacity - 1 nodes)
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._data: list[tuple | None] = [None] * capacity
        self._pos = 0
        self._size = 0
        self._max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        transition = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        priority = self._max_priority ** self._alpha
        self._data[self._pos] = transition
        self._update_tree(self._pos, priority)
        self._pos = (self._pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        indices = []
        priorities = []
        segment = self._tree[0] / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = random.uniform(low, high)
            idx, prio = self._retrieve(value)
            indices.append(idx)
            priorities.append(prio)

        priorities_arr = np.array(priorities, dtype=np.float64)
        probs = priorities_arr / self._tree[0]
        weights = (self._size * probs) ** (-beta)
        weights /= weights.max()

        batch = [self._data[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            np.array(indices, dtype=np.int64),
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + 1e-6) ** self._alpha
            self._max_priority = max(self._max_priority, abs(td) + 1e-6)
            self._update_tree(idx, priority)

    def _update_tree(self, data_idx: int, priority: float):
        tree_idx = data_idx + self._capacity - 1
        delta = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self._tree[tree_idx] += delta

    def _retrieve(self, value: float) -> tuple[int, float]:
        idx = 0
        while idx < self._capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = right
        data_idx = idx - (self._capacity - 1)
        return data_idx, self._tree[idx]

    def __len__(self):
        return self._size
