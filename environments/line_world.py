import numpy as np

from .base import Environment


class LineWorldEnv(Environment):
    """1D grid: agent starts at cell 0, goal at cell N-1."""

    def __init__(self, size: int = 5):
        self._size = size
        self._pos = 0

    def reset(self) -> np.ndarray:
        self._pos = 0
        return self.state_description()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        if action == 0:
            self._pos = max(0, self._pos - 1)
        elif action == 1:
            self._pos = min(self._size - 1, self._pos + 1)

        done = self._pos == self._size - 1
        reward = 1.0 if done else 0.0
        return self.state_description(), reward, done

    def available_actions(self) -> list[int]:
        actions = []
        if self._pos > 0:
            actions.append(0)
        if self._pos < self._size - 1:
            actions.append(1)
        return actions

    def state_description(self) -> np.ndarray:
        state = np.zeros(self._size, dtype=np.float32)
        state[self._pos] = 1.0
        return state

    def action_space_size(self) -> int:
        return 2

    def state_space_size(self) -> int:
        return self._size

    def render_text(self) -> str:
        cells = ['.' for _ in range(self._size)]
        cells[self._pos] = 'A'
        cells[-1] = 'G' if self._pos != self._size - 1 else 'A'
        return '[' + '|'.join(cells) + ']'
