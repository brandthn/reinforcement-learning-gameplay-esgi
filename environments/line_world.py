import numpy as np

from .base import Environment


class LineWorldEnv(Environment):
    """Grille 1D : l'agent demarre au milieu, case 0 = defaite (-1), case N-1 = victoire (+1)."""

    def __init__(self, size: int = 5):
        self._size = size
        self._pos = size // 2
        self._done = False

    def reset(self) -> np.ndarray:
        self._pos = self._size // 2
        self._done = False
        return self.state_description()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        if action == 0:
            self._pos = max(0, self._pos - 1)
        elif action == 1:
            self._pos = min(self._size - 1, self._pos + 1)

        if self._pos == self._size - 1:
            self._done = True
            return self.state_description(), 1.0, True
        elif self._pos == 0:
            self._done = True
            return self.state_description(), -1.0, True
        return self.state_description(), 0.0, False

    def available_actions(self) -> list[int]:
        if self._done:
            return []
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
        cells[0] = 'X'
        cells[-1] = 'G'
        cells[self._pos] = 'A'
        return '[' + '|'.join(cells) + ']'
