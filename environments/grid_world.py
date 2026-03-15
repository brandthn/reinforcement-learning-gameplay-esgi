import numpy as np

from .base import Environment


class GridWorldEnv(Environment):
    """2D grid: agent starts top-left (0,0), goal at bottom-right (rows-1, cols-1)."""

    def __init__(self, rows: int = 5, cols: int = 5):
        self._rows = rows
        self._cols = cols
        self._row = 0
        self._col = 0

    def reset(self) -> np.ndarray:
        self._row = 0
        self._col = 0
        return self.state_description()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        # 0=up, 1=down, 2=left, 3=right
        if action == 0:
            self._row = max(0, self._row - 1)
        elif action == 1:
            self._row = min(self._rows - 1, self._row + 1)
        elif action == 2:
            self._col = max(0, self._col - 1)
        elif action == 3:
            self._col = min(self._cols - 1, self._col + 1)

        done = (self._row == self._rows - 1) and (self._col == self._cols - 1)
        reward = 1.0 if done else 0.0
        return self.state_description(), reward, done

    def available_actions(self) -> list[int]:
        actions = []
        if self._row > 0:
            actions.append(0)
        if self._row < self._rows - 1:
            actions.append(1)
        if self._col > 0:
            actions.append(2)
        if self._col < self._cols - 1:
            actions.append(3)
        return actions

    def state_description(self) -> np.ndarray:
        state = np.zeros(self._rows * self._cols, dtype=np.float32)
        state[self._row * self._cols + self._col] = 1.0
        return state

    def action_space_size(self) -> int:
        return 4

    def state_space_size(self) -> int:
        return self._rows * self._cols

    def render_text(self) -> str:
        lines = []
        for r in range(self._rows):
            row = []
            for c in range(self._cols):
                if r == self._row and c == self._col:
                    row.append('A')
                elif r == self._rows - 1 and c == self._cols - 1:
                    row.append('G')
                else:
                    row.append('.')
            lines.append('|'.join(row))
        return '\n'.join(lines)
