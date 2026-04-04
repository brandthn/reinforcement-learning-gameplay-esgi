import numpy as np

from .base import Environment


class GridWorldEnv(Environment):
    """Grille 2D : l'agent demarre en (0,0), case 4 = defaite (-1), case 24 = victoire (+1)."""

    LOSE_INDEX = 4
    WIN_INDEX = 24

    def __init__(self, rows: int = 5, cols: int = 5):
        self._rows = rows
        self._cols = cols
        self._row = 0
        self._col = 0
        self._done = False

    def _flat_index(self) -> int:
        return self._row * self._cols + self._col

    def reset(self) -> np.ndarray:
        self._row = 0
        self._col = 0
        self._done = False
        return self.state_description()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        # 0=haut, 1=bas, 2=gauche, 3=droite
        if action == 0:
            self._row = max(0, self._row - 1)
        elif action == 1:
            self._row = min(self._rows - 1, self._row + 1)
        elif action == 2:
            self._col = max(0, self._col - 1)
        elif action == 3:
            self._col = min(self._cols - 1, self._col + 1)

        idx = self._flat_index()
        if idx == self.WIN_INDEX:
            self._done = True
            return self.state_description(), 1.0, True
        elif idx == self.LOSE_INDEX:
            self._done = True
            return self.state_description(), -1.0, True
        return self.state_description(), 0.0, False

    def available_actions(self) -> list[int]:
        if self._done:
            return []
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
        lose_r, lose_c = self.LOSE_INDEX // self._cols, self.LOSE_INDEX % self._cols
        win_r, win_c = self.WIN_INDEX // self._cols, self.WIN_INDEX % self._cols
        lines = []
        for r in range(self._rows):
            row = []
            for c in range(self._cols):
                if r == self._row and c == self._col:
                    row.append('A')
                elif r == win_r and c == win_c:
                    row.append('G')
                elif r == lose_r and c == lose_c:
                    row.append('X')
                else:
                    row.append('.')
            lines.append('|'.join(row))
        return '\n'.join(lines)
