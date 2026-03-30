import numpy as np

from .base import Environment

_WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # lignes
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # colonnes
    (0, 4, 8), (2, 4, 6),              # diagonales
]


class TicTacToeEnv(Environment):
    """TicTacToe 3x3 avec encodage du point de vue du joueur courant (D-002)."""

    def __init__(self):
        # board[i] : 0=vide, 1=joueur0, 2=joueur1
        self._board = np.zeros(9, dtype=np.int8)
        self._current = 0
        self._done = False

    def reset(self) -> np.ndarray:
        self._board[:] = 0
        self._current = 0
        self._done = False
        return self.state_description()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        player_mark = self._current + 1  # 1 ou 2
        self._board[action] = player_mark

        if self._check_win(player_mark):
            self._done = True
            reward = 1.0
            # changer de perspective pour que l'etat soit du point de vue du prochain joueur
            self._current = 1 - self._current
            return self.state_description(), reward, True

        if not np.any(self._board == 0):
            self._done = True
            self._current = 1 - self._current
            return self.state_description(), 0.0, True

        reward = 0.0
        self._current = 1 - self._current
        return self.state_description(), reward, False

    def available_actions(self) -> list[int]:
        if self._done:
            return []
        return [i for i in range(9) if self._board[i] == 0]

    def state_description(self) -> np.ndarray:
        """3 canaux de 9 : marques du joueur courant, marques adverses, cases vides."""
        my_mark = self._current + 1
        opp_mark = 2 - self._current

        mine = (self._board == my_mark).astype(np.float32)
        opp = (self._board == opp_mark).astype(np.float32)
        empty = (self._board == 0).astype(np.float32)
        return np.concatenate([mine, opp, empty])

    def action_space_size(self) -> int:
        return 9

    def state_space_size(self) -> int:
        return 27

    def is_adversarial(self) -> bool:
        return True

    def current_player(self) -> int:
        return self._current

    def _check_win(self, mark: int) -> bool:
        for a, b, c in _WIN_LINES:
            if self._board[a] == self._board[b] == self._board[c] == mark:
                return True
        return False

    def render_text(self) -> str:
        symbols = {0: '.', 1: 'X', 2: 'O'}
        lines = []
        for row in range(3):
            cells = [symbols[self._board[row * 3 + col]] for col in range(3)]
            lines.append(' '.join(cells))
        return '\n'.join(lines)
