import numpy as np

from .base import Environment

BOARD_SIZE = 5
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]

# Positions initiales (ligne, colonne)
# Joueur 0 : rangee du bas (ligne 4)
_P0_START = [(4, c) for c in range(5)]
# Joueur 1 : rangee du haut (ligne 0)
_P1_START = [(0, c) for c in range(5)]
_BOBAIL_START = (2, 2)

PHASE_BOBAIL = 0
PHASE_PIECE = 1


def _rc_to_idx(r: int, c: int) -> int:
    return r * BOARD_SIZE + c


def _idx_to_rc(idx: int) -> tuple[int, int]:
    return divmod(idx, BOARD_SIZE)


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


class BobailEnv(Environment):
    """Bobail : jeu 5x5 a deux joueurs avec des tours en deux phases (D-003).

    Encodage des actions : case_depart * 25 + case_arrivee (espace d'actions = 625).
    Phase 0 = deplacer le bobail (1 pas), Phase 1 = deplacer sa propre piece (glissade).
    Premier tour de la partie : le joueur 0 saute la phase bobail.
    """

    def __init__(self):
        # pieces[joueur] = ensemble d'indices de cases
        self._pieces = [set(), set()]
        self._bobail = 0
        self._current = 0
        self._phase = PHASE_PIECE  # le premier tour commence a la phase piece
        self._done = False
        self._turn_number = 0
        self._first_turn = True

    def reset(self) -> np.ndarray:
        self._pieces[0] = {_rc_to_idx(r, c) for r, c in _P0_START}
        self._pieces[1] = {_rc_to_idx(r, c) for r, c in _P1_START}
        self._bobail = _rc_to_idx(*_BOBAIL_START)
        self._current = 0
        self._phase = PHASE_PIECE  # le premier tour du joueur 0 saute la phase bobail
        self._done = False
        self._turn_number = 0
        self._first_turn = True
        return self.state_description()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        from_cell = action // NUM_CELLS
        to_cell = action % NUM_CELLS

        if self._phase == PHASE_BOBAIL:
            self._bobail = to_cell
            self._phase = PHASE_PIECE

            # Ligne du bobail apres le deplacement
            br, _ = _idx_to_rc(self._bobail)
            # Camp J0 = ligne 4, Camp J1 = ligne 0
            home_row = 4 if self._current == 0 else 0
            # Bobail sur la rangee maison du joueur courant = victoire pour le joueur courant
            if br == home_row:
                self._done = True
                # Convention : current = perdant apres la fin de partie
                self._current = 1 - self._current
                return self.state_description(), 1.0, True

            # Pas de victoire, on continue a la phase piece
            return self.state_description(), 0.0, False

        # PHASE_PIECE (deplacement de piece)
        self._pieces[self._current].discard(from_cell)
        self._pieces[self._current].add(to_cell)

        if self._first_turn:
            self._first_turn = False

        # Passage a l'adversaire
        opponent = 1 - self._current
        self._current = opponent
        self._turn_number += 1

        # Le prochain tour commence par la phase bobail
        self._phase = PHASE_BOBAIL

        # Verifier si l'adversaire peut deplacer le bobail
        if not self._bobail_moves():
            self._done = True
            # L'adversaire (maintenant courant) ne peut pas deplacer le bobail -> il perd
            # current_player pointe deja vers le perdant (convention)
            return self.state_description(), 1.0, True

        return self.state_description(), 0.0, False

    def available_actions(self) -> list[int]:
        if self._done:
            return []

        if self._phase == PHASE_BOBAIL:
            return self._bobail_moves()
        return self._piece_moves()

    def _bobail_moves(self) -> list[int]:
        """Mouvements legaux du bobail : 1 pas dans n'importe quelle direction, bloque par les pieces/bords."""
        br, bc = _idx_to_rc(self._bobail)
        occupied = self._pieces[0] | self._pieces[1]
        moves = []
        for dr, dc in DIRECTIONS:
            nr, nc = br + dr, bc + dc
            if _in_bounds(nr, nc):
                target = _rc_to_idx(nr, nc)
                if target not in occupied:
                    moves.append(self._bobail * NUM_CELLS + target)
        return moves

    def _piece_moves(self) -> list[int]:
        """Mouvements legaux des pieces : glisser aussi loin que possible dans une direction."""
        occupied = self._pieces[0] | self._pieces[1] | {self._bobail}
        moves = []
        for cell in self._pieces[self._current]:
            r, c = _idx_to_rc(cell)
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                # Doit se deplacer d'au moins 1 case
                if not _in_bounds(nr, nc):
                    continue
                if _rc_to_idx(nr, nc) in occupied:
                    continue
                # Glisser jusqu'a etre bloque
                while _in_bounds(nr + dr, nc + dc) and _rc_to_idx(nr + dr, nc + dc) not in occupied:
                    nr += dr
                    nc += dc
                target = _rc_to_idx(nr, nc)
                moves.append(cell * NUM_CELLS + target)
        return moves

    def state_description(self) -> np.ndarray:
        """3 canaux de 25 : pieces du joueur courant, pieces adverses, bobail."""
        my_pieces = np.zeros(NUM_CELLS, dtype=np.float32)
        opp_pieces = np.zeros(NUM_CELLS, dtype=np.float32)
        bobail = np.zeros(NUM_CELLS, dtype=np.float32)

        for idx in self._pieces[self._current]:
            my_pieces[idx] = 1.0
        for idx in self._pieces[1 - self._current]:
            opp_pieces[idx] = 1.0
        bobail[self._bobail] = 1.0

        return np.concatenate([my_pieces, opp_pieces, bobail])

    def action_space_size(self) -> int:
        return NUM_CELLS * NUM_CELLS

    def state_space_size(self) -> int:
        return NUM_CELLS * 3

    def is_adversarial(self) -> bool:
        return True

    def current_player(self) -> int:
        return self._current

    def render_text(self) -> str:
        lines = []
        for r in range(BOARD_SIZE):
            row = []
            for c in range(BOARD_SIZE):
                idx = _rc_to_idx(r, c)
                if idx == self._bobail:
                    row.append('B')
                elif idx in self._pieces[0]:
                    row.append('0')
                elif idx in self._pieces[1]:
                    row.append('1')
                else:
                    row.append('.')
            lines.append(' '.join(row))
        return '\n'.join(lines)
