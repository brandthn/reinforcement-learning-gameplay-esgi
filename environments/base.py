from abc import ABC, abstractmethod
import copy

import numpy as np


class Environment(ABC):

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset to initial state. Returns state vector."""
        ...

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Execute action. Returns (next_state, reward, done).

        For two-player games:
        - reward is from the perspective of the player who just acted
        - next_state is from the perspective of the NEW current player
        """
        ...

    @abstractmethod
    def available_actions(self) -> list[int]:
        """Legal action indices for the current player/phase."""
        ...

    @abstractmethod
    def state_description(self) -> np.ndarray:
        """Current state as flat float32 vector."""
        ...

    @abstractmethod
    def action_space_size(self) -> int:
        """Total number of possible actions (including currently illegal ones)."""
        ...

    @abstractmethod
    def state_space_size(self) -> int:
        """Dimensionality of state_description() output."""
        ...

    def is_adversarial(self) -> bool:
        """Override to return True for two-player games."""
        return False

    def current_player(self) -> int:
        """0 for single-player. 0 or 1 for two-player."""
        return 0

    def clone(self):
        """Deep copy. Required for MCTS/AlphaZero/MuZero."""
        return copy.deepcopy(self)

    def render_text(self) -> str:
        """Optional text representation for debugging."""
        return ""
