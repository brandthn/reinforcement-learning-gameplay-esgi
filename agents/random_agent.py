import random

import numpy as np

from .base import Agent


class RandomAgent(Agent):
    """Choisit uniformement au hasard parmi les actions disponibles."""

    def __init__(self, state_size: int = 0, action_size: int = 0, **kwargs):
        pass

    def act(self, state: np.ndarray, available_actions: list[int],
            training: bool = False) -> int:
        return random.choice(available_actions)
