from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):

    @abstractmethod
    def act(self, state: np.ndarray, available_actions: list[int],
            training: bool = False) -> int:
        """Select an action.

        training=True: may explore (epsilon-greedy, stochastic, etc.)
        training=False: pure exploitation (greedy, deterministic)
        """
        ...

    def observe(self, state, action, reward, next_state, done):
        """Called after env.step() during training. Default: no-op."""
        pass

    def end_episode(self):
        """Called at end of each training episode. Default: no-op."""
        pass

    def save(self, path: str) -> None:
        """Save model/weights/tables."""
        pass

    def load(self, path: str) -> None:
        """Load model/weights/tables."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
