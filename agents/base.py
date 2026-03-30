from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):

    @abstractmethod
    def act(self, state: np.ndarray, available_actions: list[int],
            training: bool = False) -> int:
        """Choisir une action.

        training=True : peut explorer (epsilon-greedy, stochastique, etc.)
        training=False : exploitation pure (greedy, deterministe)
        """
        ...

    def observe(self, state, action, reward, next_state, done):
        """Appele apres env.step() pendant l'entrainement. Par defaut : no-op."""
        pass

    def end_episode(self):
        """Appele a la fin de chaque episode d'entrainement. Par defaut : no-op."""
        pass

    def save(self, path: str) -> None:
        """Sauvegarder le modele/poids/tables."""
        pass

    def load(self, path: str) -> None:
        """Charger le modele/poids/tables."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
