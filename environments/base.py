from abc import ABC, abstractmethod
import copy

import numpy as np


class Environment(ABC):

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reinitialise a l'etat initial. Retourne le vecteur d'etat."""
        ...

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Execute une action. Retourne (next_state, reward, done).

        Pour les jeux a deux joueurs :
        - reward est du point de vue du joueur qui vient d'agir
        - next_state est du point de vue du NOUVEAU joueur courant
        """
        ...

    @abstractmethod
    def available_actions(self) -> list[int]:
        """Indices des actions legales pour le joueur/phase courant."""
        ...

    @abstractmethod
    def state_description(self) -> np.ndarray:
        """Etat courant sous forme de vecteur float32 aplati."""
        ...

    @abstractmethod
    def action_space_size(self) -> int:
        """Nombre total d'actions possibles (y compris les actions actuellement illegales)."""
        ...

    @abstractmethod
    def state_space_size(self) -> int:
        """Dimensionnalite de la sortie de state_description()."""
        ...

    def is_adversarial(self) -> bool:
        """Surcharger pour retourner True pour les jeux a deux joueurs."""
        return False

    def current_player(self) -> int:
        """0 pour un joueur. 0 ou 1 pour deux joueurs."""
        return 0

    def clone(self):
        """Copie profonde. Necessaire pour MCTS/AlphaZero/MuZero."""
        return copy.deepcopy(self)

    def render_text(self) -> str:
        """Representation textuelle optionnelle pour le debogage."""
        return ""
