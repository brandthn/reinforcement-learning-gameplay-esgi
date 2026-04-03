import numpy as np

from .base import Agent


class HumanAgent(Agent):
    """Agent whose actions are set externally by the GUI.

    The GUI calls set_action() with the human's chosen action,
    then the game loop calls act() which returns it.
    """

    def __init__(self, state_size: int = 0, action_size: int = 0, **kwargs):
        self._pending_action: int | None = None

    def set_action(self, action: int) -> None:
        self._pending_action = action

    def act(self, state: np.ndarray, available_actions: list[int],
            training: bool = False) -> int:
        if self._pending_action is None:
            raise RuntimeError("HumanAgent.act() called without a pending action")
        action = self._pending_action
        self._pending_action = None
        return action
