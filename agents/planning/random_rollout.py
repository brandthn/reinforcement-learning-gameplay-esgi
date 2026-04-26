"""Flat Monte-Carlo (Random Rollout) planning agent.

Aucun apprentissage : a chaque appel act(), on estime Q(s,a) pour chaque
action legale en moyennant n_rollouts_per_action parties jouees uniformement
au hasard (pour les deux cotes si l'env est adversarial).
"""

import random

import numpy as np

from environments.base import Environment
from ..base import Agent


class RandomRolloutAgent(Agent):

    def __init__(self, env: Environment,
                 n_rollouts_per_action: int = 20,
                 max_rollout_depth: int = 300,
                 gamma: float = 1.0,
                 **kwargs):
        self._env = env
        self._n_rollouts = int(n_rollouts_per_action)
        self._max_depth = int(max_rollout_depth)
        self._gamma = float(gamma)

    def act(self, state: np.ndarray, available_actions: list[int],
            training: bool = False) -> int:
        if len(available_actions) == 1:
            return available_actions[0]

        root_player = self._env.current_player()
        best_action = available_actions[0]
        best_score = -float("inf")

        for action in available_actions:
            total = 0.0
            for _ in range(self._n_rollouts):
                sim = self._env.clone()
                total += self._rollout_starting_with(sim, action, root_player)
            avg = total / self._n_rollouts
            if avg > best_score:
                best_score = avg
                best_action = action

        return best_action

    def _rollout_starting_with(self, sim: Environment, first_action: int,
                               root_player: int) -> float:
        """Play `first_action` then random policy until terminal or cap.

        Returns the terminal reward converted to root_player's POV.
        Non-terminal rewards are 0 in all project envs so they are ignored
        (matches tictactoe/bobail/grid/line semantics).
        """
        actor = sim.current_player()
        _, reward, done = sim.step(first_action)
        depth = 1

        if done:
            return reward if actor == root_player else -reward

        while depth < self._max_depth:
            legal = sim.available_actions()
            if not legal:
                return 0.0
            actor = sim.current_player()
            a = random.choice(legal)
            _, reward, done = sim.step(a)
            depth += 1
            if done:
                return reward if actor == root_player else -reward

        return 0.0
