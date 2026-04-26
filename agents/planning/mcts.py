"""Monte Carlo Tree Search (UCT) planning agent.

Arbre reconstruit a chaque act(). Backup negamax base sur le CHANGEMENT de
joueur (node.player vs terminal_actor), pas sur la parite du niveau : cela
gere correctement les tours multi-phases de Bobail ou le meme joueur joue
deux coups consecutifs.
"""

import math
import random

import numpy as np

from environments.base import Environment
from ..base import Agent


class _Node:
    __slots__ = ("player", "untried", "children", "N", "W", "is_terminal",
                 "terminal_reward", "terminal_actor")

    def __init__(self, player: int, legal_actions: list[int],
                 is_terminal: bool = False,
                 terminal_reward: float = 0.0,
                 terminal_actor: int = 0):
        self.player = player
        self.untried = list(legal_actions)
        self.children: dict[int, _Node] = {}
        self.N: dict[int, int] = {a: 0 for a in legal_actions}
        self.W: dict[int, float] = {a: 0.0 for a in legal_actions}
        self.is_terminal = is_terminal
        self.terminal_reward = terminal_reward
        self.terminal_actor = terminal_actor

    def total_visits(self) -> int:
        return sum(self.N.values())

    def fully_expanded(self) -> bool:
        return not self.untried


class MCTSAgent(Agent):

    def __init__(self, env: Environment,
                 n_simulations: int = 200,
                 c_uct: float = 1.41,
                 max_rollout_depth: int = 300,
                 gamma: float = 1.0,
                 **kwargs):
        self._env = env
        self._n_simulations = int(n_simulations)
        self._c = float(c_uct)
        self._max_depth = int(max_rollout_depth)
        self._gamma = float(gamma)

    def act(self, state: np.ndarray, available_actions: list[int],
            training: bool = False) -> int:
        if len(available_actions) == 1:
            return available_actions[0]

        root_env = self._env.clone()
        root = _Node(
            player=root_env.current_player(),
            legal_actions=list(available_actions),
        )

        for _ in range(self._n_simulations):
            sim = root_env.clone()
            path: list[tuple[_Node, int]] = []
            node = root

            # Selection
            while node.fully_expanded() and not node.is_terminal:
                action = self._ucb_select(node)
                path.append((node, action))
                actor = sim.current_player()
                _, reward, done = sim.step(action)
                child = node.children[action]
                node = child
                if child.is_terminal:
                    break

            # Expansion
            if not node.is_terminal and node.untried:
                action = random.choice(node.untried)
                node.untried.remove(action)
                path.append((node, action))
                actor = sim.current_player()
                _, reward, done = sim.step(action)
                if done:
                    child = _Node(
                        player=sim.current_player(),
                        legal_actions=[],
                        is_terminal=True,
                        terminal_reward=reward,
                        terminal_actor=actor,
                    )
                    node.children[action] = child
                    terminal_reward = reward
                    terminal_actor = actor
                else:
                    legal = sim.available_actions()
                    child = _Node(
                        player=sim.current_player(),
                        legal_actions=legal,
                    )
                    node.children[action] = child
                    terminal_reward, terminal_actor = self._rollout(sim)
            elif node.is_terminal:
                terminal_reward = node.terminal_reward
                terminal_actor = node.terminal_actor
            else:
                # Pas d'actions et non-terminal (defensif) : rollout nul
                terminal_reward, terminal_actor = 0.0, node.player

            # Backup
            for pnode, paction in path:
                val = (terminal_reward
                       if pnode.player == terminal_actor
                       else -terminal_reward)
                pnode.N[paction] += 1
                pnode.W[paction] += val

        # argmax sur les visites a la racine
        best_action = max(root.N, key=lambda a: root.N[a])
        return best_action

    def _ucb_select(self, node: _Node) -> int:
        total_n = node.total_visits()
        log_total = math.log(total_n) if total_n > 0 else 0.0
        best_a = None
        best_score = -float("inf")
        for a, n_a in node.N.items():
            if n_a == 0:
                return a
            q = node.W[a] / n_a
            u = self._c * math.sqrt(log_total / n_a)
            score = q + u
            if score > best_score:
                best_score = score
                best_a = a
        return best_a

    def _rollout(self, sim: Environment) -> tuple[float, int]:
        """Random playout jusqu'a terminal/cap. Renvoie (reward, last_actor)."""
        depth = 0
        last_actor = sim.current_player()
        while depth < self._max_depth:
            legal = sim.available_actions()
            if not legal:
                return 0.0, last_actor
            last_actor = sim.current_player()
            a = random.choice(legal)
            _, reward, done = sim.step(a)
            depth += 1
            if done:
                return reward, last_actor
        return 0.0, last_actor
