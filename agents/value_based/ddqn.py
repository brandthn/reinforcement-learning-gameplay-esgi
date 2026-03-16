"""Double Deep Q-Network (DDQN) agent."""

import torch

from .dqn import DQNAgent


class DDQNAgent(DQNAgent):
    """DDQN: online net selects the best action, target net evaluates it.

    Reduces overestimation bias from standard DQN.
    All other behavior (epsilon-greedy, replay buffer, target updates)
    is inherited from DQNAgent.
    """

    def _compute_targets(self, rewards_t, next_states_t, dones_t):
        """DDQN target: r + gamma * Q_target(s', argmax_a Q_online(s', a))."""
        with torch.no_grad():
            best_actions = self._online_net(next_states_t).argmax(dim=1)
            next_q = self._target_net(next_states_t)
            next_q_max = next_q.gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)
        return rewards_t + self._gamma * next_q_max * (~dones_t).float()
