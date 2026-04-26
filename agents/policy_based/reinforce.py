
"""Vanilla REINFORCE agent."""

from agents.policy_based.base_reinforce import BaseREINFORCEAgent


class REINFORCEAgent(BaseREINFORCEAgent):
    """Policy gradient with Monte Carlo returns as advantages."""

    def _update_policy(self, states, actions, returns):
        self._optimizer.zero_grad()

        loss = 0.0
        for state, action, ret in zip(states, actions, returns):
            log_prob = self._log_prob(state, action)
            loss = loss - log_prob * ret

        loss = loss / len(returns)
        loss.backward()
        self._optimizer.step()
