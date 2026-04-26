"""REINFORCE agent with episode mean baseline."""

from agents.policy_based.base_reinforce import BaseREINFORCEAgent


class REINFORCEMeanBaselineAgent(BaseREINFORCEAgent):
    """Subtracts episode mean return as a baseline to reduce variance."""

    def _update_policy(self, states, actions, returns):
        baseline = returns.mean()
        advantages = returns - baseline

        self._optimizer.zero_grad()
        loss = 0.0
        for state, action, advantage in zip(states, actions, advantages):
            log_prob = self._log_prob(state, action)
            loss = loss - log_prob * advantage

        loss = loss / len(advantages)
        loss.backward()
        self._optimizer.step()
