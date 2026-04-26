from .random_agent import RandomAgent
from .human_agent import HumanAgent
from .tabular_q import TabularQAgent
from .value_based.dqn import DQNAgent
from .value_based.ddqn import DDQNAgent
from .value_based.ddqn_er import DDQNERAgent
from .value_based.ddqn_per import DDQNPERAgent
from .planning.random_rollout import RandomRolloutAgent
from .planning.mcts import MCTSAgent

AGENT_REGISTRY = {
    "random": RandomAgent,
    "human": HumanAgent,
    "tabular_q": TabularQAgent,
    "dqn": DQNAgent,
    "ddqn": DDQNAgent,
    "ddqn_er": DDQNERAgent,
    "ddqn_per": DDQNPERAgent,
    "random_rollout": RandomRolloutAgent,
    "mcts": MCTSAgent,
}

# Agents de planification : besoin d'une reference a l'env pour cloner.
PLANNING_AGENTS = {"random_rollout", "mcts"}


def get_agent(name: str, env, params: dict = None):
    params = params or {}
    cls = AGENT_REGISTRY[name]
    if name in PLANNING_AGENTS:
        return cls(env=env, **params)
    return cls(
        state_size=env.state_space_size(),
        action_size=env.action_space_size(),
        **params,
    )
