from .random_agent import RandomAgent
from .human_agent import HumanAgent
from .tabular_q import TabularQAgent
from .value_based.dqn import DQNAgent
from .value_based.ddqn import DDQNAgent
from .value_based.ddqn_er import DDQNERAgent
from .value_based.ddqn_per import DDQNPERAgent
from .planning.expert_apprentice import ExpertApprenticeAgent

AGENT_REGISTRY = {
    "random": RandomAgent,
    "human": HumanAgent,
    "tabular_q": TabularQAgent,
    "dqn": DQNAgent,
    "ddqn": DDQNAgent,
    "ddqn_er": DDQNERAgent,
    "ddqn_per": DDQNPERAgent,
    "expert_apprentice" : ExpertApprenticeAgent
}


def get_agent(name: str, env, params: dict = None):
    params = params or {}
    return AGENT_REGISTRY[name](
        state_size=env.state_space_size(),
        action_size=env.action_space_size(),
        **params,
    )
