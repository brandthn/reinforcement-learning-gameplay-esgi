from .random_agent import RandomAgent
from .human_agent import HumanAgent

AGENT_REGISTRY = {
    "random": RandomAgent,
    "human": HumanAgent,
}


def get_agent(name: str, env, params: dict = None):
    params = params or {}
    return AGENT_REGISTRY[name](
        state_size=env.state_space_size(),
        action_size=env.action_space_size(),
        **params,
    )
