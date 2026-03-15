from .line_world import LineWorldEnv
from .grid_world import GridWorldEnv
from .tictactoe import TicTacToeEnv
from .bobail import BobailEnv

ENV_REGISTRY = {
    "line_world": LineWorldEnv,
    "grid_world": GridWorldEnv,
    "tictactoe": TicTacToeEnv,
    "bobail": BobailEnv,
}


def get_env(name: str, **kwargs):
    return ENV_REGISTRY[name](**kwargs)
