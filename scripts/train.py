#!/usr/bin/env python3
"""Train one agent on one environment from a YAML config."""

import sys
import os
import random
import argparse

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environments import get_env
from agents import get_agent
from training.trainer import Trainer
from training.self_play import SelfPlayTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_results_dir(base_dir: str, env_name: str, agent_name: str,
                      agent_params: dict, seed: int) -> str:
    """results/{env}/{agent}/{params}_seed{N}/"""
    if agent_params:
        parts = [f"{k}{v}" for k, v in sorted(agent_params.items())]
        param_str = "_".join(parts) if parts else "default"
    else:
        param_str = "default"
    return os.path.join(base_dir, env_name, agent_name, f"{param_str}_seed{seed}")


def train_single(config: dict, seed: int) -> dict:
    set_seed(seed)

    env_name = config["env"]
    env_params = config.get("env_params", {})
    agent_name = config["agent"]
    agent_params = config.get("agent_params", {})

    env = get_env(env_name, **env_params)
    agent = get_agent(agent_name, env, agent_params)

    base_dir = config.get("results_dir", "results")
    results_dir = build_results_dir(base_dir, env_name, agent_name,
                                    agent_params, seed)

    config_with_seed = {**config, "seed": seed}

    if env.is_adversarial():
        opp_name = config.get("opponent", "random")
        opp_params = config.get("opponent_params", {})
        opponent = get_agent(opp_name, env, opp_params)
        trainer = SelfPlayTrainer(env, agent, opponent, config_with_seed)
    else:
        trainer = Trainer(env, agent, config_with_seed)

    print(f"Training {agent_name} on {env_name} (seed={seed})")
    print(f"  -> {results_dir}")

    metrics = trainer.train(results_dir)

    for checkpoint, m in sorted(metrics.items()):
        print(f"  [{checkpoint:>8} eps] "
              f"reward={m['mean_reward']:.3f}+/-{m['std_reward']:.3f}  "
              f"steps={m['mean_steps']:.1f}  "
              f"action_time={m['mean_action_time_ms']:.3f}ms")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train one agent on one env")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = config.get("seeds", [config.get("seed", 42)])

    for seed in seeds:
        train_single(config, seed)


if __name__ == "__main__":
    main()
