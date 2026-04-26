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


def apply_quick_mode(config: dict, episodes: int = 1000):
    """Override config for fast iteration: fewer episodes, single seed, dev results dir."""
    config["training"]["num_episodes"] = episodes
    config["seeds"] = [config.get("seed", 42)]
    config.setdefault("eval", {})
    config["eval"]["checkpoints"] = [episodes]
    config["results_dir"] = "results_dev"


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

# ── Add this function next to set_seed() ─────────────────────────────────────
def get_device() -> str:
    """Return 'cuda' if a GPU is available, 'mps' on Apple Silicon, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():   # Apple Silicon
            return "mps"
    except ImportError:
        pass
    return "cpu"

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

    device = get_device()
    print(f"Device: {device}")
    
    agent_params = {**agent_params, "device": device} #j'ai ajouté le device CUDA si possible

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
    parser.add_argument("--quick", action="store_true",
                        help="Fast iteration mode: 1 seed, fewer episodes, writes to results_dev/")
    parser.add_argument("--quick-episodes", type=int, default=1000,
                        help="Number of episodes in --quick mode (default: 1000)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.quick:
        apply_quick_mode(config, args.quick_episodes)

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = config.get("seeds", [config.get("seed", 42)])

    for seed in seeds:
        train_single(config, seed)


if __name__ == "__main__":
    main()
