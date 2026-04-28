#!/usr/bin/env python3
"""Train one agent on one environment from a YAML config."""

import sys
import os
import random
import argparse
import hashlib
import re

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


def _short_value(value) -> str:
    """Compact, path-safe representation for run-name values."""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        inner = "-".join(_short_value(v) for v in value)
        return f"[{inner}]"
    text = str(value)
    # Keep only path-safe characters and avoid noisy spaces/punctuation.
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^A-Za-z0-9._\-\[\]]+", "-", text)
    return text.strip("-") or "x"


def _build_param_run_name(agent_params: dict, seed: int,
                          max_name_len: int = 120) -> str:
    """Build a readable, collision-safe run name capped for Windows paths."""
    if not agent_params:
        return f"default_seed{seed}"

    full_parts = [
        f"{k}{_short_value(v)}" for k, v in sorted(agent_params.items())
    ]
    full_name = "_".join(full_parts)
    full_with_seed = f"{full_name}_seed{seed}"
    if len(full_with_seed) <= max_name_len:
        return full_with_seed

    # Keep the most informative fields, then append short hash of full params.
    preferred_keys = [
        "lr",
        "hidden_layers",
        "n_simulations",
        "c_uct",
        "max_rollout_depth",
        "batch_size",
        "learning_starts",
        "max_iterations",
        "max_epochs",
        "early_stopping_patience",
        "policy_bonus_weight",
        "opening_moves_to_skip",
        "device",
    ]
    compact_parts = []
    for key in preferred_keys:
        if key in agent_params:
            compact_parts.append(f"{key}{_short_value(agent_params[key])}")

    if not compact_parts:
        compact_parts = full_parts[:4]

    digest = hashlib.sha1(full_with_seed.encode("utf-8")).hexdigest()[:10]
    compact = "_".join(compact_parts)
    candidate = f"{compact}_h{digest}_seed{seed}"
    if len(candidate) <= max_name_len:
        return candidate

    # If still too long, progressively trim largest parts, keep hash + seed.
    trimmed_parts = []
    remaining = max_name_len - len(f"_h{digest}_seed{seed}")
    for part in compact_parts:
        if remaining <= 0:
            break
        take = min(len(part), max(6, remaining))
        trimmed_parts.append(part[:take])
        remaining -= take + 1  # account for underscore
    trimmed = "_".join(trimmed_parts) or "params"
    return f"{trimmed}_h{digest}_seed{seed}"


def build_results_dir(base_dir: str, env_name: str, agent_name: str,
                      agent_params: dict, seed: int) -> str:
    """results/{env}/{agent}/{params}_seed{N}/"""
    param_str = _build_param_run_name(agent_params, seed)
    return os.path.join(base_dir, env_name, agent_name, param_str)


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
