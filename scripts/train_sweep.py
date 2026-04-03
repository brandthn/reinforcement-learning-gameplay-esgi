#!/usr/bin/env python3
"""Expand a sweep config into concrete experiments and train each one.

A sweep config is a normal config with an additional `sweep:` section
that declares parameter axes. Each axis maps a dot-path key to a list
of values. The script computes the cartesian product of all axes and
calls train_single() for each combination.

Example sweep config:

    env: grid_world
    agent: dqn

    agent_params:
      lr: 0.001          # default, overridden by sweep
      gamma: 0.99
      hidden_layers: [64, 64]
      ...

    sweep:
      agent_params.lr: [0.001, 0.0005, 0.0001]
      agent_params.batch_size: [32, 64]

    training:
      num_episodes: 100000

    seeds: [42, 123, 456]

This produces 3 x 2 = 6 experiments, each run with 3 seeds = 18 total runs.
"""

import sys
import os
import copy
import itertools
import argparse

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from train import train_single, apply_quick_mode


def set_nested(d: dict, dotpath: str, value):
    """Set a value in a nested dict using dot notation (e.g. 'agent_params.lr')."""
    keys = dotpath.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def expand_sweep(config: dict) -> list[dict]:
    """Expand the sweep section into a list of concrete configs."""
    sweep = config.get("sweep")
    if not sweep:
        base = copy.deepcopy(config)
        base.pop("sweep", None)
        return [base]

    axes = list(sweep.items())
    axis_names = [name for name, _ in axes]
    axis_values = [values for _, values in axes]

    configs = []
    for combination in itertools.product(*axis_values):
        concrete = copy.deepcopy(config)
        concrete.pop("sweep")
        for dotpath, value in zip(axis_names, combination):
            set_nested(concrete, dotpath, value)
        configs.append(concrete)

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Expand a sweep config and train each combination")
    parser.add_argument("config", help="Path to sweep YAML config file")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print expanded configs without training")
    parser.add_argument("--quick", action="store_true",
                        help="Fast iteration mode: 1 seed, fewer episodes, writes to results_dev/")
    parser.add_argument("--quick-episodes", type=int, default=1000,
                        help="Number of episodes in --quick mode (default: 1000)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if "sweep" not in config:
        print("No 'sweep' section found -- nothing to expand. "
              "Use scripts/train.py for regular configs.")
        sys.exit(1)

    expanded = expand_sweep(config)
    print(f"Sweep expands to {len(expanded)} experiment(s)\n")

    if args.dry_run:
        for i, cfg in enumerate(expanded, 1):
            params = cfg.get("agent_params", {})
            print(f"  [{i}] {cfg['agent']} on {cfg['env']}  {params}")
        return

    for i, cfg in enumerate(expanded, 1):
        if args.quick:
            apply_quick_mode(cfg, args.quick_episodes)
        seeds_list = (
            [args.seed] if args.seed is not None
            else cfg.get("seeds", [cfg.get("seed", 42)])
        )
        params = cfg.get("agent_params", {})
        print(f"--- Experiment {i}/{len(expanded)}  {params} ---")
        for seed in seeds_list:
            train_single(cfg, seed)
        print()


if __name__ == "__main__":
    main()
