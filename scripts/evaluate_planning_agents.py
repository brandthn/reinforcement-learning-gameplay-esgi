#!/usr/bin/env python3
"""Budget sweep evaluation for planning agents (RandomRollout, MCTS).

Justification : les agents de planification n'apprennent pas dans le temps,
la metrique "score au bout de N parties d'entrainement" est sans objet.
On la remplace par une courbe budget -> score (budget = n_simulations pour
MCTS ou n_rollouts_per_action pour RandomRollout).

Ecrit : results/{env}/{agent}/budget_sweep_seed{S}/metrics.csv + config.yaml
"""

import sys
import os
import csv
import random
import argparse

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environments import get_env
from agents import get_agent
from evaluation.evaluator import Evaluator


BUDGET_PARAM = {
    "random_rollout": "n_rollouts_per_action",
    "mcts": "n_simulations",
}

CSV_HEADER = [
    "budget",
    "mean_reward", "std_reward",
    "mean_steps", "std_steps",
    "mean_action_time_ms",
    "win_rate", "draw_rate", "loss_rate",
    "termination_rate", "truncation_rate",
    "num_games",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def run_sweep(config: dict, seed: int, out_root: str) -> None:
    set_seed(seed)

    env_name = config["env"]
    env_params = config.get("env_params", {})
    agent_name = config["agent"]
    base_agent_params = dict(config.get("agent_params", {}))
    budgets = config["budgets"]
    num_games = config.get("eval", {}).get("num_games", 100)

    if agent_name not in BUDGET_PARAM:
        raise ValueError(f"Unknown planning agent: {agent_name}")
    budget_key = BUDGET_PARAM[agent_name]

    env = get_env(env_name, **env_params)

    opponent = None
    if env.is_adversarial():
        opp_name = config.get("opponent", "random")
        opp_params = config.get("opponent_params", {})
        opponent = get_agent(opp_name, env, opp_params)

    run_dir = os.path.join(out_root, env_name, agent_name,
                           f"budget_sweep_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump({**config, "seed": seed}, f, default_flow_style=False)

    metrics_path = os.path.join(run_dir, "metrics.csv")
    evaluator = Evaluator()

    print(f"=== {env_name} / {agent_name} (seed={seed}) ===")
    print(f"  -> {run_dir}")

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for budget in budgets:
            agent_params = dict(base_agent_params)
            agent_params[budget_key] = int(budget)
            agent = get_agent(agent_name, env, agent_params)

            if opponent is not None:
                split = evaluator.evaluate_both_sides(
                    agent, env, num_games, opponent)
                res = split["combined"]
                row = [
                    budget,
                    f"{res['mean_reward']:.4f}", "",
                    f"{res['mean_steps']:.2f}", "",
                    f"{res['mean_action_time_ms']:.3f}",
                    f"{res['win_rate']:.3f}",
                    f"{res['draw_rate']:.3f}",
                    f"{res['loss_rate']:.3f}",
                    f"{res['termination_rate']:.3f}",
                    f"{res['truncation_rate']:.3f}",
                    res["num_games"],
                ]
                print(f"  [budget={budget:>5}] wr={res['win_rate']:.2%}"
                      f"  reward={res['mean_reward']:.3f}"
                      f"  t/move={res['mean_action_time_ms']:.1f}ms")
            else:
                res = evaluator.evaluate(agent, env, num_games)
                row = [
                    budget,
                    f"{res['mean_reward']:.4f}",
                    f"{res['std_reward']:.4f}",
                    f"{res['mean_steps']:.2f}",
                    f"{res['std_steps']:.2f}",
                    f"{res['mean_action_time_ms']:.3f}",
                    "", "", "",
                    f"{res['termination_rate']:.3f}",
                    f"{res['truncation_rate']:.3f}",
                    res["num_games"],
                ]
                print(f"  [budget={budget:>5}] reward={res['mean_reward']:.3f}"
                      f"  steps={res['mean_steps']:.1f}"
                      f"  t/move={res['mean_action_time_ms']:.1f}ms")

            writer.writerow(row)
            f.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Budget sweep for planning agents")
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed(s)")
    parser.add_argument("--budgets", type=int, nargs="+", default=None,
                        help="Override budget grid")
    parser.add_argument("--num-games", type=int, default=None,
                        help="Override eval.num_games")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.budgets is not None:
        config["budgets"] = args.budgets
    if args.num_games is not None:
        config.setdefault("eval", {})["num_games"] = args.num_games

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = config.get("seeds", [config.get("seed", 42)])

    for seed in seeds:
        run_sweep(config, seed, args.results_dir)


if __name__ == "__main__":
    main()
