#!/usr/bin/env python3
"""Batch evaluation of saved models in results/."""

import sys
import os
import argparse
import csv

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environments import get_env
from agents import get_agent
from evaluation.evaluator import Evaluator


def find_experiments(results_dir: str):
    """Walk results/ tree and yield (run_dir, config) for each experiment."""
    if not os.path.isdir(results_dir):
        return
    for env_dir in sorted(os.listdir(results_dir)):
        env_path = os.path.join(results_dir, env_dir)
        if not os.path.isdir(env_path):
            continue
        for agent_dir in sorted(os.listdir(env_path)):
            agent_path = os.path.join(env_path, agent_dir)
            if not os.path.isdir(agent_path):
                continue
            for run_dir in sorted(os.listdir(agent_path)):
                run_path = os.path.join(agent_path, run_dir)
                config_path = os.path.join(run_path, "config.yaml")
                if os.path.isfile(config_path):
                    with open(config_path) as f:
                        config = yaml.safe_load(f)
                    yield run_path, config


def evaluate_experiment(run_dir: str, config: dict, num_games: int):
    env_name = config["env"]
    env_params = config.get("env_params", {})
    agent_name = config["agent"]
    agent_params = config.get("agent_params", {})

    env = get_env(env_name, **env_params)
    agent = get_agent(agent_name, env, agent_params)

    model_files = sorted(
        [f for f in os.listdir(run_dir)
         if f.startswith("model_") and f.endswith(".pt")],
        key=lambda f: int(f.replace("model_", "").replace(".pt", "")),
    )

    opponent = None
    if env.is_adversarial():
        opp_name = config.get("opponent", "random")
        opp_params = config.get("opponent_params", {})
        opponent = get_agent(opp_name, env, opp_params)

    evaluator = Evaluator()
    metrics_path = os.path.join(run_dir, "metrics_reeval.csv")

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "checkpoint", "mean_reward", "std_reward",
            "mean_steps", "std_steps",
            "mean_action_time_ms", "std_action_time_ms",
        ])

        for model_file in model_files:
            checkpoint = int(
                model_file.replace("model_", "").replace(".pt", ""))
            agent.load(os.path.join(run_dir, model_file))

            result = evaluator.evaluate(agent, env, num_games,
                                        opponent=opponent)
            writer.writerow([
                checkpoint,
                result["mean_reward"], result["std_reward"],
                result["mean_steps"], result["std_steps"],
                result["mean_action_time_ms"], result["std_action_time_ms"],
            ])

            print(f"  [{checkpoint:>8} eps] "
                  f"reward={result['mean_reward']:.3f}"
                  f"+/-{result['std_reward']:.3f}  "
                  f"steps={result['mean_steps']:.1f}")

    return metrics_path


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate all saved models")
    parser.add_argument("--results-dir", default="results",
                        help="Root results directory")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Evaluation games per checkpoint")
    args = parser.parse_args()

    experiments = list(find_experiments(args.results_dir))
    if not experiments:
        print(f"No experiments found in {args.results_dir}/")
        return

    print(f"Found {len(experiments)} experiment(s)\n")

    for run_dir, config in experiments:
        env_name = config["env"]
        agent_name = config["agent"]
        run_name = os.path.basename(run_dir)
        print(f"=== {env_name} / {agent_name} ({run_name}) ===")
        evaluate_experiment(run_dir, config, args.num_games)
        print()


if __name__ == "__main__":
    main()
