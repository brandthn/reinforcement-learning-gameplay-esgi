#!/usr/bin/env python3
"""Batch re-evaluation of saved models.

For each run in results/, loads every checkpoint and evaluates the frozen policy.
- Single-player envs: standard eval
- Adversarial envs: balanced eval (N/2 games as P0, N/2 as P1 vs opponent)

Outputs:
- Per run: metrics_reeval.csv (one row per checkpoint)
- Global: results/summary.csv (one row per run x checkpoint x side)
"""

import sys
import os
import argparse
import csv

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environments import get_env
from agents import get_agent
from evaluation.evaluator import Evaluator


SUMMARY_COLUMNS = [
    "env", "agent", "run_dir", "seed", "checkpoint", "side",
    "mean_reward", "std_reward",
    "mean_steps", "std_steps",
    "mean_action_time_ms", "std_action_time_ms",
    "win_rate", "draw_rate", "loss_rate",
    "termination_rate", "truncation_rate",
    "num_games",
]


def find_experiments(results_dir: str):
    """Walk results/{env}/{agent}/{run}/ and yield (run_dir, config)."""
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


def list_checkpoints(run_dir: str) -> list[int]:
    files = [f for f in os.listdir(run_dir)
             if f.startswith("model_") and f.endswith(".pt")]
    return sorted(int(f.replace("model_", "").replace(".pt", ""))
                  for f in files)


def summary_row(env, agent, run_dir, seed, checkpoint, side, result):
    return {
        "env": env,
        "agent": agent,
        "run_dir": run_dir,
        "seed": seed,
        "checkpoint": checkpoint,
        "side": side,
        "mean_reward": result.get("mean_reward"),
        "std_reward": result.get("std_reward", ""),
        "mean_steps": result.get("mean_steps"),
        "std_steps": result.get("std_steps", ""),
        "mean_action_time_ms": result.get("mean_action_time_ms"),
        "std_action_time_ms": result.get("std_action_time_ms", ""),
        "win_rate": result.get("win_rate", ""),
        "draw_rate": result.get("draw_rate", ""),
        "loss_rate": result.get("loss_rate", ""),
        "termination_rate": result.get("termination_rate", ""),
        "truncation_rate": result.get("truncation_rate", ""),
        "num_games": result.get("num_games"),
    }


def evaluate_run(run_dir, config, num_games, summary_rows):
    env_name = config["env"]
    env_params = config.get("env_params", {})
    agent_name = config["agent"]
    agent_params = config.get("agent_params", {})
    seed = config.get("seed", "")

    env = get_env(env_name, **env_params)
    agent = get_agent(agent_name, env, agent_params)

    opponent = None
    if env.is_adversarial():
        opp_name = config.get("opponent", "random")
        opp_params = config.get("opponent_params", {})
        opponent = get_agent(opp_name, env, opp_params)

    evaluator = Evaluator()
    metrics_path = os.path.join(run_dir, "metrics_reeval.csv")

    adversarial = env.is_adversarial() and opponent is not None

    if adversarial:
        header = [
            "checkpoint",
            "mean_reward_combined", "mean_steps_combined",
            "win_rate_combined", "draw_rate_combined", "loss_rate_combined",
            "termination_rate_combined", "truncation_rate_combined",
            "mean_reward_p0", "win_rate_p0", "termination_rate_p0",
            "mean_reward_p1", "win_rate_p1", "termination_rate_p1",
            "mean_action_time_ms",
        ]
    else:
        header = [
            "checkpoint", "mean_reward", "std_reward",
            "mean_steps", "std_steps",
            "termination_rate", "truncation_rate",
            "mean_action_time_ms", "std_action_time_ms",
        ]

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ckpt in list_checkpoints(run_dir):
            agent.load(os.path.join(run_dir, f"model_{ckpt}.pt"))

            if adversarial:
                split = evaluator.evaluate_both_sides(
                    agent, env, num_games, opponent)
                r_p0, r_p1, r_c = split["as_p0"], split["as_p1"], split["combined"]
                writer.writerow([
                    ckpt,
                    f"{r_c['mean_reward']:.4f}",
                    f"{r_c['mean_steps']:.2f}",
                    f"{r_c['win_rate']:.3f}",
                    f"{r_c['draw_rate']:.3f}",
                    f"{r_c['loss_rate']:.3f}",
                    f"{r_c['termination_rate']:.3f}",
                    f"{r_c['truncation_rate']:.3f}",
                    f"{r_p0['mean_reward']:.4f}",
                    f"{r_p0['win_rate']:.3f}",
                    f"{r_p0['termination_rate']:.3f}",
                    f"{r_p1['mean_reward']:.4f}",
                    f"{r_p1['win_rate']:.3f}",
                    f"{r_p1['termination_rate']:.3f}",
                    f"{r_c['mean_action_time_ms']:.3f}",
                ])
                print(f"  [{ckpt:>6} eps] wr={r_c['win_rate']:.2%}"
                      f"  P0={r_p0['win_rate']:.2%}  P1={r_p1['win_rate']:.2%}"
                      f"  term={r_c['termination_rate']:.0%}")

                for side, res in [("combined", r_c), ("p0", r_p0), ("p1", r_p1)]:
                    summary_rows.append(summary_row(
                        env_name, agent_name, run_dir, seed, ckpt, side, res))
            else:
                result = evaluator.evaluate(agent, env, num_games)
                writer.writerow([
                    ckpt,
                    f"{result['mean_reward']:.4f}",
                    f"{result['std_reward']:.4f}",
                    f"{result['mean_steps']:.2f}",
                    f"{result['std_steps']:.2f}",
                    f"{result['termination_rate']:.3f}",
                    f"{result['truncation_rate']:.3f}",
                    f"{result['mean_action_time_ms']:.3f}",
                    f"{result['std_action_time_ms']:.3f}",
                ])
                print(f"  [{ckpt:>6} eps] reward={result['mean_reward']:.3f}"
                      f"  steps={result['mean_steps']:.1f}"
                      f"  term={result['termination_rate']:.0%}")

                summary_rows.append(summary_row(
                    env_name, agent_name, run_dir, seed, ckpt, "single", result))


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate all saved models (with player-swap for adversarial envs)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Eval games per checkpoint (split 50/50 between P0 and P1 for adversarial)")
    parser.add_argument("--summary", default=None,
                        help="Output summary CSV path (default: <results-dir>/summary.csv)")
    args = parser.parse_args()

    summary_path = args.summary or os.path.join(args.results_dir, "summary.csv")

    experiments = list(find_experiments(args.results_dir))
    if not experiments:
        print(f"No experiments found in {args.results_dir}/")
        return

    print(f"Found {len(experiments)} run(s)\n")
    summary_rows = []

    for run_dir, config in experiments:
        env_name = config["env"]
        agent_name = config["agent"]
        print(f"=== {env_name} / {agent_name} ({os.path.basename(run_dir)}) ===")
        try:
            evaluate_run(run_dir, config, args.num_games, summary_rows)
        except Exception as e:
            print(f"  FAILED: {e}")
        print()

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary written: {summary_path} ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
