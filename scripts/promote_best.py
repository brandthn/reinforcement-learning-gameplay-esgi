#!/usr/bin/env python3
"""Promote the best trained model to results/{env}/{agent}/best/.

Usage:
  Auto:   uv run scripts/promote_best.py --env line_world --agent dqn
  Manual: uv run scripts/promote_best.py --run results/line_world/dqn/...seed42/ --checkpoint 100000
  All:    uv run scripts/promote_best.py --all
"""

import argparse
import csv
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def find_best_run(env_name: str, agent_name: str,
                  results_base: str = "results") -> tuple[str | None, int | None, float | None]:
    """Scan all runs for env/agent, return (run_dir, checkpoint, mean_reward)."""
    agent_dir = os.path.join(results_base, env_name, agent_name)
    if not os.path.isdir(agent_dir):
        return None, None, None

    best_run = None
    best_checkpoint = -1
    best_reward = float("-inf")

    for entry in os.listdir(agent_dir):
        run_dir = os.path.join(agent_dir, entry)
        if entry == "best" or not os.path.isdir(run_dir):
            continue

        metrics_path = os.path.join(run_dir, "metrics.csv")
        if not os.path.isfile(metrics_path):
            continue

        with open(metrics_path) as f:
            reader = csv.DictReader(f)
            latest_cp = -1
            latest_reward = float("-inf")
            for row in reader:
                cp = int(row["checkpoint"])
                reward = float(row["mean_reward"])
                if cp > latest_cp:
                    latest_cp = cp
                    latest_reward = reward

        if latest_cp < 0:
            continue
        if (latest_reward > best_reward or
                (latest_reward == best_reward and latest_cp > best_checkpoint)):
            best_reward = latest_reward
            best_checkpoint = latest_cp
            best_run = run_dir

    if best_run is None:
        return None, None, None
    return best_run, best_checkpoint, best_reward


def find_latest_model(run_dir: str) -> tuple[str | None, int]:
    best_path = None
    best_cp = -1
    for path in glob.glob(os.path.join(glob.escape(run_dir), "model_*.pt")):
        fname = os.path.basename(path)
        try:
            cp = int(fname.replace("model_", "").replace(".pt", ""))
        except ValueError:
            continue
        if cp > best_cp:
            best_cp = cp
            best_path = path
    return best_path, best_cp


def promote(run_dir: str, checkpoint: int | None = None,
            exit_on_error: bool = True) -> bool:
    """Copy config and model from run_dir to best/. Returns True on success."""
    agent_dir = os.path.dirname(run_dir)
    best_dir = os.path.join(agent_dir, "best")

    if checkpoint is not None:
        model_path = os.path.join(run_dir, f"model_{checkpoint}.pt")
        if not os.path.isfile(model_path):
            print(f"  Skip: {model_path} not found")
            if exit_on_error:
                sys.exit(1)
            return False
    else:
        model_path, checkpoint = find_latest_model(run_dir)
        if model_path is None:
            print(f"  Skip: no model files in {run_dir}")
            if exit_on_error:
                sys.exit(1)
            return False

    config_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(config_path):
        print(f"  Skip: {config_path} not found")
        if exit_on_error:
            sys.exit(1)
        return False

    os.makedirs(best_dir, exist_ok=True)
    shutil.copy2(config_path, os.path.join(best_dir, "config.yaml"))
    shutil.copy2(model_path, os.path.join(best_dir, "model.pt"))

    with open(os.path.join(best_dir, "source.txt"), "w") as f:
        f.write(f"run_dir: {run_dir}\n")
        f.write(f"checkpoint: {checkpoint}\n")
        f.write(f"model_file: {os.path.basename(model_path)}\n")

    print(f"  Promoted: {run_dir} (checkpoint {checkpoint})")
    print(f"       -> {best_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Promote best trained model to results/{env}/{agent}/best/")
    parser.add_argument("--env", help="Environment name")
    parser.add_argument("--agent", help="Agent name")
    parser.add_argument("--run", help="Specific run directory to promote")
    parser.add_argument("--checkpoint", type=int,
                        help="Specific checkpoint number (with --run)")
    parser.add_argument("--all", action="store_true",
                        help="Auto-promote best for every env/agent combo")
    args = parser.parse_args()

    if args.run:
        run_dir = args.run.rstrip("/")
        if not os.path.isdir(run_dir):
            print(f"Error: {run_dir} is not a directory")
            sys.exit(1)
        promote(run_dir, args.checkpoint)

    elif args.all:
        results_base = "results"
        if not os.path.isdir(results_base):
            print("No results/ directory found")
            sys.exit(1)

        count = 0
        for env_name in sorted(os.listdir(results_base)):
            env_dir = os.path.join(results_base, env_name)
            if not os.path.isdir(env_dir):
                continue
            for agent_name in sorted(os.listdir(env_dir)):
                agent_dir = os.path.join(env_dir, agent_name)
                if not os.path.isdir(agent_dir):
                    continue
                run_dir, checkpoint, reward = find_best_run(
                    env_name, agent_name, results_base)
                if run_dir is None:
                    continue
                print(f"\n{env_name}/{agent_name}: best reward={reward:.3f}")
                if promote(run_dir, checkpoint, exit_on_error=False):
                    count += 1

        print(f"\nPromoted {count} model(s) total.")

    elif args.env and args.agent:
        run_dir, checkpoint, reward = find_best_run(args.env, args.agent)
        if run_dir is None:
            print(f"No results found for {args.env}/{args.agent}")
            sys.exit(1)
        print(f"Best run: reward={reward:.3f} at checkpoint {checkpoint}")
        promote(run_dir, checkpoint)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
