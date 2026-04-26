#!/usr/bin/env python3
"""Batch training: run train.py or train_sweep.py for every YAML config."""

import sys
import os
import glob
import subprocess
import argparse

import yaml


def is_sweep_config(path: str) -> bool:
    with open(path) as f:
        config = yaml.safe_load(f)
    return "sweep" in config


def main():
    parser = argparse.ArgumentParser(
        description="Train all configs in a directory")
    parser.add_argument("config_dir", nargs="?", default="configs",
                        help="Directory containing YAML config files")
    parser.add_argument("--quick", action="store_true",
                        help="Fast iteration mode: 1 seed, fewer episodes, writes to results_dev/")
    parser.add_argument("--quick-episodes", type=int, default=1000,
                        help="Number of episodes in --quick mode (default: 1000)")
    args = parser.parse_args()

    extra_args = []
    if args.quick:
        extra_args += ["--quick", "--quick-episodes", str(args.quick_episodes)]

    config_files = sorted(glob.glob(os.path.join(args.config_dir, "**/*.yaml"),
                                     recursive=True))

    if not config_files:
        print(f"No YAML configs found in {args.config_dir}/")
        return

    scripts_dir = os.path.dirname(__file__)
    train_script = os.path.join(scripts_dir, "train.py")
    sweep_script = os.path.join(scripts_dir, "train_sweep.py")
    project_root = os.path.join(scripts_dir, "..")

    print(f"Found {len(config_files)} config(s)\n")

    for config_path in config_files:
        sweep = is_sweep_config(config_path)
        script = sweep_script if sweep else train_script
        label = "sweep" if sweep else "train"
        print(f"=== {config_path} ({label}) ===")
        result = subprocess.run(
            [sys.executable, script, config_path] + extra_args,
            cwd=project_root,
        )
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
        print()


if __name__ == "__main__":
    main()
