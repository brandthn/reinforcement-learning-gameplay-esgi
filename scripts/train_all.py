#!/usr/bin/env python3
"""Batch training: run scripts/train.py for every YAML config in a directory."""

import sys
import os
import glob
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Train all configs in a directory")
    parser.add_argument("config_dir", nargs="?", default="configs",
                        help="Directory containing YAML config files")
    args = parser.parse_args()

    config_files = sorted(glob.glob(os.path.join(args.config_dir, "*.yaml")))

    if not config_files:
        print(f"No YAML configs found in {args.config_dir}/")
        return

    train_script = os.path.join(os.path.dirname(__file__), "train.py")
    print(f"Found {len(config_files)} config(s)\n")

    for config_path in config_files:
        print(f"=== {os.path.basename(config_path)} ===")
        result = subprocess.run(
            [sys.executable, train_script, config_path],
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
        print()


if __name__ == "__main__":
    main()
