#!/usr/bin/env python3
"""Benchmark: run N random games per environment, print games/second."""

import sys
import os
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environments import ENV_REGISTRY, get_env

NUM_GAMES = 1000


def benchmark_env(env_name: str, n_games: int) -> tuple[float, float]:
    """Returns (games_per_second, avg_steps_per_game)."""
    env = get_env(env_name)
    total_steps = 0

    start = time.perf_counter()
    for _ in range(n_games):
        env.reset()
        done = False
        steps = 0
        while not done:
            action = random.choice(env.available_actions())
            _, _, done = env.step(action)
            steps += 1
        total_steps += steps
    elapsed = time.perf_counter() - start

    games_per_sec = n_games / elapsed
    avg_steps = total_steps / n_games
    return games_per_sec, avg_steps


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_GAMES
    print(f"Running {n} random games per environment...\n")
    print(f"{'Environment':<15} {'Games/sec':>12} {'Avg steps':>12}")
    print("-" * 42)

    for env_name in ENV_REGISTRY:
        gps, avg = benchmark_env(env_name, n)
        print(f"{env_name:<15} {gps:>12.1f} {avg:>12.1f}")


if __name__ == "__main__":
    main()
