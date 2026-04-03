"""Generic single-player training loop."""

import os
import csv

import yaml

from environments.base import Environment
from agents.base import Agent
from evaluation.evaluator import Evaluator


class Trainer:
    """Trains a single agent on a single-player environment.

    At configured checkpoints, freezes the policy and evaluates it.
    Writes training_curve.csv (per-episode) and metrics.csv (at checkpoints).
    """

    def __init__(self, env: Environment, agent: Agent, config: dict):
        self._env = env
        self._agent = agent
        self._config = config

        train_cfg = config["training"]
        self._num_episodes = train_cfg["num_episodes"]
        self._max_steps = train_cfg.get("max_steps_per_episode", 1000)

        eval_cfg = config.get("eval", {})
        self._checkpoints = set(eval_cfg.get("checkpoints", []))
        self._eval_games = eval_cfg.get("num_games", 100)

        self._evaluator = Evaluator()

    def train(self, results_dir: str) -> dict:
        os.makedirs(results_dir, exist_ok=True)

        with open(os.path.join(results_dir, "config.yaml"), "w") as f:
            yaml.dump(self._config, f, default_flow_style=False)

        curve_path = os.path.join(results_dir, "training_curve.csv")
        metrics_path = os.path.join(results_dir, "metrics.csv")

        with open(curve_path, "w", newline="") as f:
            csv.writer(f).writerow(["episode", "reward", "steps"])

        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "checkpoint", "mean_reward", "std_reward",
                "mean_steps", "std_steps",
                "mean_action_time_ms", "std_action_time_ms",
            ])

        all_metrics = {}

        with open(curve_path, "a", newline="") as curve_f, \
             open(metrics_path, "a", newline="") as metrics_f:
            curve_writer = csv.writer(curve_f)
            metrics_writer = csv.writer(metrics_f)

            for ep in range(1, self._num_episodes + 1):
                reward, steps = self._run_episode()
                curve_writer.writerow([ep, reward, steps])

                if ep in self._checkpoints:
                    curve_f.flush()
                    eval_result = self._evaluator.evaluate(
                        self._agent, self._env, self._eval_games
                    )
                    all_metrics[ep] = eval_result

                    metrics_writer.writerow([
                        ep,
                        eval_result["mean_reward"],
                        eval_result["std_reward"],
                        eval_result["mean_steps"],
                        eval_result["std_steps"],
                        eval_result["mean_action_time_ms"],
                        eval_result["std_action_time_ms"],
                    ])
                    metrics_f.flush()

                    model_path = os.path.join(results_dir, f"model_{ep}.pt")
                    self._agent.save(model_path)

        return all_metrics

    def _run_episode(self) -> tuple[float, int]:
        state = self._env.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(self._max_steps):
            available = self._env.available_actions()
            if not available:
                break

            action = self._agent.act(state, available, training=True)
            next_state, reward, done = self._env.step(action)
            self._agent.observe(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        self._agent.end_episode()
        return total_reward, steps
