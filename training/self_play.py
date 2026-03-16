"""Two-player training loop with self-play."""

import os
import csv

import yaml

from environments.base import Environment
from agents.base import Agent
from evaluation.evaluator import Evaluator


class SelfPlayTrainer:
    """Trains a learning agent (player 0) against an opponent in adversarial envs.

    Uses deferred observe: the agent's observe() receives same-perspective states.
    When the agent acts, we store (state, action) and deliver the transition only
    when the agent gets to act again (correct next_state from its own perspective)
    or when the game ends.

    This ensures Q(s,a) targets use s' from the same player's viewpoint, which is
    required for off-policy methods (DQN family).
    """

    def __init__(self, env: Environment, agent: Agent, opponent: Agent,
                 config: dict):
        self._env = env
        self._agent = agent
        self._opponent = opponent
        self._config = config

        train_cfg = config["training"]
        self._num_episodes = train_cfg["num_episodes"]
        self._max_steps = train_cfg.get("max_steps_per_episode", 2000)

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

        for ep in range(1, self._num_episodes + 1):
            reward, steps = self._run_episode()

            with open(curve_path, "a", newline="") as f:
                csv.writer(f).writerow([ep, reward, steps])

            if ep in self._checkpoints:
                eval_result = self._evaluator.evaluate(
                    self._agent, self._env, self._eval_games,
                    opponent=self._opponent,
                )
                all_metrics[ep] = eval_result

                with open(metrics_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        ep,
                        eval_result["mean_reward"],
                        eval_result["std_reward"],
                        eval_result["mean_steps"],
                        eval_result["std_steps"],
                        eval_result["mean_action_time_ms"],
                        eval_result["std_action_time_ms"],
                    ])

                model_path = os.path.join(results_dir, f"model_{ep}.pt")
                self._agent.save(model_path)

        return all_metrics

    def _run_episode(self) -> tuple[float, int]:
        state = self._env.reset()
        done = False

        # Deferred-observe bookkeeping for the learning agent
        pending_state = None
        pending_action = None
        pending_reward = 0.0

        agent_reward = 0.0
        steps = 0

        for _ in range(self._max_steps):
            if done:
                break

            player = self._env.current_player()
            available = self._env.available_actions()
            if not available:
                break

            if player == 0:
                # Deliver previous pending transition (non-terminal)
                if pending_state is not None:
                    self._agent.observe(
                        pending_state, pending_action, pending_reward,
                        state, False,
                    )
                    pending_state = None
                    pending_reward = 0.0

                action = self._agent.act(state, available, training=True)
                pending_state = state
                pending_action = action
            else:
                action = self._opponent.act(state, available, training=False)

            next_state, reward, done = self._env.step(action)
            steps += 1

            if player == 0:
                pending_reward += reward
                agent_reward += reward

            # Terminal: flush pending transition
            if done and pending_state is not None:
                if player != 0:
                    # Opponent ended game: reward is from opponent's perspective
                    pending_reward -= reward
                    agent_reward -= reward
                self._agent.observe(
                    pending_state, pending_action, pending_reward,
                    next_state, True,
                )
                pending_state = None

            state = next_state

        self._agent.end_episode()
        return agent_reward, steps
