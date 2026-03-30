"""Boucle d'entrainement a deux joueurs avec self-play."""

import os
import csv

import yaml

from environments.base import Environment
from agents.base import Agent
from evaluation.evaluator import Evaluator


class SelfPlayTrainer:
    """Entraine un agent apprenant (joueur 0) contre un adversaire dans les env adversariels.

    Utilise l'observation differee : le observe() de l'agent recoit des etats de meme perspective.
    Quand l'agent agit, on stocke (etat, action) et on livre la transition seulement
    quand l'agent agit a nouveau (next_state correct de son propre point de vue)
    ou quand la partie se termine.

    Cela garantit que les cibles Q(s,a) utilisent s' du point de vue du meme joueur,
    ce qui est requis pour les methodes off-policy (famille DQN).
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
                        self._agent, self._env, self._eval_games,
                        opponent=self._opponent,
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
        done = False

        # Comptabilite de l'observation differee pour l'agent apprenant
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
                # Livrer la transition en attente precedente (non-terminale)
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

            # Terminal : vider la transition en attente
            if done and pending_state is not None:
                if player != 0:
                    # L'adversaire a termine la partie : la recompense est du point de vue de l'adversaire
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
