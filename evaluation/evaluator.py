"""Evaluation a politique figee aux points de controle d'entrainement."""

import time
import numpy as np

from environments.base import Environment
from agents.base import Agent


class Evaluator:
    """Execute un agent entraine en mode inference et collecte les statistiques.

    Metriques collectees par evaluation :
    - moyenne/ecart-type de la recompense sur N parties
    - moyenne/ecart-type de la longueur d'episode (pas)
    - moyenne/ecart-type du temps reel par action (ms)
    """

    def evaluate(self, agent: Agent, env: Environment, num_games: int,
                 opponent: Agent = None) -> dict:
        rewards = []
        step_counts = []
        action_times = []

        adversarial = env.is_adversarial() and opponent is not None

        for _ in range(num_games):
            if adversarial:
                r, s, times = self._play_adversarial(agent, opponent, env)
            else:
                r, s, times = self._play_single(agent, env)
            rewards.append(r)
            step_counts.append(s)
            action_times.extend(times)

        rewards_arr = np.array(rewards)
        steps_arr = np.array(step_counts)
        times_arr = np.array(action_times) if action_times else np.array([0.0])

        return {
            "mean_reward": float(np.mean(rewards_arr)),
            "std_reward": float(np.std(rewards_arr)),
            "mean_steps": float(np.mean(steps_arr)),
            "std_steps": float(np.std(steps_arr)),
            "mean_action_time_ms": float(np.mean(times_arr) * 1000),
            "std_action_time_ms": float(np.std(times_arr) * 1000),
        }

    def _play_single(self, agent, env, max_steps=10_000):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        action_times = []

        for _ in range(max_steps):
            available = env.available_actions()
            if not available:
                break

            t0 = time.perf_counter()
            action = agent.act(state, available, training=False)
            action_times.append(time.perf_counter() - t0)

            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            if done:
                break

        return total_reward, steps, action_times

    def _play_adversarial(self, agent, opponent, env, max_steps=10_000):
        state = env.reset()
        agent_reward = 0.0
        steps = 0
        action_times = []

        for _ in range(max_steps):
            player = env.current_player()
            available = env.available_actions()
            if not available:
                break

            if player == 0:
                t0 = time.perf_counter()
                action = agent.act(state, available, training=False)
                action_times.append(time.perf_counter() - t0)
            else:
                action = opponent.act(state, available, training=False)

            next_state, reward, done = env.step(action)
            steps += 1

            if player == 0:
                agent_reward += reward
            elif done:
                agent_reward -= reward

            state = next_state
            if done:
                break

        return agent_reward, steps, action_times
