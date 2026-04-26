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
                 opponent: Agent = None, agent_player: int = 0) -> dict:
        """Run the agent with frozen policy (training=False, greedy argmax).

        Returns per-game stats + counts of terminated/truncated games.
        - terminated: env reported done=True within max_steps
        - truncated : hit the internal max_steps cap (policy stuck in a loop)
        Draws/wins/losses are counted ONLY over terminated games.
        """
        rewards = []
        step_counts = []
        action_times = []
        terminations = []

        adversarial = env.is_adversarial() and opponent is not None

        for _ in range(num_games):
            if adversarial:
                r, s, times, term = self._play_adversarial(
                    agent, opponent, env, agent_player=agent_player)
            else:
                r, s, times, term = self._play_single(agent, env)
            rewards.append(r)
            step_counts.append(s)
            action_times.extend(times)
            terminations.append(term)

        rewards_arr = np.array(rewards)
        steps_arr = np.array(step_counts)
        times_arr = np.array(action_times) if action_times else np.array([0.0])

        terminated_count = sum(terminations)
        truncated_count = num_games - terminated_count

        result = {
            "mean_reward": float(np.mean(rewards_arr)),
            "std_reward": float(np.std(rewards_arr)),
            "mean_steps": float(np.mean(steps_arr)),
            "std_steps": float(np.std(steps_arr)),
            "mean_action_time_ms": float(np.mean(times_arr) * 1000),
            "std_action_time_ms": float(np.std(times_arr) * 1000),
            "num_games": num_games,
            "terminated_count": terminated_count,
            "truncated_count": truncated_count,
            "termination_rate": terminated_count / num_games,
            "truncation_rate": truncated_count / num_games,
        }
        if adversarial:
            wins = sum(1 for r, t in zip(rewards, terminations) if t and r > 0)
            draws = sum(1 for r, t in zip(rewards, terminations) if t and r == 0)
            losses = sum(1 for r, t in zip(rewards, terminations) if t and r < 0)
            result["win_rate"] = wins / num_games
            result["draw_rate"] = draws / num_games
            result["loss_rate"] = losses / num_games
            result["agent_player"] = agent_player
        return result

    def evaluate_both_sides(self, agent: Agent, env: Environment,
                            num_games: int, opponent: Agent) -> dict:
        """Adversarial eval with balanced player positions.

        Splits num_games in half: agent plays as P0 then as P1 vs same opponent.
        Returns per-side results + combined.
        """
        half = num_games // 2
        result_p0 = self.evaluate(agent, env, half,
                                  opponent=opponent, agent_player=0)
        result_p1 = self.evaluate(agent, env, num_games - half,
                                  opponent=opponent, agent_player=1)

        n0, n1 = result_p0["num_games"], result_p1["num_games"]
        total = n0 + n1

        def wavg(key):
            return (result_p0[key] * n0 + result_p1[key] * n1) / total

        combined = {
            "mean_reward": wavg("mean_reward"),
            "mean_steps": wavg("mean_steps"),
            "mean_action_time_ms": wavg("mean_action_time_ms"),
            "win_rate": wavg("win_rate"),
            "draw_rate": wavg("draw_rate"),
            "loss_rate": wavg("loss_rate"),
            "termination_rate": wavg("termination_rate"),
            "truncation_rate": wavg("truncation_rate"),
            "num_games": total,
        }
        return {"as_p0": result_p0, "as_p1": result_p1, "combined": combined}

    def _play_single(self, agent, env, max_steps=10_000):
        """Returns (reward, steps, action_times, terminated).

        terminated = True iff env set done within max_steps.
        """
        state = env.reset()
        total_reward = 0.0
        steps = 0
        action_times = []
        terminated = False

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
                terminated = True
                break

        return total_reward, steps, action_times, terminated

    def _play_adversarial(self, agent, opponent, env, max_steps=10_000,
                          agent_player: int = 0):
        """Returns (agent_reward, steps, action_times, terminated)."""
        state = env.reset()
        agent_reward = 0.0
        steps = 0
        action_times = []
        terminated = False

        for _ in range(max_steps):
            player = env.current_player()
            available = env.available_actions()
            if not available:
                break

            if player == agent_player:
                t0 = time.perf_counter()
                action = agent.act(state, available, training=False)
                action_times.append(time.perf_counter() - t0)
            else:
                action = opponent.act(state, available, training=False)

            next_state, reward, done = env.step(action)
            steps += 1

            if player == agent_player:
                agent_reward += reward
            elif done:
                agent_reward -= reward

            state = next_state
            if done:
                terminated = True
                break

        return agent_reward, steps, action_times, terminated
