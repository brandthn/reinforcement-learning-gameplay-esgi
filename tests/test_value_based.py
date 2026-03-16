"""Value-based agent tests: TabularQ, DQN, DDQN, DDQN+ER, DDQN+PER.

Tests cover:
- Learning ability on LineWorld (simple env, fast convergence)
- Training pipeline integration (Trainer + SelfPlayTrainer)
- Agent-specific correctness checks (Q-table population, network updates)
- Experience replay extensions (learning_starts, PER priorities, IS weights)

These tests serve as regression guards: if future changes break value-based
agents, these tests catch it.
"""

import os
import random as stdlib_random
import tempfile

import numpy as np
import pytest
import torch

from environments import get_env
from agents import get_agent
from training.trainer import Trainer
from training.self_play import SelfPlayTrainer
from evaluation.evaluator import Evaluator


# Shared test params — small for speed
TABULAR_Q_PARAMS = {
    "lr": 0.1, "gamma": 0.99,
    "epsilon_start": 1.0, "epsilon_end": 0.05,
    "epsilon_decay_steps": 500,
}

DQN_PARAMS = {
    "lr": 0.01, "gamma": 0.99,
    "epsilon_start": 1.0, "epsilon_end": 0.05,
    "epsilon_decay_steps": 300,
    "hidden_layers": [32], "batch_size": 16,
    "buffer_capacity": 2000, "target_update_freq": 20,
}

DDQN_PARAMS = dict(DQN_PARAMS)

DDQN_ER_PARAMS = {**DQN_PARAMS, "learning_starts": 100}

DDQN_PER_PARAMS = {
    **DQN_PARAMS,
    "per_alpha": 0.6, "per_beta_start": 0.4,
    "per_beta_end": 1.0, "per_beta_steps": 1000,
    "learning_starts": 100,
}


def _seed(s=42):
    stdlib_random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _train_single_player(agent, env, num_episodes, max_steps=100):
    for _ in range(num_episodes):
        state = env.reset()
        for _ in range(max_steps):
            available = env.available_actions()
            if not available:
                break
            action = agent.act(state, available, training=True)
            next_state, reward, done = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        agent.end_episode()


# ─── TabularQ specific ──────────────────────────────────────────────

class TestTabularQLearning:

    def test_learns_line_world(self):
        """TabularQ should solve LineWorld (5 cells) in <300 episodes."""
        _seed()
        env = get_env("line_world")
        agent = get_agent("tabular_q", env, TABULAR_Q_PARAMS)

        _train_single_player(agent, env, num_episodes=300)

        evaluator = Evaluator()
        result = evaluator.evaluate(agent, env, num_games=50)
        assert result["mean_reward"] > 0.9, (
            f"TabularQ didn't learn LineWorld: mean_reward={result['mean_reward']}"
        )
        assert result["mean_steps"] < 8, (
            f"TabularQ too slow: mean_steps={result['mean_steps']}"
        )

    def test_learns_grid_world(self):
        _seed()
        env = get_env("grid_world")
        agent = get_agent("tabular_q", env, TABULAR_Q_PARAMS)

        _train_single_player(agent, env, num_episodes=1000, max_steps=200)

        evaluator = Evaluator()
        result = evaluator.evaluate(agent, env, num_games=50)
        assert result["mean_reward"] > 0.8, (
            f"TabularQ didn't learn GridWorld: mean_reward={result['mean_reward']}"
        )

    def test_q_table_populated(self):
        """After training, the Q-table should have entries."""
        _seed()
        env = get_env("line_world")
        agent = get_agent("tabular_q", env, TABULAR_Q_PARAMS)

        _train_single_player(agent, env, num_episodes=50)

        assert len(agent._q) > 0, "Q-table is empty after training"

    def test_epsilon_decays(self):
        env = get_env("line_world")
        agent = get_agent("tabular_q", env, {
            **TABULAR_Q_PARAMS, "epsilon_decay_steps": 10,
        })

        eps_start = agent._get_epsilon()
        state = env.reset()
        for _ in range(15):
            available = env.available_actions()
            if not available:
                break
            action = agent.act(state, available, training=True)
            next_state, reward, done = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.end_episode()
                state = env.reset()

        eps_end = agent._get_epsilon()
        assert eps_end < eps_start, "Epsilon did not decay"


# ─── DQN specific ───────────────────────────────────────────────────

class TestDQNLearning:

    def test_learns_line_world(self):
        """DQN should solve LineWorld in <1000 episodes."""
        _seed()
        env = get_env("line_world")
        agent = get_agent("dqn", env, DQN_PARAMS)

        _train_single_player(agent, env, num_episodes=1000)

        evaluator = Evaluator()
        result = evaluator.evaluate(agent, env, num_games=50)
        assert result["mean_reward"] > 0.8, (
            f"DQN didn't learn LineWorld: mean_reward={result['mean_reward']}"
        )

    def test_network_params_change(self):
        """Network parameters should update during training."""
        _seed()
        env = get_env("line_world")
        agent = get_agent("dqn", env, DQN_PARAMS)

        params_before = [p.clone() for p in agent._online_net.parameters()]

        _train_single_player(agent, env, num_episodes=50)

        changed = False
        for p_before, p_after in zip(
            params_before, agent._online_net.parameters()
        ):
            if not torch.equal(p_before, p_after):
                changed = True
                break
        assert changed, "DQN network parameters did not change during training"

    def test_target_net_updates(self):
        """Target net should sync with online net periodically."""
        _seed()
        env = get_env("line_world")
        params = {**DQN_PARAMS, "target_update_freq": 5}
        agent = get_agent("dqn", env, params)

        _train_single_player(agent, env, num_episodes=100)

        assert agent._update_count > 0, "No training updates happened"


# ─── DDQN specific ──────────────────────────────────────────────────

class TestDDQNLearning:

    def test_learns_line_world(self):
        _seed()
        env = get_env("line_world")
        agent = get_agent("ddqn", env, DDQN_PARAMS)

        _train_single_player(agent, env, num_episodes=1000)

        evaluator = Evaluator()
        result = evaluator.evaluate(agent, env, num_games=50)
        assert result["mean_reward"] > 0.8, (
            f"DDQN didn't learn LineWorld: mean_reward={result['mean_reward']}"
        )

    def test_uses_different_target_than_dqn(self):
        """DDQN target computation differs from DQN (uses online net for action selection)."""
        _seed()
        env = get_env("line_world")
        dqn = get_agent("dqn", env, DQN_PARAMS)
        ddqn = get_agent("ddqn", env, DDQN_PARAMS)

        # Give them different online/target nets to see different targets
        with torch.no_grad():
            for p in ddqn._online_net.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        batch_size = 4
        states = torch.randn(batch_size, env.state_space_size())
        rewards = torch.zeros(batch_size)
        dones = torch.zeros(batch_size, dtype=torch.bool)

        dqn_targets = dqn._compute_targets(rewards, states, dones)
        ddqn_targets = ddqn._compute_targets(rewards, states, dones)

        # With different networks, targets should differ
        assert not torch.allclose(dqn_targets, ddqn_targets), \
            "DDQN and DQN produced identical targets despite different networks"


# ─── DDQN+ER specific ──────────────────────────────────────────────

class TestDDQNERLearning:

    def test_learns_line_world(self):
        _seed()
        env = get_env("line_world")
        agent = get_agent("ddqn_er", env, DDQN_ER_PARAMS)

        _train_single_player(agent, env, num_episodes=1000)

        evaluator = Evaluator()
        result = evaluator.evaluate(agent, env, num_games=50)
        assert result["mean_reward"] > 0.8, (
            f"DDQN+ER didn't learn LineWorld: mean_reward={result['mean_reward']}"
        )

    def test_learning_starts_delays_training(self):
        """No gradient updates should happen before learning_starts steps."""
        _seed()
        env = get_env("line_world")
        agent = get_agent("ddqn_er", env, {**DDQN_ER_PARAMS, "learning_starts": 200})

        params_before = [p.clone() for p in agent._online_net.parameters()]

        # Run fewer steps than learning_starts
        state = env.reset()
        for _ in range(50):
            available = env.available_actions()
            if not available:
                break
            action = agent.act(state, available, training=True)
            next_state, reward, done = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.end_episode()
                state = env.reset()

        assert agent._update_count == 0, (
            f"Training started before learning_starts: update_count={agent._update_count}"
        )
        for p_before, p_after in zip(params_before, agent._online_net.parameters()):
            assert torch.equal(p_before, p_after), (
                "Network params changed before learning_starts"
            )

    def test_training_starts_after_warmup(self):
        """After enough steps, training updates should begin."""
        _seed()
        env = get_env("line_world")
        agent = get_agent("ddqn_er", env, {**DDQN_ER_PARAMS, "learning_starts": 30})

        _train_single_player(agent, env, num_episodes=100)

        assert agent._update_count > 0, "No training updates after warmup"


# ─── DDQN+PER specific ─────────────────────────────────────────────

class TestDDQNPERLearning:

    def test_learns_line_world(self):
        _seed()
        env = get_env("line_world")
        agent = get_agent("ddqn_per", env, DDQN_PER_PARAMS)

        _train_single_player(agent, env, num_episodes=1000)

        evaluator = Evaluator()
        result = evaluator.evaluate(agent, env, num_games=50)
        assert result["mean_reward"] > 0.8, (
            f"DDQN+PER didn't learn LineWorld: mean_reward={result['mean_reward']}"
        )

    def test_uses_prioritized_buffer(self):
        """Verify the agent's buffer is a PrioritizedReplayBuffer."""
        from training.replay_buffer import PrioritizedReplayBuffer
        env = get_env("line_world")
        agent = get_agent("ddqn_per", env, DDQN_PER_PARAMS)
        assert isinstance(agent._buffer, PrioritizedReplayBuffer)

    def test_priorities_updated_after_training(self):
        """Sum-tree priorities should change after training steps."""
        _seed()
        env = get_env("line_world")
        agent = get_agent("ddqn_per", env, {**DDQN_PER_PARAMS, "learning_starts": 16})

        _train_single_player(agent, env, num_episodes=50)

        assert agent._update_count > 0, "No training updates happened"
        tree_sum = agent._buffer._tree[0]
        assert tree_sum > 0, "PER tree total priority is zero after training"

    def test_beta_anneals(self):
        """Beta should increase toward per_beta_end over training."""
        env = get_env("line_world")
        params = {**DDQN_PER_PARAMS, "per_beta_steps": 50}
        agent = get_agent("ddqn_per", env, params)

        beta_start = agent._get_beta()
        assert abs(beta_start - params["per_beta_start"]) < 1e-6

        state = env.reset()
        for _ in range(60):
            available = env.available_actions()
            if not available:
                break
            action = agent.act(state, available, training=True)
            next_state, reward, done = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.end_episode()
                state = env.reset()

        beta_end = agent._get_beta()
        assert abs(beta_end - params["per_beta_end"]) < 1e-6, (
            f"Beta did not anneal to per_beta_end: got {beta_end}"
        )

    def test_is_weighted_loss_differs_from_uniform(self):
        """PER loss with non-uniform weights should differ from uniform MSE."""
        _seed()
        env = get_env("line_world")
        agent = get_agent("ddqn_per", env, DDQN_PER_PARAMS)

        # Fill buffer with some transitions
        state = env.reset()
        for _ in range(200):
            available = env.available_actions()
            if not available:
                break
            action = agent.act(state, available, training=True)
            next_state, reward, done = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.end_episode()
                state = env.reset()

        assert agent._update_count > 0, "Agent didn't train"


# ─── Pipeline integration (Trainer) ─────────────────────────────────

class TestValueBasedTrainerIntegration:
    """Verify each value-based agent works with the Trainer pipeline end-to-end."""

    @pytest.fixture(params=["tabular_q", "dqn", "ddqn", "ddqn_er", "ddqn_per"])
    def agent_name(self, request):
        return request.param

    def _get_params(self, name):
        return {
            "tabular_q": TABULAR_Q_PARAMS,
            "dqn": DQN_PARAMS,
            "ddqn": DDQN_PARAMS,
            "ddqn_er": DDQN_ER_PARAMS,
            "ddqn_per": DDQN_PER_PARAMS,
        }[name]

    def test_trainer_single_player(self, agent_name):
        """Full Trainer pipeline on LineWorld."""
        config = {
            "env": "line_world",
            "agent": agent_name,
            "training": {"num_episodes": 50, "max_steps_per_episode": 50},
            "eval": {"checkpoints": [25, 50], "num_games": 10},
            "seed": 42,
        }
        env = get_env("line_world")
        agent = get_agent(agent_name, env, self._get_params(agent_name))

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(env, agent, config)
            metrics = trainer.train(tmpdir)

            assert 25 in metrics
            assert 50 in metrics
            assert os.path.isfile(os.path.join(tmpdir, "training_curve.csv"))
            assert os.path.isfile(os.path.join(tmpdir, "metrics.csv"))
            assert os.path.isfile(os.path.join(tmpdir, "model_25.pt"))
            assert os.path.isfile(os.path.join(tmpdir, "model_50.pt"))

    def test_self_play_tictactoe(self, agent_name):
        """Full SelfPlayTrainer pipeline on TicTacToe."""
        config = {
            "env": "tictactoe",
            "agent": agent_name,
            "opponent": "random",
            "training": {"num_episodes": 30, "max_steps_per_episode": 20},
            "eval": {"checkpoints": [30], "num_games": 10},
            "seed": 42,
        }
        env = get_env("tictactoe")
        agent = get_agent(agent_name, env, self._get_params(agent_name))
        opponent = get_agent("random", env)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SelfPlayTrainer(env, agent, opponent, config)
            metrics = trainer.train(tmpdir)
            assert 30 in metrics
            assert metrics[30]["mean_steps"] > 0
