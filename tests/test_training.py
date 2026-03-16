"""Tests for Phase 3 training & evaluation infrastructure.

Covers: networks, replay buffers, Trainer, SelfPlayTrainer, Evaluator,
and the full train.py config-driven pipeline.
"""

import os
import csv
import tempfile

import numpy as np
import pytest
import torch
import yaml

from training.networks import build_mlp
from training.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from training.trainer import Trainer
from training.self_play import SelfPlayTrainer
from evaluation.evaluator import Evaluator
from environments import get_env
from agents import get_agent


# ─── MLP builder ────────────────────────────────────────────────────

class TestBuildMLP:

    def test_output_shape(self):
        net = build_mlp(10, 4, [32, 16])
        x = torch.randn(5, 10)
        out = net(x)
        assert out.shape == (5, 4)

    def test_single_hidden_layer(self):
        net = build_mlp(3, 2, [8])
        x = torch.randn(1, 3)
        out = net(x)
        assert out.shape == (1, 2)

    def test_no_hidden_layers(self):
        net = build_mlp(5, 3, [])
        x = torch.randn(2, 5)
        out = net(x)
        assert out.shape == (2, 3)

    def test_tanh_activation(self):
        net = build_mlp(4, 2, [8], activation="tanh")
        x = torch.randn(1, 4)
        out = net(x)
        assert out.shape == (1, 2)

    def test_different_activations(self):
        for act in ["relu", "tanh", "elu"]:
            net = build_mlp(4, 2, [8], activation=act)
            assert net(torch.randn(1, 4)).shape == (1, 2)


# ─── Replay Buffer ──────────────────────────────────────────────────

class TestReplayBuffer:

    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0
        for i in range(10):
            buf.push(np.zeros(4), 0, 0.0, np.zeros(4), False)
        assert len(buf) == 10

    def test_sample_shapes(self):
        buf = ReplayBuffer(capacity=100)
        for _ in range(20):
            buf.push(np.ones(5), 1, 0.5, np.ones(5), False)
        states, actions, rewards, next_states, dones = buf.sample(8)
        assert states.shape == (8, 5)
        assert actions.shape == (8,)
        assert rewards.shape == (8,)
        assert next_states.shape == (8, 5)
        assert dones.shape == (8,)

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(25):
            buf.push(np.array([float(i)]), 0, 0.0, np.array([0.0]), False)
        assert len(buf) == 10

    def test_sample_content(self):
        buf = ReplayBuffer(capacity=10)
        buf.push(np.array([1.0, 2.0]), 3, 0.5, np.array([4.0, 5.0]), True)
        states, actions, rewards, next_states, dones = buf.sample(1)
        np.testing.assert_array_equal(states[0], [1.0, 2.0])
        assert actions[0] == 3
        assert rewards[0] == 0.5
        np.testing.assert_array_equal(next_states[0], [4.0, 5.0])
        assert dones[0] is np.True_


class TestPrioritizedReplayBuffer:

    def test_push_and_len(self):
        buf = PrioritizedReplayBuffer(capacity=64)
        for _ in range(10):
            buf.push(np.zeros(3), 0, 0.0, np.zeros(3), False)
        assert len(buf) == 10

    def test_sample_returns_weights_and_indices(self):
        buf = PrioritizedReplayBuffer(capacity=64)
        for _ in range(20):
            buf.push(np.ones(4), 1, 1.0, np.ones(4), False)
        result = buf.sample(8, beta=0.4)
        assert len(result) == 7
        states, actions, rewards, next_states, dones, indices, weights = result
        assert states.shape == (8, 4)
        assert indices.shape == (8,)
        assert weights.shape == (8,)
        assert np.all(weights > 0)

    def test_update_priorities(self):
        buf = PrioritizedReplayBuffer(capacity=64)
        for _ in range(20):
            buf.push(np.zeros(2), 0, 0.0, np.zeros(2), False)
        _, _, _, _, _, indices, _ = buf.sample(5, beta=0.4)
        td_errors = np.random.rand(5)
        buf.update_priorities(indices, td_errors)
        # No crash = success; priorities updated internally

    def test_capacity_limit(self):
        buf = PrioritizedReplayBuffer(capacity=8)
        for _ in range(20):
            buf.push(np.zeros(2), 0, 0.0, np.zeros(2), False)
        assert len(buf) == 8


# ─── Evaluator ───────────────────────────────────────────────────────

class TestEvaluator:

    def test_single_player_metrics(self):
        env = get_env("line_world")
        agent = get_agent("random", env)
        evaluator = Evaluator()
        result = evaluator.evaluate(agent, env, num_games=20)

        assert "mean_reward" in result
        assert "std_reward" in result
        assert "mean_steps" in result
        assert "std_steps" in result
        assert "mean_action_time_ms" in result
        assert "std_action_time_ms" in result
        assert result["mean_action_time_ms"] >= 0

    def test_adversarial_metrics(self):
        env = get_env("tictactoe")
        agent = get_agent("random", env)
        opponent = get_agent("random", env)
        evaluator = Evaluator()
        result = evaluator.evaluate(agent, env, num_games=20,
                                    opponent=opponent)

        assert "mean_reward" in result
        assert result["mean_steps"] > 0

    def test_adversarial_bobail(self):
        env = get_env("bobail")
        agent = get_agent("random", env)
        opponent = get_agent("random", env)
        evaluator = Evaluator()
        result = evaluator.evaluate(agent, env, num_games=10,
                                    opponent=opponent)
        assert result["mean_steps"] > 0


# ─── Trainer (single-player) ────────────────────────────────────────

class TestTrainer:

    def _make_config(self, num_episodes=50, checkpoints=None):
        return {
            "env": "line_world",
            "agent": "random",
            "training": {
                "num_episodes": num_episodes,
                "max_steps_per_episode": 100,
            },
            "eval": {
                "checkpoints": checkpoints or [25, 50],
                "num_games": 10,
            },
            "seed": 42,
        }

    def test_produces_output_files(self):
        config = self._make_config()
        env = get_env("line_world")
        agent = get_agent("random", env)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(env, agent, config)
            trainer.train(tmpdir)

            assert os.path.isfile(os.path.join(tmpdir, "config.yaml"))
            assert os.path.isfile(os.path.join(tmpdir, "training_curve.csv"))
            assert os.path.isfile(os.path.join(tmpdir, "metrics.csv"))

    def test_training_curve_has_correct_rows(self):
        config = self._make_config(num_episodes=30, checkpoints=[30])
        env = get_env("line_world")
        agent = get_agent("random", env)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(env, agent, config)
            trainer.train(tmpdir)

            with open(os.path.join(tmpdir, "training_curve.csv")) as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)

            assert header == ["episode", "reward", "steps"]
            assert len(rows) == 30

    def test_metrics_at_checkpoints(self):
        config = self._make_config(num_episodes=50, checkpoints=[20, 50])
        env = get_env("line_world")
        agent = get_agent("random", env)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(env, agent, config)
            metrics = trainer.train(tmpdir)

            assert 20 in metrics
            assert 50 in metrics
            assert "mean_reward" in metrics[20]

            with open(os.path.join(tmpdir, "metrics.csv")) as f:
                reader = csv.reader(f)
                next(reader)  # header
                rows = list(reader)
            assert len(rows) == 2
            assert rows[0][0] == "20"
            assert rows[1][0] == "50"

    def test_config_snapshot_saved(self):
        config = self._make_config()
        env = get_env("line_world")
        agent = get_agent("random", env)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(env, agent, config)
            trainer.train(tmpdir)

            with open(os.path.join(tmpdir, "config.yaml")) as f:
                saved = yaml.safe_load(f)
            assert saved["env"] == "line_world"
            assert saved["agent"] == "random"


# ─── SelfPlayTrainer (two-player) ───────────────────────────────────

class TestSelfPlayTrainer:

    def _make_config(self, num_episodes=50, checkpoints=None):
        return {
            "env": "tictactoe",
            "agent": "random",
            "opponent": "random",
            "training": {
                "num_episodes": num_episodes,
                "max_steps_per_episode": 20,
            },
            "eval": {
                "checkpoints": checkpoints or [25, 50],
                "num_games": 10,
            },
            "seed": 42,
        }

    def test_produces_output_files(self):
        config = self._make_config()
        env = get_env("tictactoe")
        agent = get_agent("random", env)
        opponent = get_agent("random", env)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SelfPlayTrainer(env, agent, opponent, config)
            trainer.train(tmpdir)

            assert os.path.isfile(os.path.join(tmpdir, "config.yaml"))
            assert os.path.isfile(os.path.join(tmpdir, "training_curve.csv"))
            assert os.path.isfile(os.path.join(tmpdir, "metrics.csv"))

    def test_training_curve_rows(self):
        config = self._make_config(num_episodes=30, checkpoints=[30])
        env = get_env("tictactoe")
        agent = get_agent("random", env)
        opponent = get_agent("random", env)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SelfPlayTrainer(env, agent, opponent, config)
            trainer.train(tmpdir)

            with open(os.path.join(tmpdir, "training_curve.csv")) as f:
                rows = list(csv.reader(f))
            # header + 30 data rows
            assert len(rows) == 31

    def test_metrics_at_checkpoints(self):
        config = self._make_config(num_episodes=50, checkpoints=[25, 50])
        env = get_env("tictactoe")
        agent = get_agent("random", env)
        opponent = get_agent("random", env)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SelfPlayTrainer(env, agent, opponent, config)
            metrics = trainer.train(tmpdir)

            assert 25 in metrics
            assert 50 in metrics

    def test_bobail_self_play(self):
        """SelfPlayTrainer works on Bobail (two-phase turns)."""
        config = {
            "env": "bobail",
            "agent": "random",
            "opponent": "random",
            "training": {
                "num_episodes": 20,
                "max_steps_per_episode": 500,
            },
            "eval": {
                "checkpoints": [20],
                "num_games": 5,
            },
            "seed": 42,
        }
        env = get_env("bobail")
        agent = get_agent("random", env)
        opponent = get_agent("random", env)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SelfPlayTrainer(env, agent, opponent, config)
            metrics = trainer.train(tmpdir)
            assert 20 in metrics
            assert metrics[20]["mean_steps"] > 0


# ─── End-to-end: train.py config pipeline ───────────────────────────

class TestTrainScript:
    """Tests train_single() from scripts/train.py as an integration test."""

    def test_line_world_end_to_end(self):
        config = {
            "env": "line_world",
            "agent": "random",
            "training": {
                "num_episodes": 30,
                "max_steps_per_episode": 100,
            },
            "eval": {
                "checkpoints": [15, 30],
                "num_games": 10,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config["results_dir"] = tmpdir

            # Import here to avoid sys.path side-effects at module level
            import importlib
            import sys
            scripts_dir = os.path.join(
                os.path.dirname(__file__), "..", "scripts")
            sys.path.insert(0, scripts_dir)
            from train import train_single
            sys.path.pop(0)

            metrics = train_single(config, seed=42)
            assert 15 in metrics
            assert 30 in metrics

            # Verify directory structure
            results_path = os.path.join(
                tmpdir, "line_world", "random", "default_seed42")
            assert os.path.isdir(results_path)
            assert os.path.isfile(
                os.path.join(results_path, "config.yaml"))
            assert os.path.isfile(
                os.path.join(results_path, "training_curve.csv"))
            assert os.path.isfile(
                os.path.join(results_path, "metrics.csv"))
