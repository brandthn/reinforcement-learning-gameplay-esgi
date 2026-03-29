"""Agent interface conformance tests.

Parameterized over all non-human agents in the registry.
Every agent must:
- Return actions within available_actions (both training modes)
- Accept observe() and end_episode() without crashing
- Survive save/load round-trip (loaded agent acts identically)

Adding a new agent? Add its name + test params to AGENT_TEST_PARAMS below.
"""

import os
import random
import tempfile

import numpy as np
import pytest

from environments import get_env
from agents import get_agent


# Maps agent name -> minimal params for instantiation.
# Small values so tests run fast.
AGENT_TEST_PARAMS = {
    "random": {},
}

ALL_ENVS = ["line_world", "grid_world", "tictactoe", "bobail"]
AGENT_NAMES = list(AGENT_TEST_PARAMS.keys())


def _make_agent(agent_name, env):
    return get_agent(agent_name, env, AGENT_TEST_PARAMS[agent_name])


# ─── Parameterized: every agent × every env ─────────────────────────

class TestAgentInterface:
    """Interface conformance: these tests MUST pass for every agent on every env."""

    @pytest.fixture(params=[
        (a, e) for a in AGENT_NAMES for e in ALL_ENVS
    ], ids=lambda x: f"{x[0]}-{x[1]}")
    def agent_env(self, request):
        agent_name, env_name = request.param
        env = get_env(env_name)
        agent = _make_agent(agent_name, env)
        return agent, env

    def test_act_returns_valid_action(self, agent_env):
        agent, env = agent_env
        state = env.reset()
        available = env.available_actions()
        action = agent.act(state, available, training=False)
        assert action in available

    def test_act_training_returns_valid_action(self, agent_env):
        agent, env = agent_env
        state = env.reset()
        available = env.available_actions()
        action = agent.act(state, available, training=True)
        assert action in available

    def test_observe_does_not_crash(self, agent_env):
        agent, env = agent_env
        state = env.reset()
        available = env.available_actions()
        action = agent.act(state, available, training=True)
        next_state, reward, done = env.step(action)
        agent.observe(state, action, reward, next_state, done)

    def test_end_episode_does_not_crash(self, agent_env):
        agent, env = agent_env
        env.reset()
        agent.end_episode()

    def test_never_picks_illegal_action(self, agent_env):
        """Run 50 steps of a random game, verify act() always returns legal action."""
        agent, env = agent_env
        state = env.reset()
        for _ in range(50):
            available = env.available_actions()
            if not available:
                break
            action = agent.act(state, available, training=True)
            assert action in available, (
                f"Illegal action {action}, available: {available}"
            )
            next_state, reward, done = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.end_episode()
                state = env.reset()


# ─── Save / Load round-trip ──────────────────────────────────────────

class TestSaveLoad:

    @pytest.fixture(params=AGENT_NAMES)
    def agent_name(self, request):
        return request.param

    def test_save_load_deterministic(self, agent_name):
        """After save+load, greedy action on the same state is identical."""
        env = get_env("line_world")
        agent = _make_agent(agent_name, env)

        # Do a few training steps so the agent has state
        state = env.reset()
        for _ in range(20):
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

        # Record greedy action
        test_state = env.reset()
        test_available = env.available_actions()
        action_before = agent.act(test_state, test_available, training=False)

        # Save, create fresh agent, load
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            save_path = f.name

        try:
            agent.save(save_path)
            agent2 = _make_agent(agent_name, env)
            agent2.load(save_path)
            action_after = agent2.act(test_state, test_available, training=False)
            assert action_before == action_after
        finally:
            os.unlink(save_path)
