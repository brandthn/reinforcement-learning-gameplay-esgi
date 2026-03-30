"""Tests de conformite de l'interface Agent.

Parametrise sur tous les agents non-humains du registre.
Chaque agent doit :
- Retourner des actions dans available_actions (les deux modes d'entrainement)
- Accepter observe() et end_episode() sans planter
- Survivre a un aller-retour save/load (l'agent charge agit de maniere identique)

Ajout d'un nouvel agent ? Ajoutez son nom + parametres de test dans AGENT_TEST_PARAMS ci-dessous.
"""

import os
import random
import tempfile

import numpy as np
import pytest

from environments import get_env
from agents import get_agent


# Associe nom d'agent -> parametres minimaux pour l'instanciation.
# Petites valeurs pour que les tests soient rapides.
AGENT_TEST_PARAMS = {
    "random": {},
}

ALL_ENVS = ["line_world", "grid_world", "tictactoe", "bobail"]
AGENT_NAMES = list(AGENT_TEST_PARAMS.keys())


def _make_agent(agent_name, env):
    return get_agent(agent_name, env, AGENT_TEST_PARAMS[agent_name])


# ─── Parametrise : chaque agent x chaque environnement ─────────────────────────

class TestAgentInterface:
    """Conformite de l'interface : ces tests DOIVENT passer pour chaque agent sur chaque env."""

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
        """Jouer 50 pas d'une partie aleatoire, verifier que act() retourne toujours une action legale."""
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


# ─── Aller-retour Save / Load ──────────────────────────────────────────

class TestSaveLoad:

    @pytest.fixture(params=AGENT_NAMES)
    def agent_name(self, request):
        return request.param

    def test_save_load_deterministic(self, agent_name):
        """Apres save+load, l'action greedy sur le meme etat est identique."""
        env = get_env("line_world")
        agent = _make_agent(agent_name, env)

        # Faire quelques pas d'entrainement pour que l'agent ait un etat
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

        # Enregistrer l'action greedy
        test_state = env.reset()
        test_available = env.available_actions()
        action_before = agent.act(test_state, test_available, training=False)

        # Sauvegarder, creer un nouvel agent, charger
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
