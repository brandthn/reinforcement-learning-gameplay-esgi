"""Tests de conformite pour tous les environnements.

Verifie que chaque environnement implemente correctement l'interface Environment :
- reset() retourne la bonne forme
- available_actions() retourne une liste non vide
- step() retourne le bon tuple
- les parties aleatoires se terminent
- clone() produit des copies independantes
- les environnements a deux joueurs alternent correctement les joueurs
"""
import random
import numpy as np
import pytest

from environments import get_env

ALL_ENVS = ["line_world", "grid_world", "tictactoe", "bobail"]
ADVERSARIAL_ENVS = ["tictactoe", "bobail"]


@pytest.fixture(params=ALL_ENVS)
def env(request):
    return get_env(request.param)


class TestEnvironmentInterface:

    def test_reset_returns_correct_shape(self, env):
        state = env.reset()
        assert isinstance(state, np.ndarray)
        assert state.shape == (env.state_space_size(),)
        assert state.dtype == np.float32

    def test_available_actions_after_reset(self, env):
        env.reset()
        actions = env.available_actions()
        assert len(actions) > 0
        for a in actions:
            assert 0 <= a < env.action_space_size()

    def test_step_returns_correct_tuple(self, env):
        env.reset()
        action = random.choice(env.available_actions())
        result = env.step(action)
        assert len(result) == 3
        state, reward, done = result
        assert isinstance(state, np.ndarray)
        assert state.shape == (env.state_space_size(),)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_random_game_terminates(self, env):
        env.reset()
        done = False
        steps = 0
        max_steps = 5000
        while not done and steps < max_steps:
            action = random.choice(env.available_actions())
            _, _, done = env.step(action)
            steps += 1
        assert done, f"La partie ne s'est pas terminee en {max_steps} pas"

    def test_clone_is_independent(self, env):
        env.reset()
        action = random.choice(env.available_actions())
        clone = env.clone()

        # Avancer l'original
        env.step(action)
        original_state = env.state_description()
        clone_state = clone.state_description()

        assert not np.array_equal(original_state, clone_state), \
            "L'etat du clone a change quand l'original a ete modifie"

    def test_state_description_matches_reset(self, env):
        state_from_reset = env.reset()
        state_from_desc = env.state_description()
        np.testing.assert_array_equal(state_from_reset, state_from_desc)


class TestAdversarialEnvs:

    @pytest.fixture(params=ADVERSARIAL_ENVS)
    def adv_env(self, request):
        return get_env(request.param)

    def test_is_adversarial(self, adv_env):
        assert adv_env.is_adversarial() is True

    def test_current_player_after_reset(self, adv_env):
        adv_env.reset()
        assert adv_env.current_player() in (0, 1)

    def test_player_switches(self, adv_env):
        """Apres un tour complet, le joueur courant doit changer."""
        adv_env.reset()
        initial_player = adv_env.current_player()

        # Jouer des actions jusqu'a ce que le joueur change (Bobail necessite 1 deplacement
        # de piece au premier tour, les autres jeux necessitent 1 coup)
        done = False
        steps = 0
        while not done and steps < 100:
            action = random.choice(adv_env.available_actions())
            _, _, done = adv_env.step(action)
            steps += 1
            if adv_env.current_player() != initial_player:
                break

        if not done:
            assert adv_env.current_player() != initial_player, \
                "Le joueur n'a jamais change"


class TestSinglePlayerEnvs:

    @pytest.fixture(params=["line_world", "grid_world"])
    def sp_env(self, request):
        return get_env(request.param)

    def test_not_adversarial(self, sp_env):
        assert sp_env.is_adversarial() is False

    def test_current_player_always_zero(self, sp_env):
        sp_env.reset()
        for _ in range(10):
            actions = sp_env.available_actions()
            if not actions:
                break
            _, _, done = sp_env.step(random.choice(actions))
            assert sp_env.current_player() == 0
            if done:
                break
