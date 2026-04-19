"""
agents/policy_gradient/ppo.py

PPO (Proximal Policy Optimization) — A2C style.

Architecture:
  - Actor  : MLP → logits sur l'espace d'actions (masqués lors du sampling)
  - Critic : MLP → valeur scalaire V(s)

Mécanisme d'entraînement:
  - Collecte une trajectoire complète par épisode via observe()
  - À end_episode() : calcule les avantages (GAE) et les returns
  - Fait n_epochs passes de mini-batch SGD avec l'objectif PPO clipé
  - Reset la trajectoire entre chaque épisode (on-policy)

Conformité avec le projet:
  - Hérite de Agent (base.py)
  - Utilise build_mlp() de training/networks.py
  - Masquage d'actions dans act() uniquement (cf. D-012)
  - Format de sauvegarde .pt cohérent avec la famille value_based
  - Compatible SelfPlayTrainer (le pending_state est géré côté Trainer,
    l'agent reçoit ses transitions via observe() normalement)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional

# Adapter l'import selon l'emplacement réel de votre package
from agents.base import Agent
from training.networks import build_mlp


class PPOAgent(Agent):
    """
    PPO Proximal Policy Optimization — style A2C (on-policy, épisodique).

    Paramètres
    ----------
    state_size : int
        Dimension du vecteur d'état (fourni par env.state_space_size()).
    action_size : int
        Nombre d'actions (fourni par env.action_space_size()).
    lr : float
        Learning rate Adam (partagé actor + critic).
    gamma : float
        Facteur de discount.
    gae_lambda : float
        Paramètre λ du GAE (Generalized Advantage Estimation).
        0 → TD(0) pur (biais fort, variance faible).
        1 → Monte-Carlo pur (biais nul, variance forte).
        0.95 est le réglage standard.
    clip_epsilon : float
        Borne du clipping PPO (ratio ∈ [1-ε, 1+ε]).
    entropy_coef : float
        Coefficient du bonus d'entropie (encourage l'exploration).
    value_coef : float
        Coefficient de la perte critique dans la perte totale.
    hidden_layers : list[int]
        Architecture des MLP actor et critic.
    n_epochs : int
        Nombre de passes d'optimisation par trajectoire collectée.
    batch_size : int
        Taille des mini-batches lors des mises à jour.
    max_grad_norm : float
        Norme max pour le gradient clipping (stabilité).
    """

    name = "ppo"

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        hidden_layers: Optional[List[int]] = None,
        n_epochs: int = 4,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        self._state_size = state_size
        self._action_size = action_size
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._clip_epsilon = clip_epsilon
        self._entropy_coef = entropy_coef
        self._value_coef = value_coef
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._max_grad_norm = max_grad_norm
        self._device = torch.device(device)

        hidden = hidden_layers or [128, 128]

        # Deux réseaux séparés : plus facile à déboguer et à configurer
        # indépendamment (lr critique plus élevé si nécessaire)
        self._actor: nn.Sequential = build_mlp(state_size, action_size, hidden).to(self._device)
        self._critic: nn.Sequential = build_mlp(state_size, 1, hidden).to(self._device)

        self._optimizer = optim.Adam(
            list(self._actor.parameters()) + list(self._critic.parameters()),
            lr=lr,
        )

        # ── Stockage de trajectoire (réinitialisé à chaque épisode) ──────────
        # Chaque step stocke un dict avec les clés suivantes :
        #   state      : np.ndarray  — état observé avant l'action
        #   action     : int         — action choisie
        #   reward     : float       — récompense reçue
        #   log_prob   : float       — log π(a|s) au moment du sampling
        #   value      : float       — V(s) estimé par le critic
        #   done       : bool        — True si épisode terminé après ce step
        self._trajectory: list[dict] = []

        # Stocke temporairement les sorties de act() jusqu'au prochain observe()
        self._pending: Optional[dict] = None

        # Compteur global de steps (pour statistiques / logs externes)
        self._step_count: int = 0

    # ────────────────────────────────────────────────────────────────────────
    # Interface Agent
    # ────────────────────────────────────────────────────────────────────────

    def act(
        self,
        state: np.ndarray,
        available_actions: List[int],
        training: bool = True,
    ) -> int:
        """
        Sélectionne une action.

        En mode training : échantillonnage stochastique depuis la politique masquée.
        En mode inference : action greedy (argmax des logits masqués).

        Le masquage consiste à mettre -inf sur les logits des actions illégales
        avant le softmax, garantissant que p(action illégale) = 0.
        Cf. D-012 : le masquage ne s'applique qu'ici, pas dans _compute_gae().
        """
        # state_t = torch.FloatTensor(state).unsqueeze(0)  # (1, state_size)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0)

        with torch.no_grad():
            logits = self._actor(state_t).squeeze(0)          # (action_size,)
            value  = self._critic(state_t).squeeze().item()   # scalaire

        # ── Masquage des actions illégales ────────────────────────────────
        mask = torch.full((self._action_size,), float("-inf"))
        mask[available_actions] = 0.0
        masked_logits = logits + mask

        if training:
            probs = torch.softmax(masked_logits, dim=-1)
            dist  = torch.distributions.Categorical(probs)
            action_t = dist.sample()
            log_prob  = dist.log_prob(action_t).item()
            action    = action_t.item()
        else:
            # Greedy : pas d'exploration, politique figée
            action   = int(masked_logits.argmax().item())
            probs    = torch.softmax(masked_logits, dim=-1)
            log_prob = torch.log(probs[action] + 1e-8).item()

        # Stocker pour le prochain observe()
        self._pending = {
            "state"    : state.copy(),
            "action"   : action,
            "log_prob" : log_prob,
            "value"    : value,
        }

        self._step_count += 1
        return action

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Reçoit la transition (s, a, r, s', done) et l'ajoute à la trajectoire.

        Note : dans SelfPlayTrainer, le Trainer gère le "deferred observe"
        (pending_state) pour les jeux adversariaux. PPOAgent reçoit ses
        transitions exactement comme les agents value-based — aucune adaptation
        spécifique n'est nécessaire côté agent.
        """
        if self._pending is None:
            # Ne devrait pas arriver si act() est toujours appelé avant observe()
            return

        self._trajectory.append({
            "state"    : self._pending["state"],
            "action"   : self._pending["action"],
            "reward"   : float(reward),
            "log_prob" : self._pending["log_prob"],
            "value"    : self._pending["value"],
            "done"     : bool(done),
        })
        self._pending = None

    def end_episode(self) -> None:
        """
        Déclenché par le Trainer à la fin de chaque épisode.
        Lance la mise à jour PPO sur la trajectoire collectée, puis la réinitialise.
        """
        if len(self._trajectory) < 2:
            # Trajectoire trop courte pour un batch utile
            self._trajectory = []
            return

        self._update()
        self._trajectory = []

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor"      : self._actor.state_dict(),
                "critic"     : self._critic.state_dict(),
                "optimizer"  : self._optimizer.state_dict(),
                "step_count" : self._step_count,
            },
            path,
        )

    # def load(self, path: str) -> None:
    #     ckpt = torch.load(path, map_location="cpu")
    #     self._actor.load_state_dict(ckpt["actor"])
    #     self._critic.load_state_dict(ckpt["critic"])
    #     self._optimizer.load_state_dict(ckpt["optimizer"])
    #     self._step_count = ckpt.get("step_count", 0)
    
    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")

        self._actor.load_state_dict(ckpt["actor"])
        self._actor.to(self._device)

        self._critic.load_state_dict(ckpt["critic"])
        self._critic.to(self._device)

        self._optimizer.load_state_dict(ckpt["optimizer"])
        self._step_count = ckpt.get("step_count", 0)

    # ────────────────────────────────────────────────────────────────────────
    # Méthodes internes
    # ────────────────────────────────────────────────────────────────────────

    def _compute_gae(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcule les avantages via GAE (Generalized Advantage Estimation)
        et les returns cibles pour le critic.

        GAE(λ) :
            δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
            A_t = δ_t + γ · λ · (1 - done_t) · A_{t+1}

        Bootstrap : si le dernier step est done=True, V(s_T) = 0.
                    Sinon, on pourrait bootstrapper avec le critic — ici on
                    suppose que end_episode() est toujours appelé après done=True
                    (comportement standard des deux trainers du projet).

        Retourne
        --------
        advantages : np.ndarray (T,) — avantages GAE non normalisés
        returns    : np.ndarray (T,) — targets pour le critic (A_t + V(s_t))
        """
        T = len(self._trajectory)
        advantages = np.zeros(T, dtype=np.float32)

        gae        = 0.0
        next_value = 0.0   # Bootstrap à 0 car dernier step est done=True

        for t in reversed(range(T)):
            r    = self._trajectory[t]["reward"]
            v    = self._trajectory[t]["value"]
            done = float(self._trajectory[t]["done"])

            delta = r + self._gamma * next_value * (1.0 - done) - v
            gae   = delta + self._gamma * self._gae_lambda * (1.0 - done) * gae
            advantages[t] = gae
            next_value    = v   # V(s_t) devient next_value pour t-1

        # Returns = avantages + baseline (= targets du critic, cf. A3C/PPO paper)
        values  = np.array([tr["value"] for tr in self._trajectory], dtype=np.float32)
        returns = advantages + values
        return advantages, returns

    def _update(self) -> None:
        """
        Mise à jour PPO sur la trajectoire complète de l'épisode.

        Étapes :
          1. Calculer les avantages GAE et les returns
          2. Normaliser les avantages (stabilité numérique)
          3. Pour n_epochs itérations :
               a. Mélanger les indices
               b. Pour chaque mini-batch :
                    - Recalculer log π(a|s) et V(s) avec les poids courants
                    - Calculer le ratio r_t = π_new / π_old
                    - Perte acteur : -mean( min(r·A, clip(r,1±ε)·A) )
                    - Perte critique : MSE( V(s), returns )
                    - Bonus entropie : +entropy_coef · H(π)
                    - Rétropropagation + gradient clipping
        """
        advantages, returns = self._compute_gae()

        # ── Normalisation des avantages ───────────────────────────────────
        # Réduit la variance et stabilise l'entraînement
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / adv_std

        # ── Construction des tenseurs une seule fois ──────────────────────
        # states_t       = torch.FloatTensor(
        #     np.stack([tr["state"] for tr in self._trajectory])
        # )   # (T, state_size)

        states_t = torch.as_tensor(
            np.stack([tr["state"] for tr in self._trajectory])
            , dtype=torch.float32, device=self._device
        )
        
        # actions_t      = torch.LongTensor(
        #     [tr["action"] for tr in self._trajectory]
        # )   # (T,)
        actions_t    = torch.as_tensor([tr["action"] for tr in self._trajectory],
                                dtype=torch.long,    device=self._device)

        # old_log_probs_t = torch.FloatTensor(
        #     [tr["log_prob"] for tr in self._trajectory]
        # )   # (T,)
        old_log_probs_t = torch.as_tensor(
            [tr["log_prob"] for tr in self._trajectory]
            , dtype=torch.float32, device=self._device
        )# (T,)


        # returns_t = torch.FloatTensor(returns)      # (T,)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self._device)
        # advantages_t   = torch.FloatTensor(advantages)   # (T,)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self._device)

        T       = len(self._trajectory)
        indices = np.arange(T)

        for _ in range(self._n_epochs):
            np.random.shuffle(indices)

            for start in range(0, T, self._batch_size):
                batch_idx = indices[start : start + self._batch_size]

                # PPO est sensible aux mini-batches d'un seul élément
                # (BatchNorm ou calculs de variance instables) — on les saute
                if len(batch_idx) < 2:
                    continue

                b_states       = states_t[batch_idx]
                b_actions      = actions_t[batch_idx]
                b_old_lp       = old_log_probs_t[batch_idx]
                b_returns      = returns_t[batch_idx]
                b_advantages   = advantages_t[batch_idx]

                # ── Sorties des réseaux ──────────────────────────────────
                logits    = self._actor(b_states)           # (B, action_size)
                new_values = self._critic(b_states).squeeze(-1)  # (B,)

                # Log-probas et entropie de la distribution courante
                # Note : pas de masquage ici — le réseau apprend naturellement
                # à attribuer de faibles probabilités aux actions jamais choisies
                # (cf. D-012). Les actions stockées dans la trajectoire sont
                # toujours légales (garanties par act()).
                log_probs  = torch.log_softmax(logits, dim=-1)      # (B, A)
                new_lp     = log_probs.gather(
                    1, b_actions.unsqueeze(1)
                ).squeeze(1)                                          # (B,)
                entropy    = -(
                    torch.softmax(logits, dim=-1) * log_probs
                ).sum(dim=-1).mean()                                  # scalaire

                # ── Objectif PPO clipé ───────────────────────────────────
                # ratio = π_new(a|s) / π_old(a|s)
                ratio  = torch.exp(new_lp - b_old_lp)
                surr1  = ratio * b_advantages
                surr2  = torch.clamp(
                    ratio,
                    1.0 - self._clip_epsilon,
                    1.0 + self._clip_epsilon,
                ) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # ── Perte du critic ──────────────────────────────────────
                # Régression simple sur les returns (pas de clipping value
                # car nos environnements sont simples et la clipping value
                # peut ralentir l'apprentissage sur des épisodes courts)
                value_loss = nn.functional.mse_loss(new_values, b_returns)

                # ── Perte totale ─────────────────────────────────────────
                loss = (
                    actor_loss
                    + self._value_coef * value_loss
                    - self._entropy_coef * entropy
                )

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self._actor.parameters())
                    + list(self._critic.parameters()),
                    self._max_grad_norm,
                )
                self._optimizer.step()