"""Shared MLP builder used by DQN, DDQN, REINFORCE, PPO, etc."""

import torch.nn as nn


def build_mlp(input_dim: int, output_dim: int, hidden_layers: list[int],
              activation: str = "relu") -> nn.Sequential:
    activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}
    act_cls = activations[activation]

    layers = []
    prev = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)
