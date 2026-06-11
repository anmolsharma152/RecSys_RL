from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def build_mlp(
    in_dim: int,
    hidden_dims: list[int],
    activation: str = "tanh",
) -> nn.Sequential:
    layers = []
    act = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "elu": nn.ELU()}[activation]
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act)
        prev = h
    return nn.Sequential(*layers)


class DiscreteActor(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dims: list[int], activation: str = "tanh"):
        super().__init__()
        self.backbone = build_mlp(state_dim, hidden_dims, activation)
        self.logits = nn.Linear(hidden_dims[-1], num_actions)

    def forward(self, x: torch.Tensor) -> Categorical:
        x = self.backbone(x)
        return Categorical(logits=self.logits(x))


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int], activation: str = "tanh"):
        super().__init__()
        self.backbone = build_mlp(state_dim, hidden_dims, activation)
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor) -> Normal:
        x = self.backbone(x)
        mean = self.mean(x)
        return Normal(mean, self.log_std.exp())


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: list[int], activation: str = "tanh"):
        super().__init__()
        self.backbone = build_mlp(state_dim, hidden_dims, activation)
        self.value = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.value(x).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = None,
        continuous: bool = False,
        activation: str = "tanh",
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        self.continuous = continuous

        if continuous:
            self.actor = GaussianActor(state_dim, action_dim, hidden_dims, activation)
        else:
            self.actor = DiscreteActor(state_dim, action_dim, hidden_dims, activation)
        self.critic = Critic(state_dim, hidden_dims, activation)

    def forward(self, x: torch.Tensor) -> tuple:
        return self.actor(x), self.critic(x)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def act(self, x: torch.Tensor) -> tuple:
        dist = self.actor(x)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(x)
        return action, log_prob, value

    def evaluate(self, x: torch.Tensor, action: torch.Tensor) -> tuple:
        dist = self.actor(x)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        value = self.critic(x)
        return log_prob, entropy, value
