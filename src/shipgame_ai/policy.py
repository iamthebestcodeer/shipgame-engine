from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(
        self, observation_dimension: int, action_count: int, hidden_size: int = 128
    ) -> None:
        super().__init__()
        self.observation_dimension = observation_dimension
        self.action_count = action_count
        self.network = nn.Sequential(
            nn.Linear(observation_dimension, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden_size, action_count)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.network(observations)
        return self.policy(features), self.value(features).squeeze(-1)

    def distribution(
        self, observations: torch.Tensor, action_mask: torch.Tensor
    ) -> Categorical:
        logits, _ = self(observations)
        return Categorical(logits=logits.masked_fill(~action_mask, -1e9))

    @torch.no_grad()
    def act(
        self,
        observations: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations, action_mask)
        action = (
            distribution.probs.argmax(dim=-1)
            if deterministic
            else distribution.sample()
        )
        return action, distribution.log_prob(action), distribution.entropy()

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations, action_mask)
        return (
            distribution.log_prob(actions),
            distribution.entropy(),
            self(observations)[1],
        )
