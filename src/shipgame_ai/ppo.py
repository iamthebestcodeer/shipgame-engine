from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from .environment import NavalEnv
from .policy import ActorCritic


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 20_000
    steps_per_update: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 64
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    device: str = "cpu"
    seed: int = 7


@dataclass
class Rollout:
    observations: torch.Tensor
    action_masks: torch.Tensor
    actions: torch.Tensor
    log_probabilities: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


@dataclass(frozen=True)
class UpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float


class PPOTrainer:
    def __init__(
        self,
        env: NavalEnv,
        config: PPOConfig | None = None,
        model: ActorCritic | None = None,
    ) -> None:
        self.env = env
        self.config = config or PPOConfig()
        torch.manual_seed(self.config.seed)
        self.device = torch.device(self.config.device)
        self.model = model or ActorCritic(env.dimension, env.action_count).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.total_steps = 0

    def collect(self, steps: int) -> Rollout:
        observations: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        actions: list[int] = []
        log_probabilities: list[float] = []
        values: list[float] = []
        rewards: list[float] = []
        dones: list[float] = []
        observation = self.env.reset()
        for _ in range(steps):
            observation_tensor = (
                torch.as_tensor(observation.vector()).unsqueeze(0).to(self.device)
            )
            mask_tensor = (
                torch.as_tensor(observation.mask()).unsqueeze(0).to(self.device)
            )
            with torch.no_grad():
                action, log_probability, _ = self.model.act(
                    observation_tensor, mask_tensor
                )
                value = self.model(observation_tensor)[1]
            next_result = self.env.step(int(action.item()))
            observations.append(observation_tensor.squeeze(0).cpu())
            masks.append(mask_tensor.squeeze(0).cpu())
            actions.append(int(action.item()))
            log_probabilities.append(float(log_probability.item()))
            values.append(float(value.item()))
            rewards.append(next_result.reward)
            dones.append(float(next_result.terminated or next_result.truncated))
            observation = next_result.observation
            if next_result.terminated or next_result.truncated:
                observation = self.env.reset()
        return Rollout(
            observations=torch.stack(observations),
            action_masks=torch.stack(masks),
            actions=torch.tensor(actions, dtype=torch.long),
            log_probabilities=torch.tensor(log_probabilities, dtype=torch.float32),
            values=torch.tensor(values, dtype=torch.float32),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            dones=torch.tensor(dones, dtype=torch.float32),
        )

    def advantages(self, rollout: Rollout) -> tuple[torch.Tensor, torch.Tensor]:
        returns = torch.zeros_like(rollout.rewards)
        last_gae = 0.0
        for index in reversed(range(len(rollout.rewards))):
            next_value = (
                0.0
                if index == len(rollout.rewards) - 1
                else rollout.values[index + 1].item()
            )
            non_terminal = 1.0 - rollout.dones[index].item()
            delta = (
                rollout.rewards[index].item()
                + self.config.gamma * next_value * non_terminal
                - rollout.values[index].item()
            )
            last_gae = (
                delta
                + self.config.gamma * self.config.gae_lambda * non_terminal * last_gae
            )
            returns[index] = last_gae
        advantages = returns - rollout.values
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        return returns, advantages

    def update(self, rollout: Rollout) -> UpdateStats:
        returns, advantages = self.advantages(rollout)
        observations = rollout.observations.to(self.device)
        action_masks = rollout.action_masks.to(self.device)
        actions = rollout.actions.to(self.device)
        old_log_probabilities = rollout.log_probabilities.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        count = len(actions)
        stats = UpdateStats(0.0, 0.0, 0.0, 0.0)
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(count)
            for start in range(0, count, self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                log_probabilities, entropy, values = self.model.evaluate_actions(
                    observations[indices], actions[indices], action_masks[indices]
                )
                ratio = torch.exp(log_probabilities - old_log_probabilities[indices])
                unclipped = ratio * advantages[indices]
                clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_ratio,
                        1.0 + self.config.clip_ratio,
                    )
                    * advantages[indices]
                )
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * (values - returns[indices]).pow(2).mean()
                entropy_loss = entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = (
                        old_log_probabilities[indices] - log_probabilities
                    ).mean()
                stats = UpdateStats(
                    float(policy_loss.item()),
                    float(value_loss.item()),
                    float(entropy_loss.item()),
                    float(approx_kl.item()),
                )
        return stats

    def train(self) -> list[UpdateStats]:
        history: list[UpdateStats] = []
        while self.total_steps < self.config.total_steps:
            steps = min(
                self.config.steps_per_update, self.config.total_steps - self.total_steps
            )
            rollout = self.collect(steps)
            self.total_steps += steps
            history.append(self.update(rollout))
        return history

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            path,
        )

    def load(self, path: Path) -> None:
        checkpoint: dict[str, Any] = torch.load(
            path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = int(checkpoint["total_steps"])
