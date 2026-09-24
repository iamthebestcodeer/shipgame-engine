"""CPU-first simulator and reinforcement-learning agent for naval combat."""

from .config import GameConfig
from .environment import NavalEnv
from .observation import Observation, StepResult
from .policy import ActorCritic
from .ppo import PPOConfig, PPOTrainer

__all__ = [
    "ActorCritic",
    "GameConfig",
    "NavalEnv",
    "Observation",
    "PPOConfig",
    "PPOTrainer",
    "StepResult",
]
