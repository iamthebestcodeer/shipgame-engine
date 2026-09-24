from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import GameConfig
from .environment import NavalEnv
from .policy import ActorCritic
from .ppo import PPOConfig, PPOTrainer


def train(steps: int, seed: int, output: Path) -> None:
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    env = NavalEnv(GameConfig(seed=seed))
    trainer = PPOTrainer(env, PPOConfig(total_steps=steps, seed=seed))
    history = trainer.train()
    trainer.save(output)
    final = history[-1] if history else None
    if final is not None:
        print(
            f"steps={trainer.total_steps} policy_loss={final.policy_loss:.4f} "
            f"value_loss={final.value_loss:.4f} entropy={final.entropy:.4f}"
        )
    print(f"checkpoint={output}")


def evaluate(steps: int, seed: int, checkpoint: Path | None) -> None:
    env = NavalEnv(GameConfig(seed=seed))
    model = ActorCritic(env.dimension, env.action_count)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
    model.eval()
    observation = env.reset()
    total_reward = 0.0
    for _ in range(steps):
        action, _, _ = model.act(
            torch.as_tensor(observation.vector()).unsqueeze(0),
            torch.as_tensor(observation.mask()).unsqueeze(0),
            deterministic=True,
        )
        result = env.step(int(action.item()))
        total_reward += result.reward
        observation = result.observation
        if result.terminated or result.truncated:
            observation = env.reset()
    print(f"steps={steps} average_reward={total_reward / max(1, steps):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or evaluate the CPU naval agent"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--steps", type=int, default=20_000)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/agent.pt")
    )
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--steps", type=int, default=1_000)
    eval_parser.add_argument("--seed", type=int, default=7)
    eval_parser.add_argument("--checkpoint", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "train":
        train(arguments.steps, arguments.seed, arguments.output)
    else:
        evaluate(arguments.steps, arguments.seed, arguments.checkpoint)


if __name__ == "__main__":
    main()
