from __future__ import annotations

import numpy as np

from shipgame_ai import GameConfig, NavalEnv, PPOConfig, PPOTrainer


def test_ppo_collect_and_update_finite() -> None:
    env = NavalEnv(GameConfig(bot_count=1, crate_count=2, max_steps=20))
    trainer = PPOTrainer(
        env,
        PPOConfig(
            total_steps=8,
            steps_per_update=4,
            update_epochs=1,
            minibatch_size=4,
        ),
    )

    history = trainer.train()

    assert trainer.total_steps == 8
    assert len(history) == 2
    assert np.isfinite(history[-1].policy_loss)
    assert np.isfinite(history[-1].value_loss)
    assert np.isfinite(history[-1].entropy)


def test_rollout_shapes_follow_action_mask() -> None:
    env = NavalEnv(GameConfig(bot_count=1, crate_count=1))
    trainer = PPOTrainer(env, PPOConfig(steps_per_update=4, minibatch_size=2))

    rollout = trainer.collect(4)

    assert rollout.observations.shape == (4, env.dimension)
    assert rollout.action_masks.shape == (4, env.action_count)
    assert rollout.actions.shape == (4,)
    assert rollout.rewards.shape == (4,)
