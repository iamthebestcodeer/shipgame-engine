from __future__ import annotations

import math

import numpy as np
import pytest

from shipgame_ai import GameConfig, NavalEnv


def test_reset_is_seeded_and_observation_is_json_safe() -> None:
    config = GameConfig(bot_count=2, crate_count=3)
    first = NavalEnv(config)
    second = NavalEnv(config)
    first_observation = first.reset(seed=42)
    second_observation = second.reset(seed=42)

    assert first_observation.values == second_observation.values
    assert first_observation.dimension == first.dimension
    assert first_observation.to_json()
    assert len(first_observation.action_mask) == first.action_count


def test_action_mask_and_invalid_action() -> None:
    env = NavalEnv(GameConfig(bot_count=1, crate_count=0))
    observation = env.reset(seed=3)

    assert any(observation.action_mask)
    assert len(observation.action_mask) == env.action_count
    with pytest.raises(ValueError, match="action must be"):
        env.step(env.action_count)


def test_player_can_collect_a_crate() -> None:
    env = NavalEnv(GameConfig(bot_count=0, crate_count=1))
    observation = env.reset(seed=5)
    crate = next(entity for entity in env.entities.values() if entity.name == "Crate")
    crate.x = env.player.x
    crate.y = env.player.y

    result = env.step(0)

    assert result.reward >= 2.0
    assert result.info["score"] == 20
    assert observation.dimension == result.observation.dimension


def test_weapon_reload_changes_action_mask() -> None:
    env = NavalEnv(GameConfig(bot_count=1, crate_count=0))
    env.reset(seed=2)
    shell_action = 1
    result = env.step(shell_action)

    events = result.info["events"]
    assert isinstance(events, list)
    assert "damage" in events or len(env.projectiles) > 0
    assert not env.action_mask()[shell_action]


def test_terrain_distance_is_bounded() -> None:
    env = NavalEnv(GameConfig(bot_count=1, crate_count=0, terrain_rays=8))
    observation = env.reset(seed=4)
    terrain_start = 12 + env.config.contact_slots * 10

    terrain_features = np.asarray(observation.values[terrain_start:])
    assert np.all((terrain_features >= 0.0) & (terrain_features <= 1.0))
    assert math.isfinite(float(terrain_features.mean()))
