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


def test_submergence_request_waits_eight_ticks() -> None:
    env = NavalEnv(GameConfig(bot_count=0, crate_count=0))
    env.reset(seed=1)
    player = env.player
    player.set_submerge(True)

    for _ in range(7):
        env.step(0)
        assert not player.submerged

    env.step(0)
    assert player.submerged
    assert player.submerge_delay == 0


def test_sensor_deactivation_waits_five_ticks() -> None:
    env = NavalEnv(GameConfig(bot_count=0, crate_count=0))
    env.reset(seed=1)
    player = env.player
    player.set_active(True)
    env.step(0)
    player.set_active(False)

    for _ in range(4):
        env.step(0)
        assert player.active

    env.step(0)
    assert not player.active
    assert player.deactivate_delay == 0


def test_level_one_spawn_protection_expires_after_twenty_ticks() -> None:
    env = NavalEnv(GameConfig(bot_count=0, crate_count=0))
    env.reset(seed=1)
    player = env.player

    assert player.spawn_protection_remaining == 20
    player.health = 50.0
    env._damage(player, 10.0)
    assert player.health == 50.0

    for _ in range(20):
        env.step(0)

    assert player.spawn_protection_remaining == 0
    player.health = 50.0
    env._damage(player, 10.0)
    assert player.health == 40.0


def test_repair_uses_length_rate_and_terrain_eligibility() -> None:
    env = NavalEnv(GameConfig(bot_count=0, crate_count=0))
    env.reset(seed=1)
    env.terrain = []
    player = env.player

    player.health = 50.0
    player.length = 30.0
    env.step(0)
    assert player.health == 51.0

    player.health = 50.0
    player.length = 150.0
    env.step(0)
    assert player.health == 52.0

    player.health = 50.0
    player.length = 250.0
    env.step(0)
    assert player.health == 53.0

    env.terrain = [(player.x, player.y, player.radius + 1.0)]
    player.health = 50.0
    player.clear_spawn_protection()
    env.step(0)
    assert player.health == 47.5


def test_weapon_reload_uses_truncated_source_ticks_and_same_tick_reload() -> None:
    env = NavalEnv(GameConfig(bot_count=1, crate_count=0))
    env.reset(seed=2)
    enemy = next(entity for entity in env.entities.values() if entity.identifier != 0)
    enemy.x = 100.0
    enemy.y = 0.0

    env.step(1)

    assert env._reload_ticks(1.29) == 12
    assert env.player.reloads is not None
    assert env.player.reloads["shell"] == 7


def test_terrain_collision_pushes_signed_velocity_and_blocks_repair() -> None:
    env = NavalEnv(GameConfig(bot_count=0, crate_count=0))
    env.reset(seed=1)
    env.terrain = [(5.0, 0.0, 0.5)]
    player = env.player
    player.clear_spawn_protection()

    reward = env._resolve_terrain()
    env._repair_ships()

    assert reward == -2.0
    assert player.speed == -5.0
    assert player.vx == -5.0
    assert player.vy == 0.0
    assert player.health == 97.5
    assert "terrain" in env.last_events


def test_world_boundary_clamps_velocity_and_damages_in_same_tick() -> None:
    env = NavalEnv(GameConfig(bot_count=0, crate_count=0))
    env.reset(seed=1)
    player = env.player
    player.x = 480.1
    player.y = 0.0
    player.heading = 0.0
    player.speed = 1.0
    player.vx = 1.0
    player.vy = 0.0
    player.clear_spawn_protection()

    env._clamp_to_world(player)
    reward = env._resolve_boundaries()
    env._repair_ships()

    assert reward == -2.0
    assert player.x == 480.0
    assert player.speed == -10.0
    assert player.health == 90.0


def test_ship_collision_integration_uses_oriented_hulls_and_impulses() -> None:
    env = NavalEnv(GameConfig(bot_count=1, crate_count=0))
    env.reset(seed=1)
    env.terrain = []
    player = env.player
    enemy = env.entities[1]
    player.clear_spawn_protection()
    enemy.clear_spawn_protection()
    player.x = 0.0
    player.y = 0.0
    player.heading = 0.0
    enemy.x = 28.0
    enemy.y = 0.0
    enemy.heading = math.pi / 2.0

    env._resolve_ship_collisions()
    assert "collision" not in env.last_events

    enemy.x = 20.0
    enemy.heading = 0.0
    env._resolve_ship_collisions()

    assert "collision" in env.last_events
    assert player.health == 99.7
    assert player.speed == -2.5625
    assert enemy.speed == 1.53125


def test_submerged_hull_still_collides_with_land() -> None:
    env = NavalEnv(GameConfig(bot_count=0, crate_count=0))
    env.reset(seed=1)
    env.terrain = [(5.0, 0.0, 0.5)]
    player = env.player
    player.clear_spawn_protection()
    player.submerged = True

    env._resolve_terrain()

    assert player.speed == -5.0
    assert player.health == 97.5
    assert "terrain" in env.last_events
