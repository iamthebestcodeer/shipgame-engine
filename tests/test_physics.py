from __future__ import annotations

import math

from shipgame_ai.config import GameConfig
from shipgame_ai.environment import NavalEnv
from shipgame_ai.physics import (
    Altitude,
    Transform,
    Velocity,
    boat_turn_rate,
    max_acceleration,
    sat_collision,
    terrain_collision,
)


def test_throttle_affects_movement() -> None:
    stopped = NavalEnv(GameConfig(bot_count=0, crate_count=0, terrain_rays=0))
    moving = NavalEnv(GameConfig(bot_count=0, crate_count=0, terrain_rays=0))
    stopped.reset(seed=1)
    moving.reset(seed=1)

    stopped.step(4 * 6 * 5)
    moving.step(4 * 6 * 5 + 4 * 6)

    assert moving.player.x > stopped.player.x


def test_reverse_speed_is_bounded() -> None:
    transform = Transform(x=0.0, y=0.0, direction=0.0)

    for _ in range(100):
        transform.apply_guidance(
            direction_target=0.0,
            velocity_target=-20.0,
            max_speed=20.0,
            dt=0.1,
            length=30.0,
        )

    assert transform.velocity == Velocity.from_mps(-20.0 / 3.0)


def test_velocity_and_altitude_are_quantized() -> None:
    velocity = Velocity.from_mps(1.015625)
    altitude = Altitude.from_meters(4.0)

    assert velocity == Velocity(32)
    assert velocity.to_mps() == 1.0
    assert altitude.units == 2
    assert altitude.to_meters() == 4.0


def test_turn_rate_uses_hull_length() -> None:
    short = Transform(x=0.0, y=0.0, direction=0.0)
    long = Transform(x=0.0, y=0.0, direction=0.0)

    short.apply_guidance(1.0, 0.0, 20.0, 0.1, 20.0)
    long.apply_guidance(1.0, 0.0, 20.0, 0.1, 40.0)

    assert short.direction > long.direction
    assert boat_turn_rate(20.0) == 1.125
    assert boat_turn_rate(40.0) == 0.625


def test_signed_velocity_integrates_against_heading() -> None:
    transform = Transform(
        x=1.0,
        y=2.0,
        direction=0.0,
        velocity=Velocity(-32),
    )

    transform.do_kinematics(0.1)

    assert transform.x == 0.9
    assert transform.y == 2.0


def test_boat_acceleration_matches_source_ticks() -> None:
    transform = Transform(x=0.0, y=0.0, direction=0.0)

    transform.apply_guidance(0.0, 20.0, 20.0, 0.1, 30.0)

    assert transform.velocity == Velocity(21)
    assert max_acceleration(0.1, 20.0) == 2.0 / 3.0


def test_sat_collision_uses_hull_orientation() -> None:
    parallel = sat_collision(
        0.0,
        0.0,
        0.0,
        0.0,
        40.0,
        8.0,
        15.0,
        20.0,
        0.0,
        0.0,
        0.0,
        40.0,
        8.0,
        15.0,
        0.1,
    )
    perpendicular = sat_collision(
        0.0,
        0.0,
        0.0,
        0.0,
        40.0,
        8.0,
        15.0,
        28.0,
        0.0,
        math.pi / 2.0,
        0.0,
        40.0,
        8.0,
        15.0,
        0.1,
    )

    assert parallel
    assert not perpendicular


def test_terrain_collision_uses_oriented_swept_hull() -> None:
    ahead = terrain_collision(
        0.0,
        0.0,
        0.0,
        10.0,
        2.0,
        8.0,
        [(1.8, 0.0, 0.1)],
        0.1,
    )
    beside = terrain_collision(
        0.0,
        0.0,
        0.0,
        10.0,
        2.0,
        8.0,
        [(0.5, 10.0, 0.1)],
        0.1,
    )

    assert ahead == (1.8, 0.0)
    assert beside is None
