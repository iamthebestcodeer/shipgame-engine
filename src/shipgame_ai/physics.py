from __future__ import annotations

import math
from dataclasses import dataclass, field

VELOCITY_UNITS_PER_MPS = 32
ALTITUDE_METERS_PER_UNIT = 2
MAX_REVERSE_SCALE = -1.0 / 3.0
MAX_SPEED_ACCELERATION_DIVISOR = 3.0
MIN_ACCELERATION = 15.0
MAX_ACCELERATION = 500.0
BOAT_TURN_RATE_BASE = 0.125
BOAT_TURN_RATE_LENGTH_DIVISOR = 20.0
TORPEDO_U_TURN = math.radians(80.0)
ANGLE_REPRESENTATION = 32767


def _quantize(value: float, scale: float) -> int:
    return math.trunc(value * scale)


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return min(maximum, max(minimum, value))


def angle_from_radians(radians: float) -> float:
    representation = math.trunc(
        radians * ANGLE_REPRESENTATION / math.pi
    ) % 65536
    if representation > 32767:
        representation -= 65536
    return representation * math.pi / ANGLE_REPRESENTATION


def angle_difference(target: float, current: float) -> float:
    target_representation = math.trunc(
        target * ANGLE_REPRESENTATION / math.pi
    )
    current_representation = math.trunc(
        current * ANGLE_REPRESENTATION / math.pi
    )
    difference = (target_representation - current_representation) % 65536
    if difference > 32767:
        difference -= 65536
    return difference * math.pi / ANGLE_REPRESENTATION


@dataclass(frozen=True)
class Velocity:
    mps: int

    @classmethod
    def from_mps(cls, value: float) -> Velocity:
        return cls(
            _clamp(
                _quantize(value, VELOCITY_UNITS_PER_MPS),
                -32768,
                32767,
            )
        )

    def to_mps(self) -> float:
        return self.mps / VELOCITY_UNITS_PER_MPS

    @property
    def speed(self) -> float:
        return self.to_mps()

    def approach(self, target: Velocity, max_delta: float) -> Velocity:
        delta = target.to_mps() - self.to_mps()
        if abs(delta) <= max_delta:
            return target
        return Velocity.from_mps(self.to_mps() + math.copysign(max_delta, delta))


@dataclass(frozen=True)
class Altitude:
    units: int

    @classmethod
    def from_meters(cls, meters: float) -> Altitude:
        return cls(_quantize(meters, 1.0 / ALTITUDE_METERS_PER_UNIT))

    def to_meters(self) -> float:
        return self.units * ALTITUDE_METERS_PER_UNIT

    @property
    def is_submerged(self) -> bool:
        return self.units < 0

    @property
    def is_airborne(self) -> bool:
        return self.units > 0


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def boat_turn_rate(length: float) -> float:
    if length <= 0.0:
        raise ValueError("length must be positive")
    return BOAT_TURN_RATE_BASE + BOAT_TURN_RATE_LENGTH_DIVISOR / length


def max_acceleration(dt: float, max_speed: float) -> float:
    bounded_speed = min(MAX_ACCELERATION, max(MIN_ACCELERATION, max_speed))
    return bounded_speed * dt / MAX_SPEED_ACCELERATION_DIVISOR


def guidance_velocity(velocity_target: float, max_speed: float) -> Velocity:
    target = min(max_speed, max(max_speed * MAX_REVERSE_SCALE, velocity_target))
    return Velocity.from_mps(target)


@dataclass
class Transform:
    x: float
    y: float
    direction: float = 0.0
    velocity: Velocity = field(default_factory=lambda: Velocity(0))
    altitude: Altitude = field(default_factory=lambda: Altitude(0))

    def apply_guidance(
        self,
        direction_target: float,
        velocity_target: float,
        max_speed: float,
        dt: float,
        length: float,
        kind: str = "boat",
        sub_kind: str = "boat",
    ) -> None:
        if kind != "collectible" and sub_kind not in {
            "shell",
            "rocket",
            "mine",
            "depth_charge",
        }:
            delta_angle = angle_difference(direction_target, self.direction)
            if kind == "boat":
                turn_rate = boat_turn_rate(length)
            else:
                turn_rate = max(
                    0.3, 1.0 - abs(self.velocity.speed) / (1.0 + max_speed)
                )
            turn_max = min(turn_rate * dt, math.pi)
            self.direction = angle_from_radians(
                self.direction + max(-turn_max, min(turn_max, delta_angle))
            )
            if sub_kind == "torpedo" and abs(delta_angle) > TORPEDO_U_TURN:
                max_speed /= 3.0
        target = guidance_velocity(velocity_target, max_speed)
        max_delta = max_acceleration(dt, max_speed)
        self.velocity = self.velocity.approach(target, max_delta)

    def do_kinematics(self, dt: float) -> None:
        self.x += math.cos(self.direction) * self.velocity.to_mps() * dt
        self.y += math.sin(self.direction) * self.velocity.to_mps() * dt


def radius_collision(
    x: float,
    y: float,
    speed: float,
    radius: float,
    other_x: float,
    other_y: float,
    other_speed: float,
    other_radius: float,
    dt: float,
) -> bool:
    sweep = speed * dt
    other_sweep = other_speed * dt
    distance_squared = (x - other_x) ** 2 + (y - other_y) ** 2
    combined_radius = radius + other_radius + sweep + other_sweep
    return distance_squared <= combined_radius**2


def _sat_collision_half(
    x: float,
    y: float,
    other_x: float,
    other_y: float,
    axis_x: float,
    axis_y: float,
    other_axis_x: float,
    other_axis_y: float,
    length: float,
    width: float,
    other_length: float,
    other_width: float,
) -> bool:
    other_tangent_x = -other_axis_y
    other_tangent_y = other_axis_x
    other_points = (
        (
            other_x
            + other_axis_x * other_length
            + other_tangent_x * other_width,
            other_y
            + other_axis_y * other_length
            + other_tangent_y * other_width,
        ),
        (
            other_x
            + other_axis_x * other_length
            - other_tangent_x * other_width,
            other_y
            + other_axis_y * other_length
            - other_tangent_y * other_width,
        ),
        (
            other_x
            - other_axis_x * other_length
            - other_tangent_x * other_width,
            other_y
            - other_axis_y * other_length
            - other_tangent_y * other_width,
        ),
        (
            other_x
            - other_axis_x * other_length
            + other_tangent_x * other_width,
            other_y
            - other_axis_y * other_length
            + other_tangent_y * other_width,
        ),
    )
    for edge in range(4):
        dimension = length if edge % 2 == 0 else width
        minimum = x * axis_x + y * axis_y - dimension
        maximum = x * axis_x + y * axis_y + dimension
        all_less = True
        all_greater = True
        for point_x, point_y in other_points:
            projection = point_x * axis_x + point_y * axis_y
            all_less = all_less and projection < minimum
            all_greater = all_greater and projection > maximum
        if all_less or all_greater:
            return False
        axis_x, axis_y = -axis_y, axis_x
    return True


def sat_collision(
    x: float,
    y: float,
    direction: float,
    speed: float,
    length: float,
    width: float,
    radius: float,
    other_x: float,
    other_y: float,
    other_direction: float,
    other_speed: float,
    other_length: float,
    other_width: float,
    other_radius: float,
    dt: float,
) -> bool:
    sweep = speed * dt
    other_sweep = other_speed * dt
    distance_squared = (x - other_x) ** 2 + (y - other_y) ** 2
    combined_radius = radius + other_radius + sweep + other_sweep
    if distance_squared > combined_radius**2:
        return False
    axis_x = math.cos(direction)
    axis_y = math.sin(direction)
    other_axis_x = math.cos(other_direction)
    other_axis_y = math.sin(other_direction)
    x += axis_x * sweep * 0.5
    y += axis_y * sweep * 0.5
    other_x += other_axis_x * other_sweep * 0.5
    other_y += other_axis_y * other_sweep * 0.5
    half_length = (length + sweep) * 0.5
    half_width = width * 0.5
    other_half_length = (other_length + other_sweep) * 0.5
    other_half_width = other_width * 0.5
    return _sat_collision_half(
        x,
        y,
        other_x,
        other_y,
        axis_x,
        axis_y,
        other_axis_x,
        other_axis_y,
        half_length,
        half_width,
        other_half_length,
        other_half_width,
    ) and _sat_collision_half(
        other_x,
        other_y,
        x,
        y,
        other_axis_x,
        other_axis_y,
        axis_x,
        axis_y,
        other_half_length,
        other_half_width,
        half_length,
        half_width,
    )


def terrain_collision(
    x: float,
    y: float,
    direction: float,
    speed: float,
    length: float,
    width: float,
    terrain: list[tuple[float, float, float]],
    dt: float,
) -> tuple[float, float] | None:
    sweep = speed * dt
    axis_x = math.cos(direction)
    axis_y = math.sin(direction)
    tangent_x = -axis_y
    tangent_y = axis_x
    center_x = x + axis_x * sweep * 0.5
    center_y = y + axis_y * sweep * 0.5
    half_length = max(0.0, (length + sweep) * 0.5 * 0.9)
    half_width = width * 0.5 * 0.9
    collision_positions: list[tuple[float, float]] = []
    for terrain_x, terrain_y, radius in terrain:
        delta_x = terrain_x - center_x
        delta_y = terrain_y - center_y
        along = delta_x * axis_x + delta_y * axis_y
        across = delta_x * tangent_x + delta_y * tangent_y
        closest_x = center_x + axis_x * max(-half_length, min(half_length, along))
        closest_y = center_y + axis_y * max(-half_length, min(half_length, along))
        closest_x += tangent_x * max(-half_width, min(half_width, across))
        closest_y += tangent_y * max(-half_width, min(half_width, across))
        if math.hypot(terrain_x - closest_x, terrain_y - closest_y) <= radius:
            collision_positions.append((terrain_x, terrain_y))
    if not collision_positions:
        return None
    return (
        sum(position[0] for position in collision_positions)
        / len(collision_positions),
        sum(position[1] for position in collision_positions)
        / len(collision_positions),
    )


def apply_guidance(
    transform: Transform,
    direction_target: float,
    velocity_target: float,
    max_speed: float,
    dt: float,
    length: float,
) -> None:
    transform.apply_guidance(
        direction_target,
        velocity_target,
        max_speed,
        dt,
        length,
    )


def do_kinematics(transform: Transform, dt: float) -> None:
    transform.do_kinematics(dt)
