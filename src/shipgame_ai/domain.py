from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum


class Faction(IntEnum):
    PLAYER = 0
    ENEMY = 1
    NEUTRAL = 2


class EntityKind(IntEnum):
    SHIP = 0
    CRATE = 1
    PROJECTILE = 2


@dataclass
class Entity:
    identifier: int
    kind: EntityKind
    name: str
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    health: float
    max_health: float
    faction: Faction
    ship_key: str | None = None
    level: int = 1
    submerged: bool = False
    active: bool = False
    speed: float = 0.0
    target_speed: float = 0.0
    turn_rate: float = 1.0
    acceleration: float = 1.0
    draft: float = 1.0
    depth: float = 0.0
    weapons: tuple[str, ...] = ()
    reloads: dict[str, float] | None = None
    alive: bool = True

    @property
    def speed_magnitude(self) -> float:
        return float((self.vx * self.vx + self.vy * self.vy) ** 0.5)

    @property
    def direction(self) -> float:
        if self.speed_magnitude < 1e-6:
            return 0.0
        return math.atan2(self.vy, self.vx)


@dataclass
class Projectile:
    identifier: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    damage: float
    kind: str
    faction: Faction
    lifetime: float
