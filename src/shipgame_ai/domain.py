from __future__ import annotations

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


SUBMERGE_DELAY = 8
DEACTIVATE_DELAY = 5
SPAWN_PROTECTION_INITIAL = 20


@dataclass
class Entity:
    identifier: int
    kind: EntityKind
    name: str
    x: float
    y: float
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
    reloads: dict[str, int] | None = None
    alive: bool = True
    length: float = 0.0
    heading: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    submerge_requested: bool = False
    submerge_delay: int = 0
    active_requested: bool = False
    deactivate_delay: int = 0
    spawn_protection_remaining: int = 0

    @property
    def speed_magnitude(self) -> float:
        return abs(self.speed)

    @property
    def direction(self) -> float:
        return self.heading

    def set_submerge(self, requested: bool) -> None:
        if requested and not self.submerge_requested:
            self.submerge_delay = SUBMERGE_DELAY
        self.submerge_requested = requested
        self.submerged = requested and self.submerge_delay == 0

    def set_active(self, requested: bool) -> None:
        if not requested and self.active:
            self.deactivate_delay = DEACTIVATE_DELAY
        self.active_requested = requested
        self.active = requested or self.deactivate_delay > 0

    def update_tickers(self, ticks: int = 1) -> None:
        self.submerge_delay = max(0, self.submerge_delay - ticks)
        self.deactivate_delay = max(0, self.deactivate_delay - ticks)
        self.spawn_protection_remaining = max(
            0, self.spawn_protection_remaining - ticks
        )
        self.submerged = self.submerge_requested and self.submerge_delay == 0
        self.active = self.active_requested or self.deactivate_delay > 0

    def spawn_protection(self) -> float:
        return (
            SPAWN_PROTECTION_INITIAL - self.spawn_protection_remaining
        ) / SPAWN_PROTECTION_INITIAL

    def clear_spawn_protection(self) -> None:
        self.spawn_protection_remaining = 0

    def advance_reloads(self, ticks: int = 1) -> None:
        if self.reloads is None:
            return
        for name, cooldown in tuple(self.reloads.items()):
            self.reloads[name] = max(0, cooldown - ticks)

    def consume_reload(self, name: str, ticks: int) -> None:
        if self.reloads is not None:
            self.reloads[name] = ticks

    def repair(self, amount: float) -> None:
        self.health = min(self.max_health, self.health + amount)


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
