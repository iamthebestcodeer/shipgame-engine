from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WeaponSpec:
    name: str
    kind: str
    speed: float
    damage: float
    radius: float
    lifetime: float
    reload: float
    submerged: bool


@dataclass(frozen=True)
class ShipSpec:
    name: str
    kind: str
    level: int
    speed: float
    acceleration: float
    turn_rate: float
    radius: float
    health: float
    draft: float
    depth: float
    weapons: tuple[WeaponSpec, ...]
    upgrade: str | None = None


SHIPS: dict[str, ShipSpec] = {
    "corvette": ShipSpec(
        name="Corvette",
        kind="boat",
        level=1,
        speed=22.0,
        acceleration=14.0,
        turn_rate=1.2,
        radius=15.0,
        health=100.0,
        draft=1.0,
        depth=0.0,
        weapons=(
            WeaponSpec("shell", "shell", 95.0, 8.0, 2.0, 3.0, 0.8, False),
            WeaponSpec("torpedo", "torpedo", 48.0, 18.0, 3.0, 5.0, 2.0, True),
        ),
        upgrade="destroyer",
    ),
    "destroyer": ShipSpec(
        name="Destroyer",
        kind="boat",
        level=2,
        speed=19.0,
        acceleration=11.0,
        turn_rate=0.85,
        radius=22.0,
        health=180.0,
        draft=1.4,
        depth=0.0,
        weapons=(
            WeaponSpec("shell", "shell", 105.0, 14.0, 2.5, 3.0, 0.9, False),
            WeaponSpec("torpedo", "torpedo", 50.0, 24.0, 3.0, 5.0, 2.2, True),
        ),
        upgrade="battleship",
    ),
    "battleship": ShipSpec(
        name="Battleship",
        kind="boat",
        level=3,
        speed=16.0,
        acceleration=8.0,
        turn_rate=0.55,
        radius=34.0,
        health=320.0,
        draft=2.0,
        depth=0.0,
        weapons=(
            WeaponSpec("shell", "shell", 115.0, 28.0, 3.5, 3.5, 1.2, False),
            WeaponSpec("torpedo", "torpedo", 48.0, 30.0, 3.5, 5.0, 2.8, True),
        ),
    ),
    "submarine": ShipSpec(
        name="Submarine",
        kind="submarine",
        level=2,
        speed=18.0,
        acceleration=9.0,
        turn_rate=0.8,
        radius=17.0,
        health=150.0,
        draft=0.0,
        depth=1.0,
        weapons=(
            WeaponSpec("torpedo", "torpedo", 48.0, 25.0, 3.0, 5.0, 2.0, True),
            WeaponSpec("shell", "shell", 95.0, 8.0, 2.0, 3.0, 1.0, True),
        ),
        upgrade="destroyer",
    ),
}


@dataclass(frozen=True)
class GameConfig:
    world_radius: float = 480.0
    dt: float = 0.1
    max_steps: int = 3000
    contact_slots: int = 16
    terrain_rays: int = 16
    crate_count: int = 24
    bot_count: int = 5
    seed: int = 7

    @classmethod
    def from_file(cls, path: Path) -> GameConfig:
        values: dict[str, Any] = json.loads(path.read_text())
        return cls(**values)

    @classmethod
    def default(cls) -> GameConfig:
        return cls()
