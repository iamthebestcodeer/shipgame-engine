from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .config import SHIPS, GameConfig, WeaponSpec
from .domain import Entity, EntityKind, Faction, Projectile
from .observation import Observation, StepResult, observation_dimension

TURN_ACTIONS = 9
THROTTLE_ACTIONS = 5
AUXILIARY_ACTIONS = 6


@dataclass(frozen=True)
class DecodedAction:
    turn: float
    throttle: float
    auxiliary: int


class NavalEnv:
    action_count: ClassVar[int] = TURN_ACTIONS * THROTTLE_ACTIONS * AUXILIARY_ACTIONS

    def __init__(self, config: GameConfig | None = None) -> None:
        self.config = config or GameConfig.default()
        self.dimension = observation_dimension(
            self.config.contact_slots, self.config.terrain_rays
        )
        self.rng = np.random.default_rng(self.config.seed)
        self.entities: dict[int, Entity] = {}
        self.projectiles: dict[int, Projectile] = {}
        self.terrain: list[tuple[float, float, float]] = []
        self.score = 0
        self.elapsed = 0.0
        self.step_count = 0
        self.next_identifier = 1
        self.dead = False
        self.last_events: list[str] = []

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.entities.clear()
        self.projectiles.clear()
        self.score = 0
        self.elapsed = 0.0
        self.step_count = 0
        self.next_identifier = 1_000_000
        self.dead = False
        self.last_events = []
        self.terrain = self._create_terrain()
        self._create_ships()
        self._create_crates()
        return self._observation()

    def step(self, action: int) -> StepResult:
        if not 0 <= action < self.action_count:
            raise ValueError(f"action must be in [0, {self.action_count})")
        if self.dead:
            raise RuntimeError("reset the environment before stepping a dead episode")
        self.last_events = []
        self._advance_reloads()
        decoded = self._decode(action)
        self._control_player(decoded)
        for entity in self._bot_entities():
            self._control_bot(entity)
        for entity in list(self.entities.values()):
            if entity.alive and entity.kind is EntityKind.SHIP:
                self._drive(
                    entity,
                    decoded.turn if entity.identifier == 0 else self._bot_steer(entity),
                    0.8,
                )
        reward = 0.01
        for projectile in self.projectiles.values():
            self._move_projectile(projectile)
        reward += self._resolve_projectiles()
        reward += self._resolve_crates()
        reward += self._resolve_ship_collisions()
        reward += self._resolve_terrain()
        self._remove_dead_entities()
        self.step_count += 1
        self.elapsed += self.config.dt
        truncated = self.step_count >= self.config.max_steps and not self.dead
        if self.player.health <= 0.0:
            self.dead = True
        return StepResult(
            observation=self._observation(),
            reward=reward,
            terminated=self.dead,
            truncated=truncated,
            info=self._info(),
        )

    def action_mask(self) -> tuple[bool, ...]:
        player = self.player
        valid: list[bool] = []
        for _turn in range(TURN_ACTIONS):
            for _throttle in range(THROTTLE_ACTIONS):
                for auxiliary in range(AUXILIARY_ACTIONS):
                    valid.append(self._is_auxiliary_valid(player, auxiliary))
        return tuple(valid)

    def observation_space_dimension(self) -> int:
        return self.dimension

    def _create_ships(self) -> None:
        self.entities[0] = self._make_ship(0, "corvette", 0.0, 0.0, Faction.PLAYER)
        radius = self.config.world_radius * 0.62
        for index in range(self.config.bot_count):
            angle = 2.0 * math.pi * index / max(1, self.config.bot_count)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            ship_key = "submarine" if index == self.config.bot_count - 1 else "corvette"
            self.entities[index + 1] = self._make_ship(
                index + 1, ship_key, x, y, Faction.ENEMY
            )

    def _make_ship(
        self, identifier: int, ship_key: str, x: float, y: float, faction: Faction
    ) -> Entity:
        spec = SHIPS[ship_key]
        reloads = {weapon.name: 0.0 for weapon in spec.weapons}
        return Entity(
            identifier=identifier,
            kind=EntityKind.SHIP,
            name=spec.name,
            x=x,
            y=y,
            vx=0.0,
            vy=0.0,
            radius=spec.radius,
            health=spec.health,
            max_health=spec.health,
            faction=faction,
            ship_key=ship_key,
            level=spec.level,
            speed=0.0,
            target_speed=spec.speed,
            turn_rate=spec.turn_rate,
            acceleration=spec.acceleration,
            draft=spec.draft,
            depth=spec.depth,
            weapons=tuple(weapon.name for weapon in spec.weapons),
            reloads=reloads,
        )

    def _create_crates(self) -> None:
        for index in range(self.config.crate_count):
            angle = self.rng.uniform(0.0, 2.0 * math.pi)
            distance = self.rng.uniform(80.0, self.config.world_radius * 0.9)
            identifier = 100 + index
            self.entities[identifier] = Entity(
                identifier=identifier,
                kind=EntityKind.CRATE,
                name="Crate",
                x=distance * math.cos(angle),
                y=distance * math.sin(angle),
                vx=0.0,
                vy=0.0,
                radius=8.0,
                health=1.0,
                max_health=1.0,
                faction=Faction.NEUTRAL,
            )

    def _create_terrain(self) -> list[tuple[float, float, float]]:
        islands: list[tuple[float, float, float]] = []
        for _ in range(7):
            angle = self.rng.uniform(0.0, 2.0 * math.pi)
            distance = self.rng.uniform(100.0, self.config.world_radius * 0.85)
            radius = self.rng.uniform(20.0, 55.0)
            islands.append(
                (distance * math.cos(angle), distance * math.sin(angle), radius)
            )
        return islands

    def _control_player(self, action: DecodedAction) -> None:
        if action.auxiliary in (1, 2):
            self._fire(self.player, action.auxiliary - 1)
        if action.auxiliary == 3:
            self.player.submerged = not self.player.submerged
        if action.auxiliary == 4:
            self.player.active = not self.player.active
        if action.auxiliary == 5:
            self._upgrade_player()

    def _advance_reloads(self) -> None:
        for entity in self.entities.values():
            if entity.reloads is None:
                continue
            for name, cooldown in tuple(entity.reloads.items()):
                entity.reloads[name] = max(0.0, cooldown - self.config.dt)

    def _control_bot(self, entity: Entity) -> None:
        target = self._nearest_enemy(entity)
        if target is None:
            return
        if self._can_see(entity, target) and entity.reloads:
            ready = [
                name for name, cooldown in entity.reloads.items() if cooldown <= 0.0
            ]
            if ready:
                self._fire(entity, entity.weapons.index(ready[0]))
        if entity.ship_key == "submarine" and entity.health < entity.max_health * 0.55:
            entity.submerged = True
        else:
            entity.submerged = False

    def _drive(self, entity: Entity, turn: float, throttle: float) -> None:
        if entity.ship_key is None:
            return
        spec = SHIPS[entity.ship_key]
        if entity.speed_magnitude < 0.01:
            entity.vx = spec.speed
            entity.vy = 0.0
        current = math.atan2(entity.vy, entity.vx)
        target = current + turn * spec.turn_rate * self.config.dt
        target_speed = spec.speed * throttle
        entity.speed = float(
            np.clip(
                entity.speed
                + np.sign(target_speed - entity.speed)
                * min(
                    spec.acceleration * self.config.dt, abs(target_speed - entity.speed)
                ),
                0.0,
                spec.speed,
            )
        )
        entity.vx = math.cos(target) * entity.speed
        entity.vy = math.sin(target) * entity.speed
        entity.x += entity.vx * self.config.dt
        entity.y += entity.vy * self.config.dt
        self._clamp_to_world(entity)

    def _fire(self, entity: Entity, weapon_index: int) -> None:
        if weapon_index < 0 or weapon_index >= len(entity.weapons):
            return
        weapon = self._weapon_spec(entity, weapon_index)
        if weapon is None or entity.reloads is None:
            return
        if entity.reloads.get(weapon.name, 0.0) > 0.0:
            return
        if weapon.kind == "shell" and entity.submerged:
            return
        target = self._nearest_enemy(entity)
        if target is None:
            return
        dx = target.x - entity.x
        dy = target.y - entity.y
        distance = max(1.0, math.hypot(dx, dy))
        direction_x = dx / distance
        direction_y = dy / distance
        identifier = self.next_identifier
        self.next_identifier += 1
        self.projectiles[identifier] = Projectile(
            identifier=identifier,
            x=entity.x + direction_x * (entity.radius + weapon.radius),
            y=entity.y + direction_y * (entity.radius + weapon.radius),
            vx=direction_x * weapon.speed,
            vy=direction_y * weapon.speed,
            radius=weapon.radius,
            damage=weapon.damage,
            kind=weapon.kind,
            faction=entity.faction,
            lifetime=weapon.lifetime,
        )
        entity.reloads[weapon.name] = weapon.reload

    def _move_projectile(self, projectile: Projectile) -> None:
        projectile.x += projectile.vx * self.config.dt
        projectile.y += projectile.vy * self.config.dt
        projectile.lifetime -= self.config.dt
        if math.hypot(projectile.x, projectile.y) > self.config.world_radius + 40.0:
            projectile.lifetime = 0.0

    def _resolve_projectiles(self) -> float:
        reward = 0.0
        for projectile in list(self.projectiles.values()):
            if projectile.lifetime <= 0.0:
                del self.projectiles[projectile.identifier]
                continue
            for target in list(self.entities.values()):
                if target.kind is not EntityKind.SHIP or not target.alive:
                    continue
                if target.faction == projectile.faction:
                    continue
                if projectile.kind == "shell" and target.submerged:
                    continue
                if (
                    math.hypot(target.x - projectile.x, target.y - projectile.y)
                    > target.radius + projectile.radius
                ):
                    continue
                target.health -= projectile.damage
                if projectile.faction is Faction.PLAYER:
                    reward += projectile.damage * 0.05
                    self.last_events.append("damage")
                if target.health <= 0.0:
                    target.alive = False
                    if projectile.faction is Faction.PLAYER:
                        self.score += 100
                        reward += 10.0
                        self.last_events.append("kill")
                del self.projectiles[projectile.identifier]
                break
        return reward

    def _resolve_crates(self) -> float:
        reward = 0.0
        for crate in list(self.entities.values()):
            if crate.kind is not EntityKind.CRATE or not crate.alive:
                continue
            if (
                math.hypot(crate.x - self.player.x, crate.y - self.player.y)
                > self.player.radius + crate.radius
            ):
                continue
            crate.alive = False
            self.score += 20
            reward += 2.0
            self.last_events.append("crate")
        return reward

    def _resolve_ship_collisions(self) -> float:
        reward = 0.0
        ships = [
            entity
            for entity in self.entities.values()
            if entity.kind is EntityKind.SHIP and entity.alive
        ]
        for index, first in enumerate(ships):
            for second in ships[index + 1 :]:
                if (
                    math.hypot(first.x - second.x, first.y - second.y)
                    > first.radius + second.radius
                ):
                    continue
                first.health -= 10.0
                second.health -= 10.0
                if first.faction is Faction.PLAYER or second.faction is Faction.PLAYER:
                    reward -= 2.0
                    self.last_events.append("collision")
        return reward

    def _resolve_terrain(self) -> float:
        reward = 0.0
        for entity in self.entities.values():
            if (
                entity.kind is not EntityKind.SHIP
                or not entity.alive
                or entity.submerged
            ):
                continue
            for x, y, radius in self.terrain:
                if math.hypot(entity.x - x, entity.y - y) > entity.radius + radius:
                    continue
                if entity.faction is Faction.PLAYER:
                    reward -= 2.0
                self.last_events.append("terrain")
                entity.health -= 8.0
        return reward

    def _upgrade_player(self) -> None:
        player = self.player
        if player.ship_key is None or self.score < 100:
            return
        upgrade = SHIPS[player.ship_key].upgrade
        if upgrade is None:
            return
        spec = SHIPS[upgrade]
        player.name = spec.name
        player.ship_key = upgrade
        player.level = spec.level
        player.radius = spec.radius
        player.max_health = spec.health
        player.health = spec.health
        player.speed = 0.0
        player.target_speed = spec.speed
        player.turn_rate = spec.turn_rate
        player.acceleration = spec.acceleration
        player.weapons = tuple(weapon.name for weapon in spec.weapons)
        player.reloads = {weapon.name: 0.0 for weapon in spec.weapons}
        self.score -= 100

    def _remove_dead_entities(self) -> None:
        for entity in self.entities.values():
            if entity.kind is EntityKind.SHIP and entity.health <= 0.0:
                entity.alive = False

    def _observation(self) -> Observation:
        player = self.player
        values: list[float] = [
            player.x / self.config.world_radius,
            player.y / self.config.world_radius,
            player.vx / max(1.0, player.speed),
            player.vy / max(1.0, player.speed),
            player.direction / (2.0 * math.pi),
            player.speed / max(1.0, self._max_speed()),
            player.health / max(1.0, player.max_health),
            float(player.submerged),
            float(player.active),
            player.level / 3.0,
            *[
                float((player.reloads or {}).get(name, 0.0) <= 0.0)
                for name in player.weapons
            ],
        ]
        contacts = self._contacts(player)
        for index in range(self.config.contact_slots):
            if index < len(contacts):
                entity = contacts[index]
                dx = entity.x - player.x
                dy = entity.y - player.y
                distance = max(1.0, math.hypot(dx, dy))
                values.extend(
                    [
                        dx / self.config.world_radius,
                        dy / self.config.world_radius,
                        distance / self.config.world_radius,
                        math.sin(math.atan2(dy, dx)),
                        math.cos(math.atan2(dy, dx)),
                        entity.vx / 100.0,
                        entity.vy / 100.0,
                        float(entity.kind is EntityKind.SHIP),
                        entity.health / max(1.0, entity.max_health),
                        float(entity.submerged),
                    ]
                )
            else:
                values.extend([0.0] * 10)
        for index in range(self.config.terrain_rays):
            angle = 2.0 * math.pi * index / self.config.terrain_rays
            distance = self._terrain_distance(player, angle)
            values.extend(
                [distance / self.config.world_radius, 0.0 if distance < 160.0 else 1.0]
            )
        if len(values) != self.dimension:
            raise RuntimeError("observation dimension mismatch")
        info = self._info()
        return Observation(tuple(values), self.action_mask(), info)

    def _contacts(self, player: Entity) -> list[Entity]:
        contacts = [
            entity
            for entity in self.entities.values()
            if entity.alive and entity.identifier != player.identifier
        ]
        contacts.sort(
            key=lambda entity: (entity.x - player.x) ** 2 + (entity.y - player.y) ** 2
        )
        return contacts

    def _terrain_distance(self, player: Entity, angle: float) -> float:
        dx = math.cos(angle)
        dy = math.sin(angle)
        for distance in range(0, 161, 10):
            point_x = player.x + dx * distance
            point_y = player.y + dy * distance
            if any(
                math.hypot(point_x - x, point_y - y) <= radius
                for x, y, radius in self.terrain
            ):
                return float(distance)
        return 160.0

    def _info(self) -> dict[str, object]:
        return {
            "score": self.score,
            "health": self.player.health,
            "alive": self.player.alive,
            "enemies": sum(
                entity.kind is EntityKind.SHIP
                and entity.faction is Faction.ENEMY
                and entity.alive
                for entity in self.entities.values()
            ),
            "crates": sum(
                entity.kind is EntityKind.CRATE and entity.alive
                for entity in self.entities.values()
            ),
            "step": self.step_count,
            "events": list(self.last_events),
        }

    def _decode(self, action: int) -> DecodedAction:
        turn = (action // (THROTTLE_ACTIONS * AUXILIARY_ACTIONS)) % TURN_ACTIONS
        throttle = (action // AUXILIARY_ACTIONS) % THROTTLE_ACTIONS
        auxiliary = action % AUXILIARY_ACTIONS
        return DecodedAction((turn - 4) / 4.0, throttle / 4.0, auxiliary)

    def _is_auxiliary_valid(self, player: Entity, auxiliary: int) -> bool:
        if auxiliary in (0, 4):
            return True
        if auxiliary in (1, 2):
            weapon_index = auxiliary - 1
            if weapon_index >= len(player.weapons):
                return False
            weapon = self._weapon_spec(player, weapon_index)
            if weapon is None:
                return False
            if weapon.kind == "shell" and player.submerged:
                return False
            return (player.reloads or {}).get(weapon.name, 0.0) <= 0.0
        if auxiliary == 3:
            return player.depth > 0.0
        return (
            player.ship_key is not None
            and SHIPS[player.ship_key].upgrade is not None
            and self.score >= 100
        )

    def _weapon_spec(self, entity: Entity, index: int) -> WeaponSpec | None:
        if entity.ship_key is None or index >= len(entity.weapons):
            return None
        return SHIPS[entity.ship_key].weapons[index]

    def _nearest_enemy(self, entity: Entity) -> Entity | None:
        candidates = [
            candidate
            for candidate in self.entities.values()
            if candidate.alive
            and candidate.kind is EntityKind.SHIP
            and candidate.faction is not entity.faction
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda candidate: (
                (candidate.x - entity.x) ** 2 + (candidate.y - entity.y) ** 2
            ),
        )

    def _can_see(self, entity: Entity, target: Entity) -> bool:
        distance = math.hypot(target.x - entity.x, target.y - entity.y)
        range_ = 380.0 if entity.active else 220.0
        if target.submerged and entity.ship_key != "submarine":
            range_ *= 0.45
        return distance < range_

    def _bot_steer(self, entity: Entity) -> float:
        target = self._nearest_enemy(entity)
        if target is None:
            return 0.0
        desired = math.atan2(target.y - entity.y, target.x - entity.x)
        current = (
            math.atan2(entity.vy, entity.vx) if entity.speed_magnitude > 0.01 else 0.0
        )
        difference = math.atan2(
            math.sin(desired - current), math.cos(desired - current)
        )
        return float(
            np.clip(difference / (entity.turn_rate * self.config.dt + 1e-6), -1.0, 1.0)
        )

    def _clamp_to_world(self, entity: Entity) -> None:
        distance = math.hypot(entity.x, entity.y)
        limit = self.config.world_radius - entity.radius
        if distance > limit:
            scale = limit / distance
            entity.x *= scale
            entity.y *= scale

    def _bot_entities(self) -> list[Entity]:
        return [
            entity
            for entity in self.entities.values()
            if entity.kind is EntityKind.SHIP
            and entity.faction is Faction.ENEMY
            and entity.alive
        ]

    @property
    def player(self) -> Entity:
        return self.entities[0]

    def _max_speed(self) -> float:
        return max(spec.speed for spec in SHIPS.values())
