from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .config import SHIPS, GameConfig, WeaponSpec
from .domain import (
    SPAWN_PROTECTION_INITIAL,
    Entity,
    EntityKind,
    Faction,
    Projectile,
)
from .observation import Observation, StepResult, observation_dimension
from .physics import Transform, Velocity, sat_collision, terrain_collision

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
        self._blocked_from_repair: set[int] = set()
        self._outside_border: set[int] = set()

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
        self._blocked_from_repair = set()
        self._outside_border = set()
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
        decoded = self._decode(action)
        self._control_player(decoded)
        for entity in self._bot_entities():
            self._control_bot(entity)
        self._outside_border.clear()
        for entity in list(self.entities.values()):
            if entity.alive and entity.kind is EntityKind.SHIP:
                self._drive(
                    entity,
                    decoded.turn if entity.identifier == 0 else self._bot_steer(entity),
                    decoded.throttle if entity.identifier == 0 else 0.8,
                )
        self._blocked_from_repair.clear()
        reward = 0.01
        for projectile in self.projectiles.values():
            self._move_projectile(projectile)
        reward += self._resolve_projectiles()
        reward += self._resolve_crates()
        reward += self._resolve_ship_collisions()
        reward += self._resolve_boundaries()
        reward += self._resolve_terrain()
        for entity in self.entities.values():
            if entity.kind is EntityKind.SHIP and entity.alive and entity.health > 0.0:
                entity.update_tickers()
                entity.advance_reloads()
        self._repair_ships()
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
        reloads = {weapon.name: 0 for weapon in spec.weapons}
        return Entity(
            identifier=identifier,
            kind=EntityKind.SHIP,
            name=spec.name,
            x=x,
            y=y,
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
            length=spec.length,
            draft=spec.draft,
            depth=spec.depth,
            weapons=tuple(weapon.name for weapon in spec.weapons),
            reloads=reloads,
            active=True,
            active_requested=True,
            spawn_protection_remaining=(
                SPAWN_PROTECTION_INITIAL if spec.level == 1 else 0
            ),
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
            self.player.set_submerge(not self.player.submerge_requested)
        if action.auxiliary == 4:
            self.player.set_active(not self.player.active_requested)
        if action.auxiliary == 5:
            self._upgrade_player()

    def _advance_reloads(self) -> None:
        for entity in self.entities.values():
            entity.advance_reloads()

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
            entity.set_submerge(True)
        else:
            entity.set_submerge(False)

    def _drive(self, entity: Entity, turn: float, throttle: float) -> None:
        if entity.ship_key is None:
            return
        spec = SHIPS[entity.ship_key]
        transform = Transform(
            x=entity.x,
            y=entity.y,
            direction=entity.heading,
            velocity=Velocity.from_mps(entity.speed),
        )
        transform.apply_guidance(
            direction_target=entity.heading + turn * math.pi,
            velocity_target=spec.speed * throttle,
            max_speed=spec.speed,
            dt=self.config.dt,
            length=entity.length,
        )
        transform.do_kinematics(self.config.dt)
        entity.x = transform.x
        entity.y = transform.y
        entity.heading = transform.direction
        self._set_motion(entity, transform.velocity.to_mps())
        self._clamp_to_world(entity)

    def _set_motion(self, entity: Entity, speed: float) -> None:
        velocity = Velocity.from_mps(speed)
        entity.speed = velocity.to_mps()
        entity.vx = math.cos(entity.heading) * entity.speed
        entity.vy = math.sin(entity.heading) * entity.speed

    def _fire(self, entity: Entity, weapon_index: int) -> None:
        if weapon_index < 0 or weapon_index >= len(entity.weapons):
            return
        weapon = self._weapon_spec(entity, weapon_index)
        if weapon is None or entity.reloads is None:
            return
        if entity.reloads.get(weapon.name, 0) > 0:
            return
        if not weapon.submerged and entity.submerged:
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
        entity.consume_reload(weapon.name, self._reload_ticks(weapon.reload))
        entity.clear_spawn_protection()

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
            target = self._projectile_target(projectile)
            if target is None:
                continue
            reward += self._apply_projectile_hit(projectile, target)
            del self.projectiles[projectile.identifier]
        return reward

    def _projectile_target(self, projectile: Projectile) -> Entity | None:
        for target in self.entities.values():
            if not self._can_hit_projectile(projectile, target):
                continue
            if self._projectile_intersects(projectile, target):
                return target
        return None

    def _can_hit_projectile(self, projectile: Projectile, target: Entity) -> bool:
        return (
            target.kind is EntityKind.SHIP
            and target.alive
            and target.faction is not projectile.faction
            and not (projectile.kind == "shell" and target.submerged)
        )

    def _projectile_intersects(
        self, projectile: Projectile, target: Entity
    ) -> bool:
        return (
            math.hypot(target.x - projectile.x, target.y - projectile.y)
            <= target.radius + projectile.radius
        )

    def _apply_projectile_hit(
        self, projectile: Projectile, target: Entity
    ) -> float:
        self._damage(target, projectile.damage)
        if projectile.faction is not Faction.PLAYER:
            return 0.0
        reward = projectile.damage * 0.05
        self.last_events.append("damage")
        if target.health > 0.0:
            return reward
        target.alive = False
        self.score += 100
        reward += 10.0
        self.last_events.append("kill")
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
                if not self._ship_pair_collides(first, second):
                    continue
                reward += self._apply_ship_pair_collision(first, second)
        return reward

    def _ship_pair_collides(self, first: Entity, second: Entity) -> bool:
        return sat_collision(
            first.x,
            first.y,
            first.heading,
            first.speed,
            first.length,
            first.radius,
            first.radius,
            second.x,
            second.y,
            second.heading,
            second.speed,
            second.length,
            second.radius,
            second.radius,
            self.config.dt,
        )

    def _apply_ship_pair_collision(self, first: Entity, second: Entity) -> float:
        self._apply_boat_collision(first, second)
        self._apply_boat_collision(second, first)
        if first.faction is Faction.PLAYER or second.faction is Faction.PLAYER:
            self.last_events.append("collision")
            return -2.0
        return 0.0

    def _apply_boat_collision(self, boat: Entity, other: Entity) -> None:
        if boat.faction is not other.faction:
            front_x = other.x + math.cos(other.heading) * other.length * 0.5
            front_y = other.y + math.sin(other.heading) * other.length * 0.5
            front_distance_squared = (boat.x - front_x) ** 2 + (boat.y - front_y) ** 2
            base_damage = min(
                boat.max_health - boat.health * 0.5,
                other.max_health - other.health * 0.5,
            ) * self.config.dt / 10.0
            multiplier = self._collision_multiplier(
                front_distance_squared, boat.radius**2, boat.ship_key
            )
            if boat.ship_key == "submarine":
                multiplier *= 1.5
            elif boat.submerged:
                multiplier *= 10.0
            self._damage(boat, base_damage * multiplier)
        self._apply_collision_impulse(boat, other)

    def _apply_collision_impulse(self, boat: Entity, other: Entity) -> None:
        relative_mass = other.length * other.radius / (boat.length * boat.radius)
        if boat.faction is other.faction:
            relative_mass *= 3.0
        closest_x, closest_y = self._closest_point_on_keel(other, boat)
        difference_x = boat.x - closest_x
        difference_y = boat.y - closest_y
        difference_length = math.hypot(difference_x, difference_y)
        if difference_length > 0.0:
            difference_x /= difference_length
            difference_y /= difference_length
        impulse = 2.0 * (
            difference_x * math.cos(boat.heading)
            + difference_y * math.sin(boat.heading)
        ) * relative_mass
        self._set_motion(boat, max(-15.0, min(15.0, boat.speed + impulse)))

    def _closest_point_on_keel(
        self, boat: Entity, position: Entity
    ) -> tuple[float, float]:
        difference_x = position.x - boat.x
        difference_y = position.y - boat.y
        if difference_x * difference_x + difference_y * difference_y < 1.0:
            return boat.x, boat.y
        along = difference_x * math.cos(boat.heading) + difference_y * math.sin(
            boat.heading
        )
        along = max(-boat.length * 0.5, min(boat.length * 0.5, along))
        return (
            boat.x + math.cos(boat.heading) * along,
            boat.y + math.sin(boat.heading) * along,
        )

    def _collision_multiplier(
        self, distance_squared: float, radius_squared: float, ship_key: str | None
    ) -> float:
        minimum = 0.8 if ship_key == "submarine" else 0.6
        multiplier = (
            (radius_squared - distance_squared) / radius_squared * (1.0 - minimum)
            + minimum
        )
        return max(minimum, min(1.0, multiplier))

    def _resolve_boundaries(self) -> float:
        reward = 0.0
        for entity in self.entities.values():
            if entity.kind is not EntityKind.SHIP or not entity.alive:
                continue
            if entity.identifier not in self._outside_border:
                continue
            if entity.faction is Faction.PLAYER:
                reward -= 2.0
            self.last_events.append("terrain")
            self._blocked_from_repair.add(entity.identifier)
            self._damage(entity, max(entity.max_health, 0.1) * self.config.dt)
        return reward

    def _resolve_terrain(self) -> float:
        reward = 0.0
        for entity in self.entities.values():
            if entity.kind is not EntityKind.SHIP or not entity.alive:
                continue
            collision = terrain_collision(
                entity.x,
                entity.y,
                entity.heading,
                entity.speed,
                entity.length,
                entity.radius,
                self.terrain,
                self.config.dt,
            )
            if collision is None:
                continue
            delta_x = (collision[0] - entity.x) / entity.length
            delta_y = (collision[1] - entity.y) / entity.length
            dot = math.cos(entity.heading) * delta_x + math.sin(
                entity.heading
            ) * delta_y
            push = Velocity.from_mps(dot * -150.0)
            speed = entity.speed + push.to_mps()
            self._set_motion(entity, max(-5.0, min(5.0, speed)))
            if entity.faction is Faction.PLAYER:
                reward -= 2.0
            self.last_events.append("terrain")
            self._blocked_from_repair.add(entity.identifier)
            damage = max(entity.max_health / 4.0, 0.1) * self.config.dt
            self._damage(entity, damage)
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
        player.vx = 0.0
        player.vy = 0.0
        player.target_speed = spec.speed
        player.turn_rate = spec.turn_rate
        player.acceleration = spec.acceleration
        player.length = spec.length
        player.weapons = tuple(weapon.name for weapon in spec.weapons)
        player.reloads = {weapon.name: 0 for weapon in spec.weapons}
        player.spawn_protection_remaining = (
            SPAWN_PROTECTION_INITIAL if spec.level == 1 else 0
        )
        self.score -= 100

    def _damage(self, entity: Entity, amount: float) -> None:
        entity.health -= amount * entity.spawn_protection()

    def _repair_eligible(self, entity: Entity) -> bool:
        if entity.health <= 0.0:
            return False
        if math.hypot(entity.x, entity.y) > self.config.world_radius:
            return False
        return entity.identifier not in self._blocked_from_repair

    def _repair_ships(self) -> None:
        for entity in self.entities.values():
            if entity.kind is not EntityKind.SHIP or not entity.alive:
                continue
            if not self._repair_eligible(entity):
                continue
            if entity.length > 200.0:
                rate = 3.0
            elif entity.length > 100.0:
                rate = 2.0
            else:
                rate = 1.0
            entity.repair(rate)

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
            if not weapon.submerged and player.submerged:
                return False
            return (player.reloads or {}).get(weapon.name, 0.0) <= 0.0
        if auxiliary == 3:
            return player.depth > 0.0
        return (
            player.ship_key is not None
            and SHIPS[player.ship_key].upgrade is not None
            and self.score >= 100
        )

    def _reload_ticks(self, seconds: float) -> int:
        return min(65535, max(0, math.trunc(seconds * 10.0)))

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
        current = entity.heading
        difference = math.atan2(
            math.sin(desired - current), math.cos(desired - current)
        )
        return float(
            np.clip(difference / (entity.turn_rate * self.config.dt + 1e-6), -1.0, 1.0)
        )

    def _clamp_to_world(self, entity: Entity) -> None:
        distance = math.hypot(entity.x, entity.y)
        if distance <= self.config.world_radius:
            return
        normal_x = -entity.x / distance
        normal_y = -entity.y / distance
        scale = self.config.world_radius / distance
        entity.x *= scale
        entity.y *= scale
        self._outside_border.add(entity.identifier)
        inward_speed = 10.0 * (
            normal_x * math.cos(entity.heading) + normal_y * math.sin(entity.heading)
        )
        self._set_motion(entity, inward_speed)

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
