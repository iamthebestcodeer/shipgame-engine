from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Observation:
    values: tuple[float, ...]
    action_mask: tuple[bool, ...]
    info: dict[str, object]

    @property
    def dimension(self) -> int:
        return len(self.values)

    def vector(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float32)

    def mask(self) -> np.ndarray:
        return np.asarray(self.action_mask, dtype=bool)

    def to_dict(self) -> dict[str, object]:
        return {
            "values": list(self.values),
            "action_mask": list(self.action_mask),
            "info": self.info,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


def observation_dimension(contact_slots: int, terrain_rays: int) -> int:
    return 12 + contact_slots * 10 + terrain_rays * 2
