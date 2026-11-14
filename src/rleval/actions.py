"""The complete, deliberately small action space shared by every model."""

import math
from dataclasses import asdict, dataclass

AXES = ("throttle", "steer", "pitch", "yaw", "roll")
BUTTONS = ("jump", "boost", "handbrake")
NEUTRAL = {**dict.fromkeys(AXES, 0.0), **dict.fromkeys(BUTTONS, False)}
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "controls": {
            "type": "object",
            "properties": {
                **{k: {"type": "number", "minimum": -1, "maximum": 1} for k in AXES},
                **{k: {"type": "boolean"} for k in BUTTONS},
            },
            "required": list(AXES + BUTTONS),
            "additionalProperties": False,
        },
        "hold_ms": {"type": "integer", "minimum": 50, "maximum": 1000},
        "memory": {"type": "string", "maxLength": 1000},
    },
    "required": ["controls", "hold_ms", "memory"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Action:
    controls: dict
    hold_ms: int
    memory: str

    def to_dict(self) -> dict:
        return asdict(self)


def parse_action(value: dict) -> Action:
    """Reject invalid inputs instead of silently changing a model's action."""
    if not isinstance(value, dict) or set(value) != {"controls", "hold_ms", "memory"}:
        raise ValueError("Action must contain exactly controls, hold_ms, memory")
    controls = value["controls"]
    if not isinstance(controls, dict) or set(controls) != set(AXES + BUTTONS):
        raise ValueError("Action controls must contain all eight controller fields")
    for name in AXES:
        v = controls[name]
        if type(v) not in (int, float) or not math.isfinite(v) or not -1 <= v <= 1:
            raise ValueError(f"{name} must be a finite number in [-1, 1]")
    for name in BUTTONS:
        if type(controls[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    if type(value["hold_ms"]) is not int or not 50 <= value["hold_ms"] <= 1000:
        raise ValueError("hold_ms must be an integer in [50, 1000]")
    if not isinstance(value["memory"], str) or len(value["memory"]) > 1000:
        raise ValueError("memory must be a string of at most 1000 characters")
    return Action(dict(controls), value["hold_ms"], value["memory"])
