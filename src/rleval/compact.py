"""Compact, measured motor observations for basic controller calibration.

This module performs coordinate conversion and rounding, never chooses controls.
Wheel contact is explicitly distinguished from being upright on the flat floor.
"""

from __future__ import annotations

# ruff: noqa: TRY004 -- invalid observations share one ValueError boundary.
import math

from .actions import parse_action
from .observation import orientation_vectors


def _number(value, label):
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be finite")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{label} must be finite") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _vector(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a vector")
    return {axis: _number(value.get(axis), label) for axis in ("x", "y", "z")}


def _round(value, digits=2):
    value = _number(value, "derived measurement")
    result = round(value, digits)
    return 0.0 if result == 0 else result


def _dot(a, b):
    return sum(a[axis] * b[axis] for axis in ("x", "y", "z"))


def _controls(value):
    if value is None:
        return None
    parsed = parse_action({"controls": value, "hold_ms": 50, "memory": ""})
    return {
        key: _round(val, 3) if type(val) is float else val for key, val in parsed.controls.items()
    }


def _feedback(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("control_feedback must be an object")
    result = {"source": "command_lease_not_game_acknowledgement"}
    for key in ("commanded_controls", "selected_controls"):
        if key in value:
            result[key] = _controls(value[key])
    for key in ("requested_hold_ms", "command_age_ms", "remaining_ms", "command_id"):
        if key in value:
            number = _number(value[key], key)
            if number < 0:
                raise ValueError(f"{key} must be nonnegative")
            result[key] = _round(number)
    if "active" in value:
        if type(value["active"]) is not bool:
            raise ValueError("active must be boolean")
        result["active"] = value["active"]
    complete = value.get("last_completed")
    result["last_completed"] = None
    if complete is not None:
        if not isinstance(complete, dict):
            raise ValueError("last_completed must be an object")
        reason = complete.get("end_reason")
        if not isinstance(reason, str) or len(reason) > 100:
            raise ValueError("end_reason must be short text")
        result["last_completed"] = {
            "controls": _controls(complete.get("controls")),
            "end_reason": reason,
            **{
                key: _round(_number(complete.get(key), key))
                for key in ("command_id", "requested_hold_ms", "commanded_duration_ms")
            },
        }
    return result


TRANSITION_NUMBERS = (
    "actual_duration_ms",
    "commanded_duration_ms",
    "elapsed_ms",
    "start_forward_speed_uu_s",
    "end_forward_speed_uu_s",
    "delta_speed_uu_s",
    "delta_heading_degrees",
    "start_target_bearing_degrees",
    "end_target_bearing_degrees",
    "delta_target_distance_uu",
    "observed_interval_ms",
    "displacement_uu",
    "heading_change_degrees",
    "speed_before_uu_s",
    "speed_after_uu_s",
)


def _transitions(history):
    if history is None:
        return []
    if not isinstance(history, list):
        raise ValueError("feedback_history must be a list")
    result = []
    for item in history[-3:]:
        if not isinstance(item, dict):
            raise ValueError("A feedback transition must be an object")
        entry = {key: _round(_number(item[key], key)) for key in TRANSITION_NUMBERS if key in item}
        if "controls" in item:
            entry["controls"] = _controls(item["controls"])
        for key in ("acknowledged_input_at_start", "acknowledged_input_at_end"):
            if key in item:
                entry[key] = _controls(item[key])
        if "completed_command" in item:
            entry["completed_command"] = _feedback({"last_completed": item["completed_command"]})[
                "last_completed"
            ]
        if "note" in item:
            if not isinstance(item["note"], str) or len(item["note"]) > 300:
                raise ValueError("Transition note must be short text")
            entry["note"] = item["note"]
        if "reason" in item:
            if not isinstance(item["reason"], str) or len(item["reason"]) > 100:
                raise ValueError("Transition reason must be short text")
            entry["reason"] = item["reason"]
        result.append(entry)
    return result


def build_motor_observation(
    snapshot: dict,
    target: dict | None,
    desired_speed: float,
    task: str,
    feedback_history: list | None = None,
) -> dict:
    """Allowlist compact geometry, current input feedback, and three past transitions."""
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be an object")
    if not isinstance(task, str) or not task.strip() or len(task) > 1000:
        raise ValueError("task must contain 1 to 1000 characters")
    desired_speed = _number(desired_speed, "desired_speed")
    if not 0 <= desired_speed <= 2300:
        raise ValueError("desired_speed must be in [0, 2300]")
    position = _vector(snapshot.get("car_position"), "car_position")
    velocity = _vector(snapshot.get("car_velocity"), "car_velocity")
    angular = _vector(snapshot.get("car_angular_velocity"), "car_angular_velocity")
    axes = orientation_vectors(snapshot.get("car_rotation"))
    contact = snapshot.get("car_has_wheel_contact", snapshot.get("car_on_ground"))
    if type(contact) is not bool:
        raise ValueError("wheel contact must be boolean")
    if not contact:
        surface = "air"
    elif axes["up"]["z"] > 0.9 and position["z"] < 100:
        surface = "floor"
    elif axes["up"]["z"] < -0.9:
        surface = "ceiling"
    else:
        surface = "wall_or_ramp"
    target_data = None
    if target is not None:
        waypoint = _vector(target, "target")
        delta = {axis: waypoint[axis] - position[axis] for axis in ("x", "y", "z")}
        forward, right, up = (_dot(delta, axes[name]) for name in ("forward", "right", "up"))
        target_data = {
            "bearing_degrees": _round(math.degrees(math.atan2(right, forward))),
            "distance_uu": _round(math.hypot(*delta.values())),
            "forward_uu": _round(forward),
            "right_uu": _round(right),
            "up_uu": _round(up),
        }
    age = _number(snapshot.get("packet_age_s"), "packet_age_s")
    if age < 0:
        raise ValueError("packet_age_s must be nonnegative")
    return {
        "schema_version": "compact-motor-v1",
        "task": task,
        "target": target_data,
        "desired_speed_uu_s": _round(desired_speed),
        "motion": {
            "forward_speed_uu_s": _round(_dot(velocity, axes["forward"])),
            "lateral_speed_uu_s": _round(_dot(velocity, axes["right"])),
            "vertical_speed_uu_s": _round(velocity["z"]),
            "yaw_rate_degrees_s": _round(math.degrees(angular["z"])),
        },
        "surface": {
            "wheel_contact": contact,
            "classification": surface,
            "up_z": _round(axes["up"]["z"], 3),
            "height_uu": _round(position["z"]),
        },
        "acknowledged_input": _controls(snapshot.get("last_input")),
        "command_feedback": _feedback(snapshot.get("control_feedback")),
        "recent_transitions": _transitions(feedback_history),
        "packet_age_ms": _round(age * 1000),
    }


def validate_motor_observation(value: dict) -> dict:
    """Reject incomplete compact observations before any model request."""
    required = {
        "schema_version",
        "task",
        "target",
        "desired_speed_uu_s",
        "motion",
        "surface",
        "acknowledged_input",
        "command_feedback",
        "recent_transitions",
        "packet_age_ms",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Compact observation has missing or unexpected fields")
    if value["schema_version"] != "compact-motor-v1":
        raise ValueError("Unknown compact observation version")
    if not isinstance(value["task"], str) or not value["task"].strip() or len(value["task"]) > 1000:
        raise ValueError("Compact task must contain 1 to 1000 characters")
    speed = _number(value["desired_speed_uu_s"], "desired_speed_uu_s")
    if not 0 <= speed <= 2300:
        raise ValueError("desired_speed_uu_s is outside [0, 2300]")
    motion = value["motion"]
    motion_keys = {
        "forward_speed_uu_s",
        "lateral_speed_uu_s",
        "vertical_speed_uu_s",
        "yaw_rate_degrees_s",
    }
    if not isinstance(motion, dict) or set(motion) != motion_keys:
        raise ValueError("Compact motion has missing or unexpected fields")
    for key in motion_keys:
        _number(motion[key], key)
    target = value["target"]
    if target is not None:
        keys = {"bearing_degrees", "distance_uu", "forward_uu", "right_uu", "up_uu"}
        if not isinstance(target, dict) or set(target) != keys:
            raise ValueError("Compact target has missing or unexpected fields")
        for key in keys:
            _number(target[key], key)
        if not -180 <= target["bearing_degrees"] <= 180 or target["distance_uu"] < 0:
            raise ValueError("Compact target bearing or distance is invalid")
    surface = value["surface"]
    if not isinstance(surface, dict) or set(surface) != {
        "wheel_contact",
        "classification",
        "up_z",
        "height_uu",
    }:
        raise ValueError("Compact surface has missing or unexpected fields")
    if type(surface["wheel_contact"]) is not bool or surface["classification"] not in {
        "air",
        "floor",
        "wall_or_ramp",
        "ceiling",
    }:
        raise ValueError("Compact surface classification is invalid")
    if not -1 <= _number(surface["up_z"], "up_z") <= 1:
        raise ValueError("up_z is outside [-1, 1]")
    _number(surface["height_uu"], "height_uu")
    if _number(value["packet_age_ms"], "packet_age_ms") < 0:
        raise ValueError("packet_age_ms must be nonnegative")
    return dict(
        value,
        acknowledged_input=_controls(value["acknowledged_input"]),
        command_feedback=_feedback(value["command_feedback"]),
        recent_transitions=_transitions(value["recent_transitions"]),
    )
