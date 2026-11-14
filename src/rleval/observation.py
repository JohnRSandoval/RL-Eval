"""Explicit opt-in state observations; never serialize a whole backend snapshot.

Axes/units: https://wiki.rlbot.org/v5/botmaking/useful-game-values/
Orientation convention: https://github.com/RLBot/RLBotPythonExample/blob/master/src/util/orientation.py
Relative velocity here is the world velocity difference projected into the
instantaneous car frame, not a rotating-frame derivative or ball prediction.
"""

from __future__ import annotations

# ruff: noqa: TRY004 -- invalid telemetry uses one ValueError validation boundary.
import math

COORDINATES = {
    "frame": "Rocket League world X/Y ground plane, +Z up; +X points left when facing orange",
    "goals": "blue/team 0 at negative Y; orange/team 1 at positive Y",
    "position_units": "Unreal units (uu); 1 uu = 1 cm",
    "velocity_units": "uu per second",
    "angular_velocity_units": "radians per second about world X/Y/Z axes",
    "rotation_units": "Euler pitch/yaw/roll radians; yaw 0 faces +X, pi/2 faces +Y",
    "rotation_signs": "positive pitch noses up; positive roll tilts right; yaw increases clockwise from above",
    "local_frame": "forward/right/up relative to the controlled car",
    "relative_velocity": "ball velocity minus car velocity, projected into current car axes; not a prediction",
    "bearing": "atan2(local right, local forward), radians; positive right, negative left; 0 when coincident",
}


def _number(value, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _integer(value, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _boolean(value, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _vector(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an x/y/z vector")
    return {axis: _number(value.get(axis), f"{label}.{axis}") for axis in ("x", "y", "z")}


def _rotation(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain pitch/yaw/roll")
    return {axis: _number(value.get(axis), f"{label}.{axis}") for axis in ("pitch", "yaw", "roll")}


def orientation_vectors(rotation: dict) -> dict:
    """World-space car axes for RLBot's YZX Euler and left-handed world convention."""
    rotation = _rotation(rotation, "rotation")
    pitch, yaw, roll = (rotation[name] for name in ("pitch", "yaw", "roll"))
    cp, sp, cy, sy, cr, sr = (
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
        math.cos(roll),
        math.sin(roll),
    )
    return {
        "forward": {"x": cp * cy, "y": cp * sy, "z": sp},
        "right": {"x": cy * sp * sr - cr * sy, "y": sy * sp * sr + cr * cy, "z": -cp * sr},
        "up": {"x": -cr * cy * sp - sr * sy, "y": -cr * sy * sp + sr * cy, "z": cp * cr},
    }


def _car(value: dict, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a car object")
    rotation = _rotation(value.get("rotation"), f"{label}.rotation")
    team = _integer(value.get("team"), f"{label}.team")
    if team not in (0, 1):
        raise ValueError(f"{label}.team must be 0 or 1")
    boost = _number(value.get("boost"), f"{label}.boost")
    if not 0 <= boost <= 100:
        raise ValueError(f"{label}.boost must be in [0, 100]")
    return {
        "index": _integer(value.get("index"), f"{label}.index"),
        "team": team,
        "position": _vector(value.get("position"), f"{label}.position"),
        "velocity": _vector(value.get("velocity"), f"{label}.velocity"),
        "angular_velocity": _vector(value.get("angular_velocity"), f"{label}.angular_velocity"),
        "rotation_radians": rotation,
        **orientation_vectors(rotation),
        "boost": boost,
        "has_wheel_contact": _boolean(
            value.get("has_wheel_contact", value.get("on_ground")), f"{label}.has_wheel_contact"
        ),
        **{
            flag: _boolean(value.get(flag), f"{label}.{flag}")
            for flag in ("has_jumped", "has_double_jumped", "has_dodged")
        },
    }


def _difference(a: dict, b: dict) -> dict:
    return {axis: _number(a[axis] - b[axis], "vector difference") for axis in ("x", "y", "z")}


def _local(vector: dict, axes: dict) -> dict:
    return {
        name: _number(
            sum(vector[axis] * axes[name][axis] for axis in ("x", "y", "z")), "projection"
        )
        for name in ("forward", "right", "up")
    }


def relative_target(player: dict, target: dict | None) -> dict | None:
    """Project a planner target into the current car axes; no policy or prediction."""
    if target is None:
        return None
    displacement = _difference(_vector(target, "target"), player["position"])
    position = _local(displacement, player)
    return {
        "position": position,
        "distance": _number(math.hypot(*displacement.values()), "target distance"),
        "bearing_radians": math.atan2(position["right"], position["forward"]),
    }


def build_game_state(snapshot: dict) -> dict:
    """Allowlist fresh measured state for the state/pixels_and_state eval tracks.

    Rejects missing or nonfinite required data. The runner must also check ready
    and packet age. Ball absence is represented explicitly, never fabricated.
    This intentionally excludes raw packets, paths, logs, credentials, prediction
    arrays, evaluator internals, and player/account names.
    """
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be a dictionary")
    player = _car(
        {
            "index": snapshot.get("player_index"),
            "team": snapshot.get("team"),
            "position": snapshot.get("car_position"),
            "velocity": snapshot.get("car_velocity"),
            "angular_velocity": snapshot.get("car_angular_velocity"),
            "rotation": snapshot.get("car_rotation"),
            "boost": snapshot.get("car_boost"),
            **{
                flag: snapshot.get(f"car_{flag}")
                for flag in ("on_ground", "has_jumped", "has_double_jumped", "has_dodged")
            },
        },
        "player",
    )
    ball = relative = None
    if snapshot.get("ball_position") is not None:
        ball = {
            "position": _vector(snapshot.get("ball_position"), "ball.position"),
            "velocity": _vector(snapshot.get("ball_velocity"), "ball.velocity"),
            "angular_velocity": _vector(
                snapshot.get("ball_angular_velocity"), "ball.angular_velocity"
            ),
        }
        displacement = _difference(ball["position"], player["position"])
        local_position = _local(displacement, player)
        relative = {
            "position": local_position,
            "velocity": _local(_difference(ball["velocity"], player["velocity"]), player),
            "distance": _number(math.hypot(*displacement.values()), "ball distance"),
            "bearing_radians": math.atan2(local_position["right"], local_position["forward"]),
        }
    opponents = snapshot.get("opponents", [])
    if not isinstance(opponents, list):
        raise ValueError("opponents must be a list")
    phase = snapshot.get("phase")
    if phase not in {
        "Inactive",
        "Countdown",
        "Kickoff",
        "Active",
        "GoalScored",
        "Replay",
        "Paused",
        "Ended",
    }:
        raise ValueError("Unknown match phase")
    scores = snapshot.get("scores", {})
    if not isinstance(scores, dict):
        raise ValueError("scores must be an object")
    age = _number(snapshot.get("packet_age_s"), "packet_age_s")
    if age < 0:
        raise ValueError("packet_age_s must be nonnegative")
    return {
        "schema_version": 2,
        "synthetic": bool(snapshot.get("synthetic", False)),
        "frame_num": _integer(snapshot.get("frame_num"), "frame_num"),
        "game_time_s": _number(snapshot.get("game_time"), "game_time"),
        "packet_received_monotonic_s": _number(
            snapshot.get("packet_received_monotonic_s"), "packet_received_monotonic_s"
        ),
        "packet_age_s": age,
        "coordinates": dict(COORDINATES),
        "match": {
            "phase": phase,
            "time_remaining_s": _number(snapshot.get("game_time_remaining"), "game_time_remaining"),
            "is_overtime": _boolean(snapshot.get("is_overtime", False), "is_overtime"),
            "scores": {
                team: _integer(scores.get(team), f"scores.{team}") for team in ("blue", "orange")
            },
        },
        "player": player,
        "ball": ball,
        "relative_ball": relative,
        "opponents": [_car(item, f"opponents[{i}]") for i, item in enumerate(opponents)],
    }
