"""Explicit local mechanics for tactical plans, not model-generated controls.

This deliberately modest controller drives on the floor, predicts ball XY with
constant velocity, makes short ground jumps, and attempts to level in the air.
It does not solve aerial shots, wall play, dodges, or ball bounces. Physics/sign
conventions follow https://wiki.rlbot.org/v5/botmaking/useful-game-values/ and
https://github.com/RLBot/RLBot/blob/master/src/main/python/rlbot/agents/base_agent.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .actions import NEUTRAL
from .observation import orientation_vectors

SKILLS = frozenset({"drive_to", "intercept", "shoot", "defend", "recover", "stop"})


@dataclass(frozen=True)
class MotorOutput:
    controls: dict
    target: dict | None
    skill: str
    status: str
    metrics: dict


def _number(value, name):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _vector(value, name):
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain x/y/z")
    result = {axis: _number(value.get(axis), f"{name}.{axis}") for axis in "xyz"}
    if any(abs(v) > 1_000_000 for v in result.values()):
        raise ValueError(f"{name} is outside supported bounds")
    return result


def _clip(value, low=-1.0, high=1.0):
    return max(low, min(high, value))


def _dot(a, b):
    return sum(a[axis] * b[axis] for axis in "xyz")


def _point(x, y, z=17.0):
    x, y = _clip(x, -3900, 3900), _clip(y, -5000, 5000)
    # Keep a margin inside the standard arena's rounded corner walls.
    dx, dy = max(0, abs(x) - 2944), max(0, abs(y) - 3968)
    radius = math.hypot(dx, dy)
    if dx and dy and radius > 1000:
        x = math.copysign(2944 + dx * 1000 / radius, x)
        y = math.copysign(3968 + dy * 1000 / radius, y)
    return {"x": x, "y": y, "z": z}


def _drive_target(requested, car):
    """Project a ground waypoint into the field or central goal opening.

    This is a simple goal-mouth gate, not a general obstacle/path planner.
    Targets beyond a back wall only enter the goal through its central opening.
    """
    target = _point(requested["x"], requested["y"])
    if abs(requested["x"]) < 750 and abs(requested["y"]) > 5000:
        target["y"] = _clip(requested["y"], -5600, 5600)
    for sign in (-1, 1):
        wall_y = sign * 5120
        car_side, target_side = sign * car["y"], sign * target["y"]
        if (car_side - 5120) * (target_side - 5120) < 0:
            fraction = (wall_y - car["y"]) / (target["y"] - car["y"])
            crossing_x = car["x"] + fraction * (target["x"] - car["x"])
            if abs(crossing_x) > 700:
                return {"x": 0.0, "y": sign * 4750, "z": 17.0}
    return target


def _plan(skill, speed=1400.0, target=None, boost="allowed", jump="allowed"):
    return {
        "skill": skill,
        "objective": f"Scripted {skill}",
        "target_position": target,
        "desired_speed": speed,
        "boost_policy": boost,
        "jump_policy": jump,
        "guidance": "Explicit scripted baseline; local mechanics recompute from current telemetry.",
    }


def scripted_plan(snapshot: dict) -> dict:
    """Transparent heuristic tactics, intended to share the exact same motor."""
    try:
        car = _vector(snapshot.get("car_position"), "car_position")
        ball = _vector(snapshot.get("ball_position"), "ball_position")
        axes = orientation_vectors(snapshot.get("car_rotation"))
        team = snapshot.get("team", 0)
        if type(team) is not int or team not in (0, 1):
            raise ValueError("team must be 0 or 1")
        contact = snapshot.get("car_has_wheel_contact", snapshot.get("car_on_ground", False))
        if not contact or car["z"] > 120 or axes["up"]["z"] < 0.9:
            return _plan("recover", 0, boost="conserve")
        attack = 1 if team == 0 else -1
        if attack * ball["y"] < attack * car["y"] - 350:
            return _plan("defend", 1300, boost="allowed", jump="avoid")
        if attack * ball["y"] < -2600:
            return _plan("intercept", 1500)
        return _plan("shoot", 1700)
    except (TypeError, ValueError, OverflowError):
        return _plan("stop", 0, boost="conserve", jump="avoid")


class LocalMotorController:
    """Tick at 60 Hz. Output is explicitly assisted/scripted motor control."""

    MAX_PACKET_AGE_S = 0.15

    def __init__(self):
        self.reset()

    def reset(self):
        self._last_now = None
        self._jump_until = -math.inf
        self._last_jump = -math.inf

    def _neutral(self, status, reason=""):
        self._jump_until = -math.inf
        return MotorOutput(dict(NEUTRAL), None, "stop", status, {"reason": reason})

    @staticmethod
    def _validated_plan(plan):
        if not isinstance(plan, dict) or plan.get("skill") not in SKILLS:
            raise ValueError("Plan requires a supported skill")
        if plan.get("boost_policy") not in {"conserve", "allowed", "use"}:
            raise ValueError("Unknown boost policy")
        if plan.get("jump_policy") not in {"avoid", "allowed", "needed"}:
            raise ValueError("Unknown jump policy")
        speed = _number(plan.get("desired_speed"), "desired_speed")
        if not 0 <= speed <= 2300:
            raise ValueError("desired_speed must be between 0 and 2300")
        target = plan.get("target_position")
        if target is not None:
            target = _vector(target, "target_position")
        if plan["skill"] == "drive_to" and target is None:
            raise ValueError("drive_to requires target_position")
        return {**plan, "desired_speed": speed, "target_position": target}

    @staticmethod
    def _target(skill, plan, car, ball, ball_velocity, team, speed):
        attack = 1 if team == 0 else -1
        if skill == "drive_to":
            return _drive_target(plan["target_position"], car), 0.0
        if skill in {"stop", "recover"}:
            return None, 0.0
        distance = math.hypot(ball["x"] - car["x"], ball["y"] - car["y"])
        horizon = _clip(distance / max(650, abs(speed), plan["desired_speed"] * 0.65), 0.05, 1.2)
        predicted = _point(
            ball["x"] + ball_velocity["x"] * horizon,
            ball["y"] + ball_velocity["y"] * horizon,
        )
        if skill == "intercept":
            return predicted, horizon
        if skill == "defend":
            own_goal_y = -attack * 5000
            return _point(
                predicted["x"] * 0.45,
                -attack
                * _clip(-attack * (own_goal_y + (predicted["y"] - own_goal_y) * 0.32), 600, 4600),
            ), horizon
        shot_x, shot_y = -predicted["x"], attack * 5120 - predicted["y"]
        length = max(1.0, math.hypot(shot_x, shot_y))
        shot_x, shot_y = shot_x / length, shot_y / length
        car_ball_x, car_ball_y = ball["x"] - car["x"], ball["y"] - car["y"]
        alignment = (car_ball_x * shot_x + car_ball_y * shot_y) / max(1.0, distance)
        offset = 300 if distance < 600 and alignment > 0.75 else -220
        return _point(predicted["x"] + shot_x * offset, predicted["y"] + shot_y * offset), horizon

    @staticmethod
    def _throttle(speed, desired, tick_dt):
        # Braking is ~3500 uu/s² even for small opposite throttle. Release
        # before one tick could cross zero; neutral drag finishes the stop.
        stop_margin = max(65.0, 3500 * tick_dt + 20)
        if desired <= 0:
            return -math.copysign(1.0, speed) if abs(speed) > stop_margin else 0.0
        if speed < -stop_margin:
            return 1.0
        error = desired - speed
        if error < -100 and speed > stop_margin:
            return -1.0
        if error <= 30:
            return 0.0
        return _clip(error / 450, 0.0, 1.0)

    def tick(self, snapshot: dict, plan: dict | None, now: float) -> MotorOutput:
        try:
            now = _number(now, "now")
            if now < 0 or (self._last_now is not None and now < self._last_now):
                self.reset()
                return self._neutral("invalid_time", "Clock moved backwards")
            tick_dt = (
                _clip(now - self._last_now, 1 / 120, 0.1) if self._last_now is not None else 1 / 60
            )
            self._last_now = now
            if not isinstance(snapshot, dict):
                raise TypeError("snapshot must be an object")
            if (
                snapshot.get("ready") is not True
                or snapshot.get("ended")
                or snapshot.get("active") is False
                or snapshot.get("error")
            ):
                return self._neutral("inactive")
            age = _number(snapshot.get("packet_age_s", 0), "packet_age_s")
            if age < 0 or age > self.MAX_PACKET_AGE_S:
                return self._neutral("stale", "Packet is outside freshness limit")
            car = _vector(snapshot.get("car_position"), "car_position")
            velocity = _vector(snapshot.get("car_velocity"), "car_velocity")
            ball = _vector(snapshot.get("ball_position"), "ball_position")
            ball_velocity = _vector(snapshot.get("ball_velocity"), "ball_velocity")
            angular = _vector(snapshot.get("car_angular_velocity"), "car_angular_velocity")
            rotation = snapshot.get("car_rotation")
            axes = orientation_vectors(rotation)
            contact = snapshot.get("car_has_wheel_contact", snapshot.get("car_on_ground"))
            if type(contact) is not bool:
                raise ValueError("Wheel contact must be boolean")
            jumped = snapshot.get("car_has_jumped", False)
            double_jumped = snapshot.get("car_has_double_jumped", False)
            if type(jumped) is not bool or type(double_jumped) is not bool:
                raise ValueError("Jump flags must be boolean")
            boost = _number(snapshot.get("car_boost", 0), "car_boost")
            team = snapshot.get("team", 0)
            if type(team) is not int or team not in (0, 1) or not 0 <= boost <= 100:
                raise ValueError("Invalid team or boost")
        except (TypeError, ValueError, OverflowError) as exc:
            return self._neutral("invalid_state", str(exc))
        fallback = plan is None
        if fallback:
            plan = _plan("defend", 1000, boost="conserve", jump="avoid")
        try:
            plan = self._validated_plan(plan)
        except (TypeError, ValueError, OverflowError) as exc:
            return self._neutral("invalid_plan", str(exc))
        skill = plan["skill"]
        speed = _dot(velocity, axes["forward"])
        flat = contact and car["z"] < 120 and axes["up"]["z"] > 0.9
        surface = "floor" if flat else "wall" if contact else "air"
        controls = dict(NEUTRAL)
        metrics = {"surface": surface, "forward_speed_uu_s": speed, "fallback": fallback}
        target, horizon = self._target(skill, plan, car, ball, ball_velocity, team, speed)
        metrics.update(prediction_horizon_s=horizon, prediction_kind="linear_xy_no_bounces")
        jump_allowed = plan["jump_policy"] != "avoid"
        if not jump_allowed:
            self._jump_until = -math.inf
        if not flat:
            target = None
            skill, status = "recover", "recover_wall" if contact else "recover_air"
            if contact:
                downhill = math.atan2(-axes["right"]["z"], -axes["forward"]["z"])
                controls["steer"] = _clip(1.5 * downhill)
                controls["throttle"] = 0.4 if abs(downhill) < math.pi / 2 else -0.3
            else:
                # Pitch/roll input and rotation share signs; angular velocity
                # about car-right/car-forward has the opposite sign.
                controls["pitch"] = _clip(
                    -2.5 * rotation["pitch"] + 0.35 * _dot(angular, axes["right"])
                )
                roll_error = math.remainder(rotation["roll"], 2 * math.pi)
                controls["roll"] = _clip(-2.5 * roll_error + 0.35 * _dot(angular, axes["forward"]))
                controls["yaw"] = _clip(-0.25 * _dot(angular, axes["up"]))
            if (
                jump_allowed
                and not jumped
                and not double_jumped
                and car["z"] < 140
                and abs(speed) < 150
                and axes["up"]["z"] < 0.5
                and now - self._last_jump > 1.5
            ):
                self._jump_until, self._last_jump = now + 0.10, now
                status = "recovery_pop"
            controls["jump"] = jump_allowed and now < self._jump_until
        elif skill in {"stop", "recover"}:
            controls["throttle"] = self._throttle(speed, 0, tick_dt)
            status = "stopped" if abs(speed) < 80 else "braking"
            metrics["desired_speed_uu_s"] = 0.0
        else:
            delta = {axis: target[axis] - car[axis] for axis in "xyz"}
            distance = math.hypot(delta["x"], delta["y"])
            bearing = math.atan2(_dot(delta, axes["right"]), _dot(delta, axes["forward"]))
            controls["steer"] = _clip(2.8 * bearing - 0.12 * angular["z"])
            desired = min(plan["desired_speed"], max(350, 2100 / (1 + 2.5 * abs(bearing))))
            if skill in {"drive_to", "defend"}:
                desired = min(desired, math.sqrt(2 * 1200 * max(0, distance - 100)))
            if distance < 100 and skill in {"drive_to", "defend"}:
                desired, controls["steer"] = 0.0, 0.0
            controls["throttle"] = self._throttle(speed, desired, tick_dt)
            controls["boost"] = bool(
                plan["boost_policy"] in {"allowed", "use"}
                and boost > 0
                and abs(bearing) < 0.12
                and speed > 600
                and desired > speed + 200
                and distance > 600
                and speed < 2250
                and controls["throttle"] > 0
            )
            status = "arrived" if desired == 0 and abs(speed) < 80 else "tracking"
            ball_delta = {axis: ball[axis] - car[axis] for axis in "xyz"}
            ball_ahead, ball_right = (
                _dot(ball_delta, axes["forward"]),
                _dot(ball_delta, axes["right"]),
            )
            if (
                jump_allowed
                and not jumped
                and not double_jumped
                and skill in {"intercept", "shoot"}
                and 120 < ball["z"] < 320
                and 50 < ball_ahead < 300
                and abs(ball_right) < 80
                and speed > 150
                and now - self._last_jump > 1.5
            ):
                self._jump_until, self._last_jump = now + 0.12, now
                status = "ball_pop"
            controls["jump"] = jump_allowed and now < self._jump_until
            metrics.update(
                heading_error_deg=math.degrees(bearing),
                distance_uu=distance,
                desired_speed_uu_s=desired,
            )
        if fallback:
            status = "fallback_defend:" + status
        return MotorOutput(controls, target, skill, status, metrics)
