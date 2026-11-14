"""Repeatable ground-control tasks and a separate diagnostic reference policy.

Success is measured from game telemetry, never from model assertions. The
reference policy is explicitly a hand-written baseline, not part of a model's
action execution. It uses the same discrete axes as ground_axes_v1.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .actions import NEUTRAL
from .observation import orientation_vectors


def _finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _vector(value, label):
    if not isinstance(value, dict) or set(value) != {"x", "y", "z"}:
        raise ValueError(f"{label} must contain x, y, z")
    return {axis: _finite(value[axis], f"{label}.{axis}") for axis in ("x", "y", "z")}


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str
    position: dict
    velocity: dict
    yaw: float
    target: dict | None
    desired_speed: float
    duration_s: float
    description: str

    def __post_init__(self):
        if self.kind not in {"turn", "align", "waypoint", "brake"}:
            raise ValueError("Unknown calibration task kind")
        if not self.name or not self.description:
            raise ValueError("Scenario name and description are required")
        _vector(self.position, "position")
        _vector(self.velocity, "velocity")
        _finite(self.yaw, "yaw")
        if self.target is not None:
            _vector(self.target, "target")
        elif self.kind != "brake":
            raise ValueError("Turn, align, and waypoint tasks require a target")
        if not 0 <= _finite(self.desired_speed, "desired_speed") <= 2300:
            raise ValueError("desired_speed must be between 0 and 2300")
        if not 0 < _finite(self.duration_s, "duration_s") <= 60:
            raise ValueError("duration_s must be between 0 and 60")

    def to_dict(self):
        return asdict(self)


def default_scenarios() -> list[Scenario]:
    """Eight mirrored or signed tasks, 54 seconds maximum per controller."""
    scenarios = []
    for side, name in [(-1, "left"), (1, "right")]:
        scenarios.append(
            Scenario(
                f"turn_{name}",
                "turn",
                {"x": -1000.0, "y": 0.0, "z": 17.0},
                {"x": 600.0, "y": 0.0, "z": 0.0},
                0.0,
                {"x": 1800.0, "y": side * 2800 * math.tan(math.pi / 6), "z": 17.0},
                600.0,
                6.0,
                f"Turn toward the {name} waypoint, then straighten. Keep heading error within 8 degrees for 0.25 seconds.",
            )
        )
    for side, name in [(-1, "left"), (1, "right")]:
        yaw = -side * math.pi / 3
        scenarios.append(
            Scenario(
                f"align_{name}",
                "align",
                {"x": -1200.0, "y": 0.0, "z": 17.0},
                {"x": 350 * math.cos(yaw), "y": 350 * math.sin(yaw), "z": 0.0},
                yaw,
                {"x": 1600.0, "y": 0.0, "z": 17.0},
                450.0,
                8.0,
                f"Align with the waypoint initially 60 degrees to your {name}; hold error within 5 degrees for 0.25 seconds.",
            )
        )
    for side, name in [(-1, "left"), (1, "right")]:
        scenarios.append(
            Scenario(
                f"waypoint_{name}",
                "waypoint",
                {"x": -1200.0, "y": 0.0, "z": 17.0},
                {"x": 0.0, "y": 0.0, "z": 0.0},
                0.0,
                {"x": 800.0, "y": side * 600.0, "z": 17.0},
                900.0,
                8.0,
                "Drive to the waypoint, then brake. Stay within 250 units at less than 250 units/second for 0.25 seconds.",
            )
        )
    for velocity, name in [(1000.0, "forward"), (-800.0, "reverse")]:
        scenarios.append(
            Scenario(
                f"brake_{name}",
                "brake",
                {"x": 0.0, "y": 0.0, "z": 17.0},
                {"x": velocity, "y": 0.0, "z": 0.0},
                0.0,
                None,
                0.0,
                5.0,
                "Brake to below 80 units/second and stay stopped for 0.25 seconds. Do not accelerate in the opposite direction.",
            )
        )
    return scenarios


def measurements(snapshot: dict, scenario: Scenario) -> dict:
    """Measured instantaneous geometry; positive heading error means right."""
    position = _vector(snapshot.get("car_position"), "car_position")
    velocity = _vector(snapshot.get("car_velocity"), "car_velocity")
    rotation = snapshot.get("car_rotation")
    axes = orientation_vectors(rotation)
    angular = _vector(snapshot.get("car_angular_velocity"), "car_angular_velocity")
    contact = snapshot.get("car_has_wheel_contact", snapshot.get("car_on_ground", False))
    flat_floor = contact is True and position["z"] < 120 and axes["up"]["z"] > 0.9
    distance = bearing = None
    if scenario.target is not None:
        delta = {axis: scenario.target[axis] - position[axis] for axis in ("x", "y", "z")}
        ahead = sum(delta[a] * axes["forward"][a] for a in delta)
        right = sum(delta[a] * axes["right"][a] for a in delta)
        distance = math.hypot(delta["x"], delta["y"])
        bearing = math.atan2(right, ahead)
    return {
        "distance_uu": distance,
        "heading_error_rad": bearing,
        "heading_error_deg": math.degrees(bearing) if bearing is not None else None,
        "speed_uu_s": math.hypot(*velocity.values()),
        "forward_speed_uu_s": sum(velocity[a] * axes["forward"][a] for a in velocity),
        "yaw_rate_rad_s": angular["z"],
        "yaw_rad": rotation["yaw"],
        "flat_floor": flat_floor,
    }


class TrialScorer:
    """Require a stable, physically observed success condition for 250 ms.

    Braking success means a stable stop, not zero reverse overshoot. Signed
    forward-speed extrema and peak reversal are reported separately.
    """

    DWELL_S = 0.25
    MAX_PACKET_AGE_S = 0.1

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.initial = self.latest = None
        self.elapsed_s = 0.0
        self.samples = 0
        self.min_distance = None
        self.min_forward_speed = self.max_forward_speed = None
        self.total_yaw = self.net_yaw = 0.0
        self._dwell_started = None
        self._last_frame_num = None
        self._last_fresh_elapsed = None
        self.fresh_samples = 0
        self.success_time = None

    def update(self, elapsed_s: float, snapshot: dict) -> bool:
        elapsed_s = _finite(elapsed_s, "elapsed_s")
        if elapsed_s < self.elapsed_s or elapsed_s < 0:
            raise ValueError("Trial timestamps must be nonnegative and nondecreasing")
        packet_age = _finite(snapshot.get("packet_age_s", 0.0), "packet_age_s")
        if packet_age < 0:
            raise ValueError("packet_age_s must be nonnegative")
        frame_num = snapshot.get("frame_num")
        if frame_num is not None and (type(frame_num) is not int or frame_num < 0):
            raise ValueError("frame_num must be a nonnegative integer")
        new_frame = frame_num is None or frame_num != self._last_frame_num
        fresh = packet_age <= self.MAX_PACKET_AGE_S
        if not fresh or (
            frame_num is not None
            and self._last_fresh_elapsed is not None
            and elapsed_s - self._last_fresh_elapsed > self.MAX_PACKET_AGE_S
        ):
            # Time spent waiting for a frozen/missing packet is not evidence
            # that the physical success condition continued to hold.
            self._dwell_started = None
        if fresh and new_frame:
            self.fresh_samples += 1
            self._last_fresh_elapsed = elapsed_s
        self._last_frame_num = frame_num
        current = measurements(snapshot, self.scenario)
        if self.initial is None:
            self.initial = current
        if self.latest is not None:
            delta = math.remainder(current["yaw_rad"] - self.latest["yaw_rad"], 2 * math.pi)
            self.total_yaw += abs(delta)
            self.net_yaw += delta
        self.latest = current
        self.elapsed_s = elapsed_s
        self.samples += 1
        forward_speed = current["forward_speed_uu_s"]
        self.min_forward_speed = (
            forward_speed
            if self.min_forward_speed is None
            else min(self.min_forward_speed, forward_speed)
        )
        self.max_forward_speed = (
            forward_speed
            if self.max_forward_speed is None
            else max(self.max_forward_speed, forward_speed)
        )
        distance = current["distance_uu"]
        if distance is not None:
            self.min_distance = (
                distance if self.min_distance is None else min(self.min_distance, distance)
            )
        qualifies = fresh and current["flat_floor"] and snapshot.get("ready", True)
        if self.scenario.kind in {"turn", "align"}:
            tolerance = 8 if self.scenario.kind == "turn" else 5
            qualifies = qualifies and abs(current["heading_error_deg"]) <= tolerance
            qualifies = qualifies and abs(current["yaw_rate_rad_s"]) <= 0.6
        elif self.scenario.kind == "waypoint":
            qualifies = qualifies and distance <= 250 and current["speed_uu_s"] < 250
        else:
            qualifies = qualifies and current["speed_uu_s"] < 80
        if qualifies and elapsed_s <= self.scenario.duration_s and new_frame:
            if self._dwell_started is None:
                self._dwell_started = elapsed_s
            if elapsed_s - self._dwell_started >= self.DWELL_S - 1e-9 and self.success_time is None:
                self.success_time = elapsed_s
        elif not qualifies or elapsed_s > self.scenario.duration_s:
            self._dwell_started = None
        return self.success_time is not None

    def finish(self) -> dict:
        first, last = self.initial or {}, self.latest or {}
        reverse_overshoot = None
        initial_forward = first.get("forward_speed_uu_s", 0)
        if self.scenario.kind == "brake" and initial_forward != 0:
            reverse_overshoot = (
                max(0.0, -self.min_forward_speed)
                if initial_forward > 0
                else max(0.0, self.max_forward_speed)
            )
        return {
            "scenario": self.scenario.name,
            "kind": self.scenario.kind,
            "success": self.success_time is not None,
            "success_time_s": self.success_time,
            "elapsed_s": self.elapsed_s,
            "samples": self.samples,
            "fresh_samples": self.fresh_samples,
            "initial_distance_uu": first.get("distance_uu"),
            "min_distance_uu": self.min_distance,
            "final_distance_uu": last.get("distance_uu"),
            "initial_heading_error_deg": first.get("heading_error_deg"),
            "final_heading_error_deg": last.get("heading_error_deg"),
            "initial_speed_uu_s": first.get("speed_uu_s"),
            "final_speed_uu_s": last.get("speed_uu_s"),
            "min_forward_speed_uu_s": self.min_forward_speed,
            "max_forward_speed_uu_s": self.max_forward_speed,
            "reverse_overshoot_speed_uu_s": reverse_overshoot,
            "final_flat_floor": last.get("flat_floor"),
            "total_absolute_yaw_rad": self.total_yaw,
            "net_yaw_rad": self.net_yaw,
            "circling": abs(self.net_yaw) >= 2 * math.pi,
            "success_dwell_s": self.DWELL_S,
            "max_success_packet_age_s": self.MAX_PACKET_AGE_S,
        }


class ReferencePolicy:
    """Hand-written diagnostic baseline using exactly the ground-axes choices.

    The caller must report this as reference-assisted control, never as model
    output. It intentionally has no network delay; equal request rate does not
    make it a latency-matched model comparison.
    """

    @staticmethod
    def _stop_throttle(speed: float) -> float:
        # At 5 Hz a full brake command removes about 700 uu/s before the next
        # decision. Below this, neutral coasting avoids repeated reversals.
        return -math.copysign(1.0, speed) if abs(speed) > 750 else 0.0

    @staticmethod
    def decide(snapshot: dict, scenario: Scenario) -> dict:
        state = measurements(snapshot, scenario)
        controls = dict(NEUTRAL)
        if not state["flat_floor"]:
            return controls
        speed = state["forward_speed_uu_s"]
        if scenario.kind == "brake":
            controls["throttle"] = ReferencePolicy._stop_throttle(speed)
            return controls
        error = state["heading_error_rad"]
        # Ground yaw quickly settles after releasing steer. Countersteering for
        # an entire 200 ms interval instead creates an alternating yaw cycle.
        predicted_error = error - 0.08 * state["yaw_rate_rad_s"]
        if (
            abs(error) > math.radians(3)
            and abs(predicted_error) > math.radians(3)
            and error * predicted_error > 0
        ):
            controls["steer"] = math.copysign(0.5, error)
        desired_speed = scenario.desired_speed
        if abs(error) > math.pi / 3:
            desired_speed = min(desired_speed, 350.0)
        if scenario.kind == "waypoint":
            # Coast toward the stopping point using observed neutral drag
            # (~500 uu/s²); full braking remains available at higher speeds.
            desired_speed = min(
                desired_speed, math.sqrt(2 * 500 * max(0.0, state["distance_uu"] - 120))
            )
            if state["distance_uu"] < 250:
                controls["steer"] = 0.0
            if state["distance_uu"] < 220:
                desired_speed = 0.0
        if desired_speed == 0 or (scenario.kind == "waypoint" and speed > desired_speed + 80):
            controls["throttle"] = ReferencePolicy._stop_throttle(speed)
        elif speed < desired_speed - 120:
            controls["throttle"] = 1.0
        return controls
