"""Observed state transitions and acknowledged inputs, not inferred model reasoning."""

import copy
import math
from collections import deque


class FeedbackTracker:
    def __init__(self, *, sample_interval_s=0.2, maxlen=4):
        self.sample_interval_s = sample_interval_s
        self.history = deque(maxlen=maxlen)
        self.previous = None
        self.previous_time = None

    def clear(self):
        self.history.clear()
        self.previous = self.previous_time = None

    def observe(self, elapsed_s, snapshot):
        if not snapshot.get("ready"):
            self.clear()
            return
        if (
            self.previous_time is not None
            and elapsed_s - self.previous_time < self.sample_interval_s
        ):
            return
        current = copy.deepcopy(snapshot)
        if self.previous is not None:
            before = self.previous
            yaw0 = before.get("car_rotation", {}).get("yaw", 0.0)
            yaw1 = current.get("car_rotation", {}).get("yaw", 0.0)
            yaw_delta = math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))
            p0, p1 = before.get("car_position", {}), current.get("car_position", {})
            v0, v1 = before.get("car_velocity", {}), current.get("car_velocity", {})
            speed = lambda v: math.hypot(*(v.get(k, 0.0) for k in ("x", "y", "z")))
            item = {
                "observed_interval_ms": round((elapsed_s - self.previous_time) * 1000, 1),
                "displacement_uu": round(
                    math.dist(
                        [p0.get(k, 0.0) for k in ("x", "y", "z")],
                        [p1.get(k, 0.0) for k in ("x", "y", "z")],
                    ),
                    1,
                ),
                "heading_change_degrees": round(math.degrees(yaw_delta), 1),
                "speed_before_uu_s": round(speed(v0), 1),
                "speed_after_uu_s": round(speed(v1), 1),
                "acknowledged_input_at_start": before.get("last_input", {}),
                "acknowledged_input_at_end": current.get("last_input", {}),
                "completed_command": current.get("control_feedback", {}).get("last_completed"),
                "note": "Measured interval may include command replacements; not attributed to one action.",
            }
            self.history.append(item)
        self.previous, self.previous_time = current, elapsed_s

    def recent(self):
        return copy.deepcopy(list(self.history))
