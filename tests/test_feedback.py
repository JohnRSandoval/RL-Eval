import math

import pytest

from rleval.feedback import FeedbackTracker


def state(x=0, yaw=0, speed=0, ready=True):
    return {
        "ready": ready,
        "car_position": {"x": x, "y": 0, "z": 17},
        "car_rotation": {"yaw": yaw},
        "car_velocity": {"x": speed, "y": 0, "z": 0},
        "last_input": {"steer": 0.5},
        "control_feedback": {"last_completed": {"commanded_duration_ms": 150}},
    }


def test_feedback_records_observed_motion_and_actual_lease_separately():
    tracker = FeedbackTracker()
    tracker.observe(0, state(speed=100))
    tracker.observe(0.25, state(x=40, yaw=0.1, speed=220))
    transition = tracker.recent()[0]
    assert transition["observed_interval_ms"] == 250
    assert transition["displacement_uu"] == 40
    assert transition["heading_change_degrees"] == pytest.approx(5.7)
    assert transition["completed_command"]["commanded_duration_ms"] == 150
    assert transition["speed_before_uu_s"] == 100
    assert transition["speed_after_uu_s"] == 220


def test_feedback_wraps_yaw_and_clears_across_gameplay_reset():
    tracker = FeedbackTracker()
    tracker.observe(0, state(yaw=math.pi - 0.01))
    tracker.observe(0.25, state(yaw=-math.pi + 0.01))
    assert tracker.recent()[0]["heading_change_degrees"] == 1.1
    tracker.observe(0.3, state(ready=False))
    tracker.observe(0.6, state(x=2000))
    assert tracker.recent() == []
