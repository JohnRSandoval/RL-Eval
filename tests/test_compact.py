import json
import math

import pytest

from rleval.actions import NEUTRAL
from rleval.compact import build_motor_observation, validate_motor_observation


def snapshot(**changes):
    return {
        "car_position": {"x": 0, "y": 0, "z": 17},
        "car_velocity": {"x": 500, "y": 100, "z": 0},
        "car_angular_velocity": {"x": 0, "y": 0, "z": 0.1},
        "car_rotation": {"pitch": 0, "yaw": 0, "roll": 0},
        "car_has_wheel_contact": True,
        "packet_age_s": 0.00312,
        "last_input": dict(NEUTRAL, steer=-0.5),
        **changes,
    }


def test_mirrored_targets_have_opposite_bearings_in_car_frame():
    for side in (-1, 1):
        observation = build_motor_observation(
            snapshot(), {"x": 1000, "y": side * 1000, "z": 17}, 500, "Align"
        )
        assert observation["target"]["bearing_degrees"] == side * 45
        assert observation["motion"]["forward_speed_uu_s"] == 500
        assert observation["motion"]["lateral_speed_uu_s"] == 100
        assert observation["packet_age_ms"] == 3.12
    rotated = build_motor_observation(
        snapshot(car_rotation={"pitch": 0, "yaw": math.pi / 2, "roll": 0}),
        {"x": 1000, "y": 1000, "z": 17},
        500,
        "Align",
    )
    assert rotated["target"]["bearing_degrees"] == -45
    assert rotated["motion"]["forward_speed_uu_s"] == 100
    assert rotated["motion"]["lateral_speed_uu_s"] == -500


@pytest.mark.parametrize(
    "contact,roll,z,expected",
    [
        (False, 0, 17, "air"),
        (True, 0, 17, "floor"),
        (True, math.pi / 2, 600, "wall_or_ramp"),
        (True, math.pi, 2020, "ceiling"),
        (True, 0.7, 170, "wall_or_ramp"),
    ],
)
def test_surface_contact_is_not_mislabeled_as_flat_ground(contact, roll, z, expected):
    raw = snapshot(
        car_has_wheel_contact=contact,
        car_position={"x": 0, "y": 0, "z": z},
        car_rotation={"pitch": 0, "yaw": 0, "roll": roll},
    )
    observation = build_motor_observation(raw, None, 0, "Brake")
    assert observation["surface"]["classification"] == expected
    assert observation["surface"]["wheel_contact"] is contact
    assert "on_ground" not in json.dumps(observation)


def test_feedback_distinguishes_acknowledged_input_from_command_and_limits_history():
    raw = snapshot(
        privileged_sentinel="PRIVATE",
        control_feedback={
            "commanded_controls": dict(NEUTRAL, throttle=1),
            "selected_controls": dict(NEUTRAL, throttle=1),
            "requested_hold_ms": 300,
            "command_age_ms": 120,
            "remaining_ms": 180,
            "active": True,
            "command_id": 2,
            "last_completed": {
                "command_id": 1,
                "controls": dict(NEUTRAL, steer=-0.5),
                "requested_hold_ms": 300,
                "commanded_duration_ms": 158.126,
                "end_reason": "replaced",
            },
            "private": "PRIVATE",
        },
    )
    history = [
        {"delta_heading_degrees": i, "controls": dict(NEUTRAL), "private": "PRIVATE"}
        for i in range(5)
    ]
    o = build_motor_observation(raw, None, 0, "Brake", history)
    assert o["acknowledged_input"]["steer"] == -0.5
    assert o["command_feedback"]["commanded_controls"]["throttle"] == 1
    assert o["command_feedback"]["last_completed"]["commanded_duration_ms"] == 158.13
    assert [x["delta_heading_degrees"] for x in o["recent_transitions"]] == [2, 3, 4]
    assert "PRIVATE" not in json.dumps(o)
    assert validate_motor_observation(o) == o


@pytest.mark.parametrize(
    "field,value",
    [("packet_age_s", float("nan")), ("packet_age_s", -1), ("car_has_wheel_contact", 1)],
)
def test_bad_measurements_fail_closed(field, value):
    with pytest.raises(ValueError):
        build_motor_observation(snapshot(**{field: value}), None, 0, "Brake")


def test_missing_or_nonfinite_compact_fields_are_not_accepted():
    observation = build_motor_observation(snapshot(), None, 0, "Brake")
    del observation["motion"]["forward_speed_uu_s"]
    with pytest.raises(ValueError):
        validate_motor_observation(observation)
    observation = build_motor_observation(snapshot(), None, 0, "Brake")
    observation["target"] = {
        "bearing_degrees": float("inf"),
        "distance_uu": 1,
        "forward_uu": 1,
        "right_uu": 0,
        "up_uu": 0,
    }
    with pytest.raises(ValueError):
        validate_motor_observation(observation)
