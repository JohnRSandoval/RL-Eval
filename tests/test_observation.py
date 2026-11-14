"""Geometry conventions and the explicit telemetry-to-model boundary."""

import json
import math

import pytest

from rleval.backend import MockBackend
from rleval.observation import build_game_state, orientation_vectors


def snapshot(tmp_path):
    backend = MockBackend({}, tmp_path)
    backend.start()
    return backend.snapshot()


@pytest.mark.parametrize(
    ("rotation", "forward", "right", "up"),
    [
        ({"pitch": 0, "yaw": 0, "roll": 0}, (1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ({"pitch": 0, "yaw": math.pi / 2, "roll": 0}, (0, 1, 0), (-1, 0, 0), (0, 0, 1)),
        ({"pitch": math.pi / 2, "yaw": 0, "roll": 0}, (0, 0, 1), (0, 1, 0), (-1, 0, 0)),
        ({"pitch": 0, "yaw": 0, "roll": math.pi / 2}, (1, 0, 0), (0, 0, -1), (0, 1, 0)),
    ],
)
def test_cardinal_rlbot_axes(rotation, forward, right, up):
    axes = orientation_vectors(rotation)
    assert tuple(axes["forward"].values()) == pytest.approx(forward, abs=1e-12)
    assert tuple(axes["right"].values()) == pytest.approx(right, abs=1e-12)
    assert tuple(axes["up"].values()) == pytest.approx(up, abs=1e-12)


def test_combined_rotation_is_orthonormal():
    axes = orientation_vectors({"pitch": 0.3, "yaw": -0.7, "roll": 1.1})
    for name, vector in axes.items():
        assert math.hypot(*vector.values()) == pytest.approx(1)
        for other_name, other in axes.items():
            if name != other_name:
                assert sum(vector[axis] * other[axis] for axis in vector) == pytest.approx(
                    0, abs=1e-12
                )


def test_relative_ball_at_yaw_zero_and_velocity_difference(tmp_path):
    raw = snapshot(tmp_path)
    raw.update(
        car_position={"x": 0, "y": 0, "z": 0},
        car_velocity={"x": 2, "y": 1, "z": 0},
        car_rotation={"pitch": 0, "yaw": 0, "roll": 0},
        ball_position={"x": 10, "y": 10, "z": 5},
        ball_velocity={"x": 10, "y": 4, "z": 3},
    )
    state = build_game_state(raw)
    relative = state["relative_ball"]
    assert relative["position"] == {"forward": 10, "right": 10, "up": 5}
    assert relative["velocity"] == {"forward": 8, "right": 3, "up": 3}
    assert relative["distance"] == 15
    assert relative["bearing_radians"] == pytest.approx(math.pi / 4)


def test_bearing_is_left_when_facing_positive_y_with_ball_positive_x(tmp_path):
    raw = snapshot(tmp_path)
    raw.update(car_position={"x": 0, "y": 0, "z": 0}, ball_position={"x": 10, "y": 0, "z": 0})
    relative = build_game_state(raw)["relative_ball"]
    assert relative["position"]["right"] == pytest.approx(-10)
    assert relative["bearing_radians"] == pytest.approx(-math.pi / 2)


def test_allowlist_does_not_leak_raw_internal_fields(tmp_path):
    raw = snapshot(tmp_path)
    raw.update(
        api_key="secret-key-value",
        raw_packet="raw-packet-value",
        prediction=["future-ball"],
        error="sensitive-error",
    )
    raw["opponents"][0].update(name="account-name-value", api_key="opponent-secret")
    state = build_game_state(raw)
    text = json.dumps(state, allow_nan=False)
    for forbidden in (
        "secret-key-value",
        "raw-packet-value",
        "future-ball",
        "sensitive-error",
        "account-name-value",
        "opponent-secret",
    ):
        assert forbidden not in text
    assert state["frame_num"] == raw["frame_num"]
    assert state["packet_received_monotonic_s"] == raw["packet_received_monotonic_s"]
    assert state["opponents"][0]["forward"]["y"] == pytest.approx(-1)
    state["player"]["position"]["x"] = 123
    assert raw["car_position"]["x"] == 0


@pytest.mark.parametrize("field", ["car_position", "ball_velocity", "car_angular_velocity"])
def test_nonfinite_vectors_rejected(tmp_path, field):
    raw = snapshot(tmp_path)
    raw[field]["x"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_game_state(raw)


def test_nonfinite_rotation_and_opponent_rejected(tmp_path):
    raw = snapshot(tmp_path)
    raw["car_rotation"]["yaw"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        build_game_state(raw)
    raw = snapshot(tmp_path)
    raw["car_position"]["x"] = 10**1000
    with pytest.raises(ValueError, match="finite"):
        build_game_state(raw)
    raw = snapshot(tmp_path)
    raw["opponents"][0]["velocity"]["z"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        build_game_state(raw)


def test_missing_ball_is_explicit_and_missing_car_is_error(tmp_path):
    raw = snapshot(tmp_path)
    raw["ball_position"] = None
    state = build_game_state(raw)
    assert state["ball"] is None and state["relative_ball"] is None
    del raw["car_position"]
    with pytest.raises(ValueError, match="vector"):
        build_game_state(raw)


def test_coincident_ball_and_finite_zero_bearing(tmp_path):
    raw = snapshot(tmp_path)
    raw["ball_position"] = dict(raw["car_position"])
    relative = build_game_state(raw)["relative_ball"]
    assert relative["distance"] == 0
    assert relative["bearing_radians"] == 0
