import math

import pytest

from rleval.actions import NEUTRAL, parse_action
from rleval.motor import LocalMotorController, scripted_plan


def state(**changes):
    value = {
        "ready": True,
        "active": True,
        "ended": False,
        "packet_age_s": 0.0,
        "team": 0,
        "car_position": {"x": 0.0, "y": 0.0, "z": 17.0},
        "car_velocity": {"x": 500.0, "y": 0.0, "z": 0.0},
        "car_rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        "car_angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "car_has_wheel_contact": True,
        "car_boost": 50.0,
        "ball_position": {"x": 1000.0, "y": 0.0, "z": 93.0},
        "ball_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
    }
    value.update(changes)
    return value


def plan(skill="drive_to", **changes):
    value = {
        "skill": skill,
        "objective": "Test mechanics",
        "target_position": {"x": 2000.0, "y": 0.0, "z": 17.0},
        "desired_speed": 1500.0,
        "boost_policy": "allowed",
        "jump_policy": "allowed",
        "guidance": "Logged guidance has no executable meaning.",
    }
    value.update(changes)
    return value


@pytest.mark.parametrize("side", [-1, 1])
def test_ground_steering_is_signed_and_continuous(side):
    output = LocalMotorController().tick(
        state(), plan(target_position={"x": 2000, "y": side * 200, "z": 17}), 0
    )
    assert 0 < side * output.controls["steer"] < 0.5
    assert output.controls["throttle"] > 0
    assert output.metrics["surface"] == "floor"


@pytest.mark.parametrize("skill", ["intercept", "shoot", "defend"])
def test_dynamic_skills_recompute_from_live_ball_instead_of_advisory_target(skill):
    motor = LocalMotorController()
    advisory = plan(skill, target_position={"x": -3000, "y": -3000, "z": 17})
    first = motor.tick(state(), advisory, 0)
    second = motor.tick(state(ball_position={"x": 1500, "y": 800, "z": 93}), advisory, 1 / 60)
    assert first.target != second.target
    assert first.target != advisory["target_position"]
    assert first.metrics["prediction_kind"] == "linear_xy_no_bounces"
    assert 0 < first.metrics["prediction_horizon_s"] <= 1.2


def test_intercept_incorporates_ball_velocity_with_bounded_horizon():
    output = LocalMotorController().tick(
        state(ball_velocity={"x": 300, "y": 200, "z": 0}), plan("intercept"), 0
    )
    horizon = output.metrics["prediction_horizon_s"]
    assert output.target["x"] == pytest.approx(1000 + 300 * horizon)
    assert output.target["y"] == pytest.approx(200 * horizon)


@pytest.mark.parametrize("team,sign", [(0, -1), (1, 1)])
def test_defend_targets_own_half_for_either_team(team, sign):
    output = LocalMotorController().tick(state(team=team), plan("defend"), 0)
    assert sign * output.target["y"] >= 600


def test_no_plan_is_explicit_conservative_defend_fallback():
    output = LocalMotorController().tick(state(), None, 0)
    assert output.skill == "defend"
    assert output.status.startswith("fallback_defend:")
    assert output.metrics["fallback"]
    assert not output.controls["boost"] and not output.controls["jump"]


@pytest.mark.parametrize("speed", [-1000, 1000])
def test_stop_brakes_signed_motion_then_releases_before_reversal(speed):
    motor = LocalMotorController()
    output = motor.tick(state(car_velocity={"x": speed, "y": 0, "z": 0}), plan("stop"), 0)
    assert output.controls["throttle"] == -math.copysign(1, speed)
    settled = motor.tick(
        state(car_velocity={"x": math.copysign(45, speed), "y": 0, "z": 0}), plan("stop"), 1 / 60
    )
    assert settled.controls["throttle"] == 0
    assert settled.status == "stopped"


def test_drive_to_arrival_stops_and_preserves_requested_waypoint():
    target = {"x": 20, "y": 0, "z": 17}
    output = LocalMotorController().tick(
        state(car_velocity={"x": 0, "y": 0, "z": 0}), plan(target_position=target), 0
    )
    assert output.target == target
    assert output.status == "arrived"
    assert output.controls == NEUTRAL


def test_drive_to_clamps_arena_and_routes_goal_entry_through_opening():
    motor = LocalMotorController()
    outside = motor.tick(state(), plan(target_position={"x": 8192, "y": 10240, "z": 4000}), 0)
    assert outside.target["x"] <= 3900 and outside.target["y"] <= 5000
    assert math.hypot(outside.target["x"] - 2944, outside.target["y"] - 3968) <= 1000.001
    assert outside.target["z"] == 17
    direct = motor.tick(state(), plan(target_position={"x": 0, "y": 5500, "z": 17}), 0.02)
    assert direct.target["y"] == 5500
    blocked = motor.tick(
        state(car_position={"x": 3000, "y": 4900, "z": 17}),
        plan(target_position={"x": 0, "y": 5500, "z": 17}),
        0.04,
    )
    assert blocked.target == {"x": 0, "y": 4750, "z": 17}
    exiting = motor.tick(
        state(car_position={"x": 500, "y": 5500, "z": 17}),
        plan(target_position={"x": 3000, "y": 4900, "z": 17}),
        0.06,
    )
    assert exiting.target == {"x": 0, "y": 4750, "z": 17}


def test_boost_requires_permission_alignment_speed_need_and_floor():
    moving = state(car_velocity={"x": 900, "y": 0, "z": 0})
    motor = LocalMotorController()
    assert motor.tick(moving, plan(boost_policy="use", desired_speed=2200), 0).controls["boost"]
    assert not motor.tick(moving, plan(boost_policy="conserve", desired_speed=2200), 0.02).controls[
        "boost"
    ]
    assert not motor.tick(
        moving, plan(target_position={"x": 0, "y": 2000, "z": 17}, desired_speed=2200), 0.04
    ).controls["boost"]
    assert not motor.tick(moving, plan(desired_speed=800), 0.06).controls["boost"]
    assert not motor.tick(state(car_boost=0), plan(desired_speed=2200), 0.08).controls["boost"]


def test_reachable_ball_jump_is_bounded_respects_permission_and_resets():
    near = state(ball_position={"x": 180, "y": 0, "z": 180})
    motor = LocalMotorController()
    assert motor.tick(near, plan("intercept"), 0).controls["jump"]
    airborne = dict(near, car_has_wheel_contact=False, car_position={"x": 40, "y": 0, "z": 70})
    assert motor.tick(airborne, plan("intercept"), 0.06).controls["jump"]
    assert not motor.tick(airborne, plan("intercept"), 0.13).controls["jump"]
    assert not motor.tick(near, plan("intercept"), 0.3).controls["jump"]
    motor.reset()
    assert motor.tick(near, plan("intercept"), 0).controls["jump"]
    assert not motor.tick(near, plan("intercept", jump_policy="avoid"), 0.02).controls["jump"]
    motor.reset()
    assert not motor.tick(
        state(ball_position={"x": 180, "y": 0, "z": 700}), plan("shoot", jump_policy="needed"), 0
    ).controls["jump"]


def test_wall_contact_is_recovery_and_never_treated_as_drivable_floor():
    wall = state(
        car_position={"x": 4000, "y": 0, "z": 500},
        car_rotation={"pitch": 0, "yaw": 0, "roll": math.pi / 2},
    )
    output = LocalMotorController().tick(wall, plan("shoot", boost_policy="use"), 0)
    assert output.skill == "recover" and output.status == "recover_wall"
    assert output.metrics["surface"] == "wall"
    assert not output.controls["boost"] and not output.controls["jump"]


def test_air_recovery_levels_pitch_and_roll_with_bounded_controls():
    aerial = state(
        car_position={"x": 0, "y": 0, "z": 500},
        car_rotation={"pitch": 0.4, "yaw": 0, "roll": -0.5},
        car_has_wheel_contact=False,
    )
    output = LocalMotorController().tick(aerial, plan("recover"), 0)
    assert output.controls["pitch"] < 0 and output.controls["roll"] > 0
    assert not output.controls["boost"]
    parse_action({"controls": output.controls, "hold_ms": 50, "memory": ""})


def test_recovery_does_not_issue_a_second_jump_or_dodge():
    upside_down = state(
        car_position={"x": 0, "y": 0, "z": 40},
        car_velocity={"x": 0, "y": 0, "z": 0},
        car_rotation={"pitch": 0, "yaw": 0, "roll": math.pi},
        car_has_wheel_contact=False,
        car_has_jumped=True,
    )
    output = LocalMotorController().tick(upside_down, plan("recover"), 0)
    assert not output.controls["jump"]
    assert output.controls["roll"] != 0


@pytest.mark.parametrize(
    "change,status",
    [
        ({"ready": False}, "inactive"),
        ({"active": False}, "inactive"),
        ({"packet_age_s": 0.2}, "stale"),
        ({"car_position": {"x": float("nan"), "y": 0, "z": 17}}, "invalid_state"),
        ({"car_has_wheel_contact": "yes"}, "invalid_state"),
    ],
)
def test_invalid_or_stale_input_neutralizes(change, status):
    output = LocalMotorController().tick(state(**change), plan(), 0)
    assert output.controls == NEUTRAL
    assert output.status == status


def test_invalid_plan_and_backward_clock_neutralize():
    motor = LocalMotorController()
    assert motor.tick(state(), plan(skill="execute_code"), 0).status == "invalid_plan"
    assert motor.tick(state(), plan(target_position=None), 0.1).status == "invalid_plan"
    assert motor.tick(state(), plan(), 0).status == "invalid_time"


def test_scripted_baseline_is_a_plan_consumed_by_the_same_motor():
    snapshot = state()
    baseline = scripted_plan(snapshot)
    assert baseline["skill"] == "shoot"
    assert "Scripted" in baseline["objective"]
    output = LocalMotorController().tick(snapshot, baseline, 0)
    assert output.skill == "shoot"
    assert not output.metrics["fallback"]
    assert scripted_plan(state(ball_position={"x": 0, "y": -1000, "z": 93}))["skill"] == "defend"
