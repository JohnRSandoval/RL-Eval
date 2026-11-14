import math

import pytest

from rleval.actions import parse_action
from rleval.calibration_tasks import ReferencePolicy, TrialScorer, default_scenarios, measurements


def state(x=0, y=0, yaw=0, speed=0, yaw_rate=0, z=17, roll=0, contact=True):
    return {
        "ready": True,
        "car_position": {"x": x, "y": y, "z": z},
        "car_velocity": {"x": speed * math.cos(yaw), "y": speed * math.sin(yaw), "z": 0},
        "car_rotation": {"pitch": 0, "yaw": yaw, "roll": roll},
        "car_angular_velocity": {"x": 0, "y": 0, "z": yaw_rate},
        "car_has_wheel_contact": contact,
    }


def scenario(name):
    return next(s for s in default_scenarios() if s.name == name)


def test_suite_is_mirrored_bounded_and_covers_forward_and_reverse_braking():
    trials = default_scenarios()
    assert len(trials) == 8
    assert len({s.name for s in trials}) == 8
    assert sum(s.duration_s for s in trials) == 54
    left, right = scenario("turn_left"), scenario("turn_right")
    assert left.target["y"] == -right.target["y"]
    assert scenario("brake_forward").velocity["x"] > 0
    assert scenario("brake_reverse").velocity["x"] < 0


def test_signed_geometry_and_reference_turns_are_mirrored():
    for name, sign in [("turn_left", -1), ("turn_right", 1)]:
        trial = scenario(name)
        observation = state(x=trial.position["x"], speed=600)
        assert measurements(observation, trial)["heading_error_deg"] == pytest.approx(sign * 30)
        assert ReferencePolicy.decide(observation, trial)["steer"] == sign * 0.5


def test_heading_success_requires_dwell_and_low_yaw_rate():
    trial = scenario("align_left")
    scorer = TrialScorer(trial)
    assert not scorer.update(0, state(x=-1200, yaw=math.pi / 3, speed=450))
    assert not scorer.update(0.2, state(x=-1200, yaw=0, speed=450, yaw_rate=1.0))
    assert not scorer.update(0.4, state(x=-1200, yaw=0, speed=450))
    assert not scorer.update(0.6, state(x=-1200, yaw=0, speed=450))
    assert scorer.update(0.66, state(x=-1200, yaw=0, speed=450))
    assert scorer.finish()["initial_heading_error_deg"] == pytest.approx(-60)


def test_crossing_waypoint_at_speed_does_not_pass():
    trial = scenario("waypoint_right")
    scorer = TrialScorer(trial)
    assert not scorer.update(0, state(x=800, y=600, speed=900))
    assert not scorer.update(0.5, state(x=800, y=600, speed=900))
    assert not scorer.update(0.8, state(x=800, y=600, speed=100))
    assert scorer.update(1.1, state(x=800, y=600, speed=100))


def test_wall_contact_is_not_flat_floor_and_cannot_pass_braking():
    trial = scenario("brake_forward")
    scorer = TrialScorer(trial)
    wall = state(z=600, roll=math.pi / 2, speed=0)
    assert not measurements(wall, trial)["flat_floor"]
    assert not scorer.update(0, wall)
    assert not scorer.update(1, wall)
    assert ReferencePolicy.decide(wall, trial)["throttle"] == 0


def test_braking_is_signed_and_does_not_reverse_a_stopped_car():
    for speed in [-1000, -800, 800, 1000]:
        control = ReferencePolicy.decide(state(speed=speed), scenario("brake_forward"))
        assert control["throttle"] == -math.copysign(1, speed)
    for speed in [-700, -200, -50, 10, 100, 300, 700]:
        control = ReferencePolicy.decide(state(speed=speed), scenario("brake_forward"))
        assert control["throttle"] == 0


def test_reference_releases_steer_instead_of_countersteering_near_alignment():
    # Synthetic near-aligned states cover both signs of angular momentum.
    for bearing, yaw_rate in [(2.0, 50.0), (-1.0, -50.0), (6.0, 45.0), (-6.0, -45.0)]:
        observed = state(
            x=-1200, yaw=-math.radians(bearing), speed=420, yaw_rate=math.radians(yaw_rate)
        )
        assert ReferencePolicy.decide(observed, scenario("align_right"))["steer"] == 0


def test_reference_coasts_through_cruise_overspeed_and_low_speed_waypoint_stopping():
    # Coast at moderate overspeed and near a waypoint to avoid reversing.
    assert ReferencePolicy.decide(state(speed=600), scenario("align_left"))["throttle"] == 0
    near_waypoint = state(x=650, y=600, speed=500)
    control = ReferencePolicy.decide(near_waypoint, scenario("waypoint_right"))
    assert control["throttle"] == control["steer"] == 0


@pytest.mark.parametrize("name,sign", [("brake_forward", 1), ("brake_reverse", -1)])
def test_brake_reversal_is_measured_separately_from_stopping_success(name, sign):
    scorer = TrialScorer(scenario(name))
    assert not scorer.update(0, state(speed=sign * 1000))
    assert not scorer.update(0.3, state(speed=-sign * 150))
    assert not scorer.update(0.5, state(speed=0))
    assert scorer.update(0.8, state(speed=0))
    result = scorer.finish()
    assert result["reverse_overshoot_speed_uu_s"] == pytest.approx(150)
    assert result["min_forward_speed_uu_s"] == min(sign * 1000, -sign * 150)
    assert result["max_forward_speed_uu_s"] == max(sign * 1000, -sign * 150)


def test_dwell_resets_on_motion_and_cannot_succeed_after_deadline():
    scorer = TrialScorer(scenario("brake_forward"))
    assert not scorer.update(0, state(speed=0))
    assert not scorer.update(0.2, state(speed=90))
    assert not scorer.update(0.4, state(speed=0))
    assert not scorer.update(0.6, state(speed=0))
    assert scorer.update(0.7, state(speed=0))
    late = TrialScorer(scenario("brake_forward"))
    assert not late.update(4.9, state(speed=0))
    assert not late.update(5.3, state(speed=0))


def test_yaw_unwrap_does_not_count_crossing_pi_as_a_full_rotation():
    scorer = TrialScorer(scenario("brake_forward"))
    scorer.update(0, state(yaw=math.pi - 0.1, speed=1000))
    scorer.update(0.1, state(yaw=-math.pi + 0.1, speed=1000))
    assert scorer.finish()["total_absolute_yaw_rad"] == pytest.approx(0.2)
    for index in range(1, 34):
        scorer.update(0.1 + index * 0.1, state(yaw=-math.pi + 0.1 + index * 0.2, speed=1000))
    assert scorer.finish()["circling"]


def test_reference_stays_in_model_action_space():
    for trial in default_scenarios():
        for yaw in [-math.pi, -1, 0, 1, math.pi]:
            controls = ReferencePolicy.decide(state(x=-1000, yaw=yaw, speed=700), trial)
            assert controls["steer"] in {-0.5, 0, 0.5}
            assert controls["throttle"] in {-1, 0, 1}
            assert controls["boost"] is controls["jump"] is False
            parse_action({"controls": controls, "hold_ms": 300, "memory": ""})


def test_bad_or_decreasing_measurements_fail_closed():
    scorer = TrialScorer(scenario("brake_forward"))
    scorer.update(1, state())
    with pytest.raises(ValueError, match="nondecreasing"):
        scorer.update(0.5, state())
    with pytest.raises(ValueError, match="finite"):
        scorer.update(2, state(speed=float("nan")))


@pytest.mark.parametrize("packet_age", [0.0, 0.3])
def test_a_frozen_packet_cannot_satisfy_physical_dwell(packet_age):
    scorer = TrialScorer(scenario("brake_forward"))
    assert not scorer.update(0, dict(state(), frame_num=1, packet_age_s=0))
    assert not scorer.update(0.3, dict(state(), frame_num=1, packet_age_s=packet_age))
    assert not scorer.finish()["success"]
    assert scorer.finish()["fresh_samples"] == 1


def test_stale_or_missing_packets_reset_dwell_before_fresh_observations_resume():
    scorer = TrialScorer(scenario("brake_forward"))
    assert not scorer.update(0, dict(state(), frame_num=1, packet_age_s=0))
    assert not scorer.update(0.08, dict(state(), frame_num=2, packet_age_s=0))
    assert not scorer.update(0.3, dict(state(), frame_num=2, packet_age_s=0.22))
    assert not scorer.update(0.31, dict(state(), frame_num=3, packet_age_s=0))
    for index in range(1, 6):
        assert not scorer.update(
            0.31 + index * 0.04, dict(state(), frame_num=3 + index, packet_age_s=0)
        )
    assert scorer.update(0.57, dict(state(), frame_num=9, packet_age_s=0))

    missing = TrialScorer(scenario("brake_forward"))
    assert not missing.update(0, dict(state(), frame_num=1, packet_age_s=0))
    assert not missing.update(0.3, dict(state(), frame_num=40, packet_age_s=0))
    assert not missing.finish()["success"]
