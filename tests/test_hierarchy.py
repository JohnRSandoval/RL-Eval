"""The tactical planner must never stall or outlive the motor-control loop."""

import copy
import json
import threading
import time

from rleval.actions import NEUTRAL
from rleval.backend import MockBackend
from rleval.config import load_config
from rleval.planner import PlannerReply
from rleval.providers import ModelReply
from rleval.runner import run_episode

PLAN = {
    "objective": "Defend the near post",
    "target_position": {"x": 600, "y": -4700, "z": 17},
    "desired_speed": 1000,
    "boost_policy": "conserve",
    "jump_policy": "allowed",
    "guidance": "Stay between the ball and the blue goal.",
}


class Planner:
    def __init__(self, callback=None):
        self.callback = callback
        self.contexts = []
        self.closed = threading.Event()

    def decide(self, context):
        self.contexts.append(copy.deepcopy(context))
        if self.callback:
            self.callback()
        return PlannerReply(copy.deepcopy(PLAN), {"total_tokens": 123}, "planner-1", "test-planner")

    def close(self):
        self.closed.set()


class Motor:
    def __init__(self, callback=None):
        self.callback = callback
        self.contexts = []
        self.closed = threading.Event()

    def decide(self, frames, context):
        assert frames == []
        self.contexts.append(copy.deepcopy(context))
        if self.callback:
            self.callback(context, len(self.contexts))
        return ModelReply({"controls": dict(NEUTRAL, throttle=1), "hold_ms": 50, "memory": ""}, {})

    def close(self):
        self.closed.set()


def configuration(tmp_path, **episode):
    return load_config(
        overrides={
            "observation": {"mode": "state"},
            "planner": {"enabled": True, "max_requests": 1, "interval_s": 1, "max_age_s": 1},
            "control": {"scheduling": "continuous", "max_decision_hz": 20},
            "episode": {"duration_s": 2, "max_decisions": 8, "max_errors": 1, **episode},
            "recording": {"output_dir": str(tmp_path), "telemetry_hz": 30},
        }
    )


def events(path):
    return [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]


def test_fast_controls_continue_while_planning_and_then_receive_the_plan(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def planning():
        entered.set()
        assert release.wait(2), "Fast controls stalled behind planner"

    def driving(context, number):
        if number == 3:
            assert entered.is_set()
            assert "strategy" not in context
            release.set()

    planner, motor = Planner(planning), Motor(driving)
    config = configuration(tmp_path)
    try:
        path, summary = run_episode(config, synthetic=True, client=motor, planner_client=planner)
    finally:
        release.set()
    assert summary["status"] == "decision_limit"
    assert summary["accepted_actions"] == 8
    assert len(planner.contexts) == 1
    contexts = [c["strategy"] for c in motor.contexts if "strategy" in c]
    assert contexts and all(c["plan"] == PLAN for c in contexts)
    assert all(c["version"] == 1 and c["model"] == "test-planner" for c in contexts)
    plan_event = next(e for e in events(path) if e["type"] == "planner_decision")
    assert plan_event["accepted"] and plan_event["usage"]["total_tokens"] == 123
    assert planner.closed.wait(1) and motor.closed.wait(1)


def test_expired_plan_is_removed_from_later_motor_observations(tmp_path):
    planner, motor = Planner(), Motor()
    config = configuration(tmp_path, duration_s=1.35, max_decisions=100)
    path, summary = run_episode(config, synthetic=True, client=motor, planner_client=planner)
    assert summary["status"] == "time_limit"
    assert any("strategy" in c for c in motor.contexts)
    after_expiry = [c for c in motor.contexts if c["elapsed_s"] >= 1.1]
    assert after_expiry and all("strategy" not in c for c in after_expiry)
    assert any(e["type"] == "planner_invalidated" for e in events(path))


def test_gameplay_reset_invalidates_an_accepted_plan(tmp_path):
    interrupt = threading.Event()
    resumed = threading.Event()

    class InterruptedBackend(MockBackend):
        seen = False

        def snapshot(self):
            state = super().snapshot()
            if interrupt.is_set() and not self.seen:
                self.seen = True
                state.update(ready=False, active=False, phase="Countdown")
            elif self.seen:
                resumed.set()
            return state

    observations_after_resume = []

    def driving(context, number):
        if resumed.is_set():
            observations_after_resume.append(context)
        if "strategy" in context:
            interrupt.set()

    planner, motor = Planner(), Motor(driving)
    config = configuration(tmp_path)
    path, summary = run_episode(
        config,
        synthetic=True,
        backend=InterruptedBackend(config["rlbot"], tmp_path),
        client=motor,
        planner_client=planner,
    )
    assert summary["status"] == "decision_limit"
    assert any("strategy" in c for c in motor.contexts)
    assert observations_after_resume and all("strategy" not in c for c in observations_after_resume)
    assert any(e["type"] == "planner_invalidated" for e in events(path))


def test_plan_spanning_a_gameplay_reset_is_rejected(tmp_path):
    requested, resumed = threading.Event(), threading.Event()

    class InterruptedBackend(MockBackend):
        seen = False

        def snapshot(self):
            state = super().snapshot()
            if requested.is_set() and not self.seen:
                self.seen = True
                state.update(ready=False, active=False, phase="Countdown")
            elif self.seen:
                resumed.set()
            return state

    def planning():
        requested.set()
        assert resumed.wait(2)

    planner, motor = Planner(planning), Motor()
    config = configuration(tmp_path)
    path, summary = run_episode(
        config,
        synthetic=True,
        backend=InterruptedBackend(config["rlbot"], tmp_path),
        client=motor,
        planner_client=planner,
    )
    assert summary["status"] == "decision_limit"
    assert all("strategy" not in c for c in motor.contexts)
    result = next(e for e in events(path) if e["type"] == "planner_decision")
    assert not result["accepted"] and "gameplay changed" in result["error"]


def test_episode_deadline_neutralizes_without_waiting_for_planner(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def planning():
        entered.set()
        assert release.wait(3)

    planner, motor = Planner(planning), Motor()
    config = configuration(tmp_path, duration_s=0.15, max_decisions=100)
    backend = MockBackend(config["rlbot"], tmp_path)
    started = time.monotonic()
    try:
        _, summary = run_episode(
            config, synthetic=True, backend=backend, client=motor, planner_client=planner
        )
        assert entered.is_set()
        assert summary["status"] == "time_limit"
        assert time.monotonic() - started < 1.25
        assert backend.snapshot()["ended"]
        assert backend.snapshot()["controls"] == NEUTRAL
        assert not planner.closed.is_set()
    finally:
        release.set()
        assert planner.closed.wait(1)
    assert backend.snapshot()["controls"] == NEUTRAL
