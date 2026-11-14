"""Threading, lifecycle, and evidence tests; MockBackend does not simulate physics."""

import copy
import json
import threading
import time

import pytest

from rleval.actions import NEUTRAL
from rleval.assisted import MotorWorker
from rleval.backend import MockBackend
from rleval.config import load_config
from rleval.motor import MotorOutput
from rleval.planner import PlannerReply
from rleval.runner import run_episode


def config(tmp_path, *, planner=True, duration=0.25):
    return load_config(
        overrides={
            "observation": {"mode": "state"},
            "control": {"mode": "local_motor", "motor_hz": 60},
            "planner": {"enabled": planner, "interval_s": 1, "max_age_s": 1},
            "episode": {"duration_s": duration},
            "recording": {
                "output_dir": str(tmp_path / "runs"),
                "telemetry_hz": 30,
                "save_frames": False,
                "live_window": False,
            },
        }
    )


def plan():
    return {
        "skill": "drive_to",
        "objective": "Reach a fixed waypoint",
        "target_position": {"x": 0, "y": 1000, "z": 17},
        "desired_speed": 900,
        "boost_policy": "conserve",
        "jump_policy": "avoid",
        "guidance": "Human-readable description only.",
    }


def wait_for(predicate, *, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.005)
    raise AssertionError("Timed out waiting for test condition")


class TrackedBackend(MockBackend):
    def __init__(self, settings, path):
        super().__init__(settings, path)
        self._test_lock = threading.Lock()
        self._overrides = {}
        self.submitted = []
        self.closed = False

    def change(self, **changes):
        with self._test_lock:
            self._overrides.update(changes)

    def snapshot(self):
        state = super().snapshot()
        with self._test_lock:
            state.update(copy.deepcopy(self._overrides))
        return state

    def submit(self, controls, hold_ms):
        super().submit(controls, hold_ms)
        with self._test_lock:
            self.submitted.append((time.monotonic(), dict(controls), hold_ms))

    def close(self):
        super().close()
        self.closed = True


class FixedController:
    """Deliberately ignores packet safety so these tests exercise MotorWorker."""

    def __init__(self):
        self.calls = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def tick(self, state, current_plan, now):
        self.calls += 1
        return MotorOutput(
            dict(NEUTRAL, throttle=1),
            None,
            current_plan["skill"] if current_plan else "defend",
            "Test controller",
            {},
        )


class FakePlanner:
    def __init__(self, *, blocked=False):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.contexts = []
        if not blocked:
            self.release.set()

    def decide(self, observation):
        self.contexts.append(copy.deepcopy(observation))
        self.entered.set()
        if not self.release.wait(5):
            raise AssertionError("Test planner was not released")
        return PlannerReply(
            plan(),
            {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            response_id="test-response",
            resolved_model="test-strategist",
        )

    def close(self):
        self.closed.set()


def read_events(path):
    return [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]


def test_motor_keeps_running_while_planner_blocks_and_stop_cleans_up(tmp_path):
    settings = config(tmp_path, duration=5)
    backend = TrackedBackend(settings["rlbot"], tmp_path)
    planner = FakePlanner(blocked=True)
    result = []

    def run():
        result.append(
            run_episode(settings, synthetic=True, backend=backend, planner_client=planner)
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert planner.entered.wait(2)
        count = len(backend.submitted)
        # This tests independent scheduling, not a hard real-time performance guarantee.
        time.sleep(0.3)
        assert len(backend.submitted) - count >= 8
        assert len(planner.contexts) == 1
        assert thread.is_alive()
        path = next((tmp_path / "runs").iterdir())
        (path / "stop.request").write_text("stop")
        thread.join(timeout=2)
        assert not thread.is_alive(), "Stopping must not wait for the blocked API request"
        assert backend.closed
        assert backend.snapshot()["controls"] == NEUTRAL
        returned_path, summary = result[0]
        assert returned_path == path
        assert summary["status"] == "interrupted"
        assert summary["motor"]["ticks"] >= 8
        assert summary["motor"]["source_counts"]["fallback"] >= 8
        assert summary["planner"]["requests"] == 1
        assert summary["planner"]["completed"] == 0
        assert summary["accepted_actions"] == 0
        assert (path / "report.html").exists()
        assert read_events(path)[-1]["type"] == "finished"
    finally:
        planner.release.set()
        thread.join(timeout=2)
        assert planner.closed.wait(2)


def test_plans_cannot_cross_inactive_gameplay_generations(tmp_path):
    settings = config(tmp_path)
    backend = TrackedBackend(settings["rlbot"], tmp_path)
    controller = FixedController()
    worker = MotorWorker(backend, settings, controller=controller)
    backend.start()
    worker.start()
    try:
        initial = wait_for(lambda: worker.snapshot() if worker.snapshot()["ready"] else None)
        generation = initial["generation"]
        assert worker.set_plan(
            plan(), version=1, observed_at=time.monotonic(), generation=generation
        )
        wait_for(lambda: worker.snapshot()["sample"].get("policy_source") == "planner")
        backend.change(ready=False)
        wait_for(lambda: worker.snapshot()["sample"].get("policy_source") == "inactive")
        interrupted = worker.snapshot()
        assert interrupted["generation"] == generation + 1
        assert interrupted["plan"] is None and interrupted["plan_version"] is None
        assert controller.resets == 1
        assert not worker.set_plan(
            plan(), version=2, observed_at=time.monotonic(), generation=generation
        )
        backend.change(ready=True)
        wait_for(lambda: worker.snapshot()["ready"])
        assert not worker.set_plan(
            plan(), version=3, observed_at=time.monotonic(), generation=generation
        )
        assert worker.set_plan(
            plan(), version=4, observed_at=time.monotonic(), generation=generation + 1
        )
    finally:
        worker.close()
        backend.close()


def test_plan_age_is_measured_from_observation_and_expires_to_fallback(tmp_path):
    settings = config(tmp_path)
    backend = TrackedBackend(settings["rlbot"], tmp_path)
    worker = MotorWorker(backend, settings, controller=FixedController())
    backend.start()
    worker.start()
    try:
        wait_for(lambda: worker.snapshot()["ready"])
        generation = worker.snapshot()["generation"]
        assert not worker.set_plan(
            plan(), version=1, observed_at=time.monotonic() - 1.1, generation=generation
        )
        assert worker.set_plan(
            plan(), version=2, observed_at=time.monotonic() - 0.8, generation=generation
        )
        wait_for(lambda: worker.snapshot()["sample"].get("policy_source") == "planner")
        wait_for(lambda: worker.snapshot()["sample"].get("policy_source") == "fallback")
        assert worker.snapshot()["plan"] is None
        assert worker.snapshot()["plan_version"] is None
        assert worker.snapshot()["generation"] == generation
    finally:
        worker.close()
        backend.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"packet_age_s": 0.2},
        {"packet_age_s": -0.1},
        {"packet_age_s": float("nan")},
        {"packet_age_s": None},
        {"ready": False},
        {"ended": True},
    ],
)
def test_stale_or_inactive_packets_neutralize_without_calling_motor(tmp_path, changes):
    settings = config(tmp_path)
    backend = TrackedBackend(settings["rlbot"], tmp_path)
    controller = FixedController()
    worker = MotorWorker(backend, settings, controller=controller)
    backend.start()
    worker.start()
    try:
        wait_for(lambda: backend.snapshot()["controls"]["throttle"] == 1)
        backend.change(**changes)
        wait_for(lambda: worker.snapshot()["sample"].get("policy_source") == "inactive")
        sample = worker.snapshot()["sample"]
        calls = controller.calls
        wait_for(lambda: worker.snapshot()["sample"]["tick_count"] >= sample["tick_count"] + 2)
        assert controller.calls == calls
        assert backend.snapshot()["controls"] == NEUTRAL
        assert worker.snapshot()["ready"] is False
    finally:
        worker.close()
        backend.close()


def test_explicit_baseline_uses_no_model_factory_and_labels_assistance(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Explicit scripted baseline must not construct an API client")

    monkeypatch.setattr("rleval.planner.build_planner", forbidden)
    monkeypatch.setattr("rleval.runner.build_client", forbidden)
    settings = config(tmp_path, planner=False)
    backend = TrackedBackend(settings["rlbot"], tmp_path)
    path, summary = run_episode(settings, synthetic=True, backend=backend)
    assert summary["status"] == "time_limit"
    assert summary["synthetic_environment"] is True
    assert summary["evaluation_track"] == "strategy_assisted_controls"
    assert summary["motor"]["ticks"] > 0
    assert summary["motor"]["source_counts"]["scripted_baseline"] > 0
    assert summary["planner"]["requests"] == summary["planner"]["completed"] == 0
    assert summary["requests"] == summary["completed_decisions"] == 0
    assert all(value == 0 for value in summary["combined_usage_totals"].values())
    assert backend.closed and backend.snapshot()["controls"] == NEUTRAL
    assert "STRATEGY-ASSISTED CONTROLS" in (path / "report.html").read_text()


def test_assisted_reports_separate_motor_ticks_from_model_requests(tmp_path):
    settings = config(tmp_path)
    backend = TrackedBackend(settings["rlbot"], tmp_path)
    planner = FakePlanner()
    path, summary = run_episode(settings, synthetic=True, backend=backend, planner_client=planner)
    assert summary["status"] == "time_limit"
    assert summary["motor"]["ticks"] >= 4
    assert summary["motor"]["source_counts"]["planner"] > 0
    assert summary["planner"]["requests"] == 1
    assert summary["planner"]["completed"] == summary["planner"]["accepted"] == 1
    assert summary["planner"]["usage_totals"]["total_tokens"] == 120
    assert summary["combined_usage_totals"]["total_tokens"] == 120
    assert summary["usage_totals"]["total_tokens"] == 0
    assert summary["requests"] == summary["accepted_actions"] == 0
    assert summary["completed_decisions"] == 0
    assert planner.closed.wait(1)
    observation = planner.contexts[0]
    assert observation["control_mode"] == "local_motor"
    assert "motor_feedback" in observation and "recent_transitions" in observation
    events = read_events(path)
    accepted = [e for e in events if e["type"] == "planner_decision"]
    assert len(accepted) == 1 and accepted[0]["resolved_model"] == "test-strategist"
    assert all(event["type"] != "decision" for event in events)


def test_assisted_rejects_inflight_reply_after_gameplay_resumes(tmp_path):
    settings = config(tmp_path, duration=0.6)
    settings["planner"]["max_requests"] = 1
    backend = TrackedBackend(settings["rlbot"], tmp_path)
    planner = FakePlanner(blocked=True)
    result = []

    def run():
        result.append(
            run_episode(settings, synthetic=True, backend=backend, planner_client=planner)
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert planner.entered.wait(2)
        wait_for(lambda: backend.snapshot()["controls"] != NEUTRAL)
        backend.change(ready=False)
        wait_for(lambda: backend.snapshot()["controls"] == NEUTRAL)
        # The worker has processed inactivity; wait for its next active submission.
        submitted_before_resume = len(backend.submitted)
        backend.change(ready=True)
        wait_for(lambda: len(backend.submitted) > submitted_before_resume)
        planner.release.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        path, summary = result[0]
        assert summary["status"] == "time_limit"
        assert summary["planner"]["completed"] == 1
        assert summary["planner"]["accepted"] == 0
        assert summary["motor"]["source_counts"].get("planner", 0) == 0
        decisions = [e for e in read_events(path) if e["type"] == "planner_decision"]
        assert len(decisions) == 1 and decisions[0]["accepted"] is False
        assert "gameplay changed" in decisions[0]["error"]
    finally:
        planner.release.set()
        thread.join(timeout=2)
        assert planner.closed.wait(2)
