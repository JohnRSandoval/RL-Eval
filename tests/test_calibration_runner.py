"""Calibration orchestration without inventing physics or contacting models."""

import copy
import json
import threading

import pytest

from rleval.actions import NEUTRAL
from rleval.backend import MockBackend
from rleval.calibration import run_calibration
from rleval.calibration_tasks import Scenario
from rleval.config import load_config
from rleval.providers import ModelReply, ProviderError


def trial(name="brake", *, duration=0.08, kind="brake", velocity=1000):
    return Scenario(
        name,
        kind,
        {"x": 0.0, "y": 0.0, "z": 17.0},
        {"x": float(velocity), "y": 0.0, "z": 0.0},
        0.0,
        {"x": 1000.0, "y": 0.0, "z": 17.0} if kind != "brake" else None,
        0.0 if kind == "brake" else 500.0,
        duration,
        "Stop" if kind == "brake" else "Drive to the waypoint and stop",
    )


def configuration(tmp_path):
    return load_config(
        overrides={
            "recording": {"output_dir": str(tmp_path), "live_window": False, "telemetry_hz": 30},
            "control": {"max_decision_hz": 30},
            "model": {"decision_hold_ms": 300},
            "episode": {"max_observation_age_s": 0.1},
        }
    )


def reply(**controls):
    return ModelReply(
        {"controls": dict(NEUTRAL, **controls), "hold_ms": 300, "memory": ""},
        {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10},
        decision={"choice": "test controls", "catalog_version": "ground-axes-v1"},
    )


class Backend(MockBackend):
    def __init__(self, config, path):
        super().__init__(config, path)
        self.resets = []
        self.submissions = []
        self.was_closed = False

    def reset_trial(self, position, velocity, yaw, boost=33):
        self.resets.append((copy.deepcopy(position), copy.deepcopy(velocity), yaw))
        return super().reset_trial(position, velocity, yaw, boost)

    def submit(self, controls, hold_ms):
        self.submissions.append((len(self.resets), dict(controls)))
        super().submit(controls, hold_ms)

    def close(self):
        self.was_closed = True
        super().close()


class Client:
    def __init__(self, callback=None):
        self.callback = callback
        self.observations = []
        self.closed = threading.Event()

    def decide(self, frames, context):
        self.observations.append((frames, copy.deepcopy(context)))
        return self.callback() if self.callback else reply()

    def close(self):
        self.closed.set()


def events(path):
    return [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]


def test_every_policy_scenario_repeat_gets_a_reset_and_explicit_report_label(tmp_path):
    config = configuration(tmp_path)
    backend = Backend(config["rlbot"], tmp_path)
    clients = []

    def factory():
        client = Client()
        clients.append(client)
        return client

    path, summary = run_calibration(
        config,
        repeats=2,
        scenarios=[trial("first"), trial("second")],
        backend=backend,
        client_factory=factory,
    )
    assert summary["status"] == "completed"
    assert len(backend.resets) == len(summary["trials"]) == 8
    assert summary["policies"] == {
        "reference": {"passed": 0, "total": 4},
        "model": {"passed": 0, "total": 4},
    }
    assert [(t["repeat"], t["name"], t["policy"]) for t in summary["trials"]] == [
        (repeat, name, policy)
        for repeat in [1, 2]
        for name in ["first", "second"]
        for policy in ["reference", "model"]
    ]
    assert backend.was_closed and backend.snapshot()["controls"] == NEUTRAL
    assert all(client.closed.wait(1) for client in clients)
    assert json.loads((path / "summary.json").read_text()) == summary
    page = (path / "report.html").read_text(encoding="utf-8")
    assert "reference" in page and "model" in page and "not a latency-matched" in page
    assert (path / "replay.html").exists()
    requests = [e for e in events(path) if e["type"] == "request"]
    assert {e["policy"] for e in requests} == {"reference", "model"}
    for client in clients:
        for frames, context in client.observations:
            assert frames == []
            assert context["observation_representation"] == "compact"
            assert "motor_observation" in context and "game_state" not in context


def test_model_client_setup_finishes_before_each_model_trial_reset(tmp_path):
    order, clients = [], []

    class OrderedBackend(Backend):
        def reset_trial(self, position, velocity, yaw, boost=33):
            order.append(("reset", len(self.resets) + 1))
            return super().reset_trial(position, velocity, yaw, boost)

    def factory():
        client = Client()
        clients.append(client)
        order.append(("client_ready", len(clients)))
        return client

    config = configuration(tmp_path)
    backend = OrderedBackend(config["rlbot"], tmp_path)
    _, summary = run_calibration(
        config,
        repeats=2,
        scenarios=[trial("first", duration=0.02), trial("second", duration=0.02)],
        backend=backend,
        client_factory=factory,
    )
    assert summary["status"] == "completed"
    assert len(clients) == 4
    model_number = 0
    for reset_number, result in enumerate(summary["trials"], start=1):
        if result["policy"] == "model":
            model_number += 1
            assert order.index(("client_ready", model_number)) < order.index(
                ("reset", reset_number)
            )
    assert all(client.closed.wait(1) for client in clients)


def test_mock_movement_is_not_fabricated_into_a_waypoint_success(tmp_path):
    config = configuration(tmp_path)
    backend = Backend(config["rlbot"], tmp_path)
    client = Client(lambda: reply(throttle=1.0))
    _, summary = run_calibration(
        config,
        policies=("model",),
        scenarios=[trial(kind="waypoint", velocity=0, duration=0.12)],
        backend=backend,
        client_factory=lambda: client,
    )
    result = summary["trials"][0]
    assert result["accepted_actions"] > 0
    assert result["metrics"]["success"] is False
    assert result["metrics"]["initial_distance_uu"] == 1000
    assert result["metrics"]["final_distance_uu"] == 1000


def test_stationary_smoke_passes_only_after_scoring_dwell(tmp_path):
    config = configuration(tmp_path)
    backend = Backend(config["rlbot"], tmp_path)
    _, summary = run_calibration(
        config,
        policies=("model",),
        scenarios=[trial(velocity=0, duration=0.4)],
        backend=backend,
        client_factory=Client,
    )
    result = summary["trials"][0]
    assert result["metrics"]["success"]
    assert result["metrics"]["success_time_s"] >= 0.25
    assert result["initial_state"]["synthetic"] is True


def test_repeated_provider_failures_neutralize_close_and_keep_report(tmp_path):
    def fail():
        raise ProviderError("Synthetic provider failure")

    config = configuration(tmp_path)
    backend = Backend(config["rlbot"], tmp_path)
    client = Client(fail)
    path, summary = run_calibration(
        config,
        policies=("model",),
        scenarios=[trial(duration=1)],
        backend=backend,
        client_factory=lambda: client,
    )
    assert summary["status"] == "failed"
    assert "Repeated provider failures" in summary["error"]
    assert not backend.submissions
    assert backend.was_closed and backend.snapshot()["controls"] == NEUTRAL
    assert client.closed.wait(1)
    assert (path / "report.html").exists()
    assert len([e for e in events(path) if e["type"] == "decision"]) == 5


def test_response_from_finished_trial_cannot_control_the_next_trial(tmp_path):
    first_entered, release_first = threading.Event(), threading.Event()
    first_finished = threading.Event()

    def first_response():
        first_entered.set()
        assert release_first.wait(3)
        first_finished.set()
        return reply(throttle=1.0, steer=-0.5)

    def second_response():
        assert first_entered.is_set()
        release_first.set()
        assert first_finished.wait(1)
        return reply(throttle=1.0, steer=0.5)

    clients = [Client(first_response), Client(second_response)]
    remaining = iter(clients)
    config = configuration(tmp_path)
    backend = Backend(config["rlbot"], tmp_path)
    try:
        _, summary = run_calibration(
            config,
            policies=("model",),
            scenarios=[trial("old", duration=0.08), trial("new", duration=0.15)],
            backend=backend,
            client_factory=lambda: next(remaining),
        )
    finally:
        release_first.set()
        assert all(client.closed.wait(1) for client in clients)
    assert summary["status"] == "completed"
    assert summary["trials"][0]["accepted_actions"] == 0
    assert summary["trials"][1]["accepted_actions"] > 0
    assert backend.submissions
    assert all(reset == 2 and controls["steer"] == 0.5 for reset, controls in backend.submissions)


def test_motor_calibration_uses_local_worker_without_api_and_reports_separate_controls(
    tmp_path, monkeypatch
):
    from rleval import assisted

    def unexpected_api(*args, **kwargs):
        pytest.fail("Local motor calibration must not construct any API client")

    workers = []
    actual_worker = assisted.MotorWorker

    class ObservedWorker(actual_worker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.closed_by_runner = False
            workers.append(self)

        def close(self):
            self.closed_by_runner = True
            super().close()

    monkeypatch.setattr(assisted, "MotorWorker", ObservedWorker)
    monkeypatch.setattr("rleval.calibration.build_client", unexpected_api)
    monkeypatch.setattr("rleval.planner.build_planner", unexpected_api)
    config = configuration(tmp_path)
    backend = Backend(config["rlbot"], tmp_path)
    path, summary = run_calibration(
        config,
        policies=("motor",),
        scenarios=[
            trial("stationary", velocity=0, duration=0.4),
            trial("unreached", kind="waypoint", velocity=0, duration=0.12),
        ],
        backend=backend,
        client_factory=unexpected_api,
    )
    assert summary["status"] == "completed"
    assert summary["policies"] == {"motor": {"passed": 1, "total": 2}}
    assert summary["model"] == "local-motor-v1"
    assert summary["action_interface"] == "continuous_motor"
    metadata = json.loads((path / "metadata.json").read_text())
    assert metadata["action_format"] == "local_motor_controller"
    assert metadata["observation_representation"] == "full"
    assert metadata["config"]["control"]["mode"] == "local_motor"
    assert metadata["config"]["planner"]["enabled"] is False
    assert len(workers) == len(backend.resets) == 2
    assert all(worker.closed_by_runner and not worker._thread.is_alive() for worker in workers)
    assert backend.was_closed and backend.snapshot()["controls"] == NEUTRAL
    recorded = events(path)
    assert not any(
        e["type"] in {"request", "decision", "planner_request", "planner_decision"}
        for e in recorded
    )
    motor_samples = [e for e in recorded if e["type"] == "motor_sample"]
    assert motor_samples
    assert all(e["policy_source"] == "calibration" for e in motor_samples)
    for result in summary["trials"]:
        assert result["policy"] == "motor" and result["requests"] == 0
        assert result["accepted_actions"] == result["motor"]["active_ticks"] > 0
        assert result["motor"]["target_hz"] == config["control"]["motor_hz"]
        assert result["motor"]["errors"] == []
    aggregate = summary["mixed_policy_pipeline"]["motor"]
    assert aggregate["active_ticks"] == sum(t["motor"]["active_ticks"] for t in summary["trials"])
    assert len([event for event in recorded if event["type"] == "motor_summary"]) == 1


def test_motor_calibration_rejects_mixing_with_model_action_space(tmp_path):
    with pytest.raises(ValueError, match="separately"):
        run_calibration(configuration(tmp_path), policies=("motor", "model"))
