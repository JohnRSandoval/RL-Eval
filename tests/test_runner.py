"""Exercise orchestration, replay evidence, and input shutdown without a game or API."""

import copy
import io
import json
import threading
import time

import pytest
from PIL import Image

from rleval.actions import NEUTRAL
from rleval.backend import MockBackend
from rleval.capture import SyntheticCapture
from rleval.config import load_config
from rleval.providers import ModelReply, ProviderError
from rleval.runner import run_episode


def action_reply():
    return ModelReply(
        {
            "controls": dict(NEUTRAL, throttle=1.0, boost=True),
            "hold_ms": 50,
            "memory": "My own visual observation.",
        },
        {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        "test-response",
    )


class ObservedBackend(MockBackend):
    """Use the real lease mechanism while observing externally visible calls."""

    def __init__(self, config, run_dir):
        super().__init__(config, run_dir)
        self.operations = []
        self.submissions = []

    def start(self):
        self.operations.append("start")
        super().start()

    def submit(self, controls, hold_ms):
        self.operations.append("submit")
        self.submissions.append((dict(controls), hold_ms))
        super().submit(controls, hold_ms)

    def neutral(self):
        self.operations.append("neutral")
        super().neutral()

    def close(self):
        self.operations.append("close")
        super().close()

    def snapshot(self):
        return dict(
            super().snapshot(),
            privileged_sentinel="HIDDEN-STATE-MUST-NOT-REACH-THE-MODEL",
            ball_position={"x": 919191, "y": 929292, "z": 939393},
        )


class ObservedClient:
    def __init__(self, callback=None):
        self.callback = callback
        self.observations = []
        self.closed = threading.Event()

    def decide(self, frames, context):
        self.observations.append((list(frames), copy.deepcopy(context)))
        if self.callback is not None:
            return self.callback(len(self.observations))
        return action_reply()

    def close(self):
        self.closed.set()


def config_for(tmp_path, **episode):
    return load_config(
        overrides={
            "episode": {
                "duration_s": 3.0,
                "max_decisions": 2,
                "max_errors": 1,
                "decision_interval_s": 0.05,
                **episode,
            },
            "capture": {"max_width": 320, "sample_hz": 30},
            "recording": {"output_dir": str(tmp_path), "telemetry_hz": 30},
        }
    )


def read_events(path):
    return [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]


def test_synthetic_run_finishes_with_replayable_pixels_and_action_evidence(tmp_path):
    config = config_for(tmp_path)
    backend = ObservedBackend(config["rlbot"], tmp_path)
    client = ObservedClient()
    path, summary = run_episode(config, synthetic=True, backend=backend, client=client)

    assert summary["status"] == "decision_limit"
    assert summary["accepted_actions"] == 2
    assert summary["completed_decisions"] == summary["requests"] == 2
    assert summary["synthetic_environment"] is True
    assert len(backend.submissions) == 2
    assert backend.snapshot()["controls"] == NEUTRAL
    assert backend.operations.index("neutral") < backend.operations.index("close")
    assert client.closed.wait(timeout=1)

    events = read_events(path)
    requests = [event for event in events if event["type"] == "request"]
    decisions = [event for event in events if event["type"] == "decision"]
    assert events[-1]["type"] == "finished"
    assert [event["number"] for event in requests] == [1, 2]
    assert all(event["accepted"] for event in decisions)
    assert all(event["usage"]["total_tokens"] == 30 for event in decisions)
    for request, (frames, _) in zip(requests, client.observations, strict=True):
        assert [(path / frame_path).read_bytes() for frame_path in request["frames"]] == frames
        for frame in frames:
            with Image.open(io.BytesIO(frame)) as image:
                assert image.format == "JPEG"
                assert image.width == 320
    assert json.loads((path / "summary.json").read_text()) == summary
    metadata = json.loads((path / "metadata.json").read_text())
    assert metadata["observation_mode"] == "pixels"
    assert len(metadata["source_sha256"]) == 64
    assert "SYNTHETIC SMOKE TEST" in (path / "report.html").read_text(encoding="utf-8")


def test_privileged_scoring_state_stays_out_of_every_model_observation(tmp_path):
    config = config_for(tmp_path)
    backend = ObservedBackend(config["rlbot"], tmp_path)
    client = ObservedClient()
    path, summary = run_episode(config, synthetic=True, backend=backend, client=client)

    # The state is available to the scorer and recording, so this checks the
    # observation boundary rather than merely constructing an empty snapshot.
    assert summary["final_telemetry"]["ball_position"]["x"] == 919191
    telemetry = [event for event in read_events(path) if event["type"] == "telemetry"]
    assert telemetry[0]["state"]["privileged_sentinel"].startswith("HIDDEN-STATE")
    assert len(client.observations) == 2
    for frames, context in client.observations:
        assert frames
        assert set(context) <= {
            "observation_mode",
            "task",
            "team",
            "elapsed_s",
            "remaining_s",
            "frame_times_s",
            "previous_actions",
            "memory",
            "timing",
        }
        encoded = json.dumps(context)
        assert "HIDDEN-STATE" not in encoded
        assert "919191" not in encoded
        assert "ball_position" not in encoded
        assert "scores" not in context
    assert client.observations[0][1]["memory"] == ""
    assert client.observations[1][1]["memory"] == "My own visual observation."
    assert len(client.observations[1][1]["previous_actions"]) == 1


@pytest.mark.parametrize(
    "failure",
    [
        ProviderError("Provider request timed out; no automatic retry was made"),
        ProviderError("Provider returned HTTP 401"),
    ],
)
def test_provider_failure_after_an_action_releases_inputs_and_records_failure(tmp_path, failure):
    def response(number):
        if number == 1:
            return action_reply()
        raise failure

    config = config_for(tmp_path)
    backend = ObservedBackend(config["rlbot"], tmp_path)
    client = ObservedClient(response)
    path, summary = run_episode(config, synthetic=True, backend=backend, client=client)

    assert summary["status"] == "failed"
    assert summary["accepted_actions"] == 1
    assert summary["rejected_or_failed_decisions"] == 1
    assert len(backend.submissions) == 1
    assert backend.snapshot()["controls"] == NEUTRAL
    assert backend.operations.index("neutral") < backend.operations.index("close")
    assert client.closed.wait(timeout=1)
    rejected = [
        event
        for event in read_events(path)
        if event["type"] == "decision" and not event["accepted"]
    ]
    assert len(rejected) == 1
    assert str(failure) in rejected[0]["error"]
    assert (path / "report.html").is_file()


def test_response_from_an_old_frame_never_reaches_the_controller(tmp_path):
    def slow_response(_):
        time.sleep(0.2)
        return action_reply()

    config = config_for(tmp_path, max_observation_age_s=0.1, max_decisions=1)
    backend = ObservedBackend(config["rlbot"], tmp_path)
    client = ObservedClient(slow_response)
    path, summary = run_episode(config, synthetic=True, backend=backend, client=client)

    assert summary["accepted_actions"] == 0
    assert summary["rejected_or_failed_decisions"] == 1
    assert not backend.submissions
    assert backend.snapshot()["controls"] == NEUTRAL
    decision = next(event for event in read_events(path) if event["type"] == "decision")
    assert decision["observation_age_s"] > 0.1
    assert "Stale response" in decision["error"]
    assert decision["usage"]["total_tokens"] == 30  # rejected inference is still counted


def test_episode_deadline_releases_controls_without_waiting_for_model_return(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def blocked_response(_):
        entered.set()
        assert release.wait(timeout=5), "Test did not release its simulated provider"
        return action_reply()

    config = config_for(tmp_path, duration_s=0.1)
    backend = ObservedBackend(config["rlbot"], tmp_path)
    client = ObservedClient(blocked_response)
    try:
        _, summary = run_episode(config, synthetic=True, backend=backend, client=client)
        assert entered.is_set()
        assert summary["status"] == "time_limit"
        assert summary["requests"] == 1
        assert summary["completed_decisions"] == 0
        assert not backend.submissions
        assert "close" in backend.operations
        assert backend.snapshot()["controls"] == NEUTRAL
        assert not client.closed.is_set()  # Inference is still blocked; shutdown did not wait.
    finally:
        release.set()
        assert client.closed.wait(timeout=1)
    assert not backend.submissions  # Late result cannot act after the episode ends.


def test_capture_failure_during_an_active_action_neutralizes_before_shutdown(tmp_path):
    class FailedCapture(SyntheticCapture):
        def grab(self):
            if backend.submissions:
                raise RuntimeError("Synthetic capture disconnected")
            return super().grab()

    config = config_for(tmp_path)
    backend = ObservedBackend(config["rlbot"], tmp_path)
    capture = FailedCapture(config["capture"])
    client = ObservedClient()
    _, summary = run_episode(
        config, synthetic=True, backend=backend, capture=capture, client=client
    )

    assert summary["status"] == "failed"
    assert "capture disconnected" in summary["error"]
    assert len(backend.submissions) == 1
    assert backend.snapshot()["controls"] == NEUTRAL
    assert backend.operations.index("neutral") < backend.operations.index("close")
    assert client.closed.wait(timeout=1)


def test_slow_capture_cannot_enqueue_a_request_after_deadline(tmp_path):
    class SlowCapture(SyntheticCapture):
        def grab(self):
            time.sleep(0.15)
            return super().grab()

    config = config_for(tmp_path, duration_s=0.1)
    backend = ObservedBackend(config["rlbot"], tmp_path)
    client = ObservedClient()
    _, summary = run_episode(
        config,
        synthetic=True,
        backend=backend,
        capture=SlowCapture(config["capture"]),
        client=client,
    )
    assert summary["status"] == "time_limit"
    assert summary["requests"] == 0
    assert not backend.submissions


def test_capture_cleanup_failure_still_records_report(tmp_path):
    class CleanupFailure(SyntheticCapture):
        def close(self):
            raise RuntimeError("cleanup failure")

    config = config_for(tmp_path, max_decisions=1)
    client = ObservedClient()
    path, summary = run_episode(
        config, synthetic=True, capture=CleanupFailure(config["capture"]), client=client
    )
    assert summary["status"] == "decision_limit"
    assert (path / "report.html").exists()
    assert any(e["type"] == "cleanup_error" for e in read_events(path))
    assert client.closed.wait(timeout=1)


def test_state_mode_supplies_allowlisted_measurements_without_screen_capture(tmp_path):
    class NeverCapture:
        def grab(self):
            raise AssertionError("State-only mode must not capture the desktop")

        def close(self):
            pass

    config = config_for(tmp_path, max_decisions=2)
    config["observation"]["mode"] = "state"
    config["rlbot"]["require_camera"] = False
    client = ObservedClient()
    backend = ObservedBackend(config["rlbot"], tmp_path)
    path, summary = run_episode(
        config, synthetic=True, backend=backend, capture=NeverCapture(), client=client
    )
    assert summary["accepted_actions"] == 2
    assert summary["observation_mode"] == "state"
    assert not list((path / "frames").iterdir())
    for frames, context in client.observations:
        assert frames == []
        assert context["observation_mode"] == "state"
        state = context["game_state"]
        assert state["ball"]["position"]["x"] == 919191
        assert state["player"]["forward"]["y"] == pytest.approx(1)
        assert "relative_ball" in state
        assert "HIDDEN-STATE-MUST-NOT-REACH-THE-MODEL" not in json.dumps(context)


def test_continuous_schedule_replaces_action_before_its_lease_expires(tmp_path):
    def long_hold(_):
        reply = action_reply()
        reply.action["hold_ms"] = 1000
        return reply

    config = config_for(tmp_path, max_decisions=2)
    config["observation"]["mode"] = "state"
    config["control"].update(scheduling="continuous", max_decision_hz=10)
    client = ObservedClient(long_hold)
    path, summary = run_episode(config, synthetic=True, client=client)
    assert summary["accepted_actions"] == 2
    requests = [e for e in read_events(path) if e["type"] == "request"]
    delta = requests[1]["elapsed_s"] - requests[0]["elapsed_s"]
    assert 0.09 <= delta < 0.9
    assert (
        requests[1]["context"]["game_state"]["frame_num"]
        > requests[0]["context"]["game_state"]["frame_num"]
    )


def test_combined_mode_keeps_frames_and_adds_explicit_state(tmp_path):
    config = config_for(tmp_path, max_decisions=1)
    config["observation"]["mode"] = "pixels_and_state"
    client = ObservedClient()
    _, summary = run_episode(config, synthetic=True, client=client)
    assert summary["accepted_actions"] == 1
    frames, context = client.observations[0]
    assert frames and context["observation_mode"] == "pixels_and_state"
    assert context["game_state"]["ball"]["position"]["z"] == 93


@pytest.mark.parametrize("mode", ["pixels", "pixels_and_state", "state"])
def test_response_spanning_countdown_is_discarded_after_gameplay_resumes(tmp_path, mode):
    requested, resumed = threading.Event(), threading.Event()

    class InterruptedBackend(ObservedBackend):
        interruption_seen = False

        def snapshot(self):
            state = super().snapshot()
            if requested.is_set() and not self.interruption_seen:
                self.interruption_seen = True
                state.update(ready=False, active=False, phase="Countdown")
            elif self.interruption_seen:
                resumed.set()
            return state

    def response(number):
        if number == 1:
            requested.set()
            assert resumed.wait(timeout=2), "Runner did not observe countdown and resume"
        return action_reply()

    config = config_for(tmp_path, max_decisions=2, max_errors=1)
    config["observation"]["mode"] = mode
    backend = InterruptedBackend(config["rlbot"], tmp_path)
    client = ObservedClient(response)
    path, summary = run_episode(config, synthetic=True, backend=backend, client=client)

    assert summary["status"] == "decision_limit"  # Transition discard did not hit max_errors.
    assert summary["accepted_actions"] == 1
    assert len(backend.submissions) == 1
    decisions = [event for event in read_events(path) if event["type"] == "decision"]
    assert [event["accepted"] for event in decisions] == [False, True]
    assert "Gameplay changed" in decisions[0]["error"]
    assert client.observations[1][1]["previous_actions"] == []
    assert client.observations[1][1]["memory"] == ""
    if mode != "state":
        first_times = client.observations[0][1]["frame_times_s"]
        second_times = client.observations[1][1]["frame_times_s"]
        assert min(second_times) > max(first_times)
