"""Explicitly assisted strategy evaluation with a separate local motor loop."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from queue import Empty

from .backend import MockBackend, RLBotBackend
from .feedback import FeedbackTracker
from .observation import build_game_state
from .recording import Recorder
from .runner import DecisionWorker


class MotorWorker:
    """Local control never waits for an API response or a report write."""

    def __init__(self, backend, config, *, baseline=False, controller=None, initial_plan=None):
        from .motor import LocalMotorController

        self.backend = backend
        self.controller = controller or LocalMotorController()
        self.baseline = baseline
        self.hz = config["control"]["motor_hz"]
        self.max_age = config["planner"]["max_age_s"]
        self.packet_age = config["control"]["motor_max_packet_age_s"]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="local-motor", daemon=True)
        self._generation = 0
        self._ready = False
        if initial_plan is not None:
            from .planner import parse_motor_plan

            initial_plan = parse_motor_plan(initial_plan)
        self._plan = initial_plan
        self._plan_at = time.monotonic()
        self._version = 0 if initial_plan else None
        self._sample = {}
        self._error = None
        self._ticks = self._active_ticks = 0
        self._compute_total = self._compute_max = 0.0
        self._sources = Counter()
        self._started = self._ended = None

    def start(self):
        self._started = time.monotonic()
        self._thread.start()

    def set_plan(self, plan, *, version, observed_at, generation):
        from .planner import parse_motor_plan

        plan = parse_motor_plan(plan)
        with self._lock:
            if not self._ready or generation != self._generation:
                return False
            if time.monotonic() - observed_at > self.max_age:
                return False
            self._plan = copy.deepcopy(plan)
            self._plan_at = observed_at
            self._version = version
        return True

    def snapshot(self):
        with self._lock:
            return {
                "generation": self._generation,
                "ready": self._ready,
                "plan": copy.deepcopy(self._plan),
                "plan_version": self._version,
                "sample": copy.deepcopy(self._sample),
                "error": self._error,
            }

    def _run(self):
        from .motor import scripted_plan

        deadline = time.monotonic()
        try:
            while not self._stop.is_set():
                began = time.monotonic()
                state = self.backend.snapshot()
                if state.get("error"):
                    raise RuntimeError(state["error"])
                age = state.get("packet_age_s")
                ready = (
                    state.get("ready") is True
                    and not state.get("ended")
                    and type(age) in (int, float)
                    and 0 <= age <= self.packet_age
                )
                with self._lock:
                    if self._ready and not ready:
                        self._generation += 1
                        self._plan = self._version = None
                        self.controller.reset()
                    self._ready = ready
                    if self._plan and began - self._plan_at > self.max_age:
                        self._plan = self._version = None
                    plan, version = copy.deepcopy(self._plan), self._version
                if ready:
                    source = ("calibration" if version == 0 else "planner") if plan else "fallback"
                    if self.baseline:
                        plan, source = scripted_plan(state), "scripted_baseline"
                    output = self.controller.tick(state, plan, began)
                    self.backend.submit(output.controls, 100)
                    sample = {
                        "controls": output.controls,
                        "target": output.target,
                        "skill": output.skill,
                        "status": output.status,
                        "metrics": output.metrics,
                        "plan_version": version,
                        "policy_source": source,
                    }
                else:
                    self.backend.neutral()
                    source = "inactive"
                    sample = {
                        "controls": {},
                        "target": None,
                        "skill": "stop",
                        "status": "Waiting for fresh active gameplay",
                        "metrics": {},
                        "plan_version": None,
                        "policy_source": source,
                    }
                compute_ms = (time.monotonic() - began) * 1000
                with self._lock:
                    self._ticks += 1
                    self._active_ticks += int(ready)
                    self._compute_total += compute_ms
                    self._compute_max = max(self._compute_max, compute_ms)
                    self._sources[source] += 1
                    self._sample = dict(sample, tick_count=self._ticks)
                deadline += 1 / self.hz
                delay = deadline - time.monotonic()
                if delay < 0:
                    deadline = time.monotonic()
                self._stop.wait(max(0, delay))
        except Exception as exc:  # noqa: BLE001 -- stop the owned motor loop on any failure
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self.backend.neutral()
            self._ended = time.monotonic()

    def close(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError("Local motor worker did not stop")
        self.backend.neutral()

    def summary(self):
        elapsed = (self._ended or time.monotonic()) - (self._started or time.monotonic())
        with self._lock:
            return {
                "ticks": self._ticks,
                "active_ticks": self._active_ticks,
                "worker_elapsed_s": round(max(0, elapsed), 6),
                "target_hz": self.hz,
                "achieved_hz": round(self._ticks / elapsed, 3) if elapsed > 0 else 0,
                "mean_compute_ms": round(self._compute_total / self._ticks, 4)
                if self._ticks
                else 0,
                "max_compute_ms": round(self._compute_max, 4),
                "source_counts": dict(self._sources),
                "errors": [self._error] if self._error else [],
            }


def run_assisted_episode(config, *, synthetic=False, backend=None, planner_client=None):
    """Same local mechanics for model tactics and the explicit scripted baseline."""
    from .planner import build_planner

    recorder = Recorder(config["recording"]["output_dir"], config, synthetic=synthetic)
    print(f"Assisted run: {recorder.path}", flush=True)
    backend = backend or (MockBackend if synthetic else RLBotBackend)(
        config["rlbot"], recorder.path
    )
    enabled = config["planner"]["enabled"]
    motor = MotorWorker(backend, config, baseline=not enabled)
    worker = None
    started = time.monotonic()
    final, status, fatal = {}, "completed", None
    pending, requests, errors = False, 0, 0
    next_plan = next_telemetry = next_motor_sample = 0.0
    last_version = None
    last_touch = 0
    feedback = FeedbackTracker()
    try:
        if enabled:
            planner_client = planner_client or build_planner(
                dict(config["planner"], control_mode="local_motor")
            )
        if config["recording"].get("live_window"):
            with (recorder.path / "live-window.log").open("wb") as log:
                subprocess.Popen(
                    [sys.executable, "-m", "rleval.live_view", str(recorder.path)],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
        backend.start()
        started = time.monotonic()
        motor.start()
        if enabled:
            worker = DecisionWorker(planner_client, state_only=True)
        print(
            "Local motor running; stop from the live window or create stop.request in the run directory.",
            flush=True,
        )
        while True:
            now = time.monotonic()
            elapsed = now - started
            final = backend.snapshot()
            control = motor.snapshot()
            if final.get("error") or control["error"]:
                raise RuntimeError(final.get("error") or control["error"])
            if (recorder.path / "stop.request").exists():
                status = "interrupted"
                break
            if final.get("ended"):
                status = "match_ended"
                break
            if elapsed >= config["episode"]["duration_s"]:
                status = "time_limit"
                break
            feedback.observe(elapsed, final)
            if now >= next_telemetry:
                recorder.event("telemetry", elapsed, state=final)
                next_telemetry = now + 1 / config["recording"]["telemetry_hz"]
            if now >= next_motor_sample and control["sample"]:
                recorder.event("motor_sample", elapsed, **control["sample"])
                next_motor_sample = now + 0.1
            if last_version is not None and control["plan_version"] is None:
                recorder.event(
                    "planner_invalidated", elapsed, reason="Plan expired or gameplay interrupted"
                )
                next_plan = 0
            last_version = control["plan_version"]
            if worker:
                try:
                    request, reply, error, latency = worker.results.get_nowait()
                except Empty:
                    pass
                else:
                    pending = False
                    fields = {
                        "number": request["number"],
                        "latency_s": round(latency, 6),
                        "observation_age_s": round(now - request["observed_monotonic"], 6),
                        "accepted": False,
                        "usage": reply.usage if reply else {},
                        "response_id": reply.response_id if reply else None,
                        "resolved_model": reply.resolved_model if reply else None,
                    }
                    if error:
                        fields["error"] = error
                        errors += 1
                    elif motor.set_plan(
                        reply.plan,
                        version=request["number"],
                        observed_at=request["observed_monotonic"],
                        generation=request["generation"],
                    ):
                        fields.update(accepted=True, plan=reply.plan)
                        last_version = request["number"]
                        print(
                            f"  Planner {request['number']}: {reply.plan['skill']} — {reply.plan['objective']}",
                            flush=True,
                        )
                    else:
                        fields["error"] = "Planner observation expired or gameplay changed"
                        next_plan = 0
                    recorder.event("planner_decision", elapsed, **fields)
                    if errors >= config["episode"]["max_errors"]:
                        raise RuntimeError("Maximum planner failures reached")
                touch = final.get("touches", 0)
                if touch != last_touch:
                    next_plan = min(next_plan, now)
                    last_touch = touch
                if (
                    not pending
                    and now >= next_plan
                    and control["ready"]
                    and requests < config["planner"]["max_requests"]
                ):
                    context = {
                        "observation_mode": "state",
                        "task": config["episode"]["task"],
                        "team": "blue",
                        "elapsed_s": round(elapsed, 3),
                        "game_state": build_game_state(final),
                        "previous_plan": control["plan"],
                        "motor_feedback": control["sample"],
                        "recent_transitions": feedback.recent(),
                        "control_mode": "local_motor",
                    }
                    requests += 1
                    recorder.event("planner_request", elapsed, number=requests, context=context)
                    worker.requests.put_nowait(
                        {
                            "number": requests,
                            "context": context,
                            "observed_monotonic": now - float(final.get("packet_age_s") or 0),
                            "generation": control["generation"],
                        }
                    )
                    pending = True
                    next_plan = now + config["planner"]["interval_s"]
            time.sleep(0.02)
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as exc:  # noqa: BLE001 -- always neutralize and retain run evidence
        status, fatal = "failed", f"{type(exc).__name__}: {exc}"
        print(fatal, flush=True)
    finally:
        elapsed = time.monotonic() - started
        for operation in (motor.close, backend.snapshot, backend.close):
            try:
                result = operation()
                if isinstance(result, dict):
                    final = result
            except Exception as exc:  # noqa: BLE001 -- remaining cleanup must still run
                status, fatal = "failed", fatal or f"Cleanup failed: {type(exc).__name__}"
        recorder.event("motor_summary", elapsed, **motor.summary())
        if worker:
            worker.close()
        elif planner_client:
            planner_client.close()
        summary = recorder.finish(status, elapsed, final, fatal)
    print(f"Finished: {status}. Report: {recorder.path / 'report.html'}", flush=True)
    return recorder.path, summary
