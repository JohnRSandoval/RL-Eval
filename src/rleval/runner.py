"""Real-time orchestration with explicitly labeled visual and state-assisted observations."""

import os
import subprocess
import sys
import threading
import time
from collections import deque
from queue import Empty, Queue

from .actions import parse_action
from .capture import SyntheticCapture, WindowCapture
from .providers import build_client
from .recording import Recorder


class DecisionWorker:
    """One outstanding request; network work never runs in the controller loop."""

    def __init__(self, client, *, state_only=False):
        self.client = client
        self.state_only = state_only
        self.requests = Queue(maxsize=1)
        self.results = Queue()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="model-request")
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                try:
                    request = self.requests.get(timeout=0.1)
                except Empty:
                    continue
                if self.stop.is_set():
                    return
                started = time.monotonic()
                try:
                    reply = (
                        self.client.decide(request["context"])
                        if self.state_only
                        else self.client.decide(request["frames"], request["context"])
                    )
                    error = None
                except Exception as exc:  # noqa: BLE001 -- contain provider failures at the worker boundary
                    reply, error = None, f"{type(exc).__name__}: {exc}"
                self.results.put((request, reply, error, time.monotonic() - started))
        finally:
            self.client.close()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=0.25)


def run_episode(
    config: dict, *, synthetic=False, backend=None, capture=None, client=None, planner_client=None
):
    if config["control"].get("mode") == "local_motor":
        from .assisted import run_assisted_episode

        return run_assisted_episode(
            config, synthetic=synthetic, backend=backend, planner_client=planner_client
        )
    from .backend import MockBackend, RLBotBackend
    from .feedback import FeedbackTracker
    from .observation import build_game_state, relative_target

    mode = config["observation"]["mode"]
    use_pixels = mode != "state"
    continuous = config["control"]["scheduling"] == "continuous"
    min_request_interval = 1.0 / config["control"]["max_decision_hz"]
    planner_enabled = config.get("planner", {}).get("enabled", False)
    feedback = FeedbackTracker()

    recorder = Recorder(config["recording"]["output_dir"], config, synthetic=synthetic)
    print(f"Run: {recorder.path}", flush=True)
    backend = backend or (MockBackend if synthetic else RLBotBackend)(
        config["rlbot"], recorder.path
    )
    if use_pixels:
        capture = capture or (SyntheticCapture if synthetic else WindowCapture)(config["capture"])
    worker = planner_worker = None
    started = time.monotonic()
    frames = deque(maxlen=config["capture"]["frame_stack"])
    observation_generation = 0
    last_ready = False
    strategy = None
    strategy_at = 0.0
    strategy_generation = 0
    planner_pending = False
    planner_requests = 0
    next_planner = 0.0

    def gameplay_snapshot():
        nonlocal observation_generation, last_ready
        state = backend.snapshot()
        ready = bool(state.get("ready", False))
        if not ready:
            # An inference that spans a goal, pause, or lost camera belongs to
            # the previous play even if gameplay resumes before it returns.
            if last_ready:
                observation_generation += 1
            frames.clear()
            feedback.clear()
        last_ready = ready
        return state

    history = deque(maxlen=8)
    memory = ""
    pending = False
    decisions = errors = frame_number = 0
    next_capture = next_telemetry = next_decision = action_until = 0.0
    final = {}
    status, fatal_error = "completed", None
    try:
        client = client or build_client(config["model"])
        if planner_enabled:
            from .planner import build_planner

            planner_client = planner_client or build_planner(config["planner"])
        if config["recording"].get("live_window", False):
            # The monitor only tails local evidence. Closing it never stops the match.
            with (recorder.path / "live-window.log").open("wb") as live_log:
                monitor = subprocess.Popen(
                    [sys.executable, "-m", "rleval.live_view", str(recorder.path)],
                    stdin=subprocess.DEVNULL,
                    stdout=live_log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            recorder.event("live_window", 0.0, pid=monitor.pid)
        # Validate credentials/provider before launching Rocket League.
        backend.start()
        print(
            f"Offline environment ready ({mode}, {config['control']['scheduling']}). Ctrl+C stops the run.",
            flush=True,
        )
        if use_pixels:
            print("Keep the game visible for screenshot capture.", flush=True)
        worker = DecisionWorker(client)
        if planner_enabled:
            planner_worker = DecisionWorker(planner_client, state_only=True)
        started = time.monotonic()
        while True:
            now = time.monotonic()
            elapsed = now - started
            final = gameplay_snapshot()
            feedback.observe(elapsed, final)
            if (recorder.path / "stop.request").exists():
                status = "interrupted"
                break
            if final.get("error"):
                raise RuntimeError(final["error"])
            if final.get("ended"):
                status = "match_ended"
                break
            if elapsed >= config["episode"]["duration_s"]:
                status = "time_limit"
                break
            if (
                decisions >= config["episode"]["max_decisions"]
                and not pending
                and now >= action_until
            ):
                status = "decision_limit"
                break
            if not final.get("ready", False):
                # Never mix opponent/replay/countdown frames into the next observation stack.
                frames.clear()
            elif use_pixels and now >= next_capture:
                jpeg = capture.grab()
                captured_at = time.monotonic() - started
                final = gameplay_snapshot()
                if final.get("ready", False):
                    filename = recorder.frame(frame_number, jpeg)
                    frames.append(
                        {"jpeg": jpeg, "elapsed_s": round(captured_at, 6), "path": filename}
                    )
                    recorder.event("frame", captured_at, number=frame_number, path=filename)
                    frame_number += 1
                else:
                    frames.clear()
                next_capture = now + 1 / config["capture"]["sample_hz"]
            # Capture/disk IO can take time. Do not apply or request an action past the deadline.
            now = time.monotonic()
            elapsed = now - started
            if elapsed >= config["episode"]["duration_s"]:
                status = "time_limit"
                break
            if now >= next_telemetry:
                recorder.event("telemetry", elapsed, state=final)
                next_telemetry = now + 1 / config["recording"]["telemetry_hz"]
            if planner_worker:
                if strategy and (
                    strategy_generation != observation_generation
                    or elapsed - strategy_at > config["planner"]["max_age_s"]
                ):
                    recorder.event(
                        "planner_invalidated", elapsed, reason="Gameplay changed or plan expired"
                    )
                    strategy = None
                    next_planner = 0.0
                try:
                    planned_request, planned_reply, planned_error, planned_latency = (
                        planner_worker.results.get_nowait()
                    )
                except Empty:
                    pass
                else:
                    planner_pending = False
                    final = gameplay_snapshot()
                    plan_age = elapsed - planned_request["observed_at"]
                    fields = {
                        "number": planned_request["number"],
                        "latency_s": round(planned_latency, 6),
                        "observation_age_s": round(plan_age, 6),
                        "accepted": False,
                        "usage": planned_reply.usage if planned_reply else {},
                        "response_id": planned_reply.response_id if planned_reply else None,
                        "resolved_model": planned_reply.resolved_model if planned_reply else None,
                    }
                    if planned_error:
                        fields["error"] = planned_error
                    elif (
                        not final.get("ready", False)
                        or planned_request["observation_generation"] != observation_generation
                        or plan_age > config["planner"]["max_age_s"]
                    ):
                        fields["error"] = "Planner observation expired or gameplay changed"
                        next_planner = 0.0
                    else:
                        strategy = {
                            "version": planned_request["number"],
                            "model": planned_reply.resolved_model or config["planner"]["model"],
                            "plan": planned_reply.plan,
                        }
                        # Plan expiry includes the time already spent in inference.
                        strategy_at = planned_request["observed_at"]
                        strategy_generation = observation_generation
                        fields.update(accepted=True, plan=planned_reply.plan)
                        print(
                            f"  strategy {planned_request['number']}: {planned_latency:.2f}s — {planned_reply.plan['objective']}",
                            flush=True,
                        )
                    recorder.event("planner_decision", elapsed, **fields)
                if (
                    not planner_pending
                    and now >= next_planner
                    and planner_requests < config["planner"]["max_requests"]
                    and final.get("ready", False)
                ):
                    planning_state = gameplay_snapshot()
                    if planning_state.get("ready", False):
                        planning_context = {
                            "observation_mode": mode,
                            "task": config["episode"]["task"],
                            "team": "blue",
                            "elapsed_s": round(elapsed, 3),
                            "game_state": build_game_state(planning_state),
                            "previous_actions": list(history),
                            "previous_plan": strategy["plan"] if strategy else None,
                        }
                        planner_requests += 1
                        planning_request = {
                            "number": planner_requests,
                            "context": planning_context,
                            "observed_at": elapsed - float(planning_state.get("packet_age_s") or 0),
                            "observation_generation": observation_generation,
                        }
                        recorder.event(
                            "planner_request",
                            elapsed,
                            number=planner_requests,
                            context=planning_context,
                        )
                        planner_worker.requests.put_nowait(planning_request)
                        planner_pending = True
                        next_planner = time.monotonic() + config["planner"]["interval_s"]
            try:
                request, reply, error, latency = worker.results.get_nowait()
            except Empty:
                pass
            else:
                pending = False
                final = gameplay_snapshot()
                age = elapsed - request["observed_at"]
                fields = {
                    "number": request["number"],
                    "latency_s": round(latency, 6),
                    "observation_age_s": round(age, 6),
                    "accepted": False,
                    "usage": reply.usage if reply else {},
                    "response_id": reply.response_id if reply else None,
                    "model_decision": reply.decision if reply else {},
                }
                inactive = not final.get("ready", False)
                invalidated = request["observation_generation"] != observation_generation
                try:
                    if error:
                        raise ValueError(error)
                    if inactive:
                        raise ValueError("Gameplay is inactive; response discarded")
                    if invalidated:
                        raise ValueError(
                            "Gameplay changed since the observation; response discarded"
                        )
                    if age > config["episode"]["max_observation_age_s"]:
                        raise ValueError(
                            "Stale response: observation exceeded max_observation_age_s"
                        )
                    action = parse_action(reply.action)
                    backend.submit(action.controls, action.hold_ms)
                    action_until = time.monotonic() + action.hold_ms / 1000
                    fields.update(accepted=True, action=action.to_dict())
                    memory = action.memory
                    history.append(
                        {
                            "at_s": round(elapsed, 3),
                            "controls": action.controls,
                            "hold_ms": action.hold_ms,
                        }
                    )
                    print(
                        f"  decision {request['number']}: {latency:.2f}s, hold {action.hold_ms}ms",
                        flush=True,
                    )
                except (ValueError, TypeError) as exc:
                    backend.neutral()
                    action_until = 0.0
                    fields["error"] = str(exc)
                    # A goal/countdown during inference is not a provider failure.
                    errors += int(not (inactive or invalidated) or bool(error))
                    print(f"  rejected decision: {exc}", flush=True)
                recorder.event("decision", elapsed, **fields)
                if not continuous:
                    next_decision = max(
                        next_decision, time.monotonic() + config["episode"]["decision_interval_s"]
                    )
                if errors >= config["episode"]["max_errors"]:
                    raise RuntimeError(
                        "Maximum failed/rejected decisions reached; see events.jsonl"
                    )
            if (
                not pending
                and (frames or not use_pixels)
                and now >= next_decision
                and (continuous or now >= action_until)
                and decisions < config["episode"]["max_decisions"]
                and final.get("ready", False)
            ):
                # Explicit allowlist: state is opt-in and never the entire backend snapshot.
                context = {
                    "observation_mode": mode,
                    "task": config["episode"]["task"],
                    "team": "blue",
                    "elapsed_s": round(elapsed, 3),
                    "remaining_s": round(max(0, config["episode"]["duration_s"] - elapsed), 3),
                    "frame_times_s": [f["elapsed_s"] for f in frames],
                    "previous_actions": list(history),
                    "memory": memory,
                    "timing": (
                        "Real time. A new decision may replace a held action. Controls return to neutral on hold_ms expiry."
                        if continuous
                        else "Real time. Controls return to neutral after hold_ms while you think."
                    ),
                }
                if strategy and strategy_generation == observation_generation:
                    context["strategy"] = dict(strategy, age_s=round(elapsed - strategy_at, 3))
                observed_at = frames[-1]["elapsed_s"] if use_pixels else elapsed
                if mode != "pixels":
                    # Read again at dispatch so state-only calls use the freshest game packet.
                    latest = gameplay_snapshot()
                    if not latest.get("ready", False):
                        frames.clear()
                        time.sleep(0.01)
                        continue
                    context["game_state"] = build_game_state(latest)
                    context["control_feedback"] = latest.get("control_feedback", {})
                    context["recent_transitions"] = feedback.recent()
                    if "strategy" in context:
                        context["strategy"]["relative_target"] = relative_target(
                            context["game_state"]["player"],
                            context["strategy"]["plan"]["target_position"],
                        )
                    observed_at = min(
                        observed_at,
                        time.monotonic() - started - float(latest.get("packet_age_s") or 0),
                    )
                    if elapsed - observed_at > config["episode"]["max_observation_age_s"]:
                        time.sleep(0.01)
                        continue
                    if config["observation"].get("representation") == "compact":
                        from .compact import build_motor_observation

                        active_plan = context.get("strategy", {}).get("plan", {})
                        target = active_plan.get("target_position") or latest.get("ball_position")
                        task = active_plan.get("objective", context["task"])
                        if active_plan.get("guidance"):
                            task += ". " + active_plan["guidance"]
                        context = {
                            "observation_mode": "state",
                            "observation_representation": "compact",
                            "motor_observation": build_motor_observation(
                                latest,
                                target,
                                active_plan.get("desired_speed", 900.0),
                                task,
                                feedback.recent(),
                            ),
                        }
                decisions += 1
                request = {
                    "number": decisions,
                    "frames": [f["jpeg"] for f in frames],
                    "context": context,
                    "observed_at": observed_at,
                    "observation_generation": observation_generation,
                }
                recorder.event(
                    "request",
                    elapsed,
                    number=decisions,
                    context=context,
                    frames=[f["path"] for f in frames],
                )
                worker.requests.put_nowait(request)
                pending = True
                next_decision = time.monotonic() + min_request_interval
            time.sleep(0.01)
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as exc:  # noqa: BLE001 -- always release controls and persist a failed run
        status, fatal_error = "failed", f"{type(exc).__name__}: {exc}"
        print(fatal_error, flush=True)
    finally:
        episode_elapsed = time.monotonic() - started
        # Input shutdown happens immediately, before waiting on any network cleanup.
        try:
            backend.neutral()
            final = backend.snapshot()
        except Exception as exc:  # noqa: BLE001 -- remaining cleanup must still run
            recorder.event("neutral_error", time.monotonic() - started, error=type(exc).__name__)
        try:
            backend.close()
        except Exception as exc:  # noqa: BLE001 -- remaining cleanup must still run
            recorder.event("cleanup_error", time.monotonic() - started, error=type(exc).__name__)
        try:
            if capture is not None:
                capture.close()
        except Exception as exc:  # noqa: BLE001 -- preserve the report even if capture cleanup fails
            recorder.event("cleanup_error", time.monotonic() - started, error=type(exc).__name__)
        if worker:
            worker.close()
        elif client:
            client.close()
        if planner_worker:
            planner_worker.close()
        elif planner_client:
            planner_client.close()
        summary = recorder.finish(status, episode_elapsed, final, fatal_error)
    print(f"Finished: {status}. Report: {recorder.path / 'report.html'}", flush=True)
    return recorder.path, summary
