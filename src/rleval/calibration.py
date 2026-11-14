"""Repeatable offline motor trials, with explicit reference/model comparison."""

import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from html import escape
from queue import Empty

from .actions import parse_action
from .backend import MockBackend, RLBotBackend
from .calibration_tasks import ReferencePolicy, TrialScorer, default_scenarios
from .compact import build_motor_observation
from .feedback import FeedbackTracker
from .providers import build_client
from .recording import Recorder
from .runner import DecisionWorker


def _write_report(path, summary):
    rows = []
    for trial in summary["trials"]:
        metrics = trial["metrics"]
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{escape(str(value))}</td>"
                for value in (
                    trial["policy"],
                    trial["name"],
                    trial["repeat"],
                    "PASS" if metrics.get("success") else "FAIL",
                    metrics.get("elapsed_s"),
                    metrics.get(
                        "final_heading_error_degrees", metrics.get("final_heading_error_deg", "—")
                    ),
                    metrics.get("final_distance_uu", "—"),
                    metrics.get("final_speed_uu_s", "—"),
                    trial["accepted_actions"],
                    trial.get("error") or "",
                )
            )
            + "</tr>"
        )
    body = """<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Rocket League control calibration</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 24px;background:#10161b;color:#e4eef3}h1{font-size:30px}p{line-height:1.6;color:#bccbd5}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #35424b}th{color:#8bd3c8}a{color:#8bd3c8}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#19232b;padding:16px} .wrap{overflow-x:auto}</style>
<h1>Control calibration</h1><p>__DESCRIPTION__</p>
<div class='wrap'><table><tr><th>Policy</th><th>Trial</th><th>Repeat</th><th>Result</th><th>Seconds</th><th>Final angle °</th><th>Distance uu</th><th>Speed uu/s</th><th>Actions</th><th>Error</th></tr>__ROWS__</table></div>
<p><a href='replay.html'>Telemetry and action trace</a> · <a href='events.jsonl'>Events</a> · <a href='summary.json'>Summary JSON</a> · <a href='metadata.json'>Configuration</a></p>
<details><summary>Summary and scoring details</summary><pre>__SUMMARY__</pre></details>"""
    description = (
        "Local motor controller with continuous axes at a 60 Hz target, using specified offline starting states. No model API calls. This tests mechanics and uses a different action space and rate from the 5 Hz model calibration."
        if summary.get("action_interface") == "continuous_motor"
        else "Specified starting states in offline Rocket League. The reference controller and model run separately, using the same reset scenarios, request-rate ceiling, and input lease. Reference controls have no network inference delay; this is a diagnostic comparison, not a latency-matched model benchmark."
    )
    (path / "report.html").write_text(
        body.replace("__ROWS__", "".join(rows))
        .replace("__SUMMARY__", escape(json.dumps(summary, indent=2)))
        .replace("__DESCRIPTION__", description),
        encoding="utf-8",
    )


def run_calibration(
    config,
    *,
    policies=("reference", "model"),
    repeats=1,
    scenarios=None,
    backend=None,
    client_factory=None,
):
    if not policies or any(p not in {"reference", "model", "motor"} for p in policies):
        raise ValueError("Calibration policies must be reference, model, or motor")
    motor_run = "motor" in policies
    if motor_run and tuple(policies) != ("motor",):
        raise ValueError("Run motor calibration separately: it uses continuous 60 Hz controls")
    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise ValueError("Calibration repeats must be an integer in [1, 10]")
    scenarios = default_scenarios() if scenarios is None else list(scenarios)
    if not scenarios:
        raise ValueError("Select at least one calibration scenario")
    config = copy.deepcopy(config)
    config["observation"].update(mode="state", representation="compact")
    config["planner"]["enabled"] = False
    config["rlbot"].update(
        freeplay=False,
        calibration_mode=True,
        opponent_skill="None",
        require_camera=False,
        spectate_player=True,
        game_speed=1.0,
    )
    config["model"]["action_interface"] = "ground_axes_v1"
    if motor_run:
        config["observation"]["representation"] = "full"
        config["control"]["mode"] = "local_motor"
        config["model"].update(
            provider="local-motor", model="local-motor-v1", action_interface="continuous_motor"
        )
    config["control"]["scheduling"] = "continuous"
    config["calibration"] = {
        "policies": list(policies),
        "repeats": repeats,
        "scenarios": [asdict(s) for s in scenarios],
    }
    recorder = Recorder(
        config["recording"]["output_dir"], config, synthetic=isinstance(backend, MockBackend)
    )
    recorder.metadata.update(
        evaluation_kind="control_calibration",
        observation_representation="full" if motor_run else "compact",
        action_format="local_motor_controller" if motor_run else "ground_axes_v1",
    )
    (recorder.path / "metadata.json").write_text(
        json.dumps(recorder.metadata, indent=2), encoding="utf-8"
    )
    backend = backend or RLBotBackend(config["rlbot"], recorder.path)
    client_factory = client_factory or (lambda: build_client(config["model"]))
    reference = ReferencePolicy()
    trials, final = [], {}
    started = time.monotonic()
    status, fatal = "completed", None
    print(f"Calibration: {recorder.path}", flush=True)
    preflight_client = None
    try:
        if "model" in policies:
            preflight_client = client_factory()  # Credential/config check before game launch.
        if config["recording"].get("live_window"):
            with (recorder.path / "live-window.log").open("wb") as log:
                subprocess.Popen(
                    [sys.executable, "-m", "rleval.live_view", str(recorder.path)],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
        backend.start()
        started = time.monotonic()
        number = 0
        for repeat in range(1, repeats + 1):
            for scenario in scenarios:
                for policy in policies:
                    # Construct network clients before resetting physics so setup
                    # time cannot consume one policy's specified initial velocity.
                    if policy == "model" and preflight_client is None:
                        preflight_client = client_factory()
                    initial = backend.reset_trial(
                        scenario.position, scenario.velocity, scenario.yaw
                    )
                    trial_started = time.monotonic()
                    scorer, feedback = TrialScorer(scenario), FeedbackTracker()
                    worker = local_motor = None
                    if policy == "model":
                        client = preflight_client
                        preflight_client = None
                        worker = DecisionWorker(client)
                    elif policy == "motor":
                        from .assisted import MotorWorker

                        local_motor = MotorWorker(
                            backend,
                            config,
                            initial_plan={
                                "skill": "stop" if scenario.kind == "brake" else "drive_to",
                                "objective": scenario.description,
                                "guidance": "Calibration start",
                                "target_position": scenario.target,
                                "desired_speed": scenario.desired_speed,
                                "boost_policy": "conserve",
                                "jump_policy": "avoid",
                            },
                        )
                        local_motor.start()
                    trial = {
                        "name": scenario.name,
                        "policy": policy,
                        "repeat": repeat,
                        "scenario": asdict(scenario),
                        "initial_state": initial,
                        "accepted_actions": 0,
                        "requests": 0,
                        "error": None,
                    }
                    recorder.event(
                        "trial_started",
                        trial_started - started,
                        name=scenario.name,
                        policy=policy,
                        repeat=repeat,
                        description=scenario.description,
                        target=scenario.target,
                    )
                    print(f"  {policy}: {scenario.name} (repeat {repeat})", flush=True)
                    pending = False
                    next_request = next_telemetry = 0.0
                    trial_errors = 0
                    try:
                        while True:
                            now = time.monotonic()
                            elapsed, trial_elapsed = now - started, now - trial_started
                            final = backend.snapshot()
                            if (recorder.path / "stop.request").exists():
                                raise KeyboardInterrupt
                            if final.get("error") or final.get("ended"):
                                raise RuntimeError(
                                    final.get("error") or "Offline calibration ended unexpectedly"
                                )
                            feedback.observe(elapsed, final)
                            success = scorer.update(trial_elapsed, final)
                            if now >= next_telemetry:
                                recorder.event(
                                    "telemetry",
                                    elapsed,
                                    state=final,
                                    trial=scenario.name,
                                    policy=policy,
                                )
                                if local_motor:
                                    motor_state = local_motor.snapshot()
                                    if motor_state["error"]:
                                        raise RuntimeError(motor_state["error"])
                                    if motor_state["sample"]:
                                        recorder.event(
                                            "motor_sample", elapsed, **motor_state["sample"]
                                        )
                                next_telemetry = now + 1 / config["recording"]["telemetry_hz"]
                            if success or trial_elapsed >= scenario.duration_s:
                                break
                            result = None
                            if worker:
                                try:
                                    result = worker.results.get_nowait()
                                except Empty:
                                    pass
                            if result:
                                request, reply, error, latency = result
                                pending = False
                                fields = {
                                    "number": request["number"],
                                    "policy": policy,
                                    "trial": scenario.name,
                                    "latency_s": latency,
                                    "observation_age_s": elapsed - request["observed_at"],
                                    "accepted": False,
                                    "usage": reply.usage if reply else {},
                                    "model_decision": reply.decision if reply else {},
                                }
                                try:
                                    if error:
                                        raise ValueError(error)
                                    if fields["observation_age_s"] > config["episode"][
                                        "max_observation_age_s"
                                    ] or not final.get("ready"):
                                        raise ValueError(
                                            "Stale or inactive calibration observation"
                                        )
                                    action = parse_action(reply.action)
                                    backend.submit(action.controls, action.hold_ms)
                                    fields.update(accepted=True, action=action.to_dict())
                                    trial["accepted_actions"] += 1
                                except (TypeError, ValueError) as exc:
                                    backend.neutral()
                                    fields["error"] = str(exc)
                                    trial_errors += 1
                                recorder.event("decision", elapsed, **fields)
                                if trial_errors >= 5:
                                    raise RuntimeError(
                                        "Repeated provider failures during calibration"
                                    )
                            if (
                                not local_motor
                                and not pending
                                and now >= next_request
                                and final.get("ready")
                            ):
                                observation = build_motor_observation(
                                    final,
                                    scenario.target,
                                    scenario.desired_speed,
                                    scenario.description,
                                    feedback.recent(),
                                )
                                context = {
                                    "observation_mode": "state",
                                    "observation_representation": "compact",
                                    "motor_observation": observation,
                                }
                                number += 1
                                trial["requests"] += 1
                                recorder.event(
                                    "request",
                                    elapsed,
                                    number=number,
                                    policy=policy,
                                    trial=scenario.name,
                                    context=context,
                                    frames=[],
                                )
                                if worker:
                                    worker.requests.put_nowait(
                                        {
                                            "number": number,
                                            "frames": [],
                                            "context": context,
                                            "observed_at": elapsed
                                            - float(final.get("packet_age_s") or 0),
                                        }
                                    )
                                    pending = True
                                else:
                                    controls = reference.decide(final, scenario)
                                    action = parse_action(
                                        {
                                            "controls": controls,
                                            "hold_ms": config["model"].get("decision_hold_ms", 300),
                                            "memory": "",
                                        }
                                    )
                                    backend.submit(action.controls, action.hold_ms)
                                    recorder.event(
                                        "decision",
                                        elapsed,
                                        number=number,
                                        policy=policy,
                                        trial=scenario.name,
                                        latency_s=0.0,
                                        observation_age_s=final.get("packet_age_s", 0),
                                        accepted=True,
                                        usage={},
                                        model_decision={
                                            "choice": "reference controller",
                                            "catalog_version": "ground-axes-v1",
                                        },
                                        action=action.to_dict(),
                                    )
                                    trial["accepted_actions"] += 1
                                next_request = (
                                    time.monotonic() + 1 / config["control"]["max_decision_hz"]
                                )
                            time.sleep(0.01)
                    except Exception as exc:
                        trial["error"] = f"{type(exc).__name__}: {exc}"
                        raise
                    finally:
                        if local_motor:
                            local_motor.close()
                            trial["motor"] = local_motor.summary()
                            trial["accepted_actions"] = trial["motor"]["active_ticks"]
                        backend.neutral()
                        if worker:
                            worker.close()
                        trial["metrics"] = scorer.finish()
                        trials.append(trial)
                        recorder.event(
                            "trial_finished",
                            time.monotonic() - started,
                            trial=scenario.name,
                            policy=policy,
                            repeat=repeat,
                            metrics=trial["metrics"],
                        )
                        print(
                            f"    {'PASS' if trial['metrics'].get('success') else 'FAIL'}: {trial['metrics']}",
                            flush=True,
                        )
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as exc:  # noqa: BLE001 -- neutralize and preserve evidence on all failures
        status, fatal = "failed", f"{type(exc).__name__}: {exc}"
        print(fatal, flush=True)
    finally:
        if preflight_client:
            preflight_client.close()
        for operation in (backend.neutral, backend.snapshot, backend.close):
            try:
                result = operation()
                if isinstance(result, dict):
                    final = result
            except Exception as exc:  # noqa: BLE001 -- finish remaining cleanup and reporting
                recorder.event(
                    "cleanup_error", time.monotonic() - started, error=type(exc).__name__
                )
                status, fatal = "failed", fatal or f"Cleanup failed: {type(exc).__name__}"
        elapsed = time.monotonic() - started
        motors = [t["motor"] for t in trials if "motor" in t]
        if motors:
            ticks = sum(m["ticks"] for m in motors)
            motor_elapsed = sum(m["worker_elapsed_s"] for m in motors)
            sources = {}
            for m in motors:
                for source, count in m["source_counts"].items():
                    sources[source] = sources.get(source, 0) + count
            recorder.event(
                "motor_summary",
                elapsed,
                ticks=ticks,
                worker_elapsed_s=motor_elapsed,
                active_ticks=sum(m["active_ticks"] for m in motors),
                mean_compute_ms=sum(m["mean_compute_ms"] * m["ticks"] for m in motors)
                / max(1, ticks),
                max_compute_ms=max(m["max_compute_ms"] for m in motors),
                achieved_hz=ticks / motor_elapsed if motor_elapsed else 0,
                source_counts=sources,
                errors=[e for m in motors for e in m["errors"]],
            )
        pipeline = recorder.finish(status, elapsed, final, fatal)
        (recorder.path / "replay.html").write_text(
            (recorder.path / "report.html").read_text(encoding="utf-8"), encoding="utf-8"
        )
        summary = {
            "evaluation_kind": "control_calibration",
            "synthetic_environment": recorder.metadata["synthetic_environment"],
            "status": status,
            "error": fatal,
            "elapsed_s": round(elapsed, 3),
            "model": config["model"]["model"],
            "action_interface": "continuous_motor" if motor_run else "ground_axes_v1",
            "policies": {
                p: {
                    "passed": sum(
                        t["metrics"].get("success", False) for t in trials if t["policy"] == p
                    ),
                    "total": sum(t["policy"] == p for t in trials),
                }
                for p in policies
            },
            "trials": trials,
            "mixed_policy_pipeline": pipeline,
        }
        (recorder.path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_report(recorder.path, summary)
    print(f"Calibration finished: {status}. Report: {recorder.path / 'report.html'}", flush=True)
    return recorder.path, summary
