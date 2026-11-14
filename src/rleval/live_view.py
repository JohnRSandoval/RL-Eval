"""Desktop monitor for an existing or currently running episode.

Run ``python -m rleval.live_view RUN_DIRECTORY``. The monitor only reads local
recordings. Its Stop button requests a graceful stop through a local file;
closing the window never stops the match or changes controller inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path


class EventTail:
    """Incrementally read complete UTF-8 JSONL records, tolerating partial writes."""

    def __init__(self, path: Path, *, read_limit: int = 1024 * 1024):
        self.path = Path(path)
        self.read_limit = read_limit
        self.offset = 0
        self.pending = b""
        self.identity = None

    def poll(self) -> list[dict]:
        try:
            stat = self.path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if self.identity != identity or stat.st_size < self.offset:
                self.offset = 0
                self.pending = b""
                self.identity = identity
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                chunk = stream.read(self.read_limit)
                self.offset = stream.tell()
        except OSError:
            return []
        lines = (self.pending + chunk).split(b"\n")
        self.pending = lines.pop()
        events = []
        for line in lines:
            try:
                event = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(event, dict):
                events.append(event)
        return events


def _number(value, default=0.0):
    return value if type(value) in (int, float) and math.isfinite(value) else default


def ranked_probabilities(probabilities: dict, limit=5) -> list[tuple[str, float]]:
    if not isinstance(probabilities, dict):
        return []
    candidates = [
        (str(choice), float(value))
        for choice, value in probabilities.items()
        if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
    ]
    return sorted(candidates, key=lambda item: item[1], reverse=True)[:limit]


class LiveState:
    """Small display projection; request contexts and model memory are not retained."""

    def __init__(self):
        self.elapsed_s = 0.0
        self.status = "Starting offline match..."
        self.telemetry = {}
        self.decision = {}
        self.recent = deque(maxlen=12)
        self.requests = 0
        self.completed = 0
        self.accepted = 0
        self.latency_total = 0.0
        self.planner_requests = 0
        self.planner_completed = 0
        self.planner_accepted = 0
        self.planner_status = "Waiting for strategist"
        self.planner_model = ""
        self.planner_latency_s = None
        self.plan = {}
        self.plan_version = None
        self.plan_received_at = None
        self.finished = False
        self.revision = 0
        self.trial_label = ""
        self.trial_target = None
        self.motor = {}
        self.motor_summary = {}
        self.motor_samples = 0

    def update(self, event: dict):
        kind = event.get("type")
        self.elapsed_s = max(self.elapsed_s, _number(event.get("elapsed_s")))
        if kind == "motor_sample":
            self.motor_samples += 1
            self.motor = {
                key: event.get(key)
                for key in (
                    "controls",
                    "target",
                    "skill",
                    "status",
                    "metrics",
                    "plan_version",
                    "policy_source",
                    "tick_count",
                )
            }
            self.recent.appendleft(
                f"{self.elapsed_s:6.1f}s  tick {event.get('tick_count', '?')}  "
                f"{event.get('policy_source', 'unknown')}  {event.get('skill', '')}"
            )
            if not self.finished:
                self.status = "Local controller: " + str(event.get("status") or "running")
        elif kind == "motor_summary":
            self.motor_summary = {
                key: event.get(key)
                for key in (
                    "ticks",
                    "elapsed_s",
                    "worker_elapsed_s",
                    "active_ticks",
                    "mean_compute_ms",
                    "max_compute_ms",
                    "achieved_hz",
                    "source_counts",
                    "errors",
                )
            }
        elif kind == "trial_started":
            self.trial_label = f"{event.get('policy', '')} · {event.get('name', '')}"
            self.trial_target = event.get("target")
            self.status = str(event.get("description", "Calibration trial"))
        elif kind == "trial_finished":
            self.status = "Trial " + ("PASS" if event.get("metrics", {}).get("success") else "FAIL")
        elif kind == "planner_request":
            self.planner_requests += 1
            self.planner_status = f"Planning #{event.get('number', self.planner_requests)}"
        elif kind == "planner_decision":
            self.planner_completed += 1
            self.planner_latency_s = _number(event.get("latency_s"), None)
            if event.get("resolved_model"):
                self.planner_model = str(event["resolved_model"])
            if event.get("accepted") is True and isinstance(event.get("plan"), dict):
                self.planner_accepted += 1
                self.plan = {
                    name: event["plan"].get(name)
                    for name in (
                        "objective",
                        "target_position",
                        "desired_speed",
                        "boost_policy",
                        "jump_policy",
                        "guidance",
                    )
                }
                self.plan_version = event.get("number", self.planner_completed)
                self.plan_received_at = self.elapsed_s
                self.planner_status = "Plan accepted"
            else:
                self.planner_status = (
                    "Plan discarded: " + str(event.get("error") or "invalid response")[:180]
                )
        elif kind == "planner_invalidated":
            self.plan = {}
            self.plan_version = None
            self.plan_received_at = None
            self.planner_status = (
                "Plan invalidated: " + str(event.get("reason") or "gameplay changed")[:180]
            )
        elif kind == "request":
            self.requests += 1
            if not self.finished:
                self.status = "Model deciding"
        elif kind == "telemetry" and isinstance(event.get("state"), dict):
            state = event["state"]
            self.telemetry = {
                name: state.get(name)
                for name in (
                    "scores",
                    "game_time_remaining",
                    "is_overtime",
                    "phase",
                    "goals",
                    "touches",
                    "shots",
                    "saves",
                    "score",
                    "car_position",
                    "car_forward",
                    "ball_position",
                    "opponents",
                    "last_input",
                    "ready",
                )
            }
            if not self.finished and not state.get("ready"):
                self.status = str(state.get("phase", "Waiting for gameplay"))
        elif kind == "decision":
            details = event.get("model_decision")
            details = details if isinstance(details, dict) else {}
            display_probabilities = details.get("probabilities") or {
                f"{axis}.{choice}": probability
                for axis, question in details.get("question_details", {}).items()
                for choice, probability in question.get("probabilities", {}).items()
            }
            action = event.get("action")
            action = action if isinstance(action, dict) else {}
            self.completed += 1
            accepted = event.get("accepted") is True
            self.accepted += int(accepted)
            latency = _number(event.get("latency_s"))
            self.latency_total += latency
            self.decision = {
                "number": event.get("number", self.completed),
                "choice": str(details.get("choice", "controller JSON")),
                "confidence": _number(details.get("confidence"), None),
                "top_probabilities": ranked_probabilities(display_probabilities, limit=6),
                "latency_s": latency,
                "observation_age_s": _number(event.get("observation_age_s")),
                "accepted": accepted,
                "error": str(event.get("error") or "")[:300],
                "controls": action.get("controls")
                if isinstance(action.get("controls"), dict)
                else {},
                "hold_ms": action.get("hold_ms"),
            }
            mark = "OK" if accepted else "DROP"
            self.recent.appendleft(
                f"{self.elapsed_s:6.1f}s  #{self.decision['number']:<4} {mark:4} "
                f"{latency * 1000:4.0f}ms  {self.decision['choice']}"
            )
            self.status = "Action accepted" if accepted else "Response discarded"
        elif kind == "finished":
            self.finished = True
            self.status = "Finished: " + str(event.get("status", "complete"))
            if event.get("error"):
                self.status += " — " + str(event["error"])[:180]
        self.revision += 1

    def load_summary(self, summary: dict):
        pipeline = summary.get("mixed_policy_pipeline", summary)
        if isinstance(pipeline.get("motor"), dict) and pipeline["motor"].get("enabled"):
            self.update({"type": "motor_summary", **pipeline["motor"]})
        if isinstance(pipeline.get("final_telemetry"), dict):
            self.update({"type": "telemetry", "state": pipeline["final_telemetry"]})
        if summary.get("evaluation_kind") == "control_calibration":
            self.trial_label = " · ".join(
                f"{policy}: {counts.get('passed', 0)}/{counts.get('total', 0)}"
                for policy, counts in summary.get("policies", {}).items()
            )
        self.update(
            {
                "type": "finished",
                "status": summary.get("status", "complete"),
                "error": summary.get("error"),
                "elapsed_s": summary.get("elapsed_s", self.elapsed_s),
            }
        )


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def request_stop(run_dir: Path):
    """Idempotently request runner shutdown, without claiming it has stopped."""
    try:
        with (Path(run_dir) / "stop.request").open("x", encoding="utf-8") as stream:
            stream.write("Stop requested from the live outputs window.\n")
    except FileExistsError:
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(description="Watch recorded model outputs in a live window")
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_directory.resolve()

    import tkinter as tk

    root = tk.Tk()
    root.title(f"Rocket League • Live model outputs • {run_dir.name}")
    root.configure(bg="#10161b")
    width, height = (
        min(580, root.winfo_screenwidth() - 40),
        min(870, root.winfo_screenheight() - 100),
    )
    root.geometry(f"{width}x{height}+{max(0, root.winfo_screenwidth() - width - 24)}+40")
    root.minsize(min(490, width), min(720, height))
    root.attributes("-topmost", True)
    panel = tk.Frame(root, bg="#10161b", padx=18, pady=14)
    panel.pack(fill="both", expand=True)

    def label(text="", *, size=10, color="#b7c4cd", bold=False, parent=panel):
        widget = tk.Label(
            parent,
            text=text,
            bg="#10161b",
            fg=color,
            anchor="w",
            justify="left",
            font=("Segoe UI", size, "bold" if bold else "normal"),
        )
        widget.pack(fill="x", pady=2)
        return widget

    label("OFFLINE ROCKET LEAGUE / LIVE OUTPUTS", size=9, color="#7fd2c1")
    title = label("Controls: loading model...", size=12, color="#f0f4f7", bold=True)
    assistance = label("", size=9, color="#ffbb86")
    scoreboard = label("BLUE  0    :    0  ORANGE", size=20, color="#f0f4f7", bold=True)
    clock_label = label("Starting offline match...")
    stats = label("Goals 0 · touches 0 · shots 0 · saves 0")

    strategy_panel = tk.Frame(panel, bg="#10161b")
    strategy_panel.pack(fill="x", pady=(6, 3))
    strategy_model = label(
        "Strategy: loading model...", size=12, color="#d4b8fa", bold=True, parent=strategy_panel
    )
    strategy_status = label("Waiting for strategist", size=9, parent=strategy_panel)
    strategy_objective = label(
        "Fallback controls while strategy is pending",
        size=10,
        color="#f0f4f7",
        parent=strategy_panel,
    )
    strategy_target = label("", size=9, parent=strategy_panel)
    strategy_policies = label("", size=9, parent=strategy_panel)
    guidance = tk.Text(
        strategy_panel,
        height=3,
        bg="#211e2a",
        fg="#e1d5ec",
        font=("Segoe UI", 9),
        borderwidth=0,
        padx=8,
        pady=5,
        wrap="word",
        state="disabled",
    )
    guidance.pack(fill="x")
    arena = tk.Canvas(panel, height=110, bg="#14261f", highlightthickness=0)
    arena.pack(fill="x", pady=(8, 6))
    choice = label("Waiting for first decision", size=17, color="#7fd2c1", bold=True)
    detail = label("Model output will appear here.")
    error_label = label("", size=9, color="#ffbb86")
    controls = label("Controls: neutral", size=10)
    probabilities = label("", size=10)
    rates = label("0 decisions · 0 accepted", size=10)
    history_label = label("RECENT DECISIONS", size=9, color="#7fd2c1")
    history = tk.Text(
        panel,
        height=4,
        bg="#1a242c",
        fg="#d5e0e7",
        font=("Consolas", 9),
        borderwidth=0,
        padx=8,
        pady=6,
        wrap="none",
        state="disabled",
    )
    history.pack(fill="both", expand=True)
    footer = tk.Frame(panel, bg="#10161b")
    footer.pack(fill="x", pady=(8, 0))
    stop_requested = False
    stop_error = ""

    def stop_run():
        nonlocal stop_requested, stop_error
        try:
            request_stop(run_dir)
            stop_requested = True
            stop_button.configure(text="Stop requested", state="disabled")
        except OSError:
            stop_error = "Could not write the stop request; stop the runner in its terminal."

    stop_button = tk.Button(
        footer,
        text="Stop run",
        command=stop_run,
        bg="#9c483b",
        fg="#ffffff",
        activebackground="#b85647",
        activeforeground="#ffffff",
        relief="flat",
        padx=8,
    )
    stop_button.pack(side="left", padx=(0, 6))
    pinned = tk.BooleanVar(value=True)
    tk.Checkbutton(
        footer,
        text="Keep on top",
        variable=pinned,
        command=lambda: root.attributes("-topmost", pinned.get()),
        bg="#10161b",
        fg="#b7c4cd",
        selectcolor="#1a242c",
        activebackground="#10161b",
        activeforeground="#f0f4f7",
    ).pack(side="left")
    tk.Label(
        footer,
        text="Closing this window leaves the match running.",
        font=("Segoe UI", 8),
        bg="#10161b",
        fg="#8c9da9",
    ).pack(side="right")

    state = LiveState()
    tail = EventTail(run_dir / "events.jsonl")
    metadata = {}
    summary_loaded = False
    history_revision = 0
    guidance_text = None
    motor_mode = False
    motor_target_hz = 60

    def draw_arena():
        arena.delete("all")
        w, h = max(arena.winfo_width(), 490), max(arena.winfo_height(), 110)
        scale = min((w - 100) / 10240, (h - 20) / 8192)
        cx, cy = w / 2, h / 2

        def point(v):
            if not isinstance(v, dict) or any(
                type(v.get(axis)) not in (float, int) or not math.isfinite(v[axis])
                for axis in ("x", "y")
            ):
                return None
            return cx + v["y"] * scale, cy + v["x"] * scale

        left, right = cx - 5120 * scale, cx + 5120 * scale
        top, bottom = cy - 4096 * scale, cy + 4096 * scale
        arena.create_rectangle(left, top, right, bottom, outline="#58786a")
        arena.create_line(cx, top, cx, bottom, fill="#456456")
        arena.create_oval(cx - 12, cy - 12, cx + 12, cy + 12, outline="#456456")
        arena.create_text(left - 26, cy, text="BLUE", fill="#6bb7ff", font=("Segoe UI", 8))
        arena.create_text(right + 29, cy, text="ORANGE", fill="#f2a45f", font=("Segoe UI", 8))

        def symbol(position, heading, color, radius):
            p = point(position)
            if p is None:
                return
            x, y = p
            arena.create_oval(
                x - radius, y - radius, x + radius, y + radius, fill=color, outline=""
            )
            if isinstance(heading, dict):
                dx, dy = _number(heading.get("y")), _number(heading.get("x"))
                norm = math.hypot(dx, dy)
                if norm > 0.001:
                    arena.create_line(
                        x,
                        y,
                        x + dx / norm * 18,
                        y + dy / norm * 18,
                        fill=color,
                        width=2,
                        arrow="last",
                    )

        t = state.telemetry
        for opponent in t.get("opponents") or []:
            if isinstance(opponent, dict):
                symbol(opponent.get("position"), opponent.get("forward"), "#f2a45f", 4)
        symbol(t.get("car_position"), t.get("car_forward"), "#6bb7ff", 5)
        symbol(t.get("ball_position"), None, "#f7f0ce", 5)
        target = point(
            state.trial_target or state.motor.get("target") or state.plan.get("target_position")
        )
        if target is not None:
            tx, ty = target
            arena.create_line(tx - 5, ty - 5, tx + 5, ty + 5, fill="#d4b8fa", width=2)
            arena.create_line(tx - 5, ty + 5, tx + 5, ty - 5, fill="#d4b8fa", width=2)

    def refresh():
        nonlocal \
            metadata, \
            summary_loaded, \
            history_revision, \
            guidance_text, \
            motor_mode, \
            motor_target_hz
        if not metadata:
            metadata = _read_json(run_dir / "metadata.json")
            model = metadata.get("config", {}).get("model", {})
            control = metadata.get("config", {}).get("control", {})
            motor_mode = (
                control.get("mode") == "local_motor" or model.get("provider") == "local-motor"
            )
            motor_target_hz = _number(control.get("motor_hz"), 60)
            if model:
                title.configure(
                    text=(
                        f"Controls: local controller ({model.get('model', '')})"
                        if motor_mode
                        else f"Controls: {model.get('model', '')} ({model.get('provider', '')})"
                    )
                )
            if motor_mode:
                assistance.configure(text="ASSISTED CONTROL · scripted low-level mechanics")
                history_label.configure(text="RECENT LOCAL CONTROLLER SAMPLES")
                choice.configure(text="Waiting for local controller")
                detail.configure(text="Controller samples will appear here.")
                rates.configure(text=f"Local controller target: {motor_target_hz:g} Hz")
            else:
                assistance.pack_forget()
            planner = metadata.get("config", {}).get("planner", {})
            strategy_model.configure(text=f"Strategy: {planner.get('model', 'disabled')}")
            if metadata and not planner.get("enabled", False):
                strategy_panel.pack_forget()
        for event in tail.poll():
            state.update(event)
        if state.finished and not summary_loaded:
            summary = _read_json(run_dir / "summary.json")
            waiting_for_calibration = (
                metadata.get("evaluation_kind") == "control_calibration"
                and summary.get("evaluation_kind") != "control_calibration"
            )
            if summary and not waiting_for_calibration:
                state.load_summary(summary)
                summary_loaded = True
        t, d = state.telemetry, state.decision
        scores = t.get("scores") or {}
        scoreboard.configure(
            text=state.trial_label
            or f"BLUE  {scores.get('blue', 0)}    :    {scores.get('orange', 0)}  ORANGE"
        )
        remaining = _number(t.get("game_time_remaining"), None)
        game_clock = (
            "--:--"
            if remaining is None
            else f"{int(abs(remaining)) // 60}:{int(abs(remaining)) % 60:02}"
        )
        if t.get("is_overtime"):
            game_clock = "OT " + game_clock
        clock_label.configure(
            text=f"Game {game_clock}  ·  {state.elapsed_s:.1f}s elapsed  ·  {state.status}"
            + (" · STOP REQUESTED" if stop_requested and not state.finished else "")
        )
        if state.finished:
            stop_button.configure(text="Finished", state="disabled")
        stats.configure(
            text=" · ".join(
                f"{name} {t.get(name) or 0}" for name in ("goals", "touches", "shots", "saves")
            )
        )
        if state.planner_model:
            strategy_model.configure(text=f"Strategy: {state.planner_model}")
        latency_text = (
            "" if state.planner_latency_s is None else f" · {state.planner_latency_s:.2f}s latency"
        )
        plan_age = (
            max(0, state.elapsed_s - state.plan_received_at)
            if state.plan_received_at is not None
            else None
        )
        version_text = (
            ""
            if plan_age is None
            else f" · plan #{state.plan_version}, received {plan_age:.1f}s ago"
        )
        strategy_status.configure(
            text=state.planner_status + latency_text + version_text,
            wraplength=max(450, root.winfo_width() - 40),
        )
        strategy_objective.configure(
            text=str(state.plan.get("objective") or "Fallback controls while strategy is pending"),
            wraplength=max(450, root.winfo_width() - 40),
        )
        target_position = state.plan.get("target_position")
        if isinstance(target_position, dict):
            target_text = ", ".join(
                f"{axis}={_number(target_position.get(axis)):.0f}" for axis in ("x", "y", "z")
            )
            strategy_target.configure(
                text=f"Target: {target_text} · speed {state.plan.get('desired_speed', 'n/a')}"
            )
        else:
            strategy_target.configure(text="No current target")
        strategy_policies.configure(
            text=(
                f"Boost: {state.plan.get('boost_policy')} · jump: {state.plan.get('jump_policy')}"
                if state.plan
                else ""
            ),
            wraplength=max(450, root.winfo_width() - 40),
        )
        new_guidance = str(state.plan.get("guidance") or "No accepted plan yet.")
        if new_guidance != guidance_text:
            guidance.configure(state="normal")
            guidance.delete("1.0", "end")
            guidance.insert("end", new_guidance)
            guidance.configure(state="disabled")
            guidance_text = new_guidance
        if motor_mode or state.motor:
            m = state.motor
            if m:
                choice.configure(text=str(m.get("skill") or "Local controller").replace("_", " "))
                source = {
                    "planner": "Planner-guided",
                    "scripted_baseline": "Scripted baseline",
                    "fallback": "Fallback",
                    "inactive": "Gameplay inactive",
                    "calibration": "Calibration target",
                }.get(m.get("policy_source"), "Unknown source")
                version = (
                    f" · plan #{m['plan_version']}" if m.get("plan_version") is not None else ""
                )
                detail.configure(text=f"{source}{version} · tick {m.get('tick_count', '?')}")
                error_label.configure(
                    text=stop_error or str(m.get("status") or ""),
                    wraplength=max(450, root.winfo_width() - 40),
                )
                c = m.get("controls") if isinstance(m.get("controls"), dict) else {}
                active = [
                    f"{name}={value:.3g}" if type(value) in (int, float) else f"{name}={value}"
                    for name, value in c.items()
                    if value
                ]
                controls.configure(
                    text="Applied controls: " + (", ".join(active) or "neutral"), wraplength=510
                )
                target = m.get("target")
                target_text = (
                    ", ".join(f"{axis}={_number(target.get(axis)):.0f}" for axis in ("x", "y", "z"))
                    if isinstance(target, dict)
                    else str(target or "none")
                )
                probabilities.configure(text="Local target: " + target_text)
            motor_result = state.motor_summary
            ticks = motor_result.get("ticks") if motor_result else m.get("tick_count", 0)
            rate = motor_result.get("achieved_hz") if motor_result else None
            rate_text = f" · measured {rate:.2f} Hz" if type(rate) in (int, float) else ""
            rates.configure(
                text=f"{ticks} local ticks · target {motor_target_hz:g} Hz{rate_text} · {state.motor_samples} log samples"
            )
        elif d:
            choice.configure(text=d["choice"].replace("_", " "))
            confidence = d["confidence"]
            confidence_text = "n/a" if confidence is None else f"{confidence:.1%}"
            detail.configure(
                text=f"#{d['number']}  ·  confidence {confidence_text}  ·  {d['latency_s'] * 1000:.0f} ms"
                f"  ·  {'ACCEPTED' if d['accepted'] else 'DISCARDED'}"
            )
            error_label.configure(
                text=stop_error or d["error"], wraplength=max(450, root.winfo_width() - 40)
            )
            c = d["controls"]
            active = [f"{name}={value}" for name, value in c.items() if value]
            controls.configure(
                text="Selected controls: " + (", ".join(active) or "neutral"), wraplength=510
            )
            ranked = d["top_probabilities"]
            entries = [f"{p:5.1%} {name}" for name, p in ranked]
            probabilities.configure(
                text="\n".join(
                    "   ·   ".join(entries[i : i + 2]) for i in range(0, len(entries), 2)
                )
            )
        mean_ms = state.latency_total / state.completed * 1000 if state.completed else 0
        hz = state.accepted / state.elapsed_s if state.elapsed_s else 0
        if not motor_mode and not state.motor:
            rates.configure(
                text=f"{state.completed} decisions · {state.accepted} accepted · {hz:.2f}/s · mean {mean_ms:.0f} ms"
            )
        if history_revision != (state.completed, state.motor_samples):
            history.configure(state="normal")
            history.delete("1.0", "end")
            history.insert("end", "\n".join(state.recent))
            history.configure(state="disabled")
            history_revision = (state.completed, state.motor_samples)
        draw_arena()
        root.after(250, refresh)

    root.after(50, refresh)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
