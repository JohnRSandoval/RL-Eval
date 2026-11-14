"""Offline RLBot v5 transport. Raw telemetry stays inside the evaluator.

Tested against rlbot==2.0.0b56 / rlbot_flatbuffers==0.19.x. No OS input,
injection, online matchmaking, or ball prediction is used here.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

from rleval.actions import NEUTRAL, parse_action
from rleval.observation import orientation_vectors


class BackendError(RuntimeError):
    """An actionable local game/transport failure."""


class _RelayLogger(logging.LoggerAdapter):
    shutting_down = False

    def error(self, message, *args, **kwargs):
        if self.shutting_down and message == "SocketRelay disconnected unexpectedly!":
            return
        super().error(message, *args, **kwargs)


class ControlLease:
    """Bound each command by wall time, even if model inference blocks."""

    def __init__(self):
        self._lock = threading.Lock()
        self._controls = dict(NEUTRAL)
        self._expires = 0.0
        self._started_at = None
        self._requested_hold_ms = 0
        self._active = False
        self._command_id = 0
        self._last_completed = None

    def _finish(self, current: float, reason: str):
        if self._active:
            self._last_completed = {
                "command_id": self._command_id,
                "controls": dict(self._controls),
                "requested_hold_ms": self._requested_hold_ms,
                "commanded_duration_ms": round(
                    max(0.0, min(current, self._expires) - self._started_at) * 1000, 3
                ),
                "end_reason": "expired" if current >= self._expires else reason,
            }
            self._active = False

    def submit(self, controls: dict, hold_ms: int, *, now: float | None = None):
        action = parse_action({"controls": controls, "hold_ms": hold_ms, "memory": ""})
        current = time.monotonic() if now is None else now
        with self._lock:
            self._finish(current, "replaced")
            self._controls = dict(action.controls)
            self._started_at = current
            self._expires = current + hold_ms / 1000
            self._requested_hold_ms = hold_ms
            self._active = True
            self._command_id += 1

    def neutral(self, *, now: float | None = None):
        with self._lock:
            self._finish(time.monotonic() if now is None else now, "neutralized")

    def sample(self, *, now: float | None = None, active: bool = True) -> dict:
        current = time.monotonic() if now is None else now
        with self._lock:
            if not active or current >= self._expires:
                self._finish(current, "inactive")
            if not self._active:
                return dict(NEUTRAL)
            return dict(self._controls)

    def feedback(self, *, now: float | None = None, active: bool = True) -> dict:
        """Command-lease timing, not a claim of when the game applied each input."""
        current = time.monotonic() if now is None else now
        with self._lock:
            if not active or current >= self._expires:
                self._finish(current, "inactive")
            return {
                "source": "command_lease_not_game_acknowledgement",
                "command_id": self._command_id,
                "commanded_controls": dict(self._controls if self._active else NEUTRAL),
                "selected_controls": dict(self._controls),
                "requested_hold_ms": self._requested_hold_ms,
                "command_age_ms": round(max(0.0, current - self._started_at) * 1000, 3)
                if self._started_at is not None
                else 0.0,
                "remaining_ms": round(max(0.0, self._expires - current) * 1000, 3)
                if self._active
                else 0.0,
                "active": self._active,
                "last_completed": copy.deepcopy(self._last_completed),
            }


def _trial_state(position: dict, velocity: dict, yaw: float, boost: float) -> tuple:
    """Validate a bounded offline reset before any game mutation."""
    vectors = []
    for name, value, limits in (
        ("position", position, {"x": (-4096, 4096), "y": (-5120, 5120), "z": (0, 2048)}),
        ("velocity", velocity, {axis: (-6000, 6000) for axis in ("x", "y", "z")}),
    ):
        if not isinstance(value, dict) or set(value) != {"x", "y", "z"}:
            raise BackendError(f"Trial {name} must contain exactly x/y/z")
        parsed = {}
        for axis, (low, high) in limits.items():
            number = value[axis]
            if (
                type(number) not in (int, float)
                or not math.isfinite(number)
                or not low <= number <= high
            ):
                raise BackendError(f"Trial {name}.{axis} must be finite in [{low}, {high}]")
            parsed[axis] = float(number)
        vectors.append(parsed)
    if type(yaw) not in (int, float) or not math.isfinite(yaw):
        raise BackendError("Trial yaw must be finite radians")
    if type(boost) not in (int, float) or not math.isfinite(boost) or not 0 <= boost <= 100:
        raise BackendError("Trial boost must be finite in [0, 100]")
    return vectors[0], vectors[1], (float(yaw) + math.pi) % (2 * math.pi) - math.pi, float(boost)


def verify_offline_processes(processes=None) -> list[int]:
    """Fail closed if an already-running game cannot be verified as -rlbot.

    A closed game is fine: RLBot launches it with the official offline flag.
    We never close an unrelated game or change its launch arguments.
    """
    import psutil

    found = []
    for process in psutil.process_iter() if processes is None else processes:
        try:
            name = process.name().lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if name not in {"rocketleague.exe", "rocketleague"}:
            continue
        try:
            argv = process.cmdline()
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise BackendError(
                "Cannot verify the running Rocket League process. Close it, then rerun "
                "so RLBot can launch its offline instance."
            ) from exc
        if not any(arg.lower() == "-rlbot" for arg in argv):
            raise BackendError(
                "Rocket League is already running without -rlbot. Close the game, "
                "then rerun; this harness only launches offline RLBot matches."
            )
        found.append(process.pid)
    return found


def find_server_path(configured: str = "") -> Path | None:
    """Find a core executable without downloading or changing an installation."""
    name = "RLBotServer.exe" if os.name == "nt" else "RLBotServer"
    candidates = []
    if configured:
        path = Path(configured).expanduser().resolve()
        candidates.append(path / name if path.is_dir() else path)
    else:
        candidates.extend([Path.cwd() / "runtime" / name, Path.cwd() / name])
        if os.environ.get("LOCALAPPDATA"):
            candidates.append(Path(os.environ["LOCALAPPDATA"]) / "RLBot5" / "bin" / name)
    return next((p.resolve() for p in candidates if p.is_file()), None)


def _isolated_relay(agent_id: str, timeout: float):
    """Upstream uses class-level handler lists; isolate clients and serialize sends."""
    from rlbot.interface import SocketRelay

    class IsolatedRelay(SocketRelay):
        def __init__(self):
            super().__init__(agent_id, connection_timeout=timeout)
            self.logger = _RelayLogger(self.logger, {})
            for name in (
                "on_connect_handlers",
                "packet_handlers",
                "field_info_handlers",
                "match_config_handlers",
                "match_comm_handlers",
                "ball_prediction_handlers",
                "controllable_team_info_handlers",
                "raw_handlers",
            ):
                setattr(self, name, [])
            self._send_lock = threading.Lock()

        def send_bytes(self, data):
            with self._send_lock:
                return super().send_bytes(data)

    return IsolatedRelay()


class RLBotBackend:
    """One blue model-controlled car, optionally versus a Psyonix bot.

    Owns a dedicated RLBotServer process; an existing core is never commandeered.
    Controller leases are refreshed at 120 Hz independently of the model thread.
    Snapshot fields are evaluator data. Only observation.build_game_state may
    allowlist state for an explicitly selected state-assisted evaluation track.
    """

    def __init__(self, config: dict, run_dir: Path):
        self.config = dict(config)
        self._require_camera = self.config.get("require_camera", True)
        if type(self._require_camera) is not bool:
            raise BackendError("require_camera must be boolean")
        self._spectate_player = self.config.get("spectate_player", self._require_camera)
        if type(self._spectate_player) is not bool:
            raise BackendError("spectate_player must be boolean")
        self.run_dir = Path(run_dir).resolve()
        self._lease = ControlLease()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._match_accepted = threading.Event()
        self._threads = []
        self._manager = self._relay = self._core = self._core_log = None
        self._flat = None
        self._index = None
        self._player_id = None
        self._spawn_confirmed = False
        self._camera_configured = False
        self._last_camera_command = 0.0
        self._last_packet_at = None
        self._last_touch = None
        self._handshake = set()
        self._init_sent = False
        self._started_match = False
        self._closed = False
        self._state = {
            "ready": False,
            "ended": False,
            "active": False,
            "game_time": 0.0,
            "scores": {"blue": 0, "orange": 0},
            "player_index": None,
            "team": 0,
            "goals": 0,
            "touches": 0,
            "shots": 0,
            "saves": 0,
            "score": 0,
            "packets": 0,
            "phase": "Inactive",
            "error": None,
            "mock": False,
        }

    def _fail(self, message: str):
        self._lease.neutral()
        with self._lock:
            self._state.update(ready=False, active=False, ended=True, error=message)

    def _write_match(self) -> Path:
        launcher = self.config.get("launcher", "Epic")
        if launcher not in {"Epic", "Steam"}:
            raise BackendError("backend.launcher must be Epic or Steam")
        skill = self.config.get("opponent_skill", "AllStar")
        if skill not in {"AllStar", "Pro", "Rookie", "None"}:
            raise BackendError("opponent_skill must be AllStar, Pro, Rookie, or None")
        freeplay = self.config.get("freeplay", False)
        if type(freeplay) is not bool:
            raise BackendError("freeplay must be boolean")
        calibration = self.config.get("calibration_mode", False)
        if type(calibration) is not bool:
            raise BackendError("calibration_mode must be boolean")
        if calibration and skill != "None":
            raise BackendError("Calibration mode requires a solo offline match")
        agent_id = "rleval/" + uuid.uuid4().hex
        self._agent_id = agent_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "model.bot.toml").write_text(
            f'[settings]\nname = "Frontier Model"\nagent_id = "{agent_id}"\nrun_command = ""\n',
            encoding="utf-8",
        )
        match = (
            f'[rlbot]\nlauncher = "{launcher}"\nauto_start_agents = false\n'
            'wait_for_agents = true\n\n[match]\ngame_mode = "Soccar"\n'
            'game_map_upk = "Stadium_P"\nskip_replays = true\n'
            # Core rc18 reads start_without_countdown when StartCommand carries
            # a TOML path. Python beta.56's loader instead reads instant_start.
            # Keep both aliases equal so either parsing route skips Playtest in
            # freeplay, while ordinary matches retain their kickoff countdown.
            f"start_without_countdown = {str(freeplay).lower()}\n"
            f"instant_start = {str(freeplay).lower()}\n"
            'existing_match_behavior = "Restart"\n'
            "enable_rendering = false\nenable_state_setting = true\n"
            f'freeplay = {str(freeplay).lower()}\n\n[mutators]\nmatch_length = "FiveMinutes"\n'
            '\n[[cars]]\ntype = "rlbot"\nteam = 0\nconfig_file = "model.bot.toml"\n'
        )
        if skill != "None":
            match += (
                '\n[[cars]]\ntype = "psyonix"\nteam = 1\n'
                f'name = "Built-in opponent"\nskill = "{skill}"\n'
            )
        path = self.run_dir / "match.toml"
        path.write_text(match, encoding="utf-8")
        return path

    def _launch_core(self, path: Path):
        import psutil

        for proc in psutil.process_iter():
            try:
                if proc.name().lower() in {"rlbotserver", "rlbotserver.exe"}:
                    raise BackendError(
                        "RLBotServer is already running. Close RLBotGUI/other RLBot "
                        "sessions first; this harness needs its own offline match."
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        # RLBot chooses an available local port using this same mechanism.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self._core_log = (self.run_dir / "rlbot-server.log").open("wb")
        self._core = subprocess.Popen(
            [str(path), str(port)],
            cwd=path.parent,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=self._core_log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return port

    def _thread(self, target, *args):
        thread = threading.Thread(target=target, args=args, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _receive(self, relay, label: str):
        from rlbot.interface import MsgHandlingResult

        try:
            while not self._stop.is_set():
                result = relay.handle_incoming_messages(blocking=True)
                if result == MsgHandlingResult.TERMINATED:
                    if not self._stop.is_set():
                        self._fail(f"RLBot {label} connection ended")
                    break
        except Exception as exc:  # noqa: BLE001 - transport thread must report every failure to runner
            if not self._stop.is_set():
                self._fail(f"RLBot {label} connection failed ({type(exc).__name__})")

    def _handshake_part(self, part: str):
        self._handshake.add(part)
        if self._handshake == {"config", "field", "mapping"} and not self._init_sent:
            self._init_sent = True
            self._relay.send_msg(self._flat.InitComplete())

    def _mapping(self, mapping):
        if mapping.team != 0 or len(mapping.controllables) != 1:
            raise BackendError("RLBot assigned an unexpected team or number of cars")
        self._index = mapping.controllables[0].index
        self._player_id = mapping.controllables[0].identifier
        self._spawn_confirmed = False
        with self._lock:
            self._state["player_index"] = self._index
            self._state["assigned_player_id"] = self._player_id
        self._handshake_part("mapping")

    def _packet(self, packet):
        if self._index is None:
            return
        if self._state["error"] is not None:
            return
        if (
            self._index >= len(packet.players)
            or self._player_id is None
            or packet.players[self._index].player_id != self._player_id
        ):
            # Mapping is reserved before map load/spawn. Packets from the old
            # match can still have a car at the same index; its identity differs.
            self._spawn_confirmed = False
            self._lease.neutral()
            with self._lock:
                self._state.update(ready=False, active=False)
            return
        self._spawn_confirmed = True
        player = packet.players[self._index]
        car = self._player_telemetry(player, self._index)
        ball = packet.balls[0].physics if packet.balls else None
        match = packet.match_info
        phase = match.match_phase
        active = phase in {self._flat.MatchPhase.Active, self._flat.MatchPhase.Kickoff}
        ended = phase == self._flat.MatchPhase.Ended
        if not active:
            self._lease.neutral()
        with self._lock:
            self._last_packet_at = time.monotonic()
            scores = {"blue": 0, "orange": 0}
            for team in packet.teams:
                if team.team_index in {0, 1}:
                    scores["blue" if team.team_index == 0 else "orange"] = team.score
            if player.latest_touch is not None:
                stamp = (player.latest_touch.game_seconds, player.latest_touch.ball_index)
                if stamp != self._last_touch:
                    self._state["touches"] += 1
                    self._last_touch = stamp
            camera_ok = match.last_spectated == self._index
            camera_ready = not self._require_camera or not self._camera_configured or camera_ok
            self._state.update(
                ready=active and camera_ready,
                ended=ended,
                active=active,
                game_time=match.seconds_elapsed,
                game_time_remaining=match.game_time_remaining,
                is_overtime=match.is_overtime,
                game_speed=match.game_speed,
                phase=str(phase).split(".")[-1],
                scores=scores,
                goals=player.score_info.goals,
                own_goals=player.score_info.own_goals,
                shots=player.score_info.shots,
                saves=player.score_info.saves,
                score=player.score_info.score,
                frame_num=match.frame_num,
                last_spectated=match.last_spectated,
                camera_on_player=camera_ok,
                camera_required=self._require_camera,
                player_id=player.player_id,
                player_index=self._index,
                team=player.team,
                car_position=car["position"],
                car_velocity=car["velocity"],
                car_angular_velocity=car["angular_velocity"],
                car_rotation=car["rotation"],
                car_forward=car["forward"],
                car_right=car["right"],
                car_up=car["up"],
                car_boost=car["boost"],
                car_on_ground=car["on_ground"],
                car_has_wheel_contact=car["has_wheel_contact"],
                car_has_jumped=car["has_jumped"],
                car_has_double_jumped=car["has_double_jumped"],
                car_has_dodged=car["has_dodged"],
                ball_position=self._xyz(ball.location) if ball else None,
                ball_velocity=self._xyz(ball.velocity) if ball else None,
                ball_angular_velocity=self._xyz(ball.angular_velocity) if ball else None,
                opponents=[
                    self._player_telemetry(other, index)
                    for index, other in enumerate(packet.players)
                    if other.team != player.team
                ],
                packet_received_monotonic_s=self._last_packet_at,
                last_input={key: getattr(player.last_input, key) for key in NEUTRAL},
                packets=self._state["packets"] + 1,
            )
        if active:
            self._ready.set()

    @staticmethod
    def _xyz(vector) -> dict:
        return {axis: getattr(vector, axis) for axis in ("x", "y", "z")}

    def _player_telemetry(self, player, index: int) -> dict:
        physics = player.physics
        rotation = {axis: getattr(physics.rotation, axis) for axis in ("pitch", "yaw", "roll")}
        return {
            "index": index,
            "team": player.team,
            "position": self._xyz(physics.location),
            "velocity": self._xyz(physics.velocity),
            "angular_velocity": self._xyz(physics.angular_velocity),
            "rotation": rotation,
            **orientation_vectors(rotation),
            "boost": player.boost,
            "on_ground": player.air_state == self._flat.AirState.OnGround,
            "has_wheel_contact": player.air_state == self._flat.AirState.OnGround,
            "has_jumped": player.has_jumped,
            "has_double_jumped": player.has_double_jumped,
            "has_dodged": player.has_dodged,
        }

    def _send_controls(self, controls: dict):
        # Core can assign a controller index before Rocket League has spawned
        # the car. Even neutral inputs are invalid until a packet confirms it.
        if (
            self._relay is not None
            and self._index is not None
            and self._spawn_confirmed
            and self._last_packet_at is not None
            and self._relay.is_connected
        ):
            self._relay.send_msg(
                self._flat.PlayerInput(self._index, self._flat.ControllerState(**controls))
            )

    def _control_loop(self):
        packet_timeout = float(self.config.get("packet_timeout_s", 5.0))
        while not self._stop.wait(1 / 120):
            with self._lock:
                active = self._state["active"] and not self._state["ended"]
                last = self._last_packet_at
            if last is not None and time.monotonic() - last > packet_timeout:
                self._fail("No fresh RLBot game packets; controller neutralized")
                active = False
            try:
                if (
                    active
                    and (self._require_camera or self._spectate_player)
                    and self._camera_configured
                    and self._state.get("last_spectated") != self._index
                    and time.monotonic() - self._last_camera_command >= 1.0
                ):
                    self._set_camera()
                self._send_controls(self._lease.sample(active=active))
            except (OSError, AssertionError):
                if not self._stop.is_set():
                    self._fail("RLBot controller connection failed")
                break

    def start(self):
        if self._core is not None or self._closed:
            raise BackendError("Create a fresh backend for each run")
        try:
            from rlbot import flat
            from rlbot.managers.match import MatchManager
        except ImportError as exc:
            raise BackendError("Install RLBot v5: pip install 'rlbot==2.0.0b56'") from exc
        self._flat = flat
        timeout = float(self.config.get("startup_timeout_s", 90))
        if not 5 <= timeout <= 600:
            raise BackendError("startup_timeout_s must be between 5 and 600")
        speed = float(self.config.get("game_speed", 1.0))
        if not 0.05 <= speed <= 1.0:
            raise BackendError("game_speed must be between 0.05 and 1.0")
        verify_offline_processes()
        path = find_server_path(self.config.get("server_path", ""))
        if path is None:
            raise BackendError(
                "RLBot v5 core was not found. Install RLBot v5 from https://rlbot.org/v5/ "
                "or set backend.server_path to RLBotServer.exe."
            )
        match_path = self._write_match()
        deadline = time.monotonic() + timeout
        try:
            port = self._launch_core(path)
            self._manager = MatchManager(path)
            self._manager.rlbot_interface = _isolated_relay("", timeout)
            self._manager.rlbot_interface.packet_handlers.append(self._manager._packet_reporter)
            self._manager.rlbot_interface.match_config_handlers.append(
                lambda _: self._match_accepted.set()
            )
            self._manager.rlbot_server_port = port
            self._manager.connect(
                wants_match_communications=False,
                wants_ball_predictions=False,
                close_between_matches=False,
                rlbot_server_ip="127.0.0.1",
                rlbot_server_port=port,
            )
            self._thread(self._receive, self._manager.rlbot_interface, "manager")
            self._manager.start_match(match_path, wait_for_start=False, ensure_server_started=False)
            self._started_match = True
            # StartMatch first closes named bot sessions from the preceding match.
            # Wait for core's match configuration acknowledgment before connecting
            # our named controller, otherwise a fast client can lose that race.
            while not self._match_accepted.wait(0.1):
                if self._state["error"]:
                    raise BackendError(self._state["error"])
                if self._core.poll() is not None or time.monotonic() > deadline:
                    raise BackendError(
                        "RLBotServer did not accept the match configuration; "
                        "see rlbot-server.log in this run"
                    )
            self._relay = _isolated_relay(self._agent_id, max(1, deadline - time.monotonic()))
            self._relay.match_config_handlers.append(lambda _: self._handshake_part("config"))
            self._relay.field_info_handlers.append(lambda _: self._handshake_part("field"))
            self._relay.controllable_team_info_handlers.append(self._mapping)
            self._relay.packet_handlers.append(self._packet)
            self._relay.connect(
                wants_match_communications=False,
                wants_ball_predictions=False,
                # Core requires named bot clients to close between matches.
                # Only the anonymous match-manager connection may persist.
                close_between_matches=True,
                rlbot_server_ip="127.0.0.1",
                rlbot_server_port=port,
            )
            self._thread(self._receive, self._relay, "controller")
            self._thread(self._control_loop)
            while not self._ready.wait(0.1):
                if self._state["error"]:
                    raise BackendError(self._state["error"])
                if self._core.poll() is not None:
                    raise BackendError("RLBotServer exited; see rlbot-server.log in this run")
                if time.monotonic() > deadline:
                    raise BackendError(
                        "Offline match did not become ready before startup_timeout_s. "
                        "Check Epic/Steam login and rlbot-server.log, then retry."
                    )
            # Guard again after launch, before allowing model controls.
            if not verify_offline_processes():
                raise BackendError("No verifiable -rlbot Rocket League process after startup")
            self._configure_match_view(speed)
            # A command is asynchronous. Require a fresh packet acknowledging
            # the selected car before the caller starts collecting model frames.
            while self._require_camera:
                state = self.snapshot()
                if state["error"]:
                    raise BackendError(state["error"])
                if (
                    state["ready"]
                    and state.get("last_spectated") == self._index
                    and self._last_packet_at > self._last_camera_command
                ):
                    break
                if time.monotonic() > deadline:
                    raise BackendError(
                        "Could not establish the model car's spectator POV before startup timeout"
                    )
                time.sleep(0.05)
        except BaseException:
            self.close()
            raise

    def _configure_match_view(self, speed: float):
        # In current v5 flatbuffers, console commands are separate messages.
        # beta.56's MatchManager.set_game_state wrapper still passes the old
        # fourth argument to fill_desired_game_state; use the wire types directly.
        self._manager.rlbot_interface.send_msg(
            self._flat.DesiredGameState(match_info=self._flat.DesiredMatchInfo(game_speed=speed))
        )
        if self._require_camera or self._spectate_player:
            self._camera_configured = True
            self._set_camera()

    def _set_camera(self):
        self._last_camera_command = time.monotonic()
        self._manager.rlbot_interface.send_msg(self._flat.ConsoleCommand("ViewPlayer 0 0"))
        # Core rc18 queues ConsoleCommand without flushing it. An empty state
        # message flushes that command queue without altering any gameplay state.
        # Keep the order: command first, flush second, on the same socket.
        self._manager.rlbot_interface.send_msg(self._flat.DesiredGameState())

    def submit(self, controls: dict, hold_ms: int):
        with self._lock:
            if self._closed or self._state["ended"]:
                raise BackendError(self._state["error"] or "Offline match is not running")
        self._lease.submit(controls, hold_ms)

    def neutral(self):
        self._lease.neutral()
        try:
            self._send_controls(dict(NEUTRAL))
        except (OSError, AssertionError):
            pass

    def snapshot(self) -> dict:
        with self._lock:
            state = copy.deepcopy(self._state)
            state["packet_age_s"] = (
                None if self._last_packet_at is None else time.monotonic() - self._last_packet_at
            )
        state["control_feedback"] = self._lease.feedback(
            active=state["active"] and not state["ended"]
        )
        return state

    def reset_trial(self, position: dict, velocity: dict, yaw: float, boost: float = 33) -> dict:
        """Set one offline calibration start, then require a fresh matching game packet.

        Position/velocity tolerances acknowledge that real-time physics keeps
        advancing while the state travels through RLBot; this is not frame stepping.
        """
        position, velocity, yaw, boost = _trial_state(position, velocity, yaw, boost)
        reset_mode = (
            self.config.get("freeplay") is True or self.config.get("calibration_mode") is True
        )
        if not reset_mode or self.config.get("opponent_skill") != "None":
            raise BackendError(
                "Trial resets require an explicitly configured solo freeplay or calibration session"
            )
        timeout = self.config.get("reset_timeout_s", 3.0)
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or not 0.05 <= timeout <= 30
        ):
            raise BackendError("reset_timeout_s must be finite in [0.05, 30]")
        with self._lock:
            if self._closed or self._manager is None or not self._spawn_confirmed:
                raise BackendError("Cannot reset before the owned offline car has spawned")
            before_packets = self._state["packets"]
            if self._state["error"] or self._state["ended"]:
                raise BackendError(self._state["error"] or "Offline match has ended")
            if not self._state["active"] or self._state.get("opponents"):
                raise BackendError("Trial resets require an active owned solo offline match")
        self.neutral()
        flat = self._flat
        vector = lambda values: flat.Vector3Partial(**values)
        zero = {"x": 0.0, "y": 0.0, "z": 0.0}
        car_states = [flat.DesiredCarState() for _ in range(self._index + 1)]
        car_states[self._index] = flat.DesiredCarState(
            physics=flat.DesiredPhysics(
                location=vector(position),
                velocity=vector(velocity),
                angular_velocity=vector(zero),
                rotation=flat.RotatorPartial(pitch=0.0, yaw=yaw, roll=0.0),
            ),
            boost_amount=boost,
        )
        sent_at = time.monotonic()
        self._manager.rlbot_interface.send_msg(
            flat.DesiredGameState(
                car_states=car_states,
                ball_states=[
                    flat.DesiredBallState(
                        physics=flat.DesiredPhysics(
                            location=vector({"x": 3500.0, "y": 3500.0, "z": 93.0}),
                            velocity=vector(zero),
                            angular_velocity=vector(zero),
                        )
                    )
                ],
                match_info=flat.DesiredMatchInfo(game_speed=1.0),
            )
        )
        deadline = sent_at + timeout
        while time.monotonic() < deadline:
            state = self.snapshot()
            if state.get("error") or state.get("ended"):
                raise BackendError(state.get("error") or "Match ended during trial reset")
            current_position = state.get("car_position", {})
            current_velocity = state.get("car_velocity", {})
            ball_position = state.get("ball_position") or {}
            ball_velocity = state.get("ball_velocity") or {}
            rotation = state.get("car_rotation", {})
            yaw_error = abs((rotation.get("yaw", 0) - yaw + math.pi) % (2 * math.pi) - math.pi)
            if (
                state.get("ready")
                and state["packets"] > before_packets
                and state.get("packet_received_monotonic_s", 0) >= sent_at
                # Controls and state setting use separate sockets. A fresh pose
                # packet can still contain old input until neutral reaches RL.
                and all(
                    state.get("last_input", {}).get(key) == value for key, value in NEUTRAL.items()
                )
                and all(
                    abs(current_position.get(axis, float("inf")) - position[axis]) <= 80
                    for axis in zero
                )
                and all(
                    abs(current_velocity.get(axis, float("inf")) - velocity[axis]) <= 150
                    for axis in zero
                )
                and yaw_error <= 0.05
                and abs(rotation.get("pitch", 0)) <= 0.08
                and abs(rotation.get("roll", 0)) <= 0.08
                and all(
                    abs(ball_position.get(axis, float("inf")) - value) <= 80
                    for axis, value in {"x": 3500.0, "y": 3500.0, "z": 93.0}.items()
                )
                and all(abs(ball_velocity.get(axis, float("inf"))) <= 150 for axis in zero)
            ):
                state["trial_reset"] = {
                    "requested_position": position,
                    "requested_velocity": velocity,
                    "requested_yaw": yaw,
                    "requested_boost": boost,
                    "observed_boost": state.get("car_boost"),
                    "boost_matches_requested": abs(state.get("car_boost", -1000) - boost) <= 1,
                    "ack_latency_s": round(time.monotonic() - sent_at, 6),
                    "position_tolerance_uu": 80,
                    "velocity_tolerance_uu_s": 150,
                    "yaw_tolerance_rad": 0.05,
                }
                return state
            time.sleep(0.01)
        self.neutral()
        raise BackendError("Trial reset was not acknowledged by a fresh matching game packet")

    @staticmethod
    def _close_socket(relay):
        if relay is None:
            return
        relay.is_connected = False
        if relay.socket is not None:
            try:
                relay.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            relay.socket.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.neutral()
        self._stop.set()
        for relay in (self._relay, self._manager.rlbot_interface if self._manager else None):
            if relay is not None and isinstance(getattr(relay, "logger", None), _RelayLogger):
                relay.logger.shutting_down = True
        if self._manager is not None and self._started_match:
            try:
                # This core process belongs to this run, never another user's server.
                self._manager.rlbot_interface.stop_match(shutdown_server=True)
            except (OSError, AssertionError):
                pass
        self._close_socket(self._relay)
        self._close_socket(self._manager.rlbot_interface if self._manager else None)
        for thread in self._threads:
            thread.join(timeout=1)
        if self._core is not None:
            try:
                self._core.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._core.terminate()
                try:
                    self._core.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._core.kill()
                    self._core.wait(timeout=3)
        if self._core_log is not None:
            self._core_log.close()
        with self._lock:
            self._state.update(ready=False, active=False, ended=True)


class MockBackend:
    """Transport smoke test only. Its zero scores are not gameplay results."""

    def __init__(self, config: dict, run_dir: Path):
        self.config, self.run_dir = config, run_dir
        self._lease = ControlLease()
        self._started = None
        self._ended = False
        self._trial_pose = None

    def start(self):
        self._started = time.monotonic()

    def submit(self, controls: dict, hold_ms: int):
        self._lease.submit(controls, hold_ms)

    def neutral(self):
        self._lease.neutral()

    def reset_trial(self, position: dict, velocity: dict, yaw: float, boost: float = 33) -> dict:
        position, velocity, yaw, boost = _trial_state(position, velocity, yaw, boost)
        if self._started is None or self._ended:
            raise BackendError("Start the synthetic backend before resetting a trial")
        self.neutral()
        self._trial_pose = (position, velocity, yaw, boost)
        state = self.snapshot()
        state["trial_reset"] = {"synthetic": True, "ack_latency_s": 0.0}
        return state

    def snapshot(self) -> dict:
        now = time.monotonic()
        elapsed = 0 if self._started is None else now - self._started
        position, velocity, yaw, boost = self._trial_pose or (
            {"x": 0.0, "y": -2048.0, "z": 17.0},
            {"x": 0.0, "y": 0.0, "z": 0.0},
            math.pi / 2,
            33.0,
        )
        rotation = {"pitch": 0.0, "yaw": yaw, "roll": 0.0}
        axes = orientation_vectors(rotation)
        opponent_rotation = {"pitch": 0.0, "yaw": -math.pi / 2, "roll": 0.0}
        return {
            "ready": self._started is not None and not self._ended,
            "ended": self._ended,
            "active": not self._ended,
            "game_time": elapsed,
            "game_time_remaining": max(0.0, 300.0 - elapsed),
            "is_overtime": False,
            "frame_num": int(elapsed * 120),
            "packet_received_monotonic_s": now,
            "packet_age_s": 0.0,
            "phase": "Ended" if self._ended else "Active",
            "scores": {"blue": 0, "orange": 0},
            "player_index": 0,
            "team": 0,
            "car_position": dict(position),
            "car_velocity": dict(velocity),
            "car_angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "car_rotation": rotation,
            "car_forward": axes["forward"],
            "car_right": axes["right"],
            "car_up": axes["up"],
            "car_boost": boost,
            "car_on_ground": True,
            "car_has_wheel_contact": True,
            "car_has_jumped": False,
            "car_has_double_jumped": False,
            "car_has_dodged": False,
            "ball_position": {"x": 3500.0, "y": 3500.0, "z": 93.0}
            if self._trial_pose
            else {"x": 0.0, "y": 0.0, "z": 93.0},
            "ball_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "ball_angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "opponents": [
                {
                    "index": 1,
                    "team": 1,
                    "position": {"x": 0.0, "y": 2048.0, "z": 17.0},
                    "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "rotation": opponent_rotation,
                    **orientation_vectors(opponent_rotation),
                    "boost": 33.0,
                    "on_ground": True,
                    "has_wheel_contact": True,
                    "has_jumped": False,
                    "has_double_jumped": False,
                    "has_dodged": False,
                }
            ],
            "goals": 0,
            "touches": 0,
            "shots": 0,
            "saves": 0,
            "score": 0,
            "controls": self._lease.sample(),
            "last_input": self._lease.sample(),
            "control_feedback": self._lease.feedback(
                active=self._started is not None and not self._ended
            ),
            "mock": True,
            "synthetic": True,
            "error": None,
        }

    def close(self):
        self.neutral()
        self._ended = True
