"""Offline boundaries, real v5 schema compatibility, and independent input expiry."""

import time
from types import SimpleNamespace

import pytest

from rleval.actions import NEUTRAL
from rleval.backend import (
    BackendError,
    ControlLease,
    MockBackend,
    RLBotBackend,
    _isolated_relay,
    verify_offline_processes,
)


def controls(**changes):
    return dict(NEUTRAL, **changes)


def test_control_lease_releases_all_buttons_while_inference_blocks():
    lease = ControlLease()
    lease.submit(controls(throttle=1, boost=True, jump=True), 500, now=10)
    assert lease.sample(now=10.49)["boost"] is True
    assert lease.sample(now=10.5) == NEUTRAL
    assert lease.sample(now=100) == NEUTRAL


def test_lease_overwrite_and_neutral_are_immediate():
    lease = ControlLease()
    lease.submit(controls(throttle=1), 1000, now=0)
    lease.submit(controls(steer=-1), 50, now=0.1)
    assert lease.sample(now=0.11)["throttle"] == 0
    assert lease.sample(now=0.11)["steer"] == -1
    lease.neutral()
    assert lease.sample(now=0.11) == NEUTRAL


def test_invalid_controller_never_enters_lease():
    lease = ControlLease()
    with pytest.raises(ValueError):
        lease.submit(controls(throttle=float("nan")), 100)
    with pytest.raises(ValueError):
        lease.submit(controls(), 5000)
    assert lease.sample() == NEUTRAL


def test_lease_feedback_records_replacement_expiry_and_neutralization():
    lease = ControlLease()
    lease.submit(controls(throttle=1, steer=-1), 300, now=10.0)
    first = lease.feedback(now=10.1)
    assert first["requested_hold_ms"] == 300
    assert first["command_age_ms"] == pytest.approx(100)
    assert first["remaining_ms"] == pytest.approx(200)
    assert first["commanded_controls"]["steer"] == -1
    lease.submit(controls(throttle=0.5), 300, now=10.12)
    second = lease.feedback(now=10.13)
    assert second["last_completed"]["commanded_duration_ms"] == pytest.approx(120)
    assert second["last_completed"]["end_reason"] == "replaced"
    assert second["last_completed"]["controls"]["steer"] == -1
    expired = lease.feedback(now=11.0)
    assert expired["last_completed"]["commanded_duration_ms"] == 300
    assert expired["last_completed"]["end_reason"] == "expired"
    assert expired["commanded_controls"] == NEUTRAL and not expired["active"]
    lease.submit(controls(boost=True), 500, now=12)
    lease.neutral(now=12.07)
    stopped = lease.feedback(now=12.08)
    assert stopped["last_completed"]["commanded_duration_ms"] == pytest.approx(70)
    assert stopped["last_completed"]["end_reason"] == "neutralized"
    assert stopped["source"] == "command_lease_not_game_acknowledgement"


def process(name, argv, pid=1):
    return SimpleNamespace(name=lambda: name, cmdline=lambda: argv, pid=pid)


def test_guard_rejects_online_and_substring_flags():
    for argv in (["RocketLeague.exe"], ["RocketLeague.exe", "-rlbot-fake"]):
        with pytest.raises(BackendError, match="without -rlbot"):
            verify_offline_processes([process("RocketLeague.exe", argv)])
    assert verify_offline_processes(
        [process("RocketLeague.exe", ["RocketLeague.exe", "-RLBot"], 77)]
    ) == [77]
    assert verify_offline_processes([]) == []


def test_guard_fails_closed_on_unreadable_game_commandline():
    import psutil

    def denied():
        raise psutil.AccessDenied(42)

    proc = process("RocketLeague.exe", [])
    proc.cmdline = denied
    with pytest.raises(BackendError, match="Cannot verify"):
        verify_offline_processes([proc])


def test_generated_toml_parses_with_pinned_rlbot_v5(tmp_path):
    flat = pytest.importorskip("rlbot.flat")
    from rlbot.config import load_match_config

    backend = RLBotBackend({"launcher": "Epic", "opponent_skill": "AllStar"}, tmp_path)
    parsed = load_match_config(backend._write_match())
    assert parsed.launcher == flat.Launcher.Epic
    assert parsed.auto_start_agents is False
    assert parsed.wait_for_agents is True
    assert parsed.enable_state_setting is True
    assert parsed.instant_start is False
    assert parsed.enable_rendering == flat.DebugRendering.OffByDefault
    assert len(parsed.player_configurations) == 2
    assert parsed.player_configurations[0].team == 0
    assert parsed.player_configurations[0].variety.agent_id == backend._agent_id
    assert isinstance(parsed.player_configurations[1].variety, flat.PsyonixBot)


def test_solo_config_and_relays_do_not_share_handlers(tmp_path):
    pytest.importorskip("rlbot.flat")
    from rlbot.config import load_match_config

    backend = RLBotBackend({"opponent_skill": "None"}, tmp_path)
    assert len(load_match_config(backend._write_match()).player_configurations) == 1
    a, b = _isolated_relay("a", 5), _isolated_relay("b", 5)
    a.packet_handlers.append(lambda _: None)
    assert b.packet_handlers == []


def test_freeplay_calibration_config_is_explicit(tmp_path):
    import tomllib

    pytest.importorskip("rlbot.flat")
    from rlbot.config import load_match_config

    backend = RLBotBackend({"opponent_skill": "None", "freeplay": True}, tmp_path)
    path = backend._write_match()
    # Production sends a StartCommand(path): core rc18 reads the first alias;
    # the pinned Python loader below reads the second alias instead.
    table = tomllib.loads(path.read_text(encoding="utf-8"))["match"]
    assert table["start_without_countdown"] is table["instant_start"] is True
    parsed = load_match_config(path)
    assert parsed.freeplay
    assert parsed.instant_start is True
    assert len(parsed.player_configurations) == 1


@pytest.mark.parametrize("mode", ["freeplay", "calibration_mode"])
def test_trial_reset_uses_current_wire_schema_and_waits_for_matching_packet(tmp_path, mode):
    flat = pytest.importorskip("rlbot.flat")
    backend = RLBotBackend(
        {"opponent_skill": "None", mode: True, "require_camera": False}, tmp_path
    )
    backend._flat, backend._index, backend._player_id = flat, 0, 42
    backend._packet(
        flat.GamePacket(
            players=[flat.PlayerInfo(player_id=42)],
            match_info=flat.MatchInfo(match_phase=flat.MatchPhase.Active),
        )
    )
    sent = []

    def send(message):
        decoded = flat.InterfacePacket.unpack(flat.InterfacePacket(message).pack()).message
        sent.append(decoded)
        if isinstance(decoded, flat.DesiredGameState):
            car = decoded.car_states[0]
            backend._packet(
                flat.GamePacket(
                    players=[
                        flat.PlayerInfo(
                            player_id=42,
                            boost=car.boost_amount,
                            physics=flat.Physics(
                                location=flat.Vector3(
                                    **{
                                        axis: getattr(car.physics.location, axis)
                                        for axis in ("x", "y", "z")
                                    }
                                ),
                                velocity=flat.Vector3(
                                    **{
                                        axis: getattr(car.physics.velocity, axis)
                                        for axis in ("x", "y", "z")
                                    }
                                ),
                                rotation=flat.Rotator(yaw=car.physics.rotation.yaw),
                            ),
                        )
                    ],
                    balls=[
                        flat.BallInfo(
                            physics=flat.Physics(
                                location=flat.Vector3(3500, 3500, 93),
                            )
                        )
                    ],
                    match_info=flat.MatchInfo(match_phase=flat.MatchPhase.Active, frame_num=2),
                )
            )

    backend._manager = SimpleNamespace(rlbot_interface=SimpleNamespace(send_msg=send))
    state = backend.reset_trial(
        position={"x": -1000, "y": 0, "z": 17},
        velocity={"x": 500, "y": 0, "z": 0},
        yaw=0.5,
        boost=25,
    )
    assert state["packets"] == 2
    assert state["car_position"]["x"] == -1000
    assert state["car_velocity"]["x"] == 500
    assert state["car_rotation"]["yaw"] == pytest.approx(0.5)
    assert state["trial_reset"]["ack_latency_s"] >= 0
    assert state["trial_reset"]["boost_matches_requested"]
    assert sent[0].ball_states[0].physics.location.x == 3500
    assert sent[0].ball_states[0].physics.velocity.x == 0
    assert sent[0].car_states[0].physics.angular_velocity.z == 0
    assert sent[0].match_info.game_speed == 1


def test_trial_reset_rejects_invalid_values_before_mutating_game(tmp_path):
    backend = RLBotBackend({"freeplay": True, "opponent_skill": "None"}, tmp_path)
    with pytest.raises(BackendError, match="finite"):
        backend.reset_trial({"x": float("nan"), "y": 0, "z": 17}, {"x": 0, "y": 0, "z": 0}, 0)
    with pytest.raises(BackendError, match="solo freeplay"):
        RLBotBackend({}, tmp_path).reset_trial(
            {"x": 0, "y": 0, "z": 17}, {"x": 0, "y": 0, "z": 0}, 0
        )


def test_solo_soccar_calibration_config_and_reset_guards(tmp_path):
    flat = pytest.importorskip("rlbot.flat")
    from rlbot.config import load_match_config

    backend = RLBotBackend(
        {"calibration_mode": True, "freeplay": False, "opponent_skill": "None"}, tmp_path
    )
    parsed = load_match_config(backend._write_match())
    assert not parsed.freeplay
    assert len(parsed.player_configurations) == 1
    with pytest.raises(BackendError, match="requires a solo"):
        RLBotBackend(
            {"calibration_mode": True, "opponent_skill": "Rookie"}, tmp_path
        )._write_match()
    with pytest.raises(BackendError, match="boolean"):
        RLBotBackend({"calibration_mode": "true"}, tmp_path)._write_match()

    backend._flat, backend._index, backend._player_id = flat, 0, 42
    sent = []
    backend._manager = SimpleNamespace(rlbot_interface=SimpleNamespace(send_msg=sent.append))
    reset_args = ({"x": 0, "y": 0, "z": 17}, {"x": 0, "y": 0, "z": 0}, 0)
    with pytest.raises(BackendError, match="owned offline car has spawned"):
        backend.reset_trial(*reset_args)
    backend._packet(
        flat.GamePacket(
            players=[flat.PlayerInfo(player_id=42)],
            match_info=flat.MatchInfo(match_phase=flat.MatchPhase.Paused),
        )
    )
    with pytest.raises(BackendError, match="active owned solo"):
        backend.reset_trial(*reset_args)
    assert not backend.snapshot()["active"]
    assert not backend._ready.is_set()
    assert not sent


def test_trial_reset_cannot_acknowledge_itself_using_an_old_matching_packet(tmp_path):
    flat = pytest.importorskip("rlbot.flat")
    backend = RLBotBackend(
        {
            "freeplay": True,
            "opponent_skill": "None",
            "require_camera": False,
            "reset_timeout_s": 0.05,
        },
        tmp_path,
    )
    backend._flat, backend._index, backend._player_id = flat, 0, 42
    backend._packet(
        flat.GamePacket(
            players=[
                flat.PlayerInfo(player_id=42, physics=flat.Physics(location=flat.Vector3(0, 0, 17)))
            ],
            match_info=flat.MatchInfo(match_phase=flat.MatchPhase.Active),
        )
    )
    sent = []
    backend._manager = SimpleNamespace(rlbot_interface=SimpleNamespace(send_msg=sent.append))
    with pytest.raises(BackendError, match="not acknowledged"):
        backend.reset_trial({"x": 0, "y": 0, "z": 17}, {"x": 0, "y": 0, "z": 0}, 0)
    assert len(sent) == 1
    assert backend._lease.sample() == NEUTRAL


def test_trial_reset_waits_for_game_acknowledgement_of_neutral_input(tmp_path, monkeypatch):
    flat = pytest.importorskip("rlbot.flat")
    backend = RLBotBackend(
        {"calibration_mode": True, "opponent_skill": "None", "require_camera": False}, tmp_path
    )
    backend._flat, backend._index, backend._player_id = flat, 0, 42

    def packet(steer):
        return flat.GamePacket(
            players=[
                flat.PlayerInfo(
                    player_id=42,
                    physics=flat.Physics(location=flat.Vector3(0, 0, 17)),
                    last_input=flat.ControllerState(steer=steer),
                )
            ],
            balls=[flat.BallInfo(physics=flat.Physics(location=flat.Vector3(3500, 3500, 93)))],
            match_info=flat.MatchInfo(match_phase=flat.MatchPhase.Active),
        )

    backend._packet(packet(1))
    backend._manager = SimpleNamespace(
        rlbot_interface=SimpleNamespace(send_msg=lambda _: backend._packet(packet(1)))
    )
    original_snapshot, snapshots = backend.snapshot, []

    def snapshot():
        if snapshots:
            backend._packet(packet(0))
        state = original_snapshot()
        snapshots.append(state)
        return state

    monkeypatch.setattr(backend, "snapshot", snapshot)
    state = backend.reset_trial({"x": 0, "y": 0, "z": 17}, {"x": 0, "y": 0, "z": 0}, 0)
    assert len(snapshots) == 2
    assert snapshots[0]["last_input"]["steer"] == 1
    assert state["last_input"] == NEUTRAL
    assert state["packets"] == 3


def test_synthetic_reset_is_labeled_static_state_without_fake_physics(tmp_path):
    backend = MockBackend({}, tmp_path)
    backend.start()
    state = backend.reset_trial({"x": 50, "y": 25, "z": 17}, {"x": 1000, "y": 0, "z": 0}, 0)
    assert state["trial_reset"]["synthetic"]
    assert state["car_position"] == {"x": 50, "y": 25, "z": 17}
    assert backend.snapshot()["car_position"] == state["car_position"]
    assert state["car_has_wheel_contact"]
    assert state["last_input"] == NEUTRAL


def test_real_v5_packet_scores_touches_and_pause_release(tmp_path):
    flat = pytest.importorskip("rlbot.flat")
    backend = RLBotBackend({}, tmp_path)
    backend._flat = flat
    backend._index = 0
    backend._player_id = 0
    player = flat.PlayerInfo(
        score_info=flat.ScoreInfo(goals=2, saves=1, shots=4, score=250),
        latest_touch=flat.Touch(game_seconds=2, ball_index=0),
    )
    packet = flat.GamePacket(
        players=[player],
        teams=[flat.TeamInfo(team_index=0, score=2), flat.TeamInfo(team_index=1, score=1)],
        match_info=flat.MatchInfo(seconds_elapsed=3, match_phase=flat.MatchPhase.Active),
    )
    backend._packet(packet)
    backend._packet(packet)
    state = backend.snapshot()
    assert state["scores"] == {"blue": 2, "orange": 1}
    assert state["touches"] == 1
    assert state["goals"] == 2 and state["saves"] == 1
    backend.submit(controls(boost=True), 1000)
    assert backend._lease.sample()["boost"]
    paused = flat.GamePacket(
        players=[player], match_info=flat.MatchInfo(match_phase=flat.MatchPhase.Paused)
    )
    backend._packet(paused)
    assert backend._lease.sample() == NEUTRAL
    backend._fail("test disconnect")
    backend._packet(packet)
    assert backend.snapshot()["ended"]
    assert backend.snapshot()["error"] == "test disconnect"


def test_mock_matches_runner_contract(tmp_path):
    backend = MockBackend({}, tmp_path)
    backend.start()
    backend.submit(controls(boost=True), 50)
    assert backend.snapshot()["ready"]
    assert backend.snapshot()["synthetic"]
    time.sleep(0.06)
    assert backend.snapshot()["controls"] == NEUTRAL
    backend.close()
    assert backend.snapshot()["ended"]


def test_mapping_alone_does_not_send_controls_before_car_exists(tmp_path):
    flat = pytest.importorskip("rlbot.flat")
    sent = []
    backend = RLBotBackend({}, tmp_path)
    backend._flat = flat
    backend._index = 0
    backend._player_id = 42
    backend._relay = SimpleNamespace(is_connected=True, send_msg=sent.append)
    backend._send_controls(dict(NEUTRAL))
    backend.neutral()
    assert sent == []
    # An old match's same index must not count as the new spawned car.
    backend._packet(
        flat.GamePacket(
            players=[flat.PlayerInfo(player_id=7)],
            match_info=flat.MatchInfo(match_phase=flat.MatchPhase.Active),
        )
    )
    backend._send_controls(dict(NEUTRAL))
    assert sent == []
    backend._packet(
        flat.GamePacket(
            players=[flat.PlayerInfo(player_id=42)],
            match_info=flat.MatchInfo(match_phase=flat.MatchPhase.Active),
        )
    )
    backend._send_controls(controls(throttle=1))
    backend.neutral()
    assert len(sent) == 2
    assert sent[0].controller_state.throttle == 1
    assert sent[1].controller_state.throttle == 0


def test_countdown_and_replay_are_not_model_decision_time(tmp_path):
    flat = pytest.importorskip("rlbot.flat")
    backend = RLBotBackend({}, tmp_path)
    backend._flat, backend._index, backend._player_id = flat, 0, 42
    for phase in (flat.MatchPhase.Countdown, flat.MatchPhase.Replay, flat.MatchPhase.Paused):
        backend._packet(
            flat.GamePacket(
                players=[flat.PlayerInfo(player_id=42)],
                match_info=flat.MatchInfo(match_phase=phase),
            )
        )
        assert not backend.snapshot()["ready"]
        assert not backend._ready.is_set()
    backend._packet(
        flat.GamePacket(
            players=[flat.PlayerInfo(player_id=42)],
            match_info=flat.MatchInfo(match_phase=flat.MatchPhase.Kickoff),
        )
    )
    assert backend.snapshot()["ready"]
    assert backend._ready.is_set()


def test_speed_and_camera_commands_use_current_flatbuffer_wire_schema(tmp_path):
    flat = pytest.importorskip("rlbot.flat")
    wire_messages = []

    def send(message):
        wire_messages.append(flat.InterfacePacket(message).pack())

    backend = RLBotBackend({}, tmp_path)
    backend._flat = flat
    backend._manager = SimpleNamespace(rlbot_interface=SimpleNamespace(send_msg=send))
    backend._configure_match_view(0.5)
    state = flat.InterfacePacket.unpack(wire_messages[0]).message
    command = flat.InterfacePacket.unpack(wire_messages[1]).message
    flush = flat.InterfacePacket.unpack(wire_messages[2]).message
    assert isinstance(state, flat.DesiredGameState)
    assert state.match_info.game_speed == 0.5
    assert isinstance(command, flat.ConsoleCommand)
    assert command.command == "ViewPlayer 0 0"
    assert isinstance(flush, flat.DesiredGameState)
    assert flush.match_info is None
    assert list(flush.car_states) == [] and list(flush.ball_states) == []


def test_state_only_needs_no_camera_commands_or_spectator_target(tmp_path):
    flat = pytest.importorskip("rlbot.flat")
    from rleval.observation import build_game_state

    backend = RLBotBackend({"require_camera": False}, tmp_path)
    backend._flat, backend._index, backend._player_id = flat, 0, 42
    sent = []
    backend._manager = SimpleNamespace(rlbot_interface=SimpleNamespace(send_msg=sent.append))
    backend._configure_match_view(1.0)
    assert len(sent) == 1 and isinstance(sent[0], flat.DesiredGameState)
    assert not backend._camera_configured
    backend._packet(
        flat.GamePacket(
            players=[
                flat.PlayerInfo(
                    player_id=42,
                    boost=77,
                    has_jumped=True,
                    air_state=flat.AirState.Jumping,
                    physics=flat.Physics(
                        location=flat.Vector3(1, 2, 3),
                        rotation=flat.Rotator(yaw=1.0),
                        angular_velocity=flat.Vector3(0.1, 0.2, 0.3),
                    ),
                ),
                flat.PlayerInfo(player_id=43, team=1, boost=20),
            ],
            balls=[flat.BallInfo(physics=flat.Physics(location=flat.Vector3(100, 200, 93)))],
            match_info=flat.MatchInfo(match_phase=flat.MatchPhase.Active, last_spectated=1),
        )
    )
    raw = backend.snapshot()
    assert raw["ready"] and not raw["camera_on_player"]
    assert raw["car_boost"] == 77 and raw["car_has_jumped"]
    assert not raw["car_on_ground"]
    assert raw["opponents"][0]["boost"] == 20
    state = build_game_state(raw)
    assert state["ball"]["position"] == {"x": 100, "y": 200, "z": 93}
    assert state["player"]["position"] == {"x": 1, "y": 2, "z": 3}


def test_state_only_can_show_player_without_requiring_camera(tmp_path):
    flat = pytest.importorskip("rlbot.flat")
    backend = RLBotBackend({"require_camera": False, "spectate_player": True}, tmp_path)
    backend._flat, backend._index, backend._player_id = flat, 0, 42
    sent = []
    backend._manager = SimpleNamespace(rlbot_interface=SimpleNamespace(send_msg=sent.append))
    backend._configure_match_view(1.0)
    assert any(isinstance(message, flat.ConsoleCommand) for message in sent)
    backend._packet(
        flat.GamePacket(
            players=[flat.PlayerInfo(player_id=42)],
            match_info=flat.MatchInfo(
                match_phase=flat.MatchPhase.Active, last_spectated=1, is_overtime=True
            ),
        )
    )
    state = backend.snapshot()
    assert state["ready"] and not state["camera_on_player"]
    assert state["is_overtime"]


def test_start_handshake_waits_for_match_before_named_bot_connects(tmp_path, monkeypatch):
    """Core rc18 rejects persistent named clients and closes old clients on StartMatch."""
    flat = pytest.importorskip("rlbot.flat")
    import rlbot.managers.match as match_module

    import rleval.backend as module

    events = []
    relays = []
    pending_console = []
    clock = [100.0]

    def advance(seconds):
        clock[0] += seconds

    # The mocked transport delivers packets synchronously. Give its camera
    # acknowledgment a later logical tick, independent of OS clock resolution.
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=advance))

    class Relay:
        def __init__(self, agent_id):
            self.agent_id = agent_id
            self.packet_handlers = []
            self.match_config_handlers = []
            self.field_info_handlers = []
            self.controllable_team_info_handlers = []
            self.is_connected = False
            self.socket = None

        def connect(self, **kwargs):
            self.is_connected = True
            events.append(("connect", self.agent_id, kwargs["close_between_matches"]))
            if self.agent_id:
                assert events[-2] == "match-accepted"
                assert kwargs["close_between_matches"] is True
                for callback in self.match_config_handlers:
                    callback(flat.MatchConfiguration())
                for callback in self.field_info_handlers:
                    callback(flat.FieldInfo())
                for callback in self.controllable_team_info_handlers:
                    callback(
                        flat.ControllableTeamInfo(
                            team=0, controllables=[flat.ControllableInfo(index=0, identifier=22)]
                        )
                    )
                for callback in self.packet_handlers:
                    callback(
                        flat.GamePacket(
                            players=[flat.PlayerInfo(player_id=22)],
                            match_info=flat.MatchInfo(match_phase=flat.MatchPhase.Active),
                        )
                    )

        def send_msg(self, message):
            events.append(type(message).__name__)
            if isinstance(message, flat.ConsoleCommand):
                pending_console.append(message.command)
            if isinstance(message, flat.DesiredGameState) and pending_console:
                pending_console.clear()
                advance(0.02)
                backend._packet(
                    flat.GamePacket(
                        players=[flat.PlayerInfo(player_id=22)],
                        match_info=flat.MatchInfo(
                            match_phase=flat.MatchPhase.Active, last_spectated=0
                        ),
                    )
                )

        def stop_match(self, shutdown_server):
            events.append(("stop", shutdown_server))

    class Manager:
        def __init__(self, path):
            pass

        def _packet_reporter(self, packet):
            pass

        def connect(self, **kwargs):
            self.rlbot_interface.connect(**kwargs)

        def start_match(self, path, **kwargs):
            assert kwargs == {"wait_for_start": False, "ensure_server_started": False}
            for callback in self.rlbot_interface.match_config_handlers:
                callback(flat.MatchConfiguration())
            events.append("match-accepted")

    def relay_factory(agent_id, timeout):
        relay = Relay(agent_id)
        relays.append(relay)
        return relay

    backend = RLBotBackend({}, tmp_path)

    def launch(path):
        backend._core = SimpleNamespace(poll=lambda: None, wait=lambda timeout: None)
        return 23234

    monkeypatch.setattr(module, "_isolated_relay", relay_factory)
    monkeypatch.setattr(module, "verify_offline_processes", lambda: [42])
    monkeypatch.setattr(module, "find_server_path", lambda _: tmp_path / "RLBotServer.exe")
    monkeypatch.setattr(match_module, "MatchManager", Manager)
    monkeypatch.setattr(backend, "_launch_core", launch)
    monkeypatch.setattr(backend, "_thread", lambda *args: None)
    backend.start()
    assert events[0] == ("connect", "", False)
    assert len(relays) == 2
    assert backend.snapshot()["ready"]
    assert backend._last_packet_at > backend._last_camera_command
    assert "DesiredGameState" in events and "ConsoleCommand" in events
    backend.close()
    assert ("stop", True) in events
