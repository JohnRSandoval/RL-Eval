import json

from rleval.live_view import EventTail, LiveState, ranked_probabilities, request_stop


def test_tail_waits_for_newline_and_handles_split_utf8_without_duplicates(tmp_path):
    path = tmp_path / "events.jsonl"
    tail = EventTail(path)
    assert tail.poll() == []
    record = json.dumps({"type": "decision", "note": "→"}, ensure_ascii=False).encode()
    split = record.index("→".encode()) + 1
    path.write_bytes(record[:split])
    assert tail.poll() == []
    with path.open("ab") as f:
        f.write(record[split:] + b"\n")
    assert tail.poll() == [{"type": "decision", "note": "→"}]
    assert tail.poll() == []


def test_tail_ignores_bad_records_and_recovers_from_truncation(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'broken\n[]\n{"type":"telemetry"}\n')
    tail = EventTail(path)
    assert tail.poll() == [{"type": "telemetry"}]
    path.write_bytes(b'{"type":"finished"}\n')
    assert tail.poll() == [{"type": "finished"}]


def test_tail_reads_bounded_chunks_and_eventually_finishes(tmp_path):
    path = tmp_path / "events.jsonl"
    records = [{"type": "request", "number": i} for i in range(10)]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    tail = EventTail(path, read_limit=17)
    seen = []
    for _ in range(100):
        seen.extend(tail.poll())
    assert seen == records


def test_live_state_tracks_outputs_without_retaining_request_context_or_memory():
    state = LiveState()
    state.update({"type": "request", "elapsed_s": 1, "context": {"secret": "excluded"}})
    state.update(
        {
            "type": "decision",
            "number": 1,
            "elapsed_s": 1.2,
            "accepted": True,
            "latency_s": 0.2,
            "model_decision": {
                "choice": "boost_forward",
                "confidence": 0.8,
                "probabilities": {"forward": 0.2, "boost_forward": 0.8},
            },
            "action": {"controls": {"boost": True}, "hold_ms": 300, "memory": "excluded"},
        }
    )
    state.update({"type": "decision", "elapsed_s": 1.5, "accepted": False, "error": "timeout"})
    assert (state.requests, state.completed, state.accepted) == (1, 2, 1)
    assert state.latency_total == 0.2
    assert len(state.recent) == 2
    assert state.decision["error"] == "timeout"
    assert "excluded" not in repr(vars(state))


def test_summary_supplies_final_score_and_finishes():
    state = LiveState()
    state.load_summary(
        {
            "status": "match_ended",
            "elapsed_s": 330,
            "final_telemetry": {"scores": {"blue": 1, "orange": 2}, "touches": 17},
        }
    )
    assert state.finished
    assert state.elapsed_s == 330
    assert state.telemetry["scores"] == {"blue": 1, "orange": 2}
    assert state.status == "Finished: match_ended"


def test_probability_rankings_exclude_invalid_values():
    assert ranked_probabilities(
        {"a": 0.2, "b": 0.8, "c": float("nan"), "d": -1, "e": True, "f": "1"}
    ) == [("b", 0.8), ("a", 0.2)]


def test_planner_outputs_are_separate_and_rejection_keeps_accepted_plan():
    state = LiveState()
    state.update(
        {"type": "planner_request", "number": 1, "elapsed_s": 1, "context": {"secret": "excluded"}}
    )
    state.update(
        {
            "type": "planner_decision",
            "number": 1,
            "elapsed_s": 3,
            "latency_s": 2,
            "accepted": True,
            "resolved_model": "gpt-6.1-sol",
            "plan": {
                "objective": "Defend",
                "target_position": {"x": 0, "y": -4000, "z": 17},
                "desired_speed": 1400,
                "boost_policy": "conserve",
                "jump_policy": "grounded",
                "guidance": "Return between the ball and the blue goal",
                "unknown": "excluded",
            },
        }
    )
    assert (state.requests, state.completed, state.accepted) == (0, 0, 0)
    assert (state.planner_requests, state.planner_completed, state.planner_accepted) == (1, 1, 1)
    assert state.plan_version == 1
    assert state.plan_received_at == 3
    assert state.planner_model == "gpt-6.1-sol"
    assert "excluded" not in repr(vars(state))
    state.update({"type": "planner_request", "number": 2, "elapsed_s": 5})
    state.update(
        {
            "type": "planner_decision",
            "number": 2,
            "elapsed_s": 6,
            "accepted": False,
            "error": "timeout",
        }
    )
    assert state.plan["objective"] == "Defend"
    assert state.plan_version == 1
    assert "timeout" in state.planner_status


def test_invalidated_plan_is_removed_instead_of_displayed_as_current():
    state = LiveState()
    state.update(
        {
            "type": "planner_decision",
            "number": 1,
            "elapsed_s": 2,
            "accepted": True,
            "plan": {"objective": "Attack"},
        }
    )
    state.update({"type": "planner_invalidated", "elapsed_s": 12, "reason": "goal reset"})
    assert state.plan == {}
    assert state.plan_version is None
    assert state.plan_received_at is None
    assert state.planner_accepted == 1
    assert "goal reset" in state.planner_status


def test_local_motor_samples_do_not_become_model_decisions_or_confidence():
    state = LiveState()
    state.update(
        {
            "type": "motor_sample",
            "elapsed_s": 0.1,
            "controls": {"throttle": 1, "boost": True},
            "target": {"x": 0, "y": 1000, "z": 17},
            "skill": "drive_to_target",
            "status": "driving",
            "metrics": {"heading_error": 0.1},
            "plan_version": 3,
            "policy_source": "planner",
            "tick_count": 6,
        }
    )
    state.update(
        {
            "type": "motor_summary",
            "elapsed_s": 1,
            "worker_elapsed_s": 0.99,
            "ticks": 60,
            "active_ticks": 55,
            "mean_compute_ms": 0.2,
            "max_compute_ms": 1.2,
            "achieved_hz": 60.6,
            "source_counts": {"planner": 50, "fallback": 10},
            "errors": 0,
        }
    )
    assert state.motor_samples == 1
    assert (state.requests, state.completed, state.accepted, state.latency_total) == (0, 0, 0, 0)
    assert state.decision == {}
    assert "confidence" not in state.motor
    assert state.motor["policy_source"] == "planner"
    assert state.motor["plan_version"] == 3
    assert state.motor_summary["ticks"] == 60
    assert state.motor_summary["achieved_hz"] == 60.6


def test_stop_button_signal_is_idempotent_and_does_not_claim_completion(tmp_path):
    request_stop(tmp_path)
    signal = tmp_path / "stop.request"
    original = signal.read_text()
    request_stop(tmp_path)
    assert signal.read_text() == original
    assert list(tmp_path.iterdir()) == [signal]
