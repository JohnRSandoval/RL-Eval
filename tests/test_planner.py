import copy
import json

import httpx
import pytest

from rleval.planner import (
    MOTOR_PLANNER_PROMPT,
    MOTOR_PLANNER_SCHEMA,
    MOTOR_SKILLS,
    PLANNER_PROMPT,
    PLANNER_SCHEMA,
    build_planner,
    parse_motor_plan,
    parse_plan,
)
from rleval.providers import ProviderError


def plan():
    return {
        "objective": "Defend the near post",
        "target_position": {"x": 800, "y": -4800, "z": 17},
        "desired_speed": 1200,
        "boost_policy": "conserve",
        "jump_policy": "allowed",
        "guidance": "Stay between the ball and the blue goal; adapt to current ball motion.",
    }


def context():
    return {"observation_mode": "state", "game_state": {"player": {}, "ball": {}}}


def response(**changes):
    return {
        "id": "resp-plan",
        "model": "gpt-6.1-sol",
        "status": "completed",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": json.dumps(plan())}]}
        ],
        "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        **changes,
    }


def client(monkeypatch, handler, **config):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return build_planner(config, http_client=http), http


def test_planner_posts_structured_tactics_and_preserves_evidence(monkeypatch):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        assert request.url == "https://api.openai.com/v1/responses"
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json=response())

    planner, http = client(monkeypatch, handler)
    try:
        reply = planner.decide(context())
        assert reply.plan == plan()
        assert reply.usage["total_tokens"] == 150
        assert reply.response_id == "resp-plan"
        assert reply.resolved_model == "gpt-6.1-sol"
        body = seen[0]
        assert body["model"] == "gpt-6.1-sol"
        assert body["reasoning"] == {"effort": "low"}
        assert body["max_output_tokens"] == 2048 and body["store"] is False
        assert body["text"]["format"]["schema"] == PLANNER_SCHEMA
        assert body["text"]["format"]["strict"] is True
        assert all(part["type"] == "input_text" for part in body["input"][0]["content"])
    finally:
        planner.close()
        assert not http.is_closed  # Caller owns injected HTTP client.
        http.close()


@pytest.mark.parametrize(
    "change",
    [
        {"desired_speed": True},
        {"desired_speed": -1},
        {"desired_speed": 2301},
        {"desired_speed": float("nan")},
        {"desired_speed": float("inf")},
        {"objective": " "},
        {"objective": "a" * 201},
        {"guidance": "a" * 1001},
        {"boost_policy": "always"},
        {"jump_policy": "flip_macro"},
        {"target_position": {"x": 0, "y": 0}},
        {"target_position": {"x": 0, "y": 0, "z": -1}},
        {"target_position": {"x": False, "y": 0, "z": 17}},
        {"throttle": 1},
    ],
)
def test_invalid_plan_never_enters_control_context(change):
    with pytest.raises(ProviderError):
        parse_plan(dict(plan(), **change))


def test_null_waypoint_and_validation_copy_are_supported():
    original = plan()
    parsed = parse_plan(original)
    original["target_position"]["x"] = 500
    assert parsed["target_position"]["x"] == 800
    assert parse_plan(dict(plan(), target_position=None))["target_position"] is None


@pytest.mark.parametrize(
    "raw", ['{"objective": "a", "objective": "b"}', "[]", '{"a":NaN}', "```json {} ```"]
)
def test_plan_json_is_strict(raw):
    with pytest.raises(ProviderError):
        parse_plan(raw)


@pytest.mark.parametrize(
    "data",
    [
        response(status="incomplete"),
        response(output=[{"content": [{"type": "refusal", "refusal": "private content"}]}]),
        response(output=[]),
        response(output=[{"content": [{"type": "output_text", "text": "{}"}]}]),
    ],
)
def test_incomplete_refused_or_malformed_plan_is_discarded(monkeypatch, data):
    planner, http = client(monkeypatch, lambda request: httpx.Response(200, json=data))
    try:
        with pytest.raises(ProviderError) as exc:
            planner.decide(context())
        assert "private content" not in str(exc.value)
    finally:
        http.close()


@pytest.mark.parametrize(
    "observation",
    [
        {"observation_mode": "pixels", "game_state": {"player": {}, "ball": {}}},
        {"observation_mode": "state"},
        {"observation_mode": "state", "game_state": {"player": {}}},
        dict(context(), unexpected=float("nan")),
    ],
)
def test_bad_observation_fails_before_network(monkeypatch, observation):
    def handler(request):
        raise AssertionError("Must not call endpoint")

    planner, http = client(monkeypatch, handler)
    try:
        with pytest.raises(ProviderError):
            planner.decide(copy.deepcopy(observation))
    finally:
        http.close()


def test_http_failures_do_not_log_response_body(monkeypatch):
    planner, http = client(monkeypatch, lambda request: httpx.Response(401, text="secret-body"))
    try:
        with pytest.raises(ProviderError, match="HTTP 401") as exc:
            planner.decide(context())
        assert "secret-body" not in str(exc.value)
    finally:
        http.close()


def motor_plan(**changes):
    return {**plan(), "skill": "drive_to", **changes}


def response_plan(value):
    return response(output=[{"content": [{"type": "output_text", "text": json.dumps(value)}]}])


@pytest.mark.parametrize("skill", MOTOR_SKILLS)
def test_motor_skills_are_explicit_and_preserve_plan_validation(skill):
    original = {**motor_plan(), "skill": skill}
    parsed = parse_motor_plan(json.dumps(original))
    assert parsed == original
    original["target_position"]["x"] = 1
    assert parsed["target_position"]["x"] == 800


@pytest.mark.parametrize("skill", ["intercept", "shoot", "defend", "recover", "stop"])
def test_dynamic_and_non_navigation_motor_skills_allow_null_target(skill):
    value = {**motor_plan(target_position=None), "skill": skill}
    assert parse_motor_plan(value)["target_position"] is None


def test_drive_to_requires_target_and_guidance_never_selects_skill():
    with pytest.raises(ProviderError, match="drive_to requires"):
        parse_motor_plan(motor_plan(target_position=None))
    value = {**motor_plan(), "skill": "stop", "guidance": "shoot intercept drive_to"}
    assert parse_motor_plan(value)["skill"] == "stop"


@pytest.mark.parametrize("skill", [None, True, 1, [], {}, "", "Drive_To", "flip_macro"])
def test_unknown_or_untyped_motor_skill_is_rejected(skill):
    with pytest.raises(ProviderError, match="unknown motor skill"):
        parse_motor_plan({**motor_plan(), "skill": skill})


@pytest.mark.parametrize(
    "change",
    [
        {"desired_speed": True},
        {"desired_speed": float("nan")},
        {"desired_speed": 2301},
        {"objective": " "},
        {"guidance": []},
        {"target_position": {"x": 0, "y": 0, "z": -1}},
        {"target_position": {"x": False, "y": 0, "z": 17}},
        {"boost_policy": "always"},
        {"jump_policy": "flip_macro"},
        {"throttle": 1},
    ],
)
def test_motor_plan_rejects_invalid_common_fields(change):
    with pytest.raises(ProviderError):
        parse_motor_plan({**motor_plan(), **change})


@pytest.mark.parametrize(
    "raw",
    [
        '{"skill":"stop","skill":"shoot"}',
        "[]",
        '{"skill":NaN}',
        "```json {} ```",
        plan(),
    ],
)
def test_motor_plan_json_and_required_skill_are_strict(raw):
    with pytest.raises(ProviderError):
        parse_motor_plan(raw)


def test_motor_schema_does_not_change_legacy_contract():
    assert "skill" not in PLANNER_SCHEMA["required"]
    assert "skill" not in PLANNER_SCHEMA["properties"]
    assert parse_plan(plan()) == plan()
    with pytest.raises(ProviderError, match="missing or unexpected"):
        parse_plan(motor_plan())


def test_local_motor_planner_posts_typed_schema_and_execution_feedback(monkeypatch):
    seen = []
    value = {**motor_plan(target_position=None), "skill": "intercept"}

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=response_plan(value))

    planner, http = client(monkeypatch, handler, control_mode="local_motor")
    observation = dict(
        context(),
        motor_feedback={"skill": "drive_to", "distance_to_target_uu": 700, "stalled": True},
    )
    try:
        reply = planner.decide(observation)
        assert reply.plan == value
        assert reply.resolved_model == "gpt-6.1-sol"
        assert reply.usage["total_tokens"] == 150
        body = seen[0]
        assert body["instructions"] == MOTOR_PLANNER_PROMPT
        assert body["text"]["format"] == {
            "type": "json_schema",
            "name": "motor_tactical_plan",
            "strict": True,
            "schema": MOTOR_PLANNER_SCHEMA,
        }
        sent_context = json.loads(body["input"][0]["content"][0]["text"])
        assert sent_context["motor_feedback"] == observation["motor_feedback"]
        assert body["store"] is False
    finally:
        http.close()


@pytest.mark.parametrize("control_mode", [None, "", "model", "other_mode"])
def test_planner_keeps_legacy_wire_contract_unless_local_motor_selected(monkeypatch, control_mode):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=response())

    planner, http = client(monkeypatch, handler, control_mode=control_mode)
    try:
        assert planner.decide(context()).plan == plan()
        assert seen[0]["instructions"] == PLANNER_PROMPT
        assert seen[0]["text"]["format"]["schema"] == PLANNER_SCHEMA
        assert seen[0]["text"]["format"]["name"] == "tactical_plan"
    finally:
        http.close()


@pytest.mark.parametrize("value", [plan(), motor_plan(target_position=None)])
def test_local_motor_planner_rejects_response_without_executable_skill(monkeypatch, value):
    planner, http = client(
        monkeypatch,
        lambda request: httpx.Response(200, json=response_plan(value)),
        control_mode="local_motor",
    )
    try:
        with pytest.raises(ProviderError):
            planner.decide(context())
    finally:
        http.close()
