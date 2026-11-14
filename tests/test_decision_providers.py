"""Exact typed-decision HTTP contracts; all requests stay in MockTransport."""

import base64
import copy
import io
import json

import httpx
import pytest
from PIL import Image

from rleval.actions import NEUTRAL, parse_action
from rleval.decision_providers import (
    ACTION_CATALOG,
    CATALOG_VERSION,
    GROUND_AXES,
    GROUND_CATALOG_VERSION,
    QUESTION_NAME,
    build_decision_client,
)
from rleval.providers import ProviderError, build_client

STATE_CONTEXT = {
    "task": "Score in the orange goal.",
    "observation_mode": "state",
    "game_state": {
        "player": {"position": {"x": 0, "y": -2000, "z": 17}, "boost": 33},
        "ball": {"position": {"x": 0, "y": 0, "z": 93}},
        "relative_ball": {"position": {"forward": 2000, "right": 0, "up": 76}},
    },
}


def jpeg(width=640, height=360, color="navy"):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def response(provider, choice="forward_left", **answer_overrides):
    probabilities = {key: (1.0 if key == choice else 0.0) for key in ACTION_CATALOG}
    answer = {"type": "choice", "choice": choice, "confidence": 0.8}
    if provider == "openai-decisions":
        answer.update(
            name=QUESTION_NAME,
            probabilities=[
                {"value": key, "probability": value} for key, value in probabilities.items()
            ],
        )
    else:
        answer["probabilities"] = probabilities
    answer.update(answer_overrides)
    return {
        "model": "resolved-test-model",
        "answers": [answer] if provider == "openai-decisions" else {QUESTION_NAME: answer},
        "usage": {
            "input_tokens": 123,
            "output_tokens": 0,
            "total_tokens": 123,
            "input_tokens_details": {"cached_tokens": 10},
        },
    }


def client_for(monkeypatch, provider, data=None, *, handler=None, **config):
    requests = []
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-secret")
    monkeypatch.setenv("JEV_API_KEY", "test-jev-secret")

    def handle(request):
        requests.append(request)
        if handler:
            return handler(request)
        return httpx.Response(200, json=data or response(provider))

    http = httpx.Client(transport=httpx.MockTransport(handle))
    client = build_decision_client({"provider": provider, **config}, http_client=http)
    return client, http, requests


def test_openai_state_request_uses_one_joint_choice_and_preserves_distribution(monkeypatch):
    client, http, requests = client_for(monkeypatch, "openai-decisions", decision_hold_ms=150)
    with http:
        reply = client.decide([], STATE_CONTEXT)
    request = requests[0]
    payload = json.loads(request.content)
    assert str(request.url) == "https://api.openai.com/v1/decisions"
    assert request.headers["authorization"] == "Bearer test-openai-secret"
    assert payload["model"] == "gpt-6-luna"
    assert set(payload) == {"model", "input", "questions"}
    assert len(payload["questions"]) == 1
    question = payload["questions"][0]
    assert question["type"] == "choice"
    assert question["name"] == QUESTION_NAME
    assert [item["value"] for item in question["choices"]] == list(ACTION_CATALOG)
    assert payload["input"][0]["content"] == [
        {
            "type": "input_json",
            "data": {"observation": STATE_CONTEXT, "frames_order": "oldest_first"},
        }
    ]
    assert reply.action == {
        "controls": ACTION_CATALOG["forward_left"]["controls"],
        "hold_ms": 150,
        "memory": "",
    }
    assert reply.decision["choice"] == "forward_left"
    assert reply.decision["confidence"] == 0.8
    assert reply.decision["probabilities"]["forward_left"] == 1.0
    assert reply.decision["resolved_model"] == "resolved-test-model"
    assert reply.decision["catalog_version"] == CATALOG_VERSION
    assert reply.usage["cached_input_tokens"] == 10
    assert reply.usage["total_tokens"] == 123
    parse_action(reply.action)


def test_openai_combines_structured_state_with_chronological_inline_images(monkeypatch):
    frames = [jpeg(color="blue"), jpeg(color="orange")]
    context = dict(STATE_CONTEXT, observation_mode="pixels_and_state")
    client, http, requests = client_for(monkeypatch, "openai-decisions", model="explicit-model")
    with http:
        client.decide(frames, context)
    payload = json.loads(requests[0].content)
    assert payload["model"] == "explicit-model"
    content = payload["input"][0]["content"]
    assert content[0]["data"]["observation"]["game_state"] == STATE_CONTEXT["game_state"]
    assert [base64.b64decode(part["image_url"].split(",")[1]) for part in content[1:]] == frames


def test_jev_state_request_uses_named_criteria_without_vision_flag(monkeypatch):
    client, http, requests = client_for(monkeypatch, "jev")
    with http:
        reply = client.decide([], STATE_CONTEXT)
    request = requests[0]
    payload = json.loads(request.content)
    assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["authorization"] == "Bearer test-jev-secret"
    assert "TypeSafe-Danger-Zone-Images" not in request.headers
    assert payload["model"] == "jev-latest"
    assert "assets" not in payload
    assert payload["state"]["observation"] == STATE_CONTEXT
    assert set(payload["questions"]) == {QUESTION_NAME}
    question = payload["questions"][QUESTION_NAME]
    assert question["type"] == "choice"
    assert question["criteria"] == {
        key: entry["description"] for key, entry in ACTION_CATALOG.items()
    }
    assert reply.action["hold_ms"] == 200
    assert reply.decision["resolved_model"] == "resolved-test-model"


def test_jev_images_are_referenced_once_and_ordered(monkeypatch):
    frames = [jpeg(color="red"), jpeg(color="green")]
    context = dict(STATE_CONTEXT, observation_mode="pixels_and_state")
    client, http, requests = client_for(monkeypatch, "jev", model="jev-1.14.0")
    with http:
        client.decide(frames, context)
    request = requests[0]
    payload = json.loads(request.content)
    assert request.headers["TypeSafe-Danger-Zone-Images"] == "true"
    references = payload["state"]["frames"]
    assert len(references) == len(set(references)) == 2
    assert set(references) == set(payload["assets"])
    assert [base64.b64decode(payload["assets"][ref]["data_base64"]) for ref in references] == frames
    assert all(payload["assets"][ref]["format"] == "jpeg" for ref in references)
    assert payload["state"]["frames_order"] == "oldest_first"
    assert "frames" not in context  # adapter did not mutate the runner's context


@pytest.mark.parametrize("provider", ["openai-decisions", "jev"])
@pytest.mark.parametrize("explicit_mode", [True, False])
def test_pixels_mode_sends_only_context_and_ordered_screenshots(
    monkeypatch, provider, explicit_mode
):
    """Public factory accepts the runner's canonical mode, including its default."""
    frames = [jpeg(color="cyan"), jpeg(color="orange")]
    context = {"task": "Score a goal", "previous_actions": [], "memory": ""}
    if explicit_mode:
        context["observation_mode"] = "pixels"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=response(provider))

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        client = build_client({"provider": provider}, http_client=http)
        reply = client.decide(frames, context)
    assert reply.action["controls"] == ACTION_CATALOG["forward_left"]["controls"]
    payload = json.loads(requests[0].content)
    if provider == "openai-decisions":
        content = payload["input"][0]["content"]
        observation = content[0]["data"]["observation"]
        sent_frames = [base64.b64decode(part["image_url"].split(",")[1]) for part in content[1:]]
    else:
        observation = payload["state"]["observation"]
        sent_frames = [
            base64.b64decode(payload["assets"][ref]["data_base64"])
            for ref in payload["state"]["frames"]
        ]
    assert observation == context
    assert "game_state" not in observation
    assert sent_frames == frames


@pytest.mark.parametrize("provider", ["openai-decisions", "jev"])
@pytest.mark.parametrize(
    "changes",
    [
        {"type": "refusal"},
        {"type": "score"},
        {"choice": "teleport_to_goal"},
        {"choice": True},
        {"confidence": -0.1},
        {"confidence": 1.1},
        {"confidence": True},
        {"confidence": "0.8"},
        {"probabilities": {}},
    ],
)
def test_invalid_or_refused_answers_never_become_controls(monkeypatch, provider, changes):
    client, http, _ = client_for(monkeypatch, provider, response(provider, **changes))
    with http, pytest.raises(ProviderError):
        client.decide([], STATE_CONTEXT)


@pytest.mark.parametrize("provider", ["openai-decisions", "jev"])
def test_invalid_distribution_is_rejected(monkeypatch, provider):
    data = response(provider)
    answer = (
        data["answers"][0] if provider == "openai-decisions" else data["answers"][QUESTION_NAME]
    )
    if provider == "openai-decisions":
        answer["probabilities"][0]["probability"] = 0.9
    else:
        answer["probabilities"]["neutral"] = 0.9
    client, http, _ = client_for(monkeypatch, provider, data)
    with http, pytest.raises(ProviderError, match="sum"):
        client.decide([], STATE_CONTEXT)


def test_repeated_openai_probability_name_is_rejected(monkeypatch):
    data = response("openai-decisions")
    data["answers"][0]["probabilities"].append(data["answers"][0]["probabilities"][0])
    client, http, _ = client_for(monkeypatch, "openai-decisions", data)
    with http, pytest.raises(ProviderError, match="repeated"):
        client.decide([], STATE_CONTEXT)


@pytest.mark.parametrize("provider", ["openai-decisions", "jev"])
def test_another_question_cannot_be_treated_as_the_controller_answer(monkeypatch, provider):
    data = response(provider)
    if provider == "openai-decisions":
        data["answers"][0]["name"] = "other_question"
    else:
        data["answers"] = {"other_question": data["answers"][QUESTION_NAME]}
    client, http, _ = client_for(monkeypatch, provider, data)
    with http, pytest.raises(ProviderError, match="question"):
        client.decide([], STATE_CONTEXT)


@pytest.mark.parametrize("provider", ["openai-decisions", "jev"])
@pytest.mark.parametrize(
    "frames,context",
    [
        ([], {"observation_mode": "state"}),
        ([b"unexpected pixels"], STATE_CONTEXT),
        ([], dict(STATE_CONTEXT, observation_mode="pixels_and_state")),
        ([b"pixels"], dict(STATE_CONTEXT, observation_mode="pixels")),
        ([], {"observation_mode": "pixels"}),
        ([b"pixels"], {"observation_mode": "pixels_and_state"}),
        ([], dict(STATE_CONTEXT, observation_mode="unknown")),
        ([], dict(STATE_CONTEXT, elapsed_s=float("nan"))),
    ],
)
def test_observation_mode_boundaries_fail_before_network(monkeypatch, provider, frames, context):
    client, http, requests = client_for(monkeypatch, provider)
    with http, pytest.raises(ProviderError):
        client.decide(frames, context)
    assert not requests


@pytest.mark.parametrize("provider", ["openai-decisions", "jev"])
@pytest.mark.parametrize("hold_ms", [49, 1001, True, 200.5])
def test_hold_duration_is_bounded_and_not_coerced(monkeypatch, provider, hold_ms):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("JEV_API_KEY", "test")
    with pytest.raises(ProviderError, match="decision_hold_ms"):
        build_decision_client({"provider": provider, "decision_hold_ms": hold_ms})


@pytest.mark.parametrize("provider", ["openai-decisions", "jev"])
def test_timeout_does_not_retry_or_log_key(monkeypatch, provider):
    def timeout(request):
        raise httpx.ReadTimeout("test-openai-secret test-jev-secret", request=request)

    client, http, requests = client_for(monkeypatch, provider, handler=timeout)
    with http, pytest.raises(ProviderError) as error:
        client.decide([], STATE_CONTEXT)
    assert len(requests) == 1
    assert "timed out" in str(error.value)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("provider", ["openai-decisions", "jev"])
def test_rate_limit_does_not_retry_or_leak_body(monkeypatch, provider):
    client, http, requests = client_for(
        monkeypatch, provider, handler=lambda _: httpx.Response(429, json={"error": "test-secret"})
    )
    with http, pytest.raises(ProviderError) as error:
        client.decide([], STATE_CONTEXT)
    assert len(requests) == 1
    assert "429" in str(error.value)
    assert "test-secret" not in str(error.value)


def test_all_catalogue_choices_are_distinct_single_valid_vectors():
    assert 2 <= len(ACTION_CATALOG) <= 255
    vectors = set()
    for entry in ACTION_CATALOG.values():
        action = parse_action({"controls": entry["controls"], "hold_ms": 200, "memory": ""})
        assert set(action.controls) == set(NEUTRAL)
        vectors.add(tuple(action.controls.values()))
    assert len(vectors) == len(ACTION_CATALOG)


def test_mutating_a_reply_does_not_change_future_controls(monkeypatch):
    original = copy.deepcopy(ACTION_CATALOG)
    client, http, _ = client_for(monkeypatch, "openai-decisions")
    with http:
        first = client.decide([], STATE_CONTEXT)
        first.action["controls"]["throttle"] = -1
        second = client.decide([], STATE_CONTEXT)
    assert second.action["controls"]["throttle"] == 1
    assert ACTION_CATALOG == original


def test_jev_rejects_small_images_without_resizing_or_calling_api(monkeypatch):
    client, http, requests = client_for(monkeypatch, "jev")
    context = dict(STATE_CONTEXT, observation_mode="pixels_and_state")
    with http, pytest.raises(ProviderError, match="dimensions"):
        client.decide([jpeg(320, 180)], context)
    assert not requests


def test_jev_supports_configurable_key_environment(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "custom-secret")
    client, http, requests = client_for(monkeypatch, "jev", api_key_env="TYPESAFE_API_KEY")
    with http:
        client.decide([], STATE_CONTEXT)
    assert requests[0].headers["authorization"] == "Bearer custom-secret"


def compact_context(speed=500):
    from pathlib import Path

    from rleval.backend import MockBackend
    from rleval.compact import build_motor_observation

    backend = MockBackend({}, Path("."))
    backend.start()
    raw = backend.snapshot()
    raw["car_rotation"] = {"pitch": 0, "yaw": 0, "roll": 0}
    raw["car_velocity"] = {"x": speed, "y": 0, "z": 0}
    return {
        "observation_mode": "state",
        "observation_representation": "compact",
        "motor_observation": build_motor_observation(
            raw, {"x": 1000, "y": -1000, "z": 17}, 500, "Drive to target"
        ),
    }


def axis_response(provider, steering="right", longitudinal="coast"):
    answers = {}
    for name, choice in [("steering", steering), ("longitudinal", longitudinal)]:
        probabilities = {key: float(key == choice) for key in GROUND_AXES[name]}
        answer = {
            "type": "choice",
            "choice": choice,
            "confidence": 0.8,
            "probabilities": probabilities,
        }
        if provider == "openai-decisions":
            answer.update(
                name=name,
                probabilities=[
                    {"value": key, "probability": p} for key, p in probabilities.items()
                ],
            )
        answers[name] = answer
    return {
        "model": "test-resolved",
        "answers": list(reversed(list(answers.values())))
        if provider == "openai-decisions"
        else answers,
        "usage": {"input_tokens": 100, "output_tokens": 0, "total_tokens": 100},
    }


@pytest.mark.parametrize("provider", ["openai-decisions", "jev"])
def test_ground_axes_send_two_small_questions_and_keep_coasting_independent(monkeypatch, provider):
    context = compact_context()
    context["unused_private_context"] = "DO-NOT-SEND"
    client, http, requests = client_for(
        monkeypatch, provider, axis_response(provider), action_interface="ground_axes_v1"
    )
    with http:
        reply = client.decide([], context)
    payload = json.loads(requests[0].content)
    assert "DO-NOT-SEND" not in requests[0].content.decode()
    assert "game_state" not in requests[0].content.decode()
    if provider == "openai-decisions":
        assert {q["name"]: len(q["choices"]) for q in payload["questions"]} == {
            "steering": 3,
            "longitudinal": 3,
        }
    else:
        assert {name: len(q["criteria"]) for name, q in payload["questions"].items()} == {
            "steering": 3,
            "longitudinal": 3,
        }
    assert reply.action["controls"] == dict(NEUTRAL, steer=0.5)
    assert reply.decision["choices"] == {"steering": "right", "longitudinal": "coast"}
    assert reply.decision["catalog_version"] == GROUND_CATALOG_VERSION
    assert reply.decision["question_details"]["longitudinal"]["probabilities"]["coast"] == 1
    assert reply.usage["total_tokens"] == 100


@pytest.mark.parametrize("provider", ["openai-decisions", "jev"])
@pytest.mark.parametrize("speed,throttle", [(500, -1), (-500, 1), (29, 0), (-29, 0), (0, 0)])
def test_braking_opposes_observed_motion_and_stops_near_zero(
    monkeypatch, provider, speed, throttle
):
    client, http, _ = client_for(
        monkeypatch,
        provider,
        axis_response(provider, "left", "brake"),
        action_interface="ground_axes_v1",
    )
    with http:
        reply = client.decide([], compact_context(speed))
    assert reply.action["controls"] == dict(NEUTRAL, steer=-0.5, throttle=throttle)


@pytest.mark.parametrize("provider", ["openai-decisions", "jev"])
def test_ground_accelerate_does_not_force_turning(monkeypatch, provider):
    client, http, _ = client_for(
        monkeypatch,
        provider,
        axis_response(provider, "straight", "accelerate"),
        action_interface="ground_axes_v1",
    )
    with http:
        reply = client.decide([], compact_context())
    assert reply.action["controls"] == dict(NEUTRAL, throttle=1)


@pytest.mark.parametrize("provider", ["openai-decisions", "jev"])
def test_one_refused_axis_rejects_entire_controller_action(monkeypatch, provider):
    data = axis_response(provider)
    if provider == "openai-decisions":
        data["answers"][0]["type"] = "refusal"
    else:
        data["answers"]["longitudinal"]["type"] = "refusal"
    client, http, _ = client_for(monkeypatch, provider, data, action_interface="ground_axes_v1")
    with http, pytest.raises(ProviderError, match="refused"):
        client.decide([], compact_context())


@pytest.mark.parametrize("provider", ["openai-decisions", "jev"])
@pytest.mark.parametrize("fault", ["missing_motor", "raw_state", "pixels", "missing_speed"])
def test_compact_boundaries_fail_before_network(monkeypatch, provider, fault):
    context = compact_context()
    frames = []
    if fault == "missing_motor":
        context.pop("motor_observation")
    elif fault == "raw_state":
        context["game_state"] = STATE_CONTEXT["game_state"]
    elif fault == "pixels":
        frames = [b"jpeg"]
    elif fault == "missing_speed":
        context["motor_observation"]["motion"].pop("forward_speed_uu_s")
    client, http, requests = client_for(
        monkeypatch, provider, axis_response(provider), action_interface="ground_axes_v1"
    )
    with http, pytest.raises(ProviderError):
        client.decide(frames, context)
    assert not requests


def test_repeated_axis_answer_cannot_replace_missing_axis(monkeypatch):
    data = axis_response("openai-decisions")
    data["answers"][1] = copy.deepcopy(data["answers"][0])
    client, http, _ = client_for(
        monkeypatch, "openai-decisions", data, action_interface="ground_axes_v1"
    )
    with http, pytest.raises(ProviderError, match="question"):
        client.decide([], compact_context())
