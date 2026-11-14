"""Request/response contract tests use httpx.MockTransport; no paid API calls."""

import base64
import json

import httpx
import pytest

from rleval.providers import ProviderError, build_client

ACTION = {
    "controls": {
        "throttle": 1.0,
        "steer": -0.2,
        "pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
        "jump": False,
        "boost": True,
        "handbrake": False,
    },
    "hold_ms": 250,
    "memory": "Keep facing the ball.",
}
FRAMES = [b"first-jpeg", b"second-jpeg"]
CONTEXT = {"task": "Score a goal", "memory": "A previous note", "elapsed_s": 1.2}
GAME_STATE = {
    "player": {"position": {"x": 100, "y": -1000, "z": 17}},
    "ball": {"position": {"x": 0, "y": 0, "z": 93}},
}


def transport_client(response: dict, *, status: int = 200):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(status, json=response)

    return httpx.Client(transport=httpx.MockTransport(handle)), requests


def openai_response(**overrides):
    data = {
        "id": "resp-1",
        "status": "completed",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": json.dumps(ACTION)}]}
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "output_tokens_details": {"reasoning_tokens": 20},
        },
    }
    data.update(overrides)
    return data


def test_openai_sends_pixels_in_order_and_constrained_action(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-for-test")
    http, requests = transport_client(openai_response())
    adapter = build_client(
        {"provider": "openai", "model": "test-vision", "reasoning_effort": "low"}, http_client=http
    )
    reply = adapter.decide(FRAMES, CONTEXT)
    request = requests[0]
    payload = json.loads(request.content)
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert request.headers["authorization"] == "Bearer secret-for-test"
    assert payload["store"] is False
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["text"]["format"]["strict"] is True
    content = payload["input"][0]["content"]
    assert [block["image_url"].split(",")[1] for block in content[1:]] == [
        base64.b64encode(frame).decode() for frame in FRAMES
    ]
    assert "A previous note" in content[0]["text"]
    assert reply.action == ACTION
    assert reply.usage["reasoning_tokens"] == 20
    assert reply.response_id == "resp-1"
    adapter.close()
    assert not http.is_closed  # injected client belongs to caller
    http.close()


@pytest.mark.parametrize("mode", ["pixels", "pixels_and_state", "state"])
def test_openai_mode_controls_exact_state_and_image_inputs(monkeypatch, mode):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    context = dict(CONTEXT, observation_mode=mode)
    if mode != "pixels":
        context["game_state"] = GAME_STATE
    frames = [] if mode == "state" else FRAMES
    http, requests = transport_client(openai_response())
    with http:
        adapter = build_client({"provider": "openai", "model": "test"}, http_client=http)
        adapter.decide(frames, context)
    content = json.loads(requests[0].content)["input"][0]["content"]
    assert len(content) == len(frames) + 1
    assert ("game_state" in content[0]["text"]) == (mode != "pixels")
    assert [base64.b64decode(item["image_url"].split(",")[1]) for item in content[1:]] == frames


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "openai-compatible", "mock"])
@pytest.mark.parametrize(
    "frames,context",
    [
        (FRAMES, dict(CONTEXT, observation_mode="pixels", game_state=GAME_STATE)),
        (FRAMES, dict(CONTEXT, game_state=GAME_STATE)),
        ([], dict(CONTEXT, observation_mode="pixels")),
        (FRAMES, dict(CONTEXT, observation_mode="state", game_state=GAME_STATE)),
        ([], dict(CONTEXT, observation_mode="state")),
        (FRAMES, dict(CONTEXT, observation_mode="pixels_and_state")),
        (FRAMES, dict(CONTEXT, observation_mode="pixels_and_state", game_state={})),
        (FRAMES, dict(CONTEXT, observation_mode="unknown")),
        (FRAMES, "not an observation object"),
    ],
)
def test_all_generic_adapters_enforce_observation_boundary_before_network(
    monkeypatch, provider, frames, context
):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "test-key")
    http, requests = transport_client({})
    with http, pytest.raises(ProviderError):
        build_client(
            {"provider": provider, "model": "test", "api_key_env": "TEST_PROVIDER_KEY"},
            http_client=http,
        ).decide(frames, context)
    assert not requests


@pytest.mark.parametrize(
    "response",
    [
        openai_response(status="incomplete"),
        openai_response(output=[{"content": [{"type": "refusal", "refusal": "No"}]}]),
        openai_response(output=[]),
        openai_response(
            output=[{"content": [{"type": "output_text", "text": '{"hold_ms": NaN}'}]}]
        ),
        openai_response(
            output=[
                {"content": [{"type": "output_text", "text": '{"hold_ms": 50,"hold_ms": 100}'}]}
            ]
        ),
    ],
)
def test_openai_discards_refused_truncated_or_ambiguous_actions(monkeypatch, response):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-for-test")
    http, _ = transport_client(response)
    with http, pytest.raises(ProviderError):
        build_client({"provider": "openai", "model": "test"}, http_client=http).decide(
            FRAMES, CONTEXT
        )


def test_anthropic_calls_action_tool_and_preserves_reasoning(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    response = {
        "id": "msg-1",
        "stop_reason": "tool_use",
        "content": [
            {"type": "thinking", "thinking": "internal"},
            {"type": "tool_use", "name": "submit_action", "input": ACTION},
        ],
        "usage": {
            "input_tokens": 10,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 30,
            "output_tokens": 40,
        },
    }
    http, requests = transport_client(response)
    with http:
        reply = build_client(
            {"provider": "anthropic", "model": "test", "reasoning_effort": "low"}, http_client=http
        ).decide(FRAMES, CONTEXT)
    payload = json.loads(requests[0].content)
    assert str(requests[0].url) == "https://api.anthropic.com/v1/messages"
    assert requests[0].headers["authorization"] == "Bearer anthropic-secret"
    assert payload["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["tools"][0]["name"] == "submit_action"
    assert payload["messages"][0]["content"][1]["source"]["media_type"] == "image/jpeg"
    assert reply.action == ACTION
    assert reply.usage["input_tokens"] == 60
    assert reply.usage["total_tokens"] == 100


@pytest.mark.parametrize(
    "response",
    [
        {
            "stop_reason": "max_tokens",
            "content": [{"type": "tool_use", "name": "submit_action", "input": ACTION}],
        },
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": json.dumps(ACTION)}]},
        {
            "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "name": "wrong_tool", "input": ACTION}],
        },
        {
            "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "name": "submit_action", "input": ACTION}] * 2,
        },
    ],
)
def test_anthropic_rejects_missing_wrong_multiple_and_incomplete_calls(monkeypatch, response):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    http, _ = transport_client(response)
    with http, pytest.raises(ProviderError):
        build_client({"provider": "anthropic", "model": "test"}, http_client=http).decide(
            FRAMES, CONTEXT
        )


def test_gemini_sends_schema_and_ignores_thought_text(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    response = {
        "responseId": "gem-1",
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {"thought": True, "text": "private thought"},
                        {"text": json.dumps(ACTION)},
                    ]
                },
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 20,
            "thoughtsTokenCount": 30,
            "totalTokenCount": 150,
        },
    }
    http, requests = transport_client(response)
    with http:
        reply = build_client(
            {"provider": "gemini", "model": "models/test", "reasoning_effort": "low"},
            http_client=http,
        ).decide(FRAMES, CONTEXT)
    request = requests[0]
    payload = json.loads(request.content)
    assert (
        str(request.url)
        == "https://generativelanguage.googleapis.com/v1beta/models/test:generateContent"
    )
    assert "gemini-secret" not in str(request.url)
    assert request.headers["x-goog-api-key"] == "gemini-secret"
    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "LOW"}
    assert payload["generationConfig"]["responseJsonSchema"]["additionalProperties"] is False
    assert payload["contents"][0]["parts"][1]["inlineData"]["mimeType"] == "image/jpeg"
    assert reply.action == ACTION
    assert reply.usage["output_tokens"] == 50


@pytest.mark.parametrize(
    "response",
    [
        {"promptFeedback": {"blockReason": "SAFETY"}},
        {
            "candidates": [
                {"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": json.dumps(ACTION)}]}}
            ]
        },
        {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "not JSON"}]}}]},
    ],
)
def test_gemini_rejects_blocked_truncated_and_invalid_actions(monkeypatch, response):
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    http, _ = transport_client(response)
    with http, pytest.raises(ProviderError):
        build_client({"provider": "gemini", "model": "test"}, http_client=http).decide(
            FRAMES, CONTEXT
        )


def test_local_compatible_endpoint_needs_no_key_and_uses_vision_schema():
    response = {
        "id": "local-1",
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(ACTION)}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }
    http, requests = transport_client(response)
    with http:
        reply = build_client(
            {"provider": "openai-compatible", "model": "local-vision"}, http_client=http
        ).decide(FRAMES, CONTEXT)
    assert "authorization" not in requests[0].headers
    payload = json.loads(requests[0].content)
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["messages"][1]["content"][1]["type"] == "image_url"
    assert reply.action == ACTION
    assert reply.usage["total_tokens"] == 30


def test_provider_errors_do_not_leak_secrets_or_retry(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "my-sensitive-key")
    http, requests = transport_client(
        {"error": "my-sensitive-key embedded in server body"}, status=401
    )
    with http, pytest.raises(ProviderError) as error:
        build_client({"provider": "openai", "model": "test"}, http_client=http).decide(
            FRAMES, CONTEXT
        )
    assert "401" in str(error.value)
    assert "my-sensitive-key" not in str(error.value)
    assert len(requests) == 1


def test_timeout_is_sanitized_and_not_retried(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "my-sensitive-key")
    requests = []

    def handle(request):
        requests.append(request)
        raise httpx.ReadTimeout("request with my-sensitive-key", request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(handle)) as http,
        pytest.raises(ProviderError, match="timed out") as error,
    ):
        build_client({"provider": "openai", "model": "test"}, http_client=http).decide(
            FRAMES, CONTEXT
        )
    assert len(requests) == 1
    assert "my-sensitive-key" not in str(error.value)


def test_redirect_is_not_followed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "https://elsewhere.example/steal"})

    with (
        httpx.Client(transport=httpx.MockTransport(handle), follow_redirects=True) as http,
        pytest.raises(ProviderError, match="302"),
    ):
        build_client({"provider": "openai", "model": "test"}, http_client=http).decide(
            FRAMES, CONTEXT
        )
    assert len(requests) == 1


def test_missing_key_and_invalid_endpoint_fail_before_requests(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        build_client({"provider": "openai", "model": "test"})
    with pytest.raises(ProviderError, match="HTTPS"):
        build_client(
            {
                "provider": "openai-compatible",
                "model": "test",
                "base_url": "http://remote.example/v1",
            }
        )
    with pytest.raises(ProviderError, match="credentials"):
        build_client(
            {
                "provider": "openai-compatible",
                "model": "test",
                "base_url": "https://secret@remote.example/v1",
            }
        )


def test_mock_is_deterministic_and_no_credentials_are_needed():
    first, second = build_client({"provider": "mock"}), build_client({"provider": "mock"})
    assert [first.decide(FRAMES, CONTEXT) for _ in range(10)] == [
        second.decide(FRAMES, CONTEXT) for _ in range(10)
    ]
    assert first.decide(FRAMES, CONTEXT).usage["total_tokens"] == 0


def test_empty_frames_cannot_trigger_api_request(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    http, requests = transport_client(openai_response())
    with http, pytest.raises(ProviderError, match="screenshot"):
        build_client({"provider": "openai", "model": "test"}, http_client=http).decide([], CONTEXT)
    assert not requests
