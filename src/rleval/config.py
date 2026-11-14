"""Configuration is serializable and contains environment-variable names, never credentials."""

import math
import tomllib
from copy import deepcopy
from pathlib import Path

DEFAULTS = {
    "model": {
        "provider": "mock",
        "model": "deterministic-smoke",
        "timeout_s": 15.0,
        "max_output_tokens": 2048,
    },
    "planner": {
        "enabled": False,
        "provider": "openai",
        "model": "gpt-6.1-sol",
        "reasoning_effort": "low",
        "max_output_tokens": 2048,
        "timeout_s": 15.0,
        "interval_s": 5.0,
        "max_age_s": 20.0,
        "max_requests": 300,
    },
    "episode": {
        "duration_s": 60.0,
        "max_decisions": 50,
        "max_errors": 3,
        "max_observation_age_s": 15.0,
        "decision_interval_s": 0.25,
        "task": "Learn to control your blue car. Score in the orange goal and prevent goals in your own goal.",
    },
    "capture": {
        "window_title": "Rocket League",
        "max_width": 960,
        "jpeg_quality": 80,
        "frame_stack": 3,
        "sample_hz": 5.0,
        "require_foreground": True,
    },
    "observation": {"mode": "pixels", "representation": "full"},
    "control": {
        "mode": "model",
        "scheduling": "bounded",
        "max_decision_hz": 10.0,
        "motor_hz": 60.0,
        "motor_max_packet_age_s": 0.1,
    },
    "rlbot": {"launcher": "Epic", "startup_timeout_s": 90.0, "opponent_skill": "Rookie"},
    "recording": {
        "output_dir": "runs",
        "save_frames": True,
        "telemetry_hz": 5.0,
        "live_window": False,
    },
}

PROVIDER_DEFAULT_MODELS = {
    "mock": "deterministic-smoke",
    "openai-decisions": "gpt-6-luna",
    "jev": "jev-latest",
}


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    config = deepcopy(DEFAULTS)
    if path:
        with Path(path).open("rb") as f:
            data = tomllib.load(f)
        for section, values in data.items():
            if section not in config or not isinstance(values, dict):
                raise ValueError(f"Unknown or invalid config section: {section}")
            config[section].update(values)
        if "provider" in data.get("model", {}) and "model" not in data["model"]:
            config["model"]["model"] = PROVIDER_DEFAULT_MODELS.get(config["model"]["provider"], "")
    model_overrides = (overrides or {}).get("model", {})
    new_provider = model_overrides.get("provider")
    if new_provider and new_provider != config["model"]["provider"]:
        # Never carry one provider's credential selector or endpoint into another.
        for key in ("api_key_env", "base_url", "api_version", "reasoning_effort"):
            config["model"].pop(key, None)
        if model_overrides.get("model") is None:
            config["model"]["model"] = PROVIDER_DEFAULT_MODELS.get(new_provider, "")
    for section, values in (overrides or {}).items():
        config[section].update({k: v for k, v in values.items() if v is not None})
    if config["observation"]["mode"] not in {"pixels", "pixels_and_state", "state"}:
        raise ValueError("observation.mode must be pixels, pixels_and_state, or state")
    if config["observation"]["representation"] not in {"full", "compact"}:
        raise ValueError("observation.representation must be full or compact")
    if (
        config["observation"]["representation"] == "compact"
        and config["observation"]["mode"] != "state"
    ):
        raise ValueError("Compact motor observations require state mode")
    if config["control"]["scheduling"] not in {"bounded", "continuous"}:
        raise ValueError("control.scheduling must be bounded or continuous")
    if config["control"]["mode"] not in {"model", "local_motor"}:
        raise ValueError("control.mode must be model or local_motor")
    if config["control"]["mode"] == "local_motor":
        if config["observation"]["mode"] != "state":
            raise ValueError("Local motor assistance requires explicit state observation mode")
        config["model"].update(provider="local-motor", model="local-motor-v1")
        config["planner"]["control_mode"] = "local_motor"
    config["rlbot"]["require_camera"] = config["observation"]["mode"] != "state"
    if type(config["recording"]["live_window"]) is not bool:
        raise ValueError("recording.live_window must be boolean")
    if type(config["planner"]["enabled"]) is not bool:
        raise ValueError("planner.enabled must be boolean")
    if config["planner"]["enabled"] and config["observation"]["mode"] == "pixels":
        raise ValueError("The tactical planner requires state or pixels_and_state observations")
    for section, values in config.items():
        if any(k.lower() in {"api_key", "token", "secret", "authorization"} for k in values):
            raise ValueError(f"Do not put credentials in [{section}]; use api_key_env")
    for section, key, low, high in [
        ("model", "timeout_s", 0.1, 300),
        ("model", "max_output_tokens", 128, 32768),
        ("episode", "duration_s", 0.1, 86400),
        ("episode", "max_decisions", 1, 100000),
        ("episode", "max_errors", 1, 100),
        ("episode", "max_observation_age_s", 0.1, 300),
        ("episode", "decision_interval_s", 0.05, 60),
        ("capture", "max_width", 160, 3840),
        ("capture", "jpeg_quality", 30, 95),
        ("capture", "frame_stack", 1, 8),
        ("capture", "sample_hz", 1, 30),
        ("recording", "telemetry_hz", 1, 30),
        ("rlbot", "startup_timeout_s", 1, 300),
        ("control", "max_decision_hz", 0.1, 30),
        ("control", "motor_hz", 30, 120),
        ("control", "motor_max_packet_age_s", 0.02, 0.25),
        ("planner", "interval_s", 1, 300),
        ("planner", "max_age_s", 1, 300),
        ("planner", "max_requests", 1, 1000),
        ("planner", "timeout_s", 0.1, 300),
        ("planner", "max_output_tokens", 128, 32768),
    ]:
        value = config[section][key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{section}.{key} must be between {low} and {high}")
    for section, key in [
        ("model", "max_output_tokens"),
        ("episode", "max_decisions"),
        ("episode", "max_errors"),
        ("capture", "max_width"),
        ("capture", "jpeg_quality"),
        ("capture", "frame_stack"),
        ("planner", "max_requests"),
        ("planner", "max_output_tokens"),
    ]:
        if type(config[section][key]) is not int:
            raise ValueError(f"{section}.{key} must be an integer")
    if not str(config["model"].get("model", "")).strip():
        raise ValueError(
            "Set model.model or --model to an exact model ID supported by this provider"
        )
    return config
