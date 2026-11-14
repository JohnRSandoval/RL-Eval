import pytest

from rleval.config import load_config


def test_provider_swap_drops_old_credentials_and_endpoint(tmp_path):
    config_file = tmp_path / "model.toml"
    config_file.write_text(
        '[model]\nprovider="openai"\nmodel="old"\napi_key_env="OPENAI_API_KEY"\nbase_url="https://api.openai.com/v1"\nreasoning_effort="low"\n'
    )
    result = load_config(config_file, {"model": {"provider": "anthropic", "model": "new"}})
    assert result["model"]["provider"] == "anthropic"
    for key in ("api_key_env", "base_url", "reasoning_effort"):
        assert key not in result["model"]


def test_provider_swap_preserves_explicit_new_endpoint_and_key(tmp_path):
    config_file = tmp_path / "model.toml"
    config_file.write_text(
        '[model]\nprovider="openai"\nmodel="old"\napi_key_env="OPENAI_API_KEY"\n'
    )
    result = load_config(
        config_file,
        {
            "model": {
                "provider": "openai-compatible",
                "model": "new",
                "base_url": "http://localhost:8000/v1",
                "api_key_env": "",
            }
        },
    )
    assert result["model"]["api_key_env"] == ""
    assert result["model"]["base_url"] == "http://localhost:8000/v1"


def test_switch_to_decision_provider_uses_its_model_default(tmp_path):
    config_file = tmp_path / "model.toml"
    config_file.write_text('[model]\nprovider="openai"\nmodel="old"\nreasoning_effort="low"\n')
    result = load_config(config_file, {"model": {"provider": "jev", "model": None}})
    assert result["model"]["model"] == "jev-latest"
    assert "reasoning_effort" not in result["model"]


def test_generic_provider_swap_requires_an_explicit_model():
    with pytest.raises(ValueError, match="exact model ID"):
        load_config(overrides={"model": {"provider": "anthropic"}})
