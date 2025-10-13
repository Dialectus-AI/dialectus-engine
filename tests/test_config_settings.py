"""Tests for configuration models and loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dialectus.engine.config.settings import (
    AppConfig,
    DebateConfig,
    JudgingConfig,
    ModelConfig,
    SystemConfig,
)


def build_minimal_config() -> AppConfig:
    """Return a minimal but valid AppConfig instance."""
    return AppConfig(
        debate=DebateConfig(topic="Test topic", format="oxford"),
        models={"model_a": ModelConfig(name="llama", provider="ollama")},
        judging=JudgingConfig(),
        system=SystemConfig(),
    )


def test_model_config_rejects_unknown_provider() -> None:
    """Provider validator raises when an unsupported provider is given."""
    with pytest.raises(ValidationError) as excinfo:
        ModelConfig(name="model", provider="unsupported")

    assert "Provider must be one of" in str(excinfo.value)


def test_model_config_accepts_known_providers() -> None:
    """Known providers pass validation."""
    config = ModelConfig(name="model", provider="openrouter")
    assert config.provider == "openrouter"


def test_system_config_topic_source_validation() -> None:
    """debate_topic_source validator enforces allowed values."""
    system = SystemConfig(debate_topic_source="ollama")
    assert system.debate_topic_source == "ollama"

    with pytest.raises(ValidationError) as excinfo:
        # Use dict unpacking to bypass static type checking while testing runtime validation
        SystemConfig(**{"debate_topic_source": "invalid"})  # type: ignore[arg-type]

    assert "ollama" in str(excinfo.value)
    assert "openrouter" in str(excinfo.value)


def test_app_config_load_from_file_success(tmp_path: Path) -> None:
    """load_from_file parses JSON config into nested Pydantic models."""
    config_path = tmp_path / "config.json"
    config = build_minimal_config()
    config_data = config.model_dump()

    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    loaded = AppConfig.load_from_file(config_path)
    assert loaded.debate.topic == "Test topic"
    assert "model_a" in loaded.models
    assert isinstance(loaded.models["model_a"], ModelConfig)
    assert loaded.system.log_level == "INFO"


def test_app_config_load_from_file_missing_file(tmp_path: Path) -> None:
    """Missing file raises FileNotFoundError."""
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        AppConfig.load_from_file(missing)


@pytest.mark.parametrize(
    "missing_section",
    [
        ["debate"],
        ["models"],
        ["judging"],
        ["system"],
        ["debate", "judging"],
    ],
)
def test_app_config_missing_sections(
    tmp_path: Path, missing_section: list[str]
) -> None:
    """load_from_file raises when required top-level sections are missing."""
    config = build_minimal_config().model_dump()
    for section in missing_section:
        config.pop(section, None)

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        AppConfig.load_from_file(config_path)

    assert "Missing required config sections" in str(excinfo.value)


def test_app_config_requires_at_least_one_model(tmp_path: Path) -> None:
    """load_from_file enforces presence of at least one model entry."""
    config = build_minimal_config().model_dump()
    config["models"] = {}

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        AppConfig.load_from_file(config_path)

    assert "at least one model" in str(excinfo.value)
