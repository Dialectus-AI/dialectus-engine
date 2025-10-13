"""Tests for judge factory functions."""

from __future__ import annotations

import pytest

from dialectus.engine.config.settings import ModelConfig, SystemConfig
from dialectus.engine.judges.factory import (
    create_judge_with_auto_config,
    create_judges,
)
from dialectus.engine.judges.base import JudgmentCriterion


class StubModelManager:
    """Minimal stand-in for ModelManager capturing registrations."""

    def __init__(self) -> None:
        self.register_calls: list[tuple[str, ModelConfig]] = []

    def register_model(self, model_id: str, config: ModelConfig) -> None:  # noqa: D401
        self.register_calls.append((model_id, config))


@pytest.fixture
def system_config() -> SystemConfig:
    return SystemConfig()


@pytest.fixture
def stub_model_manager() -> StubModelManager:
    return StubModelManager()


def test_create_judges_returns_ensemble_with_default_criteria(
    system_config: SystemConfig, stub_model_manager: StubModelManager
) -> None:
    judges = create_judges(
        judge_models=["judge-a", "judge-b"],
        judge_provider="ollama",
        system_config=system_config,
        model_manager=stub_model_manager,  # type: ignore[arg-type]
        criteria=None,
    )

    assert len(judges) == 2
    assert len(stub_model_manager.register_calls) == 2

    expected_criteria = [
        JudgmentCriterion.LOGIC,
        JudgmentCriterion.EVIDENCE,
        JudgmentCriterion.PERSUASIVENESS,
    ]
    assert list(judges[0].criteria) == expected_criteria

    temperatures = [judge._ensemble_temperature for judge in judges]  # noqa: SLF001
    assert temperatures == [0.4, 0.5]
    assert all(judge.judge_provider == "ollama" for judge in judges)


def test_create_judges_with_custom_criteria_reuses_input_list(
    system_config: SystemConfig, stub_model_manager: StubModelManager
) -> None:
    custom_criteria = ["logic", "clarity"]
    judges = create_judges(
        judge_models=["solo"],
        judge_provider="openrouter",
        system_config=system_config,
        model_manager=stub_model_manager,  # type: ignore[arg-type]
        criteria=custom_criteria,
    )

    assert len(judges) == 1
    judge = judges[0]

    assert list(judge.criteria) == [
        JudgmentCriterion.LOGIC,
        JudgmentCriterion.CLARITY,
    ]
    assert judge.judge_provider == "openrouter"
    # Single judge keeps default ensemble temperature from AIJudge initializer
    assert judge._ensemble_temperature == pytest.approx(0.3, rel=1e-6)  # noqa: SLF001  # type: ignore[comparison-overlap]


def test_create_judges_without_models_returns_empty(
    system_config: SystemConfig, stub_model_manager: StubModelManager
) -> None:
    judges = create_judges(
        judge_models=[],
        judge_provider="ollama",
        system_config=system_config,
        model_manager=stub_model_manager,  # type: ignore[arg-type]
        criteria=None,
    )

    assert judges == []
    assert stub_model_manager.register_calls == []


def test_create_judges_requires_provider(
    system_config: SystemConfig, stub_model_manager: StubModelManager
) -> None:
    with pytest.raises(ValueError):
        create_judges(
            judge_models=["judge-a"],
            judge_provider="",
            system_config=system_config,
            model_manager=stub_model_manager,  # type: ignore[arg-type]
            criteria=None,
        )


def test_create_judge_with_auto_config_uses_defaults(
    system_config: SystemConfig, stub_model_manager: StubModelManager
) -> None:
    judge = create_judge_with_auto_config(
        system_config=system_config,
        model_manager=stub_model_manager,  # type: ignore[arg-type]
    )

    assert judge.judge_provider == "ollama"
    assert [criterion.value for criterion in judge.criteria] == [
        "logic",
        "evidence",
        "persuasiveness",
    ]
    assert len(stub_model_manager.register_calls) == 1


def test_create_judge_with_auto_config_accepts_custom_criteria(
    system_config: SystemConfig, stub_model_manager: StubModelManager
) -> None:
    custom_criteria = ["logic", "format_adherence"]
    judge = create_judge_with_auto_config(
        system_config=system_config,
        model_manager=stub_model_manager,  # type: ignore[arg-type]
        judge_model="custom-model",
        criteria=custom_criteria,
    )

    assert judge.judge_model_name == "custom-model"
    assert list(judge.criteria) == [
        JudgmentCriterion.LOGIC,
        JudgmentCriterion.FORMAT_ADHERENCE,
    ]
    assert stub_model_manager.register_calls[-1][1].name == "custom-model"
