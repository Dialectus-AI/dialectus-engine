"""Factory for creating judges."""

import logging

from dialectus.engine.models.manager import ModelManager
from dialectus.engine.config.settings import SystemConfig
from .ai_judge import AIJudge

logger = logging.getLogger(__name__)


def create_judges(
    judge_models: list[str],
    judge_provider: str,
    system_config: SystemConfig,
    model_manager: ModelManager,
    criteria: list[str] | None = None,
) -> list[AIJudge]:
    """Factory function to create list of AI judges."""

    if not judge_models or len(judge_models) == 0:
        logger.info("No judges specified - debate will complete without evaluation")
        return []

    # Use default criteria if none provided
    if criteria is None:
        criteria = ["logic", "evidence", "persuasiveness"]

    if not judge_provider:
        raise ValueError("Judge provider must be specified")

    logger.info(f"Creating {len(judge_models)} AI judges with models: {judge_models}")
    judges: list[AIJudge] = []

    for i, model_name in enumerate(judge_models):
        judge = AIJudge(
            model_manager=model_manager,
            judge_model_name=model_name,
            criteria=criteria,
            system_config=system_config,
            judge_provider=judge_provider,
        )
        # Add slightly higher temperature for ensemble judges to increase variation
        if len(judge_models) > 1:
            judge._ensemble_temperature = 0.4 + (i * 0.1)  # 0.4, 0.5, 0.6, etc.

        judges.append(judge)

    return judges


def create_judge_with_auto_config(
    system_config: SystemConfig,
    model_manager: ModelManager,
    judge_model: str = "openthinker:7b",
    criteria: list[str] | None = None,
) -> AIJudge:
    """Quick factory for creating AI judge with sensible defaults."""
    if criteria is None:
        criteria = ["logic", "evidence", "persuasiveness"]

    logger.info(f"Auto-configuring AI judge with {judge_model}")
    return AIJudge(
        model_manager=model_manager,
        judge_model_name=judge_model,
        criteria=criteria,
        system_config=system_config,
        judge_provider="ollama",  # Default for auto-config with ollama model names
    )
