"""Factory for creating different types of judges."""

import logging
from typing import Optional, List

from models.manager import ModelManager
from config.settings import SystemConfig
from .base import BaseJudge
from .ai_judge import AIJudge, EnsembleJudge

logger = logging.getLogger(__name__)


def create_judge(
    judge_models: List[str],
    judge_provider: Optional[str],
    system_config: SystemConfig,
    model_manager: ModelManager,
    criteria: Optional[List[str]] = None
) -> Optional[BaseJudge]:
    """Factory function to create appropriate judge based on judges array."""

    if not judge_models or len(judge_models) == 0:
        logger.info("No judges specified - debate will complete without evaluation")
        return None

    # Use default criteria if none provided
    if criteria is None:
        criteria = ["logic", "evidence", "persuasiveness"]

    if len(judge_models) == 1:
        # Single judge
        judge_model = judge_models[0]
        if not judge_provider:
            raise ValueError(f"Judge provider must be specified for model {judge_model}")

        logger.info(f"Creating single AI judge with model: {judge_model}")
        return AIJudge(
            model_manager=model_manager,
            judge_model_name=judge_model,
            criteria=criteria,
            system_config=system_config,
            judge_provider=judge_provider
        )

    else:
        # Ensemble of judges
        if not judge_provider:
            raise ValueError("Judge provider must be specified for ensemble judging")

        logger.info(f"Creating ensemble judge with models: {judge_models}")
        individual_judges = []
        for model_name in judge_models:
            judge = AIJudge(
                model_manager=model_manager,
                judge_model_name=model_name,
                criteria=criteria,
                system_config=system_config,
                judge_provider=judge_provider
            )
            individual_judges.append(judge)

        return EnsembleJudge(individual_judges, criteria)


def create_judge_with_auto_config(
    system_config: SystemConfig,
    model_manager: ModelManager,
    judge_model: str = "openthinker:7b",
    criteria: Optional[list] = None
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
        judge_provider="ollama"  # Default for auto-config with ollama model names
    )