"""Factory for creating different types of judges."""

import logging
from typing import Optional

from models.manager import ModelManager
from config.settings import JudgingConfig, SystemConfig
from .base import BaseJudge
from .ai_judge import AIJudge, EnsembleJudge

logger = logging.getLogger(__name__)


def create_judge(
    judging_config: JudgingConfig,
    system_config: SystemConfig,
    model_manager: ModelManager
) -> Optional[BaseJudge]:
    """Factory function to create appropriate judge based on configuration."""
    
    if judging_config.method == "none":
        logger.info("No judging selected - debate will complete without evaluation")
        return None
    
    elif judging_config.method == "ai":
        if not judging_config.judge_model:
            logger.error("AI judging requires judge_model to be specified in config")
            return None
            
        logger.info(f"Creating AI judge with model: {judging_config.judge_model}")
        return AIJudge(
            model_manager=model_manager,
            judge_model_name=judging_config.judge_model,
            criteria=judging_config.criteria,
            system_config=system_config
        )
    
    elif judging_config.method == "ensemble":
        if not judging_config.judge_model:
            logger.error("Ensemble judging requires judge_model to be specified")
            return None
        
        # For ensemble, judge_model can be comma-separated list
        judge_models = [m.strip() for m in judging_config.judge_model.split(",")]
        if len(judge_models) < 2:
            logger.warning("Ensemble judging works best with multiple models, falling back to single AI judge")
            return AIJudge(
                model_manager=model_manager,
                judge_model_name=judge_models[0],
                criteria=judging_config.criteria,
                system_config=system_config
            )
        
        logger.info(f"Creating ensemble judge with models: {judge_models}")
        individual_judges = []
        for model_name in judge_models:
            judge = AIJudge(
                model_manager=model_manager,
                judge_model_name=model_name,
                criteria=judging_config.criteria,
                system_config=system_config
            )
            individual_judges.append(judge)
        
        return EnsembleJudge(individual_judges, judging_config.criteria)
    
    elif judging_config.method == "tournament":
        # For now, tournament mode uses AI judging as base
        if not judging_config.judge_model:
            logger.error("Tournament judging requires judge_model to be specified")
            return None
            
        logger.info(f"Creating tournament AI judge with model: {judging_config.judge_model}")
        return AIJudge(
            model_manager=model_manager,
            judge_model_name=judging_config.judge_model,
            criteria=judging_config.criteria,
            system_config=system_config
        )
    
    else:
        logger.error(f"Unknown judging method: {judging_config.method}")
        return None


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
        system_config=system_config
    )