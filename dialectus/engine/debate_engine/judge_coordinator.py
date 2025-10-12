"""Judging coordination for completed debates."""

import logging
import random
from typing import Any

from dialectus.engine.judges.base import BaseJudge, JudgeDecision

from .models import DebateContext
from .types import DebatePhase

logger = logging.getLogger(__name__)


class JudgeCoordinator:
    """Coordinates judging of completed debates using AI judges."""

    @staticmethod
    async def judge_debate_with_judges(
        context: DebateContext,
        judges: list[BaseJudge],
    ) -> Any | None:
        """Judge the completed debate using a list of AI judges.

        Args:
            context: The debate context
            judges: List of judge instances

        Returns:
            Decision from single judge or ensemble result from multiple judges
        """
        if context.current_phase != DebatePhase.COMPLETED:
            raise RuntimeError("Cannot judge incomplete debate")

        if not judges:
            logger.info("No judges provided - skipping automated judging")
            return None

        from dialectus.engine.judges.ensemble_utils import calculate_ensemble_result

        if len(judges) == 1:
            # Single judge case
            judge = judges[0]
            logger.info(f"Judging debate with single judge: {judge.name}")
            decision = await judge.evaluate_debate(context)
            return decision

        else:
            # Multiple judges - ensemble case
            logger.info(
                f"Judging debate with {len(judges)} judges for ensemble evaluation"
            )

            decisions: list[JudgeDecision] = []
            failed_judges: list[str] = []

            for i, judge in enumerate(judges):
                try:
                    # Add slight randomness to avoid identical evaluations
                    random_seed = random.random()
                    logger.info(
                        f"Starting evaluation for judge {i + 1}: {judge.name} "
                        f"(seed: {random_seed:.4f})"
                    )

                    decision = await judge.evaluate_debate(context)
                    decisions.append(decision)
                    logger.info(
                        f"Judge {judge.name} completed successfully with winner: "
                        f"{decision.winner_id}, margin: {decision.winner_margin:.2f}"
                    )

                    # Debug: Log first criterion score to verify different evaluations
                    if decision.criterion_scores:
                        first_score = decision.criterion_scores[0]
                        logger.info(
                            f"Judge {i + 1} first score sample: {first_score.participant_id} - "
                            f"{first_score.criterion.value}: {first_score.score} - "
                            f"{first_score.feedback[:50]}..."
                        )

                except Exception as e:
                    logger.error(f"Judge {judge.name} failed: {e}")
                    failed_judges.append(f"{judge.name}: {str(e)}")

            # Require ALL judges to succeed for ensemble judging
            if failed_judges:
                failed_list = "; ".join(failed_judges)
                raise RuntimeError(
                    f"Ensemble judging requires ALL judges to succeed. "
                    f"Failed judges ({len(failed_judges)}/{len(judges)}): {failed_list}"
                )

            if not decisions:
                raise RuntimeError("All ensemble judges failed to evaluate the debate")

            logger.info(
                f"Ensemble evaluation: All {len(decisions)} judges completed successfully"
            )

            # Return the list of decisions for ensemble handling
            return {
                "type": "ensemble",
                "decisions": decisions,
                "ensemble_summary": calculate_ensemble_result(decisions, context),
            }
