"""Ensemble judging utilities - standalone functions for combining judge decisions."""

import logging
from typing import Any, TypedDict

from dialectus.engine.debate_engine.models import DebateContext

from .base import CriterionScore, JudgeDecision

logger = logging.getLogger(__name__)


class SerializedCriterionScore(TypedDict):
    """Serialized criterion score for storage."""

    criterion: str
    participant_id: str
    score: float
    feedback: str


class SerializedJudgeDecision(TypedDict):
    """Serialized judge decision for storage in ensemble metadata."""

    winner_id: str
    winner_margin: float
    overall_feedback: str
    reasoning: str
    criterion_scores: list[SerializedCriterionScore]
    judge_model: str
    judge_provider: str
    generation_time_ms: int | None


def calculate_ensemble_result(
    decisions: list[JudgeDecision],
    context: DebateContext,
) -> dict[str, Any]:
    """Calculate ensemble result from multiple judge decisions."""

    # Debug: Check if all decisions are identical
    if len(decisions) > 1:
        first_decision = decisions[0]
        identical_decisions = True
        for decision in decisions[1:]:
            if (
                decision.winner_id != first_decision.winner_id
                or abs(decision.winner_margin - first_decision.winner_margin) > 0.01
                or len(decision.criterion_scores)
                != len(first_decision.criterion_scores)
            ):
                identical_decisions = False
                break

        if identical_decisions:
            logger.warning(
                "WARNING: All ensemble judges produced identical decisions - this"
                " suggests a problem with randomness or caching"
            )
            for i, decision in enumerate(decisions):
                logger.warning(
                    f"Judge {i + 1} decision: winner={decision.winner_id},"
                    f" margin={decision.winner_margin:.2f},"
                    f" scores={len(decision.criterion_scores)}"
                )

    logger.info("=== ENSEMBLE DEBUG: Individual judge decisions ===")
    for i, decision in enumerate(decisions):
        logger.info(
            f"Judge {i + 1}: winner={decision.winner_id},"
            f" margin={decision.winner_margin:.2f}"
        )
        if decision.criterion_scores:
            first_score = decision.criterion_scores[0]
            logger.info(
                f"  First score: {first_score.participant_id} -"
                f" {first_score.criterion.value}: {first_score.score} -"
                f" {first_score.feedback[:50]}"
            )

    # Count votes for each participant
    winner_votes: dict[str, int] = {}
    for decision in decisions:
        winner_votes[decision.winner_id] = winner_votes.get(decision.winner_id, 0) + 1

    # Get participants and calculate average scores for tie-breaking
    participants = list(context.participants.keys())
    participant_total_scores = {p: 0.0 for p in participants}
    participant_score_counts = {p: 0 for p in participants}

    # Calculate total scores per participant across all judges
    all_scores: list[CriterionScore] = []
    for decision in decisions:
        all_scores.extend(decision.criterion_scores)

    for score in all_scores:
        participant_total_scores[score.participant_id] += score.score
        participant_score_counts[score.participant_id] += 1

    # Calculate average scores
    participant_avg_scores: dict[str, float] = {}
    for participant in participants:
        if participant_score_counts[participant] > 0:
            participant_avg_scores[participant] = (
                participant_total_scores[participant]
                / participant_score_counts[participant]
            )
        else:
            participant_avg_scores[participant] = 0.0

    # Determine winner with tie-breaking logic
    max_votes = max(winner_votes.values()) if winner_votes else 0
    winners_by_vote = [p for p, votes in winner_votes.items() if votes == max_votes]

    if len(winners_by_vote) == 1:
        # Clear winner by votes
        ensemble_winner = winners_by_vote[0]
        decision_method = f"majority vote ({max_votes}/{len(decisions)})"
    else:
        # Tie in votes - break tie using average scores
        tie_winner = max(winners_by_vote, key=lambda p: participant_avg_scores[p])
        ensemble_winner = tie_winner
        tied_votes = max_votes
        winner_score = participant_avg_scores[tie_winner]
        decision_method = (
            f"tie-breaker by scores (tied at {tied_votes}/{len(decisions)} votes,"
            f" winner scored {winner_score:.1f} avg)"
        )

    # Validate vote vs score consistency (warn if mismatch but don't fail)
    score_winner = max(participants, key=lambda p: participant_avg_scores[p])
    if score_winner != ensemble_winner:
        logger.warning(
            f"Vote winner ({ensemble_winner}) differs from score winner"
            f" ({score_winner}). Using vote-based result with tie-breaking."
        )

    # Calculate winner margin from averaged scores
    if len(participant_avg_scores) >= 2:
        averages = list(participant_avg_scores.values())
        averages.sort(reverse=True)
        ensemble_margin = averages[0] - averages[1]
    else:
        ensemble_margin = 0.0

    # Calculate consensus level (agreement rate)
    consensus_level = max_votes / len(decisions) if decisions else 0.0

    return {
        "final_winner_id": ensemble_winner,
        "final_margin": ensemble_margin,
        "ensemble_method": "majority_vote_with_tiebreaker",
        "num_judges": len(decisions),
        "consensus_level": consensus_level,
        "summary_reasoning": f"Winner determined by {decision_method}",
        "summary_feedback": (
            f"Ensemble decision from {len(decisions)} judges. Consensus level:"
            f" {consensus_level:.1%}"
        ),
    }


def serialize_individual_decision(decision: JudgeDecision) -> SerializedJudgeDecision:
    """Serialize an individual judge decision for storage in ensemble metadata."""
    return {
        "winner_id": decision.winner_id,
        "winner_margin": decision.winner_margin,
        "overall_feedback": decision.overall_feedback,
        "reasoning": decision.reasoning,
        "criterion_scores": [
            {
                "criterion": (
                    score.criterion
                    if isinstance(score.criterion, str)
                    else score.criterion.value
                ),
                "participant_id": score.participant_id,
                "score": score.score,
                "feedback": score.feedback,
            }
            for score in decision.criterion_scores
        ],
        "judge_model": decision.judge_model,
        "judge_provider": decision.judge_provider,
        "generation_time_ms": decision.generation_time_ms,
    }
