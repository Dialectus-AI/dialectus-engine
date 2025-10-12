"""Base classes and interfaces for judging systems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from dialectus.engine.debate_engine.models import DebateContext


class JudgmentCriterion(Enum):
    """Standard judging criteria."""

    LOGIC = "logic"
    EVIDENCE = "evidence"
    PERSUASIVENESS = "persuasiveness"
    CLARITY = "clarity"
    REBUTTAL = "rebuttal"
    FORMAT_ADHERENCE = "format_adherence"


@dataclass
class CriterionScore:
    """Score for a single judging criterion."""

    criterion: JudgmentCriterion
    participant_id: str
    score: float  # 0.0 to 10.0
    feedback: str


@dataclass
class JudgeDecision:
    """Complete judge decision with reasoning."""

    winner_id: str
    winner_margin: float  # How decisive the win was (0.0 to 10.0)
    criterion_scores: list[CriterionScore]
    overall_feedback: str
    reasoning: str
    judge_model: str
    judge_provider: str
    generation_time_ms: int | None = None
    cost: float | None = None
    generation_id: str | None = None
    cost_queried_at: str | None = None


class BaseJudge(ABC):
    """Abstract base class for all judges."""

    def __init__(self, criteria: list[str]):
        self.criteria = [JudgmentCriterion(c) for c in criteria]

    @abstractmethod
    async def evaluate_debate(self, context: DebateContext) -> JudgeDecision:
        """Evaluate a completed debate and return decision."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Judge name/identifier."""
        pass

    def _calculate_total_score(
        self, scores: list[CriterionScore], participant_id: str
    ) -> float:
        """Calculate total score for a participant across all criteria."""
        participant_scores = [
            s.score for s in scores if s.participant_id == participant_id
        ]
        return (
            sum(participant_scores) / len(participant_scores)
            if participant_scores
            else 0.0
        )

    def _calculate_winner_margin(self, criterion_scores: list[CriterionScore]) -> float:
        """Calculate victory margin from criterion scores (centralized logic)."""
        if not criterion_scores:
            return 0.0

        # Group scores by participant
        participant_totals: dict[str, float] = {}
        participant_counts: dict[str, int] = {}

        for score in criterion_scores:
            participant_id = score.participant_id
            if participant_id not in participant_totals:
                participant_totals[participant_id] = 0.0
                participant_counts[participant_id] = 0

            participant_totals[participant_id] += score.score
            participant_counts[participant_id] += 1

        # Calculate average scores for each participant
        participant_averages: dict[str, float] = {}
        for participant_id in participant_totals:
            if participant_counts[participant_id] > 0:
                participant_averages[participant_id] = (
                    participant_totals[participant_id]
                    / participant_counts[participant_id]
                )

        # Find margin between highest and second highest
        if len(participant_averages) < 2:
            return 0.0

        averages = list(participant_averages.values())
        averages.sort(reverse=True)
        return averages[0] - averages[1]

    def _determine_winner_from_scores(
        self, criterion_scores: list[CriterionScore]
    ) -> str:
        """Determine winner based on highest average score across criteria."""
        if not criterion_scores:
            return "unknown"

        participant_averages: dict[str, float] = {}
        participant_totals: dict[str, float] = {}
        participant_counts: dict[str, int] = {}

        for score in criterion_scores:
            participant_id = score.participant_id
            if participant_id not in participant_totals:
                participant_totals[participant_id] = 0.0
                participant_counts[participant_id] = 0

            participant_totals[participant_id] += score.score
            participant_counts[participant_id] += 1

        # Calculate averages
        for participant_id in participant_totals:
            if participant_counts[participant_id] > 0:
                participant_averages[participant_id] = (
                    participant_totals[participant_id]
                    / participant_counts[participant_id]
                )

        # Return participant with highest average
        return (
            max(participant_averages.keys(), key=lambda x: participant_averages[x])
            if participant_averages
            else "unknown"
        )
