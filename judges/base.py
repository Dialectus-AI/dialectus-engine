"""Base classes and interfaces for judging systems."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from debate_engine.models import DebateContext


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
    criterion_scores: List[CriterionScore]
    overall_feedback: str
    reasoning: str
    metadata: Dict[str, Any]


class BaseJudge(ABC):
    """Abstract base class for all judges."""
    
    def __init__(self, criteria: List[str]):
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
    
    def _calculate_total_score(self, scores: List[CriterionScore], participant_id: str) -> float:
        """Calculate total score for a participant across all criteria."""
        participant_scores = [s.score for s in scores if s.participant_id == participant_id]
        return sum(participant_scores) / len(participant_scores) if participant_scores else 0.0