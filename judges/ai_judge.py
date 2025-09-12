"""AI-powered debate judge using language models."""

import json
import logging
from typing import Dict, List, Optional
import re

from models.manager import ModelManager
from config.settings import SystemConfig
from debate_engine.models import DebateContext
from formats import format_registry
from .base import BaseJudge, JudgeDecision, CriterionScore, JudgmentCriterion

logger = logging.getLogger(__name__)


class AIJudge(BaseJudge):
    """AI judge using a dedicated language model for evaluation."""

    def __init__(
        self,
        model_manager: ModelManager,
        judge_model_name: str,
        criteria: List[str],
        system_config: SystemConfig,
        judge_provider: Optional[str] = None,
    ):
        super().__init__(criteria)
        self.model_manager = model_manager
        self.judge_model_name = judge_model_name
        self.system_config = system_config

        # Register judge model with manager
        from config.settings import ModelConfig

        # Use provided provider or raise an error if not provided
        if not judge_provider:
            raise ValueError(f"Judge provider must be specified for model {judge_model_name}")

        judge_config = ModelConfig(
            name=judge_model_name,
            provider=judge_provider,
            personality="impartial",
            max_tokens=1500,  # Longer responses for detailed evaluation
            temperature=0.3,  # Lower temperature for more consistent judging
        )
        self.model_manager.register_model("judge", judge_config)

    @property
    def name(self) -> str:
        return f"AI Judge ({self.judge_model_name})"

    async def evaluate_debate(self, context: DebateContext) -> JudgeDecision:
        """Evaluate the debate using the AI judge model."""
        logger.info(f"AI judge evaluating debate: {context.topic}")

        # Prepare transcript for evaluation
        transcript = self._format_transcript(context)

        try:
            # Generate evaluation
            evaluation = await self._generate_evaluation(transcript, context)
            decision = self._parse_evaluation(evaluation, context)

            logger.info(
                f"AI judge decision: {decision.winner_id} wins by {decision.winner_margin:.1f}"
            )
            return decision

        except Exception as e:
            logger.error(f"AI judge evaluation failed: {e}")
            # Fallback to basic scoring
            return self._create_fallback_decision(context)

    def _format_transcript(self, context: DebateContext) -> str:
        """Format debate transcript for judge evaluation."""
        lines = []
        lines.append(f"DEBATE TOPIC: {context.topic}")
        lines.append(f"FORMAT: {context.metadata.get('format', 'Unknown')}")
        lines.append("")

        # Group messages by round/phase for clarity
        current_round = 0
        for message in context.messages:
            if message.round_number != current_round:
                current_round = message.round_number
                lines.append(
                    f"\n=== ROUND {current_round} - {message.phase.value.upper()} ==="
                )

            lines.append(f"\n{message.speaker_id} ({message.position.value}):")
            lines.append(message.content)

        return "\n".join(lines)

    def _get_participant_labels(
        self, participants: List[str], context: DebateContext
    ) -> Dict[str, str]:
        """Get format-specific labels for participants."""
        try:
            # Get format name from context metadata
            format_name = context.metadata.get("format", "oxford")  # Default to oxford

            # Get format instance
            debate_format = format_registry.get_format(format_name)

            # Get side labels from format
            side_labels = debate_format.get_side_labels(participants)

            return side_labels
        except Exception as e:
            logger.warning(
                f"Failed to get format-specific labels: {e}, falling back to generic labels"
            )
            # Fallback to generic labels
            fallback_labels = {}
            for i, participant in enumerate(participants):
                fallback_labels[participant] = (
                    f"Participant {chr(65 + i)}"  # A, B, C, etc.
                )
            return fallback_labels

    def _map_enhanced_label_to_participant_id(
        self, enhanced_label: str, participants: List[str], context: DebateContext
    ) -> str:
        """Map enhanced label back to participant ID."""
        # Create mapping from enhanced labels to participant IDs
        participant_labels = self._get_participant_labels(participants, context)

        for participant in participants:
            model_name = context.participants[participant].name
            side_label = participant_labels.get(participant, participant)
            expected_label = f"{model_name} - {side_label}"

            if enhanced_label == expected_label:
                return participant

        # If no exact match found, try to find best match
        for participant in participants:
            if participant in enhanced_label:
                return participant

        # Fallback to first participant if nothing matches
        logger.warning(
            f"Could not map enhanced label '{enhanced_label}' to participant ID, using fallback"
        )
        return participants[0] if participants else "unknown"

    async def _generate_evaluation(
        self, transcript: str, context: DebateContext
    ) -> str:
        """Generate AI evaluation of the debate."""
        import time

        participants = list(context.participants.keys())
        criteria_list = [c.value for c in self.criteria]

        evaluation_prompt = self._create_evaluation_prompt(
            transcript, participants, criteria_list, context
        )

        messages = [
            {"role": "system", "content": self._get_judge_system_prompt()},
            {"role": "user", "content": evaluation_prompt},
        ]

        # Capture evaluation timing
        start_time = time.time()

        # Use model manager to generate response
        async with self.model_manager.model_session("judge"):
            response = await self.model_manager.generate_response(
                "judge", messages, max_tokens=1500, temperature=0.3
            )

        generation_time = time.time() - start_time

        # Store timing in instance variable for use in decision creation
        self._last_generation_time_ms = int(generation_time * 1000)

        return response

    def _get_judge_system_prompt(self) -> str:
        """Get system prompt for the AI judge."""
        return """You are an expert debate judge with years of experience evaluating formal debates. Your role is to provide fair, objective, and detailed evaluations of debate performances.

        JUDGING PRINCIPLES:
        1. Evaluate arguments based on logic, evidence, and persuasiveness
        2. Consider how well debaters address opponent arguments  
        3. Assess adherence to debate format and rules
        4. Be impartial - judge the arguments, not the participants
        5. Provide specific, constructive feedback
        6. Score each criterion on a scale of 0-10 (10 being exceptional)

        You must respond with a structured JSON evaluation that can be parsed programmatically. Be thorough but concise in your reasoning."""

    def _create_evaluation_prompt(
        self,
        transcript: str,
        participants: List[str],
        criteria: List[str],
        context: DebateContext,
    ) -> str:
        """Create the evaluation prompt for the judge."""
        # Get format-specific side labels
        participant_labels = self._get_participant_labels(participants, context)

        # Create enhanced participant list with side labels and model names
        enhanced_participants = []
        for participant in participants:
            model_name = context.participants[participant].name
            side_label = participant_labels.get(participant, participant)
            enhanced_label = f"{model_name} - {side_label}"
            enhanced_participants.append(enhanced_label)

        return f"""Please evaluate this debate and provide your judgment in the following JSON format:

{{
  "winner": "{enhanced_participants[0]} or {enhanced_participants[1]}",
  "overall_feedback": "2-3 sentence summary of the debate quality",
  "reasoning": "Write a detailed natural language explanation of your decision. Use complete sentences and paragraphs. Do NOT use structured data, dictionaries, or lists here - only descriptive text explaining your thought process.",
  "criterion_scores": [
    {{
      "criterion": "{criteria[0]}",
      "participant": "{enhanced_participants[0]}",
      "score": 7.5,
      "feedback": "specific feedback"
    }},
    ...repeat for all participants and criteria...
  ]
}}

EVALUATION CRITERIA: {', '.join(criteria)}

PARTICIPANTS:
{chr(10).join(f"- {label}" for label in enhanced_participants)}

CRITICAL INSTRUCTIONS:
- Reference participants by their full labels (e.g., "{enhanced_participants[0]}")
- The "reasoning" field must contain ONLY natural language text explaining your decision
- Do NOT put structured data, scores, or dictionaries in the "reasoning" field
- All numerical scores belong ONLY in the "criterion_scores" array
- Focus on argument quality, evidence, and debate performance
- Provide specific feedback for each participant and criterion

EXAMPLE of correct reasoning field:
"The pro side presented stronger evidence with three concrete examples, while the con side relied more on theoretical arguments. The pro debater also did a better job addressing counterarguments in the rebuttal phase, showing deeper engagement with the opposing viewpoint."

DEBATE TRANSCRIPT:
{transcript}

Provide your evaluation as valid JSON only, no additional text:"""

    def _parse_evaluation(
        self, evaluation: str, context: DebateContext
    ) -> JudgeDecision:
        """Parse AI evaluation response into structured decision."""
        try:
            # Extract JSON from response (handle cases where model adds extra text)
            json_match = re.search(r"\{.*\}", evaluation, re.DOTALL)
            if json_match:
                evaluation_data = json.loads(json_match.group())
            else:
                evaluation_data = json.loads(evaluation)

            participants = list(context.participants.keys())

            # Map enhanced winner label back to participant ID
            winner_enhanced_label = evaluation_data["winner"]
            winner_id = self._map_enhanced_label_to_participant_id(
                winner_enhanced_label, participants, context
            )

            # Parse criterion scores
            criterion_scores = []
            for score_data in evaluation_data.get("criterion_scores", []):
                participant_enhanced_label = score_data["participant"]
                participant_id = self._map_enhanced_label_to_participant_id(
                    participant_enhanced_label, participants, context
                )

                # Debug and fix feedback field too
                feedback = score_data.get("feedback", "")
                if not isinstance(feedback, str):
                    logger.warning(
                        f"Converting non-string feedback: {type(feedback)} -> {feedback}"
                    )
                    feedback = str(feedback) if feedback else ""

                criterion_scores.append(
                    CriterionScore(
                        criterion=JudgmentCriterion(score_data["criterion"]),
                        participant_id=participant_id,
                        score=float(score_data["score"]),
                        feedback=feedback,
                    )
                )

            # Debug: Log what we actually received for problematic fields
            overall_feedback = evaluation_data.get("overall_feedback", "")
            reasoning = evaluation_data.get("reasoning", "")

            logger.debug(
                f"Judge overall_feedback type: {type(overall_feedback)}, value: {overall_feedback}"
            )
            logger.debug(f"Judge reasoning type: {type(reasoning)}, value: {reasoning}")

            # Ensure these are strings, not objects
            if not isinstance(overall_feedback, str):
                logger.warning(
                    f"Converting non-string overall_feedback: {type(overall_feedback)} -> {overall_feedback}"
                )
                overall_feedback = str(overall_feedback) if overall_feedback else ""

            if not isinstance(reasoning, str):
                logger.warning(
                    f"Converting non-string reasoning: {type(reasoning)} -> {reasoning}"
                )
                reasoning = str(reasoning) if reasoning else ""

            # Create display labels mapping for frontend
            participant_labels = self._get_participant_labels(participants, context)
            display_labels = {}
            for participant in participants:
                model_name = context.participants[participant].name
                side_label = participant_labels.get(participant, participant)
                display_labels[participant] = f"{model_name} - {side_label}"

            return JudgeDecision(
                winner_id=winner_id,
                winner_margin=0.0,  # Will be calculated client-side from criterion scores
                criterion_scores=criterion_scores,
                overall_feedback=overall_feedback,
                reasoning=reasoning,
                metadata={
                    "judge_model": self.judge_model_name,
                    "raw_evaluation": evaluation,
                    "enhanced_labels_used": True,
                    "display_labels": display_labels,  # For frontend display
                    "generation_time_ms": getattr(
                        self, "_last_generation_time_ms", None
                    ),
                },
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse AI judge evaluation: {e}")
            logger.debug(f"Raw evaluation: {evaluation}")
            return self._create_fallback_decision(context)

    def _create_fallback_decision(self, context: DebateContext) -> JudgeDecision:
        """Create a basic fallback decision when AI evaluation fails."""
        participants = list(context.participants.keys())

        # Simple fallback: count message lengths as a proxy for engagement
        participant_scores = {}
        for participant in participants:
            messages = [
                msg for msg in context.messages if msg.speaker_id == participant
            ]
            total_length = sum(len(msg.content) for msg in messages)
            participant_scores[participant] = total_length

        # Determine winner (longest total response)
        if participant_scores:
            winner_id = max(
                participant_scores.keys(), key=lambda x: participant_scores[x]
            )
        else:
            winner_id = participants[0] if participants else "unknown"

        # Create basic criterion scores
        criterion_scores = []
        for criterion in self.criteria:
            for participant in participants:
                score = 6.0 if participant == winner_id else 5.0  # Basic scoring
                criterion_scores.append(
                    CriterionScore(
                        criterion=criterion,
                        participant_id=participant,
                        score=score,
                        feedback=f"Fallback scoring - judge evaluation failed",
                    )
                )

        return JudgeDecision(
            winner_id=winner_id,
            winner_margin=1.0,
            criterion_scores=criterion_scores,
            overall_feedback="AI judge evaluation failed, using fallback scoring",
            reasoning="Technical issue prevented detailed evaluation",
            metadata={"fallback": True},
        )


class EnsembleJudge(BaseJudge):
    """Judge using multiple AI models for ensemble voting."""

    def __init__(self, judges: List[AIJudge], criteria: List[str]):
        super().__init__(criteria)
        self.judges = judges

    @property
    def name(self) -> str:
        judge_names = [judge.judge_model_name for judge in self.judges]
        return f"Ensemble Judge ({', '.join(judge_names)})"

    async def evaluate_debate(self, context: DebateContext) -> JudgeDecision:
        """Evaluate using ensemble of judges."""
        logger.info(f"Ensemble judge evaluating with {len(self.judges)} judges")

        decisions = []
        for judge in self.judges:
            try:
                decision = await judge.evaluate_debate(context)
                decisions.append(decision)
            except Exception as e:
                logger.error(f"Judge {judge.name} failed: {e}")

        if not decisions:
            return self._create_fallback_decision(context)

        return self._combine_decisions(decisions, context)

    def _combine_decisions(
        self, decisions: List[JudgeDecision], context: DebateContext
    ) -> JudgeDecision:
        """Combine multiple judge decisions into ensemble decision."""
        # Simple majority vote for winner
        winner_votes = {}
        for decision in decisions:
            winner_votes[decision.winner_id] = (
                winner_votes.get(decision.winner_id, 0) + 1
            )

        ensemble_winner = (
            max(winner_votes.keys(), key=lambda x: winner_votes[x])
            if winner_votes
            else "unknown"
        )

        # Average scores across judges
        all_scores = []
        for decision in decisions:
            all_scores.extend(decision.criterion_scores)

        # Average by criterion and participant
        averaged_scores = []
        participants = list(context.participants.keys())

        for criterion in self.criteria:
            for participant in participants:
                participant_scores = [
                    s.score
                    for s in all_scores
                    if s.criterion == criterion and s.participant_id == participant
                ]
                if participant_scores:
                    avg_score = sum(participant_scores) / len(participant_scores)
                    feedbacks = [
                        s.feedback
                        for s in all_scores
                        if s.criterion == criterion and s.participant_id == participant
                    ]
                    combined_feedback = " | ".join(
                        feedbacks[:2]
                    )  # Limit feedback length

                    averaged_scores.append(
                        CriterionScore(
                            criterion=criterion,
                            participant_id=participant,
                            score=avg_score,
                            feedback=combined_feedback,
                        )
                    )

        return JudgeDecision(
            winner_id=ensemble_winner,
            winner_margin=sum(d.winner_margin for d in decisions) / len(decisions),
            criterion_scores=averaged_scores,
            overall_feedback=f"Ensemble decision from {len(decisions)} judges",
            reasoning=f"Winner chosen by {winner_votes[ensemble_winner]}/{len(decisions)} judges",
            metadata={
                "ensemble_size": len(decisions),
                "individual_decisions": [d.metadata for d in decisions],
            },
        )

    def _create_fallback_decision(self, context: DebateContext) -> JudgeDecision:
        """Fallback when all ensemble judges fail."""
        participants = list(context.participants.keys())

        # Create basic criterion scores
        criterion_scores = []
        for criterion in self.criteria:
            for participant in participants:
                criterion_scores.append(
                    CriterionScore(
                        criterion=criterion,
                        participant_id=participant,
                        score=5.0,  # Neutral score
                        feedback="Ensemble judging failed",
                    )
                )

        return JudgeDecision(
            winner_id=participants[0] if participants else "unknown",
            winner_margin=0.0,
            criterion_scores=criterion_scores,
            overall_feedback="All ensemble judges failed",
            reasoning="Technical issues prevented evaluation",
            metadata={"ensemble_failure": True},
        )
