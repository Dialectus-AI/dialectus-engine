"""AI-powered debate judge using language models."""

import json
import logging
import re
import uuid
import asyncio
import time
from datetime import datetime
from dialectus.engine.models.manager import ModelManager
from dialectus.engine.config.settings import SystemConfig
from dialectus.engine.debate_engine.models import DebateContext
from .base import BaseJudge, JudgeDecision, CriterionScore, JudgmentCriterion

logger = logging.getLogger(__name__)


class AIJudge(BaseJudge):
    """AI judge using a dedicated language model for evaluation."""

    def __init__(
        self,
        model_manager: ModelManager,
        judge_model_name: str,
        criteria: list[str],
        system_config: SystemConfig,
        judge_provider: str,
    ):
        super().__init__(criteria)
        self.model_manager = model_manager
        self.judge_model_name = judge_model_name
        self.judge_provider = judge_provider
        self.system_config = system_config

        # Initialize ensemble temperature (can be overridden for ensemble judges)
        self._ensemble_temperature: float = 0.3

        # Generate unique judge ID for model manager registration
        self.judge_id = str(uuid.uuid4())

        # Register judge model with manager
        from dialectus.engine.config.settings import ModelConfig

        judge_config = ModelConfig(
            name=judge_model_name,
            provider=judge_provider,
            personality="impartial",
            max_tokens=3000,  # Much longer responses for complete JSON evaluation
            temperature=0.3,  # Lower temperature for more consistent judging
        )
        self.model_manager.register_model(self.judge_id, judge_config)

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
            raise RuntimeError(f"Judge evaluation failed: {e}") from e

    def _format_transcript(self, context: DebateContext) -> str:
        """Format debate transcript for judge evaluation."""
        lines: list[str] = []
        lines.append(f"DEBATE TOPIC: {context.topic}")
        lines.append(f"FORMAT: {context.metadata.get('format', 'Unknown')}")
        lines.append("")

        # Get format-specific side labels for clean transcript display
        participants = list(context.participants.keys())
        participant_labels = self._get_participant_labels(participants, context)

        # Group messages by round/phase for clarity
        current_round = 0
        for message in context.messages:
            if message.round_number != current_round:
                current_round = message.round_number
                lines.append(
                    f"\n=== ROUND {current_round} - {message.phase.value.upper()} ==="
                )

            # Use side label only (e.g., "Proposition:", "Opposition:")
            side_label = participant_labels.get(message.speaker_id, message.speaker_id)
            lines.append(f"\n{side_label}:")
            lines.append(message.content)

        return "\n".join(lines)

    def _get_participant_labels(
        self, participants: list[str], context: DebateContext
    ) -> dict[str, str]:
        """Get format-specific labels for participants."""
        try:
            # Lazy import to avoid circular dependency
            from dialectus.engine.formats import format_registry

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
            fallback_labels: dict[str, str] = {}
            for i, participant in enumerate(participants):
                fallback_labels[participant] = (
                    f"Participant {chr(65 + i)}"  # A, B, C, etc.
                )
            return fallback_labels

    def _map_side_label_to_participant_id(
        self, side_label: str, participants: list[str], context: DebateContext
    ) -> str:
        """Map side label back to participant ID."""
        # Create mapping from side labels to participant IDs
        participant_labels = self._get_participant_labels(participants, context)

        for participant in participants:
            expected_side_label = participant_labels.get(participant, participant)
            if side_label == expected_side_label:
                return participant

        # Fail fast - no fallbacks
        raise ValueError(
            f"Could not map side label '{side_label}' to participant ID. "
            f"Available side labels: {list(participant_labels.values())}"
        )

    async def _generate_evaluation(
        self, transcript: str, context: DebateContext
    ) -> str:
        """Generate AI evaluation of the debate."""
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
        # Add some randomness for ensemble judges to get different evaluations
        temperature = getattr(self, "_ensemble_temperature", 0.3)

        async with self.model_manager.model_session(self.judge_id):
            generation_metadata = await self.model_manager.generate_response_with_metadata(
                self.judge_id, messages, max_tokens=3000, temperature=temperature
            )

        generation_time = time.time() - start_time

        # Store generation metadata for use in decision creation
        self._last_generation_time_ms = generation_metadata.generation_time_ms or int(generation_time * 1000)
        self._last_generation_cost = generation_metadata.cost
        self._last_generation_id = generation_metadata.generation_id
        self._last_generation_provider = generation_metadata.provider

        return generation_metadata.content

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
        participants: list[str],
        criteria: list[str],
        context: DebateContext,
    ) -> str:
        """Create the evaluation prompt for the judge."""
        # Get format-specific side labels only (no model names)
        participant_labels = self._get_participant_labels(participants, context)

        # Use only side labels for participant identification
        side_labels = [participant_labels.get(p, p) for p in participants]

        # Add randomization to prevent identical evaluations
        import random
        import time

        # Use current time and a random element for uniqueness
        evaluation_id = int(time.time() * 1000) % 10000
        random_instruction = random.choice(
            [
                "Pay particular attention to the strength of evidence presented.",
                "Focus especially on how well arguments address counterpoints.",
                "Consider the persuasive impact and clarity of each argument.",
                "Evaluate the logical consistency and reasoning quality.",
                "Assess how effectively each side builds their case.",
            ]
        )

        # Create complete example showing all participants and criteria
        example_scores: list[str] = []
        for criterion in criteria:
            for side_label in side_labels:
                example_scores.append(
                    f"""    {{
      "criterion": "{criterion}",
      "participant": "{side_label}",
      "score": 7.5,
      "feedback": "specific feedback for {side_label} on {criterion}"
    }}"""
                )

        participants_list = "\n".join(f"- {label}" for label in side_labels)
        scores_text = ",\n".join(example_scores)

        return f"""Please evaluate this debate and provide your judgment in the following JSON format.

EVALUATION FOCUS: {random_instruction}
EVALUATION ID: {evaluation_id}

Please evaluate this debate and provide your judgment in the following JSON format:

{{
  "winner": "{side_labels[0]} or {side_labels[1]}",
  "overall_feedback": "2-3 sentence summary of the debate quality",
  "reasoning": "Write a detailed natural language explanation of your decision. Use complete sentences and paragraphs. Do NOT use structured data, dictionaries, or lists here - only descriptive text explaining your thought process.",
  "criterion_scores": [
{scores_text}
  ]
}}

EVALUATION CRITERIA: {', '.join(criteria)}

PARTICIPANTS:
{participants_list}

CRITICAL INSTRUCTIONS:
- You MUST evaluate BOTH participants on ALL criteria - do not skip any participant-criterion combinations
- Reference participants by their side labels only (e.g., "{side_labels[0]}", "{side_labels[1]}")
- The "reasoning" field must contain ONLY natural language text explaining your decision
- Do NOT put structured data, scores, or dictionaries in the "reasoning" field
- All numerical scores belong ONLY in the "criterion_scores" array
- Focus on argument quality, evidence, and debate performance
- Provide specific feedback for each participant and criterion
- Your response must include exactly {len(criteria) * len(side_labels)} criterion_scores entries

EXAMPLE of correct reasoning field:
"The {side_labels[0]} presented stronger evidence with three concrete examples, while the {side_labels[1]} relied more on theoretical arguments. The {side_labels[0]} also did a better job addressing counterarguments in the rebuttal phase, showing deeper engagement with the opposing viewpoint."

DEBATE TRANSCRIPT:
{transcript}

Provide your evaluation as valid JSON only, no additional text:"""

    def _parse_evaluation(
        self, evaluation: str, context: DebateContext
    ) -> JudgeDecision:
        """Parse AI evaluation response into structured decision."""
        try:
            # Log the raw evaluation for debugging
            logger.info(f"Raw judge evaluation (full response): {evaluation}")

            # Extract JSON from response (handle markdown code fences and extra text)
            # First, try to extract from markdown code fences
            markdown_match = re.search(
                r"```(?:json)?\s*(\{.*\})\s*```", evaluation, re.DOTALL
            )
            if markdown_match:
                json_text = markdown_match.group(1)
                logger.debug(
                    f"Extracted JSON from markdown (first 300 chars): {json_text[:300]}"
                )
            else:
                # Fallback to looking for bare JSON - find the complete JSON object
                # Look for opening brace and find matching closing brace
                start_idx = evaluation.find("{")
                if start_idx != -1:
                    # Count braces to find the complete JSON object
                    brace_count = 0
                    end_idx = start_idx
                    for i, char in enumerate(evaluation[start_idx:], start_idx):
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i
                                break

                    if brace_count == 0:  # Found matching closing brace
                        json_text = evaluation[start_idx : end_idx + 1]
                        logger.debug(
                            f"Extracted bare JSON (first 300 chars): {json_text[:300]}"
                        )
                    else:
                        logger.debug(
                            "No complete JSON object found, attempting to parse entire response"
                        )
                        json_text = evaluation.strip()
                else:
                    logger.debug(
                        "No JSON found in response, attempting to parse entire response"
                    )
                    json_text = evaluation.strip()

            # Try to repair common JSON issues from small models
            json_text = self._repair_json(json_text)
            evaluation_data = json.loads(json_text)

            participants = list(context.participants.keys())

            # Map winner side label back to participant ID
            winner_side_label = evaluation_data["winner"]
            winner_id = self._map_side_label_to_participant_id(
                winner_side_label, participants, context
            )

            # Parse criterion scores
            criterion_scores: list[CriterionScore] = []
            for score_data in evaluation_data.get("criterion_scores", []):
                participant_side_label = score_data["participant"]
                participant_id = self._map_side_label_to_participant_id(
                    participant_side_label, participants, context
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

            # Validate that we have complete scoring
            self._validate_complete_scoring(criterion_scores, participants, context)

            # Calculate winner margin from scores using centralized logic
            calculated_margin = self._calculate_winner_margin(criterion_scores)

            # Also verify winner matches highest scoring participant
            calculated_winner = self._determine_winner_from_scores(criterion_scores)
            if calculated_winner != winner_id and calculated_winner != "unknown":
                logger.warning(
                    f"Judge declared winner {winner_id} but highest scoring participant is {calculated_winner}"
                )

            judge_decision = JudgeDecision(
                winner_id=winner_id,
                winner_margin=calculated_margin,  # Calculated from actual scores
                criterion_scores=criterion_scores,
                overall_feedback=overall_feedback,
                reasoning=reasoning,
                judge_model=self.judge_model_name,
                judge_provider=self.judge_provider,
                generation_time_ms=getattr(self, "_last_generation_time_ms", None),
                cost=getattr(self, "_last_generation_cost", None),
                generation_id=getattr(self, "_last_generation_id", None),
            )

            # Schedule background cost query for OpenRouter judges
            if (judge_decision.generation_id and
                getattr(self, "_last_generation_provider", None) == "openrouter"):
                asyncio.create_task(self._query_and_update_judge_cost(judge_decision))

            return judge_decision

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse AI judge evaluation: {e}")
            logger.debug(f"Raw evaluation: {evaluation}")
            raise RuntimeError(f"Failed to parse judge evaluation: {e}") from e

    def _repair_json(self, json_text: str) -> str:
        """Attempt to repair common JSON issues from small models."""
        repair_json = json_text.strip()

        # Remove control characters that cause JSON parsing errors
        repair_json = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', repair_json)

        # Remove any trailing comma before closing braces/brackets
        repair_json = re.sub(r",(\s*[}\]])", r"\1", repair_json)

        # Fix missing quotes around keys (common issue with small models)
        repair_json = re.sub(
            r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', repair_json
        )

        # Handle truncated JSON - try to complete it
        if not repair_json.endswith("}"):
            logger.warning("JSON appears truncated, attempting to complete it")

            # If we're in the middle of a string value, close it
            open_quotes = repair_json.count('"') - repair_json.count('\\"')
            if open_quotes % 2 == 1:  # Odd number means unclosed quote
                repair_json += '"'
                logger.debug("Closed unclosed quote")

            # Remove any trailing comma
            repair_json = repair_json.rstrip().rstrip(",")

            # Count open braces and brackets to determine what to close
            open_braces = repair_json.count("{") - repair_json.count("}")
            open_brackets = repair_json.count("[") - repair_json.count("]")

            # Close arrays first, then objects
            repair_json += "]" * open_brackets
            repair_json += "}" * open_braces

            logger.debug(
                f"Completed truncated JSON with {open_brackets} ] and {open_braces} }}"
            )

        # Remove any text after the final closing brace
        last_brace = repair_json.rfind("}")
        if last_brace != -1:
            repair_json = repair_json[: last_brace + 1]

        if repair_json != json_text:
            logger.debug(f"Repaired JSON text for parsing: {repair_json}")

        return repair_json

    def _validate_complete_scoring(
        self, criterion_scores: list[CriterionScore], participants: list[str], context: DebateContext
    ) -> None:
        """Validate that judge provided complete scoring for all participants and criteria."""
        expected_combinations = len(participants) * len(self.criteria)
        actual_combinations = len(criterion_scores)

        if actual_combinations != expected_combinations:
            raise ValueError(
                f"Incomplete judge scoring: expected {expected_combinations} "
                f"criterion scores ({len(participants)} participants × {len(self.criteria)} criteria), "
                f"but got {actual_combinations}"
            )

        # Validate each participant has scores for all criteria
        participant_labels = self._get_participant_labels(participants, context)

        for participant_id in participants:
            side_label = participant_labels[participant_id]
            participant_criteria = {
                score.criterion.value
                for score in criterion_scores
                if score.participant_id == participant_id
            }
            expected_criteria = {c.value for c in self.criteria}

            if participant_criteria != expected_criteria:
                missing_criteria = expected_criteria - participant_criteria
                extra_criteria = participant_criteria - expected_criteria

                error_parts: list[str] = []
                if missing_criteria:
                    error_parts.append(f"missing: {missing_criteria}")
                if extra_criteria:
                    error_parts.append(f"unexpected: {extra_criteria}")

                raise ValueError(
                    f"Judge provided incomplete scoring for {side_label}: {', '.join(error_parts)}"
                )

    async def _query_and_update_judge_cost(self, judge_decision: "JudgeDecision") -> None:
        """Background task to query and update cost for a judge decision with generation_id."""
        if not judge_decision.generation_id:
            return

        try:
            # Wait a bit to ensure the generation is finalized on OpenRouter's end
            await asyncio.sleep(2.0)

            cost = await self.model_manager.query_generation_cost(self.judge_id, judge_decision.generation_id)
            if cost is not None:
                # Update the judge decision object
                judge_decision.cost = cost
                judge_decision.cost_queried_at = datetime.now().isoformat()

                logger.info(f"Updated cost for judge decision {judge_decision.generation_id}: ${cost}")
            else:
                logger.warning(f"Failed to retrieve cost for judge generation {judge_decision.generation_id}")

        except Exception as e:
            logger.error(f"Failed to query cost for judge generation {judge_decision.generation_id}: {e}")


# EnsembleJudge class deleted - ensemble logic moved to core.py coordinator
