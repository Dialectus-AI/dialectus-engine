"""Core debate engine for orchestrating AI model debates."""

from typing import Any
import asyncio
import logging
import time

from config.settings import AppConfig
from models.manager import ModelManager
from formats import format_registry, FormatPhase
from .types import DebatePhase, Position
from .models import DebateContext, DebateMessage
from .transcript import TranscriptManager

logger = logging.getLogger(__name__)


class DebateEngine:
    """Core debate orchestration engine."""

    def __init__(self, config: AppConfig, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.context: DebateContext | None = None
        self._system_prompts: dict[str, str] = {}
        self.format = format_registry.get_format(config.debate.format)

        # Initialize transcript manager if enabled
        self.transcript_manager: TranscriptManager | None = None
        if config.system.save_transcripts:
            from pathlib import Path

            transcript_dir = Path(config.system.transcript_dir)
            transcript_dir.mkdir(parents=True, exist_ok=True)
            db_path = transcript_dir / "debates.db"
            self.transcript_manager = TranscriptManager(str(db_path))

    async def initialize_debate(self, topic: str | None = None) -> DebateContext:
        """Initialize a new debate with the given topic."""
        debate_topic = topic or self.config.debate.topic

        # Register models with manager
        for model_id, model_config in self.config.models.items():
            self.model_manager.register_model(model_id, model_config)

        # Create debate context
        self.context = DebateContext(
            topic=debate_topic, participants=self.config.models.copy()
        )

        # Store format information in metadata
        self.context.metadata["format"] = self.config.debate.format
        self.context.metadata["word_limit"] = self.config.debate.word_limit

        # Store judging configuration in metadata
        if self.config.judging.judge_models:
            self.context.metadata["judge_models"] = self.config.judging.judge_models

        # Generate system prompts
        await self._generate_system_prompts()

        logger.info(f"Initialized debate: '{debate_topic}'")
        return self.context

    async def _generate_system_prompts(self) -> None:
        """Generate system prompts for each participant."""
        if not self.context:
            raise RuntimeError("No active debate context")

        base_prompt = self._get_base_system_prompt()

        # Use format-specific position assignments
        model_ids = list(self.context.participants.keys())
        position_assignments = self.format.get_position_assignments(model_ids)

        for model_id in model_ids:
            position = position_assignments.get(model_id, Position.NEUTRAL)
            personality = self.context.participants[model_id].personality

            prompt = self._create_participant_prompt(
                base_prompt, position, personality, model_id
            )
            self._system_prompts[model_id] = prompt

            logger.debug(f"Generated system prompt for {model_id} ({position.value})")

    def _get_base_system_prompt(self) -> str:
        """Get the base system prompt template."""
        if not self.context:
            raise RuntimeError("No active debate context")

        format_instructions = self.format.get_format_instructions()

        return f"""You are participating in a formal debate about: "{self.context.topic}"

DEBATE FORMAT: {self.format.name.upper()}
WORD LIMIT: {self.config.debate.word_limit} words per response

{format_instructions}

GENERAL RULES:
1. Stay focused on the topic
2. Respect the word limit
3. Provide evidence and reasoning
4. Address opponent's arguments
5. Maintain a respectful tone
6. Follow the debate format structure

RESPONSE FORMAT:
- Speak directly as your assigned role without any labels, prefixes, or announcements
- Use plain text without markdown formatting (avoid **bold**, *italics*, # headers, bullet points)
- Write in natural conversational style as if speaking live to an audience
- Do not add labels like "COUNTER:", "STATEMENT:", "OPENING:", etc. - just speak your argument
- Focus on content, not formatting or structure

You will be told your specific role and speaking context for each response."""

    def _create_participant_prompt(
        self, base_prompt: str, position: Position, personality: str, model_id: str
    ) -> str:
        """Create a participant-specific system prompt using role-based identity."""
        if not self.context:
            raise RuntimeError("No active debate context")

        # Get the proper role name for this format (Opposition, Proposition, etc.)
        model_ids = list(self.context.participants.keys())
        side_labels = self.format.get_side_labels(model_ids)
        role_name = side_labels.get(model_id, position.value.title())

        position_guidance = {
            Position.PRO: f"You ARE the {role_name} speaker. You support the motion and believe it is correct.",
            Position.CON: f"You ARE the {role_name} speaker. You oppose the motion and believe it is wrong.",
        }

        personality_guidance = {
            "analytical": "Your speaking style is data-driven, logical, and systematic.",
            "passionate": "Your speaking style is persuasive and emotionally engaging while factual.",
            "scholarly": "Your speaking style references academic sources and theoretical frameworks.",
            "practical": "Your speaking style emphasizes real-world applications and practical implications.",
            "neutral": "Your speaking style is balanced with measured reasoning.",
        }

        role_instruction = position_guidance.get(position, f"You ARE the {role_name} speaker.")
        style_instruction = personality_guidance.get(personality, "Speak in your natural style.")

        return f"""{base_prompt}

YOUR ROLE: {role_instruction}
You are not describing what {role_name} would say - you ARE {role_name} speaking directly.
Speak as if you are standing at the podium addressing the audience.
Do not announce your role, add labels, or use prefixes like "COUNTER:" or "STATEMENT:" - simply speak your argument.

YOUR SPEAKING STYLE: {style_instruction}

Remember: You are embodying the {role_name} position throughout this debate. Speak naturally and directly as that person would speak."""

    async def conduct_round(self, phase: DebatePhase) -> list[DebateMessage]:
        """Conduct a single round of the debate."""
        if not self.context:
            raise RuntimeError("No active debate context")

        self.context.current_phase = phase
        round_messages = []

        # Determine speaking order based on phase
        speakers = await self._get_speaking_order(phase)

        for speaker_id in speakers:
            message = await self._get_speaker_response(speaker_id, phase)
            round_messages.append(message)
            self.context.messages.append(message)

            logger.info(
                f"Round {self.context.current_round}, {phase.value}: {speaker_id}"
            )

        return round_messages

    async def conduct_format_round(
        self, format_phase: FormatPhase
    ) -> list[DebateMessage]:
        """Conduct a round using a specific format phase."""
        if not self.context:
            raise RuntimeError("No active debate context")

        self.context.current_phase = format_phase.phase
        round_messages = []

        # Use format-defined speaking order
        for speaker_id in format_phase.speaking_order:
            message = await self._get_format_speaker_response(speaker_id, format_phase)
            round_messages.append(message)
            self.context.messages.append(message)

            logger.info(
                f"Round {self.context.current_round}, {format_phase.name}: {speaker_id}"
            )

        return round_messages

    async def _get_speaking_order(self, phase: DebatePhase) -> list[str]:
        """Determine speaking order for the current phase."""
        if not self.context:
            raise RuntimeError("No active debate context")
        model_ids = list(self.context.participants.keys())

        if phase in [DebatePhase.OPENING, DebatePhase.CLOSING]:
            return model_ids  # Pro speaks first, then con
        elif phase == DebatePhase.REBUTTAL:
            return list(reversed(model_ids))  # Con speaks first in rebuttals
        elif phase == DebatePhase.CROSS_EXAM:
            # Alternating questions
            return model_ids * 2  # Each gets to ask and answer

        return model_ids

    async def _get_speaker_response(
        self, speaker_id: str, phase: DebatePhase
    ) -> DebateMessage:
        """Get a response from a specific speaker."""
        if not self.context:
            raise RuntimeError("No active debate context")

        # Determine position
        positions = [Position.PRO, Position.CON]
        model_ids = list(self.context.participants.keys())
        speaker_position = (
            positions[model_ids.index(speaker_id)]
            if speaker_id in model_ids[:2]
            else Position.NEUTRAL
        )

        # Build conversation context
        messages = self._build_conversation_context(speaker_id, phase)

        # Generate response with timing
        start_time = time.time()
        async with self.model_manager.model_session(speaker_id):
            response_content = await self.model_manager.generate_response(
                speaker_id,
                messages,
                max_tokens=min(
                    self.config.models[speaker_id].max_tokens,
                    self._calculate_max_tokens(),
                ),
            )
        generation_time = time.time() - start_time

        # Clean the response to remove any echoed prefixes
        cleaned_response = self._clean_model_response(response_content, speaker_id)

        return DebateMessage(
            speaker_id=speaker_id,
            position=speaker_position,
            phase=phase,
            round_number=self.context.current_round,
            content=cleaned_response,
            metadata={"generation_time_ms": int(generation_time * 1000)},
        )

    async def _get_format_speaker_response(
        self, speaker_id: str, format_phase: FormatPhase
    ) -> DebateMessage:
        """Get a response from a specific speaker using format phase."""
        if not self.context:
            raise RuntimeError("No active debate context")

        # Determine position
        model_ids = list(self.context.participants.keys())
        position_assignments = self.format.get_position_assignments(model_ids)
        speaker_position = position_assignments.get(speaker_id, Position.NEUTRAL)

        # Build conversation context with format-specific instruction
        messages = self._build_format_conversation_context(speaker_id, format_phase)

        # Calculate max tokens with format time multiplier
        base_max_tokens = min(
            self.config.models[speaker_id].max_tokens, self._calculate_max_tokens()
        )
        adjusted_max_tokens = int(base_max_tokens * format_phase.time_multiplier)

        # Generate response with timing and detailed error handling
        start_time = time.time()
        try:
            logger.info(
                f"CORE ENGINE: Starting response generation for {speaker_id} ({self.config.models[speaker_id].name}) via {self.config.models[speaker_id].provider}"
            )
            async with self.model_manager.model_session(speaker_id):
                response_content = await self.model_manager.generate_response(
                    speaker_id, messages, max_tokens=adjusted_max_tokens
                )
            generation_time = time.time() - start_time
            logger.info(
                f"CORE ENGINE: Successfully generated {len(response_content)} chars for {speaker_id} in {generation_time:.2f}s"
            )
        except Exception as e:
            generation_time = time.time() - start_time
            model_config = self.config.models[speaker_id]
            error_msg = f"CORE ENGINE FAILURE: {speaker_id} ({model_config.name}) via {model_config.provider} failed after {generation_time:.2f}s: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Clean the response to remove any echoed prefixes
        cleaned_response = self._clean_model_response(response_content, speaker_id)

        return DebateMessage(
            speaker_id=speaker_id,
            position=speaker_position,
            phase=format_phase.phase,
            round_number=self.context.current_round,
            content=cleaned_response,
            metadata={"generation_time_ms": int(generation_time * 1000)},
        )

    def _build_conversation_context(
        self, speaker_id: str, phase: DebatePhase
    ) -> list[dict[str, str]]:
        """Build conversation context for the model."""
        if not self.context:
            raise RuntimeError("No active debate context")

        messages = []

        # Add system prompt
        if speaker_id in self._system_prompts:
            messages.append(
                {"role": "system", "content": self._system_prompts[speaker_id]}
            )

        # Add role-focused phase instruction
        model_ids = list(self.context.participants.keys())
        side_labels = self.format.get_side_labels(model_ids)
        role_name = side_labels.get(speaker_id, "Speaker")

        phase_instruction = self._get_phase_instruction(phase)
        messages.append(
            {
                "role": "system",
                "content": f"You are now speaking as {role_name}. {phase_instruction}",
            }
        )

        # Add relevant debate history
        for msg in self.context.messages[-6:]:  # Last 6 messages for context
            role = "assistant" if msg.speaker_id == speaker_id else "user"

            # Use a cleaner format that's less likely to be repeated by small models
            if role == "assistant":
                # When showing the model its own previous messages, just show the content
                messages.append({"role": role, "content": msg.content})
            else:
                # When showing opponent messages, use a simple format
                messages.append(
                    {
                        "role": role,
                        "content": f"Opponent ({msg.position.value}): {msg.content}",
                    }
                )

        # Add current turn prompt with role focus
        turn_prompt = f"Now speak as {role_name}. Stay under {self.config.debate.word_limit} words."
        messages.append({"role": "user", "content": turn_prompt})

        return messages

    def _build_format_conversation_context(
        self, speaker_id: str, format_phase: FormatPhase
    ) -> list[dict[str, str]]:
        """Build conversation context for the model using format phase."""
        if not self.context:
            raise RuntimeError("No active debate context")

        messages = []

        # Add system prompt
        if speaker_id in self._system_prompts:
            messages.append(
                {"role": "system", "content": self._system_prompts[speaker_id]}
            )

        # Add role-focused phase instruction instead of procedural description
        model_ids = list(self.context.participants.keys())
        side_labels = self.format.get_side_labels(model_ids)
        role_name = side_labels.get(speaker_id, "Speaker")

        # Simplify the instruction to focus on role, not procedure
        simplified_instruction = format_phase.instruction.replace("As the Proposition,", "").replace("As the Opposition,", "").strip()

        messages.append(
            {
                "role": "system",
                "content": f"You are now speaking as {role_name}. {simplified_instruction}",
            }
        )

        # Add relevant debate history
        for msg in self.context.messages[-6:]:  # Last 6 messages for context
            role = "assistant" if msg.speaker_id == speaker_id else "user"

            # Use a cleaner format that's less likely to be repeated by small models
            if role == "assistant":
                # When showing the model its own previous messages, just show the content
                messages.append({"role": role, "content": msg.content})
            else:
                # When showing opponent messages, use a simple format
                messages.append(
                    {
                        "role": role,
                        "content": f"Opponent ({msg.position.value}): {msg.content}",
                    }
                )

        # Add current turn prompt with role focus
        word_limit = int(self.config.debate.word_limit * format_phase.time_multiplier)
        turn_prompt = f"Now speak as {role_name}. Stay under {word_limit} words."
        messages.append({"role": "user", "content": turn_prompt})

        return messages

    def _clean_model_response(self, response: str, speaker_id: str) -> str:
        """Clean model response by removing any echoed prefixes or formatting."""
        import re

        # Get the model name for this speaker
        display_name = speaker_id
        if self.context and speaker_id in self.context.participants:
            display_name = self.context.participants[speaker_id].name

        # Debug logging for empty responses
        if not response.strip():
            logger.warning(
                f"Model {display_name} ({speaker_id}) returned empty response before cleaning"
            )

        cleaned = response.strip()

        # Remove position statement prefixes like "Proposition Opening Statement", etc.
        position_prefixes = [
            r"^(?:proposition|opposition|pro|con)\s+(?:opening|rebuttal|closing)\s+(?:statement|argument)[\:\-\s]*",
            r"^(?:opening|rebuttal|closing)\s+(?:statement|argument)[\:\-\s]*",
            r"^(?:proposition|opposition|pro|con)[\:\-\s]+",
            r"^\*\*(?:proposition|opposition|pro|con)\s+(?:opening|rebuttal|closing)\s+(?:statement|argument)\*\*[\:\-\s]*",  # **Position Statement**
            r"^(?:Phase\s+\d+[\:\-\s]*)?(?:proposition|opposition|pro|con)\s+(?:opening|rebuttal|closing)[\:\-\s]*",  # Phase 1: Proposition Opening
        ]

        for pattern in position_prefixes:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

        # Remove any echoed conversation prefixes that the model might repeat
        conversation_patterns = [
            rf"\[{re.escape(speaker_id)}\s*-\s*\w+\]:\s*",  # [model_a - pro]:
            rf"\[{re.escape(display_name)}\s*-\s*\w+\]:\s*",  # [llama3.2:latest - pro]:
            r"\[.*?\s*-\s*\w+\]:\s*",  # Any [something - position]: pattern
        ]

        for pattern in conversation_patterns:
            # Keep removing the pattern until no more matches (handles multiple prefixes)
            while re.match(pattern, cleaned, re.IGNORECASE):
                cleaned = re.sub(
                    pattern, "", cleaned, count=1, flags=re.IGNORECASE
                ).strip()

        # Remove markdown formatting
        # Remove headers (# ## ###)
        cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)

        # Remove bold and italic markdown (**text**, *text*)
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)  # **bold** -> bold
        cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)  # *italic* -> italic

        # Remove any remaining asterisks that might be formatting attempts
        cleaned = re.sub(
            r"^\*+\s*", "", cleaned, flags=re.MULTILINE
        )  # Remove lines starting with *

        # Debug logging if cleaning resulted in empty response
        final_cleaned = cleaned.strip()
        if response.strip() and not final_cleaned:
            logger.warning(
                f"Model {display_name} ({speaker_id}) response was cleaned to empty. Original: {repr(response)}, Cleaned: {repr(final_cleaned)}"
            )

        return final_cleaned

    def _get_phase_instruction(self, phase: DebatePhase) -> str:
        """Get instruction text for the current phase."""
        instructions = {
            DebatePhase.OPENING: "Present your opening arguments clearly and persuasively.",
            DebatePhase.REBUTTAL: "Address the opponent's arguments and strengthen your position.",
            DebatePhase.CROSS_EXAM: "Ask probing questions or provide direct answers.",
            DebatePhase.CLOSING: "Summarize your case and make your final persuasive appeal.",
        }
        return instructions.get(phase, "Participate according to the debate format.")

    def _calculate_max_tokens(self) -> int:
        """Calculate maximum tokens based on word limit."""
        # Rough estimate: 1 token ≈ 0.75 words
        return int(self.config.debate.word_limit * 1.33)

    async def run_full_debate(self) -> DebateContext:
        """Run a complete debate from start to finish."""
        if not self.context:
            raise RuntimeError("Debate not initialized")

        logger.info("Starting full debate execution")

        # Start timing the debate
        debate_start_time = time.time()

        # Get format-specific phases
        model_ids = list(self.context.participants.keys())
        format_phases = self.format.get_phases(model_ids)

        # Group format phases by their DebatePhase enum to determine round numbers
        phase_to_round = {}
        current_round = 1
        last_phase = None

        for format_phase in format_phases:
            # If we encounter a new DebatePhase enum, increment the round
            if format_phase.phase != last_phase:
                phase_to_round[format_phase.phase] = current_round
                current_round += 1
                last_phase = format_phase.phase
            else:
                # Same phase type, use existing round number
                phase_to_round[format_phase.phase] = phase_to_round[format_phase.phase]

        # Execute each format phase with correct round numbers
        for format_phase in format_phases:
            self.context.current_round = phase_to_round[format_phase.phase]

            await self.conduct_format_round(format_phase)

            # Brief pause between phases
            await asyncio.sleep(0.5)

        # Calculate total debate time
        total_debate_time_ms = int((time.time() - debate_start_time) * 1000)

        self.context.current_phase = DebatePhase.COMPLETED
        logger.info(f"Debate completed successfully in {total_debate_time_ms}ms")

        # Store debate time for later transcript saving
        self.context.metadata["total_debate_time_ms"] = total_debate_time_ms

        return self.context

    def save_transcript_with_judge_result(self, judge_result=None) -> None:
        """Save transcript to database with optional judge result (single decision or ensemble)."""
        if not self.transcript_manager or not self.context:
            return

        try:
            # Get debate time from metadata
            total_debate_time_ms = self.context.metadata.get("total_debate_time_ms", 0)

            transcript_id = self.transcript_manager.save_transcript(
                self.context, total_debate_time_ms
            )
            logger.info(f"Transcript saved to database with ID {transcript_id}")
            self.context.metadata["transcript_id"] = transcript_id

            # Save judge result to database if provided
            if judge_result:
                if (
                    isinstance(judge_result, dict)
                    and judge_result.get("type") == "ensemble"
                ):
                    # Ensemble case - save individual decisions and ensemble summary
                    self._save_ensemble_result(transcript_id, judge_result)
                else:
                    # Single judge case
                    self._save_individual_decision(transcript_id, judge_result)
            else:
                logger.info(
                    f"No judge result provided - transcript {transcript_id} saved without judge evaluation"
                )

        except Exception as e:
            logger.error(f"Failed to save transcript: {e}")
            raise

    def _save_individual_decision(self, transcript_id: int, judge_decision) -> int:
        """Save a single judge decision to the database."""
        if not self.transcript_manager:
            raise RuntimeError("Transcript manager not initialized")

        logger.info(
            f"Saving individual judge decision to database for transcript {transcript_id}"
        )

        # Save to judge_decisions table using strongly-typed JudgeDecision object
        decision_id = self.transcript_manager.db_manager.save_judge_decision(
            transcript_id, judge_decision
        )
        logger.info(
            f"Successfully saved judge decision {decision_id} to database for transcript {transcript_id}"
        )
        return decision_id

    def _save_ensemble_result(self, transcript_id: int, ensemble_result: dict) -> None:
        """Save ensemble result - individual decisions + ensemble summary."""

        decisions = ensemble_result["decisions"]
        ensemble_summary = ensemble_result["ensemble_summary"]

        logger.info(
            f"Saving ensemble result with {len(decisions)} individual decisions for transcript {transcript_id}"
        )

        # Save each individual decision to judge_decisions table
        decision_ids = []
        for i, decision in enumerate(decisions):
            decision_id = self._save_individual_decision(transcript_id, decision)
            decision_ids.append(decision_id)
            logger.info(
                f"Saved individual judge decision {i+1}/{len(decisions)} with ID {decision_id}"
            )

        # Save ensemble summary to ensemble_summary table
        ensemble_summary_with_metadata = {
            **ensemble_summary,
            "participating_judge_decision_ids": ",".join(
                map(str, decision_ids)
            ),  # Store as comma-separated string
        }

        if not self.transcript_manager:
            raise RuntimeError("Transcript manager not initialized")

        ensemble_id = self.transcript_manager.db_manager.save_ensemble_summary(
            transcript_id, ensemble_summary_with_metadata
        )
        logger.info(
            f"Successfully saved ensemble summary {ensemble_id} for transcript {transcript_id} with {len(decisions)} individual decisions"
        )

    def get_transcript_for_judging(self) -> str | None:
        """Get formatted transcript suitable for AI judging."""
        if not self.context or not self.transcript_manager:
            return None
        return self.transcript_manager.format_transcript_for_judging(self.context)

    async def judge_debate_with_judges(self, judges: list) -> Any | None:
        """Judge the completed debate using a list of AI judges."""
        if not self.context or self.context.current_phase != DebatePhase.COMPLETED:
            raise RuntimeError("Cannot judge incomplete debate")

        if not judges:
            logger.info("No judges provided - skipping automated judging")
            return None

        from judges.ensemble_utils import calculate_ensemble_result

        if len(judges) == 1:
            # Single judge case
            judge = judges[0]
            logger.info(f"Judging debate with single judge: {judge.name}")
            decision = await judge.evaluate_debate(self.context)
            return decision

        else:
            # Multiple judges - ensemble case
            logger.info(
                f"Judging debate with {len(judges)} judges for ensemble evaluation"
            )

            decisions = []
            failed_judges = []

            for i, judge in enumerate(judges):
                try:
                    # Add slight randomness to avoid identical evaluations
                    import random

                    random_seed = random.random()
                    logger.info(
                        f"Starting evaluation for judge {i+1}: {judge.judge_model_name} (seed: {random_seed:.4f})"
                    )

                    decision = await judge.evaluate_debate(self.context)
                    decisions.append(decision)
                    logger.info(
                        f"Judge {judge.name} completed successfully with winner: {decision.winner_id}, margin: {decision.winner_margin:.2f}"
                    )

                    # Debug: Log first criterion score to verify different evaluations
                    if decision.criterion_scores:
                        first_score = decision.criterion_scores[0]
                        logger.info(
                            f"Judge {i+1} first score sample: {first_score.participant_id} - {first_score.criterion.value}: {first_score.score} - {first_score.feedback[:50]}..."
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
                "ensemble_summary": calculate_ensemble_result(
                    decisions, self.context
                ),
            }
