"""Core debate engine for orchestrating AI model debates."""

from typing import Dict, List, Optional, Any
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
        self.context: Optional[DebateContext] = None
        self._system_prompts: Dict[str, str] = {}
        self.format = format_registry.get_format(config.debate.format)
        
        # Initialize transcript manager if enabled
        self.transcript_manager: Optional[TranscriptManager] = None
        if config.system.save_transcripts:
            from pathlib import Path
            transcript_dir = Path(config.system.transcript_dir)
            transcript_dir.mkdir(parents=True, exist_ok=True)
            db_path = transcript_dir / "debates.db"
            self.transcript_manager = TranscriptManager(str(db_path))
        
    async def initialize_debate(self, topic: Optional[str] = None) -> DebateContext:
        """Initialize a new debate with the given topic."""
        debate_topic = topic or self.config.debate.topic
        
        # Register models with manager
        for model_id, model_config in self.config.models.items():
            self.model_manager.register_model(model_id, model_config)
        
        # Create debate context
        self.context = DebateContext(
            topic=debate_topic,
            participants=self.config.models.copy()
        )
        
        # Store format information in metadata
        self.context.metadata['format'] = self.config.debate.format
        self.context.metadata['word_limit'] = self.config.debate.word_limit
        
        # Store judging configuration in metadata
        self.context.metadata['judging_method'] = self.config.judging.method
        if self.config.judging.judge_model:
            self.context.metadata['judge_model'] = self.config.judging.judge_model
        
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

Current phase will be specified in each turn."""
    
    def _create_participant_prompt(
        self, 
        base_prompt: str, 
        position: Position, 
        personality: str, 
        model_id: str
    ) -> str:
        """Create a participant-specific system prompt."""
        position_guidance = {
            Position.PRO: "You are arguing FOR the topic. Present strong supporting arguments.",
            Position.CON: "You are arguing AGAINST the topic. Present strong opposing arguments."
        }
        
        personality_guidance = {
            "analytical": "Focus on data, logic, and systematic reasoning.",
            "passionate": "Be persuasive and emotionally engaging while maintaining facts.",
            "scholarly": "Reference academic sources and theoretical frameworks.",
            "practical": "Emphasize real-world applications and practical implications.",
            "neutral": "Present balanced arguments with measured reasoning."
        }
        
        return f"""{base_prompt}

POSITION: {position.value.upper()}
{position_guidance.get(position, "")}

PERSONALITY: {personality.title()}
{personality_guidance.get(personality, "")}

Remember: You are {model_id} and should maintain consistency in your argumentation style throughout the debate."""
    
    async def conduct_round(self, phase: DebatePhase) -> List[DebateMessage]:
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
            
            logger.info(f"Round {self.context.current_round}, {phase.value}: {speaker_id}")
        
        return round_messages
    
    async def conduct_format_round(self, format_phase: FormatPhase) -> List[DebateMessage]:
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
            
            logger.info(f"Round {self.context.current_round}, {format_phase.name}: {speaker_id}")
        
        return round_messages
    
    async def _get_speaking_order(self, phase: DebatePhase) -> List[str]:
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
    
    async def _get_speaker_response(self, speaker_id: str, phase: DebatePhase) -> DebateMessage:
        """Get a response from a specific speaker."""
        if not self.context:
            raise RuntimeError("No active debate context")
        
        # Determine position
        positions = [Position.PRO, Position.CON]
        model_ids = list(self.context.participants.keys())
        speaker_position = positions[model_ids.index(speaker_id)] if speaker_id in model_ids[:2] else Position.NEUTRAL
        
        # Build conversation context
        messages = self._build_conversation_context(speaker_id, phase)
        
        # Generate response with timing
        start_time = time.time()
        async with self.model_manager.model_session(speaker_id):
            response_content = await self.model_manager.generate_response(
                speaker_id, 
                messages,
                max_tokens=min(self.config.models[speaker_id].max_tokens, self._calculate_max_tokens())
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
            metadata={'generation_time_ms': int(generation_time * 1000)}
        )
    
    async def _get_format_speaker_response(self, speaker_id: str, format_phase: FormatPhase) -> DebateMessage:
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
            self.config.models[speaker_id].max_tokens, 
            self._calculate_max_tokens()
        )
        adjusted_max_tokens = int(base_max_tokens * format_phase.time_multiplier)
        
        # Generate response with timing
        start_time = time.time()
        async with self.model_manager.model_session(speaker_id):
            response_content = await self.model_manager.generate_response(
                speaker_id, 
                messages,
                max_tokens=adjusted_max_tokens
            )
        generation_time = time.time() - start_time
        
        # Clean the response to remove any echoed prefixes
        cleaned_response = self._clean_model_response(response_content, speaker_id)
        
        return DebateMessage(
            speaker_id=speaker_id,
            position=speaker_position,
            phase=format_phase.phase,
            round_number=self.context.current_round,
            content=cleaned_response,
            metadata={'generation_time_ms': int(generation_time * 1000)}
        )
    
    def _build_conversation_context(self, speaker_id: str, phase: DebatePhase) -> List[Dict[str, str]]:
        """Build conversation context for the model."""
        if not self.context:
            raise RuntimeError("No active debate context")
            
        messages = []
        
        # Add system prompt
        if speaker_id in self._system_prompts:
            messages.append({
                "role": "system",
                "content": self._system_prompts[speaker_id]
            })
        
        # Add phase-specific instruction
        phase_instruction = self._get_phase_instruction(phase)
        messages.append({
            "role": "system", 
            "content": f"CURRENT PHASE: {phase.value.upper()}\n{phase_instruction}"
        })
        
        # Add relevant debate history
        for msg in self.context.messages[-6:]:  # Last 6 messages for context
            role = "assistant" if msg.speaker_id == speaker_id else "user"
            
            # Get actual model name for display
            display_name = msg.speaker_id
            if msg.speaker_id in self.context.participants:
                display_name = self.context.participants[msg.speaker_id].name
            
            # Use a cleaner format that's less likely to be repeated by small models
            if role == "assistant":
                # When showing the model its own previous messages, just show the content
                messages.append({
                    "role": role,
                    "content": msg.content
                })
            else:
                # When showing opponent messages, use a simple format
                messages.append({
                    "role": role,
                    "content": f"Opponent ({msg.position.value}): {msg.content}"
                })
        
        # Add current turn prompt
        turn_prompt = f"Your turn to speak in the {phase.value} phase. Remember to stay under {self.config.debate.word_limit} words."
        messages.append({"role": "user", "content": turn_prompt})
        
        return messages
    
    def _build_format_conversation_context(self, speaker_id: str, format_phase: FormatPhase) -> List[Dict[str, str]]:
        """Build conversation context for the model using format phase."""
        if not self.context:
            raise RuntimeError("No active debate context")
            
        messages = []
        
        # Add system prompt
        if speaker_id in self._system_prompts:
            messages.append({
                "role": "system",
                "content": self._system_prompts[speaker_id]
            })
        
        # Add format-specific phase instruction
        messages.append({
            "role": "system", 
            "content": f"CURRENT PHASE: {format_phase.name.upper()}\n{format_phase.instruction}"
        })
        
        # Add relevant debate history
        for msg in self.context.messages[-6:]:  # Last 6 messages for context
            role = "assistant" if msg.speaker_id == speaker_id else "user"
            
            # Get actual model name for display
            display_name = msg.speaker_id
            if msg.speaker_id in self.context.participants:
                display_name = self.context.participants[msg.speaker_id].name
            
            # Use a cleaner format that's less likely to be repeated by small models
            if role == "assistant":
                # When showing the model its own previous messages, just show the content
                messages.append({
                    "role": role,
                    "content": msg.content
                })
            else:
                # When showing opponent messages, use a simple format
                messages.append({
                    "role": role,
                    "content": f"Opponent ({msg.position.value}): {msg.content}"
                })
        
        # Add current turn prompt
        word_limit = int(self.config.debate.word_limit * format_phase.time_multiplier)
        turn_prompt = f"Your turn to speak in the {format_phase.name} phase. Remember to stay under {word_limit} words."
        messages.append({"role": "user", "content": turn_prompt})
        
        return messages
    
    def _clean_model_response(self, response: str, speaker_id: str) -> str:
        """Clean model response by removing any echoed prefixes or formatting."""
        import re
        
        # Get the model name for this speaker
        display_name = speaker_id
        if self.context and speaker_id in self.context.participants:
            display_name = self.context.participants[speaker_id].name
        
        # Remove any echoed conversation prefixes that the model might repeat
        patterns_to_remove = [
            rf"\[{re.escape(speaker_id)}\s*-\s*\w+\]:\s*",  # [model_a - pro]:
            rf"\[{re.escape(display_name)}\s*-\s*\w+\]:\s*",  # [llama3.2:latest - pro]:
            r"\[.*?\s*-\s*\w+\]:\s*",  # Any [something - position]: pattern
        ]
        
        cleaned = response.strip()
        for pattern in patterns_to_remove:
            # Keep removing the pattern until no more matches (handles multiple prefixes)
            while re.match(pattern, cleaned, re.IGNORECASE):
                cleaned = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE).strip()
        
        return cleaned

    def _get_phase_instruction(self, phase: DebatePhase) -> str:
        """Get instruction text for the current phase."""
        instructions = {
            DebatePhase.OPENING: "Present your opening arguments clearly and persuasively.",
            DebatePhase.REBUTTAL: "Address the opponent's arguments and strengthen your position.",
            DebatePhase.CROSS_EXAM: "Ask probing questions or provide direct answers.",
            DebatePhase.CLOSING: "Summarize your case and make your final persuasive appeal."
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
        
        # Each format phase gets its own sequential number
        phase_number = 1
        
        for format_phase in format_phases:
            self.context.current_round = phase_number
            
            await self.conduct_format_round(format_phase)
            
            # Move to next phase
            phase_number += 1
            
            # Brief pause between phases
            await asyncio.sleep(0.5)
        
        # Calculate total debate time
        total_debate_time_ms = int((time.time() - debate_start_time) * 1000)
        
        self.context.current_phase = DebatePhase.COMPLETED
        logger.info(f"Debate completed successfully in {total_debate_time_ms}ms")
        
        # Save transcript if enabled
        if self.transcript_manager:
            try:
                transcript_id = self.transcript_manager.save_transcript(self.context, total_debate_time_ms)
                logger.info(f"Transcript saved to database with ID {transcript_id}")
                self.context.metadata['transcript_id'] = transcript_id
            except Exception as e:
                logger.error(f"Failed to save transcript: {e}")
        
        return self.context
    
    def _serialize_judge_decision(self, decision) -> Dict[str, Any]:
        """Serialize JudgeDecision to a JSON-compatible dictionary."""
        return {
            'winner_id': decision.winner_id,
            'winner_margin': decision.winner_margin,
            'overall_feedback': decision.overall_feedback,
            'reasoning': decision.reasoning,
            'criterion_scores': [
                {
                    'criterion': score.criterion.value if hasattr(score.criterion, 'value') else str(score.criterion),
                    'participant_id': score.participant_id,
                    'score': score.score,
                    'feedback': score.feedback
                }
                for score in decision.criterion_scores
            ],
            'metadata': decision.metadata or {}
        }
    
    def get_transcript_for_judging(self) -> Optional[str]:
        """Get formatted transcript suitable for AI judging."""
        if not self.context or not self.transcript_manager:
            return None
        return self.transcript_manager.format_transcript_for_judging(self.context)
    
    async def judge_debate(self, judge) -> Optional[Any]:
        """Judge the completed debate using the provided judge."""
        if not self.context or self.context.current_phase != DebatePhase.COMPLETED:
            raise RuntimeError("Cannot judge incomplete debate")
        
        if judge is None:
            logger.info("No judge provided - skipping automated judging")
            return None
        
        logger.info(f"Judging debate with {judge.name}")
        decision = await judge.evaluate_debate(self.context)
        
        # Store judge decision in context metadata as serializable dict
        self.context.metadata['judge_decision'] = self._serialize_judge_decision(decision)
        
        # Update transcript with judge decision if transcript was saved
        if self.transcript_manager and 'transcript_id' in self.context.metadata:
            try:
                transcript_id = self.context.metadata['transcript_id']
                # Re-save the transcript with updated metadata including judge decision
                self.transcript_manager.save_transcript_update(transcript_id, self.context.metadata)
                logger.info(f"Updated transcript {transcript_id} with judge decision")
            except Exception as e:
                logger.error(f"Failed to update transcript with judge decision: {e}")
        
        return decision