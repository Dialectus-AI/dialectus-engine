"""Core debate engine for orchestrating AI model debates."""

import asyncio
import logging
import time
from typing import Any, Callable, Awaitable

from config.settings import AppConfig
from models.manager import ModelManager
from .types import DebatePhase
from .models import DebateContext
from .prompt_builder import PromptBuilder
from .context_builder import ContextBuilder
from .response_handler import ResponseHandler
from .round_manager import RoundManager
from .judge_coordinator import JudgeCoordinator

logger = logging.getLogger(__name__)


class DebateEngine:
    """Core debate orchestration engine."""

    def __init__(self, config: AppConfig, model_manager: ModelManager):
        """Initialize the debate engine.

        Args:
            config: Application configuration
            model_manager: ModelManager instance
        """
        from formats import format_registry

        self.config = config
        self.model_manager = model_manager
        self.context: DebateContext | None = None
        self.format = format_registry.get_format(config.debate.format)

        # Initialize specialized components
        self.prompt_builder = PromptBuilder(config, self.format)
        self.context_builder = ContextBuilder(config, self.format, self.prompt_builder)
        self.response_handler = ResponseHandler(
            config, model_manager, self.format, self.context_builder
        )
        self.round_manager = RoundManager(self.format, self.response_handler)
        self.judge_coordinator = JudgeCoordinator()

    async def initialize_debate(self, topic: str | None = None) -> DebateContext:
        """Initialize a new debate with the given topic.

        Args:
            topic: Optional debate topic (uses config topic if not provided)

        Returns:
            Initialized debate context
        """
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
        await self.prompt_builder.generate_system_prompts(self.context)

        logger.info(f"Initialized debate: '{debate_topic}'")
        return self.context

    async def run_full_debate(
        self,
        phase_callback: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
        message_callback: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
        chunk_callback: Callable[[str, bool], Awaitable[None]] | None = None,
    ) -> DebateContext:
        """Run a complete debate from start to finish.

        Args:
            phase_callback: Called when phases start/complete with event type and data
            message_callback: Called when messages start/complete with event type and data
            chunk_callback: Called for streaming chunks during message generation

        Callback event types:
            phase_callback:
                - "phase_started": {"phase": str, "instruction": str, "current_phase": int, "total_phases": int}
            message_callback:
                - "message_start": {"message_id": str, "speaker_id": str, "position": str, "phase": str, "round_number": int}
                - "message_complete": {"message_id": str, "speaker_id": str, "position": str, "content": str, ...}
            chunk_callback:
                - chunk: str, is_complete: bool (same as streaming API)

        Returns:
            Completed debate context
        """
        if not self.context:
            raise RuntimeError("Debate not initialized")

        logger.info("Starting full debate execution")

        # Start timing the debate
        debate_start_time = time.time()

        # Get format-specific phases
        model_ids = list(self.context.participants.keys())
        format_phases = self.format.get_phases(model_ids)
        total_phases = len(format_phases)

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
        for phase_index, format_phase in enumerate(format_phases):
            self.context.current_round = phase_to_round[format_phase.phase]

            # Invoke phase callback if provided
            if phase_callback:
                await phase_callback("phase_started", {
                    "phase": format_phase.name,
                    "instruction": format_phase.instruction,
                    "current_phase": self.context.current_round,
                    "total_phases": total_phases,
                    "progress_percentage": round((phase_index / total_phases) * 100),
                })

            # If streaming callbacks provided, use streaming version
            if message_callback or chunk_callback:
                await self.round_manager.conduct_format_round_stream(
                    format_phase,
                    self.context,
                    message_callback=message_callback,
                    chunk_callback=chunk_callback
                )
            else:
                await self.round_manager.conduct_format_round(format_phase, self.context)

            # Brief pause between phases
            await asyncio.sleep(0.5)

        # Calculate total debate time
        total_debate_time_ms = int((time.time() - debate_start_time) * 1000)

        self.context.current_phase = DebatePhase.COMPLETED
        logger.info(f"Debate completed successfully in {total_debate_time_ms}ms")

        # Store debate time for later transcript saving
        self.context.metadata["total_debate_time_ms"] = total_debate_time_ms

        return self.context

    async def judge_debate_with_judges(self, judges: list) -> Any | None:
        """Judge the completed debate using a list of AI judges.

        Args:
            judges: List of judge instances

        Returns:
            Decision from single judge or ensemble result from multiple judges
        """
        if not self.context:
            raise RuntimeError("No active debate context")

        return await self.judge_coordinator.judge_debate_with_judges(
            self.context, judges
        )
