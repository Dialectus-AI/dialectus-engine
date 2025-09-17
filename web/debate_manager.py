"""FastAPI web application for Dialectus AI Debate System."""

import asyncio
import uuid
from typing import Dict, List, Any
import logging

from fastapi import HTTPException, WebSocket

from config.settings import get_default_config
from models.manager import ModelManager
from debate_engine.core import DebateEngine
from formats import format_registry
from judges.factory import create_judges
from web.debate_setup_request import DebateSetupRequest

logger = logging.getLogger(__name__)


class DebateManager:
    """Manages active debates and WebSocket connections."""

    def __init__(self):
        self.active_debates: Dict[str, Dict[str, Any]] = {}
        self.connections: Dict[str, List[WebSocket]] = {}

    async def create_debate(self, setup: DebateSetupRequest) -> str:
        """Create a new debate session."""
        debate_id = str(uuid.uuid4())

        # Create configuration from existing system
        base_config = get_default_config()
        base_config.debate.topic = setup.topic
        # Validate format exists in registry
        if setup.format not in format_registry.list_formats():
            available_formats = ", ".join(format_registry.list_formats())
            raise HTTPException(
                status_code=400,
                detail=f"Invalid debate format: {setup.format}. Available formats: {available_formats}",
            )
        base_config.debate.format = setup.format  # type: ignore[assignment]
        base_config.debate.word_limit = setup.word_limit
        base_config.models = setup.models

        # Store judge configuration directly
        base_config.judging.judge_models = setup.judge_models or []
        if setup.judge_provider:
            base_config.judging.judge_provider = setup.judge_provider

        # Initialize components
        model_manager = ModelManager(base_config.system)
        debate_engine = DebateEngine(base_config, model_manager)

        # Store debate info
        self.active_debates[debate_id] = {
            "id": debate_id,
            "config": base_config,
            "engine": debate_engine,
            "manager": model_manager,
            "context": None,
            "status": "created",
            "task": None,
        }

        self.connections[debate_id] = []

        logger.info(f"Created debate {debate_id}: {setup.topic}")
        return debate_id

    async def start_debate(self, debate_id: str) -> None:
        """Start a debate session."""
        if debate_id not in self.active_debates:
            raise HTTPException(status_code=404, detail="Debate not found")

        debate_info = self.active_debates[debate_id]
        if debate_info["status"] == "running":
            raise HTTPException(status_code=400, detail="Debate already running")

        # Initialize debate
        context = await debate_info["engine"].initialize_debate()
        debate_info["context"] = context
        debate_info["status"] = "running"

        # Start debate task
        debate_info["task"] = asyncio.create_task(self._run_debate(debate_id))

        await self._broadcast_to_debate(
            debate_id,
            {"type": "debate_started", "debate_id": debate_id, "topic": context.topic},
        )

    async def cancel_debate(self, debate_id: str) -> None:
        """Cancel a running debate session."""
        if debate_id not in self.active_debates:
            raise HTTPException(status_code=404, detail="Debate not found")

        debate_info = self.active_debates[debate_id]

        # Cancel the running task if it exists
        if debate_info.get("task") and not debate_info["task"].done():
            debate_info["task"].cancel()
            try:
                await debate_info["task"]
            except asyncio.CancelledError:
                logger.info(f"Debate task {debate_id} cancelled successfully")

        # Update status
        debate_info["status"] = "cancelled"

        # Note: Ollama handles model memory management automatically

        # Broadcast cancellation
        await self._broadcast_to_debate(
            debate_id,
            {
                "type": "debate_cancelled",
                "debate_id": debate_id,
                "message": "Debate was cancelled by user",
            },
        )

    async def _run_debate(self, debate_id: str) -> None:
        """Run the debate using unified DebateEngine logic with real-time broadcasting."""
        debate_info = self.active_debates[debate_id]
        engine = debate_info["engine"]
        context = debate_info["context"]

        try:
            # Brief delay to ensure WebSocket connections are established
            await asyncio.sleep(1.0)
            # Hook into the engine to broadcast updates in real-time
            async def conduct_round_with_broadcast(format_phase):
                # Get format phases for progress calculation
                model_ids = list(context.participants.keys())
                format_phases = engine.format.get_phases(model_ids)
                total_phases = len(format_phases)
                # Use the current_round which is already set correctly in the engine
                current_phase = context.current_round

                # Calculate progress based on completed phases (current_phase - 1)
                # since current_phase represents the phase we're about to start
                completed_phases = max(0, current_phase - 1)

                # Log the speaking order for this phase
                logger.info(f"PHASE START: {format_phase.name} - Speaking order: {format_phase.speaking_order}")

                # Broadcast phase start with progress info
                await self._broadcast_to_debate(
                    debate_id,
                    {
                        "type": "phase_started",
                        "phase": format_phase.name,
                        "instruction": format_phase.instruction,
                        "current_phase": current_phase,
                        "total_phases": total_phases,
                        "progress_percentage": round(
                            (completed_phases / total_phases) * 100
                        ),
                    },
                )

                # Instead of calling original method, implement the round logic here
                # to broadcast messages as they're generated
                if not context:
                    raise RuntimeError("No active debate context")

                context.current_phase = format_phase.phase
                round_messages = []

                # Use format-defined speaking order
                for speaker_id in format_phase.speaking_order:
                    try:
                        logger.info(f"Attempting to get response from {speaker_id} for {format_phase.name}")
                        # Get individual message (this calls the engine's internal method)
                        message = await engine._get_format_speaker_response(speaker_id, format_phase)
                        round_messages.append(message)
                        context.messages.append(message)

                        # Broadcast message immediately after generation
                        await self._broadcast_to_debate(
                            debate_id,
                            {
                                "type": "new_message",
                                "message": {
                                    "speaker_id": message.speaker_id,
                                    "position": message.position.value,
                                    "phase": message.phase.value,
                                    "round_number": message.round_number,
                                    "content": message.content,
                                    "timestamp": message.timestamp.isoformat(),
                                    "word_count": len(message.content.split()),
                                    "metadata": message.metadata,
                                },
                            },
                        )

                        logger.info(f"SUCCESS: Round {context.current_round}, {format_phase.name}: {speaker_id}")

                    except Exception as e:
                        # Get model info for detailed error
                        model_config = context.participants.get(speaker_id)
                        model_name = model_config.name if model_config else "unknown"
                        provider = model_config.provider if model_config else "unknown"

                        error_msg = f"DEBATE FAILED: Model {speaker_id} ({model_name}) via {provider} failed in {format_phase.name}: {type(e).__name__}: {str(e)}"
                        logger.error(error_msg)

                        # Broadcast the error immediately
                        await self._broadcast_to_debate(
                            debate_id,
                            {
                                "type": "model_error",
                                "error": error_msg,
                                "speaker_id": speaker_id,
                                "model_name": model_name,
                                "provider": provider,
                                "phase": format_phase.name,
                                "exception_type": type(e).__name__,
                                "exception_message": str(e)
                            },
                        )

                        # FAIL FAST - re-raise the exception with enhanced context
                        raise RuntimeError(error_msg) from e

                return round_messages

            # Temporarily replace the method for broadcasting
            engine.conduct_format_round = conduct_round_with_broadcast

            # Use the unified debate running logic (includes transcript saving)
            context = await engine.run_full_debate()

            # Handle transcript saving with judge validation
            config = debate_info["config"]
            judge_models = config.judging.judge_models
            judge_provider = config.judging.judge_provider
            criteria = config.judging.criteria
            judges = create_judges(judge_models, judge_provider, config.system, debate_info["manager"], criteria)

            judge_result = None
            judges_configured = bool(judge_models)  # True if judge_models is not empty
            judging_succeeded = True

            # NOTE: Judge failures are intentionally NOT re-raised here to keep the server running.
            # Clients (CLI/web) receive judge_error WebSocket messages and handle termination themselves.
            # This allows the server to continue serving other requests while clients fail gracefully.
            if judges:
                try:
                    await self._broadcast_to_debate(
                        debate_id, {"type": "judging_started"}
                    )

                    judge_result = await engine.judge_debate_with_judges(judges)

                    # Judge result will be broadcast after saving to database
                    # No immediate broadcasting here since we need the database format
                except Exception as e:
                    judging_succeeded = False
                    logger.error(f"Judge evaluation failed for {debate_id}: {e}")
                    await self._broadcast_to_debate(
                        debate_id,
                        {
                            "type": "judge_error",
                            "error": f"Judge evaluation failed: {str(e)}",
                            "details": f"Error type: {type(e).__name__}"
                        },
                    )

            # Validate and save transcript
            try:
                # ONLY fail if judges were configured but judging failed
                if judges_configured and not judging_succeeded:
                    error_msg = f"Debate {debate_id} configured with judges but judging failed - transcript NOT saved to prevent invalid state"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                # Save transcript (with judge decision if judges succeeded, without if no judges configured)
                engine.save_transcript_with_judge_result(judge_result)

                # After saving, load the complete judging data from database and broadcast it
                if judges_configured and judging_succeeded and engine.transcript_manager:
                    transcript_id = context.metadata.get("transcript_id")
                    if transcript_id:
                        try:
                            # Load ensemble summary if it exists
                            ensemble_summary = engine.transcript_manager.db_manager.load_ensemble_summary(transcript_id)

                            if ensemble_summary:
                                # Ensemble case - load individual decisions and create unified format
                                individual_decisions = engine.transcript_manager.db_manager.load_judge_decisions(transcript_id)

                                # Aggregate all criterion scores from individual decisions
                                all_criterion_scores = []
                                for decision in individual_decisions:
                                    all_criterion_scores.extend(decision["criterion_scores"])

                                # Send unified judge_decision format with ensemble data
                                await self._broadcast_to_debate(
                                    debate_id,
                                    {
                                        "type": "judge_decision",
                                        "decision": {
                                            "winner_id": ensemble_summary["final_winner_id"],
                                            "winner_margin": ensemble_summary["final_margin"],
                                            "overall_feedback": ensemble_summary["summary_feedback"],
                                            "reasoning": ensemble_summary["summary_reasoning"],
                                            "criterion_scores": all_criterion_scores,
                                            "metadata": {
                                                "ensemble_size": ensemble_summary["num_judges"],
                                                "consensus_level": ensemble_summary["consensus_level"],
                                                "ensemble_method": ensemble_summary["ensemble_method"],
                                                "individual_decisions": individual_decisions,
                                            },
                                        },
                                    },
                                )
                            else:
                                # Single judge case - load single decision
                                judge_decision = engine.transcript_manager.db_manager.load_judge_decision(transcript_id)
                                if judge_decision:
                                    await self._broadcast_to_debate(
                                        debate_id,
                                        {
                                            "type": "judge_decision",
                                            "decision": judge_decision,
                                        },
                                    )

                        except Exception as e:
                            logger.error(f"Failed to load and broadcast judge results from database: {e}")

                debate_info["status"] = "completed"
                logger.info(f"Debate {debate_id} completed - transcript saved with judging status: judges_configured={judges_configured}, judging_succeeded={judging_succeeded}")

                await self._broadcast_to_debate(
                    debate_id, {"type": "debate_completed", "debate_id": debate_id}
                )

            except Exception as e:
                # If transcript saving fails, mark debate as error
                debate_info["status"] = "error"
                logger.error(f"Failed to save transcript for {debate_id}: {e}")
                await self._broadcast_to_debate(
                    debate_id, {"type": "error", "message": f"Failed to save transcript: {str(e)}"}
                )
                raise

        except asyncio.CancelledError:
            logger.info(f"Debate {debate_id} was cancelled")
            debate_info["status"] = "cancelled"
            # Note: Don't broadcast here - cancel_debate() already sent the message
            raise  # Re-raise to properly handle task cancellation
        except Exception as e:
            logger.error(f"Debate {debate_id} failed: {e}")
            debate_info["status"] = "error"
            await self._broadcast_to_debate(
                debate_id, {"type": "error", "message": str(e)}
            )

    async def _broadcast_to_debate(
        self, debate_id: str, message: Dict[str, Any]
    ) -> None:
        """Broadcast message to all connected clients for a debate."""
        if debate_id not in self.connections:
            return

        dead_connections = []
        for websocket in self.connections[debate_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.debug(f"WebSocket send failed: {e}")
                dead_connections.append(websocket)

        # Remove dead connections
        for conn in dead_connections:
            self.connections[debate_id].remove(conn)

    def add_connection(self, debate_id: str, websocket: WebSocket) -> None:
        """Add WebSocket connection for a debate."""
        if debate_id not in self.connections:
            self.connections[debate_id] = []
        self.connections[debate_id].append(websocket)

    def remove_connection(self, debate_id: str, websocket: WebSocket) -> None:
        """Remove WebSocket connection."""
        if debate_id in self.connections and websocket in self.connections[debate_id]:
            self.connections[debate_id].remove(websocket)
