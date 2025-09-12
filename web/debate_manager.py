"""FastAPI web application for Dialectus AI Debate System."""

import asyncio
import uuid
from typing import Dict, List, Any
import logging
from contextlib import asynccontextmanager

from fastapi import HTTPException, WebSocket

from config.settings import get_default_config
from models.manager import ModelManager
from debate_engine.core import DebateEngine
from formats import format_registry
from judges.factory import create_judge
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
        # Automatically determine judging method based on judge_models
        if setup.judge_models is None or len(setup.judge_models) == 0:
            base_config.judging.method = "none"  # type: ignore[assignment]
        elif len(setup.judge_models) == 1:
            base_config.judging.method = "ai"  # type: ignore[assignment]
            base_config.judging.judge_model = setup.judge_models[0]
        else:
            base_config.judging.method = "ensemble"  # type: ignore[assignment]
            # For ensemble, store as comma-separated string (existing factory expects this)
            base_config.judging.judge_model = ",".join(setup.judge_models)
        
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
            original_conduct_round = engine.conduct_format_round

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
                    
                    logger.info(f"Round {context.current_round}, {format_phase.name}: {speaker_id}")

                return round_messages

            # Temporarily replace the method for broadcasting
            engine.conduct_format_round = conduct_round_with_broadcast

            # Use the unified debate running logic (includes transcript saving)
            context = await engine.run_full_debate()

            # Judge the debate if configured
            config = debate_info["config"]
            judge = create_judge(config.judging, config.system, debate_info["manager"])
            if judge:
                try:
                    await self._broadcast_to_debate(
                        debate_id, {"type": "judging_started"}
                    )

                    decision = await engine.judge_debate(judge)
                    if decision:
                        await self._broadcast_to_debate(
                            debate_id,
                            {
                                "type": "judge_decision",
                                "decision": {
                                    "winner_id": decision.winner_id,
                                    "winner_margin": decision.winner_margin,
                                    "overall_feedback": decision.overall_feedback,
                                    "reasoning": decision.reasoning,
                                    "criterion_scores": [
                                        {
                                            "criterion": score.criterion.value,
                                            "participant_id": score.participant_id,
                                            "score": score.score,
                                            "feedback": score.feedback,
                                        }
                                        for score in decision.criterion_scores
                                    ],
                                    "metadata": getattr(decision, "metadata", {}),
                                },
                            },
                        )
                except Exception as e:
                    logger.error(f"Judge evaluation failed for {debate_id}: {e}")
                    await self._broadcast_to_debate(
                        debate_id,
                        {
                            "type": "judge_error",
                            "error": f"Judge evaluation failed: {str(e)}",
                            "details": f"Error type: {type(e).__name__}"
                        },
                    )

            debate_info["status"] = "completed"
            logger.info(
                f"Debate {debate_id} completed - transcript saved via DebateEngine"
            )
            await self._broadcast_to_debate(
                debate_id, {"type": "debate_completed", "debate_id": debate_id}
            )

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
