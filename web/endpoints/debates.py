"""Debate management and WebSocket endpoints."""

import logging

from fastapi import HTTPException, WebSocket, WebSocketDisconnect, APIRouter

from config.settings import get_default_config, ModelConfig
from models.manager import ModelManager
from formats import format_registry
from web.debate_manager import DebateManager
from web.debate_reponse import DebateResponse
from web.debate_setup_request import DebateSetupRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")
ws_router = APIRouter()


def setup_debate_manager():
    """Get or create the global debate manager."""
    # Import here to avoid circular imports
    from web import api
    return api.debate_manager


@router.post("/debates", response_model=DebateResponse)
async def create_debate(setup: DebateSetupRequest):
    """Create a new debate."""
    try:
        debate_manager = setup_debate_manager()
        debate_id = await debate_manager.create_debate(setup)
        debate_info = debate_manager.active_debates[debate_id]

        # Get format-specific side labels
        side_labels = None
        try:
            debate_format = format_registry.get_format(setup.format)
            participants = list(setup.models.keys())
            side_labels = debate_format.get_side_labels(participants)
        except Exception as e:
            logger.warning(f"Failed to get side labels for format {setup.format}: {e}")

        return DebateResponse(
            id=debate_id,
            topic=setup.topic,
            format=setup.format,
            status=debate_info["status"],
            current_round=0,
            current_phase="setup",
            message_count=0,
            word_limit=setup.word_limit,
            models=setup.models,
            judge_models=setup.judge_models,
            side_labels=side_labels,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/debates/{debate_id}/start")
async def start_debate(debate_id: str):
    """Start a debate."""
    try:
        debate_manager = setup_debate_manager()
        await debate_manager.start_debate(debate_id)
        return {"status": "started", "debate_id": debate_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/debates/{debate_id}/cancel")
async def cancel_debate(debate_id: str):
    """Cancel a running debate."""
    try:
        debate_manager = setup_debate_manager()
        await debate_manager.cancel_debate(debate_id)
        return {"status": "cancelled", "debate_id": debate_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debates/{debate_id}", response_model=DebateResponse)
async def get_debate(debate_id: str):
    """Get debate status and info."""
    debate_manager = setup_debate_manager()
    if debate_id not in debate_manager.active_debates:
        raise HTTPException(status_code=404, detail="Debate not found")

    debate_info = debate_manager.active_debates[debate_id]
    context = debate_info.get("context")

    config = debate_info["config"]

    # Get format-specific side labels
    side_labels = None
    try:
        debate_format = format_registry.get_format(config.debate.format)
        participants = list(config.models.keys())
        side_labels = debate_format.get_side_labels(participants)
    except Exception as e:
        logger.warning(
            f"Failed to get side labels for format {config.debate.format}: {e}"
        )

    # Judge models are already in list format
    judge_models_list = (
        config.judging.judge_models if config.judging.judge_models else None
    )

    return DebateResponse(
        id=debate_id,
        topic=config.debate.topic,
        format=config.debate.format,
        status=debate_info["status"],
        current_round=context.current_round if context else 0,
        current_phase=context.current_phase.value if context else "setup",
        message_count=len(context.messages) if context else 0,
        word_limit=config.debate.word_limit,
        models=config.models,
        judge_models=judge_models_list,
        side_labels=side_labels,
    )


@router.get("/generate-topic")
async def generate_topic(format: str = "oxford"):
    """Generate a debate topic using AI with format-specific prompts."""
    try:
        config = get_default_config()

        # Get the format instance to access format-specific prompts
        debate_format = format_registry.get_format(format)
        messages = debate_format.get_topic_generation_messages()

        # Use configured topic generation model
        topic_model = config.system.debate_topic_model
        topic_provider = config.system.debate_topic_source

        if not topic_model:
            raise HTTPException(
                status_code=500,
                detail="No topic generation model configured. Please set debate_topic_model in system config.",
            )

        # Fast Ollama detection if using Ollama
        if topic_provider == "ollama":
            from models.providers import ProviderFactory, OllamaProvider

            try:
                provider = ProviderFactory.create_provider("ollama", config.system)
                if isinstance(provider, OllamaProvider):
                    if not await provider.is_running():
                        raise HTTPException(
                            status_code=500,
                            detail="Ollama is not running. Please start Ollama or change debate_topic_source to 'openrouter' in your config.",
                        )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail="Expected OllamaProvider but got different provider type",
                    )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to connect to Ollama: {str(e)}"
                )

        # Create model configuration for topic generation
        topic_gen_config = ModelConfig(
            name=topic_model,
            provider=topic_provider,
            personality="creative",
            max_tokens=100,
            temperature=0.8,
        )

        # Initialize model manager and register the topic generation model
        model_manager = ModelManager(config.system)
        model_manager.register_model("topic_generator", topic_gen_config)

        # Generate topic using format-specific prompts
        generated_topic = await model_manager.generate_response(
            "topic_generator", messages, max_tokens=100, temperature=0.8
        )

        # Clean up the response - remove quotes, extra formatting
        topic = generated_topic.strip().strip('"').strip("'")

        # Ensure topic ends with proper punctuation if it's a statement
        if topic and not topic[-1] in ".?!":
            if topic.lower().startswith(
                ("should", "is", "are", "can", "will", "would")
            ):
                topic += "?"
            else:
                topic += "."

        return {"topic": topic}

    except Exception as e:
        logger.error(f"Topic generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate topic: {str(e)}"
        )


@ws_router.websocket("/ws/debate/{debate_id}")
async def websocket_endpoint(websocket: WebSocket, debate_id: str):
    """WebSocket endpoint for real-time debate updates."""
    await websocket.accept()
    debate_manager = setup_debate_manager()
    debate_manager.add_connection(debate_id, websocket)

    try:
        # Send current state if debate exists
        if debate_id in debate_manager.active_debates:
            debate_info = debate_manager.active_debates[debate_id]
            await websocket.send_json(
                {
                    "type": "connected",
                    "debate_id": debate_id,
                    "status": debate_info["status"],
                }
            )

        # Keep connection alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        debate_manager.remove_connection(debate_id, websocket)