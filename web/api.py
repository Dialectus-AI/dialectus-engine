"""FastAPI web application for Dialectus AI Debate System."""

import asyncio
from typing import Dict, Any, TYPE_CHECKING
from pathlib import Path
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from config.settings import get_default_config, ModelConfig
from models.manager import ModelManager
from debate_engine.transcript import TranscriptManager
from formats import format_registry
from judges.factory import create_judge
from web.debate_manager import DebateManager
from web.debate_reponse import DebateResponse
from web.debate_setup_request import DebateSetupRequest

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Pydantic models for API


# Cache cleanup scheduler
cleanup_task = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global cleanup_task

    # Startup: Start cache cleanup scheduler
    logger.info("Starting cache cleanup scheduler...")
    cleanup_task = asyncio.create_task(cache_cleanup_scheduler())

    yield

    # Shutdown: Cancel cache cleanup task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Cache cleanup scheduler stopped")


async def cache_cleanup_scheduler():
    """Background task to periodically clean up expired cache entries."""
    while True:
        try:
            # Run cleanup every hour
            await asyncio.sleep(3600)  # 1 hour

            from models.cache_manager import cache_manager

            cleaned_count = cache_manager.cleanup_expired()

            if cleaned_count > 0:
                logger.info(
                    f"Scheduled cache cleanup: removed {cleaned_count} expired entries"
                )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in cache cleanup scheduler: {e}")


# FastAPI app
app = FastAPI(
    title="Dialectus AI Debate System",
    description="Web interface for local AI model debates",
    version="1.0.0",
    lifespan=lifespan,
)

# Global debate manager
debate_manager = DebateManager()


@app.get("/api/models")
async def get_models():
    """Get available models from all providers with enhanced metadata."""
    try:
        config = get_default_config()
        model_manager = ModelManager(config.system)

        # Get enhanced models first (this includes all the data we need)
        enhanced_models = await model_manager.get_enhanced_models()

        # Build other formats from enhanced_models to avoid redundant API calls
        models_by_provider = {}
        flat_models = []
        formatted_models = []

        for model in enhanced_models:
            # Build models_by_provider
            provider = model["provider"]
            if provider not in models_by_provider:
                models_by_provider[provider] = []
            models_by_provider[provider].append(model["name"])

            # Build flat_models
            flat_models.append(model["name"])

            # Build formatted_models
            formatted_models.append(
                {
                    "id": model["id"],
                    "name": model["name"],
                    "provider": model["provider"],
                }
            )

        return {
            "models": flat_models,  # Backward compatibility
            "models_detailed": formatted_models,  # Basic format with provider info
            "models_enhanced": enhanced_models,  # Full enhanced metadata
            "models_by_provider": models_by_provider,  # Grouped by provider
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/providers")
async def get_providers():
    """Get available model providers and their status."""
    try:
        from models.providers import ProviderFactory

        providers = []

        for provider_name in ProviderFactory.get_available_providers():
            provider_info: Dict[str, Any] = {
                "name": provider_name,
                "status": "available",
            }

            # Add provider-specific status information
            if provider_name == "openrouter":
                config = get_default_config()
                import os

                api_key_configured = bool(
                    config.system.openrouter.api_key or os.getenv("OPENROUTER_API_KEY")
                )
                provider_info["api_key_configured"] = api_key_configured
                if not api_key_configured:
                    provider_info["status"] = "requires_api_key"
            elif provider_name == "ollama":
                # Use the provider's own health check method
                try:
                    config = get_default_config()
                    provider = ProviderFactory.create_provider(
                        provider_name, config.system
                    )
                    # Type-safe check for OllamaProvider
                    if provider.provider_name == "ollama":
                        from models.providers import OllamaProvider

                        if (
                            isinstance(provider, OllamaProvider)
                            and await provider.is_running()
                        ):
                            provider_info["status"] = "available"
                            provider_info["ollama_running"] = True
                        else:
                            provider_info["status"] = "offline"
                            provider_info["ollama_running"] = False
                except Exception as e:
                    logger.debug(f"Ollama provider check failed: {e}")
                    provider_info["status"] = "offline"
                    provider_info["ollama_running"] = False

            providers.append(provider_info)

        return {"providers": providers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/formats")
async def get_formats():
    """Get available debate formats."""
    return {"formats": format_registry.get_format_descriptions()}


@app.get("/api/ollama/health")
async def ollama_health():
    """Quick health check for Ollama service."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get("http://localhost:11434/api/version")
            if response.status_code == 200:
                version_data = response.json()
                return {
                    "status": "available",
                    "running": True,
                    "version": version_data.get("version", "unknown"),
                }
            else:
                return {
                    "status": "offline",
                    "running": False,
                    "error": f"HTTP {response.status_code}",
                }
    except Exception as e:
        return {"status": "offline", "running": False, "error": str(e)}


@app.get("/api/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    try:
        from models.cache_manager import cache_manager

        stats = cache_manager.get_cache_stats()
        return {"cache_stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cache/cleanup")
async def cleanup_cache():
    """Clean up expired cache entries."""
    try:
        from models.cache_manager import cache_manager

        cleaned_count = cache_manager.cleanup_expired()
        return {"message": f"Cleaned up {cleaned_count} expired cache entries"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/cache/models")
async def invalidate_models_cache():
    """Force invalidate the models cache."""
    try:
        from models.cache_manager import cache_manager

        invalidated = cache_manager.invalidate("openrouter", "models")
        if invalidated:
            return {"message": "Models cache invalidated successfully"}
        else:
            return {"message": "No models cache found to invalidate"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/debates", response_model=DebateResponse)
async def create_debate(setup: DebateSetupRequest):
    """Create a new debate."""
    try:
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
            judging_method=setup.judging_method,
            judge_model=setup.judge_model,
            side_labels=side_labels,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/debates/{debate_id}/start")
async def start_debate(debate_id: str):
    """Start a debate."""
    try:
        await debate_manager.start_debate(debate_id)
        return {"status": "started", "debate_id": debate_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/debates/{debate_id}/cancel")
async def cancel_debate(debate_id: str):
    """Cancel a running debate."""
    try:
        await debate_manager.cancel_debate(debate_id)
        return {"status": "cancelled", "debate_id": debate_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/debates/{debate_id}", response_model=DebateResponse)
async def get_debate(debate_id: str):
    """Get debate status and info."""
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
        judging_method=config.judging.method,
        judge_model=config.judging.judge_model,
        side_labels=side_labels,
    )


@app.get("/api/generate-topic")
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


@app.get("/api/transcripts")
async def get_transcripts(page: int = 1, limit: int = 20):
    """Get paginated list of saved transcripts from SQLite database."""
    try:
        # Validate pagination parameters
        if page < 1:
            page = 1
        if limit < 1 or limit > 100:
            limit = 20

        # Calculate offset for pagination
        offset = (page - 1) * limit

        # Get transcripts from database with explicit path
        config = get_default_config()
        transcript_dir = Path(config.system.transcript_dir)
        transcript_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        db_path = transcript_dir / "debates.db"
        transcript_manager = TranscriptManager(str(db_path))
        transcripts = transcript_manager.db_manager.list_debates_with_metadata(
            limit=limit, offset=offset
        )
        total_count = transcript_manager.get_debate_count()

        # Format response with pagination metadata
        return {
            "transcripts": transcripts,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total_count,
                "total_pages": (total_count + limit - 1) // limit,
                "has_next": offset + limit < total_count,
                "has_prev": page > 1,
            },
        }
    except Exception as e:
        logger.error(f"Failed to get transcripts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/transcripts/{transcript_id}")
async def get_transcript(transcript_id: int):
    """Get a specific transcript by ID with full message content."""
    try:
        config = get_default_config()
        transcript_dir = Path(config.system.transcript_dir)
        transcript_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        db_path = transcript_dir / "debates.db"
        transcript_manager = TranscriptManager(str(db_path))
        transcript_data = transcript_manager.load_transcript(transcript_id)

        if not transcript_data:
            raise HTTPException(status_code=404, detail="Transcript not found")

        # Enhance transcript data with human-readable phase names
        try:
            format_name = transcript_data["metadata"]["format"]
            debate_format = format_registry.get_format(format_name)

            # Create a mapping from phase enum values to human-readable names
            # We need to reconstruct this from the format's phases
            participants = list(transcript_data["metadata"]["participants"].keys())
            format_phases = debate_format.get_phases(participants)

            # Create phase name mapping: {enum_value: human_readable_name}
            phase_name_mapping = {}
            for format_phase in format_phases:
                phase_name_mapping[format_phase.phase.value] = format_phase.name

            # Add phase name mapping to context metadata for frontend use
            if "context_metadata" not in transcript_data:
                transcript_data["context_metadata"] = {}
            transcript_data["context_metadata"]["phase_names"] = phase_name_mapping

        except Exception as e:
            logger.warning(f"Failed to enhance transcript with phase names: {e}")
            # Continue without phase names if format lookup fails

        return transcript_data
    except Exception as e:
        logger.error(f"Failed to get transcript {transcript_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/debate/{debate_id}")
async def websocket_endpoint(websocket: WebSocket, debate_id: str):
    """WebSocket endpoint for real-time debate updates."""
    await websocket.accept()
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


# Mount static files for frontend development
try:
    from pathlib import Path

    frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dist)), name="static")
        print(f"📂 Serving frontend from: {frontend_dist}")
except Exception as e:
    print(f"⚠️ Frontend files not found: {e}")


@app.get("/")
async def read_root():
    """Serve the main web interface."""
    # Check if we have built frontend files
    frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
    frontend_src = Path(__file__).parent.parent / "frontend" / "src"

    if frontend_dist.exists() and (frontend_dist / "index.html").exists():
        # Serve built frontend
        with open(frontend_dist / "index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    elif frontend_src.exists():
        # Development mode - show build instructions
        return HTMLResponse(
            f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Dialectus AI Debate System - Dev Mode</title>
            <style>
                {get_dev_styles()}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎭 Dialectus AI Debate System</h1>
                    <p class="subtitle">Development Mode</p>
                </div>
                
                <div class="card status-card">
                    <h2>🚀 Backend Status</h2>
                    <div class="status-grid">
                        <div class="status-item">
                            <span class="status-label">FastAPI Server:</span>
                            <span class="status-value success">✅ Running</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">API Endpoints:</span>
                            <span class="status-value success">✅ Active</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">WebSocket:</span>
                            <span class="status-value success">✅ Ready</span>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h2>🛠️ Frontend Development</h2>
                    <p>To start the frontend development server:</p>
                    <div class="code-block">
                        <code>cd frontend<br>npm install<br>npm run dev</code>
                    </div>
                    <p>The frontend will be available at <strong>http://localhost:5173</strong> with hot reload!</p>
                </div>

                <div class="card">
                    <h2>📡 API Endpoints</h2>
                    <div class="endpoint-list">
                        <a href="/api/models" class="endpoint">GET /api/models</a>
                        <a href="/api/formats" class="endpoint">GET /api/formats</a>
                        <a href="/api/transcripts" class="endpoint">GET /api/transcripts</a>
                        <a href="/docs" class="endpoint docs-link">📚 API Documentation</a>
                    </div>
                </div>

                <div class="card">
                    <h2>🎯 Quick Test</h2>
                    <p>Test the API connection:</p>
                    <button onclick="testApi()" class="test-button">Test API Connection</button>
                    <div id="test-result"></div>
                </div>
            </div>

            <script>
                async function testApi() {{
                    const resultDiv = document.getElementById('test-result');
                    resultDiv.innerHTML = '<p class="testing">🔄 Testing...</p>';
                    
                    try {{
                        const response = await fetch('/api/models');
                        const data = await response.json();
                        resultDiv.innerHTML = `
                            <p class="success">✅ API Working!</p>
                            <p>Found ${{data.models?.length || 0}} models</p>
                        `;
                    }} catch (error) {{
                        resultDiv.innerHTML = `
                            <p class="error">❌ API Test Failed</p>
                            <p>${{error.message}}</p>
                        `;
                    }}
                }}
            </script>
        </body>
        </html>
        """
        )
    else:
        # Fallback basic HTML
        return HTMLResponse(
            """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dialectus AI Debate System</title>
        <style>
            * { box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 2rem;
                background: #f8fafc;
                color: #1e293b;
            }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 3rem; }
            .header h1 { font-size: 3rem; margin: 0; color: #0f172a; }
            .header p { font-size: 1.2rem; color: #64748b; margin: 0.5rem 0; }
            .card {
                background: white;
                border-radius: 12px;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .card h2 { margin-top: 0; color: #374151; }
            .status-badge {
                display: inline-block;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                font-weight: 500;
                font-size: 0.9rem;
            }
            .status-running { background: #dcfce7; color: #166534; }
            .status-building { background: #fef3c7; color: #92400e; }
            ul { list-style: none; padding: 0; }
            li { margin: 0.5rem 0; }
            a { color: #3b82f6; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎭 Dialectus AI Debate System</h1>
                <p>Modern web interface for orchestrating debates between AI models</p>
            </div>
            
            <div class="card">
                <h2>🚀 Status</h2>
                <p>
                    Backend API: <span class="status-badge status-running">✅ Running</span><br>
                    TypeScript Frontend: <span class="status-badge status-building">🔧 Building next...</span>
                </p>
            </div>
            
            <div class="card">
                <h2>📡 API Endpoints</h2>
                <ul>
                    <li>📊 <a href="/api/models">GET /api/models</a> - List available Ollama models</li>
                    <li>🏛️ <a href="/api/formats">GET /api/formats</a> - Available debate formats</li>
                    <li>📜 <a href="/api/transcripts">GET /api/transcripts</a> - Saved debate transcripts</li>
                    <li>📚 <a href="/docs">Interactive API Documentation</a></li>
                </ul>
            </div>
            
            <div class="card">
                <h2>🛠️ Next Steps</h2>
                <p>The FastAPI backend is ready! Now building the TypeScript frontend with:</p>
                <ul>
                    <li>⚡ Vite for lightning-fast development</li>
                    <li>📦 Modern ES2024 TypeScript</li>
                    <li>🧩 Native Web Components</li>
                    <li>🔄 Real-time WebSocket updates</li>
                    <li>🎨 Modern CSS Grid layouts</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
        )


def get_dev_styles() -> str:
    """Get CSS styles for development mode."""
    return """
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #1f2937;
        }
        .container { max-width: 800px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 2rem; color: white; }
        .header h1 { font-size: 3rem; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }
        .subtitle { font-size: 1.2rem; opacity: 0.9; margin: 0.5rem 0; }
        .card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .card h2 { color: #374151; margin-top: 0; }
        .status-grid { display: grid; gap: 1rem; }
        .status-item { display: flex; justify-content: space-between; align-items: center; }
        .status-label { font-weight: 500; }
        .status-value { padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.875rem; }
        .success { background: #d1fae5; color: #065f46; }
        .error { background: #fee2e2; color: #991b1b; }
        .testing { color: #0369a1; }
        .code-block {
            background: #f3f4f6;
            padding: 1rem;
            border-radius: 6px;
            margin: 1rem 0;
        }
        .code-block code { font-family: 'SF Mono', Monaco, monospace; }
        .endpoint-list { display: flex; flex-wrap: wrap; gap: 0.5rem; }
        .endpoint {
            padding: 0.5rem 1rem;
            background: #3b82f6;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-size: 0.875rem;
            transition: background 0.2s;
        }
        .endpoint:hover { background: #2563eb; }
        .docs-link { background: #10b981; }
        .docs-link:hover { background: #059669; }
        .test-button {
            padding: 0.75rem 1.5rem;
            background: #8b5cf6;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: background 0.2s;
        }
        .test-button:hover { background: #7c3aed; }
        #test-result { margin-top: 1rem; }
        @media (max-width: 640px) {
            body { padding: 1rem; }
            .header h1 { font-size: 2rem; }
            .endpoint-list { flex-direction: column; }
        }
    """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
