"""FastAPI web application for AI Debate System."""

import asyncio
import uuid
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from config.settings import get_default_config, ModelConfig
from models.manager import ModelManager
from debate_engine.core import DebateEngine
from debate_engine.transcript import TranscriptManager
from formats import format_registry
from judges.factory import create_judge

logger = logging.getLogger(__name__)

# Pydantic models for API
class DebateSetupRequest(BaseModel):
    """Request model for creating a new debate."""
    topic: str
    format: str = "oxford"
    word_limit: int = 200
    models: Dict[str, ModelConfig]
    judging_method: str = "ai"
    judge_model: Optional[str] = None

class DebateResponse(BaseModel):
    """Response model for debate information."""
    id: str
    topic: str
    format: str
    status: str
    current_round: int
    current_phase: str
    message_count: int
    # Full configuration details for frontend
    word_limit: Optional[int] = None
    models: Optional[Dict[str, ModelConfig]] = None
    judging_method: Optional[str] = None
    judge_model: Optional[str] = None
    side_labels: Optional[Dict[str, str]] = None  # Format-specific participant labels

class MessageResponse(BaseModel):
    """Response model for debate messages."""
    speaker_id: str
    position: str
    phase: str
    round_number: int
    content: str
    timestamp: str
    word_count: int

# FastAPI app
app = FastAPI(
    title="AI Debate System",
    description="Web interface for local AI model debates",
    version="1.0.0"
)

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
        # Validate and assign format with proper type casting
        if setup.format not in ["parliamentary", "oxford", "socratic", "custom"]:
            raise HTTPException(status_code=400, detail=f"Invalid debate format: {setup.format}")
        base_config.debate.format = setup.format  # type: ignore[assignment]
        base_config.debate.word_limit = setup.word_limit
        base_config.models = setup.models
        # Validate and assign judging method with proper type casting
        if setup.judging_method not in ["ai", "ensemble", "none"]:
            raise HTTPException(status_code=400, detail=f"Invalid judging method: {setup.judging_method}")
        base_config.judging.method = setup.judging_method  # type: ignore[assignment]
        if setup.judge_model:
            base_config.judging.judge_model = setup.judge_model
        
        # Initialize components
        model_manager = ModelManager(base_config.system)
        debate_engine = DebateEngine(base_config, model_manager)
        
        # Store debate info
        self.active_debates[debate_id] = {
            'id': debate_id,
            'config': base_config,
            'engine': debate_engine,
            'manager': model_manager,
            'context': None,
            'status': 'created',
            'task': None
        }
        
        self.connections[debate_id] = []
        
        logger.info(f"Created debate {debate_id}: {setup.topic}")
        return debate_id
    
    async def start_debate(self, debate_id: str) -> None:
        """Start a debate session."""
        if debate_id not in self.active_debates:
            raise HTTPException(status_code=404, detail="Debate not found")
        
        debate_info = self.active_debates[debate_id]
        if debate_info['status'] == 'running':
            raise HTTPException(status_code=400, detail="Debate already running")
        
        # Initialize debate
        context = await debate_info['engine'].initialize_debate()
        debate_info['context'] = context
        debate_info['status'] = 'running'
        
        # Start debate task
        debate_info['task'] = asyncio.create_task(
            self._run_debate(debate_id)
        )
        
        await self._broadcast_to_debate(debate_id, {
            'type': 'debate_started',
            'debate_id': debate_id,
            'topic': context.topic
        })
    
    async def cancel_debate(self, debate_id: str) -> None:
        """Cancel a running debate session."""
        if debate_id not in self.active_debates:
            raise HTTPException(status_code=404, detail="Debate not found")
        
        debate_info = self.active_debates[debate_id]
        
        # Cancel the running task if it exists
        if debate_info.get('task') and not debate_info['task'].done():
            debate_info['task'].cancel()
            try:
                await debate_info['task']
            except asyncio.CancelledError:
                logger.info(f"Debate task {debate_id} cancelled successfully")
        
        # Update status
        debate_info['status'] = 'cancelled'
        
        # Note: Ollama handles model memory management automatically
        
        # Broadcast cancellation
        await self._broadcast_to_debate(debate_id, {
            'type': 'debate_cancelled',
            'debate_id': debate_id,
            'message': 'Debate was cancelled by user'
        })
    
    async def _run_debate(self, debate_id: str) -> None:
        """Run the debate using unified DebateEngine logic with real-time broadcasting."""
        debate_info = self.active_debates[debate_id]
        engine = debate_info['engine']
        context = debate_info['context']
        
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
                await self._broadcast_to_debate(debate_id, {
                    'type': 'phase_started',
                    'phase': format_phase.name,
                    'instruction': format_phase.instruction,
                    'current_phase': current_phase,
                    'total_phases': total_phases,
                    'progress_percentage': round((completed_phases / total_phases) * 100)
                })
                
                # Run the round
                round_messages = await original_conduct_round(format_phase)
                
                # Broadcast each message
                for message in round_messages:
                    await self._broadcast_to_debate(debate_id, {
                        'type': 'new_message',
                        'message': {
                            'speaker_id': message.speaker_id,
                            'position': message.position.value,
                            'phase': message.phase.value,
                            'round_number': message.round_number,
                            'content': message.content,
                            'timestamp': message.timestamp.isoformat(),
                            'word_count': len(message.content.split()),
                            'metadata': message.metadata
                        }
                    })
                
                return round_messages
            
            # Temporarily replace the method for broadcasting
            engine.conduct_format_round = conduct_round_with_broadcast
            
            # Use the unified debate running logic (includes transcript saving)
            context = await engine.run_full_debate()
            
            # Judge the debate if configured
            config = debate_info['config']
            judge = create_judge(config.judging, config.system, debate_info['manager'])
            if judge:
                try:
                    await self._broadcast_to_debate(debate_id, {
                        'type': 'judging_started'
                    })
                    
                    decision = await engine.judge_debate(judge)
                    if decision:
                        await self._broadcast_to_debate(debate_id, {
                            'type': 'judge_decision',
                            'decision': {
                                'winner_id': decision.winner_id,
                                'winner_margin': decision.winner_margin,
                                'overall_feedback': decision.overall_feedback,
                                'reasoning': decision.reasoning,
                                'criterion_scores': [
                                    {
                                        'criterion': score.criterion.value,
                                        'participant_id': score.participant_id,
                                        'score': score.score,
                                        'feedback': score.feedback
                                    } for score in decision.criterion_scores
                                ],
                                'metadata': getattr(decision, 'metadata', {})
                            }
                        })
                except Exception as e:
                    logger.error(f"Judge evaluation failed for {debate_id}: {e}")
            
            debate_info['status'] = 'completed'
            logger.info(f"Debate {debate_id} completed - transcript saved via DebateEngine")
            await self._broadcast_to_debate(debate_id, {
                'type': 'debate_completed',
                'debate_id': debate_id
            })
            
        except asyncio.CancelledError:
            logger.info(f"Debate {debate_id} was cancelled")
            debate_info['status'] = 'cancelled'
            # Note: Don't broadcast here - cancel_debate() already sent the message
            raise  # Re-raise to properly handle task cancellation
        except Exception as e:
            logger.error(f"Debate {debate_id} failed: {e}")
            debate_info['status'] = 'error'
            await self._broadcast_to_debate(debate_id, {
                'type': 'error',
                'message': str(e)
            })
    
    async def _broadcast_to_debate(self, debate_id: str, message: Dict[str, Any]) -> None:
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

# Global debate manager
debate_manager = DebateManager()

@app.get("/api/models")
async def get_models():
    """Get available Ollama models."""
    try:
        config = get_default_config()
        model_manager = ModelManager(config.system)
        models = await model_manager.get_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/formats")
async def get_formats():
    """Get available debate formats."""
    return {
        "formats": format_registry.get_format_descriptions()
    }

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
            status=debate_info['status'],
            current_round=0,
            current_phase="setup",
            message_count=0,
            word_limit=setup.word_limit,
            models=setup.models,
            judging_method=setup.judging_method,
            judge_model=setup.judge_model,
            side_labels=side_labels
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
    context = debate_info.get('context')
    
    config = debate_info['config']
    
    # Get format-specific side labels
    side_labels = None
    try:
        debate_format = format_registry.get_format(config.debate.format)
        participants = list(config.models.keys())
        side_labels = debate_format.get_side_labels(participants)
    except Exception as e:
        logger.warning(f"Failed to get side labels for format {config.debate.format}: {e}")
    
    return DebateResponse(
        id=debate_id,
        topic=config.debate.topic,
        format=config.debate.format,
        status=debate_info['status'],
        current_round=context.current_round if context else 0,
        current_phase=context.current_phase.value if context else "setup",
        message_count=len(context.messages) if context else 0,
        word_limit=config.debate.word_limit,
        models=config.models,
        judging_method=config.judging.method,
        judge_model=config.judging.judge_model,
        side_labels=side_labels
    )

@app.get("/api/generate-topic")
async def generate_topic():
    """Generate a debate topic using AI."""
    try:
        config = get_default_config()
        model_manager = ModelManager(config.system)
        
        # Use judge model for topic generation
        judge_model = config.judging.judge_model
        if not judge_model:
            # Fallback to first available model if no judge model configured
            available_models = await model_manager.get_available_models()
            if not available_models:
                raise HTTPException(status_code=500, detail="No models available for topic generation")
            judge_model = available_models[0]
        
        # Create a temporary model configuration for topic generation
        topic_gen_config = ModelConfig(
            name=judge_model,
            personality="creative",
            max_tokens=100,
            temperature=0.8
        )
        
        # Register the model temporarily
        model_manager.register_model("topic_generator", topic_gen_config)
        
        # Generate topic with optimized prompt
        messages = [
            {
                "role": "system",
                "content": "You are an expert debate topic generator. Create engaging, balanced, and thought-provoking debate topics that have clear pro and con sides. Topics should be contemporary, relevant, and suitable for intellectual discourse. Generate topics across diverse domains like technology, society, ethics, environment, education, healthcare, economics, politics, and culture."
            },
            {
                "role": "user", 
                "content": "Generate a single debate topic that would make for an interesting and balanced debate. The topic should be phrased as a clear statement that can be argued for or against. Make it thought-provoking and current. Respond with just the topic statement, no additional text."
            }
        ]
        
        try:
            generated_topic = await model_manager.generate_response(
                "topic_generator", 
                messages,
                max_tokens=100,
                temperature=0.8
            )
            
            # Clean up the response - remove quotes, extra formatting
            topic = generated_topic.strip().strip('"').strip("'")
            
            # Ensure topic ends with proper punctuation if it's a statement
            if topic and not topic[-1] in '.?!':
                if topic.lower().startswith(('should', 'is', 'are', 'can', 'will', 'would')):
                    topic += '?'
                else:
                    topic += '.'
            
            return {"topic": topic}
            
        finally:
            # Note: Ollama handles model memory management automatically
            pass
        
    except Exception as e:
        logger.error(f"Topic generation failed: {e}")
        # Return fallback topics if generation fails
        fallback_topics = [
            "Should artificial intelligence be regulated by government oversight?",
            "Is remote work more beneficial than office-based work?",
            "Should social media platforms be liable for user-generated content?",
            "Is universal basic income a necessary policy for the future?",
            "Should genetic engineering be used to enhance human capabilities?"
        ]
        import random
        return {"topic": random.choice(fallback_topics)}

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
        transcripts = transcript_manager.db_manager.list_debates_with_metadata(limit=limit, offset=offset)
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
                "has_prev": page > 1
            }
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
            await websocket.send_json({
                'type': 'connected',
                'debate_id': debate_id,
                'status': debate_info['status']
            })
        
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
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI Debate System - Dev Mode</title>
            <style>
                {get_dev_styles()}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎭 AI Debate System</h1>
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
        """)
    else:
        # Fallback basic HTML
        return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Debate System</title>
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
                <h1>🎭 AI Debate System</h1>
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
    """)

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