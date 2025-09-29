# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Style

- You are a technical co-founder and collaborative partner, not a servant. 🤗
- Your role is to critically evaluate my claims and suggestions, and to offer alternative perspectives or challenge assumptions when appropriate.
- Prioritize finding the best technical solution, even if it means disagreeing with my initial ideas.
- Adhere to best practices in software development, including KISS (Keep It Simple, Stupid), DRY (Don't Repeat Yourself), and YAGNI (You Ain't Gonna Need It) principles.
- Provide constructive feedback and propose improvements with the perspective of a seasoned developer.
- Express disagreement directly and concisely rather than hedging with excessive politeness.

## Project Overview

**Dialectus Engine** is the Python backend for the Dialectus AI debate system. This is a FastAPI-based service providing:

- **REST API** endpoints for debate management, model discovery, and authentication
- **WebSocket** streaming for real-time debate execution
- **Multi-provider AI orchestration** (Ollama local models, OpenRouter cloud models)
- **Debate engine** implementing multiple formal debate formats (Oxford, Parliamentary, Socratic, Public Forum)
- **AI judging system** with ensemble support for objective debate evaluation
- **SQLite database** for transcript storage, user management, and cost tracking

This is one of three repositories in the Dialectus ecosystem:
- **dialectus-engine** (this repo) - Python/FastAPI backend
- **dialectus-web** - TypeScript/Vite frontend SPA
- **dialectus-cli** - Node.js cross-platform CLI tool

## Code Quality Standards

**This project has a HIGH bar.** The developer has 30 years of professional software engineering experience and expects:

### Python Standards
- **Python 3.13** with modern type annotations
- **Pylance strict mode** enabled - zero type errors tolerated
- **No legacy typing** - Use `list[str]`, `dict[str, int]`, `X | None` instead of `List[str]`, `Dict[str, int]`, `Optional[X]`
- **Avoid `Any`** - Use proper type annotations with generics, protocols, and unions
- **Pydantic v2** - Use `model_dump()` not deprecated `dict()` for serialization
- **Modern async/await** - Proper async context managers and generators
- **No unused imports or variables** - Keep the codebase clean
- **Type-safe dict access** - Use `.get()` or explicit checks for optional keys
- **Collections from built-ins** - `from collections.abc import AsyncGenerator` not typing module

### Pre-Commit Requirements
**CRITICAL**: Before ANY commit or code completion:
1. Check type annotations with Pylance in strict mode
2. Run the server to ensure no runtime errors (`python main.py --web`)
3. Test API endpoints if modifying web routes
4. Verify database migrations if touching schema

### Project Conventions
- **Pydantic for config** - All configuration via Pydantic models with validation
- **Structured logging** - Use logger with appropriate levels (DEBUG, INFO, WARNING, ERROR)
- **Context managers** - Use `with` for database connections, file handles, async sessions
- **Protocol classes** - Use `Protocol` for duck typing over abstract base classes when appropriate
- **Type-safe enums** - Use `Enum` classes for constants with fixed values

## Development Commands

```bash
# Start development server (FastAPI with auto-reload)
python main.py --web

# Production server (Railway, Docker, etc.)
python main.py  # Auto-detects production environment

# Install/update dependencies
pip install -r requirements.txt

# Recompile dependencies after editing requirements.in
pip-compile requirements.in

# Database initialization (automatic on first run)
# Creates debates.db with all tables via SchemaManager
```

## Architecture

### Technology Stack
- **Python 3.13** with modern type hints (`X | None`, `list[T]`, `dict[K, V]`)
- **FastAPI** for REST API and WebSocket endpoints
- **Pydantic v2** for data validation and settings management
- **SQLite** via context managers for debate transcripts and auth
- **OpenAI SDK** for OpenRouter API integration
- **httpx** for async HTTP requests (Ollama provider)
- **Uvicorn** ASGI server with WebSocket support

### Core Structure
```
dialectus-engine/
├── main.py                          # Entry point with CLI/web server logic
├── web/
│   ├── api.py                       # FastAPI app initialization and lifespan
│   ├── debate_manager.py            # Manages active debate WebSocket sessions
│   └── endpoints/                   # API route handlers
│       ├── models.py                # GET /v1/models (cached, multi-provider)
│       ├── system.py                # GET /v1/system/health, /formats, /providers
│       ├── transcripts.py           # GET /v1/transcripts
│       └── v1/                      # Versioned API endpoints
│           ├── auth.py              # POST /v1/auth/{login,register,logout}
│           ├── debates.py           # POST /v1/debates, GET /v1/debates/{id}
│           └── tournaments.py       # Tournament CRUD operations
├── debate_engine/
│   ├── core.py                      # DebateEngine - orchestrates debate flow
│   ├── models.py                    # DebateContext, DebateMessage (Pydantic)
│   ├── types.py                     # Enums: DebatePhase, Position, Criterion
│   ├── transcript.py                # TranscriptManager - formats and saves
│   └── database/
│       ├── database.py              # DatabaseManager with context managers
│       └── schema/
│           ├── schema_manager.py    # Loads SQL files, creates tables
│           └── tables/              # SQL DDL files (debates.sql, auth.sql, etc.)
├── models/
│   ├── manager.py                   # ModelManager - multi-provider orchestration
│   ├── base_types.py                # BaseEnhancedModelInfo (protocol/base class)
│   ├── cache_manager.py             # In-memory cache with TTL for model metadata
│   ├── openrouter_types.py          # OpenRouter model filtering and enhancement
│   └── providers/
│       ├── base_model_provider.py   # BaseModelProvider abstract class
│       ├── ollama_provider.py       # OllamaProvider (local models)
│       ├── openrouter_provider.py   # OpenRouterProvider (cloud models)
│       └── providers.py             # ProviderFactory - creates provider instances
├── judges/
│   ├── base.py                      # JudgeDecision, CriterionScore (Pydantic)
│   ├── ai_judge.py                  # AIJudge - LLM-based debate evaluation
│   ├── ensemble_utils.py            # Aggregates multiple judge decisions
│   └── factory.py                   # JudgeFactory - creates judge instances
├── formats/
│   ├── base.py                      # DebateFormat base class
│   ├── registry.py                  # FormatRegistry singleton
│   ├── oxford.py                    # OxfordFormat - classic debate structure
│   ├── parliamentary.py             # ParliamentaryFormat - British style
│   ├── public_forum.py              # PublicForumFormat - American high school
│   └── socratic.py                  # SocraticFormat - question-driven
├── config/
│   ├── settings.py                  # Pydantic config models (AppConfig, ModelConfig)
│   └── openrouter_filters.json      # Configurable model exclusion patterns
└── tournaments/
    └── manager.py                   # TournamentManager for multi-debate events
```

### Key Patterns

**Multi-Provider Architecture**: The `ModelManager` abstracts away provider differences. Register models with `register_model(id, ModelConfig)`, then call `generate_response(id, messages)`. The manager routes to the correct provider (Ollama or OpenRouter) transparently.

**Async Context Managers**: Providers use `@asynccontextmanager` for session management. The `model_session()` context ensures proper resource cleanup even on errors.

**Pydantic Everywhere**: All config, API requests/responses, and data models use Pydantic v2 with strict validation. This provides automatic FastAPI documentation and type-safe serialization.

**Database Schema Management**: SQL DDL files in `debate_engine/database/schema/tables/` are loaded by `SchemaManager` on startup. Database tables are created idempotently with proper indexes.

**WebSocket Streaming**: Real-time debate updates via `WebSocketManager` in `debate_manager.py`. Each debate gets a unique WebSocket connection with chunk-by-chunk streaming.

**Cost Tracking**: OpenRouter generations include `generation_id` for async cost queries. Background tasks update the database with actual costs after API latency.

## Multi-Repository Context

When working across repositories, you can add them to the Claude Code session:
```bash
claude-code --add-dir ../dialectus-web --add-dir ../dialectus-cli dialectus-engine
```

### Frontend Integration (dialectus-web)
The TypeScript SPA consumes this backend via:
- **REST API** at `/v1/*` for models, debates, auth, transcripts
- **WebSocket** at `/v1/ws/debate/{id}` for real-time streaming
- **CORS** configured for localhost (dev) and dialectus.ai (prod)

### CLI Tool (dialectus-cli)
Node.js CLI for headless debate execution:
- Uses same REST API endpoints
- No WebSocket - polls for debate status
- Supports batch/scripted debates

## API Endpoints

### Stable Endpoints (shared across versions)
- `GET /v1/models` - Available AI models across all providers with metadata
- `GET /v1/formats` - Debate format definitions
- `GET /v1/providers` - Provider status (Ollama availability, OpenRouter key status)
- `GET /v1/transcripts` - List saved debate transcripts
- `GET /v1/transcripts/{id}` - Full debate transcript with messages and scores
- `GET /v1/system/health` - Health check

### Version 1 Endpoints
- `POST /v1/auth/register` - Create new user account
- `POST /v1/auth/login` - Authenticate and create session
- `POST /v1/auth/logout` - End session
- `GET /v1/auth/verify` - Check session validity
- `POST /v1/debates` - Create and start new debate
- `GET /v1/debates/{id}` - Get debate metadata and status
- `DELETE /v1/debates/{id}` - Delete debate transcript
- `WebSocket /v1/ws/debate/{id}` - Real-time debate streaming

## Authentication & Authorization

**HTTP-only cookies** for session management. FastAPI middleware validates sessions on protected routes.

Auth tables (`users`, `sessions`) managed via `debate_engine/database/schema/tables/auth.sql`.

Password hashing uses `bcrypt` with proper salt rounds.

**Important**: All authentication security is enforced server-side. Frontend auth is UX only.

## Database Architecture

**SQLite database** (`debates.db`) with normalized schema:

### Core Tables
- `debates` - Metadata (topic, format, participants, timing, user_id)
- `messages` - Individual debate turns (speaker, content, cost, generation_id)
- `judge_decisions` - Individual judge evaluations with criterion scores
- `criterion_scores` - Per-participant scores for each judging criterion
- `ensemble_summary` - Aggregated results from multiple judges
- `users` - User accounts with hashed passwords
- `sessions` - Active session tokens with expiration

### Schema Management
SQL files in `debate_engine/database/schema/tables/*.sql` define the schema.

`SchemaManager` loads and executes these on startup to create tables idempotently.

Indexes defined in SQL files for query performance (by user_id, debate_id, created_at).

### Cost Tracking
OpenRouter API provides `generation_id` after streaming completes. Background tasks query the `/generation` endpoint to fetch actual costs and update the database.

Messages and judge decisions have `cost`, `generation_id`, and `cost_queried_at` fields.

## Real-Time Features

### WebSocket Protocol
1. Client connects to `/v1/ws/debate/{id}` after creating debate via REST
2. Server streams JSON messages with `type` field:
   - `new_message` - Debate turn chunk or complete message
   - `judge_decision` - Judge evaluation complete
   - `debate_completed` - All phases finished
   - `error` - Something went wrong

### Debate Flow
1. `POST /v1/debates` with topic, format, participants → returns debate_id
2. Connect WebSocket to `/v1/ws/debate/{id}`
3. Server executes debate via `DebateEngine.run_full_debate()`
4. Each message streams chunks in real-time
5. Judge evaluation runs after debate completes
6. Transcript saved to database with all costs

### Provider Streaming
- **OpenRouter**: Native SSE streaming via OpenAI SDK
- **Ollama**: Chunk-based streaming via `/api/generate` endpoint
- Both normalized to same `chunk_callback(content: str, is_final: bool)` interface

## Configuration System

### Configuration Files
- `debate_config.json` - Main config (auto-created from `debate_config.example.json`)
- `config/openrouter_filters.json` - Model filtering patterns (excludes meta-models, routers)

### Environment Variables
- `OPENROUTER_API_KEY` - OpenRouter API key (required for cloud models)
- `OLLAMA_URL` - Ollama server URL (default: `http://localhost:11434`)
- `PORT` - Web server port (default: 8000)
- `ALLOWED_ORIGINS` - CORS origins for production (comma-separated)

### Pydantic Config Models
- `AppConfig` - Top-level config (debate, judging, system)
- `ModelConfig` - Per-model config (name, provider, temperature, max_tokens)
- `SystemConfig` - System settings (Ollama URL, OpenRouter config)
- `DebateConfig` - Debate settings (topic, format, word_limit)
- `JudgingConfig` - Judge settings (criteria, judge_models)

## Testing & Development

### Local Development
1. Ensure Ollama is running: `ollama serve`
2. Pull models: `ollama pull llama3.2:latest`
3. Start server: `python main.py --web`
4. API docs: `http://localhost:8000/docs`

### Testing with Frontend
1. Start backend: `cd dialectus-engine && python main.py --web`
2. Start frontend: `cd dialectus-web && npm run dev`
3. Frontend proxies `/v1/*` and `/ws/*` to backend via Vite

### Common Issues
- **"Model not found"** - Run `ollama pull <model>` or check OpenRouter API key
- **CORS errors** - Check `ALLOWED_ORIGINS` or Vite proxy config
- **WebSocket disconnect** - Check logs for provider errors or token limits
- **Empty responses** - Small models may struggle - increase temperature or try larger model

## File Organization

### Naming Conventions
- **Modules**: `snake_case.py` (e.g., `debate_manager.py`, `ai_judge.py`)
- **Classes**: `PascalCase` (e.g., `DebateEngine`, `ModelManager`)
- **Functions**: `snake_case` (e.g., `generate_response`, `save_transcript`)
- **Constants**: `UPPER_SNAKE_CASE` for module-level constants
- **Type Aliases**: `PascalCase` (e.g., `MessageData`, `FullTranscriptData`)

### Import Conventions
```python
# Standard library
from collections.abc import AsyncGenerator
from typing import Protocol, Any

# Third-party
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel, Field

# Local absolute imports (preferred)
from debate_engine.core import DebateEngine
from models.manager import ModelManager
from config.settings import AppConfig

# Local relative imports (only within same package)
from .base import DebateFormat
from .types import DebatePhase
```

## Common Patterns

### Async Context Managers
```python
@asynccontextmanager
async def model_session(self, model_id: str):
    """Manage model lifecycle."""
    try:
        await self._acquire_model(model_id)
        yield
    finally:
        await self._release_model(model_id)
```

### Type-Safe Database Access
```python
def load_debate(self, debate_id: int) -> FullTranscriptData | None:
    """Load debate with proper typing."""
    with self._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM debates WHERE id = ?", (debate_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return FullTranscriptData(
            metadata={...},
            messages=[...],
        )
```

### Streaming Response Pattern
```python
async def generate_response_stream(
    self,
    config: ModelConfig,
    messages: list[dict[str, str]],
    chunk_callback: Callable[[str, bool], Awaitable[None]],
    **overrides
) -> str:
    """Stream chunks to callback, return full content."""
    full_content = ""
    async for chunk in self._stream_chunks(config, messages, **overrides):
        full_content += chunk
        await chunk_callback(chunk, is_final=False)
    await chunk_callback("", is_final=True)
    return full_content
```

### Pydantic Model Serialization
```python
# Modern Pydantic v2
data = model.model_dump()  # NOT model.dict() (deprecated)
json_str = model.model_dump_json()  # NOT model.json() (deprecated)

# With exclusions
data = model.model_dump(exclude={"password"})
```

## Performance Considerations

### Model Caching
`CacheManager` caches model lists with 5-minute TTL to avoid repeated API calls to OpenRouter.

### Database Connection Pooling
Use `with self._get_connection()` context manager for automatic connection cleanup.

### Async Operations
All provider calls are async to avoid blocking during LLM generation (which can take 10-60s).

### WebSocket Backpressure
Chunk streaming includes small delays to prevent overwhelming frontend with rapid updates.

## Deployment

### Production Environment
- **Railway**: Automatic deployment from git push (Dockerfile or Procfile)
- **Environment Variables**: Set `OPENROUTER_API_KEY`, `ALLOWED_ORIGINS`, `PORT`
- **Database**: SQLite persists via Railway volumes

### Production Checklist
1. Set `ALLOWED_ORIGINS` to frontend domain
2. Configure `OPENROUTER_API_KEY` in environment
3. Set `PORT` if not using default 8000
4. Ensure `debates.db` is in volume mount for persistence
5. Monitor logs for provider errors and cost tracking

## Security Considerations

- **No API keys in code** - Use environment variables
- **HTTP-only session cookies** - Not accessible to JavaScript
- **bcrypt password hashing** - Proper salt rounds (12+)
- **SQL injection prevention** - Always use parameterized queries
- **CORS restrictions** - Explicit origin whitelist for production
- **Rate limiting** - `slowapi` middleware on auth endpoints

## Future Enhancements

- **PostgreSQL support** - For production-scale deployments
- **Redis caching** - For distributed model metadata cache
- **Background job queue** - For async cost queries and long-running tasks
- **Prometheus metrics** - For monitoring debate latency and costs
- **Additional providers** - Anthropic, Gemini, local vLLM