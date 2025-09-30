# Dialectus Architecture Refactor: Pure Library + Web API

## Executive Summary

**Goal**: Transform `dialectus-engine` from a monolithic web service into a clean Python library, with web functionality living in a separate `dialectus-web-api` service.

**Status**: Phases 1-3 complete (web API built, engine cleaned), Phase 4 in progress (frontend testing), Phase 5 pending (deployment).

**Architecture**: Engine is now a pure Python library with NO web server. Web API imports engine as a library and runs debates in-process.

**Key Benefits**:
- ✅ One service to deploy (not two)
- ✅ No HTTP/WebSocket proxy complexity
- ✅ No inter-service communication overhead
- ✅ Engine is a clean, importable Python package
- ✅ Simpler security model (only web API has attack surface)
- ✅ Faster (in-process function calls vs HTTP)
- ✅ Easier local development (one server)

## Decisions Made

| Question | Answer |
|----------|--------|
| **API URL Scheme** | `/api/*` (no versioning - keeps Vite proxy simple) |
| **Database Migration** | Fresh start - wipe and rebuild (auto-create logic exists, will be moved to web API) |
| **Email Service** | Resend (environment variables for API key, configure on Railway) |
| **CLI Approach** | Imports engine as Python library (no WebSocket, just callbacks) |
| **WebSocket Handling** | Web API handles WebSockets directly (no proxy needed) |
| **User-Debate Mapping** | Web API's database stores `(user_id, debate_id)` - engine knows nothing about users |
| **Engine Persistence** | NONE - engine has no database. Web API has SQLite with all tables (debates + users) |
| **Engine Installation** | Local folder reference during dev (`-e ../dialectus-engine`), PyPI after open-source |
| **Git History** | New repo from scratch (Option A) - no history preservation needed |
| **Implementation Order** | Sequential with safety: Web API skeleton → Engine integration → Clean engine → Deploy |
| **Testing Strategy** | Manual testing via `/docs` endpoints and frontend (no automated tests for refactor) |
| **Web API Repository** | `dialectus-web-api` folder exists (empty, private GitHub repo, git tracking active) |
| **Railway Volume** | Wipe and rebuild - destroy old `/data` volume, create new `/app/dialectus.db` volume for web API |
| **Engine Callbacks** | Add callback parameter to `run_full_debate()` in Phase 2 (currently no callback support) |
| **Config Auto-Creation** | Stays in engine (useful for library users), web API uses Railway env vars |
| **Resend Email** | Resend via `RESEND_API_KEY` + `EMAIL_ENABLED=true` (Railway/prod), `AUTH_DEVELOPMENT_MODE=true` skips email (local dev) |
| **Auth Development Mode** | Defaults to `true` (auto-verify users locally), set `AUTH_DEVELOPMENT_MODE=false` in production |

## Architecture

### 1. dialectus-engine (Pure Python Library)

**Purpose**: AI debate framework - importable Python package

**What It Is**:
```python
from debate_engine.core import DebateEngine
from models.manager import ModelManager

# Just use it like any Python library
engine = DebateEngine(config)
result = await engine.run_full_debate(...)
```

**What Stays**:
- `debate_engine/` - Core debate orchestration
- `models/` - Multi-provider model management
- `judges/` - AI judging system
- `formats/` - Debate format implementations
- `tournaments/` - Tournament logic

**What Goes**:
- `web/` directory - DELETED ENTIRELY
- `debate_engine/database/` - DELETED ENTIRELY (no persistence in engine)
- All auth code
- All email code
- All FastAPI code
- All WebSocket server code (engine keeps HTTP client streaming from Ollama/OpenRouter)

**Database**:
- **NONE** - Engine has no persistence layer
- Users of the library implement their own storage if needed
- Web API will handle all persistence (debates + users in single SQLite DB)

**Configuration**:
- `debate_config.json` - AI models, formats, judging criteria (auto-created from `debate_config.example.json`)
- `debate_config.example.json` - Git-tracked template (actual config is gitignored)
- `config/settings.py` - Pydantic models that auto-create config from example template (stays in engine)
- Environment variables: `OLLAMA_URL`, `OPENROUTER_API_KEY` (engine only needs these two)
- NO database config, NO email config, NO auth config, NO PORT (it's not a server!)
- Web API-specific config (JWT, email, CORS) lives in Railway environment variables

**Streaming Architecture**:
Engine receives HTTP streaming responses from Ollama/OpenRouter and forwards them via callbacks:
- **No WebSocket server in engine** - just callback functions
- **CLI usage**: Pass simple callback to print chunks as they arrive
- **Web API usage**: Pass callback that forwards chunks to WebSocket
- **Providers**: httpx (Ollama) and OpenAI SDK (OpenRouter) handle HTTP streaming

```python
# Engine's streaming callback API
async def run_full_debate(
    topic: str,
    format: str,
    callback: Callable[[str, dict], Awaitable[None]] | None = None
) -> DebateResult:
    """
    callback(event_type, data) gets called for:
    - "chunk": streaming token chunks as they arrive from provider
    - "message_complete": full message finished
    - "judge_decision": judging complete
    - "debate_complete": entire debate done
    """
```

**Entry Point**:
- `main.py` can stay for CLI-style local usage: `python main.py --topic "AI is sentient"`
- But NO `--web` flag - that's gone
- Or remove `main.py` entirely - library doesn't strictly need an entry point

### 2. dialectus-web-api (New FastAPI Service)

**Purpose**: Production web API - the ONLY web service

**What It Does**:
- Imports engine: `from debate_engine.core import DebateEngine`
- Runs debates in-process (no HTTP calls to engine)
- Handles all auth, email, WebSockets, user management
- Stores user-to-debate mappings in its own database

**API Endpoints** (no versioning - clean `/api/*` paths):
```
Auth:
  POST   /api/auth/register
  POST   /api/auth/login
  POST   /api/auth/logout
  POST   /api/auth/verify-email
  POST   /api/auth/forgot-password
  POST   /api/auth/reset-password
  GET    /api/auth/me

Debates:
  POST   /api/debates          # Runs debate via library, stores mapping
  GET    /api/debates          # List user's debates
  GET    /api/debates/{id}     # Get debate (permission check)
  DELETE /api/debates/{id}     # Delete debate (permission check)
  WebSocket /api/ws/debate/{id}  # Real-time streaming (auth check)

Metadata:
  GET    /api/models           # From ModelManager (engine import)
  GET    /api/formats          # From FormatRegistry (engine import)
  GET    /api/transcripts      # User's transcripts only
  GET    /api/transcripts/{id} # Permission check
  GET    /api/health           # Health check

Tournaments:
  POST   /api/tournaments
  GET    /api/tournaments
  GET    /api/tournaments/{id}
```

**Technology**:
- FastAPI
- SQLite for ALL persistence (debates + users in single DB)
- HTTP email service (Resend via environment variable API key)
- Imports `dialectus-engine` as library dependency (local folder during dev)

**Database Tables** (single `dialectus.db` file):
```sql
-- User management tables
CREATE TABLE users (
  id INTEGER PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  is_verified BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE sessions (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  token TEXT UNIQUE NOT NULL,
  expires_at TIMESTAMP NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Debate storage tables (includes user_id columns)
CREATE TABLE debates (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,  -- Owned by user
  topic TEXT NOT NULL,
  format TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE messages (
  id INTEGER PRIMARY KEY,
  debate_id INTEGER NOT NULL,
  speaker TEXT NOT NULL,
  content TEXT NOT NULL,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (debate_id) REFERENCES debates(id)
);

CREATE TABLE judge_decisions (
  id INTEGER PRIMARY KEY,
  debate_id INTEGER NOT NULL,
  judge_name TEXT NOT NULL,
  winner TEXT,
  reasoning TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (debate_id) REFERENCES debates(id)
);

CREATE TABLE tournaments (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,  -- Owned by user
  name TEXT NOT NULL,
  format TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id)
);
```

**Configuration**:
- Environment variables:
  - `JWT_SECRET_KEY` - For session tokens
  - `RESEND_API_KEY` - Resend email API key
  - `DATABASE_URL` - SQLite path (default: `dialectus.db`)
  - `OPENROUTER_API_KEY` - For AI models
  - `OLLAMA_URL` - For local models (optional)
  - `PORT` - Web server port (default 8000)
  - `ALLOWED_ORIGINS` - CORS origins

**How It Runs Debates with Streaming**:
```python
# web-api/endpoints/debates.py
from debate_engine.core import DebateEngine
from fastapi import WebSocket

@router.websocket("/api/ws/debate/{id}")
async def debate_websocket(websocket: WebSocket, debate_id: int):
    await websocket.accept()

    # Create debate record in DB
    debate_id = db.execute(
        "INSERT INTO debates (user_id, topic, format, status) VALUES (?, ?, ?, ?)",
        (user_id, request.topic, request.format, "running")
    )

    # Callback bridges engine streaming to WebSocket
    async def ws_callback(event_type: str, data: dict):
        await websocket.send_json({
            "type": event_type,
            **data
        })

        # Save complete messages to DB
        if event_type == "message_complete":
            db.execute(
                "INSERT INTO messages (debate_id, speaker, content) VALUES (?, ?, ?)",
                (debate_id, data['speaker'], data['content'])
            )

    # Run debate - engine streams via callback, no HTTP between services!
    engine = DebateEngine(config)
    result = await engine.run_full_debate(
        topic=request.topic,
        format=request.format,
        callback=ws_callback  # Forwards streaming chunks to WebSocket
    )

    # Update debate status
    db.execute(
        "UPDATE debates SET status=? WHERE id=?",
        ("completed", debate_id)
    )
```

### 3. dialectus-cli (Separate Repo - Future Work)

**Not part of this refactor - ignore for now**

CLI will import engine as library with streaming via callbacks (no WebSocket needed):
```python
from debate_engine.core import DebateEngine

# Simple callback for token-by-token streaming
async def print_chunk(event_type: str, data: dict):
    if event_type == "chunk":
        print(data["content"], end="", flush=True)  # Token-by-token streaming!
    elif event_type == "message_complete":
        print("\n")  # Newline after full message

engine = DebateEngine(config)
result = await engine.run_full_debate(
    topic="AI sentience",
    format="oxford",
    callback=print_chunk  # Gets streaming chunks as they arrive from provider
)
```

### 4. dialectus-web (Existing Frontend)

**Changes Needed**: NONE (if API contract stays the same)

Frontend still talks to same endpoints:
- `POST /api/auth/login`
- `POST /api/debates`
- `WebSocket /api/ws/debate/{id}`

Just point to web API URL instead of engine URL.

## Railway Deployment Strategy

**Before Refactor** (current):
- Service #1: dialectus-engine (FastAPI with auth/email/everything)
- Service #2: dialectus-web (static site)

**After Refactor**:
- Service #1: dialectus-web-api (FastAPI with auth/email, imports engine as library)
- Service #2: dialectus-web (static site)

**Deployment Changes**:
1. Delete current `dialectus-engine` Railway service
2. Create new `dialectus-web-api` Railway service:
   - Private GitHub repo
   - Environment: `JWT_SECRET_KEY`, `RESEND_API_KEY`, `OPENROUTER_API_KEY`, `OLLAMA_URL`, `ALLOWED_ORIGINS`
   - Persistent volume: `/app/dialectus.db` (single database file)
   - URL: `https://api.dialectus.ai`
3. Update `dialectus-web` environment variables:
   - Change API URL to `https://api.dialectus.ai`

## Remaining Work

### ✅ Phase 1-3: COMPLETE
- Web API skeleton built with auth, database, email service
- Engine integrated as library dependency with callback streaming
- Engine cleaned (removed web/, database/, all web dependencies)
- Auth defaults to development mode (`AUTH_DEVELOPMENT_MODE=true` auto-verifies users)
- Frontend updated (API paths changed from `/v1/*` to `/api/*`, WebSocket path to `/api/ws/*`)

### 🚧 Phase 4: IN PROGRESS - Frontend Testing

**Completed**:
- ✅ Removed API versioning from `.env` files
- ✅ Updated Vite proxy config for `/api/*` paths (with WebSocket support)
- ✅ Updated `api-service.ts` to auto-prepend `/api` prefix
- ✅ Updated `websocket-service.ts` to use `/api/ws/debate/{id}` path
- ✅ Set `AUTH_DEVELOPMENT_MODE` default to `true` for easier local dev

**In Progress**:
- 🔄 Manual testing by user:
  - Registration/login flows
  - Debate creation
  - WebSocket streaming (token-by-token)
  - Transcript viewing
- 🔄 Fix any API contract differences discovered

### ⏳ Phase 5: Deploy to Railway

**Steps**:

1. **Create new Railway service for web API**:
   - Connect `dialectus-web-api` GitHub repo
   - Set environment variables:
     - `AUTH_DEVELOPMENT_MODE=false` (require email verification)
     - `EMAIL_ENABLED=true`
     - `RESEND_API_KEY=<key>`
     - `JWT_SECRET_KEY=<generate-secret>`
     - `OPENROUTER_API_KEY=<key>`
     - `OLLAMA_URL=<optional>`
     - `ALLOWED_ORIGINS=https://dialectus.ai`
   - Add persistent volume: `/app/dialectus.db`
   - Deploy and verify via `/api/health`

2. **Update frontend environment** (Railway):
   - Set `VITE_API_BASE_URL=https://api.dialectus.ai`
   - Redeploy `dialectus-web`

3. **Delete old engine service**:
   - Remove `dialectus-engine` Railway service (replaced by `dialectus-web-api`)
   - Archive or delete old database volume

4. **Test production end-to-end**:
   - Register new user (email verification required)
   - Login and create debate
   - Verify WebSocket streaming works
   - Check transcript persistence

## Success Criteria

- [x] Engine has no web dependencies (FastAPI, Uvicorn removed)
- [x] Engine has no database/persistence code
- [x] Engine can be imported as library: `from debate_engine.core import DebateEngine`
- [x] Engine streaming works via callbacks (no WebSocket server in engine)
- [x] Web API runs debates in-process (no HTTP calls between services)
- [x] Web API has single database with all tables (users + debates)
- [x] Web API bridges engine callbacks to WebSocket for frontend streaming
- [x] Frontend API calls updated to use `/api/*` paths
- [x] Frontend WebSocket updated to use `/api/ws/*` path
- [ ] Frontend testing complete (auth, debates, streaming, transcripts)
- [ ] Only ONE Railway service deployed (web API replaces engine)
- [ ] Engine repo is clean and ready to open-source
- [ ] Authentication and email work via web API (production with Resend)
- [ ] Production deployment tested end-to-end

## Key Benefits

1. **Simpler Architecture**: One web service instead of two
2. **Better Performance**: In-process function calls vs HTTP
3. **Cleaner Code**: Engine is pure AI logic, web API is pure business logic
4. **Easier Development**: One server to run locally
5. **Lower Costs**: One Railway service instead of two
6. **Resume Value**: Clean open-source AI framework
7. **No Proxy Complexity**: WebSocket handled directly by web API
8. **Token-by-Token Streaming Preserved**: Engine callbacks forward provider streaming chunks
9. **CLI Streaming**: No WebSocket needed - just callback functions for real-time output

## What This Looks Like in Code

### Architecture Comparison

**Before** (monolithic engine with web layer):
```python
# client → HTTP → engine FastAPI → DebateEngine
POST https://engine.dialectus.ai/api/debates
  ↓
engine/web/api.py receives request
  ↓
engine/web/endpoints/debates.py runs debate
  ↓
Returns result
```

**After** (web API imports engine):
```python
# client → HTTP → web-api FastAPI → import DebateEngine
POST https://api.dialectus.ai/api/debates
  ↓
web-api/endpoints/debates.py receives request
  ↓
from debate_engine.core import DebateEngine  # Import!
engine = DebateEngine(config)
result = await engine.run_full_debate(...)  # In-process call!
  ↓
Returns result
```

**No HTTP between services!** Just Python imports and function calls.

### Streaming Flow

**How streaming works without WebSocket in engine**:

```
Provider (Ollama/OpenRouter)
  ↓ HTTP streaming (httpx/OpenAI SDK)
Engine HTTP client receives chunks
  ↓ Invoke callback with each chunk
callback(event_type="chunk", data={"content": "token"})
  ↓
Web API: Forward chunk to WebSocket
  OR
CLI: Print chunk to terminal
```

**Key insight**: Engine receives streaming responses from providers via HTTP clients (httpx, OpenAI SDK). It just needs to invoke a callback function with each chunk. Web API bridges that callback to WebSocket. CLI bridges it to stdout. **No WebSocket server in engine needed!**
