# Tournament System Implementation Specification

## Status
**CRITICAL**: The README.md exists in `tournaments/README.md` but ALL CODE FILES ARE MISSING. Need complete reimplementation.

## Overview
Implement a March Madness-style single-elimination tournament system for AI model debates with cost-based weight classes, automatic bracket generation, and real-time progression tracking. The system was previously implemented but lost - only the README survived.

## Missing Files That Need Implementation

Based on the README and transcript, these files were created but are now MISSING:

```
tournaments/
├── __init__.py              # ❌ MISSING - Package exports  
├── models.py                # ❌ MISSING - Pydantic data models
├── database.py              # ❌ MISSING - SQLite database operations
├── manager.py               # ❌ MISSING - Core tournament logic
├── api.py                   # ❌ MISSING - FastAPI endpoint handlers
├── debate_integration.py    # ❌ MISSING - Debate system callbacks
└── README.md               # ✅ EXISTS - Documentation (survived)
```

**Also Missing:**
- `test_tournament_system.py` in project root
- `src/types/tournament-types.ts` for frontend types
- Database schema additions (tables were never created)
- API endpoint integration in `web/api.py`
## Core Features (From README Evidence)
- **Cost-based weight classes** (Free, Budget, Economy, Premium, Elite) to ensure fair competition
- **Preview model exclusion** to prevent anonymous ringers  
- **Single-elimination brackets** supporting 4, 8, 16, 32, or 64 models
- **Value-score seeding** for fair initial matchups with March Madness pairing
- **Unique topics per match** to add variety
- **Fixed judges** for entire tournament to ensure consistency
- **Automatic tournament progression** with concurrent match execution
- **Real-time WebSocket updates** for live tournament tracking
- **Full integration** with existing debate system via callbacks

## Database Schema Extensions

### New Tables to Add

```sql
-- Tournament metadata
CREATE TABLE tournaments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    weight_class TEXT NOT NULL,  -- 'free', 'budget', 'economy', 'premium', 'elite'
    format TEXT NOT NULL,
    word_limit INTEGER NOT NULL,
    status TEXT NOT NULL,  -- 'created', 'in_progress', 'completed', 'cancelled'
    bracket_size INTEGER NOT NULL,  -- 4, 8, 16, 32, 64
    current_round INTEGER DEFAULT 1,
    total_rounds INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME,
    completed_at DATETIME,
    winner_model_id TEXT,
    tournament_metadata TEXT  -- JSON: judge_models, settings
);

-- Tournament participants (seeded models)
CREATE TABLE tournament_participants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_id INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    seed_number INTEGER NOT NULL,  -- 1-64 seeding
    eliminated_in_round INTEGER,  -- NULL if still active
    FOREIGN KEY (tournament_id) REFERENCES tournaments (id)
);

-- Tournament matches (bracket matchups)
CREATE TABLE tournament_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_id INTEGER NOT NULL,
    round_number INTEGER NOT NULL,
    match_number INTEGER NOT NULL,
    model_a_id TEXT NOT NULL,
    model_b_id TEXT,  -- NULL for bye matches
    winner_model_id TEXT,  -- NULL until debate completes
    debate_id TEXT,  -- Links to existing debates table
    topic TEXT NOT NULL,  -- Unique topic per match
    status TEXT NOT NULL,  -- 'pending', 'in_progress', 'completed'
    FOREIGN KEY (tournament_id) REFERENCES tournaments (id)
);

### tournament_matches (CRITICAL TABLE)
- `id` - Primary key
- `tournament_id` - Foreign key to tournaments  
- `round_number` - 1, 2, 3... up to total_rounds
- `match_number` - Match within round (1, 2, 3...)
- `model_a_id` - First model (higher seed)
- `model_b_id` - Second model (NULL for bye matches)
- `winner_model_id` - Winner (NULL until debate completes)
- `debate_id` - Links to existing debates table
- `topic` - **UNIQUE topic per match** (not tournament-wide topic)
- `status` - 'pending', 'in_progress', 'completed', 'bye'

### tournament_judges (Fixed judges for consistency)
- `id` - Primary key  
- `tournament_id` - Foreign key to tournaments
- `judge_model_id` - Fixed judge model for entire tournament
- `judge_provider` - Judge provider ('ollama', 'openrouter')
```

**Key Schema Notes:**
- **NO topic field in tournaments table** - each match has unique topic
- **Fixed judges per tournament** - stored in separate table
- **Bye match handling** - model_b_id can be NULL
- **Integration** - debate_id links to existing debates table

## Weight Classes (Cost-Based)

## Exact Weight Classes (From README)

```python
class WeightClass(Enum):
    FREE = "free"        # 🆓 Ollama + legitimate free OpenRouter (no preview)
    BUDGET = "budget"    # 💰 $0 - $0.001/1K tokens  
    ECONOMY = "economy"  # 💼 $0.001 - $0.005/1K tokens
    PREMIUM = "premium"  # 💎 $0.005 - $0.02/1K tokens
    ELITE = "elite"      # 👑 $0.02+/1K tokens
```

**Examples from README:**
- **Free**: Llama 3.2 3B, Qwen 2.5 variants
- **Budget**: Ultra-cheap text models  
- **Economy**: Llama 3.1 8B, Gemma 2 9B
- **Premium**: Claude 3.5 Haiku, GPT-4o Mini
- **Elite**: Claude 3.5 Sonnet, GPT-4o, O1

**Model Selection Logic:**
- Filter models by cost ranges for weight class
- Exclude preview models (`is_preview == True`)
- Sort by `value_score` for seeding
- Take top N models for bracket size

## File Structure

```
tournaments/
├── __init__.py           # Module exports
├── models.py            # Pydantic data models
├── database.py          # Database operations
├── manager.py           # Core tournament logic
├── api.py              # API endpoints
├── debate_integration.py # Debate system callbacks
└── README.md           # Documentation
```

## Core Classes

### TournamentManager (tournaments/manager.py)

```python
class TournamentManager:
    def __init__(self, db_path: str = "tournaments.db", debate_manager=None):
        self.db = TournamentDatabaseManager(db_path)
        self.debate_manager = debate_manager

    async def create_tournament(self, request: TournamentCreateRequest) -> int:
        """Create tournament with bracket generation"""
        
    async def start_tournament(self, tournament_id: int):
        """Begin tournament execution"""
        
    async def advance_tournament(self, tournament_id: int):
        """Advance to next round after completion"""
        
    def get_tournament_status(self, tournament_id: int) -> Tournament:
        """Get current tournament state"""
        
    def get_bracket_view(self, tournament_id: int) -> dict:
        """Get bracket visualization data"""
```

### Bracket Generation Algorithm

1. **Get eligible models** for weight class using existing model manager
2. **Filter out preview models** to prevent ringers
3. **Rank by value_score** for seeding (1 = highest score)
4. **Create single-elimination bracket** with classic March Madness pairing:
   - #1 vs #64, #2 vs #63, #32 vs #33, etc.
5. **Handle byes** for non-power-of-2 participant counts
6. **Generate unique topics** for each match to add variety

## API Endpoints

Add to existing `web/api.py`:

```python
# Tournament API endpoints
POST /api/tournaments                    # Create tournament
GET  /api/tournaments                    # List tournaments
GET  /api/tournaments/{id}               # Tournament details  
POST /api/tournaments/{id}/start         # Begin tournament
POST /api/tournaments/{id}/advance       # Advance to next round
GET  /api/tournaments/{id}/bracket       # Bracket visualization
GET  /api/tournaments/{id}/matches       # Current round matches
DELETE /api/tournaments/{id}             # Cancel tournament
WebSocket /ws/tournament/{id}            # Real-time updates
```

## Integration with Existing Systems

### Debate System Integration
- Use existing `DebateManager.create_debate()` for each match
- Use existing `DebateEngine` and judging system  
- Store results in existing `debates`/`messages` tables
- Link via `tournament_matches.debate_id`

### Tournament Callback System
- Add callback support to `DebateManager`
- Automatically detect tournament debate completion
- Extract winner from judge results
- Update tournament match status and advance rounds

```python
# In DebateManager
def add_tournament_callback(self, callback_func):
    self.tournament_callbacks.append(callback_func)

async def _notify_tournament_completion(self, debate_id: str, judge_result):
    # Extract winner and notify tournament system
```

## Tournament Execution Flow

1. **Tournament Creation**
   - Select models for weight class
   - Generate bracket with seeding
   - Create all match records
   - Set status to 'created'

2. **Tournament Start**  
   - Create debates for first round matches
   - Start all debates concurrently
   - Set status to 'in_progress'

3. **Round Completion**
   - Detect when all matches in round finish
   - Automatically advance winners to next round
   - Generate new match records for next round

4. **Tournament Completion**
   - Crown champion when final match completes
   - Set status to 'completed'
   - Record winner and completion time

## Request/Response Models

```python
class TournamentCreateRequest(BaseModel):
    name: str
    weight_class: WeightClass
    format: str = "oxford"
    word_limit: int = 500
    bracket_size: int = 8
    judge_models: List[str] = ["llama3.1:8b"]
    judge_provider: str = "ollama"

class Tournament(BaseModel):
    id: int
    name: str
    weight_class: WeightClass
    format: str
    word_limit: int
    status: TournamentStatus
    bracket_size: int
    current_round: int
    total_rounds: int
    created_at: Optional[datetime]
    started_at: Optional[datetime] 
    completed_at: Optional[datetime]
    winner_model_id: Optional[str]
```

## Implementation Priority & Recovery Strategy

### Phase 1: Database & Core Models (IMMEDIATE)
- [ ] Create database schema (4 new tables)
- [ ] Implement `tournaments/models.py` with all Pydantic models
- [ ] Implement `tournaments/database.py` with CRUD operations
- [ ] Test database schema creation and basic operations

### Phase 2: Tournament Logic (CRITICAL PATH)  
- [ ] Implement `tournaments/manager.py` with TournamentManager class
- [ ] Bracket generation algorithm with March Madness seeding
- [ ] Model filtering by weight class and preview exclusion
- [ ] Tournament status management (created → in_progress → completed)

### Phase 3: Integration & API (ESSENTIAL)
- [ ] Implement `tournaments/debate_integration.py` callback system
- [ ] Implement `tournaments/api.py` with FastAPI handlers  
- [ ] Add tournament endpoints to `web/api.py`
- [ ] Set up debate completion callbacks in main app

### Phase 4: Testing & Polish
- [ ] Create `test_tournament_system.py` 
- [ ] WebSocket tournament updates
- [ ] Frontend TypeScript types
- [ ] Performance optimization

## Critical Integration Points (From Transcript)

### DebateManager Callback System
The tournament system MUST register callbacks with the existing DebateManager:

```python
# In web/api.py setup
tournament_callback = TournamentDebateCallback(tournament_db)
debate_manager.add_tournament_callback(tournament_callback.on_debate_completed)
```

### Debate Creation for Matches
Tournament matches create real debates using existing infrastructure:

```python
# Tournament matches use DebateSetupRequest
setup = DebateSetupRequest(
    topic=match.topic,  # Unique per match
    format=tournament.format,
    word_limit=tournament.word_limit, 
    models={"model_a": model_a_config, "model_b": model_b_config},
    judge_models=[judge.judge_model_id for judge in judges]
)
debate_id = await self.debate_manager.create_debate(setup)
await self.debate_manager.start_debate(debate_id)  # Auto-start for tournaments
```

### Winner Determination
Extract winners from judge ensemble results:

```python
async def _notify_tournament_completion(self, debate_id: str, judge_result):
    winner_id = None
    if judge_result and isinstance(judge_result, dict):
        if judge_result.get("type") == "ensemble":
            winner_id = judge_result["ensemble_summary"]["final_winner_id"]
```

## Key Integration Points

1. **Model Manager**: Use existing enhanced models API with cost filtering
2. **Debate Manager**: Extend with tournament callback system
3. **Judge System**: Use existing multi-judge ensemble decisions
4. **WebSocket**: Extend existing real-time updates for tournaments
5. **Database**: Add new tables to existing SQLite schema

## Success Criteria

- [ ] Can create tournaments in all weight classes
- [ ] Bracket generation works for all sizes (4-64)
- [ ] Preview models are properly excluded
- [ ] Tournament matches execute as real debates
- [ ] Winners advance automatically through rounds
- [ ] Real-time updates work via WebSocket
- [ ] Tournament completion properly crowns champion
- [ ] Full integration with existing debate system

## Evidence of What Was Actually Implemented

### From README (Survived):
- Complete documentation of weight classes with specific examples
- Detailed API endpoint specifications
- Database schema descriptions (but tables don't exist)
- Usage examples and integration details

### From Transcript (Lost Implementation):
- All 6+ Python files in tournaments/ directory
- Database schema creation scripts  
- Complete API integration in web/api.py
- Callback system implementation
- Test suite creation
- TypeScript definitions

### Current Reality Check:
```bash
# What exists:
✅ tournaments/README.md (comprehensive docs)

# What's missing (need to rebuild):
❌ tournaments/__init__.py
❌ tournaments/models.py  
❌ tournaments/database.py
❌ tournaments/manager.py
❌ tournaments/api.py
❌ tournaments/debate_integration.py
❌ test_tournament_system.py
❌ Database tables (were never actually created)
❌ API integration in web/api.py
❌ Frontend TypeScript types
```

## Recovery Strategy for Claude Code

1. **Start with Database Schema** - Create all 4 tables first to establish foundation
2. **Build Core Models** - Pydantic classes matching the README specifications  
3. **Implement Database Operations** - CRUD operations for all tournament entities
4. **Create Tournament Manager** - Core business logic with bracket generation
5. **Add Integration Layer** - Callbacks and debate system integration
6. **Implement API Endpoints** - FastAPI handlers and main app integration
7. **Create Test Suite** - Verify everything works end-to-end

The README provides the exact API contracts, data models, and feature specifications. Combined with the transcript details about implementation patterns, this should be sufficient to rebuild the entire system.