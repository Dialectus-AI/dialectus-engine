# 🏆 Tournament System

A March Madness-style tournament system for AI model debates, featuring cost-based weight classes and fair competition brackets.

## Overview

The tournament system creates single-elimination brackets where AI models compete head-to-head in debates within their cost-based weight classes. Models are seeded by their value scores and advance through rounds until a champion is crowned.

## Weight Classes (Cost-Based)

### 🆓 Free Division
- **Models**: Ollama models + legitimate free OpenRouter models
- **Cost**: $0 (excludes preview models)
- **Examples**: Llama 3.2 3B, Qwen 2.5 variants

### 💰 Budget Division
- **Cost Range**: $0 - $0.001/1K tokens
- **Examples**: Ultra-cheap text models

### 💼 Economy Division
- **Cost Range**: $0.001 - $0.005/1K tokens
- **Examples**: Llama 3.1 8B, Gemma 2 9B

### 💎 Premium Division
- **Cost Range**: $0.005 - $0.02/1K tokens
- **Examples**: Claude 3.5 Haiku, GPT-4o Mini

### 👑 Elite Division
- **Cost Range**: $0.02+/1K tokens
- **Examples**: Claude 3.5 Sonnet, GPT-4o, O1

## Features

### Fair Competition
- **Preview Model Exclusion**: Anonymous preview models are automatically excluded to prevent "ringers"
- **Cost-Based Classes**: Models compete against similar-capability opponents
- **Fixed Judges**: Same judges for entire tournament ensure consistency

### Tournament Structure
- **Bracket Sizes**: 4, 8, 16, 32, or 64 models
- **Single Elimination**: Lose and you're out, like March Madness
- **Seeded Competition**: Models ranked by value_score for fair initial matchups
- **Unique Topics**: Each match gets its own debate topic

### Real-Time Execution
- **Concurrent Matches**: Multiple debates run simultaneously within rounds
- **Auto-Advancement**: Tournament automatically advances when rounds complete
- **WebSocket Updates**: Live progress tracking
- **Result Integration**: Seamless integration with existing debate system

## API Endpoints

### Tournament Management
```http
POST   /tournaments                    # Create tournament
GET    /tournaments                    # List tournaments
GET    /tournaments/{id}               # Tournament details
POST   /tournaments/{id}/start         # Start tournament
POST   /tournaments/{id}/advance       # Manual advancement
DELETE /tournaments/{id}               # Cancel tournament
```

### Tournament Data
```http
GET    /tournaments/{id}/bracket       # Bracket visualization data
GET    /tournaments/{id}/matches       # Current round matches
WebSocket /ws/tournament/{id}              # Real-time updates
```

## Database Schema

### tournaments
- `id` - Tournament ID
- `name` - Tournament name
- `weight_class` - Cost-based weight class
- `format` - Debate format (oxford, parliamentary, etc.)
- `word_limit` - Word limit per response
- `status` - created, in_progress, completed, cancelled
- `bracket_size` - 4, 8, 16, 32, 64
- `current_round` - Current tournament round
- `total_rounds` - Total rounds (log₂ of bracket_size)
- `winner_model_id` - Champion model

### tournament_participants
- `tournament_id` - Foreign key to tournaments
- `model_id` - Model identifier
- `model_name` - Display name
- `seed_number` - 1-64 seeding position
- `eliminated_in_round` - Round where eliminated

### tournament_matches
- `tournament_id` - Foreign key to tournaments
- `round_number` - 1, 2, 3... up to total_rounds
- `match_number` - Match within round
- `model_a_id` - First model (higher seed)
- `model_b_id` - Second model (NULL for bye)
- `winner_model_id` - Winner (NULL until complete)
- `debate_id` - Links to existing debates table
- `topic` - Unique debate topic for this match
- `status` - pending, in_progress, completed, bye

### tournament_judges
- `tournament_id` - Foreign key to tournaments
- `judge_model_id` - Fixed judge model
- `judge_provider` - Judge provider (ollama, openrouter)

## Usage Examples

### Create Tournament
```python
from tournaments.models import TournamentCreateRequest, WeightClass

request = TournamentCreateRequest(
    name="Free Division Championship",
    weight_class=WeightClass.FREE,
    format="oxford",
    word_limit=300,
    bracket_size=16,
    judge_models=["llama3.2:3b", "qwen2.5:3b"],
    judge_provider="ollama"
)

tournament_id = await tournament_manager.create_tournament(request)
```

### Start Tournament
```python
await tournament_manager.start_tournament(tournament_id)
```

### Get Bracket Data
```python
bracket = tournament_manager.get_bracket_data(tournament_id)
print(f"Participants: {len(bracket.participants)}")
print(f"Matches: {len(bracket.matches)}")
```

## Integration

### With Debate System
- Uses existing `DebateManager` for match execution
- Integrates with `DebateEngine` for actual debates
- Leverages existing judge system with ensemble support
- Stores results in existing transcript database

### Callback System
- Tournament system registers callbacks with `DebateManager`
- Automatically receives debate completion notifications
- Extracts winners from judge decisions
- Advances tournament rounds automatically

## Testing

Run the tournament system test:

```bash
cd dialectus-engine
python test_tournament_system.py
```

This will:
- Test weight class model filtering
- Create a test tournament
- Generate bracket with seeded participants
- Verify database operations
- Show tournament structure

## File Structure

```
tournaments/
├── __init__.py              # Package exports
├── models.py                # Pydantic data models
├── database.py              # SQLite database operations
├── manager.py               # Core tournament logic
├── api.py                   # FastAPI endpoint handlers
├── debate_integration.py    # Debate system callbacks
└── README.md               # This file
```

## Architecture

### Tournament Flow
1. **Creation** - Validate request, filter eligible models, seed by value_score
2. **Bracket Generation** - Create single-elimination matches with unique topics
3. **Execution** - Start debates concurrently, track results via callbacks
4. **Advancement** - Auto-advance when rounds complete, generate next round
5. **Completion** - Determine champion, update tournament status

### Key Design Principles
- **Cost-Based Fair Play** - Models compete within similar capability ranges
- **Preview Model Exclusion** - Prevents anonymous ringers from dominating
- **Fixed Judging** - Same judges throughout ensure consistent evaluation
- **Real-Time Updates** - WebSocket integration for live tournament tracking
- **Seamless Integration** - Builds on existing debate infrastructure

## Future Enhancements

- **Round Robin Tournaments** - All models play each other
- **Swiss System** - Pair models with similar records
- **Elo Ratings** - Track model performance over time
- **Tournament Templates** - Pre-configured tournament types
- **Statistics Dashboard** - Tournament analytics and insights