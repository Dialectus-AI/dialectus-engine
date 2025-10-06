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

**Dialectus Engine** is the core Python library for the Dialectus AI debate system. This is an importable package providing:

- **Debate orchestration** - DebateEngine coordinates multi-turn debates between AI models
- **Multi-provider AI integration** - Supports Ollama (local) and OpenRouter (cloud) models
- **Format system** - Pluggable debate formats (Oxford, Parliamentary, Socratic, Public Forum)
- **AI judging** - Ensemble judge system for objective debate evaluation
- **Model management** - Provider abstraction with async streaming support
- **Configuration system** - Pydantic-based config with validation

This is one of four repositories in the Dialectus ecosystem:
- **dialectus-engine** (this repo) - Python debate orchestration library
- **dialectus-web-api** - FastAPI backend providing REST API and WebSocket endpoints
- **dialectus-web** - TypeScript/Vite frontend SPA
- **dialectus-cli** - Node.js cross-platform CLI tool

## Environment Note

**Development Platform**: Windows 11 with Git Bash

- Unix shell commands and syntax used throughout
- Standard bash redirects (e.g., `2>/dev/null`) work as expected
- **Claude Code Edit tool**: Use relative paths for cross-repo file operations
  - Example: `../dialectus-web-api/CLAUDE.md`
  - Alternative: Proper Git Bash absolute paths like `/g/ai/repos/dialectus/dialectus-web-api/CLAUDE.md`
  - Relative paths are simplest and most portable

## Code Quality Standards

**This project has a HIGH bar.** The developer has 30 years of professional software engineering experience and expects:

### Python Standards
- **Python 3.13+** - Use modern syntax and features
- **PEP 8** - Follow Python style guide strictly
- **Modern type hints** - Use `list[str]`, `dict[str, int]`, `X | None` (not `List`, `Dict`, `Optional`)
- **Pyright strict mode** - Zero type errors tolerated (see pyrightconfig.json)
- **No `Any` types** - Use proper annotations with generics, protocols, unions
- **Pydantic v2** - Use `model_dump()` not deprecated `dict()`, `model_dump_json()` not `json()`
- **Modern async/await** - Proper async context managers and generators
- **No unused imports or variables** - Keep codebase clean
- **Type-safe dict access** - Use `.get()` or explicit `in` checks for optional keys
- **Collections from built-ins** - `from collections.abc import Sequence` not `typing.Sequence`
- **Docstrings** - All public classes, methods, functions need docstrings (Google style preferred)

### Pre-Commit Requirements
**CRITICAL**: Before ANY commit or code completion:
1. Run Pyright/Pylance type check - Must pass with zero errors
2. Check PEP 8 compliance (line length, naming, imports)
3. Test with actual debate if modifying core engine logic
4. Verify both Ollama and OpenRouter providers if modifying model code

### Project Conventions
- **Pydantic for all data models** - Config, messages, decisions, responses
- **Async by default** - All model interactions must be async
- **Context managers** - Use `async with` for session management
- **Protocol classes** - Use `Protocol` for duck typing over ABC when appropriate
- **Enums for constants** - Use `Enum` classes for debate phases, positions, etc.
- **Structured logging** - Use logger with appropriate levels (DEBUG, INFO, WARNING, ERROR)

## Development Commands

```bash
# Install as editable package (for local development)
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Install/update dependencies
pip install -r requirements.txt

# Recompile dependencies after editing requirements.in
pip-compile requirements.in

# Type checking (Pyright via VS Code Pylance or CLI)
pyright .

# Run tests (when implemented)
pytest

# Build distribution packages
python -m build
```

## Architecture

### Technology Stack
- **Python 3.13** with modern type hints (`X | None`, `list[T]`, `dict[K, V]`)
- **Pydantic v2** for data validation and settings management
- **OpenAI SDK** for OpenRouter API integration (streaming support)
- **httpx** for async HTTP requests (Ollama provider)
- **asyncio** for concurrent debate operations

### Core Structure
```
dialectus-engine/
├── debate_engine/          # Core debate orchestration
│   ├── core.py             # DebateEngine - main debate coordinator
│   ├── models.py           # DebateContext, DebateMessage (Pydantic models)
│   └── types.py            # Enums: DebatePhase, Position
├── models/                 # AI model provider abstraction
│   ├── manager.py          # ModelManager - multi-provider orchestration
│   ├── base_types.py       # Base model info protocol/types
│   ├── cache_manager.py    # In-memory cache with TTL for model metadata
│   ├── providers/
│   │   ├── base_model_provider.py    # BaseModelProvider abstract class
│   │   ├── ollama_provider.py        # OllamaProvider (local models)
│   │   ├── open_router_provider.py   # OpenRouterProvider (cloud models)
│   │   └── providers.py              # ProviderFactory
│   └── openrouter/         # OpenRouter-specific types and utilities
│       ├── openrouter_model.py       # Model data structures
│       ├── openrouter_filter_config.py  # Filter configuration
│       └── ...             # Architecture, pricing, capability extraction
├── judges/                 # AI judge implementations
│   ├── base.py             # JudgeDecision, CriterionScore (Pydantic)
│   ├── ai_judge.py         # AIJudge - LLM-based debate evaluation
│   ├── ensemble_utils.py   # Aggregates multiple judge decisions
│   └── factory.py          # JudgeFactory - creates judge instances
├── formats/                # Debate format definitions
│   ├── base.py             # DebateFormat base class, FormatPhase
│   ├── registry.py         # FormatRegistry singleton
│   ├── oxford.py           # OxfordFormat - classic debate structure
│   ├── parliamentary.py    # ParliamentaryFormat - British style
│   ├── public_forum.py     # PublicForumFormat - American high school
│   └── socratic.py         # SocraticFormat - question-driven
├── config/                 # Configuration system
│   ├── settings.py         # Pydantic config models (AppConfig, ModelConfig)
│   └── openrouter_filters.json  # Model exclusion patterns
├── pyproject.toml          # Package metadata and dependencies
├── requirements.in         # Direct dependencies (source for pip-compile)
├── requirements.txt        # Pinned dependencies (generated by pip-compile)
└── pyrightconfig.json      # Pyright/Pylance type checker config
```

### Key Patterns

**Multi-Provider Architecture**: The `ModelManager` abstracts provider differences. Register models with `register_model(id, ModelConfig)`, then call `generate_response(id, messages)`. The manager routes to the correct provider (Ollama or OpenRouter) transparently.

**Async Context Managers**: Providers use `@asynccontextmanager` for session management. The pattern ensures proper resource cleanup even on errors:
```python
@asynccontextmanager
async def model_session(self, model_id: str):
    try:
        await self._acquire_model(model_id)
        yield
    finally:
        await self._release_model(model_id)
```

**Pydantic Everywhere**: All config, data models, and responses use Pydantic v2 with strict validation. This provides automatic serialization, validation, and type safety.

**Format Registry**: Debate formats are registered via `@format_registry.register(name)` decorator. The registry enables dynamic format discovery and instantiation.

**Streaming Support**: Both providers support chunk-by-chunk streaming via callback pattern:
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

## Integration with Other Repos

### Backend API (dialectus-web-api)
The FastAPI backend imports this library to:
- Create `DebateEngine` instances for WebSocket streaming
- Use `ModelManager` for model discovery endpoints
- Access `FormatRegistry` for format listings
- Import judge factories for evaluation

### CLI Tool (dialectus-cli)
The Node.js CLI tool uses the Python backend API, which in turn uses this library.

### Frontend (dialectus-web)
The TypeScript SPA consumes the API provided by dialectus-web-api, which uses this library for all debate logic.

## Configuration System

### Configuration Files
- `debate_config.json` - Main application config (created from `debate_config.example.json`)
- `config/openrouter_filters.json` - Model filtering patterns (excludes meta-models, routers)

### Environment Variables
- `OPENROUTER_API_KEY` - OpenRouter API key (required for cloud models)
- `OLLAMA_URL` - Ollama server URL (default: `http://localhost:11434`)

### Pydantic Config Models
All configuration uses Pydantic v2 models in `config/settings.py`:

- `AppConfig` - Top-level config (debate, judging, system, models)
- `ModelConfig` - Per-model config (name, provider, temperature, max_tokens, cost, context_window)
- `SystemConfig` - System settings (Ollama URL, OpenRouter config)
- `DebateConfig` - Debate settings (topic, format, word_limit)
- `JudgingConfig` - Judge settings (criteria, judge_models, judge_provider)

Example:
```python
from config.settings import AppConfig

config = AppConfig.from_json_file("debate_config.json")
debate_topic = config.debate.topic
model_config = config.models["llama3.2:latest"]
```

## Model Provider System

### Provider Abstraction
`BaseModelProvider` defines the interface:
- `get_models()` - List available models
- `generate_response()` - Single-shot completion
- `generate_response_stream()` - Streaming completion with callbacks
- `verify_availability()` - Health check

### Ollama Provider
- Connects to local Ollama instance via HTTP
- Uses `/api/generate` for streaming
- Supports hardware optimization (GPU layers, keep_alive)
- Models like `llama3.2:latest`, `qwen2.5:3b`, etc.

### OpenRouter Provider
- Cloud-based model access via OpenRouter API
- Uses OpenAI SDK for streaming
- Supports 100+ models from various providers
- Includes cost tracking and filtering
- Models like `anthropic/claude-3-haiku`, `meta-llama/llama-3.1-70b`, etc.

### Model Caching
`CacheManager` caches model lists with configurable TTL (default 5 minutes) to avoid repeated API calls.

## Debate Format System

### Base Format Class
All formats extend `DebateFormat` and define:
- `phases()` - List of `FormatPhase` objects
- `get_system_prompt(position, context)` - Position-specific instructions
- `validate_response(text, word_limit)` - Response validation
- `get_phase_instructions(phase)` - Phase-specific guidance

### Available Formats
- **Oxford** - Opening, rebuttal, closing (classic structure)
- **Parliamentary** - Government vs. opposition with points of order
- **Socratic** - Question-driven dialogue format
- **Public Forum** - American high school debate style

### Format Registration
```python
@format_registry.register("oxford")
class OxfordFormat(DebateFormat):
    def phases(self) -> list[FormatPhase]:
        return [
            FormatPhase("opening", Position.PROPOSITION, 1),
            FormatPhase("opening", Position.OPPOSITION, 1),
            # ...
        ]
```

## Judge System

### AI Judge
`AIJudge` uses an LLM to evaluate debates:
- Scores on multiple criteria (logic, evidence, persuasiveness, etc.)
- Generates detailed reasoning for decisions
- Returns structured `JudgeDecision` with scores and winner

### Ensemble Judging
`ensemble_utils.py` aggregates multiple judge decisions:
- Takes list of `JudgeDecision` objects
- Averages criterion scores
- Determines consensus winner
- Combines reasoning from all judges

### Judge Factory
`JudgeFactory` creates judge instances based on configuration:
```python
judge = JudgeFactory.create_judge(
    judge_config=config.judging,
    model_manager=model_manager
)
```

## File Organization

### Naming Conventions
- **Modules**: `snake_case.py` (e.g., `debate_engine/core.py`, `ai_judge.py`)
- **Classes**: `PascalCase` (e.g., `DebateEngine`, `ModelManager`, `AIJudge`)
- **Functions**: `snake_case` (e.g., `generate_response`, `get_models`)
- **Constants**: `UPPER_SNAKE_CASE` for module-level constants
- **Type Aliases**: `PascalCase` (e.g., `ChunkCallback`, `ModelDict`)
- **Enums**: `PascalCase` for class, `UPPER_CASE` for values

### Import Conventions
```python
# Standard library (alphabetical)
import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any, Protocol

# Third-party (alphabetical)
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Local absolute imports (preferred for cross-package)
from config.settings import AppConfig, ModelConfig
from debate_engine.core import DebateEngine
from models.manager import ModelManager

# Local relative imports (only within same package)
from .base import DebateFormat
from .types import DebatePhase, Position
```

### Module Organization
- Keep modules focused and single-purpose
- Separate data models (Pydantic) from logic
- Use `__init__.py` for package-level exports
- Avoid circular imports (use TYPE_CHECKING if needed)

## Common Patterns

### Type-Safe Async Generators
```python
from collections.abc import AsyncGenerator

async def stream_chunks(self, prompt: str) -> AsyncGenerator[str, None]:
    """Stream response chunks."""
    async with self._get_session() as session:
        async for chunk in session.stream(prompt):
            yield chunk
```

### Pydantic Model Serialization (v2)
```python
# Modern Pydantic v2 - CORRECT
data = model.model_dump()
json_str = model.model_dump_json()

# OLD Pydantic v1 - DEPRECATED (don't use)
# data = model.dict()  # ❌
# json_str = model.json()  # ❌

# With exclusions
data = model.model_dump(exclude={"internal_field"})

# With None exclusion
data = model.model_dump(exclude_none=True)
```

### Protocol-Based Duck Typing
```python
from typing import Protocol

class ModelProvider(Protocol):
    """Protocol for model providers."""

    async def generate_response(
        self,
        messages: list[dict[str, str]],
        **kwargs
    ) -> str:
        """Generate a response."""
        ...
```

### Enum Usage
```python
from enum import Enum

class DebatePhase(Enum):
    """Debate phase enumeration."""
    OPENING = "opening"
    REBUTTAL = "rebuttal"
    CLOSING = "closing"

# Usage
current_phase = DebatePhase.OPENING
if current_phase == DebatePhase.OPENING:
    print(f"Phase: {current_phase.value}")  # "opening"
```

### Config Validation
```python
from pydantic import BaseModel, Field, field_validator

class ModelConfig(BaseModel):
    """Model configuration."""
    name: str
    provider: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider is supported."""
        if v not in ['ollama', 'openrouter']:
            raise ValueError(f"Unsupported provider: {v}")
        return v
```

## Testing & Development

### Local Development Setup
1. **Clone the repository**
2. **Create virtual environment**: `python -m venv venv`
3. **Activate**: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Unix)
4. **Install package**: `pip install -e ".[dev]"`
5. **Copy config**: `cp debate_config.example.json debate_config.json`
6. **Edit config**: Set your Ollama URL and OpenRouter API key

### Testing with Ollama
1. Ensure Ollama is running: `ollama serve`
2. Pull models: `ollama pull llama3.2:latest`
3. Test model manager:
```python
from models.manager import ModelManager
from config.settings import ModelConfig

manager = ModelManager()
config = ModelConfig(
    name="llama3.2:latest",
    provider="ollama",
    temperature=0.7
)
manager.register_model("test", config)
response = await manager.generate_response(
    "test",
    [{"role": "user", "content": "Hello!"}]
)
```

### Testing with OpenRouter
1. Set environment variable: `$env:OPENROUTER_API_KEY="sk-or-..."`
2. Test provider:
```python
from models.providers.open_router_provider import OpenRouterProvider

provider = OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"])
models = await provider.get_models()
```

### Common Issues
- **"Model not found"** - Run `ollama pull <model>` or check OpenRouter API key
- **Import errors** - Install package with `pip install -e .`
- **Type errors** - Check pyrightconfig.json settings match your IDE
- **Async errors** - All model calls must be awaited in async context

## Performance Considerations

### Model Caching
- `CacheManager` caches model lists with 5-minute TTL
- Reduces API calls to OpenRouter
- Configurable cache duration

### Async Concurrency
- All provider calls are async to avoid blocking
- Use `asyncio.gather()` for concurrent requests
- Streaming prevents memory buildup for long responses

### Memory Management
- Stream responses chunk-by-chunk rather than buffering
- Release model sessions explicitly in context managers
- Clear cache periodically if running long-term

## Deployment

### As a Library
This package is designed to be imported by other projects:
```bash
# Install from local directory
pip install -e /path/to/dialectus-engine

# Or add to requirements.txt
git+https://github.com/yourusername/dialectus-engine.git@main
```

### Building Distribution
```bash
# Build wheel and source distribution
python -m build

# Install from built wheel
pip install dist/dialectus_engine-0.1.0-py3-none-any.whl
```

### Environment Variables
When deploying applications that use this library:
- `OPENROUTER_API_KEY` - Required for cloud models
- `OLLAMA_URL` - Override default Ollama endpoint
- Ensure config files are accessible to the importing application

## Security Considerations

- **No API keys in code** - Use environment variables or config files (excluded from git)
- **Validate all user inputs** - Use Pydantic validation for topic, format, etc.
- **Rate limiting** - Implement in consuming application (dialectus-web-api)
- **Cost tracking** - Monitor OpenRouter usage to avoid unexpected charges
- **Sanitize prompts** - Be cautious with user-provided debate topics

## Future Enhancements

- **Additional providers** - Anthropic Claude SDK, Google Gemini, local vLLM
- **Advanced formats** - Lincoln-Douglas, Karl Popper, World Schools
- **Judge diversity** - Human judge integration, multi-model ensemble
- **Caching improvements** - Redis backend for distributed systems
- **Metrics** - Prometheus metrics for debate latency and quality
- **Testing** - Comprehensive unit and integration test suite
