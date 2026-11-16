
<img width="500" height="117" alt="dialectus-logo-horz" src="https://github.com/user-attachments/assets/438b8963-0652-4886-acd1-5c197040c606" />

<br />

# Dialectus Engine

A Python library for orchestrating AI-powered debates with multi-provider model support.

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> **Ready-to-Use CLI:** Want to run debates right away? Check out the [dialectus-cli](https://github.com/dialectus-ai/dialectus-cli) - a command-line interface that uses this engine to run debates locally with a beautiful terminal UI.

## Overview

The Dialectus Engine is a standalone Python library that provides core debate orchestration logic, including participant coordination, turn management, AI judge integration, and multi-provider model support. It's designed to be imported and used by other applications to build debate systems.

## Components

- **Core Engine** (`debate_engine/`) - Main debate orchestration logic
- **Models** (`models/`) - AI model provider integrations (Ollama, OpenRouter, Anthropic, OpenAI)
- **Configuration** (`config/`) - System configuration management
- **Judges** (`judges/`) - AI judge implementations with ensemble support
- **Formats** (`formats/`) - Debate format definitions (Oxford, Parliamentary, Socratic, Public Forum)
- **Moderation** (`moderation/`) - Optional content safety system for debate topics

## Installation

### From PyPI

**Using uv (recommended):**
```bash
uv pip install dialectus-engine
```

**Using pip:**
```bash
pip install dialectus-engine
```

### From Source

**Using uv (recommended, faster):**
```bash
# Clone the repository
git clone https://github.com/dialectus-ai/dialectus-engine.git
cd dialectus-engine

# Install in development mode with all dev dependencies
uv sync

# Or install without dev dependencies
uv pip install -e .
```

**Using pip:**
```bash
# Clone the repository
git clone https://github.com/dialectus-ai/dialectus-engine.git
cd dialectus-engine

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### As a Dependency

Add to your `pyproject.toml`:

```toml
[project]
dependencies = [
    "dialectus-engine>=0.4.2",
]
```

Or install directly from git:

```bash
# Using uv
uv pip install git+https://github.com/dialectus-ai/dialectus-engine.git@main

# Using pip
pip install git+https://github.com/dialectus-ai/dialectus-engine.git@main
```

## Quick Start

```python
import asyncio
from pathlib import Path

from dialectus.engine import DebateEngine
from dialectus.engine.config.settings import AppConfig
from dialectus.engine.debate_engine.types import (
    MessageCompleteEventData,
    MessageEventType,
    MessageStartEventData,
)
from dialectus.engine.judges.factory import create_judges
from dialectus.engine.models.manager import ModelManager


async def on_message(
    event: MessageEventType,
    data: MessageStartEventData | MessageCompleteEventData,
) -> None:
    # Called once per turn, when the model has finished its response
    if isinstance(data, MessageCompleteEventData):
        print(f"[{data.position}] {data.speaker_id}:\n{data.content}\n")


async def main() -> None:
    config = AppConfig.load_from_file(Path("debate_config.json"))
    model_manager = ModelManager.from_config(config)
    engine = DebateEngine(config=config, model_manager=model_manager)

    # Build the debate context, then run every phase of the chosen format
    await engine.initialize_debate()
    context = await engine.run_full_debate(message_callback=on_message)

    # Optional: have AI judges evaluate the finished debate
    if config.judging.judge_provider:
        judges = create_judges(
            config.judging.judge_models,
            config.judging.judge_provider,
            config.system,
            model_manager,
            config.judging.criteria,
        )
        result = await engine.judge_debate_with_judges(judges)
        print(result)  # JudgeDecision, or a dict with an ensemble summary


asyncio.run(main())
```

`run_full_debate` returns the completed `DebateContext` (topic, participants, messages and metadata). Pass `phase_callback` and `chunk_callback` as well if you want phase headers or token-by-token streaming.

## Configuration

The engine uses `debate_config.json` for system configuration. To get started:

```bash
# Linux/Mac: Copy the example configuration
cp debate_config.example.json debate_config.json

# Windows (PowerShell):
# copy debate_config.example.json debate_config.json

# Edit with your settings and API keys
# Linux/Mac: nano debate_config.json
# Windows: notepad debate_config.json
# Or use your preferred editor (VS Code, vim, etc.)
```

Key configuration sections:
- **Models**: Define debate participants with provider, personality, and parameters
- **Providers**: Configure Ollama (local), OpenRouter, Anthropic, and OpenAI (cloud) settings
- **Judging**: Set evaluation criteria and judge models
- **Debate**: Default topic, format, and word limits
- **Moderation** (optional): Content safety for user-provided topics

For detailed configuration documentation, see [CONFIG_GUIDE.md](CONFIG_GUIDE.md).

## Development Workflows

### Running Tests and Type Checking

**Using uv (recommended):**
```bash
# Run tests
uv run pytest

# Type check with Pyright
uv run pyright

# Lint with ruff
uv run ruff check .

# Format with ruff
uv run ruff format .
```

**Using pip:**
```bash
# Ensure dev dependencies are installed
pip install -e ".[dev]"

# Run tests
pytest

# Type check with Pyright
pyright

# Lint and format
ruff check .
ruff format .
```

### Building Distribution

**Using uv:**
```bash
# Build wheel and sdist
uv build

# Install locally from wheel
uv pip install dist/dialectus_engine-*.whl
```

**Using pip:**
```bash
# Build wheel and sdist
python -m build

# Install locally
pip install dist/dialectus_engine-*.whl
```

### Releasing (maintainers)

Releases are published to PyPI by GitHub Actions (`.github/workflows/publish.yml`) using trusted publishing, so no API token is needed.

```bash
# 1. Bump `version` in pyproject.toml, then refresh the lockfile
#    (CI runs `uv sync --locked`, which fails if uv.lock is stale)
uv lock

# 2. Commit and push to main
git commit -am "Bump version to X.Y.Z" && git push origin main

# 3. Tag the release; pushing a v* tag publishes to TestPyPI, then PyPI
git tag vX.Y.Z && git push origin vX.Y.Z
```

Watch the "Publish to PyPI" run under the repository's Actions tab. Downstream projects (the web API, the CLI) need their `dialectus-engine>=` constraint bumped and their lockfile refreshed to pick up the new version.

### Managing Dependencies

**Using uv:**
```bash
# Add a new dependency
# 1. Edit pyproject.toml [project.dependencies] section
# 2. Update lock file and sync environment:
uv lock && uv sync

# Upgrade all dependencies (within version constraints)
uv lock --upgrade

# Upgrade specific package
uv lock --upgrade-package httpx

# Add dev dependency
# 1. Edit pyproject.toml [project.optional-dependencies.dev]
# 2. Run:
uv sync
```

**Using pip:**
```bash
# Add a new dependency
# 1. Edit pyproject.toml dependencies
# 2. Reinstall:
pip install -e ".[dev]"
```

### Why uv?

- **10-100x faster** than pip for installs and resolution
- **Reproducible builds** via `uv.lock` (cross-platform, includes hashes)
- **Single source of truth** - Dependencies in `pyproject.toml`, lock file auto-generated
- **Compatible** - `pip` still works perfectly with `pyproject.toml`

## Features

### Multi-Provider Model Support
- **Ollama**: Local model management with hardware optimization
- **OpenRouter**: Cloud model access to a wide variety of models
- **Anthropic**: Direct access to Claude models
- **OpenAI**: Direct access to GPT models
- **Async streaming**: Chunk-by-chunk response generation for all providers
- **Auto-discovery**: Dynamic model listing from all configured providers
- **Caching**: Model metadata cached in memory and on disk (`./cache`) with a TTL
- **Cost tracking**: Token usage and cost for cloud providers (OpenRouter reports real costs; direct Anthropic/OpenAI costs use a built-in pricing catalog that may not include the newest models)

### Debate Formats
- **Oxford**: Classic opening/rebuttal/closing structure
- **Parliamentary**: British-style government vs. opposition
- **Socratic**: Question-driven dialogue format
- **Public Forum**: American high school debate style

### AI Judge System
- **LLM-based evaluation**: Detailed criterion scoring
- **Ensemble judging**: Aggregate decisions from multiple judges
- **Structured decisions**: JSON-serializable judge results
- **Configurable criteria**: Logic, evidence, persuasiveness, etc.

### Content Moderation (Optional)
- **Multi-provider support**: Ollama (local), OpenRouter, OpenAI moderation API
- **Safety categories**: Harassment, hate speech, violence, sexual content, dangerous activities
- **Flexible deployment**: Enable for production APIs, disable for trusted environments
- **Graceful error handling**: Provider-specific rate limit handling and retry logic

## Architecture

Key architectural principles:
- **Library-first**: Designed to be imported by other applications
- **Provider agnostic**: Support for multiple AI model sources
- **Async by default**: All model interactions are async
- **Type-safe**: Strict Pyright configuration with modern type hints
- **Pydantic everywhere**: All config and data models use Pydantic v2
- **Configurable**: JSON-based configuration with validation

### Technology Stack
- **Python 3.12+** with modern type hints (`X | None`, `list[T]`, `dict[K, V]`)
- **Pydantic v2** for data validation and settings management
- **OpenAI SDK** for OpenRouter API integration (streaming support)
- **httpx** for async HTTP requests (Ollama provider)
- **asyncio** for concurrent debate operations

## Usage Examples

### Listing Available Models

```python
from pathlib import Path

from dialectus.engine.config.settings import AppConfig
from dialectus.engine.models.manager import ModelManager

async def list_models() -> None:
    config = AppConfig.load_from_file(Path("debate_config.json"))
    manager = ModelManager(config.system)

    # Curated, classified models from every configured provider
    for model in await manager.get_enhanced_models():
        print(f"{model['provider']}: {model['id']} ({model['tier']})")
```

### Exploring Debate Formats

```python
from dialectus.engine.formats import format_registry

# Available formats: oxford, parliamentary, socratic, public_forum
print(format_registry.list_formats())

# Inspect a format's phases for a given set of participants
oxford = format_registry.get_format("oxford")
for phase in oxford.get_phases(["model_a", "model_b"]):
    print(phase)
```

### Ensemble Judging

Pass more than one judge model and the engine aggregates their decisions by majority vote, breaking ties with average criterion scores.

```python
from dialectus.engine.judges.factory import create_judges

judges = create_judges(
    ["openthinker:7b", "llama3.2:3b", "qwen2.5:3b"],
    "ollama",  # provider for all judge models
    config.system,
    model_manager,
    config.judging.criteria,
)

# The debate must be complete (after run_full_debate)
result = await engine.judge_debate_with_judges(judges)
# One judge  -> a JudgeDecision
# Many judges -> {"type": "ensemble", "decisions": [...], "ensemble_summary": ...}
```

Ensemble judging requires **every** judge to succeed. If any judge fails (for example, a model your provider rejects), `judge_debate_with_judges` raises a `RuntimeError` listing the failures. Catch it if you want to keep the debate results regardless.

### Content Moderation

```python
from dialectus.engine.moderation import ModerationManager, TopicRejectedError

# Create moderation manager
manager = ModerationManager(config.moderation, config.system)

# Validate user-provided topic
user_topic = "Should AI be regulated?"

try:
    result = await manager.moderate_topic(user_topic)
    # Topic is safe, proceed with debate
    print(f"Topic approved with confidence: {result.confidence}")
except TopicRejectedError as e:
    # Topic violates content policy
    print(f"Topic rejected: {e.reason}")
    print(f"Violated categories: {', '.join(e.categories)}")
```

You can also let the engine do this for you: enable moderation in the config and call `engine.initialize_debate(topic, user_provided=True)`. See the `moderation` section of [`debate_config.example.json`](debate_config.example.json) for the available settings.

## Provider Setup

### Anthropic (Claude Models)

To use Anthropic's Claude models, you'll need an API key:

1. **Get an API key**: Sign up at [console.anthropic.com](https://console.anthropic.com/)

2. **Set your API key** (choose one method):

   **Environment variable (recommended):**
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-api03-..."
   ```

   **Or in `debate_config.json`:**
   ```json
   {
     "system": {
       "anthropic": {
         "api_key": "sk-ant-api03-...",
         "base_url": "https://api.anthropic.com/v1",
         "max_retries": 3,
         "timeout": 60
       }
     }
   }
   ```

3. **Configure a model**:
   ```json
   {
     "models": {
       "model_a": {
         "name": "claude-haiku-5-5",
         "provider": "anthropic",
         "personality": "analytical",
         "max_tokens": 300,
         "temperature": 0.7
       }
     }
   }
   ```

**Finding available models:**

Use any model ID your Anthropic account can access. Anthropic provides a `/v1/models` API endpoint to list them, and [Anthropic's model documentation](https://docs.anthropic.com/en/docs/models-overview) covers the current lineup. Older Claude 3 and 3.5 models have been retired, so check there before reusing an old config.

### OpenAI

To use OpenAI's GPT models directly:

1. **Get an API key** at [platform.openai.com](https://platform.openai.com/)

2. **Set your API key** (or set `system.openai.api_key` in `debate_config.json`):
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. **Configure a model** with an OpenAI model ID:
   ```json
   {
     "models": {
       "model_a": {
         "name": "gpt-4.1-mini",
         "provider": "openai",
         "personality": "analytical",
         "max_tokens": 300,
         "temperature": 0.7
       }
     }
   }
   ```

Any model ID your OpenAI account can access will work; see [OpenAI's model documentation](https://platform.openai.com/docs/models) for the current lineup.

### OpenRouter

To use OpenRouter's model marketplace:

1. **Get an API key**: Sign up at [openrouter.ai](https://openrouter.ai/)

2. **Set your API key**:
   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-..."
   ```

3. **Configure a model**:
   ```json
   {
     "models": {
       "model_a": {
         "name": "anthropic/claude-haiku-5.5",
         "provider": "openrouter",
         "personality": "analytical",
         "max_tokens": 300,
         "temperature": 0.7
       }
     }
   }
   ```

   Browse model IDs at [openrouter.ai/models](https://openrouter.ai/models). Variants with a `:batch` suffix only accept OpenRouter's batch API and fail on chat requests, so the engine's model discovery filters them out. Don't configure them by hand.

### Ollama (Local Models)

To use local models via Ollama:

1. **Install Ollama**: Download from [ollama.com](https://ollama.com/)

2. **Pull models**:
   ```bash
   ollama pull llama3.2:3b
   ollama pull qwen2.5:7b
   ```

3. **Configure**:
   ```json
   {
     "models": {
       "model_a": {
         "name": "llama3.2:3b",
         "provider": "ollama",
         "personality": "analytical",
         "max_tokens": 300,
         "temperature": 0.7
       }
     },
     "system": {
       "ollama_base_url": "http://localhost:11434"
     }
   }
   ```
