<img src="./assets/logo.png" alt="Dialectus Engine" width="500">

<br />

# Dialectus Engine

A Python library for orchestrating AI-powered debates with multi-provider model support.

![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)

## Overview

The Dialectus Engine is a standalone Python library that provides core debate orchestration logic, including participant coordination, turn management, AI judge integration, and multi-provider model support. It's designed to be imported and used by other applications to build debate systems.

## Components

- **Core Engine** (`debate_engine/`) - Main debate orchestration logic
- **Models** (`models/`) - AI model provider integrations (Ollama, OpenRouter)
- **Configuration** (`config/`) - System configuration management
- **Judges** (`judges/`) - AI judge implementations with ensemble support
- **Formats** (`formats/`) - Debate format definitions (Oxford, Parliamentary, Socratic, Public Forum)

## Installation

### From PyPI

```bash
pip install dialectus-engine
```

### From Source

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

Add to your `requirements.txt` or `pyproject.toml`:

```
dialectus-engine>=0.1.0
```

Or install directly from git:

```bash
pip install git+https://github.com/dialectus-ai/dialectus-engine.git@main
```

## Quick Start

```python
import asyncio
from debate_engine.core import DebateEngine
from models.manager import ModelManager
from config.settings import AppConfig, ModelConfig

async def run_debate():
    # Load configuration
    config = AppConfig.from_json_file("debate_config.json")

    # Set up model manager
    model_manager = ModelManager()

    # Register models
    for model_id, model_config in config.models.items():
        model_manager.register_model(model_id, model_config)

    # Create debate engine
    engine = DebateEngine(
        config=config,
        model_manager=model_manager
    )

    # Run debate
    transcript = await engine.run_debate()
    print(transcript)

asyncio.run(run_debate())
```

## Configuration

The engine uses `debate_config.json` for system configuration. To get started:

```bash
# Copy the example configuration
cp debate_config.example.json debate_config.json

# Edit with your settings and API keys
nano debate_config.json
```

Key configuration sections:
- **Models**: Define debate participants with provider, personality, and parameters
- **Providers**: Configure Ollama (local) and OpenRouter (cloud) settings
- **Judging**: Set evaluation criteria and judge models
- **Debate**: Default topic, format, and word limits

For detailed configuration documentation, see [CONFIG_GUIDE.md](CONFIG_GUIDE.md).

## Features

### Multi-Provider Model Support
- **Ollama**: Local model management with hardware optimization
- **OpenRouter**: Cloud model access to 100+ models
- **Async streaming**: Chunk-by-chunk response generation
- **Auto-discovery**: Dynamic model listing from all configured providers
- **Caching**: In-memory cache with TTL for model metadata

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

## Architecture

Key architectural principles:
- **Library-first**: Designed to be imported by other applications
- **Provider agnostic**: Support for multiple AI model sources
- **Async by default**: All model interactions are async
- **Type-safe**: Strict Pyright configuration with modern type hints
- **Pydantic everywhere**: All config and data models use Pydantic v2
- **Configurable**: JSON-based configuration with validation

### Technology Stack
- **Python 3.13+** with modern type hints (`X | None`, `list[T]`, `dict[K, V]`)
- **Pydantic v2** for data validation and settings management
- **OpenAI SDK** for OpenRouter API integration (streaming support)
- **httpx** for async HTTP requests (Ollama provider)
- **asyncio** for concurrent debate operations

## Usage Examples

### Listing Available Models

```python
from models.manager import ModelManager

async def list_models():
    manager = ModelManager()
    models = await manager.get_all_models()
    for model_id, model_info in models.items():
        print(f"{model_id}: {model_info.description}")
```

### Running a Custom Format

```python
from formats.registry import format_registry

# Get available formats
formats = format_registry.list_formats()

# Load a specific format
oxford = format_registry.get_format("oxford")
phases = oxford.phases()
```

### Ensemble Judging

```python
from judges.factory import JudgeFactory

# Create judge with multiple models
config.judging.judge_models = ["openthinker:7b", "llama3.2:3b", "qwen2.5:3b"]
judge = JudgeFactory.create_judge(config.judging, model_manager)

# Get aggregated decision
decision = await judge.judge_debate(context)
```
