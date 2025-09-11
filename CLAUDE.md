# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Dialectus Engine is a Python-based AI debate orchestration system that supports multiple LLM providers (Ollama, OpenRouter) and provides both REST API and WebSocket interfaces for real-time debate interactions.

## Development Commands

### Start the System
```bash
# Start web server (recommended)
python web_server.py

# Alternative CLI entry point
python main.py
```

### Dependencies
```bash
# Install dependencies
pip install -r requirements.txt

# Update dependencies (after modifying requirements.in)
pip-compile requirements.in
```

### Testing
```bash
# Test multi-provider configuration
python test_providers.py
```

## Architecture

### Core Components

- **`debate_engine/`** - Main orchestration logic, debate models, and database integration
- **`web/`** - FastAPI application (`web_server.py` entry point)
- **`models/`** - LLM provider abstractions and model management
  - `manager.py` - Central ModelManager for multi-provider support
  - `providers.py` - Provider factory and implementations (Ollama, OpenRouter)
  - `openrouter_types.py` - OpenRouter model filtering and enhancement
  - `cache_manager.py` - Model data caching with TTL
- **`config/`** - Configuration management with auto-creation from templates
  - `openrouter_filters.json` - Configurable model filtering patterns (excludes meta/router models)
- **`judges/`** - AI judge implementations for debate scoring
- **`formats/`** - Debate format definitions (Oxford, Parliamentary, Socratic)

### Configuration System

The system uses `debate_config.json` (auto-created from `debate_config.example.json`) for:
- Model provider settings (Ollama URL, OpenRouter API keys)
- Model configurations with personalities and parameters
- Debate format preferences
- Judging criteria and methods

Configuration follows the pattern: Base config → Web UI overrides → Final debate config

### Multi-Provider Architecture

- **ProviderFactory** creates provider instances based on configuration
- **ModelManager** handles model registration and discovery across providers
- Supports mixing providers in single debates (e.g., Ollama vs OpenRouter models)
- Provider-specific model formats: `model:tag` (Ollama) vs `provider/model` (OpenRouter)

### API Endpoints

Key REST endpoints:
- `GET /api/debates` - List active debates  
- `POST /api/debates` - Create new debate
- `GET /api/debates/{id}` - Get debate details
- `WebSocket /ws/debate/{id}` - Real-time debate updates
- `GET /api/models` - Available models across providers
- `GET /api/providers` - Provider status

## Frontend Integration

The engine provides REST API and WebSocket endpoints consumed by the **dialectus-web** frontend:

**Model Picker Integration** (`dialectus-web/src/components/model-picker-modal.ts`):
- Fetches models via `/api/models` endpoint (app.py:351)
- Uses enhanced models from `models/manager.py:104`
- OpenRouter models filtered by `models/openrouter_types.py:565`
- Filtering patterns configured in `config/openrouter_filters.json`

**Debate Flow**:
- Frontend creates debates via `/api/debates` → `web/app.py:454`
- Real-time updates via WebSocket `/ws/debate/{id}` → `web/app.py:697`
- Core debate logic in `debate_engine/core.py`

## Development Notes

- Entry points: `web_server.py` (web server) or `main.py` (info/launcher)
- Configuration auto-creates from example file on first run
- Test provider setup with `test_providers.py` 
- Web interface available at http://localhost:8000 with API docs at `/docs`
- Uses FastAPI with WebSocket support for real-time debate interactions
- OpenRouter meta models (auto-router, mixture-of-experts) are filtered out via configurable patterns

### Code Quality & Type Safety

- Uses strict Python type hints throughout the codebase
- Pylance/mypy compatible with proper Optional[] typing for nullable parameters
- Key patterns for clean code:
  - Use `Optional[Type]` for parameters that accept None (not `Type = None`)
  - Use `getattr()` for dynamic method access when type checking fails
  - Cache manager handles JSON serialization with Pydantic v2 `model_dump()` (not deprecated `dict()`)
  - Model filtering uses configurable JSON patterns instead of hardcoded exclusions

### Common Type Issues Fixed

- Model manager uses `getattr()` for dynamic provider method calls
- Cache manager properly handles `Optional[Path]`, `Optional[Dict]` parameters  
- OpenRouter types avoid None attribute access with proper null checking
- Import cleanup: remove unused imports (`List`, `Enum`, `Optional`, etc.)