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
- **`config/`** - Configuration management with auto-creation from templates
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

## Development Notes

- Entry points: `web_server.py` (web server) or `main.py` (CLI)
- Configuration auto-creates from example file on first run
- Test provider setup with `test_providers.py` 
- Web interface available at http://localhost:8000 with API docs at `/docs`
- Uses FastAPI with WebSocket support for real-time debate interactions