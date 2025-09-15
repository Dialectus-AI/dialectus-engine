# Dialectus Engine

Core debate orchestration engine and REST API for the Dialectus AI debate system.

## Overview

The Dialectus Engine provides the core logic for managing AI-powered debates, including participant coordination, turn management, and judge integration. It exposes a RESTful API and WebSocket interface for real-time debate interactions.

## Components

- **Core Engine** (`debate_engine/`) - Main debate orchestration logic
- **REST API** (`web/`) - FastAPI-based web service
- **Models** (`models/`) - AI model provider integrations (Ollama, OpenRouter)
- **Configuration** (`config/`) - System configuration management
- **Judges** (`judges/`) - AI judge implementations with ensemble support
- **Formats** (`formats/`) - Debate format definitions (Oxford, Parliamentary, Socratic)
- **Transcripts** (`transcripts/`) - SQLite database for debate storage

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure the system:**
```bash
cp debate_config.example.json debate_config.json
# Edit debate_config.json with your provider settings
```

3. **Start the server:**
```bash
python web_server.py
```

4. **Access the API documentation at http://localhost:8000/docs**

## Configuration

The engine uses `debate_config.json` for system configuration. Key sections:

### Provider Configuration

```json
{
  "system": {
    "ollama_base_url": "http://localhost:11434",
    "ollama": {
      "num_gpu_layers": -1,
      "keep_alive": "5m",
      "repeat_penalty": 1.1
    },
    "openrouter": {
      "api_key": "your-openrouter-api-key-here",
      "base_url": "https://openrouter.ai/api/v1",
      "timeout": 60,
      "max_retries": 3
    }
  }
}
```

### Judging Configuration

```json
{
  "judging": {
    "criteria": ["logic", "evidence", "persuasiveness"],
    "judge_models": ["openthinker:7b"],
    "judge_provider": "ollama"
  }
}
```

Supports multiple judges for ensemble decisions:
```json
{
  "judging": {
    "judge_models": ["openthinker:7b", "llama3.2:3b", "qwen2.5:3b"],
    "judge_provider": "ollama"
  }
}
```

### Topic Generation

Used by the web interface's "refresh topic" feature:
```json
{
  "system": {
    "debate_topic_source": "openrouter",
    "debate_topic_model": "anthropic/claude-3-haiku"
  }
}
```

## API Endpoints

### Core Endpoints
- `GET /api/models` - List available AI models from all providers
- `GET /api/providers` - List provider status and configuration
- `POST /api/debates` - Create new debate
- `POST /api/debates/{id}/start` - Start debate execution
- `GET /api/debates/{id}` - Get debate status
- `WebSocket /ws/debate/{id}` - Real-time debate streaming

### Model Integration
- **Ollama**: Local model management with hardware optimization
- **OpenRouter**: Cloud model access with API key authentication
- **Auto-discovery**: Dynamic model listing from all configured providers

## Architecture

The engine serves as the backend for both:
- **CLI Interface** - Command-line client for debates
- **Web Interface** - Browser-based real-time UI

Key architectural principles:
- **API-first**: All functionality exposed via REST/WebSocket
- **Provider agnostic**: Support for multiple AI model sources
- **Real-time**: WebSocket streaming for live debate updates
- **Persistent**: SQLite storage for debates and judging results
- **Configurable**: JSON-based configuration with validation

## Development

This repository was extracted from the original AI-Debate monolith to create a focused, deployable service for the debate engine and API layer. The engine handles all AI model interactions, debate logic, and data persistence while exposing clean APIs for client applications.
