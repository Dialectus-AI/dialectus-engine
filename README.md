<img src="./assets/logo.png" alt="Dialectus Engine" width="500">

<br />

# Dialectus Engine

Core debate orchestration engine and REST API for the Dialectus AI debate system.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![SQLite](https://img.shields.io/badge/database-SQLite-003B57.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)

## Overview

The Dialectus Engine provides the core logic for managing AI-powered debates, including participant coordination, turn management, judge integration, and user authentication. It exposes a RESTful API and WebSocket interface for real-time debate interactions with secure user management.

## Components

- **Core Engine** (`debate_engine/`) - Main debate orchestration logic
- **REST API** (`web/`) - FastAPI-based web service with authentication
- **User Authentication** (`web/auth_*`) - JWT-based user management with email verification
- **Models** (`models/`) - AI model provider integrations (Ollama, OpenRouter)
- **Configuration** (`config/`) - System configuration management
- **Judges** (`judges/`) - AI judge implementations with ensemble support
- **Formats** (`formats/`) - Debate format definitions (Oxford, Parliamentary, Socratic)
- **Transcripts** (`transcripts/`) - SQLite database for debate storage and user data

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
- `GET /api/health` - API health check
- `GET /api/models` - List available AI models from all providers
- `GET /api/providers` - List provider status and configuration
- `GET /api/formats` - List available debate formats

### Debate Management
- `POST /api/debates` - Create new debate
- `GET /api/debates/{id}` - Get debate status and details
- `POST /api/debates/{id}/start` - Start debate execution
- `POST /api/debates/{id}/cancel` - Cancel running debate
- `GET /api/debates/{id}/transcript` - Get debate transcript
- `WebSocket /ws/debate/{id}` - Real-time debate streaming

### User Authentication
- `POST /api/auth/register` - Register new user account
- `POST /api/auth/verify` - Verify email address with token
- `POST /api/auth/complete-registration` - Complete registration with username
- `POST /api/auth/login` - User login with JWT token
- `POST /api/auth/logout` - User logout
- `GET /api/auth/me` - Get current authenticated user

### Utilities
- `GET /api/generate-topic` - Generate random debate topic
- `GET /api/transcripts` - List stored debate transcripts
- `GET /api/transcripts/{id}` - Get specific transcript details

### System Management
- `GET /api/ollama/health` - Check Ollama provider health
- `GET /api/cache/stats` - View model cache statistics
- `POST /api/cache/cleanup` - Clean up expired cache entries
- `DELETE /api/cache/models` - Clear model cache

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

This repository creates a focused, deployable service for the debate engine and API layer. The engine handles all AI model interactions, debate logic, and data persistence while exposing clean APIs for client applications.
