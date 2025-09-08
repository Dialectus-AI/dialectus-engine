# Dialectus Engine

Core debate orchestration engine and REST API for the Dialectus AI debate system.

## Overview

The Dialectus Engine provides the core logic for managing AI-powered debates, including participant coordination, turn management, and judge integration. It exposes a RESTful API and WebSocket interface for real-time debate interactions.

## Components

- **Core Engine** (`debate_engine/`) - Main debate orchestration logic
- **REST API** (`web/`) - FastAPI-based web service
- **Models** (`models/`) - Data models and schemas
- **Configuration** (`config/`) - System configuration management
- **Judges** (`judges/`) - AI judge implementations
- **Formats** (`formats/`) - Debate format definitions

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the server:
```bash
python web_server.py
```

3. Access the API documentation at http://localhost:8000/docs

## API Endpoints

- `GET /api/debates` - List active debates
- `POST /api/debates` - Create new debate
- `GET /api/debates/{id}` - Get debate details
- `WebSocket /ws/debate/{id}` - Real-time debate updates

## Development

This repository was extracted from the original AI-Debate monolith to create a focused, deployable service for the debate engine and API layer.

For multi-repository development workflow, use:
```bash
claude-code --add-dir ../dialectus-web --add-dir ../dialectus-cli dialectus-engine
```