# Configuration Guide

## Overview

Dialectus uses a single configuration file called `debate_config.json` that contains all settings for debates, models, and system configuration.

## Configuration Files

### `debate_config.json` (Auto-created)
- **Location**: Root directory of engine/CLI projects
- **Purpose**: Your actual configuration with API keys and preferences
- **Status**: Automatically created on first run, ignored by git
- **Contains**: All your settings including API keys

### `debate_config.example.json` (Template)
- **Location**: Root directory of engine/CLI projects  
- **Purpose**: Template showing all available configuration options
- **Status**: Committed to git as documentation
- **Contains**: Example values and full structure reference

## Auto-Creation Process

When you first run the system:

1. **Checks for `debate_config.json`** - if it exists, uses it
2. **Copies from `debate_config.example.json`** - if example exists, copies it
3. **Creates from template** - fallback to built-in defaults

## Configuration Structure

### Model Providers

```json
{
  "models": {
    "model_a": {
      "name": "qwen2.5:7b",           // Model identifier
      "provider": "ollama",           // Provider: "ollama" or "openrouter"  
      "personality": "analytical",    // Debate personality
      "max_tokens": 300,             // Max response length
      "temperature": 0.7             // Creativity level (0.0-1.0)
    },
    "model_b": {
      "name": "openai/gpt-4o-mini",  // OpenRouter model format
      "provider": "openrouter",      // Cloud provider
      "personality": "passionate",    // Different personality
      "max_tokens": 300,
      "temperature": 0.8
    }
  }
}
```

### Provider Settings

#### Ollama Configuration
```json
{
  "system": {
    "ollama_base_url": "http://localhost:11434",
    "ollama": {
      "num_gpu_layers": -1,              // -1 = all GPU layers, 0 = CPU only
      "gpu_memory_utilization": null,    // GPU memory percentage (0.0-1.0)
      "main_gpu": null,                  // Primary GPU for multi-GPU setups
      "num_thread": null,                // CPU threads for processing
      "keep_alive": "5m",                // How long to keep models loaded
      "repeat_penalty": 1.1              // Repetition penalty
    }
  }
}
```

#### OpenRouter Configuration  
```json
{
  "system": {
    "openrouter": {
      "api_key": "sk-or-v1-your-api-key-here",  // 🔑 PUT YOUR API KEY HERE
      "base_url": "https://openrouter.ai/api/v1",
      "site_url": null,                          // Your site URL (optional)
      "app_name": "Dialectus AI Debate System",  // App identifier
      "max_retries": 3,                          // API retry attempts
      "timeout": 60                              // Request timeout (seconds)
    }
  }
}
```

### Debate Settings
```json
{
  "debate": {
    "topic": "Your default debate topic",
    "format": "oxford",                // "oxford", "parliamentary", "socratic"
    "time_per_turn": 120,             // Seconds per turn
    "word_limit": 200                 // Words per response
  }
}
```

### Judging Configuration
```json
{
  "judging": {
    "method": "ai",                    // "ai", "ensemble", "none"
    "criteria": [                      // Scoring criteria
      "logic",
      "evidence", 
      "persuasiveness"
    ],
    "judge_model": "openthinker:7b"    // Model for AI judging
  }
}
```

## Setting Up API Keys

### Option 1: Configuration File (Recommended)
Edit your `debate_config.json`:
```json
{
  "system": {
    "openrouter": {
      "api_key": "sk-or-v1-your-actual-api-key-here"
    }
  }
}
```

### Option 2: Environment Variable
Set environment variable (fallback):
```bash
export OPENROUTER_API_KEY="sk-or-v1-your-actual-api-key-here"
```

## Configuration Override Flow

1. **Base Config** (`debate_config.json`) - System defaults
2. **Web UI Form** - Per-debate overrides 
3. **Final Config** - Used for that specific debate

Example: If config file has `temperature: 0.7` but web form sets `0.9`, the debate uses `0.9`.

## Available Model Formats

### Ollama Models
- Format: `model_name:tag` (e.g., `llama3.2:7b`, `qwen2.5:14b`)
- Provider: `"ollama"`
- Requires: Local Ollama installation

### OpenRouter Models  
- Format: `provider/model` (e.g., `openai/gpt-4o-mini`, `anthropic/claude-3-haiku`)
- Provider: `"openrouter"`  
- Requires: OpenRouter API key

## Getting Started

1. **First Run**: System auto-creates `debate_config.json` from example
2. **Add API Key**: Edit the `openrouter.api_key` field  
3. **Customize Models**: Add your preferred model combinations
4. **Start Debating**: Use web interface or CLI

## Troubleshooting

### "No OpenRouter API key found"
- Add API key to `debate_config.json` OR set `OPENROUTER_API_KEY` environment variable

### "Provider openrouter requires API key"  
- Check that your API key starts with `sk-or-v1-`
- Verify the key is valid at https://openrouter.ai

### "Model not found"
- For Ollama: Ensure model is pulled locally (`ollama pull model_name`)
- For OpenRouter: Check available models at https://openrouter.ai/models

### Configuration not loading
- Ensure `debate_config.json` exists in the project root
- Check JSON syntax with a validator
- Look at `debate_config.example.json` for reference structure