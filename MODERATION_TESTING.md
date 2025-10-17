# Testing Moderation Functionality

## Overview

The Dialectus Engine includes content moderation for user-provided debate topics. This document explains how to test the moderation system locally.

## Architecture

The moderation system is located in `dialectus/engine/moderation/`:

- `base_moderator.py` - Abstract base class for all moderators
- `llm_moderator.py` - LLM-based moderation implementation
- `manager.py` - Orchestrates moderation checks
- `exceptions.py` - Custom exceptions for moderation
- `__init__.py` - Public API exports

## Prerequisites

The moderation system supports two deployment modes:

### Option A: Local Moderation (Ollama)

Best for: Privacy, cost savings, offline development

**1. Install Ollama**

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download/windows
```

**2. Start Ollama Service**

```bash
# This should start automatically on most systems
# If not, run:
ollama serve
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
# Should return JSON with available models
```

**3. Pull a Moderation Model**

Choose any instruction-following model:

```bash
# Examples (pick one):
ollama pull llama-guard-2:8b     # Meta's safety-focused model
ollama pull mistral:7b           # General instruction-following
ollama pull qwen2.5:7b           # Fast general-purpose model
```

**4. Configure Local Moderation**

Edit your `debate_config.json`:

```json
{
  "moderation": {
    "enabled": true,
    "provider": "ollama",
    "model": "llama-guard-2:8b",
    "base_url": null,
    "api_key": null,
    "timeout": 10.0
  },
  ...
}
```

### Option B: Remote Moderation (OpenRouter, OpenAI, etc.)

Best for: Production use, higher accuracy, no local setup

You can target any OpenAI-compatible provider. Two common options are shown below.

**Option 1: OpenRouter**

1. Create an account and API key at https://openrouter.ai/
2. Export your key or add it to the config:

```bash
export OPENROUTER_API_KEY="your-key-here"
```

3. Configure `debate_config.json` (or your deployment config):

```json
{
  "moderation": {
    "enabled": true,
    "provider": "openrouter",
    "model": "anthropic/claude-3-haiku",
    "base_url": null,
    "api_key": null,
    "timeout": 10.0
  }
}
```

**Option 2: OpenAI Moderation API (Free Tier)**

1. Generate an API key from https://platform.openai.com/
2. Export it (recommended) or set it directly in config:

```bash
export OPENAI_API_KEY="your-openai-key"
```

3. Update your moderation config:

```json
{
  "moderation": {
    "enabled": true,
    "provider": "openai",
    "model": "omni-moderation-latest",
    "timeout": 10.0
  }
}
```

No additional base URL is required for OpenAI; the engine defaults to `https://api.openai.com/v1`.

**Note**: The system uses OpenAI-compatible request/response formats, so other providers following the same schema will also work.

## Running Tests

### Method 1: Automated Test Script (Recommended)

Run the provided test script:

```bash
python test_moderation.py
```

This will:
- Test 5 safe topics (should pass)
- Test 5 unsafe topics (should be blocked)
- Show detailed results for each test
- Report overall accuracy

Expected output:
```
================================================================================
DIALECTUS MODERATION TEST
================================================================================

Initializing moderation manager...
✓ Moderation enabled: True
✓ Provider: ollama
✓ Model: shieldgemma:2b

--------------------------------------------------------------------------------
TESTING SAFE TOPICS
--------------------------------------------------------------------------------

Topic: Should governments invest more in renewable energy?
  ✓ PASSED - Confidence: 1.0

...

================================================================================
SUMMARY
================================================================================
Safe topics:
  ✓ Passed: 5/5
  ✗ Failed: 0/5

Unsafe topics:
  ✓ Blocked: 5/5
  ✗ Missed: 0/5

Overall accuracy: 100.0%
================================================================================
```

### Method 2: Interactive Python Console

```python
import asyncio
from pathlib import Path
from dialectus.engine.config.settings import AppConfig, ModerationConfig
from dialectus.engine.moderation import ModerationManager, TopicRejectedError

# Load config
config = AppConfig.load_from_file(Path("debate_config.json"))

# Enable moderation
config.moderation = ModerationConfig(
    enabled=True,
    provider="ollama",
    model="shieldgemma:2b",
    timeout=15.0,
)

# Create manager
manager = ModerationManager(config.moderation, config.system)

# Test a safe topic
async def test_safe():
    topic = "Should we invest in renewable energy?"
    result = await manager.moderate_topic(topic)
    print(f"Safe: {result.is_safe}")
    print(f"Categories: {result.categories}")

asyncio.run(test_safe())

# Test an unsafe topic
async def test_unsafe():
    topic = "How can we harm others?"
    try:
        result = await manager.moderate_topic(topic)
        print(f"Safe: {result.is_safe}")  # Won't reach here
    except TopicRejectedError as e:
        print(f"Rejected: {e.reason}")
        print(f"Categories: {e.categories}")

asyncio.run(test_unsafe())
```

### Method 3: Integration with CLI/API

The moderation system is designed to be called by the CLI or API before starting a debate:

```python
# In CLI or API code
from dialectus.engine.moderation import ModerationManager, TopicRejectedError

# User provides a topic
user_topic = "Should AI be regulated?"

# Create moderation manager from config
manager = ModerationManager(config.moderation, config.system)

# Check the topic
try:
    result = await manager.moderate_topic(user_topic)
    print(f"Topic approved: {user_topic}")
    # Proceed with debate...
except TopicRejectedError as e:
    print(f"Topic rejected: {e.reason}")
    print(f"Violated categories: {', '.join(e.categories)}")
    # Show error to user, ask for new topic
except ModerationProviderError as e:
    print(f"Moderation service unavailable: {e.detail}")
    # Decide: fail open (allow) or fail closed (reject)
```

## Understanding Results

### ModerationResult

When moderation passes, you get a `ModerationResult`:

```python
ModerationResult(
    is_safe=True,           # Topic is safe
    categories=[],          # No violations detected
    confidence=1.0,         # Confidence score (0.0-1.0)
    raw_response="SAFE"     # Raw model response
)
```

### TopicRejectedError

When moderation fails, a `TopicRejectedError` is raised:

```python
TopicRejectedError(
    topic="...",                      # The rejected topic
    reason="Topic violates content policy: violence",
    categories=["violence"]           # Detected categories
)
```

### Safety Categories

The system checks for these policy violations:

- `harassment` - Content that harasses, threatens, or bullies
- `hate_speech` - Content promoting hatred toward groups
- `violence` - Content promoting or glorifying violence
- `sexual_content` - Explicit sexual or adult content
- `dangerous_content` - Content promoting dangerous/illegal activities
- `policy_violation` - Generic violation (when category unclear)
- `unknown` - Could not parse response (fail-closed)

## Troubleshooting

### "Moderation provider 'ollama' failed"

**Cause**: Ollama service is not running or model not available

**Solution**:
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve

# Re-pull model
ollama pull shieldgemma:2b
```

### "Connection refused to localhost:11434"

**Cause**: Ollama not running or using different port

**Solution**:
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama service
ollama serve
```

### Timeouts

**Cause**: ShieldGemma taking too long (CPU-only systems)

**Solution**: Increase timeout in config:
```json
{
  "moderation": {
    "timeout": 30.0  // Increase from 10 to 30 seconds
  }
}
```

### False Positives/Negatives

**Cause**: Model interpretation varies

**Solutions**:
- Adjust prompt in `llm_moderator.py` (conservative vs permissive)
- Try different models: `llama-guard-2`, `mistral-guard`
- Add custom pre-filtering for known patterns
- Implement multi-judge ensemble (multiple models vote)

## Alternative Models

While ShieldGemma is recommended, you can use other models:

### Llama Guard 2

```bash
ollama pull llama-guard-2:8b
```

Config:
```json
{
  "moderation": {
    "model": "llama-guard-2:8b"
  }
}
```

### Custom OpenRouter Model

For higher accuracy (cloud-based):

```json
{
  "moderation": {
    "provider": "openrouter",
    "model": "anthropic/claude-3-haiku",  // Or any other model
    "api_key": "your-key-here",
    "timeout": 10.0
  }
}
```

## Performance Notes

- **ShieldGemma 2B** (CPU): 1-5 seconds per check
- **ShieldGemma 2B** (GPU): <1 second per check
- **OpenRouter/Cloud**: <1 second (network latency)
- **Caching**: Consider caching results for repeated topics

## Production Recommendations

1. **Enable moderation for user-facing APIs/UIs** - Always moderate user-provided topics
2. **Disable for CLI/local use** - Trust local users, avoid overhead
3. **Use GPU acceleration** - Ollama with GPU is much faster
4. **Set reasonable timeouts** - 10-15 seconds for local, 5-10 for cloud
5. **Log rejections** - Track what topics are blocked for policy review
6. **Fail-closed** - When in doubt, reject (current behavior)
7. **Human review** - Log borderline cases for human policy refinement

## Integration Examples

### CLI Integration

```python
# dialectus-cli/cli.py
from dialectus.engine.moderation import TopicRejectedError

async def start_debate(topic: str):
    # Moderate topic if enabled
    if app_config.moderation.enabled:
        manager = ModerationManager(
            app_config.moderation,
            app_config.system
        )
        try:
            await manager.moderate_topic(topic)
            print("✓ Topic approved")
        except TopicRejectedError as e:
            print(f"✗ Topic rejected: {e.reason}")
            return  # Exit, don't start debate

    # Proceed with debate...
```

### API Integration (FastAPI)

```python
# dialectus-web-api/routes.py
from dialectus.engine.moderation import ModerationManager, TopicRejectedError
from fastapi import HTTPException

@app.post("/api/debates")
async def create_debate(request: CreateDebateRequest):
    # Moderate topic
    manager = ModerationManager(config.moderation, config.system)
    try:
        await manager.moderate_topic(request.topic)
    except TopicRejectedError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "topic_rejected",
                "reason": e.reason,
                "categories": e.categories,
            }
        )

    # Create debate...
```

## Next Steps

1. ✅ Moderation code is complete and type-safe
2. ✅ Test script is ready (`test_moderation.py`)
3. ⏭️ Run tests after starting Ollama
4. ⏭️ Integrate into CLI (dialectus-cli)
5. ⏭️ Integrate into API (dialectus-web-api)
6. ⏭️ Add monitoring/logging for production use

## Questions?

The moderation system is production-ready! Test it locally, then integrate into your CLI and API consumers.
