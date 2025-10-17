## What's New

### Content Moderation System

This release introduces an **optional content moderation system** for debate topics. Library consumers can now validate user-provided topics for safety before starting debates.

**Key Features:**
- **Multi-provider support** - Use local models (Ollama), cloud APIs (OpenRouter), or OpenAI's dedicated moderation endpoint
- **Flexible deployment** - Enable for production APIs, disable for trusted environments
- **Safety categories** - Detects harassment, hate speech, violence, sexual content, and dangerous activities
- **Rate limit handling** - Graceful error handling with provider-specific retry logic
- **Type-safe** - Full Pydantic models and strict type checking
- **Production-ready** - Comprehensive error handling and testing

**Example Usage:**
```python
from dialectus.engine.moderation import ModerationManager, TopicRejectedError

manager = ModerationManager(config.moderation, config.system)

try:
    result = await manager.moderate_topic(user_topic)
    # Topic approved, proceed with debate
except TopicRejectedError as e:
    # Topic rejected - show error to user
    print(f"Rejected: {e.reason}")
    print(f"Categories: {', '.join(e.categories)}")
```

**Configuration:**
```json
{
  "moderation": {
    "enabled": true,
    "provider": "ollama",  // or "openrouter", "openai"
    "model": "your-model-name",
    "timeout": 10.0
  }
}
```

See `MODERATION_TESTING.md` for detailed testing instructions and integration examples.

---

## Additional Improvements

### Development Infrastructure
- **uv package manager support** - 10-100x faster dependency resolution and installation
- **Comprehensive test suite** - Added tests for config, models, formats, judges, and moderation
- **GitHub Actions CI** - Automated type checking and formatting validation
- **Enhanced documentation** - Updated README with uv workflows and examples

### Code Quality
- **Ruff formatting** - Consistent code style across the codebase
- **Type safety improvements** - Enhanced Pyright configuration and type stubs
- **Lint fixes** - Resolved long line lengths and simplified instructions
- **Rate limit error handling** - Improved error hierarchy for provider-specific rate limits

### Testing & Examples
- **Format registry tests** - Comprehensive coverage for debate format registration
- **Moderation examples** - Interactive test scripts for all supported providers
- **Cache manager tests** - Validation of model metadata caching
- **Judge factory tests** - Testing for AI judge creation and ensemble support

---

## Breaking Changes

None. This release is fully backward compatible.

---

## Migration Guide

### Enabling Moderation (Optional)

If you want to use the new moderation system:

1. **Update your config schema** - Add a `moderation` section to your `debate_config.json`:

```json
{
  "moderation": {
    "enabled": true,
    "provider": "ollama",
    "model": "your-model",
    "timeout": 10.0
  }
}
```

2. **Install a moderation provider** (if using local):
```bash
# Example: Ollama
ollama pull llama3.2:3b
```

3. **Integrate into your application**:
```python
from dialectus.engine.moderation import ModerationManager, TopicRejectedError

# Before starting debate
if config.moderation.enabled:
    manager = ModerationManager(config.moderation, config.system)
    try:
        await manager.moderate_topic(user_topic)
    except TopicRejectedError:
        # Handle rejection
        pass
```

### Using uv (Optional but Recommended)

To take advantage of faster dependency management:

```bash
# Install uv
pip install uv

# Install dialectus-engine
uv pip install dialectus-engine

# Or install from source in development mode
uv sync
```

---

## Documentation

- **MODERATION_TESTING.md** - Comprehensive testing guide for all providers
- **README.md** - Updated with uv workflows and moderation examples
- **tests/README.md** - Testing framework documentation

---

## Contributors

- [@psarno](https://github.com/psarno) - Core development and moderation system

---

## Installation

```bash
# Using pip
pip install dialectus-engine

# Using uv (recommended)
uv pip install dialectus-engine

# From source
git clone https://github.com/dialectus-ai/dialectus-engine.git
cd dialectus-engine
uv sync  # or: pip install -e ".[dev]"
```

---

## What's Next

We're working on:
- Additional debate formats
- Enhanced judge evaluation metrics
- WebSocket streaming support
- Performance optimizations

---

## Feedback

Found a bug or have a feature request? [Open an issue](https://github.com/dialectus-ai/dialectus-engine/issues)!
