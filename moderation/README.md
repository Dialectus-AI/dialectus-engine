# Moderation Models Directory

This directory is for **local moderation model files** when testing content safety functionality.

## Purpose

Users developing or testing the moderation system can place model files here (GGUF, safetensors, etc.) to use with Ollama or other compatible inference engines.

## Not Distributed with Engine

⚠️ **Important**: Model files in this directory are:
- **User-provided** - Not included in the repository or package
- **Gitignored** - `*.gguf`, `*.bin`, `*.safetensors` are excluded
- **Optional** - Most users will pull models from registries (Ollama, HuggingFace)

The `dialectus-engine` package is model-agnostic and doesn't ship with any models.

## Usage Patterns

### Option 1: Registry-Managed Models (Recommended)

Pull models from a registry like Ollama:

```bash
# Example: Using any moderation model
ollama pull <your-chosen-model>
```

Configure in `debate_config.json`:
```json
{
  "moderation": {
    "enabled": true,
    "provider": "ollama",
    "model": "<your-chosen-model>",
    "timeout": 10.0
  }
}
```

### Option 2: Local Model Files

If you have a GGUF or other model file:

1. Place it in this directory: `moderation/your-model.gguf`
2. Create an Ollama Modelfile:

```bash
cat > Modelfile <<EOF
FROM ./moderation/your-model.gguf
TEMPLATE """{{ .System }}

{{ .Prompt }}"""
EOF
```

3. Import into Ollama:

```bash
ollama create my-moderation-model -f Modelfile
```

4. Use in config:

```json
{
  "moderation": {
    "model": "my-moderation-model"
  }
}
```

## Model Selection

The moderation system works with any instruction-following LLM that can classify content as safe/unsafe. Popular choices include:

- **OpenAI moderation endpoint**: Use `provider: "openai"` with the `omni-moderation-latest` model for the native moderation API
- **Safety-focused models**: Models trained specifically for content moderation
- **General instruction models**: GPT, Claude, Llama, Mistral, etc.
- **Custom fine-tuned models**: Your own models trained on your content policy

See `MODERATION_TESTING.md` for examples and guidance.

## Cleanup

This directory can remain empty if you use registry-managed models. It exists as a convention for developers who want to test with local model files.
